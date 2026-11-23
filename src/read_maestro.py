"""Read Maestro files (.mae, .maegz), including Glide pose viewer output.

Written because a Glide pose viewer file is the natural input to pose triage
and there is no dependency-free reader for it. Nothing here needs Schrodinger
software installed.

Format, briefly
---------------
A Maestro file is a sequence of blocks. Each structure is an ``f_m_ct`` block
holding a property table and sub-blocks of atoms and bonds. A table is written
as a list of column names, then a ``:::`` separator, then the values. Column
names encode their type in the first characters: ``s_`` string, ``i_``
integer, ``r_`` real, ``b_`` boolean.

A pose viewer file (``_pv``) holds the receptor as the first structure,
flagged with ``b_glide_receptor``, followed by one structure per pose.

Streaming
---------
These files are large: a 4-million-compound Glide SP run produced a 174 MB
file with 18,899 structures. ``iter_structures`` yields one at a time and
never holds more than one in memory. Use ``read_properties_only`` when the
coordinates are not needed, which is much faster.
"""

from __future__ import annotations

import gzip
import io
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

LOG = logging.getLogger(__name__)

RECEPTOR_FLAG = "b_glide_receptor"
DOCKING_SCORE = "r_i_docking_score"
GLIDE_GSCORE = "r_i_glide_gscore"
TITLE = "s_m_title"


class MaestroParseError(RuntimeError):
    """Raised on malformed input. Never recovered from by guessing."""


@dataclass
class Structure:
    """One ``f_m_ct`` block."""

    properties: dict[str, Any] = field(default_factory=dict)
    atoms: list[dict[str, Any]] = field(default_factory=list)
    bonds: list[dict[str, Any]] = field(default_factory=list)

    @property
    def title(self) -> str | None:
        return self.properties.get(TITLE)

    @property
    def is_receptor(self) -> bool:
        return bool(self.properties.get(RECEPTOR_FLAG))

    @property
    def docking_score(self) -> float | None:
        value = self.properties.get(DOCKING_SCORE)
        return float(value) if value is not None else None

    def coordinates(self) -> list[tuple[float, float, float]]:
        return [
            (atom["r_m_x_coord"], atom["r_m_y_coord"], atom["r_m_z_coord"])
            for atom in self.atoms
        ]

    def elements(self) -> list[int]:
        """Atomic numbers. Maestro stores these as i_m_atomic_number."""
        return [atom.get("i_m_atomic_number") for atom in self.atoms]


def _open(path: Path) -> io.TextIOBase:
    if path.suffix == ".maegz" or path.name.endswith(".mae.gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def _convert(name: str, raw: str) -> Any:
    """Cast a value using the type prefix of its column name."""
    if raw == "<>":
        return None
    kind = name[:2]
    try:
        if kind == "i_":
            return int(raw)
        if kind == "r_":
            return float(raw)
        if kind == "b_":
            return bool(int(raw))
    except ValueError:
        return raw
    return raw


def _tokenise(line: str) -> list[str]:
    """Split a data row, honouring double-quoted fields with escapes."""
    tokens: list[str] = []
    current: list[str] = []
    in_quotes = False
    escaped = False
    for character in line:
        if escaped:
            current.append(character)
            escaped = False
            continue
        if character == "\\":
            escaped = True
            continue
        if character == '"':
            in_quotes = not in_quotes
            continue
        if character.isspace() and not in_quotes:
            if current:
                tokens.append("".join(current))
                current = []
            continue
        current.append(character)
    if current:
        tokens.append("".join(current))
    return tokens


def _read_table(
    lines: Iterator[str], has_index_column: bool
) -> tuple[list[str], list[list[str]]]:
    """Read column names, the ``:::`` separator, then rows until ``:::``."""
    names: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped == ":::":
            break
        if stripped and not stripped.startswith("#"):
            names.append(stripped)
    else:
        raise MaestroParseError("file ended while reading column names")

    rows: list[list[str]] = []
    for line in lines:
        stripped = line.strip()
        if stripped == ":::":
            break
        if not stripped:
            continue
        tokens = _tokenise(stripped)
        if has_index_column and tokens:
            tokens = tokens[1:]
        rows.append(tokens)
    return names, rows



def _read_property_table(lines: Iterator[str]) -> dict[str, Any]:
    """Read an f_m_ct property block.

    Unlike the atom and bond tables, the values here are written one per line
    rather than as columns, and a string value may contain spaces. Reading
    this with the row-based reader silently produces an empty property dict,
    which is how the receptor flag went missing the first time.
    """
    names: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped == ":::":
            break
        if stripped and not stripped.startswith("#"):
            names.append(stripped)
    else:
        raise MaestroParseError("file ended while reading property names")

    properties: dict[str, Any] = {}
    for name in names:
        line = next(lines, None)
        if line is None:
            raise MaestroParseError(
                f"file ended after {len(properties)}/{len(names)} property values"
            )
        raw = line.strip()
        if raw.startswith('"') and raw.endswith('"') and len(raw) >= 2:
            raw = raw[1:-1]
        properties[name] = _convert(name, raw)
    return properties


def iter_structures(
    path: Path, with_coordinates: bool = True, limit: int | None = None
) -> Iterator[Structure]:
    """Yield structures one at a time.

    Set with_coordinates=False to skip atom and bond tables, which is several
    times faster when only scores and titles are wanted.
    """
    handle = _open(path)
    lines: Iterator[str] = iter(handle)
    count = 0
    try:
        for line in lines:
            stripped = line.strip()
            if not stripped.startswith("f_m_ct"):
                continue

            structure = Structure()
            structure.properties = _read_property_table(lines)

            depth = 1
            while depth > 0:
                line = next(lines, None)
                if line is None:
                    break
                stripped = line.strip()
                if stripped.endswith("{"):
                    block_name = stripped.split("[")[0].split()[0]
                    if block_name == "m_atom" and with_coordinates:
                        names, rows = _read_table(lines, has_index_column=True)
                        structure.atoms = [
                            {
                                name: _convert(name, raw)
                                for name, raw in zip(names, row)
                            }
                            for row in rows
                        ]
                        _consume_block_close(lines)
                    elif block_name == "m_bond" and with_coordinates:
                        names, rows = _read_table(lines, has_index_column=True)
                        structure.bonds = [
                            {
                                name: _convert(name, raw)
                                for name, raw in zip(names, row)
                            }
                            for row in rows
                        ]
                        _consume_block_close(lines)
                    else:
                        _skip_block(lines)
                elif stripped == "}":
                    depth -= 1

            yield structure
            count += 1
            if limit is not None and count >= limit:
                return
    finally:
        handle.close()



def _consume_block_close(lines: Iterator[str]) -> None:
    """Eat the closing brace of a sub-block whose table has just been read.

    _read_table stops at the table's trailing ``:::`` and leaves the block's
    own ``}`` unread. Letting that brace fall through to the structure loop
    decrements the structure's depth and ends it early, which is why bonds
    came back empty: the parser closed each pose at the end of its atom table.
    """
    for line in lines:
        if line.strip() == "}":
            return
    raise MaestroParseError("file ended before a block closed")


def _skip_block(lines: Iterator[str]) -> None:
    """Consume a nested block, counting braces."""
    depth = 1
    for line in lines:
        stripped = line.strip()
        if stripped.endswith("{"):
            depth += 1
        elif stripped == "}":
            depth -= 1
            if depth == 0:
                return
    raise MaestroParseError("file ended inside a block")


def read_properties_only(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    """Scores and titles for every structure, without coordinates."""
    return [
        structure.properties
        for structure in iter_structures(path, with_coordinates=False, limit=limit)
    ]


def split_pose_viewer(path: Path) -> tuple[Structure, Iterator[Structure]]:
    """Return the receptor and an iterator over the poses.

    Raises if the first structure is not flagged as the receptor, rather than
    assuming position one is the receptor.
    """
    structures = iter_structures(path)
    try:
        first = next(structures)
    except StopIteration as exc:
        raise MaestroParseError(f"{path} contains no structures") from exc
    if not first.is_receptor:
        raise MaestroParseError(
            f"first structure in {path} is not flagged {RECEPTOR_FLAG}; "
            "this may not be a pose viewer file"
        )
    return first, structures

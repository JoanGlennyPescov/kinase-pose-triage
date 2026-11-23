"""Map a kinase structure onto the KLIFS 85-residue pocket numbering.

Why this exists
---------------
"The gatekeeper" and "the beta-3 lysine" are positions in a structural
alignment, not residue numbers. They differ in every kinase. Any rule about
where a ligand should and should not sit has to be expressed in alignment
positions and then translated into the residue numbers of the structure in
hand. That translation is this module's only job.

Provenance of the landmark positions
------------------------------------
The KLIFS scheme defines 85 binding-site residues covering the front cleft,
gate area and back cleft. The landmark positions below are taken from the
published description of the scheme, not inferred:

  * beta-3 lysine            KLIFS 17
  * gatekeeper               KLIFS 45
  * hinge hydrogen-bond anchor KLIFS 48
  * DFG motif                KLIFS 81, 82, 83
  * catalytic-loop asparagine KLIFS 75

Positions 17, 45, 46-48 and 81-83 were confirmed against the pocket sequence
KLIFS returns for JAK2 (4JI9, structure 3692): position 17 is K, 45 is M, the
JAK2 gatekeeper, 46-48 read EYL, the JAK2 hinge, and 81-83 read DFG.

The catalytic-loop asparagine is set to 75 because that is where the
asparagine falls in that same sequence, with HRD at 68-70 immediately before
it. One published description of the scheme places it at 76, which disagrees.
Rather than pick one on authority, landmarks that have a conserved identity
carry it in EXPECTED_RESIDUE below and every mapping is checked against the
structure. A wrong constant then announces itself instead of silently
mislabelling a residue.

The hinge region is taken as 46-48: the gatekeeper at 45 immediately precedes
the hinge, and 48 is the position at which most ligands hydrogen bond.

Two ways to obtain the mapping
------------------------------
1. From a KLIFS pocket export saved locally (``from_klifs_export``). This is
   the reproducible route and the one a published pipeline should use, because
   the mapping is then pinned in the repository rather than re-fetched.
2. From the KLIFS web API (``fetch_from_api``). Convenient, but the result can
   change under you when KLIFS is updated.

The API endpoint paths are collected in KLIFS_ENDPOINTS below. Both were confirmed by hand on 2026-11-23 against JAK2
structure 3692. ``python src/klifs_pocket.py --check`` re-exercises them and
reports exactly which call failed rather than returning a mapping it is not
sure about.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

LOG = logging.getLogger(__name__)

KLIFS_API_BASE = "https://klifs.net/api_v2"

# Collected in one place so a change to the KLIFS API is a one-line fix.
KLIFS_ENDPOINTS = {
    "structures_for_pdb": "/structures_pdb_list",   # ?pdb-codes=XXXX
    # Confirmed working 2026-11-23 against structure_ID 3692 (JAK2, 4JI9 chain A).
    # Returns rows of {index, Xray_position, KLIFS_position}.
    "pocket_residues": "/interactions_match_residues",  # ?structure_ID=NNNN
}

POCKET_LENGTH = 85

# Landmark alignment positions. See module docstring for provenance.
BETA3_LYSINE = 17
CATALYTIC_ASPARAGINE = 75
GATEKEEPER = 45
HINGE = (46, 47, 48)
HINGE_ANCHOR = 48
DFG = (81, 82, 83)
HRD = (68, 69, 70)

# Positions whose residue identity is conserved across the kinome. Checked on
# every mapping; a mismatch means either the constant is wrong or the
# structure is unusual, and both are things to see rather than to absorb.
EXPECTED_RESIDUE: dict[int, str] = {
    BETA3_LYSINE: "K",
    CATALYTIC_ASPARAGINE: "N",
    HRD[0]: "H",
    HRD[1]: "R",
    HRD[2]: "D",
    DFG[0]: "D",
    DFG[1]: "F",
    DFG[2]: "G",
}

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    # Modified residues seen in activated kinases.
    "SEP": "S", "TPO": "T", "PTR": "Y", "MSE": "M",
    # Protonation-state variants written by Schrodinger's Protein Preparation
    # Wizard. These are ordinary residues with an explicit tautomer or charge
    # state, not modifications, and must not be read as unknown.
    "HIE": "H", "HID": "H", "HIP": "H",
    "ASH": "D", "GLH": "E", "LYN": "K",
    "CYX": "C", "CYM": "C", "ARN": "R",
}

# Named groups used by the plausibility rules downstream.
POCKET_REGIONS: dict[str, tuple[int, ...]] = {
    "beta3_lysine": (BETA3_LYSINE,),
    "gatekeeper": (GATEKEEPER,),
    "hinge": HINGE,
    "hinge_anchor": (HINGE_ANCHOR,),
    "dfg": DFG,
    "hrd": HRD,
    "catalytic_asparagine": (CATALYTIC_ASPARAGINE,),
}


def check_expected_residues(
    pocket_sequence: str,
) -> list[tuple[int, str, str]]:
    """Compare a KLIFS 85-character pocket sequence against the conserved
    landmark identities. Returns (position, expected, observed) for each
    mismatch, empty when all agree.

    A gap character ('-' or '_') is not a mismatch; it means the residue is
    not modelled.
    """
    if len(pocket_sequence) != POCKET_LENGTH:
        raise KlifsError(
            f"pocket sequence is {len(pocket_sequence)} characters, "
            f"expected {POCKET_LENGTH}"
        )
    mismatches = []
    for position, expected in sorted(EXPECTED_RESIDUE.items()):
        observed = pocket_sequence[position - 1]
        if observed in "-_":
            continue
        if observed != expected:
            mismatches.append((position, expected, observed))
    return mismatches


class KlifsError(RuntimeError):
    """Raised when a mapping cannot be obtained. Never silently substituted."""


@dataclass(frozen=True)
class PocketResidue:
    """One position of the KLIFS pocket, resolved onto a structure."""

    klifs_position: int
    residue_name: str | None
    residue_number: int | None
    chain: str | None
    region: str | None = None

    @property
    def is_resolved(self) -> bool:
        """False for gaps: positions with no residue modelled in the structure."""
        return self.residue_number is not None


def parse_region_label(label: str) -> tuple[str, int] | None:
    """Split a KLIFS_position label such as 'GK.45' or 'hinge.46'.

    KLIFS returns the structural region alongside the index, so the region is
    read from the data rather than inferred from a hard-coded integer. Observed
    region labels: I, II, III, IV, V, VI, VII, VIII, g.l, b.l, c.l, a.l, aC,
    aD, aE, GK, hinge, linker, xDFG. Note that 'xDFG' covers four positions,
    80 to 83, where 80 is the x preceding the motif.
    """
    if not label:
        return None
    match = re.match(r"^(?P<region>.+)\.(?P<index>\d+)$", label.strip())
    if not match:
        return None
    return match.group("region"), int(match.group("index"))


@dataclass(frozen=True)
class PocketMapping:
    """KLIFS pocket for one structure."""

    pdb_id: str
    chain: str | None
    structure_id: int | None
    residues: tuple[PocketResidue, ...]
    source: str

    def by_position(self, position: int) -> PocketResidue | None:
        for residue in self.residues:
            if residue.klifs_position == position:
                return residue
        return None

    def residue_numbers(self, region: str) -> list[int]:
        """Residue numbers for a named region, omitting unmodelled positions."""
        if region not in POCKET_REGIONS:
            raise KeyError(
                f"unknown region {region!r}; known: {sorted(POCKET_REGIONS)}"
            )
        numbers = []
        for position in POCKET_REGIONS[region]:
            residue = self.by_position(position)
            if residue is not None and residue.is_resolved:
                numbers.append(residue.residue_number)
        return numbers

    def missing_regions(self) -> list[str]:
        """Regions with no modelled residue at all. Worth refusing to score on."""
        return [
            region for region in POCKET_REGIONS if not self.residue_numbers(region)
        ]

    def describe(self) -> str:
        lines = [
            f"{self.pdb_id} chain {self.chain} "
            f"(KLIFS structure {self.structure_id}, source: {self.source})",
            f"  resolved positions: "
            f"{sum(1 for r in self.residues if r.is_resolved)}/{POCKET_LENGTH}",
        ]
        for region in POCKET_REGIONS:
            residues = [
                self.by_position(position) for position in POCKET_REGIONS[region]
            ]
            rendered = ", ".join(
                f"{r.residue_name}{r.residue_number}"
                if r is not None and r.is_resolved
                else "unmodelled"
                for r in residues
            )
            lines.append(f"  {region:<22}: {rendered}")
        missing = self.missing_regions()
        if missing:
            lines.append(f"  MISSING REGIONS: {', '.join(missing)}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Route 1: a local export. Reproducible, and what a published run should use.
# ---------------------------------------------------------------------------


def from_klifs_export(
    path: Path, pdb_id: str, chain: str | None = None
) -> PocketMapping:
    """Build a mapping from a KLIFS residue export saved as CSV or JSON.

    Expected fields per row, under any of the accepted aliases:
        KLIFS position : 'index', 'klifs_position', 'KLIFS_position'
        residue number : 'Xray_position', 'residue_number', 'resnum'
        residue name   : 'resname', 'residue_name', 'amino_acid'

    A row whose residue number is '_' or blank marks an unmodelled position and
    is preserved as a gap rather than dropped, because a gap at the hinge is a
    reason to exclude the structure, not something to paper over.
    """
    if not path.exists():
        raise KlifsError(f"KLIFS export not found: {path}")

    rows: list[dict[str, Any]]
    text = path.read_text()
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
        rows = payload if isinstance(payload, list) else payload.get("residues", [])
    else:
        rows = list(csv.DictReader(text.splitlines()))

    if not rows:
        raise KlifsError(f"no rows in {path}")

    residues = _rows_to_residues(rows, chain)
    return PocketMapping(
        pdb_id=pdb_id.lower(),
        chain=chain,
        structure_id=None,
        residues=residues,
        source=f"export:{path.name}",
    )


def _first_present(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


def _rows_to_residues(
    rows: Iterable[dict[str, Any]], chain: str | None
) -> tuple[PocketResidue, ...]:
    resolved: dict[int, PocketResidue] = {}
    for row in rows:
        position_raw = _first_present(
            row, "index", "klifs_position", "KLIFS_position", "position"
        )
        if position_raw is None:
            continue
        try:
            position = int(position_raw)
        except (TypeError, ValueError):
            continue

        number_raw = _first_present(
            row, "Xray_position", "residue_number", "resnum", "residue.id"
        )
        residue_number: int | None
        try:
            residue_number = int(str(number_raw).strip())
        except (TypeError, ValueError):
            residue_number = None  # '_' or blank: unmodelled position

        label = _first_present(row, "KLIFS_position", "residue.klifs_region", "region")
        parsed = parse_region_label(str(label)) if label else None
        region = parsed[0] if parsed else None
        if parsed and parsed[1] != position:
            LOG.warning(
                "row index %s disagrees with its label %s; trusting the label",
                position,
                label,
            )
            position = parsed[1]

        resolved[position] = PocketResidue(
            klifs_position=position,
            residue_name=_first_present(
                row, "resname", "residue_name", "amino_acid", "residue.name"
            ),
            residue_number=residue_number,
            chain=chain,
            region=region,
        )

    if not resolved:
        raise KlifsError("no usable rows: expected a KLIFS position column")

    missing_positions = set(range(1, POCKET_LENGTH + 1)) - set(resolved)
    if missing_positions:
        LOG.warning(
            "export covers %d/%d positions; %d absent entirely",
            len(resolved),
            POCKET_LENGTH,
            len(missing_positions),
        )
    for position in missing_positions:
        resolved[position] = PocketResidue(position, None, None, chain, None)

    return tuple(resolved[position] for position in sorted(resolved))



def annotate_from_structure(
    mapping: PocketMapping, pdb_path: Path, chain: str | None = None
) -> tuple[PocketMapping, list[str]]:
    """Fill residue names from the structure, and validate the mapping against it.

    Returns the annotated mapping and a list of problems. Problems are not
    raised, because the caller may legitimately want to see a partially
    resolvable structure; they must not be ignored silently either, which is
    why they are returned rather than logged and dropped.
    """
    chain = chain or mapping.chain
    observed: dict[int, str] = {}
    for line in pdb_path.read_text().splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        if chain is not None and line[21] != chain:
            continue
        try:
            observed[int(line[22:26])] = line[17:20].strip()
        except ValueError:
            continue

    problems: list[str] = []
    annotated: list[PocketResidue] = []
    for residue in mapping.residues:
        name = None
        if residue.residue_number is not None:
            name = observed.get(residue.residue_number)
            if name is None:
                problems.append(
                    f"KLIFS position {residue.klifs_position} maps to residue "
                    f"{residue.residue_number}, which is not modelled in "
                    f"{pdb_path.name} chain {chain}"
                )
        annotated.append(
            PocketResidue(
                klifs_position=residue.klifs_position,
                residue_name=name,
                residue_number=residue.residue_number,
                chain=chain,
                region=residue.region,
            )
        )

    for position, expected in sorted(EXPECTED_RESIDUE.items()):
        residue = next(
            (r for r in annotated if r.klifs_position == position), None
        )
        if residue is None or residue.residue_name is None:
            continue
        one_letter = THREE_TO_ONE.get(residue.residue_name.upper())
        if one_letter is None:
            problems.append(
                f"KLIFS position {position}: unrecognised residue "
                f"{residue.residue_name}"
            )
        elif one_letter != expected:
            problems.append(
                f"KLIFS position {position}: expected {expected}, structure has "
                f"{residue.residue_name} ({one_letter}) at {residue.residue_number}"
            )

    return (
        PocketMapping(
            pdb_id=mapping.pdb_id,
            chain=chain,
            structure_id=mapping.structure_id,
            residues=tuple(annotated),
            source=f"{mapping.source}+{pdb_path.name}",
        ),
        problems,
    )


# ---------------------------------------------------------------------------
# Route 2: the web API.
# ---------------------------------------------------------------------------


def _get(path: str, params: dict[str, Any], timeout: int = 30) -> Any:
    url = f"{KLIFS_API_BASE}{path}?{urlencode(params, doseq=True)}"
    try:
        request = Request(url, headers={"Accept": "application/json"})
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise KlifsError(
            f"KLIFS returned HTTP {exc.code} for {url}. If this is 404, the "
            f"endpoint path in KLIFS_ENDPOINTS is wrong for the current API."
        ) from exc
    except (URLError, TimeoutError) as exc:
        raise KlifsError(f"could not reach KLIFS at {url}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise KlifsError(f"KLIFS returned non-JSON for {url}: {exc}") from exc


def fetch_from_api(pdb_id: str, chain: str | None = None) -> PocketMapping:
    """Resolve the pocket via the KLIFS API.

    Raises rather than guessing when the entry is absent or ambiguous: a kinase
    structure KLIFS does not know about is one this pipeline cannot score.
    """
    pdb_id = pdb_id.lower()
    structures = _get(
        KLIFS_ENDPOINTS["structures_for_pdb"], {"pdb-codes": pdb_id}
    )
    if not structures:
        raise KlifsError(
            f"KLIFS has no structure for {pdb_id}. It may not be a kinase, or "
            "may not yet be curated. Exclude it rather than substituting a "
            "different entry."
        )

    candidates = [
        entry
        for entry in structures
        if chain is None or str(entry.get("chain", "")).upper() == chain.upper()
    ]
    if not candidates:
        available = sorted({str(e.get("chain")) for e in structures})
        raise KlifsError(
            f"{pdb_id}: no KLIFS structure for chain {chain!r}; "
            f"available chains: {available}"
        )
    if len(candidates) > 1 and chain is None:
        chains = sorted({str(e.get("chain")) for e in candidates})
        raise KlifsError(
            f"{pdb_id}: {len(candidates)} KLIFS structures across chains "
            f"{chains}. Pass chain explicitly; copies can differ."
        )

    entry = candidates[0]
    structure_id = entry.get("structure_ID") or entry.get("structure_id")
    if structure_id is None:
        raise KlifsError(f"{pdb_id}: KLIFS entry carries no structure id: {entry}")

    rows = _get(KLIFS_ENDPOINTS["pocket_residues"], {"structure_ID": structure_id})
    residues = _rows_to_residues(rows, chain or entry.get("chain"))
    return PocketMapping(
        pdb_id=pdb_id,
        chain=chain or entry.get("chain"),
        structure_id=int(structure_id),
        residues=residues,
        source="klifs-api",
    )


def check_api() -> int:
    """Exercise the endpoints and report precisely what works."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print(f"KLIFS base: {KLIFS_API_BASE}")
    for name, path in KLIFS_ENDPOINTS.items():
        print(f"  {name}: {path}")
    print("\nAttempting a known kinase entry (4JI9, JAK2)...")
    try:
        mapping = fetch_from_api("4ji9", chain="A")
    except KlifsError as exc:
        print(f"\nFAILED: {exc}")
        print(
            "\nFix KLIFS_ENDPOINTS at the top of this module, or export the "
            "pocket residues from klifs.net and use from_klifs_export instead."
        )
        return 1
    print("\n" + mapping.describe())
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="test API connectivity")
    parser.add_argument("--pdb", help="PDB id to resolve")
    parser.add_argument("--chain", default=None)
    parser.add_argument("--export", type=Path, default=None, help="local KLIFS export")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.check:
        return check_api()
    if not args.pdb:
        parser.error("give --pdb, or --check")

    try:
        if args.export:
            mapping = from_klifs_export(args.export, args.pdb, args.chain)
        else:
            mapping = fetch_from_api(args.pdb, args.chain)
    except KlifsError as exc:
        print(f"ERROR: {exc}")
        return 1
    print(mapping.describe())
    return 0


if __name__ == "__main__":
    sys.exit(main())


# ---------------------------------------------------------------------------
# Route 3: align the KLIFS pocket onto a structure KLIFS has no entry for,
# such as a homology or chimeric model.
# ---------------------------------------------------------------------------

KLIFS_REGION_SPANS: dict[str, tuple[int, int]] = {
    "I": (1, 3), "g.l": (4, 9), "II": (10, 13), "III": (14, 19),
    "aC": (20, 30), "b.l": (31, 37), "IV": (38, 41), "V": (42, 44),
    "GK": (45, 45), "hinge": (46, 48), "linker": (49, 52), "aD": (53, 59),
    "aE": (60, 64), "VI": (65, 67), "c.l": (68, 75), "VII": (76, 78),
    "VIII": (79, 79), "xDFG": (80, 83), "a.l": (84, 85),
}


@dataclass(frozen=True)
class AlignedPosition:
    """One KLIFS position after alignment onto a model."""

    klifs_position: int
    region: str
    expected: str
    residue_number: int | None
    observed: str | None
    status: str  # mapped | outside_construct | mismatch | alignment_gap


def map_pocket_by_alignment(
    pocket_sequence: str,
    residue_numbers: list[int],
    residue_letters: str,
) -> list[AlignedPosition]:
    """Map the 85 KLIFS positions onto a model by region anchoring.

    Regions whose whole sequence occurs exactly once in the model are used as
    anchors. Every other position is then resolved by walking outward from an
    anchor one residue at a time, checking identity as it goes.

    This two-stage approach exists because a region can be only partly present.
    A construct beginning part-way through the glycine-rich loop contains the
    last residues of that region and not the first; matching whole regions
    reports nothing for it, which reads as "absent" when it is "truncated".

    Status values:
      mapped            resolved and the residue identity agrees
      mismatch          resolved by position but the residue differs
      outside_construct the position falls before or after the model
      alignment_gap     the position is a gap in the KLIFS pocket itself
    """
    if len(pocket_sequence) != POCKET_LENGTH:
        raise KlifsError(
            f"pocket sequence is {len(pocket_sequence)}, expected {POCKET_LENGTH}"
        )
    if len(residue_numbers) != len(residue_letters):
        raise KlifsError("residue_numbers and residue_letters differ in length")

    region_of = {
        position: region
        for region, (start, end) in KLIFS_REGION_SPANS.items()
        for position in range(start, end + 1)
    }

    # Stage one: anchor on regions that occur exactly once.
    anchors: dict[int, int] = {}  # klifs position -> index into the model
    for region, (start, end) in KLIFS_REGION_SPANS.items():
        fragment = pocket_sequence[start - 1 : end].replace("-", "")
        if len(fragment) < 2:
            continue  # too short to place unambiguously
        hits = [
            i
            for i in range(len(residue_letters) - len(fragment) + 1)
            if residue_letters[i : i + len(fragment)] == fragment
        ]
        if len(hits) != 1:
            continue
        offset = 0
        for position in range(start, end + 1):
            if pocket_sequence[position - 1] == "-":
                continue
            anchors[position] = hits[0] + offset
            offset += 1

    if not anchors:
        raise KlifsError(
            "no KLIFS region matched the model uniquely; the sequences may be "
            "unrelated, or the model may be too short to anchor"
        )

    # Stage two: extend outward from the anchors, one residue at a time.
    resolved: dict[int, int] = dict(anchors)
    for position in range(1, POCKET_LENGTH + 1):
        if position in resolved or pocket_sequence[position - 1] == "-":
            continue
        # Try anchors nearest first, but accept an extension only when the
        # residue it lands on is the one KLIFS expects. Regions are not
        # contiguous with each other in a structure: in JAK2 the gap between
        # VII.78 and VIII.79 is eight residues. Walking across a boundary and
        # assuming consecutive numbering silently lands on the wrong residue,
        # which is what this check prevents.
        # Only the nearest anchor below and the nearest above are candidates.
        # Searching all anchors until one matches lets a single residue match
        # by chance far from where it belongs: KLIFS positions 2 and 3 were
        # resolved onto residues 790 and 791 of a construct that starts at 766,
        # because E and L happen to occur there.
        below = [p for p in anchors if p < position]
        above = [p for p in anchors if p > position]
        candidates = []
        if below:
            candidates.append(max(below))
        if above:
            candidates.append(min(above))

        for anchor_position in sorted(candidates, key=lambda p: abs(p - position)):
            step = 1 if position > anchor_position else -1
            index = anchors[anchor_position]
            walk = anchor_position
            ok = True
            while walk != position:
                walk += step
                if pocket_sequence[walk - 1] == "-":
                    continue
                index += step
                if not 0 <= index < len(residue_numbers):
                    ok = False
                    break
            if not ok or not 0 <= index < len(residue_numbers):
                continue
            if residue_letters[index] == pocket_sequence[position - 1]:
                resolved[position] = index
                break

    out: list[AlignedPosition] = []
    for position in range(1, POCKET_LENGTH + 1):
        expected = pocket_sequence[position - 1]
        region = region_of.get(position, "?")
        if expected == "-":
            out.append(
                AlignedPosition(position, region, "-", None, None, "alignment_gap")
            )
            continue
        index = resolved.get(position)
        if index is None:
            out.append(
                AlignedPosition(
                    position, region, expected, None, None, "outside_construct"
                )
            )
            continue
        observed = residue_letters[index]
        out.append(
            AlignedPosition(
                position,
                region,
                expected,
                residue_numbers[index],
                observed,
                "mapped" if observed == expected else "mismatch",
            )
        )
    return out


def summarise_alignment(positions: list[AlignedPosition]) -> str:
    counts: dict[str, int] = {}
    for entry in positions:
        counts[entry.status] = counts.get(entry.status, 0) + 1
    lines = [
        "  ".join(f"{status}={count}" for status, count in sorted(counts.items()))
    ]
    truncated = [e for e in positions if e.status == "outside_construct"]
    if truncated:
        spans = ", ".join(str(e.klifs_position) for e in truncated)
        lines.append(f"positions outside the construct: {spans}")
    mismatches = [e for e in positions if e.status == "mismatch"]
    for entry in mismatches:
        lines.append(
            f"position {entry.klifs_position} ({entry.region}): KLIFS says "
            f"{entry.expected}, model has {entry.observed}{entry.residue_number}"
        )
    return "\n".join(lines)

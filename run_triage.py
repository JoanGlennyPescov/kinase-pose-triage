#!/usr/bin/env python3
"""Triage a Glide pose-viewer file and say which compounds are worth an MD run.

Usage:
    python run_triage.py <file_pv.maegz or .mae> [--top 50] [--scan 3000] [-o out.txt]

Prints, and optionally writes, two lists: compounds to reject with the reason,
and compounds recommended for MD. Decisions are made per compound using that
compound's best clean pose, not per pose.
"""

from __future__ import annotations

import argparse
import gzip
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from triage import (  # noqa: E402
    triage, RULESET_VERSION, CLASH_OVERLAP, AROMATIC_FRACTION_MIN,
)


def _maybe_gunzip(path: Path) -> tuple[Path, Path | None]:
    """Glide writes .maegz. Decompress to a temp file if needed."""
    if path.suffix.lower() != ".maegz":
        return path, None
    tmp = Path(tempfile.mkdtemp()) / (path.stem + ".mae")
    with gzip.open(path, "rb") as src, tmp.open("wb") as dst:
        shutil.copyfileobj(src, dst)
    return tmp, tmp.parent


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("posefile", type=Path)
    ap.add_argument("--top", type=int, default=50, help="how many top-scoring poses to triage (default 50)")
    ap.add_argument("--scan", type=int, default=3000, help="poses read to build the strain baseline (default 3000)")
    ap.add_argument("-o", "--output", type=Path, default=None)
    ap.add_argument("--json", type=Path, default=None,
                    help="write a machine-readable report for a downstream agent")
    ap.add_argument("--hinge", type=str, default=None,
                    help="comma-separated receptor residue numbers for KLIFS pocket "
                         "positions 46-48; omit to skip the aromatic-facing warning")
    args = ap.parse_args()

    if not args.posefile.exists():
        print(f"no such file: {args.posefile}", file=sys.stderr)
        return 1

    hinge = tuple(int(x) for x in args.hinge.split(",")) if args.hinge else None

    work, tmpdir = _maybe_gunzip(args.posefile)
    try:
        ranked, strain_cut, screened, gridfile, softened = triage(
            work, top_n=args.top, limit=args.scan, hinge_residues=hinge
        )
    finally:
        if tmpdir is not None:
            shutil.rmtree(tmpdir, ignore_errors=True)

    # per compound, keep the best-scoring pose that is clean; else its best pose
    best: dict[str, object] = {}
    for r in ranked:
        cur = best.get(r.title)
        if cur is None:
            best[r.title] = r
            continue
        better = (r.verdict == "KEEP" and cur.verdict == "REJECT") or (
            r.verdict == cur.verdict and r.score < cur.score
        )
        if better:
            best[r.title] = r

    keeps = sorted([r for r in best.values() if r.verdict == "KEEP"], key=lambda r: r.score)
    rejects = sorted([r for r in best.values() if r.verdict == "REJECT"], key=lambda r: r.score)

    lines: list[str] = []
    w = lines.append
    w(f"file            : {args.posefile.name}")
    w(f"grid            : {gridfile}")
    if softened:
        w("                  ** SOFTENED GRID (vdW radii scaled down). Glide was told to")
        w("                     allow interpenetration, so close contacts here are produced")
        w("                     by the protocol. Poses from this file are NOT score-comparable")
        w("                     with standard-grid runs, and clashes below should be treated")
        w("                     as 'needs re-minimisation on a standard grid'. **")
    w(f"poses screened  : {screened}")
    w(f"triaged         : top {len(ranked)} poses -> {len(best)} distinct compounds")
    w(f"strain cutoff   : einternal > {strain_cut:.1f}")
    w("")
    w(f"REJECT ({len(rejects)} compounds) - do not spend MD time on these")
    for r in rejects:
        w(f"  {r.title:22}{r.score:7.2f}   {r.reason()}")
    w("")
    w(f"RECOMMENDED FOR MD ({len(keeps)} compounds)")
    for r in keeps:
        extra = "   [" + "; ".join(r.warnings) + "]" if r.warnings else ""
        arom = "n/a" if r.aromatic_fraction is None else f"{r.aromatic_fraction:.0%}"
        strain = "n/a" if r.strain is None else f"{r.strain:5.1f}"
        w(f"  {r.title:22}{r.score:7.2f}   strain {strain}   vdW overlap {r.worst_overlap:4.2f}   "
          f"rings at hinge {arom}{extra}")

    text = "\n".join(lines)
    print(text)
    if args.output:
        args.output.write_text(text + "\n")
        print(f"\nwritten to {args.output}")

    if args.json:
        import json
        payload = {
            "ruleset_version": RULESET_VERSION,
            "source_file": args.posefile.name,
            "grid": gridfile,
            "grid_softened": bool(softened),
            "poses_screened": screened,
            "top_n": len(ranked),
            "thresholds": {
                "strain_cutoff": strain_cut,
                "strain_source": "r_i_glide_einternal",
                "strain_validated": False,
                "clash_overlap_A": CLASH_OVERLAP,
                "aromatic_fraction_min": AROMATIC_FRACTION_MIN,
                "aromatic_rule_is_warning_only": True,
            },
            "compounds": [
                {
                    "compound_id": r.title,
                    "docking_score": r.score,
                    "ligand_efficiency": r.ligand_efficiency,
                    "strain": r.strain,
                    "worst_vdw_overlap_A": r.worst_overlap,
                    "closest_nonpolar_contact_A": (
                        None if r.closest_nonpolar == float("inf") else r.closest_nonpolar
                    ),
                    "aromatic_hinge_fraction": r.aromatic_fraction,
                    "grid_softened": bool(softened),
                    "verdict": r.verdict,
                    "reason": r.reason(),
                    "warnings": r.warnings,
                }
                for r in sorted(best.values(), key=lambda x: x.score)
            ],
        }
        args.json.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"json written to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

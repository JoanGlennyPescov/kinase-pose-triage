"""Triage kinase docking poses: rank by score, then flag those that cannot bind.

Hand it a docking output and it returns which compounds are worth an MD run,
with a measured reason for each. Every number is read or computed from the pose
file; nothing is inferred.

Two signals can reject a pose, one only warns.

1. Steric clash  (computed from coordinates)
   Van der Waals overlap, not raw distance. Any ligand-receptor heavy-atom pair
   more than CLASH_OVERLAP inside the sum of their Bondi radii, excluding N/O
   to N/O pairs, which are hydrogen bonds rather than clashes. A flat distance
   cutoff cannot work: a ligand N 2.69 A from a carbon is 0.56 A inside vdW and
   occurs in experimentally validated poses, while a nitrogen 1.48 A from a
   carbon is 1.77 A inside and is a covalent bond length.

2. Ligand internal strain  (engine-specific, NOT VALIDATED)
   Read from r_i_glide_einternal where Glide wrote it. This rule currently does
   all of the rejecting on standard grids and none of it on softened ones, and
   its threshold rests on a small number of compounds. An MMFF energy
   difference does NOT reproduce it (Spearman 0.35-0.44 over 400 poses, with
   4-6 of 40 flagged poses in common across every minimisation variant tried),
   so the threshold cannot be transferred to another engine and at least one of
   the two metrics is not measuring real strain. Treat accordingly.

3. Aromatic facing at the hinge  (warns only)
   The fraction of hinge contacts made by ring atoms rather than polar atoms.
   The hinge is largely hydrophobic and the conserved backbone contacts across
   the kinome are with ring systems, so a pose presenting only polar atoms
   there has its rings pointed at solvent. This never rejects: a compound with
   a stable MD trajectory was measured at 13%, in the bottom 5% of its library.

Hinge positions come from the KLIFS 85-position pocket numbering (positions
46-48), mapped onto whatever receptor is supplied. No residue numbers are
hardcoded; pass them via `hinge_residues`.

Deliberately NOT used:

- Any b_glide_conssat_* flag. These record docking constraints imposed by the
  person who ran the docking. Where a hinge hydrogen-bond constraint was used,
  the flag is True on essentially every pose and carries no information.
  Check for it before trusting any hinge-contact feature.
- The hinge hydrogen-bond distance, for the same reason.
- Any predicted-activity columns a library may carry.

Known limits, stated so they are not overclaimed:

- Thresholds are percentile-based WITHIN one pose file. Absolute contact counts
  are not comparable across receptor conformations or across grids built with
  different van der Waals scaling.
- This does not predict MD stability, and the evidence suggests static pose
  geometry may be insufficient for that: two near-identical chemotypes with
  similar contact maps can differ entirely in whether the ligand stays bound,
  with the difference appearing only over hundreds of nanoseconds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from read_maestro import split_pose_viewer
from contacts import receptor_atoms, ligand_atoms

# A receptor-scale entry is a complex (receptor + ligand) written into the pose
# stream, not a pose. Scoring one yields a meaningless 0.00 A "contact".
RULESET_VERSION = "0.1.0"

MAX_POSE_ATOMS = 300

POLAR_ELEMENTS = {7, 8}
CLASH_LIMIT = 3.10
CLASH_SEVERE = 2.90

# Bondi van der Waals radii (A). A clash is judged by how far inside the sum of
# radii a pair sits, not by a flat distance: N..C at 2.69 is 0.56 A inside and
# occurs in validated poses, while C..C at 2.36 is 1.04 A inside and cannot.
VDW_RADII = {1: 1.20, 6: 1.70, 7: 1.55, 8: 1.52, 9: 1.47, 15: 1.80, 16: 1.80, 17: 1.75, 35: 1.85}
CLASH_OVERLAP = 0.80

# Grids built with a van der Waals scaling factor let ligand and receptor atoms
# interpenetrate on purpose. Absolute clash detection is meaningless on those,
# and poses from a softened grid cannot be score-compared with standard ones.
SOFT_GRID_MARKERS = ("sf05", "sf0.5", "sf_0.5", "sf04", "sf0.4")


def grid_is_softened(gridfile: str | None) -> bool:
    if not gridfile:
        return False
    name = gridfile.lower()
    return any(m in name for m in SOFT_GRID_MARKERS)

STRAIN_PERCENTILE = 90.0
STRAIN_FLOOR = 8.0
AROMATIC_FRACTION_MIN = 0.35

# KLIFS pocket positions for the hinge. These are alignment positions in the
# 85-residue kinase pocket numbering, valid for any kinase; the receptor residue
# numbers they map to are supplied per target, never hardcoded here.
KLIFS_HINGE_POSITIONS = (46, 47, 48)
KLIFS_HINGE_ANCHOR = 48


@dataclass
class PoseReport:
    title: str
    score: float
    strain: float | None
    ligand_efficiency: float | None
    closest_nonpolar: float
    nonpolar_count: int
    aromatic_fraction: float | None
    worst_overlap: float = 0.0
    flags: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        return "REJECT" if self.flags else "KEEP"

    def reason(self) -> str:
        if not self.flags:
            bits = [f"no non-polar contact under {CLASH_LIMIT} A"] if self.nonpolar_count == 0 else []
            if self.strain is not None:
                bits.append(f"low strain ({self.strain:.1f})")
            if self.aromatic_fraction is not None:
                bits.append(f"rings at hinge ({self.aromatic_fraction:.0%})")
            return "; ".join(bits) or "no disqualifying geometry"
        return "; ".join(self.flags)


def _nonpolar_contacts(ligand, tree, rz, positions):
    """Ligand-receptor approaches that cannot be hydrogen bonds, as vdW overlap.

    Returns (distances, overlaps). A pair of N/O atoms is skipped: that is a
    hydrogen bond, not a clash. Everything else is scored by how far inside the
    sum of van der Waals radii it sits.
    """
    distances, overlaps = [], []
    for atom in ligand:
        ri = VDW_RADII.get(atom.atomic_number, 1.70)
        for j in tree.query_ball_point(atom.position, CLASH_LIMIT):
            if atom.atomic_number in POLAR_ELEMENTS and rz[j] in POLAR_ELEMENTS:
                continue
            d = float(np.linalg.norm(np.asarray(atom.position) - positions[j]))
            distances.append(d)
            overlaps.append((ri + VDW_RADII.get(int(rz[j]), 1.70)) - d)
    return distances, overlaps


def _aromatic_fraction(profile, hinge_residues) -> float | None:
    if not hinge_residues:
        return None
    counts = profile.contacts_with(list(hinge_residues))
    total = sum(counts.values())
    if not total:
        return None
    aromatic = counts.get("aromatic", 0) + counts.get("aromatic_polar", 0)
    return aromatic / total


def triage(
    path: Path,
    top_n: int = 50,
    limit: int | None = None,
    hinge_residues: tuple[int, ...] | None = None,
):
    """Rank poses by docking score and flag implausible geometry among the top N.

    hinge_residues: receptor residue numbers for KLIFS pocket positions 46-48.
    Obtain them from the KLIFS mapping for the target (see klifs_pocket.py).
    When omitted, the aromatic-facing warning is skipped; clash and strain still
    apply, since neither depends on the pocket alignment.
    """
    from contacts import PocketIndex

    receptor, poses = split_pose_viewer(path)
    rec_atoms = receptor_atoms(receptor)
    positions = np.array([a.position for a in rec_atoms])
    rz = np.array([a.atomic_number for a in rec_atoms])
    tree = cKDTree(positions)
    index = PocketIndex(rec_atoms)

    collected = []
    gridfile = None
    for pose in poses:
        if limit is not None and len(collected) >= limit:
            break
        if len(pose.atoms) > MAX_POSE_ATOMS:
            continue  # receptor-scale complex entry, not a pose
        ligand = ligand_atoms(pose)
        if not ligand:
            continue
        score = pose.properties.get("r_i_docking_score")
        if score is None:
            continue
        if gridfile is None:
            gridfile = pose.properties.get("s_i_glide_gridfile")
        dists, overlaps = _nonpolar_contacts(ligand, tree, rz, positions)
        profile = index.profile(ligand, pose.title, score)
        collected.append(
            PoseReport(
                title=pose.title,
                score=float(score),
                strain=pose.properties.get("r_i_glide_einternal"),
                ligand_efficiency=pose.properties.get("r_i_glide_ligand_efficiency"),
                closest_nonpolar=min(dists) if dists else float("inf"),
                nonpolar_count=len(dists),
                worst_overlap=max(overlaps) if overlaps else 0.0,
                aromatic_fraction=_aromatic_fraction(profile, hinge_residues),
            )
        )

    strains = [r.strain for r in collected if r.strain is not None]
    strain_cut = max(STRAIN_FLOOR, float(np.percentile(strains, STRAIN_PERCENTILE))) if strains else STRAIN_FLOOR

    softened = grid_is_softened(gridfile)

    ranked = sorted(collected, key=lambda r: r.score)[:top_n]
    for r in ranked:
        if r.strain is not None and r.strain > strain_cut:
            r.flags.append(f"strained conformer (einternal {r.strain:.1f} > {strain_cut:.1f})")
        if r.worst_overlap > CLASH_OVERLAP:
            note = " (grid was softened, so Glide permitted this)" if softened else ""
            r.flags.append(
                f"steric clash ({r.closest_nonpolar:.2f} A, {r.worst_overlap:.2f} A inside vdW){note}"
            )
        # Aromatic facing is a WARNING, never a rejection: a compound with a
        # stable MD trajectory was measured at 13%, in the bottom 5% of its
        # library, so rejecting on this would discard a known-good compound.
        if r.aromatic_fraction is not None and r.aromatic_fraction < AROMATIC_FRACTION_MIN:
            r.warnings.append(f"rings mostly face away from hinge ({r.aromatic_fraction:.0%} aromatic)")

    return ranked, strain_cut, len(collected), gridfile, softened

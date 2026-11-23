"""Contacts between a docked pose and the pocket, in kinase terms.

Reports what is geometrically true. It does not decide whether a pose is good;
that belongs to the rules, which need these facts first.

Three things are computed:

  * which pocket residues each part of the ligand touches, split by whether the
    ligand atom is aromatic, apolar or polar, because "the ligand contacts the
    hinge" means something different for a ring carbon than for a carbonyl;
  * steric overlap with the receptor, since a pose can score well and still
    push into the protein;
  * polar ligand atoms that are buried with no hydrogen-bond partner, which is
    the signature of a flipped binding mode.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np
from rdkit import Chem, RDLogger
from scipy.spatial import cKDTree

RDLogger.DisableLog("rdApp.*")

LOG = logging.getLogger(__name__)

# Bondi van der Waals radii, angstrom, for the elements seen in these files.
VDW_RADII: dict[int, float] = {
    1: 1.20, 6: 1.70, 7: 1.55, 8: 1.52, 9: 1.47, 15: 1.80,
    16: 1.80, 17: 1.75, 35: 1.85, 53: 1.98, 34: 1.90, 5: 1.92,
}
DEFAULT_VDW = 1.70

CONTACT_CUTOFF = 4.0          # heavy-atom contact distance
HBOND_CUTOFF = 3.5            # donor/acceptor heavy-atom separation
CLASH_TOLERANCE = 0.5         # overlap beyond this is reported
BURIAL_NEIGHBOUR_CUTOFF = 5.0 # receptor heavy atoms within this count as buried
BURIAL_THRESHOLD = 12         # neighbours needed before a polar atom is "buried"

HBOND_ELEMENTS = {7, 8}       # N, O. S is a weak partner and excluded here.


@dataclass
class LigandAtom:
    index: int
    atomic_number: int
    position: tuple[float, float, float]
    is_aromatic: bool = False
    in_ring: bool = False
    is_polar: bool = False

    @property
    def kind(self) -> str:
        """Primary label, kept for compatibility. See kinds() for the real answer."""
        if self.is_aromatic:
            return "aromatic"
        if self.is_polar:
            return "polar"
        return "apolar"

    def kinds(self) -> tuple[str, ...]:
        """Every label that applies.

        These categories are not exclusive and treating them as such was wrong.
        A kinase hinge binder typically donates or accepts through a nitrogen
        that is part of its aromatic core. Labelling that atom "polar" and not
        "aromatic" made a validated hinge binder report zero aromatic contact
        at the hinge, which is the opposite of what is happening.
        """
        labels = []
        if self.is_aromatic:
            labels.append("aromatic")
        if self.is_polar:
            labels.append("polar")
        if self.in_ring and not self.is_aromatic:
            labels.append("ring")
        if not self.is_polar and not self.is_aromatic:
            labels.append("apolar")
        if self.is_aromatic and self.is_polar:
            labels.append("aromatic_polar")
        return tuple(labels)


@dataclass
class ReceptorAtom:
    index: int
    atomic_number: int
    position: tuple[float, float, float]
    residue_number: int
    residue_name: str
    atom_name: str
    chain: str

    @property
    def is_backbone(self) -> bool:
        return self.atom_name.strip() in {"N", "CA", "C", "O"}


@dataclass
class Clash:
    ligand_index: int
    receptor_residue: int
    receptor_atom: str
    distance: float
    overlap: float


@dataclass
class BuriedPolar:
    ligand_index: int
    atomic_number: int
    neighbour_count: int
    nearest_partner: float | None


@dataclass
class ContactProfile:
    """Everything measured for one pose."""

    title: str | None
    docking_score: float | None
    residue_contacts: dict[int, dict[str, int]] = field(default_factory=dict)
    clashes: list[Clash] = field(default_factory=list)
    buried_polars: list[BuriedPolar] = field(default_factory=list)
    worst_overlap: float = 0.0

    def contacts_with(self, residues: Iterable[int]) -> dict[str, int]:
        """Counts by ligand atom kind against a set of residues."""
        totals = {"aromatic": 0, "apolar": 0, "polar": 0,
                  "ring": 0, "aromatic_polar": 0}
        for residue in residues:
            for kind, count in self.residue_contacts.get(residue, {}).items():
                totals[kind] = totals.get(kind, 0) + count
        return totals

    def touches(self, residues: Iterable[int]) -> bool:
        return any(self.residue_contacts.get(r) for r in residues)


def _vdw(atomic_number: int) -> float:
    return VDW_RADII.get(atomic_number, DEFAULT_VDW)


def molecule_from_structure(structure: Any) -> Chem.Mol | None:
    """Build an RDKit molecule from a Maestro structure, for ring perception.

    Returns None when sanitisation fails. Callers fall back to element-only
    classification rather than silently treating a failure as "no aromatics".
    """
    editable = Chem.RWMol()
    index_map: dict[int, int] = {}
    for position, atom in enumerate(structure.atoms, start=1):
        atomic_number = atom.get("i_m_atomic_number")
        if atomic_number is None:
            return None
        rd_atom = Chem.Atom(int(atomic_number))
        # LigPrep protonates basic centres, so a nitrogen can carry four bonds.
        # The charge lives in its own column; ignoring it makes RDKit reject the
        # molecule for exceeding nitrogen's valence, and the failure is silent.
        # That dropped ring and aromaticity perception for about 40% of poses.
        formal_charge = atom.get("i_m_formal_charge")
        if formal_charge:
            rd_atom.SetFormalCharge(int(formal_charge))
        index_map[position] = editable.AddAtom(rd_atom)

    order_map = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE,
                 3: Chem.BondType.TRIPLE}
    for bond in structure.bonds:
        start, end = bond.get("i_m_from"), bond.get("i_m_to")
        if start not in index_map or end not in index_map:
            continue
        editable.AddBond(
            index_map[start],
            index_map[end],
            order_map.get(bond.get("i_m_order", 1), Chem.BondType.SINGLE),
        )

    conformer = Chem.Conformer(editable.GetNumAtoms())
    for position, atom in enumerate(structure.atoms, start=1):
        conformer.SetAtomPosition(
            index_map[position],
            (atom["r_m_x_coord"], atom["r_m_y_coord"], atom["r_m_z_coord"]),
        )
    molecule = editable.GetMol()
    molecule.AddConformer(conformer)
    try:
        Chem.SanitizeMol(molecule)
    except Exception:
        return None
    return molecule


def ligand_atoms(structure: Any) -> list[LigandAtom]:
    """Heavy atoms of a pose, classified aromatic, apolar or polar."""
    molecule = molecule_from_structure(structure)
    atoms: list[LigandAtom] = []
    for position, raw in enumerate(structure.atoms):
        atomic_number = int(raw.get("i_m_atomic_number") or 0)
        if atomic_number <= 1:
            continue  # hydrogens handled implicitly
        atom = LigandAtom(
            index=position + 1,
            atomic_number=atomic_number,
            position=(raw["r_m_x_coord"], raw["r_m_y_coord"], raw["r_m_z_coord"]),
            is_polar=atomic_number in HBOND_ELEMENTS,
        )
        if molecule is not None:
            rd_atom = molecule.GetAtomWithIdx(position)
            atom.is_aromatic = rd_atom.GetIsAromatic()
            atom.in_ring = rd_atom.IsInRing()
        atoms.append(atom)
    return atoms


def receptor_atoms(structure: Any) -> list[ReceptorAtom]:
    """Heavy atoms of the receptor, with residue identity."""
    out: list[ReceptorAtom] = []
    for position, raw in enumerate(structure.atoms):
        atomic_number = int(raw.get("i_m_atomic_number") or 0)
        if atomic_number <= 1:
            continue
        out.append(
            ReceptorAtom(
                index=position + 1,
                atomic_number=atomic_number,
                position=(
                    raw["r_m_x_coord"],
                    raw["r_m_y_coord"],
                    raw["r_m_z_coord"],
                ),
                residue_number=int(raw.get("i_m_residue_number") or 0),
                residue_name=(raw.get("s_m_pdb_residue_name") or "").strip(),
                atom_name=(raw.get("s_m_pdb_atom_name") or "").strip(),
                chain=(raw.get("s_m_chain_name") or "").strip(),
            )
        )
    return out


class PocketIndex:
    """Spatial index over the receptor, built once and reused for every pose."""

    def __init__(self, atoms: Sequence[ReceptorAtom]) -> None:
        self.atoms = list(atoms)
        self.coordinates = np.array([a.position for a in self.atoms])
        self.tree = cKDTree(self.coordinates)
        self.radii = np.array([_vdw(a.atomic_number) for a in self.atoms])

    def profile(
        self,
        pose_atoms: Sequence[LigandAtom],
        title: str | None = None,
        docking_score: float | None = None,
    ) -> ContactProfile:
        profile = ContactProfile(title=title, docking_score=docking_score)
        if not pose_atoms:
            return profile

        ligand_coordinates = np.array([a.position for a in pose_atoms])

        # Contacts, by residue and ligand atom kind.
        neighbour_lists = self.tree.query_ball_point(
            ligand_coordinates, r=CONTACT_CUTOFF
        )
        for ligand_position, neighbours in enumerate(neighbour_lists):
            labels = pose_atoms[ligand_position].kinds()
            for receptor_position in neighbours:
                residue = self.atoms[receptor_position].residue_number
                bucket = profile.residue_contacts.setdefault(
                    residue,
                    {"aromatic": 0, "apolar": 0, "polar": 0,
                     "ring": 0, "aromatic_polar": 0},
                )
                for label in labels:
                    bucket[label] = bucket.get(label, 0) + 1

        # Steric overlap. A clash is measured against summed vdW radii, not a
        # fixed distance, so a contact that is close for carbon is not reported
        # as a clash for fluorine.
        clash_lists = self.tree.query_ball_point(ligand_coordinates, r=5.0)
        for ligand_position, neighbours in enumerate(clash_lists):
            ligand_radius = _vdw(pose_atoms[ligand_position].atomic_number)
            for receptor_position in neighbours:
                distance = float(
                    np.linalg.norm(
                        ligand_coordinates[ligand_position]
                        - self.coordinates[receptor_position]
                    )
                )
                allowed = ligand_radius + self.radii[receptor_position]
                overlap = allowed - distance
                if overlap > CLASH_TOLERANCE:
                    receptor_atom = self.atoms[receptor_position]
                    profile.clashes.append(
                        Clash(
                            ligand_index=pose_atoms[ligand_position].index,
                            receptor_residue=receptor_atom.residue_number,
                            receptor_atom=receptor_atom.atom_name,
                            distance=round(distance, 2),
                            overlap=round(float(overlap), 2),
                        )
                    )
                    profile.worst_overlap = max(
                        profile.worst_overlap, round(float(overlap), 2)
                    )

        # Buried polar atoms with no partner.
        for ligand_position, atom in enumerate(pose_atoms):
            if not atom.is_polar:
                continue
            neighbours = self.tree.query_ball_point(
                ligand_coordinates[ligand_position], r=BURIAL_NEIGHBOUR_CUTOFF
            )
            if len(neighbours) < BURIAL_THRESHOLD:
                continue
            partners = [
                float(
                    np.linalg.norm(
                        ligand_coordinates[ligand_position]
                        - self.coordinates[receptor_position]
                    )
                )
                for receptor_position in neighbours
                if self.atoms[receptor_position].atomic_number in HBOND_ELEMENTS
            ]
            nearest = min(partners) if partners else None
            if nearest is None or nearest > HBOND_CUTOFF:
                profile.buried_polars.append(
                    BuriedPolar(
                        ligand_index=atom.index,
                        atomic_number=atom.atomic_number,
                        neighbour_count=len(neighbours),
                        nearest_partner=round(nearest, 2) if nearest else None,
                    )
                )
        return profile

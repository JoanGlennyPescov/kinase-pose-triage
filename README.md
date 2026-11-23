# kinase-pose-triage

Geometric triage for kinase docking poses. Flags the ones that cannot bind that way, before you spend GPU hours on MD.

Docking scores rank compounds, but a good score does not mean the pose is physically sensible. In a typical top-50 list some poses have atoms interpenetrating, some hold a strained conformer, and some point their rings at solvent instead of into the hinge. Finding those by eye in Maestro takes hours.

Kinases only. The pocket logic is anchored on KLIFS 85-position numbering, so there is no meaningful port to GPCRs or other target classes.

```bash
python run_triage.py path/to/poses_pv.maegz --hinge 123,124,125 --json report.json
```

Reads `.maegz` directly. No Schrodinger licence, no Maestro, no GPU.

## Start here: check your grid

The most useful thing this tool does needs no calibration and rests on no assumptions.

Glide writes the grid name into every pose as `s_i_glide_gridfile`. A grid built with a van der Waals scaling factor, marked `sf05` or `sf0.5` in the name, shrinks the receptor's radii so that Glide is explicitly permitted to let atoms interpenetrate. Across five docking runs on one target, the effect was absolute:

| grid | worst vdW overlap in top 50 |
| --- | --- |
| standard (2 runs) | 0.63 A, 0.58 A |
| softened `sf05` (3 runs) | 1.77 A, 1.78 A, 1.73 A |

The standard-grid runs contained no genuine clashes at all. Every physically impossible contact came from a softened grid, including in one run the single best-scoring compound in the file, which had a 1.48 A heavy-atom contact, shorter than a carbon-nitrogen bond, and still reported a favourable `evdw` because the potential used to score it was soft.

Two consequences:

- **Scores from a softened run are not comparable with scores from a standard run.** The soft potential inflates the favourable vdW term. Do not merge them into one ranking.
- A clash flagged on a softened run means "re-minimise on a standard grid before trusting this", not necessarily "bad compound".

If you have been pooling soft-grid and standard-grid results into one top-N, that is probably the largest single source of implausible poses in your list.

## What it measures

Three signals. Two can reject, one only warns.

### Steric clash (rejects)

Van der Waals overlap, not raw distance. Any ligand-receptor heavy-atom pair more than 0.80 A inside the sum of their Bondi radii, excluding N/O to N/O pairs, which are hydrogen bonds rather than clashes.

A flat distance threshold cannot work:

| contact | distance | inside vdW | verdict |
| --- | --- | --- | --- |
| ligand N to backbone C | 2.69 A | 0.56 A | fine, appears in validated poses |
| ligand F to side-chain O | 2.36 A | 0.63 A | fine |
| ligand N to side-chain C | 1.48 A | 1.77 A | impossible, covalent bond length |

Any single cutoff either admits the third case or rejects the first two.

### Ligand internal strain (rejects, experimental)

Read from `r_i_glide_einternal`, which Glide already writes.

On standard grids this signal produces essentially every rejection the tool currently makes, and the clash rule is silent. On softened grids the opposite is true. So on ordinary docking output, strain *is* the filter.

**This rule is not validated.** Its threshold was set so as not to reject a small number of compounds that remained stable in MD. An independent MMFF energy difference computed on the same poses does not reproduce it: over 400 poses, Spearman 0.35 to 0.44, and at matched 90th-percentile cutoffs only 4 to 6 of 40 flagged poses are the same ones. Adding a distance-dependent dielectric or positional restraints changes the magnitude but not the agreement. Therefore:

- the threshold cannot be transferred to another docking engine,
- at least one of the two metrics is not measuring the physically relevant quantity,
- any KEEP or REJECT that rests solely on strain should be treated as provisional.

The JSON output carries `"strain_validated": false` so a downstream tool cannot silently treat the signal as settled. Establishing a defensible threshold from the distribution of real crystallographic poses is the purpose of evaluation 1 in `benchmark/`.

### Aromatic facing at the hinge (warns only)

The fraction of hinge contacts made by ring atoms rather than polar atoms. A low value means the rings point at solvent, which is the artifact this tool was built to catch.

It warns and never rejects: a compound with a stable MD trajectory measured 13%, in the bottom 5% of its library. Rejecting on it would discard a compound known to be good.

Pass the three receptor residue numbers corresponding to KLIFS pocket positions 46 to 48 for your structure. Nothing is hardcoded, so this works on any kinase. If you do not know the mapping, `src/klifs_pocket.py` will derive it from a KLIFS export, the KLIFS API, or by sequence alignment for a model KLIFS does not know. Or simply omit the flag: clash and strain still run, since neither depends on the pocket alignment.

## What this does not do

**It does not predict MD stability.** Two near-identical chemotypes with similar contact maps can differ entirely in whether the ligand stays bound, with the difference appearing only over hundreds of nanoseconds. No static geometric feature measured here separates such a pair.

This is a plausibility filter. It removes poses that cannot be right. It does not tell you which of the survivors will hold.

**The thresholds are not yet validated against negatives.** They were set so as not to reject a small set of compounds known to be good. That is a defensible starting point, not a validation: the filter has not been shown to reject something that deserves rejecting. See `benchmark/`.

## Benchmark (in progress)

`benchmark/structures.csv` holds 31 public kinase structures selected from KLIFS, one per kinase, for a validation that needs no proprietary data:

- Resolution 0.98 to 2.40 A, median 1.50 A
- All eight kinase groups; 29 drug-like inhibitors, 2 nucleotide controls
- Non-covalent, DFG-in, ATP-site ligand, no missing pocket atoms

Planned evaluations:

1. **Crystal-pose retention.** Score each crystallographic ligand in its own receptor. Every one must be kept; these are experimentally observed binding modes. This sets thresholds against real geometry rather than against a handful of compounds, and it is the next thing to run.
2. **Implausibility detection.** Negatives generated by soft-grid redocking and cross-docking, which produce genuinely impossible geometry rather than merely mislocated poses.
3. **Enrichment.** Whether filtering the top N raises the fraction of near-native poses, measured as MD runs saved.

Note on method: RMSD to the crystallographic pose is used only as an outcome measure in (3), never as the label the rules are judged against. This tool answers "is this pose physically possible", not "is this the near-native pose". A clean pose 6 A from the crystal position is correctly kept, and an RMSD-based label would score that as a failure.

## Notes found while building this

- **A hinge hydrogen-bond constraint makes any hinge H-bond metric useless.** If the docking used one, `b_glide_conssat_*` is True on essentially every pose and scoring that feature just re-measures your own input. An early version of these rules kept 97.6% of 3000 poses for exactly this reason. Check the constraint flags before trusting any hinge-contact feature, including this tool's aromatic warning.
- Pose files are sorted best-score-first, so a top-N only needs the head of the file.
- Entries above 300 atoms are receptor-scale complexes, not poses. Scoring one yields a meaningless 0.00 A contact.
- The receptor entry carries hydrogens; they are filtered before contact analysis, which works on heavy atoms with vdW radii.
- Ligand formal charges must be read from the file. Without them, protonated nitrogens fail sanitisation and ring perception silently dies for a large fraction of a library.

## Layout

```
run_triage.py              entry point
src/triage.py              rules, verdict, reason strings
src/contacts.py            contact geometry and atom typing
src/read_maestro.py        dependency-free Maestro .mae / .maegz reader
src/klifs_pocket.py        KLIFS pocket numbering: local export, API, or alignment
benchmark/structures.csv   the 31 public structures
```

`src/read_maestro.py` is standalone and may be useful on its own: it parses Maestro files with no Schrodinger dependency, including pose-viewer splitting and per-atom property tables.

## Status

v0.1, early.

The clash rule and soft-grid detection are ready for use: both are direct geometry and neither needs calibration. The strain rule is experimental and Glide-specific. Public validation on the 31 crystal structures is the next milestone, and it is what will decide whether the strain rule survives, and how aggressive the clash cutoff can be.

## Requirements

Python 3.10+, numpy, scipy, RDKit.

## Licence

MIT

"""
Pristine graphene reference run.

Reference-matched protocol:

    966 total atoms
    722 free atoms
    244 fixed atoms

    10 ps NVT relaxation
    30 ps NPT vibration

    300 K
    1 fs timestep
    20 THz trajectory sampling
"""

from pathlib import Path
from graphene_defect_ml import (build_graphene_sheet, assemble_graphene, 
                                write_data, write_in, modify_airebo_cutoff, run_lammps)


RUN_DIR       = Path(__file__).parent
POTENTIAL_SRC = RUN_DIR.parent.parent / "potentials" / "CH.airebo"
POTENTIAL_DST = RUN_DIR / "CH_modified.airebo"

SEED = 12345

sheet = build_graphene_sheet()
print(f"free:{sheet['free_xy'].shape[0]}, fixed: {sheet['fixed_xy'].shape[0]}")

atoms = assemble_graphene(sheet)
write_data(str(RUN_DIR / "sheet.data"), atoms)

modify_airebo_cutoff(str(POTENTIAL_SRC), str(POTENTIAL_DST))
write_in(str(RUN_DIR/ "in.Vibration"), "sheet.data", "CH_modified.airebo", seed=SEED, ensemble='npt')

print("Files generated, launching LAMMPS....")
run_lammps(str(RUN_DIR), "in.Vibration")
print("Done.")
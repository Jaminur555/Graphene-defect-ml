from pathlib import Path
from graphene_defect_ml import (build_graphene_sheet, assemble_graphene, write_data,
                                modify_airebo_cutoff, run_lammps)


RUN_DIR       = Path(__file__).parent
POTENTIAL_SRC = RUN_DIR.parent.parent / "potentials" / "CH.airebo"
POTENTIAL_DST = RUN_DIR / "CH_modified.airebo"

sheet = build_graphene_sheet()
print(f"free:{sheet['free_xy'].shape[0]}, fixed: {sheet['fixed_xy'].shape[0]}")

atoms = assemble_graphene(sheet)
write_data(str(RUN_DIR / "sheet.data"), atoms)

modify_airebo_cutoff(str(POTENTIAL_SRC), str(POTENTIAL_DST))

print("Files generated, launching LAMMPS minimization....")
run_lammps(str(RUN_DIR), "in.relax")
print("Done.")
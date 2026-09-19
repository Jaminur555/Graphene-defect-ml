"""
Single-atom vacancy validation run

Generates sheet.data, in.vibration  and then runs LAMMPS.
"""

from pathlib import Path
from graphene_defect_ml import (build_graphene_sheet, remove_vacancy, assemble_graphene,
                                write_data, write_in, modify_airebo_cutoff, run_lammps)


RUN_DIR       = Path(__file__).parent
POTENTIAL_SRC = RUN_DIR.parent.parent
POTENTIAL_DST = RUN_DIR / "CH_modified.airebo"


VACANCY_ROW = 17
VACANCY_COL = 10
SEED        = 12345


sheet     = build_graphene_sheet()
sheet_vac = remove_vacancy(sheet, target_row=VACANCY_ROW, target_col=VACANCY_COL)
print(f"free: {sheet_vac['free_xy'].shape[0]} fixed: {sheet_vac['fixed_xy'].shape[0]}")

atoms = assemble_graphene(sheet_vac)
write_data(str(RUN_DIR / "sheet.data"), atoms)

modify_airebo_cutoff(str(POTENTIAL_SRC), str(POTENTIAL_DST))
write_in(str(RUN_DIR / "in.vibration"), "sheet.data", "CH_modified.airebo", seed=SEED)

print("Files generated, launching LAMMPS...")
run_lammps(str(RUN_DIR), "in.vibration")
print("Done")
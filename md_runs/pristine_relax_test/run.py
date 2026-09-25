"""
Pristine baseline, relax-duration test.

Stage 1 (NVT relax) is extended from 10 ps to 30 ps, to test whether the 
original relax duration was long enough for PotEng to plateau before the 
vibration reference configuration is taken.
"""

from pathlib import Path
from graphene_defect_ml import (build_graphene_sheet, assemble_graphene,
                                write_data, write_in, modify_airebo_cutoff, run_lammps,
                                check_relax_convergence)


RUN_DIR       = Path(__file__).parent
POTENTIAL_SRC = RUN_DIR.parent.parent / "potentials" / "CH.airebo"
POTENTIAL_DST = RUN_DIR / "CH_modified.airebo"

SEED = 12345
RELAX_STEPS = 30000  # 30 ps, vs. 10 ps in the original pristine run

sheet = build_graphene_sheet()
print(f"free:{sheet['free_xy'].shape[0]}, fixed: {sheet['fixed_xy'].shape[0]}")

atoms = assemble_graphene(sheet)
write_data(str(RUN_DIR / "sheet.data"), atoms)

modify_airebo_cutoff(str(POTENTIAL_SRC), str(POTENTIAL_DST))
write_in(str(RUN_DIR / "in.Vibration"), "sheet.data", "CH_modified.airebo",
        seed=SEED, relax_steps=RELAX_STEPS)

print("Files generated, launching LAMMPS....")
run_lammps(str(RUN_DIR), "in.Vibration")
print("Done.")

result = check_relax_convergence(str(RUN_DIR / "log.lammps"), run_index=0)
print(f"\nRelax convergence check (Stage 1, {RELAX_STEPS} steps):")
print(f"  PotEng change (last vs. prior tail window): {result['pct_change']:.4f}%")
print(f"  Converged (< 0.05% threshold): {result['converged']}")
"""
Validate the pristine-sheet vibration amplitude against Paper 1's reported benchmark of ~0.3 A.
"""

from pathlib import Path
from graphene_defect_ml import vibration_amplitude

RUN_DIR   = Path(__file__).parent
DUMP_PATH = RUN_DIR / "vibration.lammpstrj"

PAPER_BENCHMARK_A = 0.3  # Angstrom, Paper 1's reported pristine-sheet vibration amplitude

if not DUMP_PATH.exists():
    raise FileNotFoundError(
        f"{DUMP_PATH} not found — run run.py in this directory first "
        f"to generate the vibration trajectory."
    )

result = vibration_amplitude(DUMP_PATH)

print(f"Parsed {result['n_frames']} frames, {result['n_atoms']} free atoms\n")
print(f"RMS amplitude:        {result['rms_amplitude']:.4f} Å")
print(f"Mean peak amplitude:  {result['mean_peak_amplitude']:.4f} Å")
print(f"Paper 1 benchmark:    ~{PAPER_BENCHMARK_A} Å\n")

for label, value in [('RMS', result['rms_amplitude']),
                      ('mean peak', result['mean_peak_amplitude'])]:
    pct_diff = 100 * (value - PAPER_BENCHMARK_A) / PAPER_BENCHMARK_A
    print(f"{label} amplitude is {pct_diff:+.1f}% relative to the paper's benchmark")
"""
Sanity check for the single-vacancy run: per-atom vibrational energy map, compared with the 
pristine NVT run.

For each free atom: z(t) -> subtract mean -> FFt -> S = sum |z(f)|^2 (band-integrated energy).
compare with Fig. 1c fo Zheng and Gu (2020): the vacancy should create a local energy maximum.
"""

from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from graphene_defect_ml import parse_lammps_dump, vibration_amplitude, build_graphene_sheet


RUN_DIR       = Path(__file__).parent
VAC_DUMP      = RUN_DIR/ "vibration.lammpstrj"
PRISTINE_DUMP = RUN_DIR.parent/ "pristine_nvt" / "vibration.lammpstrj"

VACANCY_ROW, VACANCY_Col = 17, 10
NEAR_RADIUS              = 3.0


def energy_map(dump_path):
    frames     = parse_lammps_dump(str(dump_path))
    cols       = {c: i for i, c in enumerate(frames[0]['columns'])}
    n_fr, n_at = len(frames), frames[0]['n_atoms']
    
    disp = np.zeros((n_fr, n_at))
    pos  = np.zeros((n_fr, n_at, 2))
    
    for k, fr in enumerate(frames):
        d       = fr['data'][np.argsort(fr['data'][:, cols['id']])]
        disp[k] = d[:, cols['c_ydisp[2]']]
        pos[k]  = d[:, [cols['x'], cols['z']]]
        
    z = disp - disp.mean(axis=0)
    S = (np.abs(np.fft.rfft(z, axis=0)) ** 2).sum(axis = 0)
    
    return pos.mean(axis=0), S


for p in (VAC_DUMP, PRISTINE_DUMP):
    if not p.exists():
        raise FileNotFoundError(f"{p} not found - run the correspondin run.py first")
    
amp = vibration_amplitude(VAC_DUMP)
print(f"Vacancy run: {amp['n_frames']} frames, {amp['n_atoms']} free atoms,"
    f"RMS amplitude : {amp['rms_amplitude']:.4f} A")

sheet = build_graphene_sheet()

i      = np.where((sheet['free_row'] == VACANCY_ROW) & (sheet['free_col'] == VACANCY_Col))[0][0]
vx, vz = sheet['free_xy'][i] 

fig, axes = plt.subplots(1, 2, figsize=(11, 6))
ratios = {}
for ax, (name, path) in zip(axes, [('Pristine', PRISTINE_DUMP), ('Vacancy', VAC_DUMP)]):
    xz, S = energy_map(path)
    S     = S / S.max()
    
    near         = np.linalg.norm(xz - np.array([vx, vz]), axis=1) < NEAR_RADIUS
    ratios[name] = S[near].mean() / S.mean()
    
    sc = ax.scatter(xz[:, 0], xz[:, 1], c=S, s=40, cmap='jet', vmin=0, vmax=1)
    
    ax.plot(vx, vz, 'kx', ms = 12, mew=2)
    ax.set_title(f"{name} (x = vacancy site)")
    ax.set_aspect("equal")
    ax.set_xlabel('x(A)')
    ax.set_ylabel("z (A)")

fig.colorbar(sc, ax = axes, label='S/ S_max')
fig.savefig(RUN_DIR/'vacancy_enery_map.png', dpi = 150, bbox_inches='tight')

print(f"\nEnergy near the vacancy site / sheet average (within {NEAR_RADIUS} A):")
print(f"  pristine: {ratios['Pristine']:.2f}   vacancy: {ratios['Vacancy']:.2f}")
print("Saved vacancy_energy_map.png")

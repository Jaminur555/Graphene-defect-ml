"""
Analyze the single-vacancy vibration-energy map.

Reference coordinate convention:

    x = long in-plane direction
    y = short in-plane direction
    z = out-of-plane direction

For each free atom:

    z(t) -> subtract temporal mean -> FFT -> S(f) = integral |z(f)|^2 df

The vacancy should produce a localized vibrational-energy
enhancement around the defect.
"""

from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use("Agg")           # Non-interactive rendering backend.
import matplotlib.pyplot as plt

from graphene_defect_ml import (vibration_amplitude, build_graphene_sheet, remove_vacancy,
                                load_displacement, spectral_energy, vacancy_label)


RUN_DIR       = Path(__file__).parent
VAC_DUMP      = RUN_DIR / "vibration.lammpstrj"
PRISTINE_DUMP = RUN_DIR.parent / "pristine_nvt" / "vibration.lammpstrj"

VACANCY_ROW = 17
VACANCY_COL = 10
NEAR_RADIUS = 3.0


def energy_map(dump_path, reference_xy):
    """
    Per-atom vibrational-energy feature S for one trajectory.
 
    Parameters
    ----------
    dump_path : Path
        LAMMPS vibration trajectory (ids 1..N, row-major order).
    reference_xy : ndarray, shape (N, 2)
        Reference x-y coordinates of the free atoms, in the same order as the ids.
 
    Returns
    -------
    xy : ndarray (N, 2)   reference coordinates
    S  : ndarray (N,)     integrated spectral energy per atom (unnormalized)
    dt : float            sampling interval in seconds, derived from the dump itself
    n_frames : int
    """
    z, dt = load_displacement(dump_path)
    
    if len(reference_xy) != z.shape[1]:
        raise ValueError(f"Reference coordinate count ({len(reference_xy)}) does not"
                         f"match trajectory atom count ({z.shape[1]}) in {dump_path}")
    
    S, _, _ = spectral_energy(z, dt)
    
    return reference_xy.copy(), S, dt, z.shape[0]


for path in (VAC_DUMP, PRISTINE_DUMP):
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run the corresponding run.py first")


sheet = build_graphene_sheet()

print(f"Reference graphene: {len(sheet['free_xy'])} free atoms, "
      f"{sheet['n_rows_free']} rows x {sheet['n_cols_free']} columns")


vacancy_indices = np.where((sheet["free_row"] == VACANCY_ROW) & (sheet["free_col"] == VACANCY_COL))[0]

if len(vacancy_indices) != 1:
    raise RuntimeError("Could not uniquely locate the requested vacancy.")

vacancy_index = vacancy_indices[0]
vx, vy        = sheet["free_xy"][vacancy_index]

label = vacancy_label(sheet, VACANCY_ROW, VACANCY_COL)
assert label == vacancy_index, "sheet is not in canonical row-major order (apply Fix 1)"

print(f"Vacancy location: row={VACANCY_ROW}, col={VACANCY_COL}, x={vx:.4f} Å, y={vy:.4f} Å")

amp = vibration_amplitude(VAC_DUMP, column="c_zdisp[3]")

print(f"Vacancy run: {amp['n_frames']} frames, {amp['n_atoms']} free atoms, "
      f"RMS amplitude : {amp['rms_amplitude']:.4f} A")

pristine_xy = sheet["free_xy"]

vacancy_sheet = remove_vacancy(sheet, VACANCY_ROW, VACANCY_COL)
vacancy_xy    = vacancy_sheet["free_xy"]

xy_pristine, S_pristine_raw, dt_p, nf_p = energy_map(PRISTINE_DUMP, pristine_xy)
xy_vacancy,  S_vacancy_raw,  dt_v, nf_v = energy_map(VAC_DUMP, vacancy_xy)

assert dt_p == dt_v and nf_p == nf_v, \
    f"runs differ: dt {dt_p} vs {dt_v}, frames {nf_p} vs {nf_v}"
 
print(f"Sampling: dt = {dt_v * 1e15:.1f} fs, {nf_v} frames")

common_max = max(S_pristine_raw.max(),S_vacancy_raw.max())

S_pristine = S_pristine_raw / common_max
S_vacancy  = S_vacancy_raw / common_max


#----------------------------------------------------------------energy maps
fig, axes = plt.subplots(1, 2, figsize=(11, 6))
ratios    = {}

for ax, (name, xy, S) in zip(axes,[("Pristine", xy_pristine, S_pristine), ("Vacancy", xy_vacancy, S_vacancy)]):
    near = (np.linalg.norm(xy - np.array([vx, vy]),axis=1) < NEAR_RADIUS)

    ratios[name] = S[near].mean() / S.mean()

    sc = ax.scatter(xy[:, 0], xy[:, 1], c=S, s=40, cmap="jet", vmin=0, vmax=1)

    ax.plot(vx, vy, "kx", ms=12, mew=2)
    ax.set_aspect("equal")

    ax.set_title(f"{name} (x = vacancy site)")
    ax.set_xlabel("X(A)")
    ax.set_ylabel("Y (A)")

    ax.set_xlim(sheet["free_xy"][:, 0].min() - 2.0,sheet["free_xy"][:, 0].max() + 2.0)
    ax.set_ylim(sheet["free_xy"][:, 1].min() - 2.0,sheet["free_xy"][:, 1].max() + 2.0)


fig.colorbar(sc, ax=axes,label="S / S_max")

output_path = RUN_DIR / "vacancy_energy_map.png"
fig.savefig(output_path,dpi=150,bbox_inches="tight")
plt.close(fig)


print(f"\nEnergy near the vacancy site / sheet average "
      f"(within {NEAR_RADIUS} A):")

print(f"  pristine: {ratios['Pristine']:.2f} vacancy: {ratios['Vacancy']:.2f}")

print(f"\nSaved corrected figure:")
print(output_path)


#------------------------------------------------------------- paired per-atom ration map
S_vac_aligned = np.insert(S_vacancy_raw, label, np.nan)
ratio         = S_vac_aligned / S_pristine_raw
 
k_max   = int(np.nanargmax(ratio))
d_max   = float(np.linalg.norm(pristine_xy[k_max] - np.array([vx, vy])))
 
print(f"\nPaired ratio S_vacancy / S_pristine (same atom):")
print(f"  max ratio {ratio[k_max]:.2f} at grid index {k_max} "
      f"(row={sheet['free_row'][k_max]}, col={sheet['free_col'][k_max]}), "
      f"{d_max:.2f} A from the vacancy site")
print(f"  median ratio over all atoms: {np.nanmedian(ratio):.2f}")
 
fig, ax = plt.subplots(figsize=(6.5, 6))
sc = ax.scatter(pristine_xy[:, 0], pristine_xy[:, 1], c=ratio, s=40, cmap="jet",
                vmin=0, vmax=np.nanpercentile(ratio, 99))
ax.plot(vx, vy, "kx", ms=12, mew=2)
ax.set_aspect("equal")
ax.set_title("Paired ratio S_vacancy / S_pristine")
ax.set_xlabel("X (A)")
ax.set_ylabel("Y (A)")
fig.colorbar(sc, ax=ax, label="ratio")
fig.savefig(RUN_DIR / "vacancy_ratio_map.png", dpi=150, bbox_inches="tight")
plt.close(fig)
 
np.savetxt(RUN_DIR / "vacancy_ratio_722.csv",
           np.column_stack((np.arange(1, len(ratio) + 1), ratio)),
           delimiter=",", header="N_paper,ratio_vac_over_pristine")


# ------------------------------------------------------------------------ 1D energy vector
S      = S_vacancy_raw
S_max  = S.max()
S_norm = S / S_max

n = np.arange(1, len(S_norm) + 1)

top = np.argsort(S_norm)[::-1][:6] + 1

print(f"\n1D energy vector:")
print(f"  Number of free atoms : {len(S_norm)}")
print(f"  S_max                : {S_max:.6e}")
print(f"  Vacancy row / column : {VACANCY_ROW} / {VACANCY_COL}")
print(f"  Vacancy label N      : {label + 1}  (vacancy sits between vector entries "
      f"{label} and {label + 1})")
print(f"  Top-6 vector indices : {top.tolist()}")

fig, ax = plt.subplots(figsize=(7, 8))

ax.plot(S_norm, n, linewidth=0.8)
ax.axhline(label + 0.5, color="r", linestyle="--", linewidth=0.6, label="vacancy position")
ax.legend(loc="lower right")

ax.set_xlabel(r"$S/S_{\max}$", fontsize=13)
ax.set_ylabel(r"$N$", fontsize=13)
ax.set_title("Single-vacancy vibrational-energy vector", fontsize=13)

ax.set_ylim(1, len(S_norm))
ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.3)

fig.tight_layout()

fig.savefig(RUN_DIR / 'vacancy_1d_S_over_Smax.png', dpi=300, bbox_inches='tight')

plt.close(fig)

output = np.column_stack((n, S, S_norm))

np.savetxt(RUN_DIR/'vacancy_1d_S_over_Smax.csv', output, delimiter=",", header='n,S,S_over_Smax')

print(f"Saved {RUN_DIR/'vacancy_1d_S_over_Smax.png'}")
print(f"Saved {RUN_DIR/'vacancy_1d_S_over_Smax.csv'}")
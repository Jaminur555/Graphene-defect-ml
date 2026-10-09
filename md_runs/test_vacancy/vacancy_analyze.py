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
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from graphene_defect_ml import parse_lammps_dump, vibration_amplitude, build_graphene_sheet, remove_vacancy


RUN_DIR       = Path(__file__).parent
VAC_DUMP      = RUN_DIR / "vibration.lammpstrj"
PRISTINE_DUMP = RUN_DIR.parent / "pristine_nvt" / "vibration.lammpstrj"

VACANCY_ROW = 17
VACANCY_COL = 10
NEAR_RADIUS = 3.0


def energy_map(dump_path, reference_xy):
    """
    Calculate the per-atom vibrational-energy feature.

    Parameters
    ----------
    dump_path : Path
        LAMMPS vibration trajectory.

    reference_xy : ndarray, shape (N, 2)
        Fixed reference x-y coordinates corresponding to the free atoms
        in the trajectory, in exactly the same atom ordering.

    Returns
    -------
    xy : ndarray, shape (N, 2)
        Reference graphene x-y coordinates.

    S : ndarray, shape (N,)
        Band-integrated FFT energy for each atom.

    atom_ids : ndarray
        Atom IDs found in the trajectory.
    """

    frames = parse_lammps_dump(str(dump_path))

    if not frames:
        raise ValueError(f"No frames found in {dump_path}")

    first = frames[0]
    cols  = {c: i for i, c in enumerate(first["columns"])}

    required_columns = {"id", "c_zdisp[3]"}
    missing          = required_columns - set(cols)

    if missing:
        raise ValueError(f"{dump_path} is missing required columns: {sorted(missing)}")

    n_frames = len(frames)
    n_atoms  = first["n_atoms"]

    if len(reference_xy) != n_atoms:
        raise ValueError(f"Reference coordinate count ({len(reference_xy)}) does not "
                         f"match trajectory atom count ({n_atoms}) in {dump_path}.")

    disp     = np.zeros((n_frames, n_atoms))
    atom_ids = None

    for k, frame in enumerate(frames):

        data  = frame["data"]
        order = np.argsort(data[:, cols["id"]])
        data  = data[order]

        current_ids = data[:, cols["id"]].astype(int)

        if atom_ids is None:
            atom_ids = current_ids

        elif not np.array_equal(current_ids, atom_ids):
            raise ValueError(f"Atom IDs/order changed during trajectory {dump_path}.")

        disp[k, :] = data[:, cols["c_zdisp[3]"]]

    z     = disp - disp.mean(axis=0, keepdims=True)
    fft_z = np.fft.rfft(z, axis=0)
    power = np.abs(fft_z) ** 2

    sample_dt   = 50.0e-15
    frequencies = np.fft.rfftfreq(n_frames, d=sample_dt)

    S = np.trapezoid(power, frequencies, axis=0)

    return reference_xy.copy(), S, atom_ids


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

print(f"Vacancy location: row={VACANCY_ROW}, col={VACANCY_COL}, x={vx:.4f} Å, y={vy:.4f} Å")

amp = vibration_amplitude(VAC_DUMP, column="c_zdisp[3]")

print(f"Vacancy run: {amp['n_frames']} frames, {amp['n_atoms']} free atoms, "
      f"RMS amplitude : {amp['rms_amplitude']:.4f} A")

pristine_xy = sheet["free_xy"]

vacancy_sheet = remove_vacancy(sheet, VACANCY_ROW, VACANCY_COL)
vacancy_xy    = vacancy_sheet["free_xy"]


xy_pristine, S_pristine, pristine_ids = energy_map(PRISTINE_DUMP,pristine_xy)
xy_vacancy, S_vacancy, vacancy_ids    = energy_map(VAC_DUMP,vacancy_xy)

common_max = max(S_pristine.max(),S_vacancy.max())

S_pristine = S_pristine / common_max
S_vacancy  = S_vacancy / common_max


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


print(f"\nEnergy near the vacancy site / sheet average "
      f"(within {NEAR_RADIUS} A):")

print(f"  pristine: {ratios['Pristine']:.2f} vacancy: {ratios['Vacancy']:.2f}")

print(f"\nSaved corrected figure:")
print(output_path)

xy, S, atom_ids = energy_map(VAC_DUMP, vacancy_xy)

S_max  = S.max()
S_norm = S / S_max

n = np.arange(1, len(S_norm) + 1)

vacancy_N = (19 * (VACANCY_ROW - 1)) + VACANCY_COL


print(f"1D energy vector:",
      f"Number of free atoms: {len(S_norm)}",
      f"S_max:              : {S_max:.6e}",
      f"Vacancy row         : {VACANCY_ROW}",
      f"vacancy column      : {VACANCY_COL}",
      f"Original vacancy N  : {vacancy_N}")

fig, ax = plt.subplots(figsize=(7, 8))

ax.plot(S_norm, n, linewidth=0.8)

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

print(f"Saved{RUN_DIR/'vacancy_1d_S_over_Smax.png'}")
print(f"Saved{RUN_DIR/'vacancy_1d_S_over_Smax.csv'}")
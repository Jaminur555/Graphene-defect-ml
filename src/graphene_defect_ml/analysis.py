import numpy as np


def parse_lammps_dump(path: str) -> list:
    """
    Parse a LAMMPS custom-style dump file into a list of per-frame records.

    Args:
        path (str): path to the .lammpstrj file

    Returns:
        list[dict]: one entry per dumped frame, each with:
            'timestep' (int), 'n_atoms' (int), 'columns' (list[str]),
            'data' (np.ndarray, shape (n_atoms, n_columns))
    """
    
    with open(path, 'r') as f:
        lines = f.readlines()

    frames   = []
    n_lines  = len(lines)
    timestep = None
    n_atoms  = None
    
    i = 0
    while i < n_lines:
        line = lines[i]
        if line.startswith('ITEM: TIMESTEP'):
            timestep = int(lines[i + 1])
            i += 2
        elif line.startswith('ITEM: NUMBER OF ATOMS'):
            n_atoms = int(lines[i + 1])
            i += 2
        elif line.startswith('ITEM: BOX BOUNDS'):
            i += 4  # header line + 3 bound lines
        elif line.startswith('ITEM: ATOMS'):
            columns = line.split()[2:]
            i += 1
            data = np.zeros((n_atoms, len(columns)))
            for row in range(n_atoms):
                data[row] = [float(v) for v in lines[i + row].split()]
            i += n_atoms
            frames.append({
                'timestep': timestep,
                'n_atoms': n_atoms,
                'columns': columns,
                'data': data,
            })
        else:
            i += 1

    return frames


def vibration_amplitude(dump_path: str, column: str = 'c_ydisp[2]') -> dict:
    """
    Compute the out-of-plane vibration amplitude from a `vibration.lammpstrj` dump,
    for comparison against reported ~0.3 A benchmark on a pristine sheet

    Two amplitude conventions are implemented:
        - rms_amplitude      : sqrt(mean(disp^2)) over all atoms and all dumped frames
        - mean_peak_amplitude: average, over atoms, of (max(disp) - min(disp)) / 2

    Args:
        dump_path (str): path to vibration.lammpstrj
        column (str)   : dump column holding the out-of-plane displacement.
                         Defaults to 'c_ydisp[2]', matching the `compute ydisp`
                         defined in lammps_io.write_in().

    Returns:
        dict: {'rms_amplitude': float, 'mean_peak_amplitude': float,
               'n_frames': int, 'n_atoms': int}
    """
    frames = parse_lammps_dump(dump_path)
    if not frames:
        raise ValueError(f"No frames parsed from {dump_path}")

    col_idx = frames[0]['columns'].index(column)
    id_idx  = frames[0]['columns'].index('id')
    n_atoms = frames[0]['n_atoms']

    # (n_frames, n_atoms) matrix of displacement values, atoms sorted by id
    # so each column tracks the same physical atom across all frames.
    disp = np.zeros((len(frames), n_atoms))
    for f_idx, frame in enumerate(frames):
        order       = np.argsort(frame['data'][:, id_idx])
        disp[f_idx] = frame['data'][order, col_idx]

    rms_amplitude = float(np.sqrt(np.mean(disp ** 2)))
    per_atom_peak = (disp.max(axis=0) - disp.min(axis=0)) / 2
    mean_peak_amplitude = float(per_atom_peak.mean())

    return {
        'rms_amplitude': rms_amplitude,
        'mean_peak_amplitude': mean_peak_amplitude,
        'n_frames': len(frames),
        'n_atoms': n_atoms,
    }
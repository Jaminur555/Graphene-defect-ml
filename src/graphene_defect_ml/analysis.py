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


def parse_lammps_log(path: str) -> list:
    """
    Parse a LAMMPS log file into a list of per-run thermo blocks (one block per
    'run N' command executed). Each block covers the thermo table printed for
    that run, e.g. block 0 is Stage 1 (relax), block 1 is Stage 2 (vibrate).
 
    Args:
        path (str): path to log.lammps
 
    Returns:
        list[dict]: one entry per run block, each with:
            'columns' (list[str]), 'data' (np.ndarray, shape (n_rows, n_columns))
    """
    with open(path, 'r') as f:
        lines = f.readlines()
 
    blocks = []
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped.startswith('Step'):
            columns = stripped.split()
            rows = []
            i += 1
            while i < len(lines):
                parts = lines[i].split()
                if len(parts) != len(columns):
                    break
                try:
                    rows.append([float(p) for p in parts])
                except ValueError:
                    break
                i += 1
            blocks.append({'columns': columns, 'data': np.array(rows)})
        else:
            i += 1
 
    return blocks


def check_relax_convergence(log_path: str, run_index: int = 0, tail_frac: float = 0.3,
                             pe_tol_pct: float = 0.05) -> dict:
    """
    Check whether PotEng has plateaued by the end of a given run block in a LAMMPS log,
    to sanity-check whether the Stage-1 relax duration is actually long enough before
    it's used as the reference configuration for downstream vibration analysis.
 
    Compares the mean PotEng over the final `tail_frac` of the block against the mean
    PotEng over the `tail_frac` immediately preceding it; flags convergence if they
    differ by less than pe_tol_pct percent.
 
    Args:
        log_path (str)      : path to log.lammps
        run_index (int)     : which run block to check. Defaults to 0 (Stage 1 relax).
        tail_frac (float)   : fraction of the block's rows used for each comparison window. Defaults to 0.3.
        pe_tol_pct (float)  : percent-change threshold below which PotEng is considered converged. Defaults to 0.05.
 
    Returns:
        dict: {'converged': bool, 'pct_change': float,
               'mean_pe_last_tail': float, 'mean_pe_prev_tail': float, 'n_rows': int}
    """
    blocks = parse_lammps_log(log_path)
    if run_index >= len(blocks):
        raise ValueError(f"Log has only {len(blocks)} run block(s); requested index {run_index}")
 
    block = blocks[run_index]
    pe_idx = block['columns'].index('PotEng')
    pe = block['data'][:, pe_idx]
    n = len(pe)
 
    tail_n = max(2, int(n * tail_frac))
    if n < 2 * tail_n:
        raise ValueError(f"Run block {run_index} has too few thermo rows ({n}) "
                         f"for tail_frac={tail_frac}; lower thermo output interval or tail_frac.")
 
    last_tail = pe[-tail_n:]
    prev_tail = pe[-2 * tail_n:-tail_n]
    mean_last = float(last_tail.mean())
    mean_prev = float(prev_tail.mean())
    pct_change = 100 * abs(mean_last - mean_prev) / abs(mean_prev)
 
    return {
        'converged': pct_change < pe_tol_pct,
        'pct_change': pct_change,
        'mean_pe_last_tail': mean_last,
        'mean_pe_prev_tail': mean_prev,
        'n_rows': n,
    }
 

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
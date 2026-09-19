import os
import subprocess


def run_lammps(run_dir: str, in_filename: str, log_filename: str = 'log.lammps',
               n_threads: int = 4) -> None:
    """
    Run LAMMPS on an already-written .in script, using the conda-installed 'lmp' binary.

    Args:
        run_dir (str): directory containing the .in/.data/.airebo files
        in_filename (str): name of the .in script (relative to run_dir)
        log_filename (str, optional): LAMMPS log output. Defaults to 'log.lammps'.
        n_threads (int, optional): OpenMP threads to use. Defaults to 4.
    """

    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = str(n_threads)
    subprocess.run(['lmp', '-in', in_filename, '-log', log_filename],
                   cwd=run_dir, env=env, check=True)
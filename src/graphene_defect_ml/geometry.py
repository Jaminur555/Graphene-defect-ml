import numpy as np
from ase.build import graphene_nanoribbon

BOND_LENGTH = 1.3992    # Angstrom, C-C bond length in graphene


def build_graphene_sheet(n: int = 21, m: int = 23, row_border: int = 4) -> dict:
    """
    Build a rectangular graphene flake and split its atoms into a fixed border
    (clamped edges) and a free interior (vibrating region).

    Args:
        n (int)         : ASE graphene_nanoribbon width parameter. Defaults to 21.
        m (int)         : ASE graphene_nanoribbon length parameter. Defaults to 23.
        row_border (int): Number of outermost atom rows/columns treated as the fixed border. Defaults to 4.

    Returns:
        dict:
        'fixed_xy'   : (n_fixed, 2) array of fixed-atom (x, z) coords
        'free_xy'    : (n_free, 2) array of free-atom (x, z) coords
        'free_row'   : (n_free,) row index of each free atom (0-based)
        'free_col'   : (n_free,) column index of each free atom (0-based)
        'n_rows_free', 'n_cols_free' : interior grid dimensions
    """

    ribbon = graphene_nanoribbon(n, m, type='zigzag', saturated=False,
                                 C_C=BOND_LENGTH, vacuum=15.0)

    pos  = ribbon.get_positions()
    x, z = pos[:, 0], pos[:, 2]

    z_round   = np.round(z, 2)
    z_levels  = np.sort(np.unique(z_round))
    row_index = np.searchsorted(z_levels, z_round)

    is_row_border = (row_index < row_border) | (row_index >= len(z_levels) - row_border)

    free_mask = np.zeros(len(pos), dtype=bool)
    free_row  = np.full(len(pos), -1, dtype=int)
    free_col  = np.full(len(pos), -1, dtype=int)

    interior_rows = z_levels[row_border: len(z_levels) - row_border]
    for local_row, zval in enumerate(interior_rows):
        row_sel   = np.isclose(z_round, zval)
        xs_in_row = np.sort(x[row_sel])
        lo, hi    = xs_in_row[0], xs_in_row[-1]

        # drop the outermost atom on each end of this row (column border)
        row_free  = row_sel & (x > lo + 0.01) & (x < hi - 0.01)
        free_mask |= row_free
        free_row[row_free] = local_row
        xs_free_sorted     = np.sort(x[row_free])
        col_lookup = {val: idx for idx, val in enumerate(xs_free_sorted)}

        for atom_idx in np.where(row_free)[0]:
            free_col[atom_idx] = col_lookup[x[atom_idx]]

    fixed_mask  = ~free_mask
    n_cols_free = int(free_col[free_mask].max() + 1)
    n_rows_free = len(interior_rows)

    return {
        'fixed_xy': pos[fixed_mask][:, [0, 2]],
        'free_xy' : pos[free_mask][:, [0, 2]],
        'free_row': free_row[free_mask],
        'free_col': free_col[free_mask],
        'n_rows_free': n_rows_free,
        'n_cols_free': n_cols_free
    }


def remove_vacancy(sheet: dict, target_row: int, target_col: int) -> dict:
    """
    Remove a single free (interior) atom at (target_row, target_col).

    Args:
        sheet (dict): dict returned by build_graphene_sheet()
        target_row (int): 0-based row index of the vacancy
        target_col (int): 0-based column index of the vacancy

    Returns:
        dict: a new sheet dict with that atom removed.
    """

    keep = ~((sheet['free_row'] == target_row) & (sheet['free_col'] == target_col))
    if keep.sum() == len(keep):
        raise ValueError(f"No atom found at (row = {target_row}, col = {target_col});"
                        f"valid ranges are row < {sheet['n_rows_free']}"
                         f"col < {sheet['n_cols_free']}")

    new_sheet = dict(sheet)

    new_sheet['free_xy']  = sheet['free_xy'][keep]
    new_sheet['free_row'] = sheet['free_row'][keep]
    new_sheet['free_col'] = sheet['free_col'][keep]

    return new_sheet


def assemble_graphene(sheet: dict) -> np.ndarray:
    """
    Assemble fixed and free atoms into a single array formatted for a LAMMPS 'atom_style molecular'
    data file. Columns are [molecule_id, atom_type, x, y, z]; molecule_id = 1 free (vibrating),
    molecule_id = 2 (fixed)

    Args:
        sheet (dict): dict returned by build_graphene_sheet() or remove_vacancy()

    Returns:
        np.ndarray: (n_atoms, 5) array
    """
    n_free  = sheet['free_xy'].shape[0]
    n_fixed = sheet['fixed_xy'].shape[0]

    free_rows = np.hstack((
        np.ones((n_free, 1)),      # molecule id: 1 = free
        np.ones((n_free, 1)),      # atom type  : 1 = carbon
        sheet['free_xy'][:, [0]],  # x
        np.zeros((n_free, 1)),     # y (flat sheet)
        sheet['free_xy'][:, [1]]   # z (in-plane 2nd axis)
    ))

    fixed_rows = np.hstack((
        2 * np.ones((n_fixed, 1)), # molecule id: 2 = fixed
        np.ones((n_fixed, 1)),     # atom type  : 1 = carbon
        sheet['fixed_xy'][:, [0]],
        np.zeros((n_fixed, 1)),
        sheet['fixed_xy'] [:, [1]]
    ))

    return np.vstack((free_rows, fixed_rows))
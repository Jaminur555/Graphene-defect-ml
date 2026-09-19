def modify_airebo_cutoff(src_path: str, dst_path: str, new_rcmin_cc: float = 1.92) -> None:
    """
    Copy an AIREBO potential file, modifying rcmin_CC (the smaller of the two REBO switching-
    function cutoffs). Default rcmin_CC = 1.7 A is changed to 1.92 A (rcmax_CC stays at its
    default 2.0 A), narrowing the switching region to better match DFT-benchmarked mechanical
    behaviour.

    Args:
        src_path (str): path to the original CH.airebo file
        dst_path (str): path to write the modified copy
        new_rcmin_cc (float): replacement value for rcmin_CC. Defaults to 1.92.
    """

    with open(src_path, 'r') as f:
        lines = f.readlines()

    modified = False
    with open(dst_path, 'w') as f:
        for line in lines:
            if 'rcmin_CC' in line:
                f.write(f"{new_rcmin_cc} rcmin_CC\n")
                modified = True
            else:
                f.write(line)

    if not modified:
        raise ValueError(f"Could not find 'rcmin_CC' line in {src_path}"
                         f"check the file format before proceeding.")
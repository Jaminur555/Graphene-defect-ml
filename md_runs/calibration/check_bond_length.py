"""
Measure the mean nearest-neighbor (bond) distance from the relaxed structure,
"""

from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from graphene_defect_ml.analysis import parse_lammps_dump

RUN_DIR = Path(__file__).parent
N_FREE  = 722

frames = parse_lammps_dump(str(RUN_DIR / "relaxed.lammpstrj"))
frame  = frames[-1]
cols   = frame['columns']
idx    = {c: i for i, c in enumerate(cols)}
pos    = frame['data']

ids    = pos[:, idx['id']].astype(int)
xyz    = pos[:, [idx['x'], idx['y'], idx['z']]]

tree = cKDTree(xyz)
dists, _ = tree.query(xyz, k=2)     # k=1 is self (dist 0), k=2 is nearest neighbor
nn_dist  = dists[:, 1]

# keep only plausible C-C bonded distances, excluding any numerical noise
plausible = (nn_dist > 1.2) & (nn_dist < 1.7)
is_free   = (ids <= N_FREE)
is_all    = np.ones(len(nn_dist), dtype=bool)

def report(label, mask):
      mask = np.asarray(mask, dtype=bool)
      d = nn_dist[mask & plausible]
      print(f"{label:<22}: {d.mean():.4f} +/- {d.std():.4f} A over {len(d)} atoms")

report("All atoms", is_all)
report("Free (interior) atoms", is_free)
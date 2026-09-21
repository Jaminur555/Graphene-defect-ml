"""
Measure the mean nearest-neighbor (bond) distance from the relaxed structure,
"""

from pathlib import Path
from scipy.spatial import cKDTree
from graphene_defect_ml.analysis import parse_lammps_dump

RUN_DIR = Path(__file__).parent

frames = parse_lammps_dump(str(RUN_DIR / "relaxed.lammpstrj"))
frame  = frames[-1]
cols   = frame['columns']
idx    = {c: i for i, c in enumerate(cols)}
pos    = frame['data']
xyz    = pos[:, [idx['x'], idx['y'], idx['z']]]

tree = cKDTree(xyz)
dists, _ = tree.query(xyz, k=2)     # k=1 is self (dist 0), k=2 is nearest neighbor
nn_dist = dists[:, 1]

# keep only plausible C-C bonded distances, excluding any numerical noise
bonded = nn_dist[(nn_dist > 1.2) & (nn_dist < 1.7)]

print(f"Mean nearest-neighbor distance: {bonded.mean():.4f} +/- {bonded.std():.4f} A "
      f"over {len(bonded)} atoms")

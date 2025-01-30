#%%
import numpy as np
import gdstk

# Constants
px = 0.4  # Period of the structure in um
py = px
gds_file_name = "NIR_achrom_test_small.gds"
# Load data from text files
file_path = "NIR_achrom_test_small.npz"
lx = np.load(file_path)["lx"]/1E-6  # Length of sqauares
ly = lx
L = px * np.shape(lx)[0]

# Create a new GDS cell
cell = gdstk.Cell("cell")
n, m = lx.shape
for i in range(n):
    for j in range(m):
        xx = lx[i, j]
        yy = ly[i, j]
        angle = 0
        center = [(i + 1/2) * px - L/2, (j + 1/2) * py - L/2] 
        rectangle_points = [
            (-xx / 2, -yy / 2),
            (xx / 2, -yy / 2),
            (xx / 2, yy / 2),
            (-xx / 2, yy / 2),
        ]
        rectangle = gdstk.Polygon(rectangle_points).rotate(angle).translate(*center)
        cell.add(rectangle)

substrate = gdstk.Cell("substrate")
center = [0,0]
rectangle_points = [
    (-L/ 2, -L / 2),
    (L / 2, -L / 2),
    (L / 2, L / 2),
    (-L / 2, L / 2),
]
rectangle = gdstk.Polygon(rectangle_points).rotate(angle).translate(*center)
cell.add(rectangle)

gds_lib = gdstk.Library()
gds_lib.add(cell)
gds_lib.add(substrate)
gds_lib.write_gds(gds_file_name)
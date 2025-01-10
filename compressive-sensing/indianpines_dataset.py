#%%
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tifffile as tiff

path = r"10_4231_R7RX991C/bundle/aviris_hyperspectral_data/19920612_AVIRIS_IndianPine_EW-line_R.tif"
dat = tiff.imread(path)
print(np.shape(dat))

plt.figure(dpi=150)
plt.imshow(dat[10,300:400,100:200])
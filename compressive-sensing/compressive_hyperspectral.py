#%%
import meep as mp
import numpy as np
import matplotlib.pyplot as plt
from psf_model import rs_psf, rs_psf_fftconv, focus_phase, fft_convolve2d
from metalens_from_lib import build_lens_from_lib
from image_model import phi_matrix
from PIL import Image
import scipy
import json
import os

def gaussian_kernel(x, y, s):
    r = np.sqrt(x**2 + y**2)
    return np.exp(-r**2/(2*s**2))

def random_c(x, y, wvl, rfac, s):
    xx, yy = np.meshgrid(x, y)
    mask = (xx**2 + yy**2 <= x.max()**2)
    mask = np.reshape(mask, (*np.shape(mask), 1))
    rand = np.random.rand(len(x), len(y), len(wvl))
    rc = np.empty(np.shape(rand))
    for i in range(len(wvl)):
        kernel = gaussian_kernel(xx, yy, s)
        rc[:,:,i] = fft_convolve2d(rand[:,:,i], kernel).real
    return mask*np.exp(2j*np.pi*rfac*rc/rc.max()*mask)

#%%
wvl_cen = 0.9
focl = 200
sx = 60
res = 2
x = np.arange(0, sx + 1/res, 1/res)
num_cells = len(x[1:])
x = np.append(-np.flip(x), x[1:])
xx, yy = np.meshgrid(x, x)

lib_path = "lib_NIR_aSi_merge_FILTERED.json"
with open(lib_path, "r") as jsonFile:
    lib = json.load(jsonFile)

rfac = 8
s = 1.9
# idx_cen = np.argmin(np.abs(wvl_cen - np.array(lib["mode_wvl"])))
# wvl_idx = np.arange(idx_cen - nf, idx_cen + nf + 1, 1)
# wvl_use = np.take(lib["mode_wvl"], wvl_idx) + 0.05
wvl_use = [0.630, 0.532, 0.467]
c_targ = np.empty((len(x), len(x), len(wvl_use)), dtype="complex")
for i in range(len(wvl_use)):
    cen = (0.2*np.random.rand()*xx.max(), 0.2*np.random.rand()*yy.max())
    cen = (0,0)
    c_targ[:,:,i] = focus_phase(x, x, focl, [wvl_use[i]], cen=cen)[:,:,0]
# c_targ *= np.exp(1j*rfac*2*np.pi*np.random.rand(*np.shape(c_targ)))

rc = random_c(x, x, wvl_use, rfac, s)
c_targ *= rc
plt.imshow(np.angle(c_targ[:,:,0]))
#%%
gparam_grid, c_opt = build_lens_from_lib(lib_path, c_targ, wvl_idx)

#%%
i = idx_cen
c = c_opt[:,:,i]
tran = np.abs(c)
phase = np.angle(c)%(-2*np.pi)
fig, ax = plt.subplots(1,2)
ax[0].imshow(tran)
ax[1].imshow(phase, cmap="jet")

#%%
psf_rs_fft = rs_psf_fftconv(x, x, [focl], wvl_use, c_targ)

i = 0
I = np.abs(psf_rs_fft[:,:,0,:])**2
fig, ax = plt.subplots(1,3,dpi=150)
col = ["R","G","B"]
for i in range(3):
    ax[i].imshow(I[83:158,83:158,i], vmin=np.min(I), vmax=np.max(I))
    ax[i].set_title(col[i])

#%%
psf_spect = I[83:158, 83:158, :]
phi = phi_matrix(psf_spect)

#%%
fig, ax = plt.subplots(1,2, dpi=150)
test_img = np.zeros(np.shape(psf_spect))
test_img[-18:-8, -15:-5, 0] = 1
# test_img[6:16, 26:36,1] = 1
# test_img[27:37,20:30,2] = 1
test_img_conv = np.reshape(phi@np.ndarray.flatten(test_img), np.shape(test_img)[:2])
ax[0].imshow(test_img)
ax[1].imshow(test_img_conv)
#%%
# LOAD IMAGES
directory = r"forest_images"
dir_list = os.listdir(directory)[:]
valid_idx = np.random.randint(0, len(dir_list), 100)
train_list = np.delete(dir_list, valid_idx)
valid_list = [dir_list[i] for i in valid_idx]
imgs_t = []
imgs_v = []
shape = [75,75]
for file in train_list:
    fname = os.fsdecode(file)
    if fname[-4:] == ".jpg":
        img = Image.open(directory+"/"+fname)
        img = img.resize(shape)
        img = np.asarray(img)
        imgs_t.append(np.ndarray.flatten(img))
for file in valid_list:
    fname = os.fsdecode(file)
    if fname[-4:] == ".jpg":
        img = Image.open(directory+"/"+fname)
        img = img.resize(shape)
        img = np.asarray(img)
        imgs_v.append(np.ndarray.flatten(img))

#%%
# LOAD/SHOW DICTIONARY REPRESENTATIONS
filepath = directory + r"/forest_images_dictlearning.npz"
dict_learned = np.load(filepath)

sparse_coeffs = dict_learned["sparse_coeffs"]
sparse_dict = dict_learned["sparse_dict"]
sparse_reps = sparse_coeffs @ sparse_dict

i = np.random.randint(len(imgs_t))
img0 = np.reshape(imgs_t[i], (*shape,3))
img1 = np.reshape((sparse_reps[i,:]/np.max(sparse_reps[i,:])*255).astype(np.uint8), (*shape,3))
fig, ax = plt.subplots(1, 2, dpi=200)
ax[0].imshow(img0)
ax[1].imshow(img1)
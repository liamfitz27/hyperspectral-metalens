#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import dctn, idctn
from image_model import phi_matrix
from sklearn.decomposition import SparseCoder
from PIL import Image
import scipy
import json
from psf_model import rs_psf, rs_psf_fftconv, focus_phase, fft_convolve2d
from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib as mpl
mpl.rcParams["backend"] = "qtagg"
%matplotlib auto


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

def group_delay(x, y, focl):
    xx, yy = np.meshgrid(x, y)
    gd = 2*np.pi*(focl - np.sqrt(focl**2 + (xx-cen[0])**2 + (yy-cen[1])**2))
    return gd

def focus_phase(x, y, focl, wvl_cen, cen=(0,0)):
    c_foc = np.empty((len(x), len(y), len(wvl_cen)), dtype="complex")
    xx, yy = np.meshgrid(x, y)
    mask = (xx**2 + yy**2 <= np.max(x)**2)
    for i in range(len(wvl_cen)):
        phi_foc = 2*np.pi/wvl_cen[i] * (focl - np.sqrt(focl**2 + (xx-cen[0])**2 + (yy-cen[1])**2))
        c_foc[:,:,i] = mask*np.exp(1j*phi_foc*mask)
    return c_foc

def DCT_dict(n, nf):
    dct_dict = np.zeros((n**2*nf, n, n, nf))
    for i in range(n):
        for j in range(n):
            for k in range(nf):
                coeff = np.zeros((n,n))
                coeff[i,j] = 1
                basis_ijk = np.zeros((n,n,nf))
                basis_ijk[:,:,k] = idctn(coeff)
                basis_ijk = np.ndarray.flatten(basis_ijk)
                dct_dict[:, i, j, k] = basis_ijk
    return np.reshape(dct_dict, (n**2*nf, n**2*nf))


#%%
wvl_rgb = np.array([0.8, 1.0, 1.3])
col = ["R","G","B"]

sweep_data = np.load(r"/Users/liam/hyperspectral-metalens/metalens_design/jan16_sweep.npz")
f = sweep_data["f"][0]
wvl = 3E8/f/1E-6
wvl_idx = []
for i in range(len(wvl_rgb)):
    wvl_idx.append(np.argmin(np.abs(wvl - wvl_rgb[i])))

trans = np.minimum(1, np.abs(sweep_data["Es"][wvl_idx,:]))
phase = np.angle(sweep_data["Es"][wvl_idx,:])
c_sweep = trans*np.exp(1j*phase)
o = 0 + 0j
c0 = [[o],[o],[o]]
# c_sweep = np.hstack((c_sweep, c0))

focl = 60
sx = 30
res = 1/0.4
x = np.arange(0, sx + 1/res, 1/res)
num_cells = len(x[1:])
x = np.append(-np.flip(x), x[1:])
xx, yy = np.meshgrid(x, x)
mask = (xx**2 + yy**2 <= x.max()**2)

r = 8
s = 1.6
c_targ = focus_phase(x, x, focl, wvl_rgb)
c_targ *= random_c(x, x, wvl_rgb, r, s)

c_opt = np.zeros([len(x), len(x), len(wvl_rgb)], dtype="complex")
for i in range(len(x)):
    for j in range(len(x)):
        s = 0
        for l in range(len(wvl_rgb)):
            s += (i+1)**1*np.abs(c_targ[i,j,l] - c_sweep[l,:])
        idx_opt = np.argmin(s)
        c_opt[i,j,:] = c_sweep[:,idx_opt]

for l in range(len(wvl_rgb)):
    c_opt[:,:,l][np.logical_not(mask)] = 0 + 0j

fig, ax = plt.subplots(2, 3, figsize=(12, 8))
fig.suptitle('Comparison of Target and Optimized Phases')

fig.text(0.5, 0.92, 'Target Phase', ha='center', va='center', fontsize=12)
fig.text(0.5, 0.5, 'Optimized Phase', ha='center', va='center', fontsize=12)

for i in range(3):
    ax[0, i].imshow(np.angle(c_targ[:, :, i]) % (-2 * np.pi))
    ax[0, i].set_title(col[i])
    ax[1, i].imshow(np.angle(c_opt[:, :, i]) % (-2 * np.pi))
    ax[1, i].set_title(col[i])
#%%

psf = rs_psf_fftconv(x, x, [focl], wvl_rgb, c_opt, pad=20)

I = np.abs(psf[:,:,0,:])**2

frac = 0.05
thresh = frac*np.max(I)
I[I<thresh] = 0

fig, ax = plt.subplots(1,3,dpi=150)
col = ["R","G","B"]
for i in range(3):
    ax[i].imshow(I[:,:,i], vmin=np.min(I), vmax=np.max(I))
    ax[i].set_title(col[i])
fig.tight_layout()

#%%
phi = phi_matrix(np.abs(psf[:,:,0,:])**2)

#%%
sp = np.shape(phi)
test_img = np.zeros((len(x), len(x), 3))
test_img[-10:, -10:, 0] = 1
test_img[6:16, 26:36,1] = 1
test_img[27:37, 7:17,2] = 1
test_img = np.ndarray.flatten(test_img)

sensor_img = phi@test_img
sensor_img = np.reshape(sensor_img, (len(x), len(x)))
plt.imshow(sensor_img)

#%%

nplots = 8
fig, ax = plt.subplots(nplots,nplots)
for i in range(nplots):
    for j in range(nplots):
        f = np.zeros(np.shape(sensor_img))
        f[i,j] = 1
        ax[i,j].imshow(idctn(f)/np.max(idctn(f)), cmap="Greys", vmin=0, vmax=1)
        ax[i,j].set_axis_off()
        ax[i,j].set_aspect("equal")
fig.suptitle("DCT Basis")
fig.tight_layout()

#%%
cmp = dctn(sensor_img)
cmp[np.shape(cmp)[0]//20:,np.shape(cmp)[0]//20:] = 0
cmp = idctn(cmp)
plt.imshow(sensor_img)
plt.figure()
plt.imshow(cmp)

#%%
ratio = 25
alpha = dctn(sensor_img)
alpha[len(x)//ratio:, len(x)//ratio:] = 0
dct_dict = DCT_dict(np.shape(sensor_img)[0])
cmp_img = dct_dict@np.ndarray.flatten(alpha)
cmp_img = np.reshape(cmp_img, (len(x), len(x)))
fig, ax = plt.subplots(1,2)
ax[0].imshow(sensor_img)
ax[0].set_title("Original img.")
ax[1].imshow(cmp_img)
ax[1].set_title(f"DCT Transform compressed \n img., ratio = 1/{ratio}")

#%%
test_img = np.reshape(test_img, (len(x), len(x), 3))
xyl_dict = DCT_dict(len(x), 3)
alpha = np.zeros(np.shape(test_img))
for i in range(3):
    alpha[:,:,i] = dctn(test_img[:,:,i])
alpha = np.ndarray.flatten(alpha)
cmp_img = xyl_dict@alpha
cmp_img = np.reshape(cmp_img, (len(x), len(x), 3))
plt.imshow((cmp_img/np.max(cmp_img)))

#%%
import scipy.sparse as sp
xyl_dict = DCT_dict(len(x), 3)
xyl_dict[xyl_dict < thresh] = 0

phi_sparse = sp.csr_matrix(phi)
xyl_dict_sparse = sp.csr_matrix(xyl_dict)

#%%
decoding_dict = phi_sparse.dot(xyl_dict_sparse.T).toarray()


#%%
coder = SparseCoder(
    dictionary=decoding_dict.T,
    transform_algorithm="lasso_lars", 
    transform_alpha=0.01, 
    )
alpha_decode = coder.transform(np.ndarray.flatten(sensor_img).reshape(1, -1))
img_decode = xyl_dict@alpha_decode[0,:]
# %%

#%%
img_decode = np.reshape(img_decode, (len(x), len(x), 3))
plt.imshow(img_decode[:,:,2])
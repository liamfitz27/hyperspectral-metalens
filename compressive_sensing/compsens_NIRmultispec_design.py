#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import dctn, idctn
from pywt import dwt2, idwt2
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
    rand = np.random.rand(len(x), len(y), len(wvl))
    rc = np.empty(np.shape(rand))
    for i in range(len(wvl)):
        kernel = gaussian_kernel(xx, yy, s)
        rc[:,:,i] = fft_convolve2d(rand[:,:,i], kernel).real
    return np.exp(2j*np.pi*rfac*rc/rc.max())

def focus_phase(x, y, focl, wvl_cen, cen=(0,0)):
    c_foc = np.empty((len(x), len(y), len(wvl_cen)), dtype="complex")
    xx, yy = np.meshgrid(x, y)
    for i in range(len(wvl_cen)):
        phi_foc = 2*np.pi/wvl_cen[i] * (focl - np.sqrt(focl**2 + (xx-cen[0])**2 + (yy-cen[1])**2))
        c_foc[:,:,i] = np.exp(1j*phi_foc)
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

def DWT_dict(n, nf, type):
    dwt_dict = np.zeros((n**2*nf, n, n, nf))
    for i in range(n):
        for j in range(n):
            for k in range(nf):
                ns = n//2
                coeff1 = np.ndarray.flatten(np.zeros((ns,ns)))
                coeff2 = np.ndarray.flatten(np.zeros((3, ns, ns)))
                coeff = np.append(coeff1, coeff2)
                ii = i*n + j
                coeff[ii] = 1
                basis_ijk = np.zeros((n,n,nf))
                coeff = (np.reshape(coeff[:ns**2], (ns, ns)), np.reshape(coeff[ns**2:], (3, ns, ns)))
                basis_ijk[:,:,k] = idwt2(coeff, type)
                basis_ijk = np.ndarray.flatten(basis_ijk)
                dwt_dict[:, i, j, k] = basis_ijk
    return np.reshape(dwt_dict, (n**2*nf, n**2*nf))

#%%
# LOAD SIM DATA
sweep_data = np.load(r"jan16_sweep.npz")
f = sweep_data["f"][0]
lx = sweep_data["lx"][0]

#%%
# CHOOSE PARAMETERS
wvl_rgb = np.array([0.8, 1.0, 1.3])
col = ["R","G","B"]
wvl = 3E8/f/1E-6
wvl_idx = []
for i in range(len(wvl_rgb)):
    wvl_idx.append(np.argmin(np.abs(wvl - wvl_rgb[i])))

trans = np.minimum(1, np.abs(sweep_data["Es"][wvl_idx,:]))
phase = np.angle(sweep_data["Es"][wvl_idx,:])
c_sweep = trans*np.exp(1j*phase)
o = 0 + 0j
# c0 = [[o],[o],[o]] # empty cell
# c_sweep = np.hstack((c_sweep, c0))

focl = 40
sx = 20
px = 0.4
res = 1/px
x = np.linspace(-sx, sx, int(2*sx*res))
xx, yy = np.meshgrid(x, x)

# CALC RANDOM PHASE, FIND BEST MATCH FROM DATA
r = 8 # Amplitude
s = 1.7 # Size of randomness correlation
c_targ = focus_phase(x, x, focl, wvl_rgb)
c_targ *= random_c(x, x, wvl_rgb, r, s)

c_opt = np.zeros([len(x), len(x), len(wvl_rgb)], dtype="complex")
lx_opt = np.zeros([len(x), len(x)])
for i in range(len(x)):
    for j in range(len(x)):
        s = 0
        for l in range(len(wvl_rgb)):
            s += (i+1)**2*np.abs(c_targ[i,j,l] - c_sweep[l,:])
        idx_opt = np.argmin(s)
        c_opt[i,j,:] = c_sweep[:,idx_opt]
        lx_opt[i,j] = lx[idx_opt]

# for l in range(len(wvl_rgb)):
#     lx_opt[np.logical_not(mask)] = 0
#     c_opt[:,:,l][np.logical_not(mask)] = 0 + 0j

#%%
# PLOT PHASES, PSFs
fig, ax = plt.subplots(2, 3, figsize=(12, 8))
fig.suptitle('Comparison of Target and Optimized Phases')

fig.text(0.5, 0.92, 'Target Phase', ha='center', va='center', fontsize=12)
fig.text(0.5, 0.5, 'Optimized Phase', ha='center', va='center', fontsize=12)

for i in range(3):
    ax[0, i].imshow(np.angle(c_targ[:, :, i]) % (-2 * np.pi))
    ax[0, i].set_title(col[i])
    ax[1, i].imshow(np.angle(c_opt[:, :, i]) % (-2 * np.pi))
    ax[1, i].set_title(col[i])
fig.tight_layout()

psf = rs_psf_fftconv(x, x, [focl], wvl_rgb, c_opt, pad=20)
I = np.abs(psf[:,:,0,:])**2

fig, ax = plt.subplots(1,3,dpi=150)
col = ["R","G","B"]
for i in range(3):
    ax[i].imshow(I[:,:,i], vmin=np.min(I), vmax=np.max(I))
    ax[i].set_title(col[i])
fig.tight_layout()

#%%
# MEASUREMENT MATRIX, SENSOR IMAGE
phi = phi_matrix(I)

sp = np.shape(phi)
test_img = np.zeros((len(x), len(x), 3))
test_img[-30:-20, -30:-20, 0] = 1
test_img[26:36, 36:46,1] = 1
test_img[27:37, 77:87,2] = 1
test_img_f = np.ndarray.flatten(test_img)

# test_img = np.zeros((len(x), len(x), 3))
# w = len(x)//3
# test_img[w:2*w, w:2*w, 0] = 1
# test_img[w:2*w, w:2*w, 1] = 1
# test_img[w:2*w, w:2*w, 2] = 1
# test_img_f = np.ndarray.flatten(test_img)

# path = r"/Users/liam/Downloads/Basic_Color_Mixing_Chart.png"
# from PIL import Image
# test_img = np.array(Image.open(path).resize((len(x), len(x))))[:,:,:-1]
# test_img[test_img[:,:,0]==129] = 0
# test_img_f = np.ndarray.flatten(test_img)

sensor_img = phi@test_img_f
sensor_img = np.reshape(sensor_img, (len(x), len(x)))
fig, ax = plt.subplots(1,2)
ax[0].imshow(test_img)
ax[1].imshow(sensor_img)

#%%
# DECODE SENSOR IMAGE
xyl_dict = DWT_dict(len(x), 3, type="haar")
decoding_dict = phi@xyl_dict

coder = SparseCoder(
    dictionary=decoding_dict.T,
    transform_algorithm="lasso_lars",
    transform_alpha=0.01,
    )
alpha_decode = coder.transform(np.ndarray.flatten(sensor_img).reshape(1, -1))
img_decode = xyl_dict@alpha_decode[0,:]
img_decode = np.reshape(img_decode, (len(x), len(x), 3))

plt.figure()
plt.imshow(img_decode/img_decode.max())

#%%
# SAVE DESIGN
filename = r"designs/NIR_3f_rand0.npz"
np.savez(filename, lx=lx_opt, wvl=wvl_rgb, p=px, h=0.7)

#%%
# PLOT DESIGN
import matplotlib.patches as patches

p = 0.4
fig, ax = plt.subplots(dpi=150)
for i in range(lx_opt.shape[0]):
    for j in range(lx_opt.shape[1]):
        lx = lx_opt[i, j] / 1E-6
        # Bottom-left corner of the square
        x = (j + 1/2) * p - lx/2
        y = (i + 1/2) * p - lx/2
        square = patches.Rectangle((x, y), lx, lx, edgecolor='black', facecolor='black')
        ax.add_patch(square)

ax.set_xlim(0, lx_opt.shape[1] * p)
ax.set_ylim(0, lx_opt.shape[0] * p)
ax.set_aspect("equal")
plt.show()

#%%
# WAVELET BASIS
n = 8
dict_plot = DWT_dict(n, 1, type='haar')
fig, ax = plt.subplots(n,n)

for i in range(n):
    for j in range(n):
        e = np.reshape(dict_plot/dict_plot.max(), (n, n, n, n))[:,:,i, j]
        ax[i,j].imshow(e, cmap="Greys", vmin=0, vmax=1)
        ax[i,j].set_aspect("equal")
        ax[i,j].set_yticks([])
        ax[i,j].set_xticks([])
fig.suptitle("DWT Basis (type: haar)")
fig.tight_layout()
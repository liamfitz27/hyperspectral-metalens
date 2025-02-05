#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('qtagg')
from scipy.optimize import minimize, Bounds
from psf_model import rs_psf_fftconv, focus_phase
from decoding_dict import DWT_dict_full, decode_dict_convolve
from tqdm import tqdm

# Realistic dispersion
def refract_index_fill(n0, lx, h, wvl):
    Afrac = lx**2
    phase = np.multiply.outer(Afrac, 2*np.pi*(1-n0)/wvl * h)
    return phase

#%%

f = np.linspace(1/3.0, 1/0.8, 20)
wvl = 1/f

L = 7.5
p = 0.6
nx = int(2*L/p)
ms_x = np.linspace(-L+p/2, L-p/2, nx)

c0 = np.zeros((nx, nx, len(wvl)), dtype="complex")
s0 = 6
for i in range(len(wvl)):
    s = s0*np.exp(1j*i*2*np.pi/len(wvl))
    cen = (int(s.real), int(s.imag))
    c0 += focus_phase(ms_x, ms_x, 4, [wvl[i]]*len(wvl), cen=cen)

phase0 = np.angle(c0)%(-2*np.pi)
plt.imshow(phase0[:,:,0])
plt.colorbar()

#%%
n0 = 4 # aSi
h = 0.7 # 700nm

def objective(lx):
    phase = refract_index_fill(n0, lx, h, wvl)
    return np.sum(np.abs(phase - phase_targ)**2)

lx0 = np.zeros((nx, nx))
for i in tqdm(range(nx)):
    for j in range(nx):
        phase_targ = phase0[i,j,:]
        bounds = Bounds(0,1)
        lx0[i,j] = minimize(objective, 0.5, method='trust-constr', bounds=bounds).x

phase0_r = refract_index_fill(n0, lx0, h, wvl)
c0_r = np.exp(1j*phase0_r)

#%%
import matplotlib.patches as patches

fig, ax = plt.subplots(1, 2, dpi=150)
p = 0.4
for i in range(lx0.shape[0]):
    for j in range(lx0.shape[1]):
        lx = lx0[i, j] * p
        # Bottom-left corner of the square
        xc = (j + 1/2) * p - lx/2
        yc = (i + 1/2) * p - lx/2
        square = patches.Rectangle((xc, yc), lx, lx, edgecolor='black', facecolor='black')
        ax[0].add_patch(square)

ax[0].set_xlim(0, lx0.shape[1] * p)
ax[0].set_ylim(0, lx0.shape[0] * p)
ax[0].set_aspect("equal")
ax[1].imshow(phase0_r[:,:,7])

#%%
zh = 4
response = rs_psf_fftconv(ms_x, ms_x, np.arange(1, 11), wvl, c0_r, pad=0)
response = np.abs(response)**2

plt.figure()
plt.imshow(response[:,:,4,0], vmin=0, vmax=np.max(response[:,:,4,:]))

px = np.array([-5,0,5])

#%%
import numpy as np
from scipy.ndimage import zoom

def resize_with_average(array, new_size):
    # Calculate the zoom factors for each dimension
    zoom_factors = (new_size[0] / array.shape[0], new_size[1] / array.shape[1], new_size[2] / array.shape[2])
    
    # Resize the array using bilinear interpolation (order=1)
    resized_array = zoom(array, zoom_factors, order=1)
    
    return resized_array

plt.figure()
response_resize = resize_with_average(response[:,:,3,:], [3,3,len(wvl)])
plt.imshow(response_resize[:,:,0])
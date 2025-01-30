#%%
import numpy as np
import matplotlib.pyplot as plt
from psf_model import rs_psf_fftconv, focus_phase
import matplotlib as mpl
mpl.rcParams["backend"] = "qtagg"
%matplotlib auto

#%%
# LOAD SIM DATA
sweep_data = np.load(r"/Users/liam/hyperspectral-metalens/compressive_sensing/jan16_sweep.npz")
f = sweep_data["f"][0]
lx = sweep_data["lx"][0]

#%%
# CHOOSE PARAMETERS
wvl_rgb = np.array([0.8, 1.0, 1.3])
wvl = 3E8/f/1E-6
wvl_idx = []
for i in range(len(wvl_rgb)):
    wvl_idx.append(np.argmin(np.abs(wvl - wvl_rgb[i])))

trans = np.minimum(1, np.abs(sweep_data["Es"][wvl_idx,:]))
phase = np.angle(sweep_data["Es"][wvl_idx,:])
c_sweep = trans*np.exp(1j*phase)

focl = 5
sx = 4
res = 1/0.4
x = np.linspace(-sx, sx, int(2*sx*res)+1)
xx, yy = np.meshgrid(x, x)
c_targ = focus_phase(x, x, focl, wvl_rgb)

c_opt = np.zeros([len(x), len(x), len(wvl_rgb)], dtype="complex")
lx_opt = np.zeros([len(x), len(x)])
for i in range(len(x)):
    for j in range(len(x)):
        s = 0
        for l in range(len(wvl_rgb)):
            s += np.abs(c_targ[i,j,l] - c_sweep[l,:])
        idx_opt = np.argmin(s)
        c_opt[i,j,:] = c_sweep[:,idx_opt]
        lx_opt[i,j] = lx[idx_opt]

#%%
# PLOT PHASES, PSFs
fig, ax = plt.subplots(3, 3, figsize=(12, 8))
fig.suptitle('Comparison of Target and Optimized Phases')

fig.text(0.08, 0.76, 'Target Phase', ha='center', va='center', fontsize=12)
fig.text(0.08, 0.5, 'Optimized Phase', ha='center', va='center', fontsize=12)

for i in range(3):
    ax[0, i].imshow(np.angle(c_targ[:, :, i]) % (-2 * np.pi))
    ax[0, i].set_title(f"{round(wvl_rgb[i]*1000)}nm")
    ax[1, i].imshow(np.angle(c_opt[:, :, i]) % (-2 * np.pi))

psf = rs_psf_fftconv(x, x, [focl], wvl_rgb, c_opt, pad=20)
I = np.abs(psf[:,:,0,:])**2
for i in range(3):
    ax[2, i].imshow(I[:,:,i], vmin=np.min(I), vmax=np.max(I))

#%%
# SAVE DESIGN
file_name = "NIR_achrom_test_small"
np.savez(file_name + ".npz", lx=lx_opt, c=c_opt)
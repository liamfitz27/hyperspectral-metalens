#%%
import numpy as np
import matplotlib.pyplot as plt
import h5py

file_path = r'/Users/liam/Downloads/Jan. 16/empty_run/E_fin.mat'
data = h5py.File(file_path, 'r')
E_fin = np.array(data['Ex_fin'])
E0 = np.zeros(len(E_fin[:,0,0]), dtype="complex")
for i in range(len(E_fin[:,0,0])):
    E0[i] = E_fin[i,0,0][0] + 1j*E_fin[i,0,0][1]

file_path = r'/Users/liam/Downloads/Jan. 16/sweep/E_fin.mat'
data = h5py.File(file_path, 'r')
E_fin = np.array(data['Ex_fin'])[:,:,2]
Es = np.zeros(np.shape(E_fin), dtype="complex")
for i in range(len(E_fin[:,0])):
    for j in range(len(E_fin[0,:])):
        Es[i,j] = E_fin[i,j][0] + 1j*E_fin[i,j][1]
        Es[i,j] *= np.exp(-1j*np.angle(E0[i]))

file_path = r'/Users/liam/Downloads/Jan. 16/sweep/para.mat'
data = h5py.File(file_path, 'r')
f = np.array(data['f'])
lx = np.array(data['l1'])
ff, ll = np.meshgrid(f, lx)

fig, ax = plt.subplots(1,2)
ax[0].set_title("Transmission")
ax[0].pcolormesh(3E8/ff/1E-9, ll/(1E-9), np.minimum(1, np.abs(Es).T), cmap="hot")
ax[0].set_xlabel('Wavelength (nm)')
ax[0].set_ylabel('Length (nm)')
ax[0].set_aspect(3)
ax[1].set_title("Phase")
ax[1].pcolormesh(3E8/ff/1E-9, ll/(1E-9), np.angle(Es).T, cmap="jet")
ax[1].set_xlabel('Wavelength (nm)')
ax[1].set_ylabel('Length (nm)')
ax[1].set_aspect(3)
fig.tight_layout()
#%%
save_path = r"metalens-design/jan16_sweep.npz"
np.savez(save_path, f=f, lx=lx, Es=Es)

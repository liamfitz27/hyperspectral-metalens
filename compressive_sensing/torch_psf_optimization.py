#%%
# OPTIMIZE PHASE 
import torch
from torch_model import *
from decoding_dict import DWT_dict_full
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import time


def objective(lx, xx, n0, h, focl):
    nx = len(xx)
    img_shape = [2*nx-2, 2*nx-2, nf]
    phase = refract_index_fill(n0, lx, h, wvl)
    c = torch.exp(1j*phase)
    psf_spect = psf_fftconv(xx, xx, torch.tensor([focl]), wvl, c, pad=nx//2)
    psf_spect = torch.abs(psf_spect[:,:,0,:])**2
    decode_dict = decode_dict_fftconv(psf_spect, wavelet_dict, img_shape)
    mc = mutual_coherence(decode_dict)
    return mc


#%%
# PARAMETERS
nf = 6
f = torch.linspace(1/0.8, 1/1.3, nf)
nf = len(f)
wvl = 1/f
focl = 50
sx = 20
res = 1
nx = int(2*sx*res)
xx = torch.linspace(-sx, sx, nx)  # x coordinates
n0 = 4
h = 7
wavelet_dict = torch.tensor(DWT_dict_full(nx, nf, nx, type="haar"))
objective_partial = partial(objective, xx=xx, n0=n0, h=h, focl=focl)

# OPTMIZATION
lx = torch.rand((nx, nx)).requires_grad_(True) # requires_grad=True for gradient calc.
optimizer = torch.optim.Adam([lx], lr=0.01)

# Optimization loop
for i in range(20):
    t0 = time.time()
    optimizer.zero_grad()
    loss = objective_partial(lx)  # Now partial_objective depends only on lx
    loss.backward()
    optimizer.step()
    t1 = time.time()
    print(f"Iteration {i}, Loss: {loss.item()}, Time: {np.round(t1-t0, 2)}")

#%%
from psf_optimization import plot_phase_psf_design

lx_opt = lx
phase_opt = refract_index_fill(n0, lx_opt, h, wvl)
c_opt = torch.exp(1j*phase_opt)
psf_opt = psf_fftconv(xx, xx, torch.tensor([focl]), wvl, c_opt, pad=20)
psf_opt = torch.abs(psf_opt[:,:,0,:])

plot_phase_psf_design(c_opt.detach().numpy(), psf_opt.detach().numpy(), lx_opt.detach().numpy(), wvl.detach().numpy())

#%%
# COMPRESSIVE SENSING IMAGE DECODING
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import SparseCoder
from decoding_dict import decode_dict_convolve
from image_model import phi_matrix, phi_matrix_full
from image_model import psf_spect_convolve, phi_matrix_full
from image_decoding import plot_results

def compressive_sense_img(actual_img, psf_spect, sparse_dict):
    # Calc. measurement matrix phi (~PSF convolution)
    phi = phi_matrix_full(psf_spect, np.shape(psf_spect))
    pad = (psf_spect.shape[0] - 2)//2 # same size
    actual_img_pad = np.pad(actual_img, ((pad,pad), (pad,pad), (0,0)))
    actual_img_f = actual_img_pad.flatten()
    # Calc. sensor image and decoding dictionary (basis elements convolved with PSFs)
    sensor_img = phi @ actual_img_f
    sensor_img += sensor_img.max()*np.random.rand(*sensor_img.shape) * 0.05# Add noise amount 0.05
    decoding_dict = phi @ sparse_dict
    decoding_dict = decode_dict_fftconv(torch.tensor(psf_spect), torch.tensor(sparse_dict), torch.tensor(actual_img_pad.shape))
    decoding_dict = decoding_dict.detach().numpy()
    # Sparse decoding algortihm
    coder = SparseCoder(
        dictionary=decoding_dict.T,
        transform_algorithm="lasso_lars",
        transform_alpha=0.01,
        )
    alpha_decode = coder.transform(sensor_img.flatten().reshape(1, -1))
    img_decode = sparse_dict @ alpha_decode[0,:]
    # Reshape img
    nwvl = psf_spect.shape[-1]
    sensor_img = np.reshape(sensor_img, np.shape(actual_img_pad)[:-1])
    img_decode = np.reshape(img_decode, np.shape(actual_img_pad))
    return sensor_img, img_decode


frac = (nx)//5
overlap = 4
test_img = np.zeros((nx, nx, nf))
test_img[1*frac-overlap:2*frac+overlap,1*frac-overlap:2*frac+overlap,0] = 1
test_img[1*frac-overlap:2*frac+overlap,2*frac-overlap:3*frac+overlap,1] = 1
test_img[1*frac-overlap:2*frac+overlap,3*frac-overlap:4*frac+overlap,2] = 1
test_img[2*frac-overlap:3*frac+overlap,1*frac-overlap:2*frac+overlap,3] = 1
test_img[2*frac-overlap:3*frac+overlap,2*frac-overlap:3*frac+overlap,4] = 1
test_img[2*frac-overlap:3*frac+overlap,3*frac-overlap:4*frac+overlap,5] = 1

sparse_dict = DWT_dict_full(nx, nf, nx, type="haar")
# sparse_dict = wavelet_dict
sensor_img, img_decode = compressive_sense_img(test_img, psf_opt.detach().numpy(), sparse_dict)

plot_results(test_img, sensor_img, img_decode, psf_opt.detach().numpy())
plot_phase_psf_design(c_opt.vdetach().numpy(), psf_opt.detach().numpy(), lx_opt.detach().numpy())
#%%
fig, axes = plt.subplots(3, 3, figsize=(10, 10))
for i, ax in enumerate(axes.flat):
    ax.imshow(img_decode[:, :, i])
    ax.set_title(f"Channel {i+1}") 
    ax.axis('off')
plt.tight_layout()
plt.show()
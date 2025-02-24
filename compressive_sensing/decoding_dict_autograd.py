#%%
# FUNCTIONS FOR FFT DECODING DICT CALCULATION USING AUTOGRAD
import jax.numpy as jnp
import jax
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
from decoding_dict import DWT_dict, DWT_dict_full
from fftconv_autograd import fft_convolve2d
from psf_fft_autograd import psf_fftconv, focus_phase
from psf_model import focus_phase, rs_psf_fftconv
from image_model import phi_matrix_full
import time
from functools import partial

# Mutual coherence for objective function
def mutual_coherence(phi):
    phi_sum = jnp.sum(phi, axis=1, keepdims=True)
    phi_norm = jnp.where(phi_sum == 0, 0, phi / phi_sum)
    a = phi_sum/jnp.size(phi)
    mc_mat = phi_norm @ phi_norm.T
    mc_mat = jnp.where(np.diag(np.ones(mc_mat.shape[0]))==1, -jnp.sqrt(phi@phi.T), mc_mat)
    unique_pairs = jnp.triu(mc_mat, k=0)
    sum = jnp.sum(unique_pairs)
    return sum

# def decode_dict_fftconv(psf_spect, sparse_dict, img_shape):
#     s0 = int(sparse_dict.shape[0]/img_shape[-1])
#     s1 = sparse_dict.shape[1]
#     pad = (int(np.sqrt(s0)) - int(np.sqrt(s1/img_shape[-1])))//2
#     decode_dict = jnp.zeros((s0, s1))
#     for i in range(s1):
#         basis = jnp.reshape(sparse_dict[:,i], img_shape)
#         basis_psf_conv = 0
#         for l in range(psf_spect.shape[-1]):
#             basis_psf_conv += fft_convolve2d(basis[:,:,l], psf_spect[:,:,l], mode="same")
#         decode_dict = decode_dict.at[:,i].set(basis_psf_conv.flatten())
#     return decode_dict

def decode_dict_fftconv(psf_spect, sparse_dict, img_shape):
    s0 = int(sparse_dict.shape[0] / img_shape[-1])  # Full spatial size (nx, ny)
    s1 = sparse_dict.shape[1]  # Number of basis functions
    basis_set = jnp.reshape(sparse_dict, (*img_shape, s1))  # (nx_full, ny_full, nwvl, s1)
    psf_spect = jnp.repeat(psf_spect[..., jnp.newaxis], s1, axis=-1)  # (nx, ny, wvl, s1)
    # Vectorized convolution over wvls and basis elements
    vmap_convolve = jax.vmap(jax.vmap(fft_convolve2d, in_axes=(2, 2), out_axes=2), in_axes=(3, 3), out_axes=3)
    basis_psf_conv = vmap_convolve(basis_set, psf_spect)  # Shape: (78, 78, 3, s1)
    # Sum along wvl dimension, reshape
    basis_psf_conv = np.sum(basis_psf_conv, axis=2) 
    decode_dict = basis_psf_conv.reshape(s0, s1)
    return decode_dict


#%%
if __name__ == "__main__":
    # PARAMETERS
    wvl_rgb = jnp.array([0.8, 1.0, 1.3])
    focl = 80
    sx = 20
    res = 1
    x = jnp.linspace(-sx, sx, int(2*sx*res))
    xx, yy = jnp.meshgrid(x, x)
    c0 = focus_phase(x, x, focl, wvl_rgb)
    wavelet_dict = DWT_dict_full(len(x), len(wvl_rgb), len(x), type="haar")
    wavelet_dict_reshaped = jnp.concatenate([jnp.reshape(wavelet_dict[:, i], (78, 78, 3)) for i in range(wavelet_dict.shape[1])], axis=0)
    
    plt.imshow(wavelet_dict_reshaped[:400,:,0]);plt.colorbar()

    #%%
    # COMPARE CALCULATION TIMES AND ACCURACY
    # Full linear transform.
    t0 = time.time()
    psf_spect = rs_psf_fftconv(x, x, [focl], wvl_rgb, c0, pad=20)
    psf_spect = jnp.abs(psf_spect[:,:,0,:])**2
    phi = phi_matrix_full(psf_spect, psf_spect.shape[:-1])
    decode_dict1 = phi @ wavelet_dict
    mc1 = mutual_coherence(jnp.array(decode_dict1))
    t1 = time.time()
    print(f"Full linear transform: {t1 - t0}s")
    #%%
    # FFTCONV
    psf_spect = rs_psf_fftconv(x, x, [focl], wvl_rgb, c0, pad=20)
    psf_spect = jnp.abs(psf_spect[:,:,0,:])**2
    img_shape = [78, 78, 3]
    t0 = time.time()
    decode_dict2 = decode_dict_fftconv(psf_spect, wavelet_dict, img_shape)
    mc2 = mutual_coherence(decode_dict2)
    t1 = time.time()
    print(f"FFTconv: {t1 - t0}s")

    #%%
    # FFTCONV2
    psf_spect = rs_psf_fftconv(x, x, [focl], wvl_rgb, c0, pad=20)
    psf_spect = jnp.abs(psf_spect[:,:,0,:])**2
    img_shape = [78, 78, 3]
    t0 = time.time()
    decode_dict3 = decode_dict_fftconv2(psf_spect, wavelet_dict, img_shape)
    mc2 = mutual_coherence(decode_dict2)
    t1 = time.time()
    print(f"FFTconv2: {t1 - t0}s")

    #%%
    plt.figure()
    plt.imshow(decode_dict1)
    plt.figure()
    plt.imshow(decode_dict2)
    # plt.figure()
    # plt.imshow(decode_dict3)

# %%
#
    print(f"Mutual coherence % error: {jnp.abs((mc1 - mc2)/mc1)*100}")
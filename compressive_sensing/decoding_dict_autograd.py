#%%
# FUNCTIONS FOR FFT DECODING DICT CALCULATION USING AUTOGRAD
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
from decoding_dict import DWT_dict, DWT_dict_full
from fftconv_autograd import fft_convolve2d
from psf_fft_autograd import psf_fftconv, focus_phase
from psf_model import focus_phase, rs_psf_fftconv
from image_model import phi_matrix_full
import time

# Mutual coherence for objective function
def mutual_coherence(phi):
    phi /= jnp.sum(phi, axis=1, keepdims=True)
    mc_mat = phi @ phi.T
    idx_diag = jnp.diag(np.ones(mc_mat.shape[0], dtype=bool))
    mc_mat = mc_mat.at[idx_diag].set(-mc_mat[idx_diag]) # Want high intensity
    unique_pairs = jnp.triu(mc_mat, k=1)
    return jnp.sum(unique_pairs)

def decode_dict_fftconv(psf_spect, sparse_dict, img_shape):
    s1 = int(sparse_dict.shape[0]/img_shape[-1])
    s2 = sparse_dict.shape[1]
    pad = (int(np.sqrt(s1)) - int(np.sqrt(s2/img_shape[-1])))//2
    decode_dict = jnp.zeros((s1, s2))
    for i in range(s2):
        basis = jnp.reshape(sparse_dict[:,i], img_shape)
        basis_psf_conv = 0
        for l in range(psf_spect.shape[-1]):
            basis_psf_conv += fft_convolve2d(basis[pad+1:-pad,pad+1:-pad,l], psf_spect[:,:,l], mode="full")
        decode_dict = decode_dict.at[:,i].set(basis_psf_conv.flatten())
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

    #%%
    # COMPARE CALCULATION TIMES AND ACCURACY
    wavelet_dict = DWT_dict_full(len(x), len(wvl_rgb), len(x), type="haar")

    # Full linear transform.
    t0 = time.time()
    psf_spect = rs_psf_fftconv(x, x, [focl], wvl_rgb, c0, pad=20)
    psf_spect = jnp.abs(psf_spect[:,:,0,:])**2
    phi = phi_matrix_full(psf_spect, psf_spect.shape[:-1])
    decode_dict1 = phi @ wavelet_dict
    mc1 = mutual_coherence(decode_dict1)
    t1 = time.time()
    print(f"Full linear transform: {t1 - t0}s")
    #%%
    # FFTCONV
    t0 = time.time()
    psf_spect = rs_psf_fftconv(x, x, [focl], wvl_rgb, c0, pad=20)
    psf_spect = jnp.abs(psf_spect[:,:,0,:])**2
    img_shape = [78, 78, 3]
    decode_dict2 = decode_dict_fftconv(psf_spect, wavelet_dict, img_shape)
    mc2 = mutual_coherence(decode_dict2)
    t1 = time.time()
    print(f"FFTconv: {t1 - t0}s")

    plt.figure()
    plt.imshow(decode_dict1)
    plt.figure()
    plt.imshow(decode_dict2)

    print(f"Mutual coherence % error: {jnp.abs((mc1 - mc2)/mc1)*100}")
#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
from pywt import dwt2, idwt2
from psf_model import focus_phase, rs_psf_fftconv, fft_convolve2d
from image_model import psf_spect_convolve, phi_matrix_full
import time

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

def DWT_dict_full(n, nf, nimg, type):
    nfull = n + nimg - 2
    dwt_dict = np.zeros((nfull**2*nf, n, n, nf))
    for i in range(n):
        for j in range(n):
            for k in range(nf):
                ns = n//2
                coeff1 = np.ndarray.flatten(np.zeros((ns,ns)))
                coeff2 = np.ndarray.flatten(np.zeros((3, ns, ns)))
                coeff = np.append(coeff1, coeff2)
                ii = i*n + j
                coeff[ii] = 1
                basis_ijk = np.zeros((nfull,nfull,nf))
                coeff = (np.reshape(coeff[:ns**2], (ns, ns)), np.reshape(coeff[ns**2:], (3, ns, ns)))
                wavelet = idwt2(coeff, type)
                wavelet = np.pad(wavelet, (nimg - 1)//2)
                if n % 2 != 0:
                    basis_ijk[:-1,:-1,k] = wavelet
                else:
                    basis_ijk[:,:,k] = wavelet
                basis_ijk = np.ndarray.flatten(basis_ijk)
                dwt_dict[:, i, j, k] = basis_ijk
    return np.reshape(dwt_dict, (nfull**2*nf, n**2*nf))

def mutual_coherence(phi):
    mc = phi @ phi.T
    return np.max(mc)

def decode_dict_convolve(psf_spect, sparse_dict, img_shape):
    s1 = int(sparse_dict.shape[0]/3)
    s2 = sparse_dict.shape[1]
    decode_dict = np.zeros((s1, s2))
    for i in range(s2):
        basis = np.reshape(sparse_dict[:,i], img_shape)
        basis_psf_conv = 0
        for l in range(psf_spect.shape[-1]):
            basis_psf_conv += fftconvolve(basis[:,:,l], psf_spect[:,:,l], mode="same")
        decode_dict[:,i] = np.ndarray.flatten(basis_psf_conv)
    return decode_dict

#%%
if __name__ == "__main__":
    # PARAMETERS
    wvl_rgb = np.array([0.8, 1.0, 1.3])
    focl = 80
    sx = 20
    res = 1
    x = np.linspace(-sx, sx, int(2*sx*res))
    xx, yy = np.meshgrid(x, x)
    c0 = focus_phase(x, x, focl, wvl_rgb)

    #%%
    psf_spect = rs_psf_fftconv(x, x, [focl], wvl_rgb, c0, pad=20)
    psf_spect = np.abs(psf_spect[:,:,0,:])**2
    phi = phi_matrix_full(psf_spect, psf_spect.shape[:-1])

    #%%
    # FULL LINEAR TRANSFORMATION DECODING DICT vs. FFTCONV DECODING DICT (TRANSL INVAR.)
    wavelet_dict = DWT_dict_full(len(x), len(wvl_rgb), len(x), type="haar")

    # full linear transformation
    decode_dict1 = phi @ wavelet_dict
    xy_shape = np.shape(fftconvolve(c0[:,:,0], c0[:,:,0], mode="full"))
    img_shape = [xy_shape[0]-1, xy_shape[1]-1, len(wvl_rgb)]
    # transl. invar. psf and fftconv
    decode_dict2 = decode_dict_convolve(psf_spect, wavelet_dict, img_shape)

    plt.imshow(decode_dict1 - decode_dict2, vmin=-2*decode_dict1.max(), vmax=2*decode_dict1.max())
    plt.colorbar()

    #%%
    # COMPARE CALCULATION TIMES AND ACCURACY

    # Full linear transform.
    t0 = time.time()
    psf_spect = rs_psf_fftconv(x, x, [focl], wvl_rgb, c0, pad=20)
    psf_spect = np.abs(psf_spect[:,:,0,:])**2
    phi = phi_matrix_full(psf_spect, psf_spect.shape[:-1])
    decode_dict1 = phi @ wavelet_dict
    mc1 = mutual_coherence(decode_dict1)
    t1 = time.time()
    print(f"Full linear transform: {t1 - t0}s")

    # FFTconv
    t0 = time.time()
    psf_spect = rs_psf_fftconv(x, x, [focl], wvl_rgb, c0, pad=20)
    psf_spect = np.abs(psf_spect[:,:,0,:])**2
    img_shape = [78, 78, 3]
    decode_dict2 = decode_dict_convolve(psf_spect, wavelet_dict, img_shape)
    mc2 = mutual_coherence(decode_dict2)
    t1 = time.time()
    print(f"FFTconv: {t1 - t0}s")

    plt.imshow(decode_dict1 - decode_dict2, vmin=-2*decode_dict1.max(), vmax=2*decode_dict1.max())
    plt.colorbar()

    print(f"Mutual coherence % error: {np.abs((mc1 - mc2)/mc1)*100}")

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import SparseCoder
from decoding_dict import decode_dict_convolve
from image_model import phi_matrix, phi_matrix_full
from image_model import psf_spect_convolve, phi_matrix_full

def compressive_sense_img(actual_img, psf_spect, sparse_dict):
    # Calc. measurement matrix phi (~PSF convolution)
    phi = phi_matrix_full(psf_spect, np.shape(actual_img))
    # Calc. padding for "full" convolution
    p_extra = (psf_spect.shape[0]%2==0, psf_spect.shape[1]%2==0)
    npad = (psf_spect.shape[0]-1-p_extra[0], psf_spect.shape[1]-1-p_extra[1])
    actual_img_pad = np.pad(actual_img, ((npad[0]//2,npad[0]//2), (npad[1]//2,npad[1]//2), (0,0)))
    actual_img_pad_f = actual_img_pad.flatten()
    # Calc. sensor image and decoding dictionary (basis elements convolved with PSFs)
    sensor_img = phi @ actual_img_pad_f
    decoding_dict = phi @ sparse_dict
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
    sensor_img =np.reshape(sensor_img, np.shape(actual_img_pad)[:-1])
    img_decode = np.reshape(img_decode, np.shape(actual_img_pad))
    return sensor_img, img_decode


def plot_results(actual_img, sensor_img, img_decode, psf_spect):
    fig, ax = plt.subplots(1, 3, dpi=150)
    ax[0].set_title("Actual Image")
    ax[0].imshow(actual_img)
    ax[1].set_title("Sensor Image")
    ax[1].imshow(sensor_img)
    ax[2].set_title("Decoded Image")
    ax[2].imshow(img_decode/img_decode.max())

    nwvl = psf_spect.shape[-1]
    fig, ax = plt.subplots(1, nwvl, dpi=150)
    for i in range(nwvl):
        ax[i].imshow(psf_spect[:,:,i], vmin=psf_spect.min(), vmax=psf_spect.max())


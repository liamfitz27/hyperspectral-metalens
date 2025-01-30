#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.signal import fftconvolve
from psf_model import rs_psf_fftconv, focus_phase
from tqdm import tqdm

def phi_pixel(psf_spect, i, j):
    sp = np.shape(psf_spect)
    cen = (sp[0]//2, sp[1]//2)
    phi_ij = np.empty(sp)
    for n in range(sp[0]):
        for m in range(sp[1]):
            idx = (cen[0] + i - n, cen[1] + j - m)
            if idx[0] < 0 or idx[1] < 0 or idx[0] >= sp[0] or idx[1] >= sp[1]:
                phi_ij[n, m, :] = 0
            else:
                phi_ij[n, m, :] = psf_spect[idx[0], idx[1], :]
    return phi_ij

def phi_matrix(psf_spect):
    sp = np.shape(psf_spect)
    phi = np.empty((sp[0]*sp[1], sp[0]*sp[1]*sp[2]))
    for i in tqdm(range(sp[0])):
        for j in range(sp[1]):
            phi_ij = phi_pixel(psf_spect, i, j)
            phi[i * sp[0] + j, :] = np.ndarray.flatten(phi_ij)
    return phi

def phi_pixel_full(psf_spect, conv_shape, i, j):
    psf_shape = np.shape(psf_spect)
    cen = (psf_shape[0]//2, psf_shape[1]//2)
    phi_ij = np.zeros([*conv_shape, psf_shape[-1]])
    for n in range(conv_shape[0]):
        for m in range(conv_shape[1]):
            idx = (cen[0] + i - n, cen[1] + j - m)
            if idx[0] < 0 or idx[1] < 0 or idx[0] >= psf_shape[0] or idx[1] >= psf_shape[1]:
                phi_ij[n, m, :] = 0
            else:
                phi_ij[n, m, :] = psf_spect[idx[0], idx[1], :]
    return phi_ij

def phi_matrix_full(psf_spect, img_shape):
    psf_shape = np.shape(psf_spect)
    conv_shape = np.array([0,0])
    conv_shape[0] = img_shape[0] + psf_shape[0] - 1 - (psf_shape[0]%2==0)
    conv_shape[1] = img_shape[1] + psf_shape[1] - 1 - (psf_shape[1]%2==0)
    phi = np.empty((conv_shape[0]*conv_shape[1], conv_shape[0]*conv_shape[1]*psf_shape[2]))
    for i in tqdm(range(conv_shape[0])):
        for j in range(conv_shape[1]):
            phi_ij = phi_pixel_full(psf_spect, conv_shape, i, j)
            phi[i * conv_shape[0] + j, :] = np.ndarray.flatten(phi_ij)
    return phi

def psf_spect_convolve(psf_spect, source, mode="full"):
    img = 0
    for l in range(np.shape(psf_spect)[-1]):
        img += fftconvolve(source[:,:,l], psf_spect[:,:,l], mode="full")
    return img


#%%
if __name__ == "__main__":
    R = 50
    F = 100
    x = np.linspace(-50, 50, 801)
    u = np.linspace(-5, 5, 81)
    idx1 = np.argmin(np.abs(x+5))
    idx2 = np.argmin(np.abs(x-5))
    xx, yy = np.meshgrid(x, x)
    mask = (xx**2 + yy**2 <= R**2)
    wvl = [0.630, 0.532, 0.467]  # R=630nm, G=532nm, and B=467nm.

    c_foc = focus_phase(x, x, F, wvl)
    psf_spect = rs_psf_fftconv(x, x, [F], wvl, c_foc, pad=20)[idx1-1:idx2,idx1-1:idx2,0,:]
    psf_spect = np.abs(psf_spect)**2
    phi = phi_matrix_full(psf_spect, [len(u), len(u)])

    test_img = np.zeros((len(u), len(u), 3))
    test_img[-10:, -10:, 0] = 1
    test_img[6:16, 26:36,1] = 1
    test_img[27:37, 7:17,2] = 1

    test_img_pad = np.zeros((2*len(u)-1, 2*len(u)-1, 3))
    for i in range(test_img_pad.shape[-1]):
        test_img_pad[:,:,i] = np.pad(test_img[:,:,i], len(u)//2)

    test_img_conv1 = phi@np.ndarray.flatten(test_img_pad)
    test_img_conv1 = np.reshape(test_img_conv1, (2*len(u)-1, 2*len(u)-1))

    test_img_conv2 = psf_spect_convolve(psf_spect, test_img)

    #%%
    # PSFs plot
    fig, ax = plt.subplots(1,3, dpi=150)
    color = ["R", "G", "B"]
    for i in range(len(wvl)):
        ax[i].set_title(color[i])
        ax[i].imshow(psf_spect[:,:,i], vmin=np.min(psf_spect), vmax=np.max(psf_spect))
    fig.suptitle("Point Spread Function")
    fig.tight_layout()

    # Image plot
    fig, ax = plt.subplots(1,3, dpi=150)
    ax[0].set_title("RGB image")
    ax[0].imshow(test_img)
    ax[1].set_title("Sensor image 1")
    ax[1].imshow(test_img_conv1)
    ax[2].set_title("Sensor image 2")
    ax[2].imshow(test_img_conv2)
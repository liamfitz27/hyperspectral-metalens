#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
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

    #%%
    phi = phi_matrix(np.abs(psf_spect)**2)

    #%%
    fig, ax = plt.subplots(1,3, dpi=150)
    color = ["R", "G", "B"]
    for i in range(len(wvl)):
        I = np.abs(psf_spect)**2
        ax[i].set_title(color[i])
        ax[i].imshow(I[:,:,i], vmin=np.min(I), vmax=np.max(I))
    fig.suptitle("Point Spread Function")
    fig.tight_layout()

    fig, ax = plt.subplots(1,2, dpi=150)
    test_img = np.zeros((len(u), len(u), 3))
    test_img[-10:, -10:, 0] = 1
    uu, vv = np.meshgrid(u, u)
    rcirc = 0.5
    cen = (3, -2)
    circ = ((uu-cen[0])**2+(vv-cen[1])**2 <= rcirc**2)
    test_img[6:16, 26:36,1] = 1
    test_img[27:37, 7:17,2] = 1
    ax[0].set_title("RGB image")
    ax[0].imshow(test_img)


    test_img_conv = phi@np.ndarray.flatten(test_img)
    test_img_conv = np.reshape(test_img_conv, np.shape(uu))
    ax[1].set_title("Sensor image")
    ax[1].imshow(test_img_conv)
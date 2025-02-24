#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from scipy import interpolate
from tqdm import tqdm
import time

# Rayeigh-Sommerfeld Green's function
def rs_kernel(x, y, z, wvl):
    k = 2*np.pi/wvl
    r = np.sqrt(x**2 + y**2 + z**2)
    kernel = 1/(2*np.pi) * (1/r - 1j*k) * z/r * np.exp(1j*k*r)/r
    return kernel

# Rayleigh-Sommerfeld PSF calc using FFT convolution
def psf_fftconv(x, y, z, wvl, c, pad=0):
    psf = np.zeros((len(x), len(y), len(z), len(wvl)), dtype="complex")
    xx, yy = np.meshgrid(x, y)
    for k in range(len(z)):
        for l in range(len(wvl)):
            kernel = rs_kernel(xx, yy, z[k], wvl[l])
            if pad:
                c_pad = np.pad(c[:,:,l], pad)
                kernel_pad = np.pad(kernel, pad)
                psf_pad = fft_convolve2d(c_pad, kernel_pad)
                psf[:,:,k,l] = psf_pad[pad:-pad, pad:-pad]
            else:
                psf[:,:,k,l] = fft_convolve2d(c[:,:,l], kernel)
    return psf

# Lens focusing phase
def focus_phase(x, y, focl, wvl_cen, cen=(0,0)):
    c_foc = np.empty((len(x), len(y), len(wvl_cen)), dtype="complex")
    xx, yy = np.meshgrid(x, y)
    for i in range(len(wvl_cen)):
        phi_foc = 2*np.pi/wvl_cen[i] * (focl - np.sqrt(focl**2 + (xx-cen[0])**2 + (yy-cen[1])**2))
        c_foc[:,:,i] = np.exp(1j*phi_foc)
    return c_foc

# Calculate Rayleigh-Sommerfeld integral
def rayleigh_sommerfeld(x, y, z, wvl, xp, yp, c):
    xs = xp - x
    ys = yp - y
    xx, yy = np.meshgrid(xs, ys)
    kernel = c*rs_kernel(xx, yy, z, wvl)
    kernel_interp_r = interpolate.RectBivariateSpline(xs, ys, kernel.real)
    kernel_interp_i = interpolate.RectBivariateSpline(xs, ys, kernel.imag)
    bounds = [xs.min(), xs.max(), ys.min(), ys.max()]
    intg = kernel_interp_r.integral(*bounds) + 1j*kernel_interp_i.integral(*bounds)
    return intg

# Rayleigh-Sommerfeld diffraction PSF
def rs_psf(x, y, z, wvl, xp, yp, c):
    psf = np.zeros((len(x), len(y), len(z), len(wvl)), dtype="complex")
    psf_f = psf.flatten()
    for n in tqdm(range(len(psf_f))):
        i = int(n/(len(y)*len(z)*len(wvl)))
        j = int((n - i*len(y)*len(z)*len(wvl))/(len(z)*len(wvl)))
        k = int((n - i*len(y)*len(z)*len(wvl) - j*len(z)*len(wvl))/len(wvl))
        l = n - i*len(y)*len(z)*len(wvl) - j*len(z)*len(wvl) - k*len(wvl)
        psf_f[n] = rayleigh_sommerfeld(x[i], y[j], z[k], wvl[l], xp, yp, c[:,:,l])
    psf = np.reshape(psf_f, np.shape(psf))
    return psf

# Realistic dispersion
def refract_index_fill(n0, lx, h, wvl):
    Afrac = lx**2
    phase = np.multiply.outer(Afrac, 2*np.pi*(1-n0)/wvl * h)
    return phase

# 2D FFT convolution with 0-padding
def fft_convolve2d(x, y, pad=0):
    if pad:
        xp = np.zeros((np.shape(x)[0]+2*pad, np.shape(x)[1]+2*pad), dtype="complex")
        xp[pad:-pad, pad:-pad] = x
        yp = np.zeros((np.shape(x)[0]+2*pad, np.shape(x)[1]+2*pad), dtype="complex")
        yp[pad:-pad, pad:-pad] = y
        xf = np.fft.fftshift(np.fft.fft2(xp))
        yf = np.fft.fftshift(np.fft.fft2(yp))
        return np.fft.fftshift(np.fft.ifft2(xf * yf))[pad:-pad, pad:-pad]
    else:
        xf = np.fft.fft2(x)
        yf = np.fft.fft2(y)
        return np.fft.fftshift(np.fft.ifft2(xf * yf))


#%%
if __name__ == "__main__":
    # PARAMETERS
    N = 80
    R = 50 # Lens rad.
    F = 200 # Lens foc.
    z = [F]
    wvl = np.array([0.630, 0.532, 0.467]) # R=630nm, G=532nm, and B=467nm.

    #%%
    # CALC RS PSF USING INTEGRAL FOR EACH PIXEL
    lp = 50
    ls = 5
    xp = np.linspace(-lp, lp, N+1) # x-phase
    xs = np.linspace(-ls, ls, N+1) # x-sensor
    c = focus_phase(xp, xp, F, wvl)
    t0 = time.time()
    psf_rs = rs_psf(xs, xs, z, wvl, xp, xp, c)
    psf_rs = np.abs(psf_rs[:,:,0,:])**2
    t1 = time.time()
    print(f"Calc. time = {t1 - t0}s")

    #%%
    # CALC RS PSF USING FFT CONVOLUTION
    x = np.linspace(-50, 50, 10*N+1)
    c = focus_phase(x, x, F, wvl)
    idx = (np.argmin(np.abs(-ls - x)), np.argmin(np.abs(ls - x)))
    t0 = time.time()
    psf_rs_fftconv = psf_fftconv(x, x, z, wvl, c)
    psf_rs_fftconv = np.abs(psf_rs_fftconv[idx[0]-1:idx[1], idx[0]-1:idx[1], 0, :])**2
    t1 = time.time()
    print(f"Calc. time = {t1 - t0}s")

    #%%
    # COMPARE
    l=2
    fig, ax = plt.subplots(1,2)
    fig.suptitle("Rayleigh Sommerfeld Diffraction PSF")
    ax[0].imshow(np.array(psf_rs[:,:,l]), vmin=np.min(np.array(psf_rs)), vmax=np.max(np.array(psf_rs)))
    ax[0].title.set_text("Integral for each pixel")
    ax[1].imshow(np.array(psf_rs_fftconv[:,:,l]), vmin=np.min(np.array(psf_rs_fftconv)), vmax=np.max(np.array(psf_rs_fftconv)))
    ax[1].title.set_text("FFT Convolution")
    fig.tight_layout()

    plt.figure()
    plt.title("Convolution vs integral % err")
    plt.imshow((np.array(psf_rs[:,:,l]-psf_rs_fftconv[:,:,l])/np.array(psf_rs[:,:,l]))*100)
    plt.colorbar()
    
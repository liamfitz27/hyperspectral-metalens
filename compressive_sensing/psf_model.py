#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from tqdm import tqdm
from scipy.signal import fftconvolve
import time

# Huygen's Diffraction PSF
def point_source(x, y, z, c, wvl):
    k = 2*np.pi/wvl
    r = np.sqrt(x**2 + y**2 + z**2)
    return c*np.exp(1j*k*r) / r

# Calc 
def point_source_sum(x, y, z, wvl, xp, yp, c):
    xs = xp - x
    ys = yp - y
    xx, yy = np.meshgrid(xs, ys)
    source = 1/(1j*wvl)*point_source(xx, yy, z, c, wvl)
    source_interp_r = interpolate.RectBivariateSpline(xs, ys, source.real)
    source_interp_i = interpolate.RectBivariateSpline(xs, ys, source.imag)
    bounds = [xs.min(), xs.max(), ys.min(), ys.max()]
    source_sum = source_interp_r.integral(*bounds) + 1j*source_interp_i.integral(*bounds)
    return source_sum

# Huygen
def huygens_psf(x, y, z, wvl, xp, yp, c):
    psf = np.zeros((len(x), len(y), len(z), len(wvl)), dtype="complex")
    psf_f = np.ndarray.flatten(psf)
    for n in tqdm(range(len(psf_f))):
        i = int(n/(len(y)*len(z)*len(wvl)))
        j = int((n - i*len(y)*len(z)*len(wvl))/(len(z)*len(wvl)))
        k = int((n - i*len(y)*len(z)*len(wvl) - j*len(z)*len(wvl))/len(wvl))
        l = n - i*len(y)*len(z)*len(wvl) - j*len(z)*len(wvl) - k*len(wvl)
        psf_f[n] = point_source_sum(x[i], y[j], z[k], wvl[l], xp, yp, c[:,:,l])
    psf = np.reshape(psf_f, np.shape(psf))
    return psf/np.max(np.abs(psf))

# Rayeigh-Sommerfeld Green's function
def rs_kernel(x, y, z, wvl):
    k = 2*np.pi/wvl
    r = np.sqrt(x**2 + y**2 + z**2)
    kernel = 1/(1j*wvl)*z/r*np.exp(1j*k*r)/r
    return kernel

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
    psf_f = np.ndarray.flatten(psf)
    for n in tqdm(range(len(psf_f))):
        i = int(n/(len(y)*len(z)*len(wvl)))
        j = int((n - i*len(y)*len(z)*len(wvl))/(len(z)*len(wvl)))
        k = int((n - i*len(y)*len(z)*len(wvl) - j*len(z)*len(wvl))/len(wvl))
        l = n - i*len(y)*len(z)*len(wvl) - j*len(z)*len(wvl) - k*len(wvl)
        psf_f[n] = rayleigh_sommerfeld(x[i], y[j], z[k], wvl[l], xp, yp, c[:,:,l])
    psf = np.reshape(psf_f, np.shape(psf))
    return psf/np.max(np.abs(psf))

# Rayleigh-Sommerfeld PSF calc using FFT convolution
def rs_psf_fftconv(x, y, z, wvl, c, pad):
    psf = np.zeros((len(x), len(y), len(z), len(wvl)), dtype="complex")
    xx, yy = np.meshgrid(x, y)
    for k in range(len(z)):
        for l in range(len(wvl)):
            kernel = rs_kernel(xx, yy, z[k], wvl[l])
            psf[:,:,k,l] = fft_convolve2d(c[:,:,l], kernel, pad=20)
    return psf/np.max(np.abs(psf))

# Fresnel diffraction integral calculated using FFT
def fresnel_fft(x, y, z, wvl, c):
    k = 2*np.pi/wvl
    xx, yy = np.meshgrid(x, y)
    kernel = np.exp(1j*k/(2*z)*(xx**2 + yy**2))
    psf = np.fft.fftshift(np.fft.fft2(c*kernel))
    psf *= np.exp(1j*k*z)/(1j*wvl*z)*np.exp(1j*k/(2*z)*(xx**2+yy**2))
    return psf/np.max(np.abs(psf))

# Fresnel diffraction PSF
def fresnel_psf(x, y, z, wvl, c):
    psf = np.zeros((len(x), len(y), len(z), len(wvl)), dtype="complex")
    for n in tqdm(range(len(z)*len(wvl))):
        k = int(n/len(wvl))
        l = n - k*len(wvl)
        psf[:,:,k,l] = fresnel_fft(x, y, z[k], wvl[l], c[:,:,l])
    return psf/np.max(np.abs(psf))

# 2D FFT convolution with 0-padding
def fft_convolve2d(x, y, pad=0):
    if pad:
        xp = np.zeros((np.shape(x)[0]+2*pad, np.shape(x)[1]+2*pad), dtype="complex")
        xp[pad:-pad, pad:-pad] = x
        yp = np.zeros((np.shape(x)[0]+2*pad, np.shape(x)[1]+2*pad), dtype="complex")
        yp[pad:-pad, pad:-pad] = y
        xf = np.fft.fftshift(np.fft.fft2(xp))
        yf = np.zeros((np.shape(y)[0]+2*pad, np.shape(y)[1]+2*pad), dtype="complex")
        yf = np.fft.fftshift(np.fft.fft2(yp))
        return np.fft.fftshift(np.fft.ifft2(xf * yf))[pad-1:-pad-1, pad-1:-pad-1]
    else:
        xf = np.fft.fft2(x)
        yf = np.fft.fft2(y)
        return np.fft.fftshift(np.fft.ifft2(xf * yf))

def fft_convolve2d_full(x, y, pad=0):
    n_conv = len(x) + len(y) - 1 - (len(y)%2 == 0)
    n_pad_x = (n_conv - len(x))//2
    n_pad_y = (n_conv - len(y))//2
    x = np.pad(x, n_pad_x)
    y = np.pad(y, n_pad_y)
    if pad:
        xp = np.zeros((np.shape(x)[0]+2*pad, np.shape(x)[1]+2*pad), dtype="complex")
        xp[pad:-pad, pad:-pad] = x
        yp = np.zeros((np.shape(x)[0]+2*pad, np.shape(x)[1]+2*pad), dtype="complex")
        yp[pad:-pad, pad:-pad] = y
        xf = np.fft.fftshift(np.fft.fft2(xp))
        yf = np.zeros((np.shape(y)[0]+2*pad, np.shape(y)[1]+2*pad), dtype="complex")
        yf = np.fft.fftshift(np.fft.fft2(yp))
        return np.fft.fftshift(np.fft.ifft2(xf * yf))[pad-1:-pad-1, pad-1:-pad-1]
    else:
        xf = np.fft.fft2(x)
        yf = np.fft.fft2(y)
        return np.fft.fftshift(np.fft.ifft2(xf * yf))

# Lens focusing phase
def focus_phase(x, y, focl, wvl_cen, cen=(0,0)):
    c_foc = np.empty((len(x), len(y), len(wvl_cen)), dtype="complex")
    xx, yy = np.meshgrid(x, y)
    for i in range(len(wvl_cen)):
        phi_foc = 2*np.pi/wvl_cen[i] * (focl - np.sqrt(focl**2 + (xx-cen[0])**2 + (yy-cen[1])**2))
        c_foc[:,:,i] = np.exp(1j*phi_foc)
    return c_foc

#%%
if __name__ == "__main__":
    # PARAMETERS
    N = 80
    R = 50 # Lens rad.
    F = 200 # Lens foc.
    z = [F]
    wvl = [0.630, 0.532, 0.467] # R=630nm, G=532nm, and B=467nm.

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
    psf_rs_fftconv = rs_psf_fftconv(x, x, z, wvl, c, pad=20)
    psf_rs_fftconv = np.abs(psf_rs_fftconv[idx[0]:idx[1]+1, idx[0]:idx[1]+1, 0, :])**2
    t1 = time.time()
    print(f"Calc. time = {t1 - t0}s")

    #%%
    # COMPARE
    l = 1
    fig, ax = plt.subplots(1,2)
    fig.suptitle("Rayleight Sommerfeld Diffraction PSF")
    ax[0].imshow(psf_rs[:,:,l], vmin=np.min(psf_rs), vmax=np.max(psf_rs))
    ax[0].title.set_text("Integral for each pixel")
    ax[1].imshow(psf_rs_fftconv[:,:,l], vmin=np.min(psf_rs_fftconv), vmax=np.max(psf_rs_fftconv))
    ax[1].title.set_text("FFT Convolution")
    fig.tight_layout()

    plt.figure()
    plt.title("Convolution vs integral residuals")
    plt.imshow(psf_rs - psf_rs_fftconv);plt.colorbar()



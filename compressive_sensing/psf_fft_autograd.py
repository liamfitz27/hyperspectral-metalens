#%%
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib import patches
from scipy import interpolate
from tqdm import tqdm
from fftconv_autograd import fft_convolve2d
import time
import jax
import nlopt
import numpy as np

# Rayeigh-Sommerfeld Green's function
def rs_kernel(x, y, z, wvl):
    k = 2*jnp.pi/wvl
    r = jnp.sqrt(x**2 + y**2 + z**2)
    kernel = 1/(2*jnp.pi) * (1/r - 1j*k) * z/r * jnp.exp(1j*k*r)/r
    return kernel

# Rayleigh-Sommerfeld PSF calc using FFT convolution
def psf_fftconv(x, y, z, wvl, c, pad=0):
    psf = jnp.zeros((len(x), len(y), len(z), len(wvl)), dtype="complex")
    xx, yy = jnp.meshgrid(x, y)
    for k in range(len(z)):
        for l in range(len(wvl)):
            kernel = rs_kernel(xx, yy, z[k], wvl[l])
            if pad:
                c_pad = jnp.pad(c[:,:,l], pad)
                kernel_pad = jnp.pad(kernel, pad)
                psf_pad = fft_convolve2d(c_pad, kernel_pad, mode="same")
                psf = psf.at[:,:,k,l].set(psf_pad[pad:-pad, pad:-pad])
            else:
                psf = psf.at[:,:,k,l].set(fft_convolve2d(c[:,:,l], kernel, mode="same"))
    return psf

# Lens focusing phase
def focus_phase(x, y, focl, wvl_cen, cen=(0,0)):
    c_foc = jnp.empty((len(x), len(y), len(wvl_cen)), dtype="complex")
    xx, yy = jnp.meshgrid(x, y)
    for i in range(len(wvl_cen)):
        phi_foc = 2*jnp.pi/wvl_cen[i] * (focl - jnp.sqrt(focl**2 + (xx-cen[0])**2 + (yy-cen[1])**2))
        c_foc = c_foc.at[:,:,i].set(jnp.exp(1j*phi_foc))
    return c_foc

# Calculate Rayleigh-Sommerfeld integral
def rayleigh_sommerfeld(x, y, z, wvl, xp, yp, c):
    xs = xp - x
    ys = yp - y
    xx, yy = jnp.meshgrid(xs, ys)
    kernel = c*rs_kernel(xx, yy, z, wvl)
    kernel_interp_r = interpolate.RectBivariateSpline(xs, ys, kernel.real)
    kernel_interp_i = interpolate.RectBivariateSpline(xs, ys, kernel.imag)
    bounds = [xs.min(), xs.max(), ys.min(), ys.max()]
    intg = kernel_interp_r.integral(*bounds) + 1j*kernel_interp_i.integral(*bounds)
    return intg

# Rayleigh-Sommerfeld diffraction PSF (integral for each pixel)
def rs_psf(x, y, z, wvl, xp, yp, c):
    psf = jnp.zeros((len(x), len(y), len(z), len(wvl)), dtype="complex")
    psf_f = psf.flatten()
    for n in tqdm(range(len(psf_f))):
        i = int(n/(len(y)*len(z)*len(wvl)))
        j = int((n - i*len(y)*len(z)*len(wvl))/(len(z)*len(wvl)))
        k = int((n - i*len(y)*len(z)*len(wvl) - j*len(z)*len(wvl))/len(wvl))
        l = n - i*len(y)*len(z)*len(wvl) - j*len(z)*len(wvl) - k*len(wvl)
        psf_f = psf_f.at[n].set(rayleigh_sommerfeld(x[i], y[j], z[k], wvl[l], xp, yp, c[:,:,l]))
    psf = jnp.reshape(psf_f, jnp.shape(psf))
    return psf

# Realistic dispersion from fill factor lx
def refract_index_fill(n0, lx, h, wvl):
    Afrac = lx**2
    phase = jnp.multiply.outer(Afrac, 2*jnp.pi*(1-n0)/wvl * h)
    return phase
    
# Plot results of design optmization
def plot_phase_psf_design(c_opt, psf_opt, lx_opt):
    # Plot PSFs at each wvl
    nwvl = psf_opt.shape[-1]
    fig, ax = plt.subplots(2, nwvl, dpi=150)
    for i in range(nwvl):
        ax[0, i].set_title(f"{wvl[i]}"+r"$\mu m$")
        ax[0, i].imshow(np.angle(c_opt[:, :, i]) % (-2 * np.pi))
        ax[1, i].imshow(psf_opt[:,:,i], vmin=np.min(psf_opt), vmax=np.max(psf_opt))
    fig.tight_layout()
    # Show design
    fig, ax = plt.subplots(1, dpi=150)
    p = 1
    for i in range(lx_opt.shape[0]):
        for j in range(lx_opt.shape[1]):
            lx = lx_opt[i, j] * p
            # Bottom-left corner of the square
            xc = (j + 1/2) * p - lx/2
            yc = (i + 1/2) * p - lx/2
            square = patches.Rectangle((xc, yc), lx, lx, edgecolor='black', facecolor='black')
            ax.add_patch(square)
    ax.set_xlim(0, lx_opt.shape[0] * p)
    ax.set_ylim(0, lx_opt.shape[1] * p)
    ax.set_aspect("equal")
    plt.show()


#%%
if __name__ == "__main__":
    # PARAMETERS
    N = 80
    R = 50 # Lens rad.
    F = 100 # Lens foc.
    z = [F]
    wvl = jnp.array([0.467, 0.532, 0.630]) # R=630nm, G=532nm, and B=467nm.

    #%%
    # CALC RS PSF USING INTEGRAL FOR EACH PIXEL
    lp = 50
    ls = 5
    xp = jnp.linspace(-lp, lp, N+1) # x-phase
    xs = jnp.linspace(-ls, ls, N+1) # x-sensor
    c = focus_phase(xp, xp, F, wvl)
    t0 = time.time()
    psf_rs = rs_psf(xs, xs, z, wvl, xp, xp, c)
    psf_rs = jnp.abs(psf_rs[:,:,0,:])**2
    t1 = time.time()
    print(f"Calc. time = {t1 - t0}s")

    #%%
    # CALC RS PSF USING FFT CONVOLUTION
    x = jnp.linspace(-50, 50, 10*N+1)
    c = focus_phase(x, x, F, wvl)
    idx = (jnp.argmin(jnp.abs(-ls - x)), jnp.argmin(jnp.abs(ls - x)))
    t0 = time.time()
    psf_rs_fftconv = psf_fftconv(x, x, z, wvl, c)
    psf_rs_fftconv = jnp.abs(psf_rs_fftconv[idx[0]-1:idx[1], idx[0]-1:idx[1], 0, :])**2
    t1 = time.time()
    print(f"Calc. time = {t1 - t0}s")

    #%%
    # COMPARE
    l=2
    fig, ax = plt.subplots(1,2)
    fig.suptitle("Rayleigh Sommerfeld Diffraction PSF")
    ax[0].imshow(jnp.array(psf_rs[:,:,l]), vmin=jnp.min(jnp.array(psf_rs)), vmax=jnp.max(jnp.array(psf_rs)))
    ax[0].title.set_text("Integral for each pixel")
    ax[1].imshow(jnp.array(psf_rs_fftconv[:,:,l]), vmin=jnp.min(jnp.array(psf_rs_fftconv)), vmax=jnp.max(jnp.array(psf_rs_fftconv)))
    ax[1].title.set_text("FFT Convolution")
    fig.tight_layout()

    plt.figure()
    plt.title("Convolution vs integral % err")
    plt.imshow((jnp.array(psf_rs[:,:,l]-psf_rs_fftconv[:,:,l])/jnp.array(psf_rs[:,:,l]))*100)
    plt.colorbar()
    
    #%%
    # TEST AUTOGRAD
    xp = jnp.linspace(-lp, lp, N) # x-phase
    c_targ = focus_phase(xp, xp, F, wvl)
    psf_targ = psf_fftconv(xp, xp, z, wvl, c_targ)[:, :, 0, :]
    psf_targ = jnp.abs(psf_targ)**2

    def objective(lx_var):
        lx_var = jnp.reshape(lx_var, (int(len(lx_var)**0.5), int(len(lx_var)**0.5)))
        n0 = 4
        h = 0.7
        phase_var = refract_index_fill(n0, lx_var, h, wvl) % (-2*jnp.pi)
        c_var = jnp.exp(1j*phase_var)
        psf_var = jnp.abs(psf_fftconv(xp, xp, z, wvl, c_var)[:,:,0,:])**2
        obj = jnp.sum(jnp.abs(psf_var - psf_targ))
        return obj
    
    objective_grad = jax.grad(objective)

    def nlopt_objective(lx, grad):
        if grad.size > 0:
            grad[:] = objective_grad(lx)
        obj = objective(lx).item()
        print(obj)
        return obj

    algorithm = nlopt.LD_MMA
    lb = jnp.zeros(N**2)
    ub = jnp.ones(N**2)
    maxtime = 20
    lx0 = 0.5 * jnp.ones(N**2)

    opt = nlopt.opt(algorithm, N**2)
    opt.set_min_objective(nlopt_objective)
    opt.set_lower_bounds(lb) # minimum side length
    opt.set_upper_bounds(ub) # maximum side length
    opt.set_maxtime(maxtime)
    xopt = opt.optimize(lx0)

    lx_opt = np.reshape(xopt, (N, N))
    phase = refract_index_fill(4, lx_opt, 0.7, wvl)%(-2*np.pi)
    c_opt = np.exp(1j*phase)
    psf_opt = np.abs(psf_fftconv(xp, xp, z, wvl, c_opt, pad=0)[:,:,0,:])**2

    plot_phase_psf_design(c_opt, psf_opt, lx_opt)

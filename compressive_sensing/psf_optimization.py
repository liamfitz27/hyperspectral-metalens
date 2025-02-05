#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import matplotlib; matplotlib.use('qtagg')
from scipy.optimize import minimize, Bounds
from psf_fft import psf_fftconv, focus_phase
from decoding_dict import DWT_dict_full, decode_dict_convolve
from image_decoding import compressive_sense_img, plot_results
import nlopt
from numpy import *
from tqdm import tqdm

# Realistic dispersion
def refract_index_fill(n0, lx, h, wvl):
    Afrac = lx**2
    phase = np.multiply.outer(Afrac, 2*np.pi*(1-n0)/wvl * h)
    return phase

# Mutual coherence for objective function
def mutual_coherence(phi):
    phi /= np.sum(phi, axis=1, keepdims=True)
    mc_mat = phi @ phi.T
    idx_diag = np.diag(np.ones(mc_mat.shape[0], dtype=bool))
    mc_mat[idx_diag] = -mc_mat[idx_diag] # Want high intensity
    unique_pairs = np.triu(mc_mat, k=1)
    return np.sum(unique_pairs)

# Metropolis algorithm step
def metropolis_step(x, x_lim, E, T, nflip):
        x_try = x.copy()
        indices = np.random.randint(0, x.shape[0], size=(nflip, 2))
        x_try[indices[:, 0], indices[:, 1]] = np.random.uniform(x_lim[0], x_lim[1], size=nflip)
        E_try = objective(x_try)
        accept_prob = np.min([1, np.exp(-1/T * (E_try - E))]) # If better, accept
        if np.random.rand() < accept_prob:
            return x_try, E_try
        else:
            return x, E
        
# Metropolis algorithm
def metropolis(x0, x_lim, E0, T, nflip, nsteps):
    fig, ax = plt.subplots(1)
    ax.set_title("Metropolis MCMC Optimization")
    plot = plt.plot([0], [E0])
    x = x0.copy()
    E = E0.copy()
    for i in range(1, nsteps+1):
        x, E = metropolis_step(x, x_lim, E, T, nflip)
        plot[0].set_xdata(np.append(plot[0].get_xdata(), i))
        plot[0].set_ydata(np.append(plot[0].get_ydata(), E))
        ax.set_ylabel("Mutual Coherence")
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw_idle()
        fig.canvas.start_event_loop(0.001)
    return x, E

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
    # INITIAL GUESS
    wvl = np.array([0.8, 1.0, 1.3])
    focl = 80
    sx = 30
    res = 1/0.6
    nx = int(2*sx*res)
    xx = np.linspace(-sx, sx, int(2*sx*res))
    c0 = np.zeros((nx, nx, len(wvl)), dtype="complex")
    s0 = 5
    for i in range(len(wvl)):
        s = s0*np.exp(1j*i*2*np.pi/3)
        cen = (int(s.real), int(s.imag))
        c0 += focus_phase(xx, xx, focl, [wvl[i]]*3, cen=cen)

    phase0 = np.angle(c0)%(-2*np.pi)
    plt.imshow(phase0[:,:,0])
    plt.colorbar()

    #%%
    # FIND REALISTIC SIDE LENGTHS FOR PHASE PROFILE
    n0 = 4 # aSi
    h = 0.7 # 700nm

    def objective(lx):
        phase = refract_index_fill(n0, lx, h, wvl)
        return np.sum(np.abs(phase - phase_targ)**2)

    lx0 = np.zeros((nx, nx))
    for i in tqdm(range(nx)):
        for j in range(nx):
            phase_targ = phase0[i,j,:]
            bounds = Bounds(0,1)
            lx0[i,j] = minimize(objective, 0.5, method='trust-constr', bounds=bounds).x

    phase0_r = refract_index_fill(n0, lx0, h, wvl)
    c0_r = np.exp(1j*phase0_r)

    #%%
    # PLOT PHASES, PSFs
    psf0 = psf_fftconv(xx, xx, [focl], wvl, c0_r, pad=20)
    psf0 = np.abs(psf0[:,:,0,:])**2

    plot_phase_psf_design(c0_r, psf0, lx0)

    #%%
    # OPTIMIZE PHASE USING METROPOLIS MCMC
    wavelet_dict = DWT_dict_full(nx, len(wvl), nx, type="haar")

    def objective(lx):
        n0 = 4
        h = 0.7
        img_shape = [78, 78, 3]
        phase = refract_index_fill(n0, lx, h, wvl)
        c = np.exp(1j*phase)
        psf_spect = psf_fftconv(xx, xx, [focl], wvl, c, pad=20)
        psf_spect = np.abs(psf_spect[:,:,0,:])**2
        decode_dict = decode_dict_convolve(psf_spect, wavelet_dict, img_shape)
        mc = mutual_coherence(decode_dict)
        return mc

    x0 = lx0
    E0 = objective(lx0)
    x_lim = [0, 1]
    T = 100
    nflip = 2
    nsteps = 1000
    lx_opt, E = metropolis(x0, x_lim, E0, T, nflip, nsteps)

    #%%
    # OPTIMIZE PHASE USING METHOD OF MOVING ASYMPTOTES (MMA)
    # Import functions for jax autograd
    from psf_fft_autograd import psf_fftconv, refract_index_fill
    from decoding_dict_autograd import mutual_coherence, decode_dict_fftconv
    from fftconv_autograd import fft_convolve2d
    import jax.numpy as jnp
    import jax

    wavelet_dict = DWT_dict_full(nx, len(wvl), nx, type="haar")

    def objective(lx):
        lx = jnp.reshape(lx, (int(len(lx)**0.5), int(len(lx)**0.5)))
        n0 = 4
        h = 0.7
        img_shape = [198, 198, 3]
        phase = refract_index_fill(n0, lx, h, wvl)
        c = jnp.exp(1j*phase)
        psf_spect = psf_fftconv(xx, xx, [focl], wvl, c, pad=0)
        psf_spect = jnp.abs(psf_spect[:,:,0,:])**2
        decode_dict = decode_dict_fftconv(psf_spect, wavelet_dict, img_shape)
        mc = mutual_coherence(decode_dict)
        return mc
    
    objective_grad = jax.grad(objective)

    def nlopt_objective(lx, grad):
        if grad.size > 0:
            grad[:] = objective_grad(lx)
        obj = objective(lx).item()
        print(obj, grad)
        return obj
    
    algorithm = nlopt.LD_MMA
    lb = jnp.zeros(nx**2)
    ub = jnp.ones(nx**2)
    maxtime = 3600
    lx0 = 0.5 * jnp.ones(nx**2)

    opt = nlopt.opt(algorithm, nx**2)
    opt.set_min_objective(nlopt_objective)
    opt.set_lower_bounds(lb) # minimum side length
    opt.set_upper_bounds(ub) # maximum side length
    opt.set_maxtime(maxtime)
    xopt = opt.optimize(lx0)

    #%%
    lx_opt = np.reshape(xopt, (nx, nx))
    phase = refract_index_fill(4, lx_opt, 0.7, wvl)%(-2*np.pi)
    c_opt = np.exp(1j*phase)
    psf_opt = np.abs(psf_fftconv(xx, xx, [focl], wvl, c_opt, pad=0)[:,:,0,:])**2

    plot_phase_psf_design(c_opt, psf_opt, lx_opt)

    #%%
    # PLOT PHASES, PSFs, DESIGN
    phase_opt = refract_index_fill(n0, lx_opt, h, wvl)
    c_opt = np.exp(1j*phase_opt)
    psf_opt = psf_fftconv(xx, xx, [focl], wvl, c_opt, pad=0)
    psf_opt = np.abs(psf_opt[:,:,0,:])**2

    plot_phase_psf_design(c_opt, psf_opt, lx_opt)
    
    #%%
    # SAVE DESIGN
    file = "mma_opt1.npz"
    np.savez(file, lx_opt=lx_opt, c_opt=c_opt, psf_opt=psf_opt)

    #%%
    # LOAD DESIGN
    file = "mma_opt1.npz"
    data = np.load(file)
    lx_opt = data["lx_opt"]
    c_opt = data["c_opt"]
    psf_opt = data["psf_opt"]

    #%%
    # COMPRESSIVE SENSING IMAGE DECODING
    from PIL import Image
    path = r"/Users/liam/Downloads/Basic_Color_Mixing_Chart.png" # Test image
    test_img = np.array(Image.open(path).resize((len(xx), len(xx))))[:,:,:-1]
    test_img[test_img[:,:,0]==129] = 0
    
    sparse_dict = DWT_dict_full(len(xx), 3, len(xx), type="haar")
    sensor_img, img_decode = compressive_sense_img(test_img, psf_opt, sparse_dict)

    plot_results(test_img, sensor_img, img_decode, psf_opt)
    plot_phase_psf_design(c_opt, psf_opt, lx_opt)

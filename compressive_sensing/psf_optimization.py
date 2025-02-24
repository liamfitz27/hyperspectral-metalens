#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from scipy.optimize import minimize, Bounds
from psf_fft import psf_fftconv, focus_phase
from decoding_dict import DWT_dict, DWT_dict_full, decode_dict_convolve
from image_decoding import compressive_sense_img, plot_results
import nlopt
from numpy import *
from tqdm import tqdm
import tracemalloc

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
def plot_phase_psf_design(c_opt, psf_opt, lx_opt, wvl):
    # Plot PSFs at each wvl
    nwvl = psf_opt.shape[-1]
    nrows = int(np.ceil(np.sqrt(nwvl)))
    ncols = int(np.ceil(nwvl / nrows))

    # Figure for phase plots
    fig1, ax1 = plt.subplots(nrows, ncols, dpi=150)
    fig1.suptitle("Phase Plots")
    for i in range(nwvl):
        row, col = divmod(i, ncols)
        ax1[row, col].set_title(f"{wvl[i]}"+r"$\mu m$")
        ax1[row, col].imshow(np.angle(c_opt[:, :, i]) % (-2 * np.pi))
    fig1.tight_layout()

    # Figure for PSF plots
    fig2, ax2 = plt.subplots(nrows, ncols, dpi=150)
    fig2.suptitle("PSF Plots")
    for i in range(nwvl):
        row, col = divmod(i, ncols)
        ax2[row, col].imshow(psf_opt[:,:,i], vmin=np.min(psf_opt), vmax=np.max(psf_opt))
    fig2.tight_layout()
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
    # wvl = np.array([0.8, 1.0, 1.3])
    f = np.linspace(1/1.2, 1/2.5, 9)
    nf = len(f)
    wvl = 1/f
    nf = len(wvl)
    focl = 50
    sx = 20
    res = 1
    nx = int(2*sx*res)
    xx = np.linspace(-sx, sx, int(2*sx*res))
    c0 = np.zeros((nx, nx, nf), dtype="complex")
    s0 = 10
    for i in range(nf):
        s = s0*np.exp(1j*i*2*np.pi/nf)
        cen = (int(s.real), int(s.imag))
        c0 += focus_phase(xx, xx, focl, [wvl[i]]*nf, cen=cen)

    phase0 = np.angle(c0)%(-2*np.pi)
    plt.imshow(phase0[:,:,0])
    plt.colorbar()

    #%%
    # FIND REALISTIC SIDE LENGTHS FOR PHASE PROFILE
    n0 = 4 # aSi
    h = 0.7 # 700nm

    def objective(lx):
        phase = refract_index_fill(n0, lx, h, wvl)
        return np.sum(np.abs(phase - phase_targ)**2*wvl**2)

    lx0 = np.zeros((nx, nx))
    bounds = Bounds(0,1)
    for i in tqdm(range(nx)):
        for j in range(nx):
            phase_targ = phase0[i,j,:]
            lx0[i,j] = minimize(objective, 0.5, method='trust-constr', bounds=bounds).x

    phase0_r = refract_index_fill(n0, lx0, h, wvl)
    c0_r = np.exp(1j*phase0_r)

    #%%
    # INITAL PLOT PHASES, PSFs, DESIGN
    psf0 = psf_fftconv(xx, xx, [focl], wvl, c0_r, pad=20)
    psf0 = np.abs(psf0[:,:,0,:])**2

    plot_phase_psf_design(c0_r, psf0, lx0)

    #%%
    # OPTIMIZE PHASE USING METHOD OF MOVING ASYMPTOTES (MMA)
    # Import functions for jax autograd
    from psf_fft_autograd import psf_fftconv, refract_index_fill, fft_convolve2d, rs_kernel
    from decoding_dict_autograd import mutual_coherence, decode_dict_fftconv
    from fftconv_autograd import fft_convolve2d
    import jax.numpy as jnp
    import jax
    import time

    h = 0.7
    n0 = 4
    wavelet_dict = jnp.array(DWT_dict_full(nx, nf, nx, type="haar"))
    # wavelet_dict = jnp.array(DWT_dict_full(nx//2, nf, nx//2+2, type="haar"))

    def mutual_coherence(phi):
        phi_norm = jnp.sum(phi, axis=0, keepdims=True)
        phi_norm = jnp.where(phi_norm == 0, 0, phi / phi_norm)
        mc_mat = phi_norm.T @ phi_norm
        unique_pairs = jnp.triu(mc_mat, k=1)
        sum = jnp.sum(unique_pairs)
        return sum
    
    mutual_coherence = jax.jit(mutual_coherence)
    
    def mutual_coherence2(phi):
        phi_norm = jnp.sum(phi, axis=1, keepdims=True)
        phi_norm = jnp.where(phi_norm == 0, 0, phi / phi_norm)
        term1 = jnp.sum(jnp.sum(phi_norm, axis=1)**2)
        term2 = jnp.sum(phi_norm**2)
        return term1 - term2
    
    mutual_coherence2 = jax.jit(mutual_coherence2)
    
    def objective(lx):
        lx = jnp.reshape(lx, (int(len(lx)**0.5), int(len(lx)**0.5)))
        # Set these parameters
        n0 = 4
        h = 0.7
        alpha = 0
        img_shape = [2*nx-2, 2*nx-2, nf]
        phase = refract_index_fill(n0, lx, h, wvl)
        c = jnp.exp(1j*phase)
        t0 = time.time()
        psf_spect = psf_fftconv(xx, xx, [focl], wvl, c, pad=nx//2)
        t1 = time.time()
        print("     psf_fftconv t =", t1-t0)
        psf_spect = jnp.abs(psf_spect[:,:,0,:])**2
        decode_dict = decode_dict_fftconv(psf_spect, wavelet_dict, img_shape)
        t2 = time.time()
        print("     decode_dict_fftconv t =", t2-t1)
        # mc = mutual_coherence(decode_dict)
        mc = mutual_coherence2(decode_dict)
        t3 = time.time()
        print("     mutual_coherence t =", t3-t2)
        return mc
    
    jax_grad = jax.grad(objective)

    def analytic_grad(lx):
        h = 0.7
        n0 = 4
        img_shape = [nx, nx, nf]
        lx = jnp.reshape(lx, (int(len(lx)**0.5), int(len(lx)**0.5)))
        phase = refract_index_fill(n0, lx, h, wvl)
        c = jnp.exp(1j*phase)
        psf_spect = psf_fftconv(xx, xx, [focl], wvl, c, pad=20)
        U_mat = psf_spect[:,:,0,:]
        PSF_mat = jnp.abs(psf_spect[:,:,0,:])**2
        D_mat = decode_dict_fftconv(PSF_mat, wavelet_dict, img_shape)
        A_mat = jnp.sqrt(jnp.sum(D_mat**2, axis=0, keepdims=True))
        A_mat = jnp.where(A_mat == 0, 0, D_mat / A_mat)
        C_mat = D_mat.T @ D_mat
        F_mat = jnp.sum(A_mat, axis=1, keepdims=True) - A_mat
        C_dd = jnp.diag(C_mat)[jnp.newaxis, :]
        term1 = jnp.divide(F_mat, C_dd)
        FtD = F_mat.T @ D_mat
        FtD_dd = jnp.diag(FtD)[jnp.newaxis, :]
        term2 = D_mat / C_dd**1.5 * FtD_dd
        dobj_dD = 2*(term1 - term2)
        dobj_dD_unpack = dobj_dD.reshape((nx,nx,-1))

        # h_mat = jnp.array([rs_kernel(*jnp.meshgrid(xx, xx), focl, w) for w in wvl]).reshape((nx, nx, nf))
        # h_mat_offcenter_ij = jnp.roll(jnp.roll(h_mat, i, axis=0), j, axis=1)
        # # Reshape and align dimensions for broadcasting
        # h = h[np.newaxis, np.newaxis, :, :, :]  # Shape: (1, 1, a_prime-i, b_prime-j, k)
        # U_mat = U_mat[:, :, np.newaxis, np.newaxis, :]  # Shape: (a_prime, b_prime, 1, 1, k)
        # theta_mat = jnp.multiply.outer(-lx**2 * 2*np.pi * (1-n0)*h, 1/wvl)
        # exp_theta = np.exp(theta_mat)[np.newaxis, np.newaxis, :, :, :]  # Shape: (1, 1, i, j, k)
        # # Compute Q using broadcasting
        # Q = 1j * h_mat + h_mat_offcenter_ij * U_mat * exp_theta  # Shape: (a_prime, b_prime, i, j, k)
        # conv

        # dD_dtheta 

        h_mat = jnp.array([rs_kernel(*jnp.meshgrid(xx, xx), focl, w) for w in wvl]).reshape((nx, nx, nf))
        Psi_mat = wavelet_dict
        Psi_mat_unpack = jnp.reshape(Psi_mat, (nx, nx, nf, -1))
        theta_mat = jnp.multiply.outer(-lx**2 * 2*np.pi * (1-n0)*h, 1/wvl)
        J_mat = jax.vmap(
            lambda k: jax.vmap(
            lambda ijk: fft_convolve2d(
                U_mat[:,:,k] * fft_convolve2d(dobj_dD_unpack[:,:,ijk], Psi_mat_unpack[:, :, k, ijk], corr=True), 
                h_mat[:, :, k], 
                corr=True
            ) * (1j * jnp.exp(1j * theta_mat[:, :, k]))
            )(jnp.arange(Psi_mat_unpack.shape[-1]))
        )(jnp.arange(nf))
        grad = -8*np.pi * lx * (1-n0)*h * jnp.sum((1/wvl)[:, None, None] * jnp.sum(jnp.real(J_mat), axis=1), axis=0)
        return grad.flatten()
    
    analytic_grad = jax.jit(analytic_grad)


    def nlopt_objective(lx, grad):
        if grad.size > 0:
            grad[:] = jax_grad(lx)
            #grad[:] = analytic_grad(lx)
        obj = objective(lx).item()
        # fig, ax = plt.subplots(1, 2)
        # ax[0].imshow(grad.reshape(nx,nx))
        # anal_grad = analytic_grad(lx)
        # ax[1].imshow(anal_grad.reshape(nx,nx))
        print("Objective func. =", obj)
        return obj
    
    algorithm = nlopt.LD_MMA
    lb = jnp.zeros(nx**2)
    ub = jnp.ones(nx**2)
    maxtime = 60 * 60
    # lx0 = 0.5 * jnp.ones(nx**2)

    opt = nlopt.opt(algorithm, nx**2)
    opt.set_min_objective(nlopt_objective)
    opt.set_lower_bounds(lb) # minimum side length
    opt.set_upper_bounds(ub) # maximum side length
    opt.set_maxtime(maxtime)
    # lx0 = np.random.rand(nx,nx)
    xopt = opt.optimize(lx0.flatten())
    lx_opt = np.reshape(xopt, (nx, nx))

    #%%
    # PLOT PHASES, PSFs, DESIGN
    phase_opt = refract_index_fill(n0, lx_opt, h, wvl)%(-2*np.pi)
    c_opt = np.exp(1j*phase_opt)
    psf_opt = np.abs(psf_fftconv(xx, xx, [focl], wvl, c_opt, pad=nx//2)[:,:,0,:])**2

    plot_phase_psf_design(c_opt, psf_opt, lx_opt)

    #%%
    # OPTIMIZE PHASE USING METROPOLIS MCMC
    wavelet_dict = DWT_dict_full(nx, nf, nx, type="haar")

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
    # PLOT PHASES, PSFs, DESIGN
    phase_opt = refract_index_fill(n0, lx_opt, h, wvl)
    c_opt = np.exp(1j*phase_opt)
    psf_opt = psf_fftconv(xx, xx, [focl], wvl, c_opt, pad=0)
    psf_opt = np.abs(psf_opt[:,:,0,:])**2

    plot_phase_psf_design(c_opt, psf_opt, lx_opt)
    
    #%%
    # SAVE DESIGN
    file = "mma_opt2.npz"
    np.savez(file, lx_opt=lx_opt, c_opt=c_opt, psf_opt=psf_opt)

    #%%
    # LOAD DESIGN
    file = "mma_opt2.npz"
    data = np.load(file)
    lx_opt = data["lx_opt"]
    c_opt = data["c_opt"]
    psf_opt = data["psf_opt"]

    #%%
    # COMPRESSIVE SENSING IMAGE DECODING
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.decomposition import SparseCoder
    from decoding_dict import decode_dict_convolve
    from image_model import phi_matrix, phi_matrix_full
    from image_model import psf_spect_convolve, phi_matrix_full

    def compressive_sense_img(actual_img, psf_spect, sparse_dict):
        # Calc. measurement matrix phi (~PSF convolution)
        phi = phi_matrix_full(psf_spect, np.shape(psf_spect))
        pad = (psf_spect.shape[0] - 2)//2 # same size
        actual_img_pad = np.pad(actual_img, ((pad,pad), (pad,pad), (0,0)))
        actual_img_f = actual_img_pad.flatten()
        # Calc. sensor image and decoding dictionary (basis elements convolved with PSFs)
        sensor_img = phi @ actual_img_f
        sensor_img += sensor_img.max()*np.random.rand(*sensor_img.shape) * 0.01# Add noise amount 0.05
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
        sensor_img = np.reshape(sensor_img, np.shape(actual_img_pad)[:-1])
        img_decode = np.reshape(img_decode, np.shape(actual_img_pad))
        return sensor_img, img_decode


    from PIL import Image
    path = r"/Users/liam/Downloads/Basic_Color_Mixing_Chart.png" # Test image
    test_img = np.array(Image.open(path).resize((len(xx), len(xx))))[:,:,:-1]
    test_img[test_img[:,:,0]==129] = 0

    frac = (nx)//5
    overlap = 0
    test_img = np.zeros((nx, nx, nf))
    test_img[1*frac-overlap:2*frac+overlap,1*frac-overlap:2*frac+overlap,0] = 1
    test_img[1*frac-overlap:2*frac+overlap,2*frac-overlap:3*frac+overlap,1] = 1
    test_img[1*frac-overlap:2*frac+overlap,3*frac-overlap:4*frac+overlap,2] = 1
    test_img[2*frac-overlap:3*frac+overlap,1*frac-overlap:2*frac+overlap,3] = 1
    test_img[2*frac-overlap:3*frac+overlap,2*frac-overlap:3*frac+overlap,4] = 1
    test_img[2*frac-overlap:3*frac+overlap,3*frac-overlap:4*frac+overlap,5] = 1
    test_img[3*frac-overlap:4*frac+overlap,1*frac-overlap:2*frac+overlap,6] = 1
    test_img[3*frac-overlap:4*frac+overlap,2*frac-overlap:3*frac+overlap,7] = 1
    test_img[3*frac-overlap:4*frac+overlap,3*frac-overlap:4*frac+overlap,8] = 1
    
    sparse_dict = DWT_dict_full(nx, nf, nx, type="haar")
    # sparse_dict = wavelet_dict
    sensor_img, img_decode = compressive_sense_img(test_img, psf_opt, sparse_dict)

    plot_results(test_img, sensor_img, img_decode, psf_opt)
    plot_phase_psf_design(c_opt, psf_opt, lx_opt)

    #%%
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        ax.imshow(img_decode[:, :, i])
        ax.set_title(f"Channel {i+1}") 
        ax.axis('off')
    plt.tight_layout()
    plt.show()
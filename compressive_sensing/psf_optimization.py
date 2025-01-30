#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('Qt5Agg')
from scipy.optimize import minimize, Bounds
from psf_model import rs_psf_fftconv, focus_phase
from decoding_dict import DWT_dict_full, decode_dict_convolve
import nlopt
from numpy import *
from tqdm import tqdm

# Realistic dispersion
def refract_index_fill(n0, lx, h, wvl):
    Afrac = lx**2
    phase = np.multiply.outer(Afrac, 2*np.pi*(1-n0)/wvl * h)
    return phase

def mutual_coherence(phi):
    mc = phi @ phi.T
    return np.max(mc)

#%%
# INITIAL GUESS
wvl = np.array([0.8, 1.0, 1.3])
focl = 80
sx = 20
res = 1
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
# PLOT REALIZED PASE & DESIGN
import matplotlib.patches as patches

fig, ax = plt.subplots(1, 2, dpi=150)
p = 0.4
for i in range(lx0.shape[0]):
    for j in range(lx0.shape[1]):
        lx = lx0[i, j] * p
        # Bottom-left corner of the square
        xc = (j + 1/2) * p - lx/2
        yc = (i + 1/2) * p - lx/2
        square = patches.Rectangle((xc, yc), lx, lx, edgecolor='black', facecolor='black')
        ax[0].add_patch(square)

ax[0].set_xlim(0, lx0.shape[1] * p)
ax[0].set_ylim(0, lx0.shape[0] * p)
ax[0].set_aspect("equal")
ax[1].imshow(phase0_r[:,:,0])

#%%
# PLOT PHASES, PSFs
psf0 = rs_psf_fftconv(xx, xx, [focl], wvl, c0_r, pad=20)
psf0 = np.abs(psf0[:,:,0,:])**2

fig, ax = plt.subplots(3, 3, figsize=(12, 8))
for i in range(3):
    ax[0, i].imshow(np.angle(c0[:, :, i]) % (-2 * np.pi))
    ax[1, i].imshow(np.angle(c0[:, :, i]) % (-2 * np.pi))
    ax[2, i].imshow(psf0[:,:,i], vmin=np.min(psf0), vmax=np.max(psf0))
fig.tight_layout()

#%%
# OPTIMIZE PHASE USING METROPOLIS MCMC
wavelet_dict = DWT_dict_full(nx, len(wvl), nx, type="haar")

def objective(x):
    n0 = 4
    h = 0.7
    img_shape = [78, 78, 3]
    phase = refract_index_fill(n0, x, h, wvl)
    c = np.exp(1j*phase)
    psf_spect = rs_psf_fftconv(xx, xx, [focl], wvl, c, pad=20)
    psf_spect = np.abs(psf_spect[:,:,0,:])**2
    decode_dict = decode_dict_convolve(psf_spect, wavelet_dict, img_shape)
    mc = mutual_coherence(decode_dict)
    return mc

def metropolis_step(x, x_lim, E, T, nflip):
    x_try = x.copy()
    indices = np.random.randint(0, x.shape[0], size=(nflip, 2))
    x_try[indices[:, 0], indices[:, 1]] = np.random.uniform(x_lim[0], x_lim[1], size=nflip)
    E_try = objective(x_try)
    accept_prob = np.min([1, np.exp(-1/T * (E_try - E))])
    if np.random.rand() < accept_prob:
        return x_try, E_try
    else:
        return x, E
    
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

x0 = lx0
E0 = objective(lx0)
x_lim = [0, 1]
T = 0.1
nflip = 2
nsteps = 1000
lx_opt, E = metropolis(x0, x_lim, E0, T, nflip, nsteps)

#%%
# PLOT PHASES, PSFs, DESIGN
phase_opt = refract_index_fill(n0, lx_opt, h, wvl)
c_opt = np.exp(1j*phase_opt)
psf_opt = rs_psf_fftconv(xx, xx, [focl], wvl, c_opt, pad=20)
psf_opt = np.abs(psf_opt[:,:,0,:])**2

fig, ax = plt.subplots(2, 3, figsize=(12, 8))
for i in range(3):
    ax[0, i].imshow(np.angle(c_opt[:, :, i]) % (-2 * np.pi))
    ax[1, i].imshow(psf_opt[:,:,i], vmin=np.min(psf_opt), vmax=np.max(psf_opt))
fig.tight_layout()

fig, ax = plt.subplots(1)
p = 0.4
for i in range(lx_opt.shape[0]):
    for j in range(lx_opt.shape[1]):
        lx = lx_opt[i, j] * p
        # Bottom-left corner of the square
        xc = (j + 1/2) * p - lx/2
        yc = (i + 1/2) * p - lx/2
        square = patches.Rectangle((xc, yc), lx, lx, edgecolor='black', facecolor='black')
        ax.add_patch(square)

ax.set_xlim(0, lx_opt.shape[1] * p)
ax.set_ylim(0, lx_opt.shape[0] * p)
ax.set_aspect("equal")

#%% 
# OPTIMIZE PHASE
wavelet_dict = DWT_dict_full(nx, len(wvl), nx, type="haar")

def objective(lx, grad):
    lx = np.reshape(lx, (int(len(lx)**0.5), int(len(lx)**0.5)))
    if grad.size > 0:
        grad[:] = 0
    n0 = 4
    h = 0.7
    img_shape = [78, 78, 3]
    dphi = refract_index_fill(n0, lx, h, wvl)
    c = np.exp(1j*dphi)
    psf_spect = rs_psf_fftconv(xx, xx, [focl], wvl, c, pad=20)
    psf_spect = np.abs(psf_spect[:,:,0,:])**2
    decode_dict = decode_dict_convolve(psf_spect, wavelet_dict, img_shape)
    mc = mutual_coherence(decode_dict)
    print(mc)
    return mc

algorithm = nlopt.GN_CRS2_LM
n = nx**2
opt = nlopt.opt(algorithm, n)

opt.set_min_objective(objective)

lb = np.zeros(n)
ub = np.ones(n)
opt.set_lower_bounds(lb) # minimum side length
opt.set_upper_bounds(ub) # maximum side length

maxtime = 120
opt.set_maxtime(maxtime)

xopt = opt.optimize(np.ndarray.flatten(lx0))

#%%

lx_opt2 = np.reshape(xopt, (40, 40))
dphi_opt2 = refract_index_fill(n0, lx_opt2, h, wvl)

fig, ax = plt.subplots(2, 2, dpi=150)
p = 0.4
for i in range(lx_opt.shape[0]):
    for j in range(lx_opt.shape[1]):
        lx = lx_opt[i, j] * p
        # Bottom-left corner of the square
        x = (j + 1/2) * p - lx/2
        y = (i + 1/2) * p - lx/2
        square = patches.Rectangle((x, y), lx, lx, edgecolor='black', facecolor='black')
        ax[0,0].add_patch(square)

        lx = lx_opt2[i, j] * p
        # Bottom-left corner of the square
        x = (j + 1/2) * p - lx/2
        y = (i + 1/2) * p - lx/2
        square = patches.Rectangle((x, y), lx, lx, edgecolor='black', facecolor='black')
        ax[1,0].add_patch(square)

ax[0,0].set_xlim(0, lx_opt2.shape[1] * p)
ax[0,0].set_ylim(0, lx_opt2.shape[0] * p)
ax[0,0].set_aspect("equal")
ax[0,1].imshow(dphi_opt[:,:,0])

ax[1,0].set_xlim(0, lx_opt2.shape[1] * p)
ax[1,0].set_ylim(0, lx_opt2.shape[0] * p)
ax[1,0].set_aspect("equal")
ax[1,1].imshow(dphi_opt2[:,:,0])


    
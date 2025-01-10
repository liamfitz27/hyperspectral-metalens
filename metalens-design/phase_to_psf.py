#%%
import autograd.numpy as np
from autograd import grad, elementwise_grad
from autograd.extend import primitive
import torch.autograd as ad
import torch
from torch.autograd import Variable
from autograd.numpy import fft
import matplotlib.pyplot as plt
import zernike
import autograd.scipy as scipy
from tqdm import tqdm

def pt_src(x, y, wvl, Np, Z, cen=(0,0)):
    k = 2*np.pi / wvl
    r = np.sqrt((x-cen[0])**2 + (y-cen[1])**2 + Z**2)
    return Np*np.cos(k*r) / r

def focus_phase(x, y, focl, wvl_cen):
    phi_foc = 2*np.pi/wvl_cen * (focl - np.sqrt(focl**2 + x**2 + y**2))
    # ang = np.angle(x+1j*y)
    # phi_foc += 2*ang
    return phi_foc

def circ_basis(x, y, n, m):
    r = np.sqrt(x**2 + y**2)
    ang = np.angle(x + 1j*y)
    e1 = np.cos(n*ang)
    e2 = scipy.special.jn(m, r)
    return e1*e2

def sq_basis(x, y, n, m):
    e1 = np.cos(n*np.pi*x/(x.max()-x.min()))
    e2 = np.cos(m*np.pi*y/(y.max()-y.min()))
    return e1*e2

def perturb_phase(x, y, var):
    f = np.zeros(np.shape(x))
    nterms = len(var[:-1])
    for i in range(nterms):
        n = i // int(np.sqrt(nterms))
        m = i % int(np.sqrt(nterms))
        # f += var[i]*circ_basis(x, y, n, m)
        f += var[i]*sq_basis(x, y, n, m)
    f = (f + np.rot90(f.T, 3) + np.rot90(f, 2) + np.rot90(f.T, 1))/4
    return f

def zernike_pol(x, y, var):
    cart = zernike.RZern(10)
    cart.make_cart_grid(x, y)
    c = np.zeros(cart.nk)
    for i in range(cart.nk):
        try:
            c[i] = var[i]
        except:
            c[i] = var._value[i]
    phi = cart.eval_grid(c, matrix=True)
    return phi

def perturb_phase(x, y, var):
    @primitive
    def fun(v):
        return zernike_pol(x, y, v)
    phi_var = fun(var)
    # phi_var = (phi_var + np.rot90(phi_var, 2))/2
    phi_var = (phi_var + np.rot90(phi_var.T, 3) + np.rot90(phi_var, 2) + np.rot90(phi_var.T, 1))/4
    # phi_var = (phi_var + np.rot90(phi_var.T, 2) + phi_var.T + np.rot90(phi_var, 2))/4
    return phi_var

def total_phase(x, y, var, focl, wvl_cen):
    # phi_tot = focus_phase(x, y, focl, wvl_cen) + perturb_phase(x, y, var)
    # phi_tot = perturb_phase(x, y, var)
    phi_tot = focus_phase(x, y, focl, wvl_cen)
    # ang = np.angle(x+1j*y)
    # phi_tot += (2*ang+var[-1])
    phi_tot %= -2*np.pi
    return phi_tot

def spiral_phase(x, y, wvl_min, wvl_max, N=2):
    theta = np.angle(x + 1j*y) + np.pi
    wvl = wvl_min + (wvl_max - wvl_min)*2/(2*np.pi)*theta

def pupil(x, y, var, focl, wvl_cen, R):
    r = np.sqrt(x**2 + y**2)
    phase = total_phase(x, y, var, focl, wvl_cen)
    return np.exp(1j*phase)*(r <= R)

# def library_pupil(x, y, var, focl, wvl_cen, R, lib_path):
#     r = np.sqrt(x**2 + y**2)
#     phase_opt = total_phase(x, y, var, focl, wvl_cen)


def psf(x, y, x_cen, y_cen, wvl, var, focl, wvl_cen, R, Z, Np):
    E_ppl = pt_src(x - x_cen, y - y_cen, wvl, Np, Z)
    ppl =  pupil(x, y, var, focl, wvl_cen, R)
    try:
        intens = np.abs(fft.fftshift(fft.fft2(E_ppl*ppl)))**2
    except:
        intens = np.abs(fft.fftshift(fft.fft2(E_ppl._value*ppl)))**2
    return intens

def crlb(theta, Nx, var):
    nt = len(theta)
    idx_cen = (np.argmin(np.abs(x_cen-x)), np.argmin(np.abs(y_cen-y)))
    x_psf = x[idx_cen[0]-Nx:idx_cen[0]+Nx+1]
    y_psf = y[idx_cen[1]-Nx:idx_cen[1]+Nx+1]
    xxp, yyp = np.meshgrid(x_psf, y_psf)
    mu_theta = psf(xxp, yyp, x_cen, y_cen, *theta, var, focl, wvl_cen, R, Z, Np)
    fisher_diag = np.empty(np.shape(theta))
    def theta_fun(wvl):
        return psf(xxp, yyp, x_cen, y_cen, wvl, var, focl, wvl_cen, R, Z, Np)
    def dtheta_fun_i(x_cen, y_cen, wvl, i):
        dtheta_i = elementwise_grad(theta_fun, i)(x_cen, y_cen, wvl)
        return dtheta_i
    for it in range(nt):
        smnd = 1/(mu_theta)*elementwise_grad(theta_fun)(wvl)**2
        try:
            fisher_diag[it] = np.sum(smnd)
        except:
            fisher_diag[it] = np.sum(smnd._value)
    crlb = 1/(fisher_diag)
    return crlb, mu_theta

def loss_fun(var):
    L = 0
    for i in range(len(x_range)):
        for j in range(len(y_range)):
            for k in range(len(wvl_range)):
                theta = [wvl_range[k]]
                c, p = crlb(theta, Nx, var)
                L += np.sqrt(c)
                # L += 1E-24*np.sum((p/np.sum(p))**0.5)
    return np.sum(L)

def psf_ang_dist(x, y, wvl, wvl_min, wvl_max):
    xxf, yyf = (np.ndarray.flatten(x), np.ndarray.flatten(y))
    ang = 0.5*np.pi*(wvl - wvl_min)/(wvl_max - wvl_min)
    dist = 0.1*sx*np.abs((wvl - (wvl_min + wvl_max)/2)/(wvl_max - wvl_min))+0.01*sx
    cen1 = (dist/2*np.cos(ang), dist/2*np.sin(ang))
    cen2 = (-dist/2*np.cos(ang), -dist/2*np.sin(ang))
    idx1 = (np.argmin(np.abs(x-cen1[0])), np.argmin(np.abs(y-cen1[1])))
    idx2 = (np.argmin(np.abs(x-cen2[0])), np.argmin(np.abs(y-cen2[1])))
    x1_shift = np.reshape(xxf - xxf[idx1[0]], np.shape(x))
    y1_shift = np.reshape(yyf - yyf[idx1[1]], np.shape(y))
    x2_shift = np.reshape(xxf - xxf[idx2[0]], np.shape(x))
    y2_shift = np.reshape(yyf - yyf[idx2[1]], np.shape(y))
    I1 = np.exp(-0.5*((x1_shift)**2+(y1_shift)**2)/(0.0035*sx)**2)
    I2 = np.exp(-0.5*((x2_shift)**2+(y2_shift)**2)/(0.0035*sx)**2)
    return I1 + I2

def loss_fun(var):
    L = 0
    idx_cen = (np.argmin(np.abs(x)), np.argmin(np.abs(y)))
    x_psf = x[idx_cen[0]-Nx:idx_cen[0]+Nx+1]
    y_psf = y[idx_cen[1]-Nx:idx_cen[1]+Nx+1]
    xxp, yyp = np.meshgrid(x_psf, y_psf)
    for i in range(len(wvl_range)):
        psf_target = psf_ang_dist(xxp, yyp, wvl_range[i], wvl_range.min(), wvl_range.max())
        psf_sim = psf(xxp, yyp, 0, 0, wvl_range[i], var, focl, wvl_cen, R, Z, Np)
        L += np.sum((psf_target - psf_sim/np.max(psf_sim))**2)
    return L

def resid_fun(var):
    idx_cen = (np.argmin(np.abs(x)), np.argmin(np.abs(y)))
    x_psf = x[idx_cen[0]-Nx:idx_cen[0]+Nx+1]
    y_psf = y[idx_cen[1]-Nx:idx_cen[1]+Nx+1]
    xxp, yyp = np.meshgrid(x_psf, y_psf)
    resid = np.zeros((*np.shape(xxp), len(wvl_range)))
    for i in range(len(wvl_range)):
        psf_target = psf_ang_dist(xxp, yyp, wvl_range[i], wvl_range.min(), wvl_range.max())
        psf_sim = psf(xxp, yyp, 0, 0, wvl_range[i], var, focl, wvl_cen, R, Z, Np)
        try:
            resid[:,:,i] = psf_sim - psf_target
        except:
            resid[:,:,i] = (psf_sim - psf_target)._value
    return np.ndarray.flatten(resid)

def adam(loss_fun, x, callback=None, num_iters=100,
         step_size=0.001, b1=0.9, b2=0.999, eps=10**-8):
    """Adam as described in http://arxiv.org/pdf/1412.6980.pdf.
    It's basically RMSprop with momentum and some correction terms."""
    m = np.zeros(len(x))
    v = np.zeros(len(x))
    for i in range(num_iters):
        g = grad(loss_fun)(x)
        if callback: callback(x, i, g)
        m = (1 - b1) * g      + b1 * m  # First  moment estimate.
        v = (1 - b2) * (g**2) + b2 * v  # Second moment estimate.
        mhat = m / (1 - b1**(i + 1))    # Bias correction.
        vhat = v / (1 - b2**(i + 1))
        x = x - step_size*mhat/(np.sqrt(vhat) + eps)
    return x

# INIT PARAMETERS
if __name__ == "__main__":
    NM = 1E-9
    UM = 1E-6

    wvl_cen = 1000*NM
    wvl = np.arange(800*NM, 1800.1*NM, 50*NM)
    focl = 500*UM
    Z = focl
    R = 50*UM
    Np = 15

    sx = 50*UM
    res = 2/UM
    x = np.arange(0, sx + 1/res, 1/res)
    x = np.append(-np.flip(x), x[1:])
    y = x
    xx, yy = np.meshgrid(x, x)
    xxx, yyy, lll = np.meshgrid(x, x, wvl)
    Nx = 15
    x_range = np.arange(0, sx/4 + 1/res, 4/res)
    y_range = x_range
    wvl_range = wvl

    x_cen = 0
    y_cen = 0

    psf_foc = psf(xx, yy, 0, 0, wvl_cen, [0], focl, wvl_cen, R, Z, Np)
    plt.imshow(psf_foc)

    #%%
    

    # self.sim.plot2D(
    #     output_plane=mp.Volume(
    #         center=mp.Vector3(-0.5*self.sx+self.dpml+self.dsub+0.5*self.gh,0,0),
    #         size=mp.Vector3(0,self.sy,self.sz)
    #     )
    # )
    # assert 1==0

    plt.figure()
    plt.title("Optim. phase")
    plt.imshow(np.angle(c_opt)+np.pi)
    plt.colorbar()
    plt.figure()
    plt.title("Optim. tran")
    plt.imshow(np.abs(c_opt))
    plt.colorbar()
    plt.figure()
    plt.title("Targ. phase")
    plt.imshow(phase_targ)
    assert 1==0


    def find_lib_min(phase_targ, lib_path):
        with open(lib_path, "r") as jsonFile:
            lib = json.load(jsonFile)
        types = ["block", "hollow_block", "cross", "interior_block", "interior_cross", "window"]
        geoms = []
        for t in types:
            geoms.append(lib[t])
        for i in range(len(np.ndarray.flatten(phase_targ))):
                tran = lib[t][1]
                phase = lib[t][2]
                idx = np.argmin(score(tran, phase, phase_targ))


    plt.plot(np.ones())


    #%%
    # from PIL import Image
    # img = Image.open('mcgill_logo2.jpg').convert('L')
    # img = img.resize((201,201), Image.Resampling.LANCZOS)
    # img = np.array(img)
    # img = np.max(img)/np.array(img)
    # img -= np.min(img)
    # # plt.imshow(img)
    # # plt.colorbar()

    # E_ppl = np.zeros(np.shape(xx))
    # for i in range(np.shape(xx)[0]):
    #     for j in range(np.shape(xx)[0]):
    #         E_ppl += pt_src(xx, yy, wvl_cen, img[i,j], Z, cen=(xx[i,j], yy[i,j]))

    # ppl = ((xx**2+yy**2) <= R**2)
    # plt.figure(dpi=200)
    # plt.imshow(E_ppl*ppl)

    # intens = np.abs(fft.fftshift(fft.fft2(E_ppl*ppl)))**2
    # plt.imshow(intens)

    var = np.zeros(20)
    # var[11] = 0.000000001 #13
    # var[13] = 1

    E_ppl = pt_src(xxx-x_cen, yyy-y_cen, lll, 1, Z)

    I = np.empty(np.shape(xxx))
    for i in range(len(wvl)):
        I[:,:,i] = psf(xx, yy, 0, 0, wvl[i], var, focl, wvl_cen, R, Z, Np)

    plt.figure(dpi=200)
    plt.xticks([])
    plt.yticks([])
    plt.gca().set(frame_on=False)
    for i in range(1,8):
        wvl_idx = 2*(i-1)
        plt.subplot(3,3,i)
        plt.title(r"$\lambda=$"+f"{round(wvl[wvl_idx]/NM)}nm", fontsize=5)
        plt.pcolormesh(xx[len(x)//2-Nx:len(x)//2+Nx+1,len(x)//2-Nx:len(x)//2+Nx+1], yy[len(x)//2-Nx:len(x)//2+Nx+1,len(x)//2-Nx:len(x)//2+Nx+1], I[len(x)//2-Nx:len(x)//2+Nx+1,len(x)//2-Nx:len(x)//2+Nx+1,wvl_idx], vmax=np.max(I))
        plt.xticks([])
        plt.yticks([])
    # plt.savefig("psf_spect_focus.png", bbox_inches="tight")

    plt.figure(dpi=150)
    plt.imshow(focus_phase(xx, yy, focl, wvl_cen)%(-2*np.pi))
    # plt.savefig("phase_prof_focus.png", bbox_inches="tight")

    # wvl_plot = 1000*nm
    wvl_plot = wvl_cen
    wvl_idx = np.argmin(np.abs(wvl - wvl_plot))
    plt.figure(dpi=150)
    plt.title(r"Point spread function ($\lambda=$"+f"{round(wvl[wvl_idx]/NM)}nm)")
    plt.pcolormesh(xx, yy, I[:,:,wvl_idx])
    plt.xlabel("y coordinate")
    plt.ylabel("x coordinate")
    plt.colorbar()
    # plt.savefig("psf_focus.png", bbox_inches="tight")

    plt.figure(dpi=150)
    plt.title(r"PSF over spectrum ($\lambda_{\text{cen.}}=$"+f"{round(wvl_cen/NM)}nm)")
    plt.pcolormesh(np.meshgrid(x, wvl)[1], np.meshgrid(x, wvl)[0], I[:,len(x)//2,:].T)
    np.shape(I[:,100,:])
    plt.xlabel("wavelength")
    plt.ylabel("x coordinate")
    plt.colorbar()
    # plt.savefig("psf_spect_slice_focus.png", bbox_inches="tight")

    #%%
    plt.figure(dpi=200)
    cart = zernike.RZern(10)
    var = np.zeros(cart.nk+1)
    score = 1E100

    var0 = (np.random.rand(len(var))-0.5)
    var_opt = adam(loss_fun, var0, num_iters=100, step_size=0.01)
    print(loss_fun(var_opt))
    #%%
    score = 1E100
    cart = zernike.RZern(10)
    for i in tqdm(range(10)):
        var_new = 1E7*(np.random.rand(cart.nk+1)-0.5)
        var_new[-1] = 0
        x_range = [0.0]
        y_range = [0.0]
        sol = adam(loss_fun, var_new, num_iters=20)
        score_new = loss_fun(sol)
        if score_new < score:
            score = score_new
            var_opt = sol
            print(score)

    # # plt.imshow(perturb_phase(xx, yy, res.x)%(-2*np.pi))

    # plt.imshow(total_phase(xx, yy, res.x, focl, wvl_cen))

    I = np.empty(np.shape(xxx))
    for i in tqdm(range(len(wvl))):
        I[:,:,i] = psf(xx, yy, 0, 0, wvl[i], var_opt, focl, wvl_cen, R, Z, Np)
        # I[:,:,i] = psf_ang_dist(xx, yy, wvl[i], wvl.min(), wvl.max())#

    plt.imshow(total_phase(xx, yy, var_opt, focl, wvl_cen))
    # plt.savefig("optpsf1_phase.png")

    wvl_plot = wvl_cen
    wvl_idx = np.argmin(np.abs(wvl - wvl_plot))
    plt.figure(dpi=150)
    plt.title(r"Point spread function ($\lambda=$"+f"{round(wvl[wvl_idx]/NM)}nm)")
    plt.pcolormesh(xx, yy, I[:,:,wvl_idx], vmax=np.max(I))
    plt.colorbar()
    plt.xlabel("y coordinate")
    plt.ylabel("x coordinate")


    plt.figure(dpi=150)
    plt.xticks([])
    plt.yticks([])
    plt.gca().set(frame_on=False)
    for i in range(1,10):
        wvl_idx = 2*i
        plt.subplot(3,3,i)
        plt.title(r"$\lambda=$"+f"{round(wvl[wvl_idx]/NM)}nm", fontsize=5)
        plt.pcolormesh(xx[len(x)//2-Nx:len(x)//2+Nx+1,len(x)//2-Nx:len(x)//2+Nx+1], yy[len(x)//2-Nx:len(x)//2+Nx+1,len(x)//2-Nx:len(x)//2+Nx+1], I[len(x)//2-Nx:len(x)//2+Nx+1,len(x)//2-Nx:len(x)//2+Nx+1,wvl_idx], vmax=np.max(I))
        plt.xticks([])
        plt.yticks([])
    # plt.savefig("optpsf3_spect.png")

    plt.show()
    print(loss_fun(var_opt))
    #%%
    I_targ = np.empty(np.shape(xxx))
    for i in tqdm(range(len(wvl))):
        I_targ[:,:,i] = psf_ang_dist(xx, yy, wvl[i], wvl.min(), wvl.max())


    wvl_plot = wvl_cen
    wvl_idx = np.argmin(np.abs(wvl - wvl_plot))
    plt.figure(dpi=150)
    plt.title(r"Point spread function ($\lambda=$"+f"{round(wvl[wvl_idx]/NM)}nm)")
    plt.pcolormesh(xx, yy, I_targ[:,:,wvl_idx], vmax=np.max(I_targ))
    plt.colorbar()
    plt.xlabel("y coordinate")
    plt.ylabel("x coordinate")


    plt.figure(dpi=150)
    plt.xticks([])
    plt.yticks([])
    plt.gca().set(frame_on=False)
    for i in range(1,10):
        wvl_idx = 2*i
        plt.subplot(3,3,i)
        plt.title(r"$\lambda=$"+f"{round(wvl[wvl_idx]/NM)}nm", fontsize=5)
        plt.pcolormesh(xx[len(x)//2-Nx:len(x)//2+Nx+1,len(x)//2-Nx:len(x)//2+Nx+1], yy[len(x)//2-Nx:len(x)//2+Nx+1,len(x)//2-Nx:len(x)//2+Nx+1], I_targ[len(x)//2-Nx:len(x)//2+Nx+1,len(x)//2-Nx:len(x)//2+Nx+1,wvl_idx], vmax=np.max(I_targ))
        plt.xticks([])
        plt.yticks([])
    # plt.savefig("optpsf3_spect.png")
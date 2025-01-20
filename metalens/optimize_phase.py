#%%
import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt
import zernike
import scipy.optimize as optimize
from tqdm import tqdm


class OptimizePSF():

    def __init__(self, sx, px, res, ra, focl, zf, wvl_cen, x_r, y_r, wvl_r, c_idx):
        self.sx = sx
        self.px = px
        self.res = res
        self.ra = ra
        self.focl = focl
        self.zf = zf
        self.wvl_cen = wvl_cen
        self.x_r = x_r
        self.y_r = y_r
        self.wvl_r = wvl_r
        self.c_idx = c_idx

        self.rx = np.linspace(-sx, sx, round(2*(sx*res)+1))
        self.x, self.y = np.meshgrid(self.rx, self.rx)
        

    def pt_source(self, x, y, wvl):
        k = 2*np.pi / wvl
        r = np.sqrt(x**2 + y**2 + self.zf**2)
        return np.cos(k*r) / r**2
    
    def focus_phase(self, x, y):
        phi_foc = 2*np.pi/self.wvl_cen*(self.focl - np.sqrt(self.focl**2 + x**2 + y**2))
        return phi_foc
    
    def perturb_phase(self, x, y, wvl, var):
        cart = zernike.RZern(9)
        cart.make_cart_grid(x, y)
        c = np.zeros(cart.nk)
        for i in range(len(self.c_idx)):
            c[self.c_idx[i]] = var[i]
        phi_var = cart.eval_grid(c, matrix=True)
        phi_var
        return phi_var
    
    def cylind_basis(self, x, y, wvl, var, m):
        L = self.wvl_r.max()-self.wvl_r.min()
        z_basis = np.sin(m*np.pi*wvl/L)
        circ_basis = self.perturb_phase(x, y, var)
        basis_fun = np.outer(circ_basis, z_basis)
        return basis_fun
    
    def pupil_fun(self, x, y, var):
        r = np.sqrt(x**2 + y**2)
        phase = self.focus_phase(x, y) + self.perturb_phase(x, y, var)
        # phase = self.perturb_phase(x, y, var)
        phase %= -2*np.pi
        return np.exp(1j*phase)*(r <= self.ra)

    def psf(self, x, y, x_cen, y_cen, wvl, var):
        ppl_E = self.pt_source(x - x_cen, y - y_cen, wvl)
        ppl_fun =  self.pupil_fun(x, y, var)
        intens = np.abs(fft.fftshift(fft.fft2(ppl_E * ppl_fun)))**2
        return intens

    def dtheta_i(self, fun, theta, i):
        eps = 1E-5
        theta_c = theta.copy()
        theta_c[i] = theta[i] - eps
        f_l = fun(*theta_c)
        theta_c[i] = theta[i] + eps
        f_r = fun(*theta_c)
        return (f_r - f_l) / (2*eps)

    def crlb(self, x, y, theta, var):
        nt = len(theta)
        idx_cen = (np.argmin(np.abs(x-theta[0])), np.argmin(np.abs(x-theta[1])))
        idx_x = (idx_cen[0] - self.px, idx_cen[0] + self.px + 1)
        idx_y = (idx_cen[1] - self.px, idx_cen[1] + self.px + 1)
        xp = x[idx_x[0]:idx_x[1], idx_y[0]:idx_y[1]]
        yp = y[idx_x[0]:idx_x[1], idx_y[0]:idx_y[1]]
        psf_theta = self.psf(xp, yp, *theta, var)
        crlb = np.empty(len(theta))
        for it in range(nt):
            smnd = 1/(psf_theta)*self.dtheta_i(self.psf, [xp, yp, *theta, var], 2+it)**2
            crlb[it] = 1/np.sum(smnd)
        return crlb, psf_theta

    def loss_fun(self, var):
        L = 0
        for i in range(len(self.x_r)):
            for j in range(len(self.y_r)):
                for k in range(len(self.wvl_r)):
                    theta = (self.x_r[i], self.y_r[j], self.wvl_r[k])
                    crlb, psf_theta = self.crlb(self.x, self.y, theta, var[k*len(self.c_idx):(k+1)*len(self.c_idx)])
                    L += np.sqrt(crlb)
                    # constraint = np.sum(np.log(psf_theta/np.sum(psf_theta) + 1))
                    # L += 1E-3*constraint
        return np.sum(L)

    def optimizePhase(self, ni):
        self.score = 1E100
        for i in tqdm(range(ni)):
            var0 = np.zeros(len(self.c_idx)*len(self.wvl_r))
            for j in range(len(self.wvl_r)):
                var0[j*len(self.c_idx):(j+1)*len(self.c_idx)] = 5E8*np.random.rand(len(self.c_idx))
            res = optimize.minimize(self.loss_fun, var0)
            if res.fun < self.score:
                self.score = res.fun
                self.var_opt = res.x

    def plotPSF(self):
        psf_spect = np.empty((*np.shape(self.x), len(self.wvl_r)))
        for i in range(len(self.wvl_r)):
            psf_spect[:,:,i] = self.psf(self.x, self.y, 0, 0, self.wvl_r[i], self.var_opt)

        wvl_plot = self.wvl_cen
        wvl_idx = np.argmin(np.abs(wvl_r - wvl_plot))
        plt.figure(dpi=200)
        N = int(np.sqrt(len(self.wvl_r))) + 1
        for i in range(len(self.wvl_r)):
            plt.subplot(N, N, i+1)
            plt.title(r"$\lambda=$"+f"{round(self.wvl_r[i]/NM)}nm", fontsize=5)
            plt.pcolormesh(self.x, self.y,
                (self.focus_phase(self.x, self.y)+self.perturb_phase(self.x, self.y, self.var_opt[i]))%(-2*np.pi)
        )
        # plt.savefig("optpsf2_phase.png")

        # plt.figure(dpi=200)
        # plt.title(r"Point spread function ($\lambda=$"+f"{round(wvl_r[wvl_idx]/NM)}nm)")
        # plt.pcolormesh(self.x, self.y, psf_spect[:,:,wvl_idx])
        # plt.xlabel("y coordinate")
        # plt.ylabel("x coordinate")
        # # plt.savefig("optpsf2_foc.png")

        plt.figure(dpi=200)
        plt.xticks([])
        plt.yticks([])
        plt.gca().set(frame_on=False)
        N = int(np.sqrt(len(self.wvl_r))) + 1
        for i in range(len(self.wvl_r)):
            plt.subplot(N, N, i+1)
            plt.title(r"$\lambda=$"+f"{round(self.wvl_r[i]/NM)}nm", fontsize=5)
            x_plot = self.x[len(self.rx)//2-self.px:len(self.rx)//2+self.px+1,len(self.rx)//2-self.px:len(self.rx)//2+self.px+1]
            y_plot = self.y[len(self.rx)//2-self.px:len(self.rx)//2+self.px+1,len(self.rx)//2-self.px:len(self.rx)//2+self.px+1]
            psf_plot = psf_spect[len(self.rx)//2-self.px:len(self.rx)//2+self.px+1,len(self.rx)//2-self.px:len(self.rx)//2+self.px+1, i]
            plt.pcolormesh(x_plot, y_plot, psf_plot)
            plt.xticks([])
            plt.yticks([])
        # plt.savefig("optpsf2_spect.png")

    # def plotCrossSection(self):
    #     psf_spect = np.empty((*np.shape(self.x), len(self.wvl_r)))
    #     for i in range(len(self.wvl_r)):
    #         psf_spect[:,:,i] = self.psf(self.x, self.y, 0, 0, self.wvl_r[i], self.var_opt)

    #     psf_spect = psf_spect[:,:,1:-1]
    #     xxx, yyy, lll = np.meshgrid(self.rx, self.rx, self.wvl_r[1:-1])
    #     voxels = (psf_spect >= 0.5*np.max(psf_spect))
    #     ax = plt.figure(dpi=300).add_subplot(projection='3d')
    #     ax.voxels(voxels, facecolors="red", edgecolor='k')
#%%

if __name__=="__main__":
    NM = 1E-9
    UM = 1E-6

    sx = 50*UM
    px = 8
    res = 2/UM
    ra = 25*UM
    focl = 250*UM
    zf = focl
    wvl_cen = 900*NM
    
    x_r = [0]
    y_r = [0]
    wvl_r = np.arange(600*NM, 1201*NM, 100*NM)

    c_idx = np.arange(55)
    mask = [1,2,6,7,15,16,28,29,45,46]
    mask = [0,1,2,4,6,7,9,12,14,15,16,17,19,22,24,26,28,29,31,33,35,38,40,42,44,45,46,47,49,51,53]
    c_idx = np.delete(c_idx, mask)

    solve = OptimizePSF(sx, px, res, ra, focl, zf, wvl_cen, x_r, y_r, wvl_r, c_idx)

    solve.optimizePhase(10)
#%%
    psf_spect = np.empty((*np.shape(solve.x), len(solve.wvl_r)))
    for i in range(len(solve.wvl_r)):
        psf_spect[:,:,i] = solve.psf(solve.x, solve.y, 0, 0, solve.wvl_r[i], solve.var_opt)

    wvl_plot = solve.wvl_cen
    wvl_idx = np.argmin(np.abs(wvl_r - wvl_plot))
    plt.figure(dpi=200)
    plt.title(r"Point spread function ($\lambda=$"+f"{round(wvl_r[wvl_idx]/NM)}nm)")
    plt.pcolormesh(solve.x, solve.y, psf_spect[:,:,wvl_idx])
    plt.xlabel("y coordinate")
    plt.ylabel("x coordinate")

    plt.figure(dpi=200)
    N = int(np.sqrt(len(solve.wvl_r))) + 1
    for i in range(len(solve.wvl_r)):
        plt.subplot(N, N, i+1)
        plt.title(r"$\lambda=$"+f"{round(solve.wvl_r[i]/NM)}nm", fontsize=5)
        plt.pcolormesh(solve.x, solve.y,
            (solve.focus_phase(solve.x, solve.y)+ solve.perturb_phase(solve.x, solve.y, solve.var_opt[i*len(solve.c_idx):(i+1)*len(solve.c_idx)]))%(-2*np.pi)
        )
        plt.xticks([])
        plt.yticks([])
    
    plt.figure(dpi=200)
    plt.xticks([])
    plt.yticks([])
    plt.gca().set(frame_on=False)
    N = int(np.sqrt(len(solve.wvl_r))) + 1
    for i in range(len(solve.wvl_r)):
        plt.subplot(N, N, i+1)
        plt.title(r"$\lambda=$"+f"{round(solve.wvl_r[i]/NM)}nm", fontsize=5)
        x_plot = solve.x[len(solve.rx)//2-solve.px:len(solve.rx)//2+solve.px+1,len(solve.rx)//2-solve.px:len(solve.rx)//2+solve.px+1]
        y_plot = solve.y[len(solve.rx)//2-solve.px:len(solve.rx)//2+solve.px+1,len(solve.rx)//2-solve.px:len(solve.rx)//2+solve.px+1]
        psf_plot = psf_spect[len(solve.rx)//2-solve.px:len(solve.rx)//2+solve.px+1,len(solve.rx)//2-solve.px:len(solve.rx)//2+solve.px+1, i]
        plt.pcolormesh(x_plot, y_plot, psf_plot)
        plt.xticks([])
        plt.yticks([])
#%%
import meep as mp
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate

class UnitCell3D():

    def __init__(self, cell_size, dpml, res, wvl_min, wvl_max, nfreq,
                  sub_mat, grat_mat, gparams, symmetries=[]):
        self.cell_size = cell_size
        self.sx = cell_size[0]
        self.sy = cell_size[1]
        self.sz = cell_size[2]
        self.dpml = dpml
        self.res = res
        self.wvl_min = wvl_min
        self.wvl_max = wvl_max
        self.nfreq = nfreq
        self.symmetries = symmetries
        self.num_cells = 0
        self.sub_mat = sub_mat
        self.grat_mat = grat_mat
        self.gparams = gparams
        self.geometry = []

        self.pml_layers = [mp.PML(thickness=dpml, direction=mp.X)]
        self.k_point = mp.Vector3(0, 0, 0)
        self.src_pt = mp.Vector3(-0.5 * self.sx + 2*dpml, 0, 0)
        self.mon_pt = mp.Vector3(0.5 * self.sx - 2*dpml, 0, 0)

        fmin = 1/self.wvl_max
        fmax = 1/self.wvl_min
        self.fcen = 0.5*(fmin + fmax)
        self.df = fmax - fmin

        self.initSource()
        self.initGeometry()
        

    def initSource(self):
        self.sources = [mp.Source(
                mp.GaussianSource(self.fcen, fwidth=self.df),
                component = mp.Ez,
                center = self.src_pt,
                size = mp.Vector3(0, self.sy*(2*self.num_cells+1), self.sz*(2*self.num_cells+1)),
            )
        ]


    def initGeometry(self, center=(0,0)):
        if center == (0,0):
            self.dsub = self.gparams[0]
            self.geometry.append(
                mp.Block(
                    material=sub_mat,
                    size=mp.Vector3(self.dpml + self.dsub, mp.inf, mp.inf),
                    center=mp.Vector3(-0.5*self.sx + 0.5*(self.dpml + self.dsub), 0, 0),
                )
            )
        if self.gparams[1] == "block":
            self.gdc = self.gparams[2]
            self.gh = self.gparams[3]
            self.geometry.append(
                mp.Block(
                    material=grat_mat,
                    size=mp.Vector3(self.gdc, self.gdc*self.sy, self.gdc*self.sz),
                    center=mp.Vector3(-0.5*self.sx + self.dpml + self.dsub + 0.5*self.gh, *center),
                )
            )


    def calcInputFlux(self):
        self.sim = mp.Simulation(
            resolution = self.res,
            cell_size = self.cell_size,
            boundary_layers = self.pml_layers,
            k_point = self.k_point,
            default_material = self.sub_mat,
            sources = self.sources,
            symmetries = self.symmetries,
        )

        flux_size = mp.Vector3(0, self.sy*(2*self.num_cells+1), self.sz*(2*num_cells+1))
        flux_obj = self.sim.add_flux(
            self.fcen, self.df, self.nfreq, mp.FluxRegion(center=self.mon_pt, size=flux_size)
        )

        self.sim.run(until_after_sources=100)
        self.input_flux = mp.get_fluxes(flux_obj)
        self.sim.reset_meep()


    def calcOutputFlux(self):
        self.sim = mp.Simulation(
            resolution = self.res,
            cell_size = self.cell_size,
            boundary_layers = self.pml_layers,
            geometry = self.geometry,
            k_point = self.k_point,
            sources = self.sources,
            symmetries = self.symmetries,
        )
        
        flux_size = mp.Vector3(0, self.sy*(2*self.num_cells+1), self.sz*(2*num_cells+1))
        flux_obj = self.sim.add_flux(
            self.fcen, self.df, self.nfreq, mp.FluxRegion(center=self.mon_pt, size=flux_size)
        )

        self.sim.run(until_after_sources=300)

        freqs = mp.get_eigenmode_freqs(flux_obj)
        res = self.sim.get_eigenmode_coefficients(
            flux_obj, [1], eig_parity=mp.ODD_Z + mp.EVEN_Y
        )
        coeffs = res.alpha
        
        self.mode_wvl = [1 / freqs[nf] for nf in range(self.nfreq)]
        self.mode_tran = [abs(coeffs[0, nf, 0]) ** 2 / self.input_flux[nf] for nf in range(self.nfreq)]
        self.mode_phase = [np.angle(coeffs[0, nf, 0]) for nf in range(self.nfreq)]



class Metalens(UnitCell3D):

    def __init__(self, num_cells, focal_length, gvar_range, phase_range, ff_res, *args):
        super().__init__(*args)
        
        self.num_cells = num_cells
        self.lens_size = mp.Vector3(self.cell_size[0], (2*num_cells+1)*self.cell_size[1], (2*num_cells+1)*self.cell_size[2])
        self.lx = self.lens_size[0]
        self.ly = self.lens_size[1]
        self.lz = self.lens_size[2]
        self.focal_length = focal_length
        self.gvar_range = gvar_range
        self.phase_range = phase_range
        self.ff_res = ff_res

        self.interpGratingPhase()
        self.buildSurface()


    def interpGratingPhase(self):
        wvl_cen = 0.5*(self.wvl_max + self.wvl_min)
        r = np.arange(-self.num_cells, self.num_cells + 1)
        ii, jj = np.meshgrid(r, r)
        f = self.focal_length
        self.phase_grid = 2*np.pi/wvl_cen * (f - ((ii*self.sy)**2 + (jj*self.sz)**2 + f**2)**0.5)
        self.phase_grid %= (-2*np.pi)

        gvar_interp = interpolate.CubicSpline(self.phase_range, self.gvar_range, extrapolate=True)
        self.gvar_grid = gvar_interp(np.reshape(self.phase_grid, -1))
        self.gvar_grid = np.reshape(self.gvar_grid, np.shape(self.phase_grid))


    def buildSurface(self):
        for i in range(2*self.num_cells + 1):
            for j in range(2*self.num_cells + 1):
                center = ((i - self.num_cells)*self.sy, (j - self.num_cells)*self.sz)
                gvar_idx = self.gparams[-1]
                self.gparams[gvar_idx] = self.gvar_grid[i, j]
                self.initGeometry(center=center)
                self.interpGratingPhase()


    def initSim(self):
        self.sim = mp.Simulation(
            resolution=self.res,
            cell_size=self.lens_size,
            boundary_layers=self.pml_layers,
            geometry=self.geometry,
            k_point=self.k_point,
            sources=self.sources,
            symmetries=self.symmetries,
        )
    

    def plotLensProfile(self):
        self.initSim()
        plt.figure(dpi=400)
        self.sim.plot2D(
            output_plane=mp.Volume(
                center=mp.Vector3(-0.5*self.lx+self.dpml+self.dsub+0.5*self.gh,0,0),
                size=mp.Vector3(0,self.ly,self.lz)
            )
        )
    

    def runSim(self):
        self.initSim()
        self.n2f_obj = self.sim.add_near2far(
            fcen, 0, 1, mp.Near2FarRegion(center=self.mon_pt, size=mp.Vector3(0, self.ly, self.lz))
        )
        self.sim.run(until_after_sources=100)


    def getFarFields(self):
        self.imgplane_int = abs(
            self.sim.get_farfields(
                    self.n2f_obj,
                    self.ff_res,
                    center=mp.Vector3(-0.5*self.lx+self.dpml+self.dsub+self.gh+self.focal_length, 0, 0),
                    size=mp.Vector3(0, self.ly/20, self.lz/20),
                )["Ez"]
        )** 2
        
        self.tanplane_int = abs(
            self.sim.get_farfields(
                    self.n2f_obj,
                    self.ff_res,
                    center=mp.Vector3(-0.5*self.lx+self.dpml+self.dsub+self.gh+self.focal_length, 0, 0),
                    size=mp.Vector3(self.focal_length/2, 0, 0),
                )["Ez"]
        )** 2

#%%
if __name__ == "__main__": 

    wvl_min = 0.4  # min wavelength
    wvl_max = 0.6  # max wavelength
    fmin = 1 / wvl_max  # min frequency
    fmax = 1 / wvl_min  # max frequency
    fcen = 0.5 * (fmin + fmax)  # center frequency
    df = fmax - fmin  # frequency width
    nfreq = 21  # number of frequency bins
    k_point = mp.Vector3(0, 0, 0)

    dsub = 3.0  # substrate thickness
    dpad = 3.0  # padding between grating and PML
    gp = 0.3  # grating periodicity
    gh = 1.8  # grating height
    gdc = np.linspace(0.1, 0.9, 30) # grating duty cycle

    glass = mp.Medium(index=1.5)
    sub_mat = glass  # substrate material
    grat_mat = glass  # grating material

    dpml = 1.0  # PML thickness
    res = 50  # resolution
    cell_size = mp.Vector3(dpml+dsub+gh+dpad+dpml, gp, gp)
    symmetries = [mp.Mirror(mp.Y)]

    num_cells = 25
    focal_length = 100
    ff_res = res

    # Load unit cell phase simulation
    data = np.load("wvl_tran_phase_dsubdpad30_gh18_gp03_glass.npz")
    var_range = data["gdc_range"][11:27]
    phase_range = data["mode_phase"][11:27, nfreq//2]

    lens_params = [num_cells, focal_length, var_range, phase_range, ff_res]
    gparams = [dsub, "block", gdc[0], gh, 2]
    cell_params = [cell_size, dpml, res, wvl_min, wvl_max, nfreq, sub_mat, grat_mat, gparams]
    
    block_metalens = Metalens(*lens_params, *cell_params)

    block_metalens.plotLensProfile()


#%%
if __name__ == "__main__": 

    # Simulation spectrum
    wvl_min = 0.4  # min wavelength
    wvl_max = 0.6  # max wavelength
    fmin = 1 / wvl_max  # min frequency
    fmax = 1 / wvl_min  # max frequency
    fcen = 0.5 * (fmin + fmax)  # center frequency
    df = fmax - fmin  # frequency width
    nfreq = 21  # number of frequency bins
    k_point = mp.Vector3(0, 0, 0)

    # Simulation geometry
    dpml = 1.0  # PML thickness
    dsub = 3.0  # substrate thickness
    dpad = 3.0  # padding between grating and PML
    gp = 0.3  # grating periodicity
    gh = 1.8  # grating height
    gdc = np.linspace(0.1, 0.9, 30) # grating duty cycle

    glass = mp.Medium(index=1.5)
    sub_mat = glass  # substrate material
    grat_mat = glass  # grating material

    cell_size = mp.Vector3(dpml+dsub+gh+dpad+dpml, gp, gp)
    res = 50  # resolution
    symmetries = [mp.Mirror(mp.Y)]

    gparams = [dsub, "block",  gdc[0], gh]

    # Input Flux
    empty_cell = UnitCell3D(cell_size, dpml, res, wvl_min, wvl_max, nfreq, sub_mat, grat_mat, symmetries)
    empty_cell.calcInputFlux()

    # Output flux phase & transmission
    mode_tran = np.empty((gdc.size, nfreq))
    mode_phase = np.empty((gdc.size, nfreq))
    for i in range(len(gdc)):
        gparams[2] = gdc[i]
        block_cell = UnitCell3D(cell_size, dpml, res, wvl_min, wvl_max, nfreq, 
                                sub_mat, grat_mat, gparams, symmetries)
        
        block_cell.input_flux = empty_cell.input_flux
        block_cell.calcOutputFlux()

        mode_wvl = block_cell.mode_wvl
        mode_tran[i, :] = block_cell.mode_tran
        mode_phase[i, :] = block_cell.mode_phase



    # # Input Flux (center wavelength)
    # empty_cell = UnitCell3D(cell_size, dpml, res, wvl_min, wvl_max, nfreq, symmetries)
    # empty_cell.nfreq = 1
    # empty_cell.df = 0
    # empty_cell.calcInputFlux()

    # # Output flux phase & transmission (center wavelength)

    

#%%

plt.figure(dpi=200)
plt.subplot(1, 2, 1)
plt.pcolormesh(
    mode_wvl,
    gdc,
    mode_tran,
    cmap="hot_r",
    shading="gouraud",
    vmin=0,
    vmax=mode_tran[:-1,:].max(),
)
plt.axis([wvl_min, wvl_max, gdc[0], gdc[-1]])
plt.xlabel("wavelength (μm)")
plt.xticks([t for t in np.linspace(wvl_min, wvl_max, 3)])
plt.ylabel("grating duty cycle (gdc)")
plt.yticks([t for t in np.arange(gdc[0], gdc[-1] + 0.1, 0.1)])
plt.title("transmittance")
cbar = plt.colorbar()
cbar.set_ticks([t for t in np.arange(0, 1.2, 0.2)])
cbar.set_ticklabels(["{:.1f}".format(t) for t in np.linspace(0, 1, 6)])

plt.subplot(1, 2, 2)
plt.pcolormesh(
    mode_wvl,
    gdc,
    mode_phase,
    cmap="RdBu",
    shading="gouraud",
    vmin=mode_phase.min(),
    vmax=mode_phase.max(),
)
plt.axis([wvl_min, wvl_max, gdc[0], gdc[-1]])
plt.xlabel("wavelength (μm)")
plt.xticks([t for t in np.linspace(wvl_min, wvl_max, 3)])
plt.ylabel("grating duty cycle (gdc)")
plt.yticks([t for t in np.arange(gdc[0], gdc[-1] + 0.1, 0.1)])
plt.title("phase (radians)")
cbar = plt.colorbar()
cbar.set_ticks([t for t in range(-3, 4)])
cbar.set_ticklabels(["{:.1f}".format(t) for t in range(-3, 4)])
plt.subplots_adjust(wspace=0.5)

plt.savefig("blockcell_phase_tran_spectrum.png")



#%%
plt.figure(dpi=100)
plt.subplot(1, 2, 1)
plt.plot(gdc[11:27], mode_tran[11:27,nfreq//2], "bo-")
plt.xlim(gdc[11], gdc[26])
plt.xticks([t for t in np.linspace(0.1, 0.9, 5)])
plt.xlabel("grating duty cycle")
plt.ylim(0.96, 1.00)
plt.yticks([t for t in np.linspace(0.96, 1.00, 5)])
plt.title("transmittance")

plt.subplot(1, 2, 2)
plt.plot(gdc[11:27], mode_phase[11:27,nfreq//2] % (-2*np.pi), "rs-")
plt.grid(True)
plt.xlim(gdc[11], gdc[26])
plt.xticks([t for t in np.linspace(0.1, 0.9, 5)])
plt.xlabel("grating duty cycle")
plt.ylim(-2 * np.pi, 0)
plt.yticks([t for t in np.linspace(-6, 0, 7)])
plt.title("phase (radians)")

plt.tight_layout(pad=0.5)

# plt.savefig("blockcell_phase_range.png")



    
# def addFFRegion(self, center, size, ff_res):
#     self.n2f_regions.append((center, size, ff_res))
#     self.n2f_objs.append(
#         self.sim.add_near2far(
#             self.fcen, 0, 1, mp.Near2FarRegion(center=center, size=size)))
    

# def runSim(self):
#     self.sim.run(until_after_sources=100)

#     self.far_fields = []
#     for i in range(len(self.n2f_objs)):
#         Ez = abs(self.sim.get_farfields(
#                 self.n2f_objs[i],
#                 resolution = self.n2f_regions[2],
#                 center = self.n2f_regions[0],
#                 size = self.n2f_regions[1])["Ez"])
#         self.far_field.append(abs(Ez)**2)



        
        
    
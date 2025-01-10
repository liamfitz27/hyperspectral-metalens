#%%
import meep as mp
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate
from tqdm import tqdm
from meep.materials import SiO2
import json

class UnitCell3D():
    def __init__(self, cell_size, dpml, dsub, sub_mat, grat_mat, res, wvl_min, wvl_max, nfreq, symmetries=[]):
        self.cell_size = cell_size
        self.sx = cell_size[0]
        self.sy = cell_size[1]
        self.sz = cell_size[2]
        self.dpml = dpml
        self.dsub = dsub
        self.sub_mat = sub_mat
        self.grat_mat = grat_mat
        self.res = res
        self.wvl_min = wvl_min
        self.wvl_max = wvl_max
        self.nfreq = nfreq
        self.symmetries = symmetries

        self.num_cells = 0
        self.geometry = []
        self.pml_layers = [mp.PML(thickness=dpml, direction=mp.X)]
        self.k_point = mp.Vector3(0, 0, 0)
        self.src_pt = mp.Vector3(-0.5 * self.sx + dpml + 0.5*self.dsub, 0, 0)
        self.mon_pt = mp.Vector3(0.5 * self.sx - 2*dpml, 0, 0)
        # self.refl_pt = mp.Vector3(-0.5 * self.sx + dpml + 0.25*dsub, 0, 0)

        self.fmin = 1/self.wvl_max
        self.fmax = 1/self.wvl_min
        self.fcen = 0.5*(self.fmin + self.fmax)
        self.df = self.fmax - self.fmin

        self.initSource()

    def initSource(self):
        self.sources = [mp.Source(
                mp.GaussianSource(self.fcen, fwidth=self.df),
                component = mp.Ez,
                center = self.src_pt,
                size = mp.Vector3(0, self.sy*(2*self.num_cells+1), self.sz*(2*self.num_cells+1)),
            )
        ]

    def initGeometry(self, gparams, center=(0,0)):
        if center == (0,0):
            self.geometry.append(
                mp.Block(
                    material=self.sub_mat,
                    size=mp.Vector3(self.dsub, mp.inf, mp.inf),
                    center=mp.Vector3(-0.5*self.sx + self.dpml + 0.5*self.dsub, 0, 0),
                )
            )
        if gparams[0] == "block":
            self.gh = gparams[1]
            self.gw = gparams[2]
            self.geometry.append(
                mp.Block(
                    material=self.grat_mat,
                    size=mp.Vector3(self.gh, self.gw*self.sy, self.gw*self.sz),
                    center=mp.Vector3(-0.5*self.sx + self.dpml + self.dsub + 0.5*self.gh, *center),
                )
            )
        if gparams[0] == "rectang":
            self.gh = gparams[1]
            self.gl = gparams[2]
            self.gw = gparams[3]
            self.geometry.append(
                mp.Block(
                    material=self.grat_mat,
                    size=mp.Vector3(self.gh, self.gl*self.sy, self.gw*self.sz),
                    center=mp.Vector3(-0.5*self.sx + self.dpml + self.dsub + 0.5*self.gh, *center),
                )
            )
        if gparams[0] == "four_blocks":
            self.gh = gparams[1]
            self.gw = gparams[2]
            self.gwi = gparams[3]
            self.geometry.append(
                mp.Block(
                    material=self.grat_mat,
                    size=mp.Vector3(self.gh, self.gw*self.sy, self.gw*self.sz),
                    center=mp.Vector3(-0.5*self.sx + self.dpml + self.dsub + 0.5*self.gh, *center),
                )
            )
            self.geometry.append(
                mp.Block(
                    material=mp.Medium(index=1),
                    size=mp.Vector3(self.gh, self.gw*self.sy, self.gwi*self.sz),
                    center=mp.Vector3(-0.5*self.sx + self.dpml + self.dsub + 0.5*self.gh, *center),
                )
            )
            self.geometry.append(
                mp.Block(
                    material=mp.Medium(index=1),
                    size=mp.Vector3(self.gh, self.gwi*self.sy, self.gw*self.sz),
                    center=mp.Vector3(-0.5*self.sx + self.dpml + self.dsub + 0.5*self.gh, *center),
                )
            )
        if gparams[0] == "hollow_block":
            self.gh = gparams[1]
            self.gw = gparams[2]
            self.gwi = gparams[3]
            self.geometry.append(
                mp.Block(
                    material=self.grat_mat,
                    size=mp.Vector3(self.gh, self.gw*self.sy, self.gw*self.sz),
                    center=mp.Vector3(-0.5*self.sx + self.dpml + self.dsub + 0.5*self.gh, *center),
                )
            )
            self.geometry.append(
                mp.Block(
                    material=mp.Medium(index=1),
                    size=mp.Vector3(self.gh, self.gwi*self.sy, self.gwi*self.sz),
                    center=mp.Vector3(-0.5*self.sx + self.dpml + self.dsub + 0.5*self.gh, *center),
                )
            )
        if gparams[0] == "window":
            self.gh = gparams[1]
            self.gw = gparams[2]
            self.gwi = gparams[3]
            self.gwii = gparams[4]
            self.geometry.append(
                mp.Block(
                    material=self.grat_mat,
                    size=mp.Vector3(self.gh, self.gw*self.sy, self.gw*self.sz),
                    center=mp.Vector3(-0.5*self.sx + self.dpml + self.dsub + 0.5*self.gh, *center),
                )
            )
            self.geometry.append(
                mp.Block(
                    material=mp.Medium(index=1),
                    size=mp.Vector3(self.gh, self.gwi*self.sy, self.gwi*self.sz),
                    center=mp.Vector3(-0.5*self.sx + self.dpml + self.dsub + 0.5*self.gh, *center),
                )
            )
            self.geometry.append(
                mp.Block(
                    material=self.grat_mat,
                    size=mp.Vector3(self.gh, self.gwi*self.sy, self.gwii*self.sz),
                    center=mp.Vector3(-0.5*self.sx + self.dpml + self.dsub + 0.5*self.gh, *center),
                )
            )
            self.geometry.append(
                mp.Block(
                    material=self.grat_mat,
                    size=mp.Vector3(self.gh, self.gwii*self.sy, self.gwi*self.sz),
                    center=mp.Vector3(-0.5*self.sx + self.dpml + self.dsub + 0.5*self.gh, *center),
                )
            )
        if gparams[0] == "interior_block":
            self.gh = gparams[1]
            self.gw = gparams[2]
            self.gwi = gparams[3]
            self.gwii = gparams[4]
            self.geometry.append(
                mp.Block(
                    material=self.grat_mat,
                    size=mp.Vector3(self.gh, self.gw*self.sy, self.gw*self.sz),
                    center=mp.Vector3(-0.5*self.sx + self.dpml + self.dsub + 0.5*self.gh, *center),
                )
            )
            self.geometry.append(
                mp.Block(
                    material=mp.Medium(index=1),
                    size=mp.Vector3(self.gh, self.gwi*self.sy, self.gwi*self.sz),
                    center=mp.Vector3(-0.5*self.sx + self.dpml + self.dsub + 0.5*self.gh, *center),
                )
            )
            self.geometry.append(
                mp.Block(
                    material=self.grat_mat,
                    size=mp.Vector3(self.gh, self.gwii*self.sy, self.gwii*self.sz),
                    center=mp.Vector3(-0.5*self.sx + self.dpml + self.dsub + 0.5*self.gh, *center),
                )
            )
        if gparams[0] == "cross":
            self.gh = gparams[1]
            self.gl = gparams[2]
            self.gw = gparams[3]
            self.geometry.append(
                mp.Block(
                    material = self.grat_mat,
                    size = mp.Vector3(self.gh, self.gl*self.sy, self.gw*self.sz),
                    center = mp.Vector3(-0.5*self.sx + self.dpml + self.dsub + 0.5*self.gh, *center),
                )
            )
            self.geometry.append(
                mp.Block(
                    material = self.grat_mat,
                    size = mp.Vector3(self.gh, self.gw*self.sy, self.gl*self.sz),
                    center = mp.Vector3(-0.5*self.sx + self.dpml + self.dsub + 0.5*self.gh, *center),
                )
            )
        if gparams[0] == "cross_hole":
            self.gh = gparams[1]
            self.gl = gparams[2]
            self.gw = gparams[3]
            self.gwi = gparams[4]
            self.geometry.append(
                mp.Block(
                    material = self.grat_mat,
                    size = mp.Vector3(self.gh, self.gl*self.sy, self.gw*self.sz),
                    center = mp.Vector3(-0.5*self.sx + self.dpml + self.dsub + 0.5*self.gh, *center),
                )
            )
            self.geometry.append(
                mp.Block(
                    material = self.grat_mat,
                    size = mp.Vector3(self.gh, self.gw*self.sy, self.gl*self.sz),
                    center = mp.Vector3(-0.5*self.sx + self.dpml + self.dsub + 0.5*self.gh, *center),
                )
            )
            self.geometry.append(
                mp.Block(
                    material = mp.Medium(index=1),
                    size = mp.Vector3(self.gh, self.gwi*self.sy, self.gwi*self.sz),
                    center = mp.Vector3(-0.5*self.sx + self.dpml + self.dsub + 0.5*self.gh, *center),
                )
            )
        if gparams[0] == "interior_cross":
            self.gh = gparams[1]
            self.gw = gparams[2]
            self.gli = gparams[3]
            self.gwi = gparams[4]
            self.geometry.append(
                mp.Block(
                    material=self.grat_mat,
                    size=mp.Vector3(self.gh, self.gw*self.sy, self.gw*self.sz),
                    center=mp.Vector3(-0.5*self.sx + self.dpml + self.dsub + 0.5*self.gh, *center),
                )
            )
            self.geometry.append(
                mp.Block(
                    material = mp.Medium(index=1),
                    size = mp.Vector3(self.gh, self.gli*self.sy, self.gwi*self.sz),
                    center = mp.Vector3(-0.5*self.sx + self.dpml + self.dsub + 0.5*self.gh, *center),
                )
            )
            self.geometry.append(
                mp.Block(
                    material = mp.Medium(index=1),
                    size = mp.Vector3(self.gh, self.gwi*self.sy, self.gli*self.sz),
                    center = mp.Vector3(-0.5*self.sx + self.dpml + self.dsub + 0.5*self.gh, *center),
                )
            )
        if gparams[0] == "cylinder":
            self.gh = gparams[1]
            self.gr = gparams[2]
            self.geometry.append(
                mp.Cylinder(
                    radius = self.gr*self.sy/2,
                    height = self.gh,
                    axis = mp.Vector3(1,0,0),
                    center = mp.Vector3(-0.5*self.sx + self.dpml + self.dsub + 0.5*self.gh, *center),
                )
            )
        if gparams[0] == "hollow_cylinder":
            self.gh = gparams[1]
            self.gd = gparams[2]
            self.gdi = gparams[3]
            self.geometry.append(
                mp.Cylinder(
                    material = self.grat_mat,
                    radius = self.gd*self.sy/2,
                    height = self.gh,
                    axis = mp.Vector3(1,0,0),
                    center = mp.Vector3(-0.5*self.sx + self.dpml + self.dsub + 0.5*self.gh, *center),
                )
            )
            self.geometry.append(
                mp.Cylinder(
                    material = mp.Medium(index=1),
                    radius = self.gdi*self.gd*self.sy/2,
                    height = self.gh,
                    axis = mp.Vector3(1,0,0),
                    center = mp.Vector3(-0.5*self.sx + self.dpml + self.dsub + 0.5*self.gh, *center),
                )
            )

    def plotCell(self):
        self.sim = mp.Simulation(
            resolution = self.res,
            cell_size = self.cell_size,
            boundary_layers = self.pml_layers,
            geometry = self.geometry,
            k_point = self.k_point,
            sources = self.sources,
            symmetries = self.symmetries,
        )
        self.sim.plot2D(
            output_plane=mp.Volume(
                center=mp.Vector3(-0.5*self.sx+self.dpml+self.dsub+0.5*self.gh,0,0),
                size=mp.Vector3(0,self.sy,self.sz)
            )
        )

    def calcInputFlux(self):
        self.sim = mp.Simulation(
            resolution = self.res,
            cell_size = self.cell_size,
            boundary_layers = self.pml_layers,
            geometry = self.geometry,
            k_point = self.k_point,
            sources = self.sources,
            symmetries = self.symmetries,
        )

        flux_size = mp.Vector3(0, self.sy*(2*self.num_cells+1), self.sz*(2*self.num_cells+1))

        flux_obj = self.sim.add_flux(
            self.fcen, self.df, self.nfreq, mp.FluxRegion(center=self.mon_pt, size=flux_size)
        )

        self.sim.run(until_after_sources=mp.stop_when_fields_decayed(5,mp.Ez,self.mon_pt,1E-3))

        self.freqs = mp.get_eigenmode_freqs(flux_obj)
        res = self.sim.get_eigenmode_coefficients(
            flux_obj, [1], eig_parity=mp.ODD_Z + mp.EVEN_Y
        )
        coeffs = res.alpha
        self.input_flux = np.abs(coeffs[0,:,0])**2
        self.input_phase = np.angle(coeffs[0,:,0])

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
        fig = plt.figure(dpi=200)
        self.sim.plot2D(
            output_plane=mp.Volume(
                center=mp.Vector3(-0.5*self.sx+self.dpml+self.dsub+0.5*self.gh,0,0),
                size=mp.Vector3(0,self.sy,self.sz)
            )
        )
        # self.sim.plot2D(
        #     output_plane=mp.Volume(
        #         center=mp.Vector3(0,0,0),
        #         size=mp.Vector3(self.sx,self.sy,0)
        #     )
        # )
        ax = plt.gca()
        ax.xaxis.set_tick_params(labelbottom=False)
        ax.yaxis.set_tick_params(labelleft=False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel(r"$y$", fontsize=20)
        ax.set_xlabel(r"$x$", fontsize=20)
        assert 1==0
        
        flux_size = mp.Vector3(0, self.sy*(2*self.num_cells+1), self.sz*(2*self.num_cells+1))
        
        flux_obj = self.sim.add_flux(
            self.fcen, self.df, self.nfreq, mp.FluxRegion(center=self.mon_pt, size=flux_size)
        )

        self.sim.run(until_after_sources=mp.stop_when_fields_decayed(5,mp.Ez,self.mon_pt,1E-3))

        self.freqs = mp.get_eigenmode_freqs(flux_obj)
        res = self.sim.get_eigenmode_coefficients(
            flux_obj, [1], eig_parity=mp.ODD_Z + mp.EVEN_Y
        )
        coeffs = res.alpha
        self.output_flux = np.abs(coeffs[0,:,0])**2
        self.output_phase = np.angle(coeffs[0,:,0])

        self.mode_tran = self.output_flux/self.input_flux
        self.mode_phase = (self.output_phase - self.input_phase)%(2*np.pi)
        
        self.sim.reset_meep()

    def animFields(self, file_path):
        sources = [mp.Source(
                mp.ContinuousSource(wavelength=1),
                component = mp.Ez,
                center = self.src_pt,
                size = mp.Vector3(0, self.sy*(2*self.num_cells+1), self.sz*(2*self.num_cells+1)),
            )
        ]
        self.sim = mp.Simulation(
            resolution = self.res,
            cell_size = self.cell_size,
            boundary_layers = self.pml_layers,
            geometry = self.geometry,
            k_point = self.k_point,
            sources = sources,
            symmetries = self.symmetries,
        )
        f1 = plt.figure(dpi=200)
        f2 = plt.figure(dpi=200)
        anim1 = mp.Animate2D(fields=mp.Ez, f=f1, realtime=False, normalize=False,
            output_plane=mp.Volume(
                center=mp.Vector3(0,0,0),
                size=mp.Vector3(self.sx,self.sy,0)
            )
        )
        anim2 = mp.Animate2D(fields=mp.Ez, f=f2, realtime=False, normalize=False,
            output_plane=mp.Volume(
                center=mp.Vector3(-0.5*self.sx+self.dpml+self.dsub+0.5*self.gh,0,0),
                size=mp.Vector3(0,self.sy,self.sz)
            )
        )
        self.sim.run(mp.at_every(0.1, anim1, anim2), until=50)
        fps = 10
        anim1.to_mp4(fps, file_path+"_parax.mp4")
        anim2.to_mp4(fps, file_path+"_transv.mp4")
        # self.sim.run(until=50)

        self.sim.plot2D(fields=mp.Ez, ax=f1.gca(),
            output_plane=mp.Volume(
                center=mp.Vector3(0,0,0),
                size=mp.Vector3(self.sx,self.sy,0)
            )
        )
        self.sim.plot2D(fields=mp.Ez, ax=f2.gca(),
            output_plane=mp.Volume(
                center=mp.Vector3(-0.5*self.sx+self.dpml+self.dsub+0.5*self.gh,0,0),
                size=mp.Vector3(0,self.sy,self.sz)
            )
        )
        f1.gca().xaxis.set_tick_params(labelbottom=False)
        f1.gca().yaxis.set_tick_params(labelleft=False)
        f1.gca().set_xticks([])
        f1.gca().set_yticks([])
        f1.gca().set_ylabel("")
        f1.gca().set_xlabel(r"$z$")
        f2.gca().xaxis.set_tick_params(labelbottom=False)
        f2.gca().yaxis.set_tick_params(labelleft=False)
        f2.gca().set_xticks([])
        f2.gca().set_yticks([])
        f2.gca().set_ylabel(r"$y$")
        f2.gca().set_xlabel(r"$x$")

        f1.savefig("xy_mode_ex1")
        f2.savefig("yz_mode_ex1")
        
        self.sim.reset_meep()


def init_library(lib_path, cell_size, dpml, dsub, sub_mat, grat_mat, res, wvl_min, wvl_max, nfreq):
    empty_cell = UnitCell3D(cell_size, dpml, dsub, material(sub_mat), material(grat_mat), res, wvl_min, wvl_max, nfreq)
    empty_cell.initGeometry([0])
    empty_cell.calcInputFlux()

    lib_dict = {            # Init library with constant parameters among unit cells
        "cell_size" : list(cell_size),
        "dpml" : dpml,
        "dsub" : dsub,
        "sub_mat" : sub_mat,
        "grat_mat" : grat_mat,
        "res" : res,
        "wvl_min" : wvl_min,
        "wvl_max" : wvl_max,
        "nfreq" : nfreq,
        "mode_wvl" : [1/empty_cell.freqs[i] for i in range(nfreq)],
        "input_flux" : list(empty_cell.input_flux),
        "input_phase" : list(empty_cell.input_phase),
        "cell_types" : []
    }
    save = ""
    while save not in ["Y", "N"]:
        save = input("\nSave data? (Y/N)")
        if save == "Y":
            with open(lib_path, "w") as outfile: 
                json.dump(lib_dict, outfile, indent=2)


def sweep_params(lib_path, gparams, symmetries, dupl_params=False):
    with open(lib_path, "r") as jsonFile:
        lib = json.load(jsonFile)

    with open(lib_path[:-5]+"_BACKUP.json", "w") as jsonFile:
        json.dump(lib, jsonFile, indent=2, default=str)
            
    cell_type = gparams[0]
    if cell_type not in lib["cell_types"]:
        lib["cell_types"].append(cell_type)

    gparam_mesh = np.meshgrid(*gparams[1:])
    gparam_mesh = [np.ndarray.flatten(gparam_mesh[i]) for i in range(len(gparam_mesh))]

    if gparams[0] not in lib.keys():
        lib[gparams[0]] = []        # Init array if grating type not included yet
    ns = len(gparam_mesh[0])

    # try:    
    for i in tqdm(range(ns)):
        gparams_new = [cell_type, *[gparam_mesh[j][i] for j in range(len(gparam_mesh))]]
        print(gparams_new)
        if cell_type in ["hollow_block", "cross", "interior_block", "interior_cross", "window"]:
            if gparams_new[2] <= gparams_new[3]+0.051:
                print("Invalid geometry parameters.")
                print("Next iteration...")
                continue
        if cell_type in ["interior_block", "interior_cross", "window"]:
            if gparams_new[3] <= gparams_new[4]+0.051:
                print("Invalid geometry parameters.")
                print("Next iteration...")
                continue
        gparams_used = [lib[gparams[0]][k][0] for k in range(len(lib[gparams[0]]))]
        if gparams_new not in gparams_used:  # Do not simulate if already simulated
            unit_cell = UnitCell3D(
                lib["cell_size"], lib["dpml"], lib["dsub"], material(lib["sub_mat"]), material(lib["grat_mat"]),
                lib["res"], lib["wvl_min"], lib["wvl_max"], lib["nfreq"], symmetries,
            )
            unit_cell.initGeometry(gparams_new)

            unit_cell.input_flux = lib["input_flux"]
            unit_cell.input_phase = lib["input_phase"]
            # unit_cell.animFields("field_anim_ex1")
            # assert 1==0
            unit_cell.calcOutputFlux()

            unit_cell_params = [gparams_new, list(unit_cell.mode_tran), list(unit_cell.mode_phase)]
            lib[gparams[0]].append(unit_cell_params)

            print("Transmittance: \n", unit_cell.mode_tran)
            print("Phase: \n", unit_cell.mode_phase)
        else:
            print("Parameters already in library: \n", gparams_new)
            print("Next iteration...")

        # save = ""
        # while save not in ["Y", "N"]:
        #     save = input("\nSave data? (Y/N)")
        #     if save == "Y":
        #         with open(lib_path, "w") as jsonFile:
        #             json.dump(lib, jsonFile, indent=2, default=str)
        
    with open(lib_path, "w") as jsonFile:
        json.dump(lib, jsonFile, indent=2, default=str)

    # except:
    #     with open(lib_path, "w") as jsonFile:
    #         json.dump(lib, jsonFile, indent=2, default=str)
    #     print("Sweep aborted, data saved.")
    #     pass
        
    return lib[gparams[0]][-ns:]


def material(name):
    if name == "cSi":
        from meep.materials import cSi
        return cSi
    if name == "SiO2":
        from meep.materials import SiO2
        return SiO2
    if name == "SiN":
        # from meep.materials import SiN
        SiN = mp.Medium(index=2)
        return SiN
    if name == "TiO2":
        # TiO2_frq1 = 3.54397200e+07
        # TiO2_gam1 = 3.48553574e+03
        # TiO2_sig1 = 0
        # TiO2_susc = [mp.LorentzianSusceptibility(frequency=TiO2_frq1, gamma=TiO2_gam1, sigma=TiO2_sig1)]
        # TiO2 = mp.Medium(epsilon=3.0, E_susceptibilities=TiO2_susc)
        TiO2 = mp.Medium(index=2.15)
        return TiO2
    if name == "aSi":
        # aSi = mp.Medium(index=4.0777, D_conductivity=2*np.pi*0.550*0.005/4.0777)
        # aSi = mp.Medium(index=4.17, D_conductivity=2*np.pi*0.565*0.36/4.17)
        # aSi = mp.Medium(index=3.5)
        data = np.loadtxt("aSi_lorentzfit_NIR.txt")
        eps_inf = 13.3
        E_susceptibilities = []
        for n in range(len(data)//3):
            mymaterial_freq = data[3 * n + 1]
            mymaterial_gamma = data[3 * n + 2]

            if mymaterial_freq == 0:
                mymaterial_sigma = data[3 * n + 0]
                E_susceptibilities.append(
                    mp.DrudeSusceptibility(
                        frequency=1.0, gamma=mymaterial_gamma, sigma=mymaterial_sigma
                    )
                )
            else:
                mymaterial_sigma = data[3 * n + 0] / mymaterial_freq**2
                E_susceptibilities.append(
                    mp.LorentzianSusceptibility(
                        frequency=mymaterial_freq,
                        gamma=mymaterial_gamma,
                        sigma=mymaterial_sigma,
                    )
                )
        aSi = mp.Medium(epsilon=eps_inf, E_susceptibilities=E_susceptibilities)
        return aSi
    if name == "aSi:H":
        # https://www.mdpi.com/2079-9292/12/13/2953
        n = 3.247
        k = 0.0471
        w0 = 2*np.pi*1.964
        return mp.Medium(index=n, D_conductivity=w0*k/n)
    if name == "glass":
        return mp.Medium(index=1.5)


def pb_phase_tran(lib_path, pb_lib_path):
    with open(lib_path, "r") as jsonFile:
        lib = json.load(jsonFile)

    params = lib["rectang"]
    fig, ax = plt.subplots(2, sharex=True)
    ax[0].set_ylabel("PB Tran.")
    ax[1].set_ylabel("PB Phase")
    ax[1].set_xlabel("Freq. (1/um)")
    for i in range(len(params)):
        A0 = np.clip(np.array(params[i][1]), 0, 1)
        phi0 = np.array(params[i][2])
        t0 = A0*np.exp(1j*phi0)

        li = params[i][0][2]
        wi = params[i][0][3]
        for j in range(len(params)):
            lj = params[j][0][2]
            wj = params[j][0][3]
            if (li, wi) == (wj, lj):
                A1 = np.clip(np.array(params[j][1]), 0, 1)
                phi1 = np.array(params[j][2])
                t1 = A1*np.exp(1j*phi1)
        sign = 2*(wi >= wj) - 1
        pb_phase = np.angle((t0-t1)*sign)
        pb_tran = np.abs((t0-t1)/2)**2
        
        print(pb_tran)
        ax[0].plot(1/np.array(lib["mode_wvl"]), pb_tran)
        ax[1].plot(1/np.array(lib["mode_wvl"]), pb_phase)

        lib["rectang"][i].append(list(pb_tran))
        lib["rectang"][i].append(list(pb_phase))
    plt.savefig("PB_phase_tran.png")

    with open(pb_lib_path, "w") as outfile: 
        json.dump(lib, outfile, indent=2)

#%%
# INIT LIBRARY
if __name__ == "__main__":
    # Simulation spectrum
    wvl_min = 0.8  # min wavelength
    wvl_max = 1.6  # max wavelength
    nfreq = 41  # number of frequency bins
    k_point = mp.Vector3(0, 0, 0)

    # Simulation geometry
    dpml = 1.0  # PML thickness
    dsub = 1.0  # substrate thickness
    dpad = 3.0  # padding between substrate and PML (>grating height)
    gp = 0.5  # grating periodicity
    sub_mat = "glass"  # substrate material
    grat_mat = "aSi"  # grating material
    cell_size = mp.Vector3(dpml+dsub+dpad+dpml, gp, gp)
    res = 50  # resolution
    symmetries = []
    lib_path = "lib_NIR_aSi_TEST.json"

    # init_library(lib_path, cell_size, dpml, dsub, sub_mat, grat_mat, res, wvl_min, wvl_max, nfreq)
    
    #%%
    # SWEEP GEOMETRY PARAMETERS
    for i in range(1):
        gh = [0.7] 
        gw = np.linspace(0.1, 1.0, 19)[15]
        gwi = np.linspace(0.1, 1.0, 19)[8]
        gwii = np.linspace(0.1, 1.0, 19)[4]
        gparams = ['window', gh, gw, gwi, gwii]
        # ['window', 0.7, 0.35, 0.2, 0.1]
        sweep = sweep_params(lib_path, ['window', 0.7, 0.75, 0.55, 0.45000000000000007], symmetries)
    
    #%%
    

    #%%

    # lib_path_save = "lib_NIR_aSi_merge.json"
    # lib_path_load = []
    # for i in range(14):
    #     lib_path_load.append(f"lib_NIR_aSi_IBLOCK{i}.json")
    #     lib_path_load.append(f"lib_NIR_aSi_ICROSS{i}.json")
    #     lib_path_load.append(f"lib_NIR_aSi_WINDOW{i}.json")
    #     lib_path_load.append(f"lib_NIR_aSi_CHOLE{i}.json")
    # for i in range(3):
    #     lib_path_load.append(f"lib_NIR_aSi_CROSS{i}.json")
    #     lib_path_load.append(f"lib_NIR_aSi_HBLOCK{i}.json")
    #     lib_path_load.append(f"lib_NIR_aSi_FBLOCKS{i}.json")

    # def merge_libraries(lib_path_save, lib_path_load):
    #     with open(lib_path_load[0], "r") as jsonFile:
    #         lib_save = json.load(jsonFile)
    #     for i in range(len(lib_path_load)):
    #         with open(lib_path_load[i], "r") as jsonFile:
    #             lib_load = json.load(jsonFile)
    #         for cell_type in lib_load["cell_types"]:
    #             if cell_type not in lib_save["cell_types"]:
    #                 lib_save["cell_types"].append(cell_type)
    #                 lib_save[cell_type] = []
    #             for j in range(len(lib_load[cell_type])):
    #                 gparams_new = lib_load[cell_type][j][0]
    #                 gparams_used = [lib_save[cell_type][k][0] for k in range(len(lib_save[cell_type]))]
    #                 if gparams_new not in gparams_used:
    #                     lib_save[cell_type].append(lib_load[cell_type][j])
    #     for i in range(len(lib_save["cell_types"])):
    #         print(lib_save["cell_types"][i], len(lib_save[lib_save["cell_types"][i]]))

    #     with open(lib_path_save, "w") as outfile: 
    #         json.dump(lib_save, outfile, indent=2)

    # merge_libraries(lib_path_save, lib_path_load)
                



    #%%
    # PLOT PHASE AT 2 WVLS
    with open(lib_path_save, "r") as jsonFile:
        lib = json.load(jsonFile)

    i1 = np.argmin(np.abs(0.8 - np.array(lib["mode_wvl"])))
    i2 = np.argmin(np.abs(1.55 - np.array(lib["mode_wvl"])))
    trans_thresh = 0.9
    plt.figure(dpi=100)
    sum = 0

    # x_pts = []
    # y_pts = []
    # for i in range(len(lib["block"])):
    #     if lib["block"][i][1][i1] > trans_thresh and lib["block"][i][1][i2] > trans_thresh:
    #         x_pts.append(lib["block"][i][2][i1])
    #         y_pts.append(lib["block"][i][2][i2])
    # plt.scatter(x_pts, y_pts, c="b", label="Block", s=2)

    x_pts = []
    y_pts = []
    for i in range(len(lib["hollow_block"])):
        if lib["hollow_block"][i][1][i1] > trans_thresh and lib["hollow_block"][i][1][i2] > trans_thresh:
            x_pts.append(lib["hollow_block"][i][2][i1])
            y_pts.append(lib["hollow_block"][i][2][i2])
    plt.scatter(x_pts, y_pts, c="g", label="Hollow block", s=2)
    sum += len(x_pts)

    x_pts = []
    y_pts = []
    for i in range(len(lib["cross"])):
        if lib["cross"][i][1][i1] > trans_thresh and lib["cross"][i][1][i2] > trans_thresh:
            x_pts.append(lib["cross"][i][2][i1])
            y_pts.append(lib["cross"][i][2][i2])
    plt.scatter(x_pts, y_pts, c="purple", label="Cross", s=2)
    sum += len(x_pts)

    x_pts = []
    y_pts = []
    for i in range(len(lib["interior_block"])):
        if lib["interior_block"][i][1][i1] > trans_thresh  and lib["interior_block"][i][1][i2] > trans_thresh:
            x_pts.append(lib["interior_block"][i][2][i1])
            y_pts.append(lib["interior_block"][i][2][i2])
    plt.scatter(x_pts, y_pts, c="orange", label="Interior block", s=2)
    sum += len(x_pts)

    x_pts = []
    y_pts = []
    for i in range(len(lib["interior_cross"])):
        if lib["interior_cross"][i][1][i1] > trans_thresh  and lib["interior_cross"][i][1][i2] > trans_thresh:
            x_pts.append(lib["interior_cross"][i][2][i1])
            y_pts.append(lib["interior_cross"][i][2][i2])
    plt.scatter(x_pts, y_pts, c="red", label="Interior cross", s=2)
    sum += len(x_pts)

    x_pts = []
    y_pts = []
    for i in range(len(lib["window"])):
        if lib["window"][i][1][i1] > trans_thresh  and lib["window"][i][1][i2] > trans_thresh:
            x_pts.append(lib["window"][i][2][i1])
            y_pts.append(lib["window"][i][2][i2])
    plt.scatter(x_pts, y_pts, c="black", label="Window", s=2)
    sum += len(x_pts)

    x_pts = []
    y_pts = []
    for i in range(len(lib["cross_hole"])):
        if lib["cross_hole"][i][1][i1] > trans_thresh  and lib["cross_hole"][i][1][i2] > trans_thresh:
            x_pts.append(lib["cross_hole"][i][2][i1])
            y_pts.append(lib["cross_hole"][i][2][i2])
    plt.scatter(x_pts, y_pts, c="brown", label="cross_hole", s=2)

    x_pts = []
    y_pts = []
    for i in range(len(lib["four_blocks"])):
        if lib["four_blocks"][i][1][i1] > trans_thresh  and lib["four_blocks"][i][1][i2] > trans_thresh:
            x_pts.append(lib["four_blocks"][i][2][i1])
            y_pts.append(lib["four_blocks"][i][2][i2])
    plt.scatter(x_pts, y_pts, c="blue", label="four_blocks", s=2)
    sum += len(x_pts)

    plt.xlim([0, 2*np.pi])
    plt.ylim([0, 2*np.pi])
    # plt.legend()
    wvl1 = round(lib["mode_wvl"][i1], 2)
    wvl2 = round(lib["mode_wvl"][i2], 2)
    plt.xlabel(f"Phase at {wvl1}"+r"$\mu$m wavelength")
    plt.ylabel(f"Phase at {wvl2}"+r"$\mu$m wavelength")
    # plt.savefig("SiN_phase.png")


    #%%
    # DISPERSION FITTING
    lib_path = "lib_NIR_aSi_merge_FILTERED.json"
    with open(lib_path, "r") as jsonFile:
        lib = json.load(jsonFile)

    from scipy.optimize import curve_fit

    def poly_mod2pi(x, c0, c1):
        return np.array(c0 + c1*x)%(2*np.pi)

    phases = []
    gd = []
    losses = []
    trans = []
    for cell_type in lib["cell_types"]:
        for j in tqdm(range(len(lib[cell_type]))):
            # cell_type = "window"
            # j = 200
            freq = 1/np.array(lib["mode_wvl"])
            tran = lib[cell_type][j][1]
            phase = lib[cell_type][j][2]
            # plt.figure(dpi=200)
            # lns1 = plt.plot(freq, phase, c="b", label="Phase")
            # ax1 = plt.gca()
            # ax1.grid()
            # ax1.set_xlabel(r"Frequency (c/$\mu$m)")
            # ax1.set_ylabel("Phase")
            # ax2 = ax1.twinx()
            # lns2 = ax2.plot(freq, tran, "--", c="r", label="Simulated tran.")
            # lns3 = ax2.plot(freq, np.clip(tran, 0, 1), c="r", label="Corrected tran.")
            # ax2.grid(linestyle="--")
            # ax2.set_ylabel("Transmission")
            # lns = lns1+lns2+lns3
            # labs = [l.get_label() for l in lns]
            # plt.savefig("phase_tran_ex2")
            # plt.show()
            # print(lib[cell_type][j][0])
            # assert 1==0
            loss = 1E100
            loss_thresh = 2.7
            i=0
            while  i < 100:
                i+=1
                p0 = np.random.rand(2)*(2*np.pi, 20*np.pi)
                p_new = curve_fit(poly_mod2pi, freq, phase, p0)[0]
                resid = phase - poly_mod2pi(freq, *p_new)
                m = np.abs(resid) > np.pi
                resid = (1 - 2*m)*np.abs(resid) + m*2*np.pi
                loss_new = np.sqrt(np.sum(resid**2))
                if loss_new < loss:
                    p_fit = p_new
                    loss = loss_new
            phases.append(poly_mod2pi(freq, *p_fit))
            trans.append(tran)
            gd.append(p_fit[1])
            losses.append(loss)
    assert 1==0

    def focus_phase(x, y, focl, wvl_cen):
        phi_foc = 2*np.pi/wvl_cen * (focl - np.sqrt(focl**2 + x**2 + y**2))
        return phi_foc % (2*np.pi)

    def group_delay(x, y, focl):
        gd = 2*np.pi*(focl - np.sqrt(focl**2 + x**2 + y**2))
        return gd
    
    print(p_fit)
    phase_fit = poly_mod2pi(freq, *p_fit)

    plt.figure()
    plt.plot(freq, tran, "r")
    plt.xlabel("Frequency (1/um)")
    plt.ylabel("Transmittance")
    plt.figure()
    plt.plot(freq, phase, "b")
    plt.plot(freq, phase_fit, "b--")
    plt.xlabel("Frequency (1/um)")
    plt.ylabel("Phase")

    #%%


    # with open(lib_path, "r") as jsonFile:
    #     lib = json.load(jsonFile)
    # unit_cell = UnitCell3D(
    #     lib["cell_size"], lib["dpml"], lib["dsub"], material(lib["sub_mat"]), material(lib["grat_mat"]),
    #     lib["res"], lib["wvl_min"], lib["wvl_max"], lib["nfreq"], symmetries,
    # )
    # unit_cell.initGeometry(gvar_grid[138][35])
    # unit_cell.plotCell()
    # assert 1==0

    #%%
    def radial_grid(a, nr):
        grid_pts = []
        grid_pts.append((0, 0))
        r = 0
        for i in range(nr):
            npts = 6*(i+1)
            theta = 2*np.pi/(npts)
            r = a/(2*np.sin(theta/2))
            for j in range(npts):
                x = r*np.cos(j*theta)
                y = r*np.sin(j*theta)
                grid_pts.append((x, y))
        return grid_pts, r

    grid_pts = radial_grid(1, 20)[0]
    for pt in grid_pts:
        # dists = []
        # pts = grid_pts.copy()
        # pts.remove(pt)
        # for pt2 in pts:
        #     dist = np.sqrt((pt[0]-pt2[0])**2 + (pt[1]-pt2[1])**2)
        #     dists.append(np.min(dist))
        # print(np.min(dists))
        plt.scatter(pt[0], pt[1], c="b")

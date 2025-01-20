#%%
import meep as mp
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate
from meep.materials import SiO2
import json
import sys

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
        self.src_pt = mp.Vector3(-0.5 * self.sx + dpml + 0.5*dsub, 0, 0)
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
                    size=mp.Vector3(self.dpml + self.dsub, mp.inf, mp.inf),
                    center=mp.Vector3(-0.5*self.sx + 0.5*(self.dpml + self.dsub), 0, 0),
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
        # self.sim.plot2D(
        #     output_plane=mp.Volume(
        #         center=mp.Vector3(-0.5*self.sx+self.dpml+self.dsub+0.5*self.gh,0,0),
        #         size=mp.Vector3(0,self.sy,self.sz)
        #     )
        # )
        # assert 1==0
        
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


def init_library(lib_path, cell_size, dpml, dsub, sub_mat, grat_mat, res, wvl_min, wvl_max, nfreq, save=""):
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
    if save == "Y":
        with open(lib_path, "w") as outfile: 
            json.dump(lib_dict, outfile, indent=2)
    while save not in ["Y", "N"]:
        save = input("\nSave data? (Y/N)")
        if save == "Y":
            with open(lib_path, "w") as outfile: 
                json.dump(lib_dict, outfile, indent=2)


def sweep_params(lib_path, gparam_list, symmetries, dupl_params=False):
    with open(lib_path, "r") as jsonFile:
        lib = json.load(jsonFile)
    with open(lib_path[:-5]+"_BACKUP.json", "w") as jsonFile:
        json.dump(lib, jsonFile, indent=2, default=str)
            
    cell_type = gparam_list[0][0]
    if cell_type not in lib.keys():
        lib["cell_types"].append(cell_type)
        lib[cell_type] = [] 

    ng = len(gparam_list)
    try:    
        for i in (range(ng)):
            if i==5:
                np.savetxt(f"test_{sweep_idx}.txt", np.array([sweep_idx]))
            gparams_new = gparam_list[i]
            print(f"Iteration: {i+1}/{ng}")
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
            gparams_used = [lib[cell_type][k][0] for k in range(len(lib[cell_type]))]
            if gparams_new not in gparams_used:  # Do not simulate if already simulated
                unit_cell = UnitCell3D(
                    lib["cell_size"], lib["dpml"], lib["dsub"], material(lib["sub_mat"]), material(lib["grat_mat"]),
                    lib["res"], lib["wvl_min"], lib["wvl_max"], lib["nfreq"], symmetries,
                )
                unit_cell.initGeometry(gparams_new)

                unit_cell.input_flux = lib["input_flux"]
                unit_cell.input_phase = lib["input_phase"]
                unit_cell.calcOutputFlux()

                unit_cell_params = [gparams_new, list(unit_cell.mode_tran), list(unit_cell.mode_phase)]
                lib[cell_type].append(unit_cell_params)

                print("Transmittance: \n", unit_cell.mode_tran)
                print("Phase: \n", unit_cell.mode_phase)
            else:
                print("Parameters already in library: \n", gparams_new)
                print("Next iteration...")
                
        with open(lib_path, "w") as jsonFile:
            json.dump(lib, jsonFile, indent=2, default=str)

    except:
        with open(lib_path, "w") as jsonFile:
            json.dump(lib, jsonFile, indent=2, default=str)
        print("Sweep aborted, data saved.")
        pass
        
    return lib[gparams[0]][-ng:]


def param_grid(gparams):
    cell_type = gparams[0]
    gparam_mesh = np.meshgrid(*gparams[1:])
    gparam_mesh = [np.ndarray.flatten(gparam_mesh[i]) for i in range(len(gparam_mesh))]
    gparam_list = []

    for i in (range(len(gparam_mesh[0]))):
        gparams_new = [cell_type, *[gparam_mesh[j][i] for j in range(len(gparam_mesh))]]
        if len(gparams_new[2:]) in [2, 3]:
            if gparams_new[2] <= gparams_new[3]+0.051:
                continue
        if len(gparams_new[2:]) == 3:
            if gparams_new[3] <= gparams_new[4]+0.051:
                continue
        gparam_list.append(gparams_new)
    return gparam_list


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

# INIT LIBRARY
if __name__ == "__main__":
    sweep_idx = int(sys.argv[1])

    # Simulation spectrum
    wvl_min = 0.8  # min wavelength
    wvl_max = 1.6  # max wavelength
    nfreq = 41  # number of frequency bins
    k_point = mp.Vector3(0, 0, 0)

    # Simulation geometry
    dpml = 1.0  # PML thickness
    dsub = 1.0  # substrate thickness
    dpad = 3.0  # padding between substrate and PML (>grating height)
    gp = 0.6  # grating periodicity
    sub_mat = "glass"  # substrate material
    grat_mat = "aSi"  # grating material
    cell_size = mp.Vector3(dpml+dsub+dpad+dpml, gp, gp)
    res = 50  # resolution
    symmetries = []
    lib_path = f"lib_NIR_aSi_ICROSS{sweep_idx}.json"

    init_library(lib_path, cell_size, dpml, dsub, sub_mat, grat_mat, res, wvl_min, wvl_max, nfreq, save="Y")

    # SWEEP GEOMETRY PARAMETERS
    gh = [0.7] 
    gw = np.linspace(0.1, 1.0, 19)
    gwi = np.linspace(0.1, 1.0, 19)
    gwii = np.linspace(0.1, 1.0, 19)
    gparams = ["interior_cross", gh, gw, gwi, gwii]

    gparam_list = param_grid(gparams)
    ns = 50
    nr = round(len(gparam_list)/ns)
    if sweep_idx == nr - 1:
        sweep = sweep_params(lib_path, gparam_list[sweep_idx*ns:], symmetries)
    else:
        sweep = sweep_params(lib_path, gparam_list[sweep_idx*ns:(sweep_idx+1)*ns], symmetries)
    
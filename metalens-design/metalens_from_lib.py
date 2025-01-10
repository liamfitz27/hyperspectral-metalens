#%%
import meep as mp
import numpy as np
import matplotlib.pyplot as plt
from unit_cell import UnitCell3D, material
from psf_model import rs_psf_fftconv, focus_phase
from scipy import interpolate
from tqdm import tqdm
import json

class Metalens(UnitCell3D):
    def __init__(self, n_cells, focal_length, gparam_grid, *args):
        super().__init__(*args)
        
        self.n_cells = n_cells
        self.lens_size = mp.Vector3(self.cell_size[0], (2*n_cells+1)*self.cell_size[1], (2*n_cells+1)*self.cell_size[2])
        self.lx = self.lens_size[0]
        self.ly = self.lens_size[1]
        self.lz = self.lens_size[2]
        self.focal_length = focal_length
        self.gparam_grid = gparam_grid

    def initLensGeometry(self):
        for i in range(2*self.n_cells + 1):
            for j in range(2*self.n_cells + 1):
                center = ((i - self.n_cells)*self.sy, (j - self.n_cells)*self.sz)
                gparams = gparam_grid[i][j]
                self.initGeometry(gparams, center=center)

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
        plt.figure(dpi=600)
        self.sim.plot2D(
            output_plane=mp.Volume(
                center=mp.Vector3(-0.5*self.lx+self.dpml+self.dsub+0.5*self.gh,0,0),
                size=mp.Vector3(0,self.ly,self.lz)
            )
        )


def filter_library(lib_path, lib_path_new, size_min, wvl_idx, tran_min, tran_max=1):
    with open(lib_path, "r") as jsonFile:
        lib = json.load(jsonFile)

    lib_new = lib.copy()
    if "tran_range" in lib.keys():
        lib_new["tran_range"].append((tran_min, tran_max))
    else:
        lib_new["tran_range"] = (tran_min, tran_max)

    for t in lib["cell_types"]:
        lib_new[t] = []
        for i in range(len(lib[t])):
            # Filter tran and min size
            tran = np.clip(lib[t][i][1][wvl_idx], 0, 1)
            gparams = lib[t][i][0]
            sizes = gparams[2:]
            edge_size = (1 - gparams[2])*lib["cell_size"][1]
            sizes.append(edge_size)
            if len(gparams[2:]) in [2, 3]:
                int_size1 = 0.5*(gparams[2] - gparams[3])*lib["cell_size"][1]
                sizes.append(int_size1)
            if len(gparams[2:]) == 3:
                int_size2 = 0.5*(gparams[3] - gparams[4])*lib["cell_size"][1]
                sizes.append(int_size2)
            if all(sizes) >= size_min and (tran >= tran_min and tran <= tran_max):
                lib_new[t].append(lib[t][i])

    with open(lib_path_new, "w") as jsonFile:
        json.dump(lib_new, jsonFile, indent=2, default=str)
    return lib_new


def build_lens_from_lib(lib_path, c_targ, wvl_idx):
    with open(lib_path, "r") as jsonFile:
        lib = json.load(jsonFile)
    geoms = []
    c_use = []
    c_targ_f = np.reshape(c_targ, (np.shape(c_targ)[0]**2, np.shape(c_targ)[-1]))
    for l in range(len(wvl_idx)):
        tran = []
        phase = []
        for t in lib["cell_types"]:
            for i in range(len(lib[t])):
                geoms.append(lib[t][i][0])
                tran.append(lib[t][i][1][wvl_idx[l]])
                phase.append(-lib[t][i][2][wvl_idx[l]])
        geoms.append(["empty"])
        tran.append(1)
        phase.append(0)
        geoms.append(["opaque"])
        tran.append(0)
        phase.append(0)
        tran = np.array(tran)
        phase = np.array(phase)
        c_use.append(np.clip(tran, 0, 1)*np.exp(1j*phase))
    c_use = np.transpose(c_use)
    c_opt_f = np.zeros(np.shape(c_targ_f), dtype="complex")
    geoms_opt = []
    for i in range(np.shape(c_targ_f)[0]):
        idx_opt = np.argmin(np.sum(np.abs(c_use - c_targ_f[i,:])**2, axis=1))
        c_opt_f[i,:] = c_use[idx_opt,:]
        geoms_opt.append(geoms[idx_opt])
    c_opt = np.reshape(c_opt_f, np.shape(c_targ))
    nx = np.shape(c_opt)[0]
    gparam_grid = []
    for i in range(nx):
        row = []
        for j in range(nx):
            row.append(geoms_opt[i*(nx)+j])
        gparam_grid.append(row)
    return gparam_grid, c_opt 

#%%
if __name__ == "__main__":
    ## INIT PARAMETERS ##
    wvl_cen = 1.1
    focl = 150
    sx = 50
    res = 2
    x = np.arange(0, sx + 1/res, 1/res)
    n_cells = len(x[1:])
    x = np.append(-np.flip(x), x[1:])
    xx, yy = np.meshgrid(x, x)
    c_targ = focus_phase(x, x, focl, [wvl_cen])

    psf = rs_psf_fftconv(x, x, [focl], [wvl_cen], c_targ)
    plt.imshow(np.abs(psf[:,:,0,0])**2)

    #%%
    ## FILTER LIBRARY FOR MINIMUM FEATURE SIZE ##
    lib_path = "lib_NIR_aSi_merge.json"
    with open(lib_path, "r") as jsonFile:
        lib = json.load(jsonFile)
    size_min = 0.06  # 60nm
    wvl_idx = np.argmin(np.abs(wvl_cen/ - np.array(lib["mode_wvl"])))
    tran_min = 0.0
    lib_path_new = lib_path[:-5] + "_FILTERED.json"
    libf = filter_library(lib_path, lib_path_new, size_min, wvl_idx, tran_min)

    #%%
    ## DESIGN CHROMATIC LENS ##
    lib_path = "lib_NIR_aSi_merge_FILTERED.json"
    with open(lib_path, "r") as jsonFile:
        libf = json.load(jsonFile)

    wvl_idx = np.argmin(np.abs(np.array(libf["mode_wvl"]) - wvl_cen))
    wvl_use = libf["mode_wvl"][wvl_idx]
    c_targ = focus_phase(x, x, focl, [wvl_use])
    gparam_grid, c_opt = build_lens_from_lib(lib_path, c_targ, [wvl_idx])

    lens = Metalens(
        n_cells, focl, gparam_grid,
        libf["cell_size"], libf["dpml"], libf["dsub"], material(libf["sub_mat"]), material(libf["grat_mat"]),
        libf["res"], libf["wvl_min"], libf["wvl_max"], libf["nfreq"]
    )

    # lens.gh = libf[libf["cell_types"][0]][0][0][1]
    # lens.initLensGeometry()
    # lens.plotLensProfile()
    # plt.savefig("chromatic_lens.pdf")

    #%%
    ## DESIGN ACHROMATIC LENS ##
    lib_path = "lib_NIR_aSi_merge_FILTERED.json"
    with open(lib_path, "r") as jsonFile:
        libf = json.load(jsonFile)

    wf = 5
    wvl_idx = np.argmin(np.abs(np.array(libf["mode_wvl"]) - wvl_cen))
    wvl_use = libf["mode_wvl"][wvl_idx - wf:wvl_idx + wf + 1]
    wvl_idx = np.arange(wvl_idx-wf, wvl_idx+wf+1)
    c_targ = focus_phase(x, x, focl, wvl_use)
    gparam_grid, c_opt = build_lens_from_lib(lib_path, c_targ, wvl_idx)

    lens = Metalens(
        n_cells, focl, gparam_grid,
        libf["cell_size"], libf["dpml"], libf["dsub"], material(libf["sub_mat"]), material(libf["grat_mat"]),
        libf["res"], libf["wvl_min"], libf["wvl_max"], libf["nfreq"]
    )

    lens.gh = libf[libf["cell_types"][0]][0][0][1]
    lens.initLensGeometry()
    lens.plotLensProfile()
    plt.savefig("achromatic_lens.pdf")

    #%%
    from psf_model import rs_psf_fftconv
    i = len(wvl_idx)//2
    c = c_opt[:,:,i]
    psf = np.abs(rs_psf_fftconv(x, x, [focl], wvl_use, c_opt)[:,:,0,:])**2
    plt.figure()
    plt.title("Optim. phase")
    plt.imshow(np.angle(c)%(-2*np.pi))
    plt.colorbar()
    plt.figure()
    plt.title("Optim. tran")
    plt.imshow(np.abs(c))
    plt.colorbar()
    plt.figure()
    plt.title("Targ. phase")
    phase_targ = np.angle(c_targ[:,:,i])%(-2*np.pi)
    plt.imshow(phase_targ)
    plt.figure()
    plt.title("PSF")
    plt.imshow(psf[:,:,i], vmin=np.min(psf), vmax=np.max(psf))


    #%%
    # PLOT PHASE RANGES
    lib = libf

    trans_thresh = 0.9

    plt.figure(dpi=200)

    # xpts = []
    # ypts = []
    # for i in range(len(lib["block"])):
    #     for j in range(len(lib["mode_wvl"])):
    #         if lib["block"][i][1][j] > trans_thresh:
    #             xpts.append(1/lib["mode_wvl"][j])
    #             ypts.append(lib["block"][i][2][j])
    # plt.scatter(xpts, ypts, c="b", label="Block", s=0.5)

    cell_types_list = ["hollow_block", "cross", "four_blocks", "interior_block", "interior_cross", "window", "cross_hole"]
    cell_names_list = ["A1", "A2", "A3", "B1", "B2", "B3", "B4"]

    for it in range(len(cell_types_list)):
        cell_type = cell_types_list[it]
        xpts = []
        ypts = []
        for i in range(len(lib[cell_type])):
            for j in range(len(lib["mode_wvl"])):
                if lib[cell_type][i][1][j] > trans_thresh:
                    xpts.append(1/lib["mode_wvl"][j])
                    ypts.append(lib[cell_type][i][2][j])
        plt.scatter(xpts, ypts, c=f"C{it}", label=cell_names_list[it], s=1, zorder=-10*i)

    # xpts = []
    # ypts = []
    # for i in range(len(lib["cross"])):
    #     for j in range(len(lib["mode_wvl"])):
    #         if lib["cross"][i][1][j] > trans_thresh:
    #             xpts.append(1/lib["mode_wvl"][j])
    #             ypts.append(lib["cross"][i][2][j])
    # plt.scatter(xpts, ypts, c="purple", label="Cross", s=0.5)

    # xpts = []
    # ypts = []
    # for i in range(len(lib["interior_block"])):
    #     for j in range(len(lib["mode_wvl"])):
    #         if lib["interior_block"][i][1][j] > trans_thresh:
    #             xpts.append(1/lib["mode_wvl"][j])
    #             ypts.append(lib["interior_block"][i][2][j])
    # plt.scatter(xpts, ypts, c="orange", label="Interior block", s=0.5)

    plt.legend(loc=3)
    plt.xlabel(r"Frequency (c/$\mu$m)")
    plt.ylabel("Phase")
    plt.savefig("phase_range_filtered")
    

# %%

#%%
import meep as mp
import numpy as np
import matplotlib.pyplot as plt

import meep as mp
import numpy as np
import matplotlib.pyplot as plt

resolution = 50  # pixels/μm

dpml = 1.0  # PML thickness
dsub = 2.0  # substrate thickness
dpad = 2.0  # padding between grating and PML

lcen = 0.5  # center wavelength
fcen = 1 / lcen  # center frequency
df = 0.2 * fcen  # frequency width

focal_length = 200  # focal length of metalens
spot_length = 100  # far field line length
ff_res = 10  # far field resolution (points/μm)

k_point = mp.Vector3(0, 0, 0)

glass = mp.Medium(index=1.5)

pml_layers = [mp.PML(thickness=dpml, direction=mp.X)]

symmetries = [mp.Mirror(mp.Y)]


def grating(gp, gh, gdc_list):
    sx = dpml + dsub + gh + dpad + dpml
    src_pt = mp.Vector3(-0.5 * sx + dpml + 0.5 * dsub)
    mon_pt = mp.Vector3(0.5 * sx - dpml - 0.5 * dpad)
    geometry = [
        mp.Block(
            material=glass,
            size=mp.Vector3(dpml + dsub, mp.inf, mp.inf),
            center=mp.Vector3(-0.5 * sx + 0.5 * (dpml + dsub)),
        )
    ]

    num_cells = len(gdc_list)
    if num_cells == 1:
        sy = gp
        sz = sy
        cell_size = mp.Vector3(sx, sy, sz)

        sources = [
            mp.Source(
                mp.GaussianSource(fcen, fwidth=df),
                component=mp.Ez,
                center=src_pt,
                size=mp.Vector3(y=sy,z=sz),
            )
        ]

        sim = mp.Simulation(
            resolution=resolution,
            cell_size=cell_size,
            boundary_layers=pml_layers,
            k_point=k_point,
            default_material=glass,
            sources=sources,
            symmetries=symmetries,
        )

        flux_obj = sim.add_flux(
            fcen, 0, 1, mp.FluxRegion(center=mon_pt, size=mp.Vector3(y=sy,z=sz))
        )

        sim.run(until_after_sources=50)

        input_flux = mp.get_fluxes(flux_obj)

        sim.reset_meep()

        geometry.append(
            mp.Block(
                material=glass,
                size=mp.Vector3(gh, gdc_list[0] * gp, gdc_list[0] * gp),
                center=mp.Vector3(-0.5 * sx + dpml + dsub + 0.5 * gh),
            )
        )
        # geometry.append(
        #     mp.Cylinder(
        #         material=glass,
        #         radius=gdc_list[0],
        #         axis=mp.Vector3(1, 0, 0),
        #         center=mp.Vector3(-0.5 * sx + dpml + dsub + 0.5 * gh),
        #     )
        # )
        
        sim = mp.Simulation(
            resolution=resolution,
            cell_size=cell_size,
            boundary_layers=pml_layers,
            geometry=geometry,
            k_point=k_point,
            sources=sources,
            symmetries=symmetries,
        )

        flux_obj = sim.add_flux(
            fcen, 0, 1, mp.FluxRegion(center=mon_pt, size=mp.Vector3(y=sy,z=sz))
        )

        sim.run(until_after_sources=200)

        freqs = mp.get_eigenmode_freqs(flux_obj)
        res = sim.get_eigenmode_coefficients(
            flux_obj, [1], eig_parity=mp.ODD_Z + mp.EVEN_Y
        )
        coeffs = res.alpha

        mode_tran = abs(coeffs[0, 0, 0]) ** 2 / input_flux[0]
        mode_phase = np.angle(coeffs[0, 0, 0])
        if mode_phase > 0:
            mode_phase -= 2 * np.pi

        return mode_tran, mode_phase

    else:
        sy = num_cells * gp
        sz = sy
        cell_size = mp.Vector3(sx, sy, sz)

        sources = [
            mp.Source(
                mp.GaussianSource(fcen, fwidth=df),
                component=mp.Ez,
                center=src_pt,
                size=mp.Vector3(y=sy,z=sz),
            )
        ]

        for j in range(num_cells):
            geometry.append(
                mp.Block(
                    material=glass,
                    size=mp.Vector3(gh, gdc_list[j] * gp, gdc_list[j] * gp),
                    center=mp.Vector3(
                        -0.5 * sx + dpml + dsub + 0.5 * gh, 
                        -0.5 * sy + (j + 0.5) * gp,
                        -0.5 * sz + (j + 0.5) * gp
                    ),
                )
            )

        sim = mp.Simulation(
            resolution=resolution,
            cell_size=cell_size,
            boundary_layers=pml_layers,
            geometry=geometry,
            k_point=k_point,
            sources=sources,
            symmetries=symmetries,
        )

        n2f_obj = sim.add_near2far(
            fcen, 0, 1, mp.Near2FarRegion(center=mon_pt, size=mp.Vector3(y=sy,z=sz))
        )

        sim.run(until_after_sources=100)

        return (
            abs(
                sim.get_farfields(
                    n2f_obj,
                    ff_res,
                    center=mp.Vector3(-0.5 * sx + dpml + dsub + gh + focal_length),
                    size=mp.Vector3(spot_length),
                )["Ez"]
            )
            ** 2
        )

#%%
gp = 0.3  # grating periodicity
gh = 1.8  # grating height
gdc = np.linspace(0.1, 1, 30)  # grating duty cycle

mode_tran = np.empty((gdc.size))
mode_phase = np.empty((gdc.size))
for n in range(gdc.size):
    mode_tran[n], mode_phase[n] = grating(gp, gh, [gdc[n]])

#%%
plt.figure(dpi=100)
plt.subplot(1, 2, 1)
plt.plot(gdc, mode_tran, "bo-")
plt.xlim(gdc[0], gdc[-1])
plt.xticks([t for t in np.linspace(0.1, 0.9, 5)])
plt.xlabel("grating duty cycle")
plt.ylim(0.96, 1.00)
plt.yticks([t for t in np.linspace(0.96, 1.00, 5)])
plt.title("transmittance")

plt.subplot(1, 2, 2)
plt.plot(gdc, mode_phase, "rs-")
plt.grid(True)
plt.xlim(gdc[0], gdc[-1])
plt.xticks([t for t in np.linspace(0.1, 0.9, 5)])
plt.xlabel("grating duty cycle")
plt.ylim(-2 * np.pi, 0)
plt.yticks([t for t in np.linspace(-6, 0, 7)])
plt.title("phase (radians)")

plt.tight_layout(pad=0.5)
plt.show()
# %%

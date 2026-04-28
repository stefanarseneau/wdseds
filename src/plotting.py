import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
import os

import interpolator
from . import likelihoods

spec = interpolator.atmos.WarwickSpectrum('1d_da_nlte', units='fnu', wavl_range=(2500, 11000))

matplotlib.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "text.latex.preamble": r"\usepackage[T1]{fontenc}\usepackage{lmodern}\usepackage{amsmath}",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

plt.style.use(os.path.join(os.path.dirname(__file__), "stefan.mplstyle"))

def plot_sed(sourceid, fl, e_fl, band_names, lambda_effs, teff, e_teff, logg, e_logg, plx, interp, logg_function, folder = "sedfigs"):
    theta = np.array([teff, logg, 1000 / plx, 0])
    fl_model_phot = likelihoods.get_model_flux(theta, interp, logg_function=logg_function) * 1e23

    fl_surface = 4 * np.pi * spec.model_spec((teff, logg))   # erg s⁻¹ cm⁻² Hz⁻¹
    radius = logg_function(teff, logg)
    radius_m = radius * 6.957e8    # R_sun → m
    dist_m   = (1000 / plx)   * 3.086775e16  # pc → m
    fl_model_jy = fl_surface * (radius_m / dist_m)**2 * 1e23   # Jy

    fig, (ax, ax_res) = plt.subplots(
        2, 1, figsize=(10, 8),
        gridspec_kw={'height_ratios': [3, 1]},
        sharex=True
    )

    # --- SED panel ---
    system_style = {
        'SDSS':      ([b for b in band_names if b.startswith('Sdss')],       '#586994', 'o'),
        'PanSTARRS': ([b for b in band_names if b.startswith('Panstarrs')],  '#F05D5E', 's'),
        'SkyMapper': ([b for b in band_names if b.startswith('Sky_Mapper')], '#BBB09B', '^'),
        'Gaia': ([b for b in band_names if b.startswith('Gaia')],            '#BFAB25', 'v'),
    }

    for sysname, (bands, color, marker) in system_style.items():
        if not bands:
            continue
        idx = [band_names.index(b) for b in bands]
        leff = lambda_effs[idx]
        ax.errorbar(leff, fl[idx] * 1e3, yerr=e_fl[idx] * 1e3,
                    fmt=marker, color=color, ecolor='k', capsize=4, ms=6, lw=1.2,
                    label=sysname, zorder=5)
        # model photometry (open symbols)
        ax.plot(leff, fl_model_phot[idx] * 1e3,
                marker=marker, color=color, ms=8, mfc='none', mew=1.5,
                ls='none', zorder=6)


    ax.plot(
        spec.wavl, fl_model_jy * 1e3, color='k', lw=3, zorder=4, 
        label=f"$T_\\text{{eff}}={teff:.0f}\\pm{e_teff:.0f}$~K\n$\\log g={logg:1.2f}\\pm{e_logg:1.2f}$~dex"
    )

    ax.set_ylabel('Flux [mJy]')
    ax.legend(framealpha=0, fontsize=18)

    # --- Residuals panel ---
    residuals = (fl - fl_model_phot) / e_fl
    ax_res.axhline(0, color='k', lw=1, ls='--')
    for sysname, (bands, color, marker) in system_style.items():
        if not bands:
            continue
        idx = [band_names.index(b) for b in bands]
        ax_res.errorbar(lambda_effs[idx], residuals[idx],
                        yerr=np.ones(len(idx)),
                        fmt=marker, color=color, ecolor='k', capsize=4, ms=6, lw=1.2)

    ax_res.set_xlabel(r'Wavelength [$\AA$]')
    ax_res.set_ylabel(r'Resid. [$\sigma$]')
    ax_res.set_xlim(2800, 10500)
    ax_res.set_ylim(-5, 5)

    fig.savefig(os.path.join(folder, f"Gaia_DR3_{sourceid}.png"))
    plt.close()
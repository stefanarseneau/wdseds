import sys
import os
import argparse

import numpy as np
import pandas as pd
import multiprocessing as mp
from scipy.interpolate import interp1d as _scipy_interp1d
from tqdm import tqdm

wdmodels_dir = os.environ['WDMODELS_DIR']
if wdmodels_dir not in sys.path:
    sys.path.append(wdmodels_dir)
import WD_models

mass_sun, radius_sun, newton_G = 1.9884e30, 6.957e8, 6.674e-11

_POP_MODEL_MAP = {
    'thick': ('Bedard2020',      'H'),
    'thin':  ('Bedard2020',      'He'),
    'mixed': ('Bedard2020_thin', 'H'),
}

# ---- per-process globals (populated once per worker by _init_worker) ----
_G_INTERP_TOT  = None
_G_INTERP_AGE  = None
_G_MASS_FUNC   = None
_G_HEINTZ_COOL = None
_G_HEINTZ_IFMR = None
_G_HEINTZ_TMS  = None


def _interp_chain(chain, feh, interpolator):
    """Evaluate interpolator(teff, radius, feh) over an (N, 2) sample chain."""
    feh = np.asarray(feh, dtype=float)
    if feh.ndim == 0:
        feh = np.full(chain.shape[0], feh)
    return interpolator(chain[:, 0], chain[:, 1], feh)


def _load_heintz_interps(pop_type):
    """Build cooling-age, IFMR, and MS-lifetime interpolators for high-mass WDs.

    Cooling ages use Bedard 2020 WD_models cooling tracks (model choice depends
    on population).  IFMR and MS lifetimes come from data files pointed to by
    HEINTZ_DATA_DIR (Heintz et al. 2024).
    """
    from . import data as _data
    heintz_dir = _data.get_heintz_dir()

    bedard_model, atm_type = _POP_MODEL_MAP[pop_type]
    model = WD_models.load_model(bedard_model, bedard_model, bedard_model, atm_type)
    g_acc    = (10 ** model["logg"]) / 100.0
    rsun_arr = np.sqrt(model["mass_array"] * mass_sun * newton_G / g_acc) / radius_sun

    cool_func = WD_models.interp_xy_z_func(
        x=10 ** model["logteff"], y=rsun_arr, z=model["age_cool"],
        interp_type="linear",
    )

    # IFMR: final WD mass → initial (progenitor) mass
    # The offset a=0.06367 is taken directly from SEDs_WDMS_v5.py (Heintz+2024);
    # its origin is not documented there — verify before publication.
    ifmr_data = pd.read_csv(os.path.join(heintz_dir, 'MESA_IFMR', 'MESA_IFMR_missing_one_point.csv'))
    IFMR_func = _scipy_interp1d(
        ifmr_data['M_final'].to_numpy() + 0.06367,
        ifmr_data['M_initial'].to_numpy(),
        kind='linear', fill_value=np.nan, bounds_error=False,
    )

    mi  = np.load(os.path.join(heintz_dir, 'init_mass_to_mslife', 'mi.npy'))
    msl = np.load(os.path.join(heintz_dir, 'init_mass_to_mslife', 'msl.npy'))
    t_ms_func = _scipy_interp1d(mi, msl, fill_value='extrapolate', bounds_error=False)

    return cool_func, IFMR_func, t_ms_func


def _init_worker(pop_type='thick'):
    """Runs once per worker process to build all shared interpolators."""
    global _G_INTERP_TOT, _G_INTERP_AGE, _G_MASS_FUNC
    global _G_HEINTZ_COOL, _G_HEINTZ_IFMR, _G_HEINTZ_TMS

    from . import age_util
    _G_INTERP_TOT = age_util.call_interp(fe_h=0, outcol="log_tot_age")
    _G_INTERP_AGE = age_util.call_interp(fe_h=0, outcol="log_age")

    bedard_model, atm_type = _POP_MODEL_MAP[pop_type]
    mass_model = WD_models.load_model(bedard_model, bedard_model, bedard_model, atm_type)
    g_acc_mass = (10 ** mass_model["logg"]) / 100.0
    rsun_mass  = np.sqrt(mass_model["mass_array"] * mass_sun * newton_G / g_acc_mass) / radius_sun
    _G_MASS_FUNC = WD_models.interp_xy_z_func(
        x=rsun_mass, y=10 ** mass_model["logteff"],
        z=mass_model["mass_array"],
        interp_type="linear",
    )

    _G_HEINTZ_COOL, _G_HEINTZ_IFMR, _G_HEINTZ_TMS = _load_heintz_interps(pop_type)


def _compute_one(args):
    """Compute age and mass posteriors for one star. Returns a 10-tuple."""
    ii, teff, e_teff, radius, e_radius, cov_rt, filled_row = args

    rng = np.random.default_rng(12345 + ii)  # deterministic per row → reproducible parallel runs
    samps = np.column_stack([
        rng.normal(teff,   e_teff,   size=10000),
        rng.normal(radius, e_radius, size=10000),
    ])
    fehs = rng.uniform(-0.2, 0.1, size=10000)
    mask = np.all(np.isfinite(samps), axis=1)

    samps_mass = _G_MASS_FUNC(samps[mask, 1], samps[mask, 0])
    valid_mass = np.isfinite(samps_mass)
    if valid_mass.sum() >= 100:
        ms = samps_mass[valid_mass]
        mass_val    = float(np.percentile(ms, 50))
        mass_hi_val = float(np.percentile(ms, 84))
        mass_lo_val = float(np.percentile(ms, 16))
    else:
        mass_val = mass_hi_val = mass_lo_val = np.nan

    _null = (ii, None, None, None, None, None, None, mass_val, mass_hi_val, mass_lo_val)

    if np.all(np.isfinite(filled_row)) or mask.sum() < 100:
        return _null

    if 0.512609 < mass_val < 1.017626:
        agecool = _interp_chain(samps[mask], fehs[mask], _G_INTERP_AGE)
        agecool = agecool[~np.isnan(agecool)]
        if mass_val > 0.63:
            agetot = _interp_chain(samps[mask], fehs[mask], _G_INTERP_TOT)
            agetot = agetot[~np.isnan(agetot)]
        else:
            agetot = np.full_like(agecool, np.nan)

    elif mass_val <= 0.512609:
        cool_gyr = _G_HEINTZ_COOL(samps[mask, 0], samps[mask, 1])
        valid    = np.isfinite(cool_gyr) & (cool_gyr > 0)
        agecool  = np.log10(cool_gyr[valid]) + 9   # log10(yr)
        agetot   = np.full_like(agecool, np.nan)

    elif mass_val >= 1.017626:
        cool_gyr = _G_HEINTZ_COOL(samps[mask, 0], samps[mask, 1])
        valid    = np.isfinite(cool_gyr) & (cool_gyr > 0)
        cool_gyr = cool_gyr[valid]
        agecool  = np.log10(cool_gyr) + 9          # log10(yr)

        init_mass = float(_G_HEINTZ_IFMR(mass_val))
        if np.isfinite(init_mass):
            ms_life_gyr = float(_G_HEINTZ_TMS(init_mass))
            agetot = np.log10(np.maximum(ms_life_gyr + cool_gyr, 1e-6)) + 9
        else:
            agetot = np.full_like(agecool, np.nan)

    else:
        return _null

    if not (isinstance(agetot, np.ndarray) and isinstance(agecool, np.ndarray)):
        return _null
    if agetot.size < 100 or agecool.size < 100:
        return _null

    return (
        ii,
        np.percentile(agecool, 50), np.percentile(agecool, 84), np.percentile(agecool, 16),
        np.percentile(agetot,  50), np.percentile(agetot,  84), np.percentile(agetot,  16),
        mass_val, mass_hi_val, mass_lo_val,
    )


def parallel_forloop(outdata, outpath, pop_type='thick', nproc=None, chunksize=50):
    """Run age measurement in parallel. Checkpoints parquet every 1000 rows."""
    cols = ["log_age_cool", "log_age_cool_hi", "log_age_cool_lo",
            "log_age",      "log_age_hi",      "log_age_lo"]

    teff     = outdata["teff_best"].to_numpy(dtype=float)
    e_teff   = outdata["std_tt_best"].to_numpy(dtype=float)
    radius   = outdata["radius_best"].to_numpy(dtype=float)
    e_radius = outdata["std_rr_best"].to_numpy(dtype=float)
    cov_rt   = outdata["cov_rt_best"].to_numpy(dtype=float)
    filled   = outdata[cols].to_numpy(dtype=float)

    n = len(outdata)
    if nproc is None:
        nproc = max(1, mp.cpu_count() - 1)

    def arggen():
        for ii in range(n):
            yield (ii, teff[ii], e_teff[ii], radius[ii], e_radius[ii], cov_rt[ii], filled[ii])

    ctx = mp.get_context("fork")  # Linux: avoids spawn/pickle overhead
    with ctx.Pool(processes=nproc, initializer=_init_worker, initargs=(pop_type,)) as pool:
        for k, res in enumerate(tqdm(pool.imap_unordered(_compute_one, arggen(), chunksize=chunksize), total=n)):
            ii, lac, lac_hi, lac_lo, lat, lat_hi, lat_lo, mass, mass_hi, mass_lo = res

            idx = outdata.index[ii]
            if lac is not None:
                outdata.loc[idx, "log_age_cool"]    = lac
                outdata.loc[idx, "log_age_cool_hi"] = lac_hi
                outdata.loc[idx, "log_age_cool_lo"] = lac_lo
                outdata.loc[idx, "log_age"]         = lat
                outdata.loc[idx, "log_age_hi"]      = lat_hi
                outdata.loc[idx, "log_age_lo"]      = lat_lo
            outdata.loc[idx, "mass"]    = mass
            outdata.loc[idx, "mass_hi"] = mass_hi
            outdata.loc[idx, "mass_lo"] = mass_lo

            if (k + 1) % 1000 == 0:
                outdata.to_parquet(outpath)

    return outdata


def main():
    parser = argparse.ArgumentParser(
        description="Measure white dwarf cooling and total ages via Monte Carlo sampling.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('inpath',  type=str, help='Path to input parquet file')
    parser.add_argument('outpath', type=str, help='Path to output parquet file')
    parser.add_argument('--pop_type', choices=['thick', 'thin', 'mixed'], default=None,
                        help='Population type for Bedard WD model (default: inferred from filename)')
    parser.add_argument('--nproc',     type=int, default=None, help='Number of worker processes (default: ncpu - 1)')
    parser.add_argument('--chunksize', type=int, default=50,   help='imap chunksize for multiprocessing (default: 50)')
    args = parser.parse_args()

    if args.pop_type is not None:
        pop_type = args.pop_type
    else:
        inbase = os.path.basename(args.inpath).lower()
        if 'mixed' in inbase:
            pop_type = 'mixed'
        elif 'thin' in inbase:
            pop_type = 'thin'
        else:
            pop_type = 'thick'

    data = pd.read_parquet(args.inpath)
    outdata = data.copy()
    for col in ["log_age_cool", "log_age_cool_hi", "log_age_cool_lo",
                "log_age",      "log_age_hi",      "log_age_lo",
                "mass",         "mass_hi",          "mass_lo"]:
        outdata[col] = np.nan

    outdata = parallel_forloop(outdata, args.outpath, pop_type=pop_type,
                               nproc=args.nproc, chunksize=args.chunksize)
    outdata.to_parquet(args.outpath)


if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
import argparse
import pyvo
from functools import reduce
import logging

from astroquery.gaia import Gaia
Gaia.ROW_LIMIT = -1

import interpolator
from . import sed_util as util

# Quality-flag bitmasks (match the cuts applied in single-object analysis)
_SDSS_BAD_BITS = 2**2 | 2**5 | 2**18 | 2**19   # EDGE | DEBLENDED_AS_PSF | SATURATED | NOTCHECKED
_PS1_BAD_BITS  = 2**10 | 2**11 | 2**12          # saturation / PSF-fit failure bits

def abmag_to_flux(mag : np.ndarray, e_mag : np.ndarray):
    """AB magnitude to flux in Jy"""
    flux = np.power(10, -0.4*(mag - 8.90))
    e_flux = 1.09 * flux * e_mag
    return flux, e_flux

def _get_jplus_flux(source_ids : list[np.int64]) -> pd.DataFrame:
    # TAP returns lowercase column names, so broad-band aliases need case + suffix fix
    _BANDS = ['rSDSS', 'gSDSS', 'iSDSS', 'zSDSS', 'uJAVA',
              'J0378', 'J0395', 'J0410', 'J0430', 'J0515', 'J0660', 'J0861']
    _OUT   = {b: b.replace('SDSS', 'JPLUS') for b in _BANDS}  # rSDSS→rJPLUS, narrow bands unchanged

    selects = ', '.join(
        f'jfl.flux_auto[jplus::{b}]*1e-7 as "Jplus_flux_{b}", '
        f'jfl.flux_relerr_auto[jplus::{b}]*1e-7 as "Jplus_flux_error_{b}"'
        for b in _BANDS
    )
    QUERY = f"""select gaia.source as gaia_dr3_source_id, {selects}
                from jplus.FNuDualObj as jfl
                join jplus.xmatch_gaia_dr3 as gaia
                    on gaia.tile_id = jfl.tile_id and gaia.NUMBER = jfl.NUMBER
                where gaia.Source in {tuple(source_ids)}"""

    tap_service = pyvo.dal.TAPService("https://archive.cefca.es/catalogues/vo/tap/jplus-dr3")
    table = tap_service.search(QUERY).to_table().to_pandas()

    rename = {f'jplus_flux_{b.lower()}':       f'Jplus_flux_{out}'
              for b, out in _OUT.items()}
    rename |= {f'jplus_flux_error_{b.lower()}': f'Jplus_flux_error_{out}'
               for b, out in _OUT.items()}
    return table.rename(columns=rename)

def _get_panstarrs_flux(source_ids : list[np.int64]) -> pd.DataFrame:
    # Step 1: Gaia DR3 → PS1 cross-match
    QUERY1 = f"""select source_id as gaia_dr3_source_id, original_ext_source_id as objid
                from gaiadr3.panstarrs1_best_neighbour
                where source_id in {tuple(source_ids)}"""
    gaianames = Gaia.launch_job_async(QUERY1).get_results().to_pandas()

    # Step 2: PS1 DR2 mean-stack photometry direct from MAST
    tap_service = pyvo.dal.TAPService(" https://mast.stsci.edu/vo-tap/api/v0.1/ps1_dr2/")
    QUERY2 = f"""select
                    objID as objid,
                    gmeanpsfmag as panstarrs1_mag_gp, gmeanpsfmagerr as panstarrs1_mag_error_gp, gflags as ps1_flags_g,
                    rmeanpsfmag as panstarrs1_mag_rp, rmeanpsfmagerr as panstarrs1_mag_error_rp, rflags as ps1_flags_r,
                    imeanpsfmag as panstarrs1_mag_ip, imeanpsfmagerr as panstarrs1_mag_error_ip, iflags as ps1_flags_i,
                    zmeanpsfmag as panstarrs1_mag_zp, zmeanpsfmagerr as panstarrs1_mag_error_zp, zflags as ps1_flags_z,
                    ymeanpsfmag as panstarrs1_mag_yp, ymeanpsfmagerr as panstarrs1_mag_error_yp, yflags as ps1_flags_y
                from ps1_dr2.mean_object
                where objID in {tuple(gaianames.objid.tolist())}"""
    ps1photo = tap_service.search(QUERY2, maxrec=0).to_table().to_pandas().dropna(subset=['objid'])
    ps1photo = ps1photo.replace(-999.0, np.nan)

    for band in ["g", "r", "i", "z", "y"]:
        mag     = ps1photo[f"panstarrs1_mag_{band}p"].values
        mag_err = np.sqrt(ps1photo[f"panstarrs1_mag_error_{band}p"].values**2 + 0.03**2)
        flux, flux_err = abmag_to_flux(mag, mag_err)
        ps1photo[f"Panstarrs1_flux_{band}p"]       = flux
        ps1photo[f"Panstarrs1_flux_error_{band}p"] = flux_err

        bad = (ps1photo[f'ps1_flags_{band}'].fillna(0).astype(np.int64) & _PS1_BAD_BITS) != 0
        ps1photo.loc[bad, f'Panstarrs1_flux_{band}p']       = np.nan
        ps1photo.loc[bad, f'Panstarrs1_flux_error_{band}p'] = np.nan

    ps1photo = ps1photo.dropna(subset=[f'Panstarrs1_flux_{band}p' for band in ['g', 'r', 'i', 'z', 'y']], how='all')
    logging.info(f"Queried PanStarrs for {len(source_ids)} objects, found {len(ps1photo)} results!")
    return pd.merge(gaianames, ps1photo, on='objid').drop(columns=[f'ps1_flags_{b}' for b in 'grizy'])

def _get_sdss_flux(source_ids : list[np.int64]) -> pd.DataFrame:
    """query the SDSS archive and return fluxes in jy"""
    tap_service = pyvo.dal.TAPService("http://TAPVizieR.u-strasbg.fr/TAPVizieR/tap/")
    QUERY = f"""select GaiaEDR3 as gaia_dr3_source_id, *
                from \"J/MNRAS/508/3877/maincat\"
                where GaiaEDR3 in {tuple(source_ids)}"""
    sdssphoto = tap_service.search(QUERY).to_table().to_pandas()

    # AB zero-point offsets per band (Fukugita+1996); applied as mag - correction - 8.90
    corrections = {'u': 0.040, 'g': 0.0, 'r': 0.0, 'i': -0.015, 'z': -0.030}
    flux_cols = []
    for band, corr in corrections.items():
        flux = 10 ** (-0.4 * (sdssphoto[f'{band}mag'] - corr - 8.90))
        sdssphoto[f'Sdss_flux_{band}']       = flux
        sdssphoto[f'Sdss_flux_error_{band}'] = 1.09 * flux * np.sqrt(sdssphoto[f'e_{band}mag'] ** 2 + 0.03 ** 2)
        flux_cols += [f'Sdss_flux_{band}', f'Sdss_flux_error_{band}']

        # Apply SDSS quality flags if the catalog returned them (SELECT * brings them along)
        for flag_col in (f'flags_{band}', f'Flags_{band}'):
            if flag_col in sdssphoto.columns:
                bad = (sdssphoto[flag_col].fillna(0).astype(np.int64) & _SDSS_BAD_BITS) != 0
                sdssphoto.loc[bad, f'Sdss_flux_{band}']       = np.nan
                sdssphoto.loc[bad, f'Sdss_flux_error_{band}'] = np.nan
                break

    sdssphoto = sdssphoto.dropna(subset=[f'Sdss_flux_{band}' for band in ['u', 'g', 'r', 'i', 'z']], how='all')
    logging.info(f"Queried SDSS for {len(source_ids)} objects, found {len(sdssphoto)} results!")
    return sdssphoto[['gaia_dr3_source_id'] + flux_cols].dropna(subset=['gaia_dr3_source_id'])

def _get_gaia_flux(source_ids : list[int]) -> pd.DataFrame:
    """query the gaia archive and return gaia fluxes in jy"""
    QUERY = f"""select source_id as gaia_dr3_source_id, 
                  phot_g_mean_flux as GaiaDr3Vega_flux_G, 
                  phot_g_mean_flux_error as GaiaDr3Vega_flux_error_G, 
                  phot_bp_mean_flux as GaiaDr3Vega_flux_BP, 
                  phot_bp_mean_flux_error as GaiaDr3Vega_flux_error_BP, 
                  phot_rp_mean_flux as GaiaDr3Vega_flux_RP, 
                  phot_rp_mean_flux_error as GaiaDr3Vega_flux_error_RP
                from gaiadr3.gaia_source
                  where source_id in {tuple(source_ids)}"""
    results = Gaia.launch_job_async(QUERY).get_results().to_pandas()
    # perform unit conversions
    results[['GaiaDr3Vega_flux_G', 'GaiaDr3Vega_flux_error_G']] *= 1.736011E-33 * 1e26
    results[['GaiaDr3Vega_flux_BP', 'GaiaDr3Vega_flux_error_BP']] *= 2.620707e-33 * 1e26
    results[['GaiaDr3Vega_flux_RP', 'GaiaDr3Vega_flux_error_RP']] *= 3.298815e-33 * 1e26
    #results[['gflux', 'e_gflux']] = results[['GaiaDr3Vega_flux_G', 'GaiaDr3Vega_flux_error_G']].values
    return results

def _get_gaia_flux_ngf(source_ids : list[int]) -> pd.DataFrame:
    """query the gaia archive and return gaia fluxes in jy"""
    tap_service = pyvo.dal.TAPService("http://TAPVizieR.u-strasbg.fr/TAPVizieR/tap/")
    QUERY = f"""select GaiaEDR3 as gaia_dr3_source_id, 
                  GmagCorr as GaiaDr3Vega_mag_G, 
                  e_GmagCorr as e_Gmag, 
                  BPmag as GaiaDr3Vega_mag_BP, 
                  e_BPmag as e_BPmag, 
                  RPmag as GaiaDr3Vega_mag_RP, 
                  e_RPmag as e_RPmag
                from \"J/MNRAS/508/3877/maincat\"
                    where GaiaEDR3 in {tuple(source_ids)}"""
    results = tap_service.search(QUERY, maxrec=0).to_table().to_pandas()
    # perform unit conversions
    results["GaiaDr3Vega_flux_G"] = 10**(-0.4*(results["GaiaDr3Vega_mag_G"] + 21.48503))
    results["GaiaDr3Vega_flux_error_G"] = 1.09*results["GaiaDr3Vega_flux_G"]*results["e_Gmag"]
    results["GaiaDr3Vega_flux_BP"] = 10**(-0.4*(results["GaiaDr3Vega_mag_BP"] + 20.96683))
    results["GaiaDr3Vega_flux_error_BP"] = 1.09*results["GaiaDr3Vega_flux_BP"]*results["e_BPmag"]
    results["GaiaDr3Vega_flux_RP"] = 10**(-0.4*(results["GaiaDr3Vega_mag_RP"] + 22.22089))
    results["GaiaDr3Vega_flux_error_RP"] = 1.09*results["GaiaDr3Vega_flux_RP"]*results["e_RPmag"]
    return results

def _get_skymapper_flux(source_ids: list[np.int64]) -> pd.DataFrame:
    """query SkyMapper DR3 and return PSF fluxes in Jy"""
    # Step 1: Gaia DR3 → SkyMapper DR2 cross-match (object_ids are stable across DR2/DR3)
    QUERY1 = f"""select source_id as gaia_dr3_source_id, original_ext_source_id as object_id
                from gaiadr3.skymapperdr2_best_neighbour
                where source_id in {tuple(source_ids)}"""
    gaianames = Gaia.launch_job_async(QUERY1).get_results().to_pandas()

    # Step 2: SkyMapper DR3 PSF photometry + per-band quality flags
    tap_service = pyvo.dal.TAPService("https://api.skymapper.nci.org.au/public/tap/")
    QUERY2 = f"""select object_id,
                    u_psf as SkyMapper_mag_u, e_u_psf as SkyMapper_mag_error_u, u_flags, u_nimaflags,
                    v_psf as SkyMapper_mag_v, e_v_psf as SkyMapper_mag_error_v, v_flags, v_nimaflags,
                    g_psf as SkyMapper_mag_g, e_g_psf as SkyMapper_mag_error_g, g_flags, g_nimaflags,
                    r_psf as SkyMapper_mag_r, e_r_psf as SkyMapper_mag_error_r, r_flags, r_nimaflags,
                    i_psf as SkyMapper_mag_i, e_i_psf as SkyMapper_mag_error_i, i_flags, i_nimaflags,
                    z_psf as SkyMapper_mag_z, e_z_psf as SkyMapper_mag_error_z, z_flags, z_nimaflags
                from dr4.master
                where object_id in {tuple(gaianames.object_id.tolist())}"""
    smphoto = tap_service.search(QUERY2, maxrec=0).to_table().to_pandas()
    smphoto = smphoto.rename(columns=lambda x: x.replace('skymapper', 'Sky_Mapper'))

    flag_cols = []
    for band in ['u', 'v', 'g', 'r', 'i', 'z']:
        flux, e_flux = abmag_to_flux(
            smphoto[f'Sky_Mapper_mag_{band}'],
            np.sqrt(smphoto[f'Sky_Mapper_mag_error_{band}'] ** 2 + 0.03 ** 2),
        )
        smphoto[f'Sky_Mapper_flux_{band}']       = flux
        smphoto[f'Sky_Mapper_flux_error_{band}'] = e_flux

        bad = (smphoto[f'{band}_flags'].fillna(1) > 0) | (smphoto[f'{band}_nimaflags'].fillna(1) > 0)
        smphoto.loc[bad, f'Sky_Mapper_flux_{band}']       = np.nan
        smphoto.loc[bad, f'Sky_Mapper_flux_error_{band}'] = np.nan
        flag_cols += [f'{band}_flags', f'{band}_nimaflags']

    smphoto = smphoto.dropna(subset=[f'Sky_Mapper_flux_{band}' for band in ['u', 'v', 'g', 'r', 'i', 'z']], how='all')
    logging.info(f"Queried SkyMapper for {len(source_ids)} objects, found {len(smphoto)} results!")
    return pd.merge(gaianames, smphoto, on='object_id').drop(columns=['object_id'] + flag_cols)


def process_dataframe(df : pd.DataFrame, systems : list[str] = ['gaia', 'sdss'], use_gravz : bool = False) -> pd.DataFrame:
    assert set(systems).issubset(set(['gaia', 'sdss', 'panstarrs', 'jplus', 'skymapper'])), "Err: unsupported photometric system!"
    ff = interpolator.atmos.sed.get_default_filters()
    photometry_dict = {'gaia' : _get_gaia_flux, 'sdss' : _get_sdss_flux,
                       'panstarrs' : _get_panstarrs_flux,
                       'jplus' : _get_jplus_flux,
                       'skymapper' : _get_skymapper_flux}
    phot_system = [photometry_dict[system] for system in systems]

    band_dict = {
        'gaia' : ['Gaia_G', 'Gaia_BP', 'Gaia_RP'],
        'sdss' : ['SDSS_u', 'SDSS_g', 'SDSS_r', 'SDSS_i', 'SDSS_z'],
        'panstarrs' : ['PS1_g', 'PS1_r', 'PS1_i', 'PS1_z', 'PS1_y'],
        'jplus' : ['JPLUS_J0378', 'JPLUS_J0395', 'JPLUS_J0410', 'JPLUS_J0430',
                    'JPLUS_J0515', 'JPLUS_J0660', 'JPLUS_J0861'],
        'skymapper' : ['SkyMapper_u', 'SkyMapper_v', 'SkyMapper_g',
                       'SkyMapper_r', 'SkyMapper_i', 'SkyMapper_z'],
    }
    lambda_eff = np.array([
        ff[band].lambda_eff
        for sys in systems
        for band in band_dict[sys]
    ])

    extinction_vec = util.fetch_extinction(lambda_eff)
    #extinction_vec = np.array([0.835, 1.139, 0.650])
    # preprocess the dataframe
    df = util.check_valid(df, use_gravz)
    source_ids = df['gaia_dr3_source_id'].values.astype(np.int64).tolist()
    synphot_list = [system(source_ids) for system in phot_system]
    synphot = reduce(lambda left, right: pd.merge(left, right, on='gaia_dr3_source_id', how="outer"), synphot_list)
    # fix the Gaia photometry
    flux_dict = util.find_photocols(synphot)
    return pd.merge(df, synphot, on='gaia_dr3_source_id'), flux_dict, lambda_eff, extinction_vec

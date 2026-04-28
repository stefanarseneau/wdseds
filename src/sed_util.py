import numpy as np
import pandas as pd
import scipy

import sys, os
wdmodels_dir = os.environ['WDMODELS_DIR']
sys.path.append(wdmodels_dir)
import WD_models
import interpolator
import pyvo

### Utility functions for the pipeline
dir_path = os.path.dirname(os.path.realpath(__file__))

### Utility functions for the pipeline

def convert_names(bands = list[str]) -> list[str]:
    """convert between pyphot names and xp names"""
    xp_to_pyphot = {# Gaia conversions
                    'GaiaDr3Vega_G' : 'Gaia_G', 
                    'GaiaDr3Vega_BP' : 'Gaia_BP', 
                    'GaiaDr3Vega_RP' : 'Gaia_RP',
                    # SDSS conversions
                    'Sdss_u' : 'SDSS_u', 
                    'Sdss_g' : 'SDSS_g', 
                    'Sdss_r' : 'SDSS_r', 
                    'Sdss_i' : 'SDSS_i', 
                    'Sdss_z' : 'SDSS_z',
                    # PanSTARRS conversions
                    'Panstarrs1_gp' : 'PS1_g', 
                    'Panstarrs1_rp' : 'PS1_r', 
                    'Panstarrs1_ip' : 'PS1_i', 
                    'Panstarrs1_zp' : 'PS1_z', 
                    'Panstarrs1_yp' : 'PS1_y',
                    # J-PLUS conversions
                    'Jplus_uJAVA' : 'JPLUS_uJava', 
                    'Jplus_J0378' : 'JPLUS_J0378', 
                    'Jplus_J0395' : 'JPLUS_J0395', 
                    'Jplus_J0410' : 'JPLUS_J0410', 
                    'Jplus_J0430' : 'JPLUS_J0430', 
                    'Jplus_gJPLUS' : 'JPLUS_gSDSS', 
                    'Jplus_J0515' : 'JPLUS_J0515', 
                    'Jplus_rJPLUS' : 'JPLUS_rSDSS', 
                    'Jplus_J0660' : 'JPLUS_J0660', 
                    'Jplus_iJPLUS' : 'JPLUS_iSDSS', 
                    'Jplus_J0861' : 'JPLUS_J0861', 
                    'Jplus_zJPLUS' : 'JPLUS_zSDSS', 
                    # SkyMapper
                    'Sky_Mapper_u' : 'SkyMapper_u',
                    'Sky_Mapper_v' : 'SkyMapper_v',
                    'Sky_Mapper_g' : 'SkyMapper_g',
                    'Sky_Mapper_r' : 'SkyMapper_r',
                    'Sky_Mapper_i' : 'SkyMapper_i',
                    'Sky_Mapper_z' : 'SkyMapper_z',
                    }
    pyphot_to_xp = {val : key for key, val in xp_to_pyphot.items()}
    non_synth = [b for b in bands if not b.startswith("SYNTH_")]
    # perform the conversion
    if set(non_synth).issubset(set(xp_to_pyphot.keys())):
        return [xp_to_pyphot[band] if 'SYNTH_' not in band else band for band in bands]
    elif set(non_synth).issubset(set(pyphot_to_xp.keys())):
        return [pyphot_to_xp[band] if 'SYNTH_' not in band else band for band in bands]
    else:
        raise "Conversion error!"

def fetch_extinction(wavelengths : np.array) -> np.array:
    """return the extinction coefficients for each band in the dataset"""
    from importlib.resources import files
    csv = files('sedtool').joinpath('extinction.csv')
    data = pd.read_csv(csv)
    return np.interp(wavelengths, data["lambda_eff"], data["R_V(3.1)"]) / 2.742

def check_valid(df : pd.DataFrame, use_gravz : bool = False) -> pd.DataFrame:
    """check that the dataframe passed is valid and strip out relevant parts"""
    needed_cols = ['gaia_dr3_source_id', 'ra', 'dec', 'parallax', 'parallax_error', 'meanAV']
    if use_gravz:
        needed_cols += ['gravz', 'gravz_error']
    assert all(col in df.columns for col in needed_cols), f"Missing one or more required columns: {tuple(needed_cols)}"
    df = df[needed_cols]
    return df

def extract_data(
        df : pd.DataFrame,
        source_id : str, 
        ra : str, 
        dec : str, 
        parallax : str, 
        parallax_error: str, 
        meanAV : str,
        gravz : str = None,
        gravz_error : str = None
    ) -> pd.DataFrame:
    columns = {source_id : 'gaia_dr3_source_id', ra : 'ra', dec : 'dec', parallax : 'parallax',
               parallax_error : 'parallax_error', meanAV : 'meanAV'}
    if (gravz is not None) and (gravz_error is not None):
        columns[gravz] = 'gravz'
        columns[gravz_error] = 'gravz_error'
    df = df[list(columns.keys())]
    df = df.rename(mapper=columns, axis=1)
    df['gaia_dr3_source_id'] = df['gaia_dr3_source_id'].astype(np.int64)
    return df

def find_photocols(df : pd.DataFrame):
    """find the columns which """
    # Keep only flux and flux_error columns
    flux_cols = [col for col in df.columns if '_flux_' in col]
    flux_err_cols = [col for col in df.columns if '_flux_error_' in col]
    # Match each flux column with its corresponding error column
    flux_dict = {}
    for flux in flux_cols:
        band = ('_').join(flux.split('_flux_'))
        error_col = flux.replace('_flux_', '_flux_error_')
        if error_col in flux_err_cols:
            flux_dict[band] = (flux, error_col)
    #if "GaiaDr3Vega_G" in flux_dict.keys():
    #    del flux_dict["GaiaDr3Vega_G"]
    return flux_dict

### Functions for interpolating mass-radius relation and photometry

def zpcorrect(df : pd.DataFrame):
    source_ids = df["gaia_dr3_source_id"].values.astype(str).tolist()
    tap_service = pyvo.dal.TAPService("http://TAPVizieR.u-strasbg.fr/TAPVizieR/tap/")
    QUERY = f"""select GaiaEDR3 as gaia_dr3_source_id, ZPcor
            from \"J/MNRAS/508/3877/maincat\"
            where GaiaEDR3 in {tuple(source_ids)}"""
    res = tap_service.search(QUERY).to_table().to_pandas()
    df = pd.merge(df, res, on="gaia_dr3_source_id")
    df["parallax"] = df["parallax"] - df["ZPcor"]
    return df

def get_logg_function(low_model = 'f', mid_model = 'f', high_model = 'f', atm_type = 'H') -> scipy.interpolate:
    """compute the radial velocity from radius and effective temperature"""
    mass_sun, radius_sun, newton_G, speed_light = 1.9884e30, 6.957e8, 6.674e-11, 299792458
    font_model = WD_models.load_model(low_model, mid_model, high_model, atm_type)
    g_acc = (10**font_model['logg'])/100
    rsun = np.sqrt(font_model['mass_array'] * mass_sun * newton_G / g_acc) / radius_sun
    return WD_models.interp_xy_z_func(x = 10**font_model['logteff'], y = rsun,\
                                                z = font_model['logg'], interp_type = 'linear')

def get_actual_logg_function(low_model = 'f', mid_model = 'f', high_model = 'f', atm_type = 'H') -> scipy.interpolate:
    """compute the radial velocity from radius and effective temperature"""
    mass_sun, radius_sun, newton_G, speed_light = 1.9884e30, 6.957e8, 6.674e-11, 299792458
    font_model = WD_models.load_model(low_model, mid_model, high_model, atm_type)
    g_acc = (10**font_model['logg'])/100
    rsun = np.sqrt(font_model['mass_array'] * mass_sun * newton_G / g_acc) / radius_sun
    return WD_models.interp_xy_z_func(x = 10**font_model['logteff'], y = font_model['logg'],\
                                                z = rsun, interp_type = 'linear')

def make_interpolator(bands, units, fixedhe=None):
    """build the model SED using default filters"""
    if fixedhe == 30:
        model_name = "1d_db_nlte"
    elif fixedhe is not None:
        model_name = "1d_dba_nlte"
    else:
        model_name = "1d_da_nlte"

    #model_name = "1d_dba_nlte" if fixedhe is not None else "1d_da_nlte"
    defaults = interpolator.atmos.sed.get_default_filters()
    SED = interpolator.atmos.WarwickPhotometry(model_name, [defaults[band] for band in bands],
                                                units = units, fixedhe=fixedhe)
    interp, _, (T, L, A, grid_sansav, grid) = SED.make_cache(nAV=7)
    return interp, SED

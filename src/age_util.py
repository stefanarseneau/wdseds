import os
import re
import glob
from io import StringIO
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import tqdm
from scipy.interpolate import LinearNDInterpolator, interp1d

from . import data as _data


def parse_feh(path: str) -> float:
    """Extract [Fe/H] from filenames of the form feh_m050 / feh_p010."""
    m = re.search(r"feh_([mp])(\d+)", os.path.basename(path))
    if not m:
        return None
    sign = -1 if m.group(1) == "m" else 1
    return sign * int(m.group(2)) / 100


def parse_metadata(filename) -> dict:
    metadata = {}
    with open(filename) as f:
        for line in f:
            if line.startswith("# M_"):
                key, value = line[1:].strip().split("=")
                metadata[key.strip()] = float(value)
    return metadata


# ---- .wdcool / track file I/O ----

def read_wdcool(path: Path):
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            if line.strip().startswith('#'):
                cols = re.split(r'\s+', line.strip().lstrip('#').strip())
                break
    df = pd.read_csv(path, comment="#", sep=r"\s+", header=None, names=cols, engine="python")
    return df, cols


def parse_track_file(path: Path) -> Tuple[float, float, pd.DataFrame]:
    M_in = M_WD = cols = None
    rows = []
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            s = line.strip()
            if s.startswith('# M_in'):
                M_in = float(re.search(r'=\s*([0-9.]+)', s).group(1))
            elif s.startswith('# M_WD'):
                M_WD = float(re.search(r'=\s*([0-9.]+)', s).group(1))
            elif s.startswith('# log_cool_age'):
                cols = [c for c in re.split(r'\s+', s.lstrip('#').strip()) if c != '...']
            elif s:
                rows.append(line.replace('...', ' '))
    if M_WD is None:
        m = re.search(r'MWD_([0-9.]+)', path.name)
        if m:
            M_WD = float(m.group(1))
    df = pd.read_csv(StringIO(''.join(rows)), sep=r"\s+", header=None, names=cols, engine='python')
    return M_in, M_WD, df[['log_cool_age', 'log_tot_age']].copy()


def build_mass_to_track(tracks_dir: Path) -> Dict[float, Tuple[float, float, pd.DataFrame]]:
    mapping = {}
    for p in sorted(tracks_dir.glob("*.data")):
        try:
            M_in, M_WD, df = parse_track_file(p)
            if M_WD is not None:
                mapping[M_WD] = (M_in, M_WD, df.sort_values('log_cool_age'))
        except Exception:
            pass
    if not mapping:
        raise ValueError(f"No usable track files in {tracks_dir}")
    return mapping


def assign_log_tot_age(df_wd: pd.DataFrame, mass_to_track, mass_col='Mass', cool_col='log_age', mass_tol=0.02) -> pd.Series:
    masses = np.array(sorted(mass_to_track.keys()))
    interps = {}
    for M, (_, _, tdf) in mass_to_track.items():
        x = tdf['log_cool_age'].to_numpy()
        y = tdf['log_tot_age'].to_numpy()
        x_uniq, idx = np.unique(x, return_index=True)
        interps[M] = interp1d(x_uniq, y[idx], kind='linear', fill_value='extrapolate', assume_sorted=True)

    out = np.full(len(df_wd), np.nan, float)
    for i, (m, a) in enumerate(zip(df_wd[mass_col].to_numpy(), df_wd[cool_col].to_numpy())):
        j = np.argmin(np.abs(masses - m))
        if abs(masses[j] - m) <= mass_tol:
            out[i] = float(interps[masses[j]](a))
    return pd.Series(out, name='log_tot_age', index=df_wd.index)


def write_wdcool_with_totage(df_wd: pd.DataFrame, cols_original: list, output_path: Path) -> None:
    cols = cols_original.copy()
    if 'log_tot_age' not in cols:
        insert_at = cols.index('log_age') + 1 if 'log_age' in cols else len(cols)
        cols.insert(insert_at, 'log_tot_age')
    for c in cols:
        if c not in df_wd:
            df_wd[c] = np.nan
    header = '# ' + '   '.join(f'{c:>12}' for c in cols)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(header + '\n')
        for row in df_wd[cols].to_numpy():
            f.write(''.join(
                f'{v:13.6f}' if isinstance(v, (int, float, np.floating)) and np.isfinite(v) else f'{v:>13}'
                for v in row
            ) + '\n')


def process_wdcool_with_tracks(wdcool_path: Path, tracks_dir: Path,
                                out_suffix=".with_totage.wdcool", mass_tol=0.02) -> Path:
    df_wd, cols = read_wdcool(wdcool_path)
    mapping = build_mass_to_track(tracks_dir)
    df_wd['log_tot_age'] = assign_log_tot_age(df_wd, mapping, mass_col='Mass',
                                               cool_col='log_age', mass_tol=mass_tol)
    outpath = wdcool_path.with_suffix(out_suffix)
    extra = ['log_tot_age'] if 'log_tot_age' not in cols else []
    write_wdcool_with_totage(df_wd, cols_original=cols + extra, output_path=outpath)
    return outpath


# ---- Interpolator construction ----

def read_tracks(save: bool = False) -> pd.DataFrame:
    """Read all MIST cooling tracks from the MIST data directory and return a summary DataFrame."""
    coolingdir = str(_data.get_mist_dir())
    metalfolders = [f[:-1] for f in glob.glob(os.path.join(coolingdir, "*/"))]
    datafiles = []
    for metal in tqdm.tqdm(metalfolders, desc="Building interpolator"):
        coolpath = os.path.join(metal, f"{os.path.basename(metal)}.wdcool")
        agepath  = process_wdcool_with_tracks(Path(coolpath), Path(os.path.join(metal, "Tracks")))
        data = pd.DataFrame(np.genfromtxt(agepath, names=True))
        data['fe_h']   = parse_feh(metal) * np.ones(len(data))
        data['teff']   = 10 ** data['log_Teff']
        data['radius'] = 10 ** data['log_R']
        datafiles.append(data.drop(columns=["log_Teff", "log_R"]))
    datafile = pd.concat(datafiles).reset_index(drop=True)
    if save:
        datafile.to_parquet(os.path.join(coolingdir, "summary.pqt"))
    return datafile


def make_interpolator(datafile: pd.DataFrame, fe_h=0, outcol="log_age"):
    """Build a (teff, radius) → outcol interpolator for a given metallicity slice."""
    target = fe_h if fe_h is not None else 0
    df = datafile[np.isclose(datafile['fe_h'], target)].copy()
    if df.empty:
        raise ValueError(f"No cooling tracks for fe_h={target}")

    s_teff, s_radius = 1e-3, 1e2   # scaling improves Delaunay conditioning
    pts  = df[["teff", "radius"]].to_numpy(np.float64)
    vals = df[outcol].to_numpy(np.float64)
    P    = np.c_[pts[:, 0] * s_teff, pts[:, 1] * s_radius]
    Puniq, idx = np.unique(P, axis=0, return_index=True)
    lin = LinearNDInterpolator(Puniq, vals[idx], fill_value=np.nan)

    def predict(teff, radius, feh=None):
        teff, radius = np.broadcast_arrays(
            np.asarray(teff, dtype=np.float64),
            np.asarray(radius, dtype=np.float64),
        )
        X = np.column_stack([teff.ravel() * s_teff, radius.ravel() * s_radius])
        return lin(X).reshape(teff.shape)

    return predict


def call_interp(fe_h=0, outcol="log_tot_age"):
    """Load (or build) the age interpolator for a given metallicity and output column."""
    coolingdir = str(_data.get_mist_dir())
    summary = os.path.join(coolingdir, "summary.pqt")
    datafile = pd.read_parquet(summary) if os.path.isfile(summary) else read_tracks(save=True)
    return make_interpolator(datafile, fe_h=fe_h, outcol=outcol)

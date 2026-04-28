"""Data directory helpers and Zenodo download utilities."""
import os
import argparse
import tarfile
import urllib.request
from pathlib import Path

_ZENODO_BASE = "https://zenodo.org/records/15242047/files"
_MIST_FILES = {
    'scaled_solar': ('default_grids_scaled_solar.tgz', '177 MB'),
    'full':         ('default_grids_full.tgz',         '879 MB'),
    'nonrotating':  ('nonrotating_grids_full.tgz',     '881 MB'),
    'bc_tables':    ('BC_tables.tgz',                  '970 KB'),
}

# Fallback for _HEINTZ_DIR: two levels up from src/, then into code_for_JJ
_HEINTZ_FALLBACK = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 '..', '..', 'code_for_JJ', 'code_for_JJ', 'code_for_JJ')
)


def get_mist_dir() -> Path:
    """Return the MIST data directory.

    Checks $MIST_DATA_DIR then $COOLING_PATH (legacy), then ~/.sedtool/mist/.
    """
    for var in ('MIST_DATA_DIR', 'COOLING_PATH'):
        env = os.environ.get(var)
        if env:
            return Path(env)
    return Path.home() / '.sedtool' / 'mist'


def get_heintz_dir() -> str:
    """Return the Heintz et al. 2024 data directory (IFMR + MS-lifetime tables).

    Checks $HEINTZ_DATA_DIR first, then falls back to the legacy relative path
    used during development.
    """
    return os.environ.get('HEINTZ_DATA_DIR', _HEINTZ_FALLBACK)


def download_mist_data(subsets=('scaled_solar', 'bc_tables'), dest=None, overwrite=False):
    """Download and unpack MIST WD cooling grids from Zenodo (10.5281/zenodo.15242047).

    Parameters
    ----------
    subsets : iterable of str
        Which archives to fetch. Choices: 'scaled_solar' (177 MB, default),
        'bc_tables' (970 KB), 'full' (879 MB), 'nonrotating' (881 MB).
    dest : path-like, optional
        Directory to download into. Defaults to get_mist_dir().
    overwrite : bool
        Re-download even if the archive already exists on disk.
    """
    dest = Path(dest) if dest is not None else get_mist_dir()
    dest.mkdir(parents=True, exist_ok=True)

    for subset in subsets:
        if subset not in _MIST_FILES:
            raise ValueError(f"Unknown subset {subset!r}. Choose from: {list(_MIST_FILES)}")
        filename, size = _MIST_FILES[subset]
        url   = f"{_ZENODO_BASE}/{filename}"
        local = dest / filename

        if local.exists() and not overwrite:
            print(f"  {filename}: already present, skipping  (--overwrite to re-download)")
        else:
            print(f"  Downloading {filename} ({size}) ...")
            urllib.request.urlretrieve(url, local)

        print(f"  Unpacking {filename} ...")
        with tarfile.open(local) as tf:
            tf.extractall(dest)

    print(f"\nMIST data ready in: {dest}")
    print(f"Set MIST_DATA_DIR={dest} so sed-tool can find it.")


def _download_cli():
    parser = argparse.ArgumentParser(
        description="Download MIST WD cooling grids from Zenodo (10.5281/zenodo.15242047).",
    )
    parser.add_argument(
        '--subsets', nargs='+',
        default=['scaled_solar', 'bc_tables'],
        choices=list(_MIST_FILES),
        help='Which archive(s) to download (default: scaled_solar bc_tables)',
    )
    parser.add_argument(
        '--dest', type=str, default=None,
        help=f'Destination directory (default: $MIST_DATA_DIR or ~/.sedtool/mist/)',
    )
    parser.add_argument(
        '--overwrite', action='store_true',
        help='Re-download even if the archive already exists',
    )
    args = parser.parse_args()
    download_mist_data(subsets=args.subsets, dest=args.dest, overwrite=args.overwrite)

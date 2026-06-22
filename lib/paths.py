"""Central path configuration for ZEAL.

Every script resolves its inputs and outputs through this module, so the code is
portable: nothing is hard-coded to one machine. Defaults are relative to the
repository; override any of them with environment variables when the data lives
elsewhere (for example a copy downloaded from the figshare data record):

    ZEAL_DATA   results (JSON/npz), trained weights, and the rasters/ subfolder
                (default: <repo>/data)
    ZEAL_RAW    raw public downloads + derived geometry
                (default: <repo>/raw)
    ATLAS_CSV   full path to the Opportunity Atlas tract_outcomes.csv
                (default: <repo>/raw/atlas/tract_outcomes.csv)

Quick start: unpack the figshare `data` bundle into <repo>/data and you can run
the figures and the analysis experiments without setting anything. The raw
sources (Atlas, TIGER 2010, ACS) are only needed to rebuild geometry from
scratch and are fetched by download_data.py.
"""
import os

# repo root = parent of the lib/ folder that holds this file
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _dir(env, default):
    return os.path.abspath(os.environ.get(env, os.path.join(ROOT, default)))


DATA = _dir("ZEAL_DATA", "data")            # result JSON/npz + trained weights
RASTERS = os.path.join(DATA, "rasters")     # rasterized nested ladders (per state)
RAW = _dir("ZEAL_RAW", "raw")               # raw public downloads + derived geometry
GEOM = os.path.join(RAW, "geom")            # per-state .gpkg produced by geo_prep.py
ACS_DIR = os.path.join(RAW, "acs")          # ACS national tract table
TIGER = os.path.join(RAW, "tiger2010")      # TIGER/Line 2010 shapefiles
ATLAS = os.environ.get("ATLAS_CSV", os.path.join(RAW, "atlas", "tract_outcomes.csv"))


def data(name):    return os.path.join(DATA, name)      # a results / weights file
def raster(name):  return os.path.join(RASTERS, name)   # a rasterized-ladder npz
def geom(name):    return os.path.join(GEOM, name)       # a derived .gpkg
def acs(name):     return os.path.join(ACS_DIR, name)    # an ACS table


# make sure the output locations exist
for _d in (DATA, RASTERS):
    os.makedirs(_d, exist_ok=True)

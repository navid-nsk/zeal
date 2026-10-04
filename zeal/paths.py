"""paths.py -- the single place where the locations of the data and results packages are configured.

Two environment variables, both optional:
  ZEAL_DATA_ROOT     root of the data package     (default: ../data    relative to the repository root)
  ZEAL_RESULTS_ROOT  root of the results package  (default: ../results relative to the repository root)

Every script in this repository resolves its input and output files through the names defined here; no script contains
an absolute path.  The layout below mirrors the folders of the data and results packages (see their README files).
"""
import os

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DATA_ROOT = os.path.abspath(os.environ.get("ZEAL_DATA_ROOT") or os.path.join(REPO, "..", "data"))
RESULTS_ROOT = os.path.abspath(os.environ.get("ZEAL_RESULTS_ROOT") or os.path.join(REPO, "..", "results"))


def data(*parts):
    """a path inside the data package"""
    return os.path.join(DATA_ROOT, *parts)


def results(*parts):
    """a path inside the results package"""
    return os.path.join(RESULTS_ROOT, *parts)


def repo(*parts):
    """a path inside this repository"""
    return os.path.join(REPO, *parts)


# ---------------------------------------------------------------- data package: rasters, enclosures, fitted fields
def raster(state, N=512):
    """the pixel raster of a ladder: geo_st<state>_raster[_<N>].npz (state = '13' for Georgia, 'GMq4', 'GMbad', 'MXrwi')"""
    suf = "" if int(N) == 512 else f"_{int(N)}"
    return data("rasters", f"geo_st{state}_raster{suf}.npz")


RASTER = {"georgia": raster("13"), "gm_q4": raster("GMq4", 1024), "gm_bad": raster("GMbad", 1024), "mx_rwi": raster("MXrwi", 1024)}
FIELDS = tuple(RASTER)


def enclosure(field):
    """certified first-order enclosure of a fitted field (value and gradient boxes per pixel)"""
    return data("enclosures", f"first_order_{field}.npz")


def transport_lp(field):
    """pixel-level transport LP summary of a ladder (read by the C5 production run and the tile-budget runs)"""
    return data("transport_lp", f"transport_lp_{field}.json")


def field_file(name):
    """fitted neural fields and their fit records (e.g. 'local_cert_net.pt', 'uk_field_q4.json')"""
    return data("fields", name)


# ---------------------------------------------------------------- data package: ladders and source tables
def georgia_gpkg(state="13", level="tract"):
    """Georgia (state FIPS 13) tract / block-group geometry joined to the Opportunity Atlas field"""
    return data("georgia", f"geo_st{state}_{level}.gpkg")


GEORGIA_TRACTS = georgia_gpkg("13", "tract")
ATLAS_TRACT_OUTCOMES = data("sources", "opportunity_atlas", "tract_outcomes.csv")
US_TRACTS_NATIONAL = data("sources", "us_tracts", "us_tracts_NATIONAL_2016.csv")
TIGER2010_DIR = data("sources", "tiger2010")


def uk_dir(region="gm"):
    """England and Wales Census 2021 ladder of one region (OA -> LSOA -> MSOA); trailing separator included"""
    return os.path.join(DATA_ROOT, f"uk_{region}", "")


GM_DIR = uk_dir("gm")
MX_DIR = os.path.join(DATA_ROOT, "mexico", "")            # Mexico ladder (RWI tiles -> municipalities -> states)
SOURCES_DIR = os.path.join(DATA_ROOT, "sources", "")      # raw sources read by the preparation scripts (gadm/, worldpop/, rwi/, ...)
ELECTRICITY_DIR = data("electricity")                     # UCI ElectricityLoadDiagrams20112014
SANDBOX_DIR = data("sources", "sandbox")                  # only for the stand-alone demonstration in fields/zeal_m1.py (not deposited)

# ---------------------------------------------------------------- results package
EXP1, EXP2 = results("exp1"), results("exp2")
EXP3_ML, EXP3_ML_REAL = results("exp3_ml"), results("exp3_ml_real")

# ---------------------------------------------------------------- Lean development (in this repository)
LEAN_DIR = repo("lean")                                   # freeze record, certificate samples, checker logs
LEAN_PROJECT = repo("lean", "zeal")                       # the Lake project
CERTCHECK = os.path.join(LEAN_PROJECT, ".lake", "build", "bin", "certcheck.exe" if os.name == "nt" else "certcheck")


def lean_file(rel):
    """a file of the Lean development: looked up in the repository first (lean/...), then in the results package (lean/...)"""
    p = os.path.join(LEAN_DIR, rel)
    return p if os.path.exists(p) else results("lean", rel)


def rel(path):
    """package-relative name of a file, as recorded in the 'source' fields of the figure data:
    'results/...' (results package), 'data/...' (data package) or a repository-relative path"""
    p = os.path.abspath(path)
    for root, tag in ((RESULTS_ROOT, "results"), (DATA_ROOT, "data"), (REPO, None)):
        try:
            r = os.path.relpath(p, root)
        except ValueError:                                # different drive on Windows
            continue
        if not r.startswith(".."):
            r = r.replace("\\", "/")
            return f"{tag}/{r}" if tag else r
    return os.path.basename(p)

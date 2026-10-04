"""Fetch the England & Wales Census 2021 ladder for one group of local authorities: Output Areas (OA) -> Lower-layer
Super Output Areas (LSOA) -> Middle-layer (MSOA) -> Local Authority Districts (LAD), with OA boundaries (ONS Open
Geography Portal, BGC generalised, British National Grid) and OA-level census tables from Nomis (bulk zips).
Default region: Greater Manchester (10 districts). Output: <data root>/uk_<region>/
Usage: python uk_fetch.py [--region gm] [--tables ts067,ts037,ts001]
"""
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "zeal")); import paths  # data/results roots: ZEAL_DATA_ROOT, ZEAL_RESULTS_ROOT
import argparse, io, os, sys, time, zipfile, json
import requests, pandas as pd
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
p = argparse.ArgumentParser(); p.add_argument("--region", default="gm"); p.add_argument("--tables", default="ts067,ts037,ts001"); a = p.parse_args()
REGIONS = {"gm": ["Bolton", "Bury", "Manchester", "Oldham", "Rochdale", "Salford", "Stockport", "Tameside", "Trafford", "Wigan"]}
OUT = paths.uk_dir(a.region); os.makedirs(OUT, exist_ok=True)
S = "https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services/"
H = {"User-Agent": "Mozilla/5.0"}


def query_all(service, where, fields="*", geom=False, sr=27700):
    """Page through an ArcGIS FeatureServer layer (2,000 records per request)."""
    url = S + service + "/FeatureServer/0/query"; out, offset = [], 0
    while True:
        params = dict(where=where, outFields=fields, returnGeometry=str(geom).lower(), f="geojson" if geom else "json",
                      resultOffset=offset, resultRecordCount=2000, outSR=sr)
        r = requests.post(url, data=params, headers=H, timeout=300); r.raise_for_status(); j = r.json()
        feats = j.get("features", [])
        out.extend(feats); offset += len(feats)
        if not feats or not j.get("exceededTransferLimit", j.get("properties", {}).get("exceededTransferLimit", False)):
            if len(feats) < 2000:
                break
    return out


# 1. lookup OA -> LSOA -> MSOA -> LAD (England)
lads = ",".join(f"'{n}'" for n in REGIONS[a.region])
lu = query_all("OA21_LAD23_LSOA21_MSOA21_LEP23_EN_LU", f"LAD23NM IN ({lads})")
lu = pd.DataFrame([f["attributes"] for f in lu])
keep = [c for c in lu.columns if c.startswith(("OA21", "LSOA21", "MSOA21", "LAD23"))]
lu = lu[keep].drop_duplicates("OA21CD"); lu.to_csv(OUT + "lookup.csv", index=False)
print(f"lookup: {len(lu):,} OAs, {lu['LSOA21CD'].nunique():,} LSOAs, {lu['MSOA21CD'].nunique():,} MSOAs, {lu['LAD23CD'].nunique()} LADs")

# 2. OA boundaries (BGC, BNG) by LSOA code lists
oas = []
lsoas = sorted(lu["LSOA21CD"].unique())
for i in range(0, len(lsoas), 150):
    chunk = ",".join(f"'{c}'" for c in lsoas[i:i + 150])
    oas.extend(query_all("Output_Areas_2021_EW_BGC_V2", f"LSOA21CD IN ({chunk})", "OA21CD,LSOA21CD,BNG_E,BNG_N", geom=True))
    print(f"  boundaries: {len(oas):,} OAs", flush=True)
gj = {"type": "FeatureCollection", "crs": {"type": "name", "properties": {"name": "EPSG:27700"}}, "features": oas}
json.dump(gj, open(OUT + "oa_boundaries.geojson", "w", encoding="utf-8"))
codes = {f["properties"]["OA21CD"] for f in oas}; print(f"boundaries: {len(codes):,} OAs ({len(codes & set(lu['OA21CD'])):,} matched to the lookup)")

# 3. census tables (Nomis bulk zips; the OA file inside is '*-oa.csv')
for t in a.tables.split(","):
    z = OUT + f"census2021-{t}.zip"
    if not os.path.exists(z):
        r = requests.get(f"https://www.nomisweb.co.uk/output/census/2021/census2021-{t}.zip", headers=H, timeout=600); r.raise_for_status()
        open(z, "wb").write(r.content)
    with zipfile.ZipFile(z) as zf:
        names = zf.namelist(); oa = [n for n in names if n.endswith("-oa.csv")]
        print(f"{t}: {len(names)} files in zip; OA file: {oa}")
        if oa:
            df = pd.read_csv(zf.open(oa[0]))
            code_col = [c for c in df.columns if c.lower().startswith("geography code")][0]
            sub = df[df[code_col].isin(lu["OA21CD"])]
            sub.to_csv(OUT + f"{t}_oa.csv", index=False)
            print(f"   {len(sub):,} OA rows kept; columns: {list(df.columns)[:6]} ... ({len(df.columns)} columns)")
print("done ->", OUT)

"""
Data prep: whole-state nested ladder (BG in tract in county) on 2010 TIGER geometry,
joined to the Opportunity Atlas mobility field + population. Rich tract->county ladder.

Usage:  python geo_prep.py 13      # GA (default); 06 CA, 08 CO, 17 IL
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import paths
import numpy as np, pandas as pd, geopandas as gpd

STATE = sys.argv[1] if len(sys.argv) > 1 else "13"
TIG = paths.TIGER

# --- Atlas field + denominator (national) ---
at = pd.read_csv(paths.ATLAS)
at["GEOID"] = (at["state"].astype("Int64").astype(str).str.zfill(2)
               + at["county"].astype("Int64").astype(str).str.zfill(3)
               + at["tract"].astype("Int64").astype(str).str.zfill(6))
at = at[["GEOID", "kfr_pooled_pooled_p25", "pooled_pooled_count"]]
nat = pd.read_csv(paths.acs("us_tracts_NATIONAL_2016.csv"), dtype={"geoid": str})
nat["GEOID"] = nat["geoid"].str.zfill(11)
nat = nat[["GEOID", "pop_total", "median_hh_income"]]

# --- tract geometry (whole state), Albers equal-area; keep Atlas-covered ---
tr = gpd.read_file(f"zip://{TIG}/tl_2010_{STATE}_tract10.zip")[["GEOID10", "geometry"]]
tr = tr.rename(columns={"GEOID10": "GEOID"}).to_crs(5070)
tr = tr.merge(at, on="GEOID", how="left").merge(nat, on="GEOID", how="left")
tr = tr[tr["kfr_pooled_pooled_p25"].notna() & tr["pooled_pooled_count"].notna()].reset_index(drop=True)
tr["county"] = tr["GEOID"].str[:5]
print(f"state {STATE}: {len(tr)} Atlas-covered tracts, {tr['county'].nunique()} counties")
print(f"  kfr_p25 range [{tr['kfr_pooled_pooled_p25'].min():.3f},{tr['kfr_pooled_pooled_p25'].max():.3f}]"
      f"  pop {tr['pop_total'].notna().sum()}/{len(tr)}")

keep = set(tr["GEOID"])
bg = gpd.read_file(f"zip://{TIG}/tl_2010_{STATE}_bg10.zip")[["GEOID10", "geometry"]]
bg = bg.rename(columns={"GEOID10": "GEOID"}).to_crs(5070)
bg["tract"] = bg["GEOID"].str[:11]
bg = bg[bg["tract"].isin(keep)].reset_index(drop=True)
print(f"  bg: {len(bg)} (parent tracts {bg['tract'].nunique()}); ladder BG<tract<county")
# ladder sizes
print(f"  ladder counts -> bg {len(bg)}, tract {len(tr)}, county {tr['county'].nunique()}")

tr.to_file(paths.geom(f"geo_st{STATE}_tract.gpkg"), driver="GPKG")
bg.to_file(paths.geom(f"geo_st{STATE}_bg.gpkg"), driver="GPKG")
print(f"saved geo_st{STATE}_tract.gpkg, geo_st{STATE}_bg.gpkg")

"""Fetch the public raw inputs ZEAL needs into the locations defined by lib/paths.py.

This script downloads, into the directories declared in ``lib/paths.py``, the three
public datasets required to rebuild ZEAL's geometry from scratch:

1. Opportunity Atlas tract-level outcomes (``tract_outcomes.csv``) from Opportunity
   Insights -> directory of ``paths.ATLAS``.
   Source: https://opportunityinsights.org/data/  (ships as a .zip, unzipped here).

2. 2010 TIGER/Line census-tract and block-group shapefiles for state FIPS
   06 (CA), 08 (CO), 13 (GA), 17 (IL) -> ``paths.TIGER``.
   Source: https://www2.census.gov/geo/tiger/TIGER2010/  (each is a .zip, unzipped).

3. ACS 2012-2016 5-year national tract table giving total population and median
   household income -> ``paths.acs(...)``.
   The Census Data API now requires a key, so this step reads the key from the
   ``CENSUS_API_KEY`` environment variable. No key or token is ever hard-coded.

Files that already exist are skipped. Run directly:

    python download_data.py

For the ACS step, first set your free Census API key (https://api.census.gov/data/key_signup.html):

    # bash / zsh
    export CENSUS_API_KEY=your_key_here
    # PowerShell
    $env:CENSUS_API_KEY = "your_key_here"
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib"))
import paths

import csv
import io
import json
import urllib.error
import urllib.parse
import urllib.request
import zipfile


# --------------------------------------------------------------------------- #
# Configuration (public sources, verified June 2026)
# --------------------------------------------------------------------------- #

# Opportunity Atlas: "All Outcomes by Census Tract, Race, Gender and Parental
# Income Percentile" -- distributed as a zip that unpacks to tract_outcomes.csv.
ATLAS_ZIP_URL = "https://opportunityinsights.org/wp-content/uploads/2018/10/tract_outcomes.zip"

# 2010 TIGER/Line shapefiles. Two-digit code after the year is the state FIPS.
TIGER_BASE = "https://www2.census.gov/geo/tiger/TIGER2010"
STATE_FIPS = ["06", "08", "13", "17"]  # CA, CO, GA, IL
TIGER_LAYERS = {
    "tract": "{base}/TRACT/2010/tl_2010_{fips}_tract10.zip",
    "bg": "{base}/BG/2010/tl_2010_{fips}_bg10.zip",
}

# ACS 2012-2016 5-year detailed tables via the Census Data API.
#   B01003_001E = total population
#   B19013_001E = median household income (in 2016 inflation-adjusted dollars)
ACS_API_URL = (
    "https://api.census.gov/data/2016/acs/acs5"
    "?get=NAME,B01003_001E,B19013_001E&for=tract:*&in=state:*"
)
ACS_OUT_NAME = "acs2016_tract_pop_income.csv"

# Network politeness
USER_AGENT = "ZEAL-download_data/1.0 (research data fetch; urllib)"
TIMEOUT = 120


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

def _human(nbytes):
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024.0 or unit == "GB":
            return "%.1f %s" % (nbytes, unit)
        nbytes /= 1024.0


def fetch(url, timeout=TIMEOUT):
    """Return the raw bytes at ``url`` (with a polite User-Agent)."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def download_to(url, dest_path):
    """Download ``url`` to ``dest_path``; skip if it already exists.

    Returns True if a download happened, False if it was skipped.
    """
    if os.path.exists(dest_path):
        print("  [skip] already present: %s" % dest_path)
        return False
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    print("  [get ] %s" % url)
    data = fetch(url)
    tmp = dest_path + ".part"
    with open(tmp, "wb") as fh:
        fh.write(data)
    os.replace(tmp, dest_path)
    print("         -> %s (%s)" % (dest_path, _human(len(data))))
    return True


def unzip_to(zip_path, out_dir):
    """Extract ``zip_path`` into ``out_dir`` (created if needed)."""
    os.makedirs(out_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)
        return zf.namelist()


# --------------------------------------------------------------------------- #
# Step 1: Opportunity Atlas tract outcomes
# --------------------------------------------------------------------------- #

def get_atlas(summary):
    print("\n[1/3] Opportunity Atlas tract outcomes")
    atlas_dir = os.path.dirname(paths.ATLAS)
    os.makedirs(atlas_dir, exist_ok=True)

    if os.path.exists(paths.ATLAS):
        print("  [skip] already present: %s" % paths.ATLAS)
        summary.append(("Opportunity Atlas", paths.ATLAS, "skipped (exists)"))
        return

    zip_path = os.path.join(atlas_dir, "tract_outcomes.zip")
    try:
        download_to(ATLAS_ZIP_URL, zip_path)
    except urllib.error.URLError as exc:
        print("  [FAIL] could not download Atlas zip: %s" % exc)
        print("         See https://opportunityinsights.org/data/ -> "
              "'All Outcomes by Census Tract...' for the current link.")
        summary.append(("Opportunity Atlas", atlas_dir, "FAILED: %s" % exc))
        return

    print("  [unzip] %s" % zip_path)
    names = unzip_to(zip_path, atlas_dir)

    # Normalize the extracted CSV name to exactly paths.ATLAS if needed.
    if not os.path.exists(paths.ATLAS):
        csv_members = [n for n in names if n.lower().endswith(".csv")]
        if csv_members:
            extracted = os.path.join(atlas_dir, csv_members[0])
            if os.path.abspath(extracted) != os.path.abspath(paths.ATLAS):
                os.replace(extracted, paths.ATLAS)

    if os.path.exists(paths.ATLAS):
        try:
            os.remove(zip_path)
        except OSError:
            pass
        summary.append(("Opportunity Atlas", paths.ATLAS, "downloaded"))
    else:
        summary.append(
            ("Opportunity Atlas", atlas_dir,
             "unzipped, but tract_outcomes.csv not found; members=%s" % names))


# --------------------------------------------------------------------------- #
# Step 2: 2010 TIGER/Line tract + block-group shapefiles
# --------------------------------------------------------------------------- #

def get_tiger(summary):
    print("\n[2/3] 2010 TIGER/Line tract + block-group shapefiles")
    os.makedirs(paths.TIGER, exist_ok=True)
    got, skipped, failed = 0, 0, 0

    for fips in STATE_FIPS:
        for layer, pattern in TIGER_LAYERS.items():
            url = pattern.format(base=TIGER_BASE, fips=fips)
            zip_name = os.path.basename(url)
            zip_path = os.path.join(paths.TIGER, zip_name)
            # The .shp is the marker of a successful prior extraction.
            shp_name = zip_name[:-4] + ".shp"
            shp_path = os.path.join(paths.TIGER, shp_name)

            if os.path.exists(shp_path):
                print("  [skip] %s (FIPS %s, %s) already extracted"
                      % (shp_name, fips, layer))
                skipped += 1
                continue

            try:
                download_to(url, zip_path)
            except urllib.error.URLError as exc:
                print("  [FAIL] FIPS %s %s: %s" % (fips, layer, exc))
                failed += 1
                continue

            print("  [unzip] %s" % zip_name)
            unzip_to(zip_path, paths.TIGER)
            try:
                os.remove(zip_path)
            except OSError:
                pass
            got += 1

    summary.append(
        ("TIGER 2010 (FIPS %s)" % ",".join(STATE_FIPS), paths.TIGER,
         "%d downloaded, %d skipped, %d failed" % (got, skipped, failed)))


# --------------------------------------------------------------------------- #
# Step 3: ACS 2012-2016 5-year tract table (pop_total + median income)
# --------------------------------------------------------------------------- #

def get_acs(summary):
    print("\n[3/3] ACS 2012-2016 5-year national tract table")
    out_path = paths.acs(ACS_OUT_NAME)

    if os.path.exists(out_path):
        print("  [skip] already present: %s" % out_path)
        summary.append(("ACS 2016 5-yr tract", out_path, "skipped (exists)"))
        return

    key = os.environ.get("CENSUS_API_KEY")
    if not key:
        print("  [SKIP] CENSUS_API_KEY is not set; cannot query the Census API.")
        print("         The 2016 ACS 5-year detailed tables are only available")
        print("         keylessly in bulk via very large summary files; the")
        print("         tract-by-tract national pull is far simpler via the API.")
        print("         Get a free key at https://api.census.gov/data/key_signup.html")
        print("         then set it and re-run, e.g.:")
        print("           export CENSUS_API_KEY=your_key_here      # bash/zsh")
        print("           $env:CENSUS_API_KEY = \"your_key_here\"    # PowerShell")
        summary.append(("ACS 2016 5-yr tract", out_path,
                        "SKIPPED: set CENSUS_API_KEY and re-run"))
        return

    os.makedirs(paths.ACS_DIR, exist_ok=True)
    url = ACS_API_URL + "&key=" + urllib.parse.quote(key)
    print("  [get ] %s" % ACS_API_URL + "&key=***")
    try:
        raw = fetch(url)
    except urllib.error.URLError as exc:
        print("  [FAIL] Census API request failed: %s" % exc)
        summary.append(("ACS 2016 5-yr tract", out_path, "FAILED: %s" % exc))
        return

    # The API returns JSON: a header row followed by data rows. Convert to CSV
    # with readable column names so downstream code does not depend on the API.
    try:
        rows = json.loads(raw.decode("utf-8"))
    except ValueError as exc:
        print("  [FAIL] could not parse Census API response as JSON: %s" % exc)
        snippet = raw[:300].decode("utf-8", "replace")
        print("         response began: %s" % snippet)
        summary.append(("ACS 2016 5-yr tract", out_path, "FAILED: bad JSON"))
        return

    header = rows[0]
    rename = {"B01003_001E": "pop_total", "B19013_001E": "median_hh_income"}
    out_header = [rename.get(c, c) for c in header]

    tmp = out_path + ".part"
    with io.open(tmp, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(out_header)
        writer.writerows(rows[1:])
    os.replace(tmp, out_path)
    n = len(rows) - 1
    print("         -> %s (%d tract rows)" % (out_path, n))
    summary.append(("ACS 2016 5-yr tract", out_path, "downloaded (%d rows)" % n))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    print("ZEAL raw-data download")
    print("  ATLAS -> %s" % paths.ATLAS)
    print("  TIGER -> %s" % paths.TIGER)
    print("  ACS   -> %s" % paths.ACS_DIR)

    summary = []
    get_atlas(summary)
    get_tiger(summary)
    get_acs(summary)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, where, status in summary:
        print("  %-26s %s" % (name + ":", status))
        print("  %-26s %s" % ("", where))
    print("=" * 70)


if __name__ == "__main__":
    main()

"""make_source_data.py -- one Source Data workbook for the journal: figures/SourceData.xlsx with one sheet per figure panel
(Fig1a, Fig1b, ..., ED1a, ...), built from the builders' tidy F{n}_source_data.csv files. Large raster panels (map pixels,
thumbnail arrays) are thinned to the per-cell / per-item values that the panel draws; a README sheet lists every sheet,
its figure, the number of rows and the result files the values come from. No local paths appear in the sheets (the
'source' column is reduced to the result-file basename)."""
import os, re, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "figures", "SourceData.xlsx")
STEMS = [("F1", "Fig1"), ("F2", "Fig2"), ("F3", "Fig3"), ("F4", "Fig4"), ("F5", "Fig5"), ("F6", "Fig6"), ("ED1", "ED1"), ("ED2", "ED2"), ("ED3", "ED3")]
MAX_ROWS = 60000          # Excel sheets hold 1,048,576 rows; keep every sheet readable
readme = []
with pd.ExcelWriter(OUT, engine="openpyxl") as xw:
    for stem, label in STEMS:
        df = pd.read_csv(os.path.join(HERE, f"{stem}_source_data.csv"), dtype=str, keep_default_na=False)
        if "source" in df.columns:
            df["source"] = df["source"].map(lambda s: os.path.basename(str(s)))
        pcol = "panel"
        panels = list(dict.fromkeys(df[pcol].tolist()))
        for p in panels:
            sub = df[df[pcol] == p].drop(columns=[pcol])
            sub = sub.loc[:, [c for c in sub.columns if sub[c].astype(str).str.len().gt(0).any()]]
            thinned = ""
            if len(sub) > MAX_ROWS:
                step = -(-len(sub) // MAX_ROWS); sub = sub.iloc[::step]; thinned = f"thinned 1/{step} (raster/thumbnail pixels)"
            sheet = f"{label}{re.sub(r'[^A-Za-z0-9]', '', str(p))}"[:31]
            sub.to_excel(xw, sheet_name=sheet, index=False)
            srcs = sorted(set(sub["source"].tolist())) if "source" in sub.columns else []
            readme.append(dict(sheet=sheet, figure=label, panel=p, rows=len(sub), note=thinned, result_files="; ".join(srcs)[:1000]))
    pd.DataFrame(readme).to_excel(xw, sheet_name="README", index=False)
    wb = xw.book; wb.move_sheet("README", offset=-(len(wb.sheetnames) - 1))
print("wrote", OUT, len(readme), "sheets")
for r in readme: print(f"  {r['sheet']:<14} {r['rows']:>7} rows {r['note']}")

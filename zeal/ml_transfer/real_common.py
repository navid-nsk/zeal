"""real_common.py -- the REAL-data variant of the exp3_ml set-up (exp3_ml_real).  Imports the mlt_common machinery unchanged
(W = 168, T = 1008, N_ENT = 12, EPS_GRID, LAGS_DYADIC, models, Incr) and only replaces the data source and the calendar split.

Data (public, downloaded 2026-10-04): UCI Machine Learning Repository, "ElectricityLoadDiagrams20112014" (dataset id 321),
  URL  https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip
  zip  261,335,609 bytes, SHA-256 f6c4d0e0df12ecdb9ea008dd6eef3518adb52c559d04a9bac2e1b81dcfc8d4e1
  LD2011_2014.txt 710,998,915 bytes, SHA-256 d51565f2cb5a6b768d06ba1bbd3c084c6e2f3aab07f00c6f2dcb80e90175124b
  370 Portuguese clients, kW per 15 min, 2011-01-01 00:15 .. 2015-01-01 00:00 (time stamps = END of each quarter-hour).
Declared processing (fixed before any model, verifier or certificate was run):
  hourly value of hour [h, h+1)  = mean of the four quarter-hour values stamped h:15, h:30, h:45, (h+1):00   (kW)
  DST repair: on the March change days (2012-03-25, 2013-03-31) the hour 01:00-02:00 is anomalous in the raw file (many clients
     0; hour total measured at 0.30-0.31 of the neighbouring hours) -> replaced by the mean of the adjacent hours; on the October
     change days (2012-10-28, 2013-10-27) the hour 01:00-02:00 aggregates two hours (total measured at 1.40-1.47 x neighbours)
     -> divided by 2.  Both days lie in the training period; neither evaluation window contains a DST day.
  hour index s = 0 at 2011-12-24 00:00; data span used 2011-12-24 00:00 .. 2014-08-13 00:00 (exclusive)
  split (calendar): train outputs 2012-01-01 .. 2013-08-31, early stopping 2013-09-01 .. 2013-12-31 (as specified),
  evaluation windows (T = 1008 hourly outputs each, all labels observed):
     'jan2014' = 2014-01-01 00:00 .. 2014-02-11 23:00   (the specified window; label rate turned out to be 0.019)
     'jul2014' = 2014-07-01 00:00 .. 2014-08-11 23:00   (DECLARED SECOND WINDOW, added after the jan2014 label rate (about 0.02,
                 positives in only 2-3 of the 12 clients) was seen and BEFORE any model was trained or any certificate computed: the first
                 6 summer weeks after the early-stopping period, the season in which the training caps are exceeded)
  entities        : among clients whose repaired hourly load is strictly positive in EVERY hour of the data span, the 12 with the
                    largest mean training-period load (the same 12 clients and the same trained models serve both windows)
  task            : identical to exp3_ml: label at hour t = 1{ max load over (t, t+24] > cap_j }, cap_j = 80th percentile of the
                    client's training daily maxima; input = past 168 h of the client's z-scored load (training mean / sd)
"""
import os as _os, sys as _sys
try:
    from .. import paths                    # imported as part of the package
except ImportError:
    _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..")); import paths
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlt_common
from mlt_common import W, T, N_ENT, EPS_GRID, LAGS_DYADIC, jdump

DATA_DIR = paths.ELECTRICITY_DIR
REAL_OUT = paths.EXP3_ML_REAL
os.makedirs(REAL_OUT, exist_ok=True)
RAW = os.path.join(DATA_DIR, "LD2011_2014.txt"); HOURLY = os.path.join(DATA_DIR, "hourly_2011_2014.npz")
S0 = pd.Timestamp("2011-12-24 00:00")


def hidx(ts):
    return int((pd.Timestamp(ts) - S0) / pd.Timedelta(hours=1))


TRAIN_START, TRAIN_END, VAL_END = hidx("2012-01-01"), hidx("2013-09-01"), hidx("2014-01-01")
SPAN_END = hidx("2014-08-13")
WINDOWS = {"jan2014": hidx("2014-01-01"), "jul2014": hidx("2014-07-01")}
DST_MARCH = ["2012-03-25", "2013-03-31"]; DST_OCT = ["2012-10-28", "2013-10-27"]


def window_dir(window):
    d = os.path.join(REAL_OUT, window); os.makedirs(d, exist_ok=True); return d


def use_real_out(window):
    """route every mlt_common path (models, nominal outputs, boxes, results) to exp3_ml_real/<window>"""
    mlt_common.OUT = window_dir(window)
    return mlt_common.OUT


def hourly():
    """parse the raw file once into an hourly array (cached in DATA_DIR)"""
    if os.path.exists(HOURLY):
        z = np.load(HOURLY, allow_pickle=True); return pd.DataFrame(z["x"], index=pd.DatetimeIndex(z["idx"]), columns=list(z["cols"]))
    df = pd.read_csv(RAW, sep=";", decimal=",", index_col=0, parse_dates=True, dtype=np.float64)
    key = (df.index - pd.Timedelta(minutes=15)).floor("h")                       # quarter stamped h:15..(h+1):00 -> hour h
    cnt = pd.Series(1, index=df.index).groupby(key).sum()
    assert (cnt == 4).all(), "every hour must have four quarter-hours"
    hr = df.groupby(key).mean()
    np.savez_compressed(HOURLY, x=hr.values.astype(np.float64), idx=hr.index.values, cols=np.array(hr.columns))
    return hr


def make_real_data():
    hr = hourly()
    rep = {}
    for d in DST_MARCH:
        h1 = pd.Timestamp(d) + pd.Timedelta(hours=1); nb = 0.5 * (hr.loc[h1 - pd.Timedelta(hours=1)] + hr.loc[h1 + pd.Timedelta(hours=1)])
        rep[f"march_{d}_hour1_total_over_neighbours"] = float(hr.loc[h1].sum() / nb.sum())
        hr.loc[h1] = nb
    for d in DST_OCT:
        h1 = pd.Timestamp(d) + pd.Timedelta(hours=1)
        nb = 0.5 * (hr.loc[h1 - pd.Timedelta(hours=1)].sum() + hr.loc[h1 + pd.Timedelta(hours=1)].sum())
        rep[f"october_{d}_hour1_total_over_neighbours"] = float(hr.loc[h1].sum() / nb)
        hr.loc[h1] = hr.loc[h1] / 2.0
    span = hr.loc[S0:S0 + pd.Timedelta(hours=SPAN_END - 1)]
    assert len(span) == SPAN_END
    X = span.values.T                                                             # [370, SPAN_END]
    eligible = np.where((X > 0).all(axis=1))[0]
    mtrain = X[:, TRAIN_START:TRAIN_END].mean(1)
    chosen = eligible[np.argsort(-mtrain[eligible])[:N_ENT]]
    loads = X[chosen]
    mu = loads[:, TRAIN_START:TRAIN_END].mean(1, keepdims=True); sd = loads[:, TRAIN_START:TRAIN_END].std(1, keepdims=True)
    z = (loads - mu) / sd
    dmax = loads[:, TRAIN_START:TRAIN_END].reshape(N_ENT, -1, 24).max(2)
    cap = np.quantile(dmax, 0.8, axis=1)
    lab = np.full((N_ENT, SPAN_END), np.nan)
    for t in range(SPAN_END - 24):
        lab[:, t] = (loads[:, t + 1:t + 25].max(1) > cap).astype(float)
    meta = [dict(client=str(span.columns[c]), mean_train_kW=float(mtrain[c]), cap_kW=float(cp)) for c, cp in zip(chosen, cap)]
    rep.update(n_clients=X.shape[0], n_eligible=int(len(eligible)), chosen=[str(span.columns[c]) for c in chosen],
               span=[str(S0), str(S0 + pd.Timedelta(hours=SPAN_END - 1))], TRAIN_START=TRAIN_START, TRAIN_END=TRAIN_END, VAL_END=VAL_END,
               WINDOWS=WINDOWS, SPAN_END=SPAN_END)
    return dict(load=loads, z=z, lab=lab, cap=cap, mu=mu.ravel(), sd=sd.ravel(), meta=meta, report=rep)


def eval_series_real(D, window):
    """z-scored evaluation series per client: hours E - W + 1 .. E + T - 1 (length T + W - 1), E = WINDOWS[window]"""
    E = WINDOWS[window]
    return D["z"][:, E - W + 1:E + T].astype(np.float32)

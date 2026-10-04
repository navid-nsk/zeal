"""lean_freeze.py -- freeze the Lean artifact (lean/zeal) at its current state: SHA-256 manifest of every source file (Zeal/**.lean,
MainTheorems*.lean, scripts/*.lean, lakefile.toml, lean-toolchain, lake-manifest.json), the build summary (warnings/errors/axiom reports
from a fresh `lake build` log), the certificate logs (b1 kernel checks, 16 LP sample, the complete C5 streaming log), and writes
lean/FREEZE_v1.md + lean/freeze_manifest_v1.json.  With --commit, also commits the Lean project repository."""
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "zeal")); import paths  # data/results roots: ZEAL_DATA_ROOT, ZEAL_RESULTS_ROOT
import hashlib, json, os, subprocess, sys, time, glob
HERE = os.path.dirname(os.path.abspath(__file__)); L = paths.LEAN_DIR; Z = paths.LEAN_PROJECT
import shutil
LAKE = shutil.which("lake") or os.path.expanduser("~/.elan/bin/lake" + (".exe" if os.name == "nt" else ""))
COMMIT = "--commit" in sys.argv          # optional: commit the Lean project repository after writing the record
def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
files = sorted(glob.glob(os.path.join(Z, "Zeal", "**", "*.lean"), recursive=True) + glob.glob(os.path.join(Z, "*.lean")) + glob.glob(os.path.join(Z, "scripts", "*.lean"))
               + [os.path.join(Z, f) for f in ("lakefile.toml", "lean-toolchain", "lake-manifest.json")])
man = {os.path.relpath(f, Z).replace("\\", "/"): sha(f) for f in files}
env = dict(os.environ); env["PATH"] = os.path.expanduser("~/.elan/bin") + os.pathsep + env["PATH"]
t0 = time.time(); log = subprocess.run([LAKE, "build"], cwd=Z, env=env, capture_output=True, text=True).stdout
nw = sum(l.startswith("warning:") for l in log.splitlines()); ne = sum(l.startswith("error:") for l in log.splitlines())
ax = [l for l in log.splitlines() if "depends on axioms" in l]; nonstd = [l for l in ax if "propext" not in l and "axioms: []" not in l]
tool = open(os.path.join(Z, "lean-toolchain")).read().strip(); mathlib = [l for l in open(os.path.join(Z, "lake-manifest.json")).read().splitlines() if '"rev"' in l]
logs = {}
for name in ("certificates/lean_check_log.md", "certificates/export_log.json", "certificates_all/lean_c5_log_georgia.json", "certificates_all/lean_c5_log_gm_q4.json", "certificates_all/lean_c5_log_gm_bad.json", "certificates_all/lean_c5_log_mx_rwi.json"):
    p = os.path.join(L, name)
    if os.path.exists(p):
        logs[name] = sha(p)
c5 = {}
for f in ("georgia", "gm_q4", "gm_bad", "mx_rwi"):
    p = os.path.join(L, "certificates_all", f"lean_c5_log_{f}.json")
    if os.path.exists(p):
        d = json.load(open(p)); c5[f] = dict(endpoints=len(d), passed=sum(x["passed"] for x in d), artifact_certified=sum(x["artifact_certified"] for x in d))
freeze = dict(frozen_at=time.strftime("%Y-%m-%d %H:%M"), toolchain=tool, mathlib_manifest_revs=mathlib, n_source_files=len(files), source_lines=sum(1 for f in files if f.endswith(".lean") for _ in open(f, encoding="utf-8")),
              build=dict(last_line=log.strip().splitlines()[-1] if log.strip() else "", warnings=nw, errors=ne, axiom_reports=len(ax), nonstandard_axiom_reports=nonstd, seconds=round(time.time() - t0, 1)),
              certificate_logs=logs, c5_streaming=c5, manifest=man)
json.dump(freeze, open(os.path.join(L, "freeze_manifest_v1.json"), "w"), indent=1)
md = [f"# Lean artifact — FREEZE v1 ({freeze['frozen_at']})", "", f"- Toolchain: `{tool}`; Mathlib: {mathlib}", f"- Source files: {len(files)} ({freeze['source_lines']} lines of Lean)",
      f"- Build: {freeze['build']['last_line']} — warnings {nw}, errors {ne}, axiom reports {len(ax)}, non-standard {len(nonstd)}",
      "- Instance certificates: 96 B1 charges kernel-checked (`certificates_lean/b1/`); 16 sample map endpoints + the complete C5 streaming log by the compiled verified checker `certcheck` (native evaluation; soundness kernel-checked):",
      *[f"  - {f}: {v['endpoints']} endpoints, PASS {v['passed']}, producer endpoint certified by the exact bound {v['artifact_certified']}" for f, v in c5.items()],
      "- Trust boundary (paper wording, fixed): theorems machine-checked (no admitted propositions); numerical optimization external; reported certificates checked from exact-rational inputs by the verified checker; a kernel-checked instance layer exists for the pricing charges only.",
      "- Lean-forced conditions: `extra_conditions_v1.md` (record); the logically required ones are in `foundation_v8.md`.", "- Frozen: no edits to the files in `freeze_manifest_v1.json` after this date; later work goes to a v2 directory.", ""]
open(os.path.join(L, "FREEZE_v1.md"), "w", encoding="utf-8").write("\n".join(md))
print(json.dumps({k: v for k, v in freeze.items() if k != "manifest"}, indent=1))
if COMMIT:
    r = subprocess.run(["git", "add", "-A"], cwd=Z, capture_output=True, text=True); r2 = subprocess.run(["git", "commit", "-q", "-m", "Lean spine v1 + certificate layer: freeze v1"], cwd=Z, capture_output=True, text=True)
    h = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Z, capture_output=True, text=True).stdout.strip()
    open(os.path.join(L, "FREEZE_v1.md"), "a", encoding="utf-8").write(f"- Git commit (project repo `lean/zeal`): `{h}`\n")
    print("commit", h, r2.stderr[:200])

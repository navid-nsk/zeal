"""real_run.py -- run an UNCHANGED exp3_ml script (s2_verify.py, s4_witness.py, s5_tables.py) on the real data of one evaluation
window: mlt_common.OUT is routed to exp3_ml_real/<window> before the script executes, so the script reads that window's models,
nominal outputs, boxes and certificates and writes its outputs there.  Nothing in the original scripts is modified.
Usage: python real_run.py <window> <script.py> [script arguments ...]
   e.g. python real_run.py jul2014 s2_verify.py --model mlp --phase crown"""
import os, sys, runpy
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import real_common
window, script = sys.argv[1], sys.argv[2]
assert window in real_common.WINDOWS and script in ("s2_verify.py", "s4_witness.py", "s5_tables.py")
out = real_common.use_real_out(window)
if script == "s4_witness.py":
    import s3_certify; s3_certify.OUT = out
sys.argv = [os.path.join(HERE, script)] + sys.argv[3:]
runpy.run_path(os.path.join(HERE, script), run_name="__main__")

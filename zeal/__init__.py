"""ZEAL: certified ranges of aggregated quantities over every admissible aggregation of a field.

The package holds the certifier core (transport_core, outer_x, synth), the pricing and certificate layers
(pricing_certified, pricing_verify, cert_export, cert_run_all), the independent checkers (check_*), the
pooling-robustness transfer (ml_transfer) and the single path configuration (paths).  The modules are also
importable as flat modules when this directory is on sys.path, which is how the scripts in experiments/,
fields/ and figures/ use them.
"""
__version__ = "1.0.0"
from . import paths  # noqa: F401

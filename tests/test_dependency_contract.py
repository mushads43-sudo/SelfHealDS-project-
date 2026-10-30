"""Guards the runtime dependency contract declared in requirements.txt.

backend/repair.py writes numeric repairs into object columns; pandas 3.x uses
a strict string dtype by default and raises TypeError on those writes. This
test fails fast if an environment drifts onto an unsupported pandas.
"""
import pandas as pd


def test_pandas_major_version_supported():
    assert int(pd.__version__.split(".")[0]) < 3, f"pandas {pd.__version__} is not supported (requirements.txt pins <3)"

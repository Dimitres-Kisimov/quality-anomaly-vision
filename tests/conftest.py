"""Process-level test setup.

pandas 3.0 imports pyarrow lazily the first time a string-typed frame is
built. When that lazy DLL load happens late in the pytest process -- after
torch has spun up its thread pools and matplotlib has rendered figures -- it
intermittently dies with a Windows access violation (observed on Python
3.14 + pandas 3.0.3 + torch 2.11 during development; a fresh process imports
the same modules in any order without trouble). Importing pyarrow here, first
thing in a quiet interpreter, turns the later lazy import into a no-op.
Harmless when pyarrow is not installed.
"""

try:
    import pyarrow  # noqa: F401
except ImportError:
    pass

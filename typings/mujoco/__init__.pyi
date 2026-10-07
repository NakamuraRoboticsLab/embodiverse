# mujoco's Python bindings are compiled modules without type stubs, so
# pyright cannot see the names they define. This stub types every name in
# the package as Any; the tests check how the adapter uses them.
from typing import Any

def __getattr__(name: str) -> Any: ...

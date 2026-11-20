"""airspace — a gesture-controlled spatial desktop powered by computer vision."""

import sys as _sys


def _disable_lazy_imports() -> None:
    """Make Python 3.15+ lazy imports eager.

    With PySide6 6.12 loaded, Python 3.15's lazy imports (PEP 810) are no longer
    resolved on first use: standard-library names such as
    ``concurrent.futures.ThreadPoolExecutor`` stay unresolved ``lazy_import``
    placeholders, and MediaPipe fails with "'lazy_import' object is not
    callable". Vetoing every lazy import keeps them ordinary imports, and
    anything already left lazy is resolved now, before PySide6 is imported.
    """
    set_filter = getattr(_sys, "set_lazy_imports_filter", None)
    if set_filter is None:
        return
    set_filter(lambda *_: False)
    for module in list(_sys.modules.values()):
        namespace = getattr(module, "__dict__", None)
        if not isinstance(namespace, dict):
            continue
        for name, value in list(namespace.items()):
            if type(value).__name__ == "lazy_import":
                try:
                    getattr(module, name)
                except Exception:  # noqa: BLE001 - leave anything that cannot resolve
                    pass


_disable_lazy_imports()

__version__ = "0.1.0"

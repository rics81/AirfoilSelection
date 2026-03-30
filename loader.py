# -*- coding: utf-8 -*-
"""
pipeline/loader.py
==================
Dynamic loader for numbered pipeline modules.

Python's import system does not allow identifiers that start with a digit,
so modules named ``01_scraper.py``, ``02_analysis.py`` … are imported via
``importlib`` rather than a standard ``import`` statement.

Usage (from main.py or interactive session)
-------------------------------------------
    from pipeline.loader import load_step
    scraper = load_step(1)   # loads pipeline/01_scraper.py
    scraper.run(cfg)
"""

import importlib.util
import os
import sys

_STEP_FILES = {
    1: "01_scraper.py",
    2: "02_analysis.py",
    3: "03_features.py",
    4: ("04_ranking_pca.py", "05_ranking_ahp.py", "06_ranking_ahp_pca.py"),
    5: "07_proppaga.py",
}

_MODULE_NAMES = {
    "01_scraper.py":          "pipeline_scraper",
    "02_analysis.py":         "pipeline_analysis",
    "03_features.py":         "pipeline_features",
    "04_ranking_pca.py":      "pipeline_ranking_pca",
    "05_ranking_ahp.py":      "pipeline_ranking_ahp",
    "06_ranking_ahp_pca.py":  "pipeline_ranking_ahp_pca",
    "07_proppaga.py":         "pipeline_proppaga",
}

_PIPELINE_DIR = os.path.dirname(__file__)


def _import_file(filename: str):
    """Import a single pipeline module file and return the module object."""
    module_name = _MODULE_NAMES[filename]

    # Return cached module if already loaded
    if module_name in sys.modules:
        return sys.modules[module_name]

    file_path = os.path.join(_PIPELINE_DIR, filename)
    spec   = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_step(step: int):
    """
    Load a pipeline step module (or tuple of modules for step 4).

    Parameters
    ----------
    step : int
        Pipeline step number (1–5).

    Returns
    -------
    module or tuple of modules
        For step 4, returns (pca_module, ahp_module, ahp_pca_module).
        For all other steps, returns a single module.
    """
    entry = _STEP_FILES[step]
    if isinstance(entry, tuple):
        return tuple(_import_file(f) for f in entry)
    return _import_file(entry)

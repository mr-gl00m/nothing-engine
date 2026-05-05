"""Atomic HDF5 writes for the experiment scripts.

Opens a sibling ``<dst>.tmp`` file for writing and only publishes the
destination via :func:`os.replace` on clean exit. On any exception the
partial temp file is removed before propagating, leaving the prior
``dst`` intact. Refuses to clobber an existing destination unless
``overwrite=True``.

Mirrors :func:`nothing_engine.experiments.runner._atomic_replace` so the
phyllotaxis pipeline shares the publish semantics of ``runner.run``.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import h5py

logger = logging.getLogger(__name__)


def _tmp_path_for(dst: Path) -> Path:
    return dst.with_suffix(dst.suffix + ".tmp")


@contextmanager
def atomic_h5_write(dst: Path, overwrite: bool = False) -> Iterator[h5py.File]:
    dst = Path(dst)
    if dst.exists() and not overwrite:
        raise FileExistsError(
            f"Output file already exists: {dst}. Pass overwrite=True to replace."
        )
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path_for(dst)
    f = h5py.File(tmp, "w")
    try:
        yield f
    except BaseException:
        try:
            f.close()
        except OSError as exc:
            logger.warning("Failed to close partial HDF5 temp %s: %s", tmp, exc)
        try:
            tmp.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("Failed to remove partial HDF5 temp %s: %s", tmp, exc)
        raise
    f.close()
    os.replace(str(tmp), str(dst))

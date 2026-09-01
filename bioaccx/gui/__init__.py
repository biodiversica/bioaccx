"""Browser front end for bioaccx.

The GUI is a second way in, not a replacement for the CLI: it writes config
files, and (from Phase 2) launches the same commands a user would have typed.
It never imports :mod:`bioaccx.train`, so TensorFlow never loads in the server
process and a broken GUI cannot break a training run.

Requires the optional ``[gui]`` extra::

    uv tool install "bioaccx[cpu,gui]"
"""

__all__ = ["schema", "configio"]

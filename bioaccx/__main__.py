import signal

from bioaccx.cli import main

# The GUI cancels a run on Windows with Ctrl+Break (see gui/jobs.py), which
# would otherwise end the process without the clean "Interrupted." exit.
if hasattr(signal, "SIGBREAK"):
    signal.signal(signal.SIGBREAK, signal.default_int_handler)

main()

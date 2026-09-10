#!/usr/bin/env python3
"""Entry point for the packaged app: the window, or the command line.

freeze_support runs before anything heavy is imported. A worker process in a
frozen build starts this same executable again, and freeze_support recognises
that, runs the worker, and never returns. Importing Qt above this line would load
the whole toolkit in every worker for nothing.

With no arguments this opens the window, which is what a double click does.
With arguments it is the command line, so that a machine with no Python on it
can still run a batch and be checked:

    FaceBlur.exe footage -o out --mask face,screen --workers 4

The build is windowed, so it owns no console and its output would go nowhere.
`_attach_console` borrows the console of whatever started it, when there is
one, so that running it from a command prompt prints the way `cli.py` does.
When there is none, printing stays silent and the audit record beside each
copy is the output.
"""
import multiprocessing
import sys


def _attach_console() -> None:
    """Give this process somewhere to print, when it was started from a shell.

    A windowed build owns no console, and PyInstaller leaves `sys.stdout` as
    None, where `print` is silently a no-op. Two things can be true instead,
    and both are tried in this order:

    - the shell redirected the output to a file or a pipe, so the process
      already holds a valid file descriptor 1 and only Python's view of it is
      missing;
    - the shell has a console, which Windows will lend to a child process
      once. `AttachConsole` asks for it and `CONOUT$` opens it.

    Neither failing is worth reporting. The run is what matters, and it writes
    its audit record beside every copy either way.
    """
    if sys.stdout is not None and sys.stderr is not None:
        return
    if sys.platform.startswith("win"):
        try:
            import ctypes

            ATTACH_PARENT_PROCESS = -1
            ctypes.windll.kernel32.AttachConsole(ATTACH_PARENT_PROCESS)
        except Exception:
            pass
    for name, fd in (("stdout", 1), ("stderr", 2)):
        if getattr(sys, name) is not None:
            continue
        for target in (fd, "CONOUT$"):
            try:
                stream = open(target, "w", encoding="utf-8", buffering=1,
                              closefd=not isinstance(target, int))
            except OSError:
                continue
            setattr(sys, name, stream)
            break


if __name__ == "__main__":
    multiprocessing.freeze_support()

    if len(sys.argv) > 1:
        _attach_console()
        from cli import main
    else:
        from ui.app import main

    sys.exit(main())

#!/usr/bin/env python3
"""Entry point for the packaged app.

freeze_support runs before anything heavy is imported. A worker process in a
frozen build starts this same executable again, and freeze_support recognises
that, runs the worker, and never returns. Importing Qt above this line would load
the whole toolkit in every worker for nothing.
"""
import multiprocessing
import sys

if __name__ == "__main__":
    multiprocessing.freeze_support()

    from ui.app import main

    sys.exit(main())

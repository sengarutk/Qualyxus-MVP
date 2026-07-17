"""Database compatibility layer.

The codebase historically stored SQLAlchemy models under `src.logging.db`.
Some deployments/tools expect a `src.database` package.

This module keeps imports stable without changing runtime behavior.
"""

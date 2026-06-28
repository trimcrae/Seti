"""Seti-WD: a contamination-robust search for infrared-excess technosignatures
around white dwarfs.

The package is organised as a linear *funnel*:

    acquire  -> sed -> contamination -> discriminate -> stats

Each stage reads and writes parquet checkpoints so the pipeline is idempotent
and reproducible offline from a small committed sample.
"""

__version__ = "0.1.0"

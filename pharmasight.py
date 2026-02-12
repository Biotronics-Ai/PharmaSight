"""
Convenience facade for the core ForenSight components.
Importing `forensight` re-exports the main classes without needing
to reference individual modules.
"""

from models import (  # noqa: F401
    SampleLoader,
    SequenceMatcher,
    DoubleSampleComparator,
    ParserFactory,
    STRSearcher,
    MolSample,
)
from kernel import KernelMatrix  # noqa: F401

#from utils import calculate_effort  # noqa: F401

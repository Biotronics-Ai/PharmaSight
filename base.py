import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional, AsyncGenerator, Iterable
from tqdm.auto import tqdm


# ===========================================================
# BASE COMPONENT (common logging for all classes)
# ===========================================================

class BaseComponent(ABC):
    """Looging begavior for all classes."""

    def _log_state(self, state: str, message: str = "", error: str = None):
        """Logs the message and state in a uniform format."""
        cls = self.__class__.__name__
        if error:
            print(f"[{cls}] ❌ {state}: {message} | ERROR: {error}")
        else:
            print(f"[{cls}] ✅ {state}: {message}")

    def _progress(self, iterable: Iterable, **tqdm_kwargs):
        """
        Lightweight progress wrapper. Uses tqdm when available; otherwise returns the
        iterable unchanged. Keeps a minimal surface so subclasses can instrument loops
        without hard dependency.
        """
        if tqdm is None:
            return iterable
        return tqdm(iterable, **tqdm_kwargs)



# ===========================================================
# BASE PARSER (template + logging)
# ===========================================================

import os
import tempfile
import numpy as np
import gc
from typing import AsyncGenerator, List
from abc import abstractmethod

class BaseParser(BaseComponent):
    """Base Abstract class for reading the DNA/protein file formats with memory map support."""

    def __init__(self, mem_map_dir: str = None):
        """
        Parameters
        ----------
        mem_map_dir : str, optional
            Directory path where temporary memory-mapped files will be stored.
            If None, system temp dir will be used.
        """
        self.mem_map_dir = mem_map_dir or tempfile.gettempdir()
        os.makedirs(self.mem_map_dir, exist_ok=True)
        self._active_memmaps = []

    async def read(self, filepath: str, **kwargs):
        """Main reading workflow with memory map support, using unified extract_all."""
        try:
            self._log_state("STARTED", f"Reading {filepath}")
            raw_data = await self._read_file(filepath)
            self._log_state("READ_SUCCESS", filepath)

            result = self.extract_all(raw_data, **kwargs)
            sequence = result.get("sequence")
            metadata = result.get("metadata")
            # Optionally clean sequence if needed
            if sequence is not None:
                cleaned, n_positions = self._strip_unknowns(sequence)
            else:
                cleaned, n_positions = None, None
            self._log_state("PARSE_SUCCESS", filepath)

            self._cleanup_memmaps()
            self._log_state("FINISHED", filepath)
            return {"sequence": cleaned, "n_positions": n_positions, "metadata": metadata}

        except Exception as e:
            self._cleanup_memmaps()
            self._log_state("FAILED", filepath, error=str(e))
            raise

    async def read_batches(self, filepath: str, batch_size: int, **kwargs) -> AsyncGenerator[object, None]:
        """Batch reader with memory map support, using unified extract_all."""
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        try:
            self._log_state("STARTED", f"Reading {filepath} in batches of {batch_size}")
            raw_data = await self._read_file(filepath)
            self._log_state("READ_SUCCESS", filepath)

            result = self.extract_all(raw_data, **kwargs)
            sequence = result.get("sequence")
            metadata = result.get("metadata")
            if sequence is not None:
                cleaned, n_positions = self._strip_unknowns(sequence)
            else:
                cleaned, n_positions = None, None
            self._log_state("PARSE_SUCCESS", filepath)

            for chunk in self._chunk_sequence(cleaned, batch_size):
                yield {"sequence": chunk, "n_positions": n_positions, "metadata": metadata}

            self._cleanup_memmaps()
            self._log_state("FINISHED", filepath)

        except Exception as e:
            self._cleanup_memmaps()
            self._log_state("FAILED", filepath, error=str(e))
            raise


    @abstractmethod
    def extract_all(self, raw_data, **kwargs) -> dict:
        """
        Unified extraction method: returns a dict with at least 'sequence' and 'metadata' keys.
        """
        pass

    def _to_numeric(self, sequence) -> np.ndarray:
        """Convert sequence to numeric, using memory map if sequence is large."""
        if isinstance(sequence, np.ndarray):
            return sequence
        arr = np.array(sequence)
        if arr.nbytes > 100_000_000:  # 100MB threshold
            return self._to_memmap(arr)
        return arr

    def _to_memmap(self, array: np.ndarray) -> np.ndarray:
        """Store large array in memory-mapped file."""
        fd, path = tempfile.mkstemp(dir=self.mem_map_dir, suffix=".dat")
        os.close(fd)
        memmap = np.memmap(path, dtype=array.dtype, mode="w+", shape=array.shape)
        memmap[:] = array[:]
        self._active_memmaps.append(path)
        return memmap

    def _cleanup_memmaps(self):
        """Delete all temporary memory-mapped files and collect garbage."""
        for path in self._active_memmaps:
            try:
                os.remove(path)
            except Exception:
                pass
        self._active_memmaps = []
        gc.collect()

    def _chunk_sequence(self, sequence, batch_size: int):
        """Yield ordered chunks from a string, list/tuple, or numpy array."""
        if isinstance(sequence, str):
            for i in range(0, len(sequence), batch_size):
                yield sequence[i:i + batch_size]
            return

        if isinstance(sequence, np.ndarray):
            for i in range(0, len(sequence), batch_size):
                yield sequence[i:i + batch_size]
            return

        if isinstance(sequence, (list, tuple)):
            for i in range(0, len(sequence), batch_size):
                yield sequence[i:i + batch_size]
            return

        yield sequence

    




# ===========================================================
# BASE Visualizer 
# ===========================================================

class BaseVisualizer(BaseComponent):
    """
    Minimal visualization base: shared, step-by-step logging helpers for all
    visualization subclasses (e.g., plot renderers, CSV writers).
    """

    def log_prepare(self, message: str = ""):
        self._log_state("PREPARE", message)

    def log_encode(self, message: str = ""):
        self._log_state("ENCODE", message)

    def log_render(self, message: str = ""):
        self._log_state("RENDER", message)

    def log_write(self, message: str = ""):
        self._log_state("WRITE", message)

    def log_finish(self, message: str = ""):
        self._log_state("FINISHED", message)

    def log_error(self, message: str = "", error: str = None):
        self._log_state("FAILED", message, error=error)

    @abstractmethod
    async def visualize(self, *args, **kwargs):
        """Render or emit visualization output."""
        raise NotImplementedError

#kernel.py
# ===========================================================
# Kernel utilities for MolSample vectors
# ===========================================================

import numpy as np
from typing import List, Tuple, Optional, Iterable
import asyncio
import os
import psutil
import logging
from models import MolSample

# kernel logger
logger = logging.getLogger("KernelMatrix")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(name)s] %(levelname)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

class KernelMatrix:
    """
    Builds a full kernel (inner product) matrix over MolSample vectors and
    provides distance-based retrieval. Each MolSample.sequence can be 1-D
    (shape: length,) or tensor-like 2-D (shape: length, feature_dim). The
    kernel never flattens away feature dimensions; it computes per-feature
    kernels and sums them so no information is discarded.

    Args:
        samples: Ordered list of MolSample objects to include in the kernel.
        matrix_memmap_path: Explicit path for kernel memmap; auto-generated
            inside memmap_dir when None.
        seq_memmap_path: Explicit path for stacked sequence memmap; auto-
            generated inside memmap_dir when None.
        memmap_dir: Directory where all memmaps are written (required).
        block_size: Rows per block for multiplication; <=0 lets the kernel pick
            a safe value from available RAM.
        memory_limit_bytes: RAM budget for working buffers; <=0 auto-sets to
            ~90% of available RAM (min 1GB).
        conditional: When True, perform a light '#batch' ordering sanity check
            on ids (no length truncation is performed).
        save_vectors_path: Optional .npy path to persist the stacked vectors
            used for this kernel.
        logs_dir: Directory for auxiliary artifacts (e.g., saved ids alongside
            vectors); required.
    """

    def __init__(
        self,
        samples: List[MolSample],
        *,
        matrix_memmap_path: Optional[str] = None,
        seq_memmap_path: Optional[str] = None,
        memmap_dir: str = "",
        block_size: int = 0,
        memory_limit_bytes: int = 0,  # <=0 means auto-set from system
        conditional: bool = True,
        save_vectors_path: Optional[str] = None,
        logs_dir: str = "",
        sp_max_nodes: Optional[int] = None,
        sp_bins: Optional[int] = None,
    ):
        if not samples:
            raise ValueError("KernelMatrix requires at least one MolSample.")
        if not memmap_dir:
            raise ValueError("memmap_dir is required to build KernelMatrix safely.")
        if not logs_dir:
            raise ValueError("logs_dir is required to record kernel artifacts (ids, etc.).")
        os.makedirs(memmap_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)
        logger.info("START: building kernel for %d samples", len(samples))
        self.samples = samples
        self.ids = [s.id for s in samples]
        self._matrix_memmap_path = matrix_memmap_path
        self._seq_memmap_path = seq_memmap_path
        self.kernels = None  # optional list of per-feature kernels when tensor input provided
        self.norms_list = None  # per-feature norms when tensor input provided
        self.matrix, self.norms = self._build_kernel(
            samples,
            matrix_memmap_path=matrix_memmap_path,
            seq_memmap_path=seq_memmap_path,
            block_size=block_size,
            memory_limit_bytes=memory_limit_bytes,
            memmap_dir=memmap_dir,
            conditional=conditional,
            save_vectors_path=save_vectors_path,
            logs_dir=logs_dir,
            sp_max_nodes=sp_max_nodes,
            sp_bins=sp_bins,
        )
        logger.info("READY: ids=%d kernel_shape=%s", len(self.ids), getattr(self.matrix, 'shape', None))

    @classmethod
    async def build_async(
        cls,
        samples: List[MolSample],
        *,
        matrix_memmap_path: Optional[str] = None,
        seq_memmap_path: Optional[str] = None,
        memmap_dir: str = "",
        block_size: int = 0,
        memory_limit_bytes: int = 0,
        conditional: bool = True,
        save_vectors_path: Optional[str] = None,
        logs_dir: str = "",
        sp_max_nodes: Optional[int] = None,
        sp_bins: Optional[int] = None,
    ) -> "KernelMatrix":
        """
        Async-friendly constructor that offloads kernel construction to a worker
        thread to avoid blocking the event loop.

        Args mirror __init__; memmap_dir is required, block_size/memory_limit_bytes <=0 auto-tune.
        Logs: start/end are logged under the KernelMatrix logger.
        Returns:
            KernelMatrix instance with memmaps, norms, and ids populated.
        """
        return await asyncio.to_thread(
            cls,
            samples,
            matrix_memmap_path=matrix_memmap_path,
            seq_memmap_path=seq_memmap_path,
            block_size=block_size,
            memory_limit_bytes=memory_limit_bytes,
            memmap_dir=memmap_dir,
            conditional=conditional,
            save_vectors_path=save_vectors_path,
            logs_dir=logs_dir,
            sp_max_nodes=sp_max_nodes,
            sp_bins=sp_bins,
        )

    @classmethod
    def build_from_batches(
        cls,
        batches: Iterable[List[MolSample]],
        *,
        matrix_memmap_path: Optional[str] = None,
        seq_memmap_path: Optional[str] = None,
        memmap_dir: str = "",
        block_size: int = 0,
        memory_limit_bytes: int = 0,
        conditional: bool = True,
        save_vectors_path: Optional[str] = None,
        logs_dir: str = "",
        sp_max_nodes: Optional[int] = None,
        sp_bins: Optional[int] = None,
    ) -> "KernelMatrix":
        """
        Build a kernel from batches of MolSample objects while preserving order.

        Args mirror __init__; batches are concatenated in order before building. memmap_dir/logs_dir required.
        Logs: start/end are logged under the KernelMatrix logger.
        Returns:
            KernelMatrix instance with memmaps, norms, and ids populated.
        """
        samples: List[MolSample] = []
        for batch in batches:
            samples.extend(batch)
        return cls(
            samples,
            matrix_memmap_path=matrix_memmap_path,
            seq_memmap_path=seq_memmap_path,
            block_size=block_size,
            memory_limit_bytes=memory_limit_bytes,
            memmap_dir=memmap_dir,
            conditional=conditional,
            save_vectors_path=save_vectors_path,
            logs_dir=logs_dir,
            sp_max_nodes=sp_max_nodes,
            sp_bins=sp_bins,
        )

    def _build_kernel(
        self,
        samples: List[MolSample],
        *,
        matrix_memmap_path: Optional[str],
        seq_memmap_path: Optional[str],
        memmap_dir: str,
        block_size: int,
        memory_limit_bytes: int,
        conditional: bool,
        save_vectors_path: Optional[str],
        logs_dir: str,
        sp_max_nodes: Optional[int],
        sp_bins: Optional[int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build kernel and norms with memmap support and memory-aware block multiplication.

        Args:
            samples: MolSamples (order preserved).
            matrix_memmap_path: explicit memmap path for kernel (auto-created in memmap_dir if None).
            seq_memmap_path: explicit memmap path for stacked sequences (auto-created in memmap_dir if None).
            memmap_dir: directory used for auto-created memmaps (required).
            block_size: rows per block for matrix mult; <=0 triggers auto-tune based on memory_limit_bytes.
            memory_limit_bytes: RAM budget; <=0 auto-sets to ~90% of available RAM (min 1GB).
            conditional: If True and ids contain '#batch', only a sanity check
                is performed (ordering); no truncation or length changes occur.
            save_vectors_path: Optional .npy path to persist the stacked
                sequence matrix.
            logs_dir: Directory to save auxiliary artifacts (ids file) when
                saving vectors.

        Returns:
            kernel_matrix (np.ndarray or memmap) and norms_vector (np.ndarray).
        """
        # determine memory budget
        if memory_limit_bytes <= 0:
            avail = psutil.virtual_memory().available
            memory_limit_bytes = max(int(avail * 0.9), 1024 * 1024 * 1024)  # default: ~90% of free RAM, min 1GB
        if not memmap_dir:
            raise ValueError("memmap_dir is required for kernel computation.")
        if not logs_dir:
            raise ValueError("logs_dir is required for kernel computation.")
        os.makedirs(memmap_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)
        logger.info("Memory budget set to %.2f MB", memory_limit_bytes / 1e6)

        vectors = []
        lengths = []
        base_length = len(samples[0].sequence)
        feature_dim = None
        if conditional:
            # Optional batch ordering check: if ids carry '#batchX', ensure order per file
            grouped = {}
            for s in samples:
                if "#batch" in s.id:
                    base_id, _, batch_part = s.id.partition("#batch")
                    try:
                        batch_idx = int(batch_part)
                    except ValueError:
                        batch_idx = None
                    grouped.setdefault(base_id, []).append(batch_idx)
            for base_id, idxs in grouped.items():
                if any(i is None for i in idxs):
                    logger.warning("Batch pairing skipped for %s due to unparsable batch id", base_id)
                else:
                    sorted_idx = sorted(idxs)
                    if sorted_idx != list(range(min(sorted_idx), max(sorted_idx) + 1)):
                        logger.warning("Non-consecutive batches detected for %s; using provided order", base_id)
        for s in samples:
            vec = s.sequence
            if vec.ndim == 1:
                if feature_dim is None:
                    feature_dim = 1
                vec = vec[:, None]
            elif vec.ndim == 2:
                if feature_dim is None:
                    feature_dim = vec.shape[1]
                elif feature_dim != vec.shape[1]:
                    raise ValueError("All tensors must share the same feature dimension.")
            else:
                # flatten all metadata axes into feature channels; keep first axis as length
                new_feature_dim = int(np.prod(vec.shape[1:]))
                vec = vec.reshape(vec.shape[0], new_feature_dim)
                if feature_dim is None:
                    feature_dim = new_feature_dim
                elif feature_dim != new_feature_dim:
                    raise ValueError("All tensors must share the same flattened feature dimension.")
            vec = vec.astype(np.float32, copy=False)
            vectors.append(vec)
            lengths.append(vec.shape[0])

        if lengths and (min(lengths) != max(lengths)):
            logger.warning(
                "Sequence length mismatch detected. Using base_length=%d with truncation/padding.",
                base_length,
            )
        logger.info(
            "Stack prepared with %d vectors of base_length %d and feature_dim %d",
            len(vectors),
            base_length,
            feature_dim,
        )

        n = len(vectors)

        # memmap for stacked sequences (auto-create if not provided)
        if not seq_memmap_path:
            seq_memmap_path = os.path.join(memmap_dir, f"kernel_seq_{os.getpid()}_{np.random.randint(1e9)}.dat")
        stack = np.memmap(seq_memmap_path, dtype=np.float32, mode="w+", shape=(n, base_length, feature_dim))
        for idx, vec in enumerate(vectors):
            cur_len = vec.shape[0]
            if cur_len >= base_length:
                stack[idx] = vec[:base_length]
            else:
                stack[idx, :cur_len, :] = vec
        logger.info("Using seq memmap at %s", seq_memmap_path)

        # norms (squared L2 per vector) aggregated across feature dims
        norms_list = []
        for k in range(feature_dim):
            norms_list.append(np.sum(stack[:, :, k] * stack[:, :, k], axis=1))
        norms = np.sum(np.stack(norms_list, axis=0), axis=0)
        self.norms_list = norms_list
        logger.info("Computed norms for %d vectors across %d feature dims", len(norms), feature_dim)

        # kernel computation (block to reduce peak RAM) for each feature dim
        estimated_kernel_bytes = n * n * 4  # float32 per kernel
        if not matrix_memmap_path:
            matrix_memmap_path = os.path.join(memmap_dir, f"kernel_matrix_{os.getpid()}_{np.random.randint(1e9)}.dat")
        kernel = np.memmap(matrix_memmap_path, dtype=np.float32, mode="w+", shape=(n, n))
        kernel[:] = 0.0
        logger.info("Using matrix memmap at %s", matrix_memmap_path)

        # auto-tune block size based on memory limit if not provided or <=0
        if block_size <= 0:
            bytes_per_row_out = n * 4  # one row of kernel (float32)
            bytes_per_row_in = base_length * feature_dim * 4  # one row of stack across all features
            bytes_per_row_total = max(1, bytes_per_row_out + bytes_per_row_in)
            block_size = max(1, min(n, memory_limit_bytes // bytes_per_row_total))
        bs = max(1, block_size)
        logger.info("Block size selected: %d (rows per multiply)", bs)
        # compute kernel by summing per-feature blocks to avoid flattening tensors
        for k in range(feature_dim):
            logger.info("Accumulating feature dimension %d / %d", k + 1, feature_dim)
            for i in range(0, n, bs):
                i_end = min(i + bs, n)
                block = stack[i:i_end, :, k]
                kernel[i:i_end, :] += block @ stack[:, :, k].T

        # flush memmaps
        if isinstance(kernel, np.memmap):
            kernel.flush()
        if isinstance(stack, np.memmap):
            stack.flush()
        if save_vectors_path:
            np.save(save_vectors_path, np.asarray(stack))
            logger.info("Saved stacked vectors to %s", save_vectors_path)

        # record paths (for cleanup) if auto-created
        self._matrix_memmap_path = matrix_memmap_path or getattr(self, "_matrix_memmap_path", None)
        self._seq_memmap_path = seq_memmap_path or getattr(self, "_seq_memmap_path", None)
        logger.info("Kernel built with block_size=%d, shape=%s", bs, getattr(kernel, "shape", None))

        # optionally persist ids alongside vectors
        if save_vectors_path and logs_dir:
            try:
                ids_path = os.path.join(logs_dir, "kernel_vectors_ids.txt")
                with open(ids_path, "w") as fh:
                    fh.write("\n".join(self.ids))
                logger.info("Saved sample ids to %s", ids_path)
            except OSError:
                logger.warning("Failed to save ids to %s", logs_dir)

        return kernel, norms

    def _build_sp_kernel(
        self,
        samples: List[MolSample],
        *,
        matrix_memmap_path: Optional[str],
        memmap_dir: str,
        memory_limit_bytes: int,
        logs_dir: str,
        max_nodes: Optional[int] = None,
        bins: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Approximate shortest-path (SP) graph kernel for sequences with ndim > 2.
        We treat the spatial axes as a grid graph, sample up to max_nodes points,
        build a histogram of pairwise Manhattan shortest-path lengths, and use
        those histograms as feature vectors. Kernel = dot product of histograms.

        Returns:
            kernel_matrix (memmap) and norms (np.ndarray).
        """
        logger.info("Using SP kernel path for %d high-dimensional samples", len(samples))
        if memory_limit_bytes <= 0:
            avail = psutil.virtual_memory().available
            memory_limit_bytes = max(int(avail * 0.9), 1024 * 1024 * 1024)
        os.makedirs(memmap_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)

        rng = np.random.default_rng(0)
        n = len(samples)
        # sample once globally to ensure positional alignment across samples
        ref_shape = samples[0].sequence.shape
        total_nodes_ref = np.prod(ref_shape)
        if max_nodes is None:
            sample_size_global = total_nodes_ref
        else:
            sample_size_global = min(total_nodes_ref, max_nodes)
        coords_global = np.empty((sample_size_global, len(ref_shape)), dtype=np.int64)
        for i in range(sample_size_global):
            coords_global[i] = [rng.integers(0, d) for d in ref_shape]

        # bin count defaults to sample size if not provided
        bin_count_global = bins if bins is not None else max(2, int(sample_size_global))

        # prepare feature matrix as memmap to stay memory-friendly
        feature_len = sample_size_global + bin_count_global
        feature_mat = np.memmap(
            os.path.join(memmap_dir, f"kernel_sp_features_{os.getpid()}_{np.random.randint(1e9)}.dat"),
            dtype=np.float32,
            mode="w+",
            shape=(n, feature_len),
        )

        for row_idx, s in enumerate(samples):
            arr = s.sequence
            spatial_shape = arr.shape  # treat every axis as spatial for graph layout
            total_nodes = np.prod(spatial_shape)
            # ensure shapes match; otherwise resample for this sample with budget cap
            if spatial_shape != ref_shape:
                if max_nodes is None:
                    sample_size = total_nodes
                else:
                    sample_size = min(total_nodes, max_nodes)
                coords = np.empty((sample_size, len(spatial_shape)), dtype=np.int64)
                for i in range(sample_size):
                    coords[i] = [rng.integers(0, d) for d in spatial_shape]
                bin_count = bins if bins is not None else max(2, int(sample_size))
            else:
                coords = coords_global
                sample_size = sample_size_global
                bin_count = bin_count_global
            logger.info("SP sample %s using %d/%d nodes over shape %s", s.id, sample_size, total_nodes, spatial_shape)
            # pairwise Manhattan distances
            diffs = coords[:, None, :] - coords[None, :, :]
            manhattan = np.abs(diffs).sum(axis=-1)
            max_d = manhattan.max(initial=0)
            if bins is not None:
                bin_count = min(bin_count, bins)
            bin_edges = np.linspace(0, max(max_d, 1), num=bin_count + 1)
            hist, _ = np.histogram(manhattan, bins=bin_edges)
            # gather sampled values to capture content alongside structure
            vals = arr[tuple(coords.T)].astype(np.float32).ravel()
            feature_vec = np.concatenate([hist.astype(np.float32), vals])
            # pad or truncate to fit feature_mat width
            if feature_vec.size < feature_len:
                padded = np.zeros(feature_len, dtype=np.float32)
                padded[:feature_vec.size] = feature_vec
                feature_vec = padded
            else:
                feature_vec = feature_vec[:feature_len]
            feature_mat[row_idx] = feature_vec

        n = feature_mat.shape[0]
        if not matrix_memmap_path:
            matrix_memmap_path = os.path.join(memmap_dir, f"kernel_sp_{os.getpid()}_{np.random.randint(1e9)}.dat")
        kernel = np.memmap(matrix_memmap_path, dtype=np.float32, mode="w+", shape=(n, n))
        logger.info("SP kernel memmap: %s", matrix_memmap_path)
        # compute Gram matrix
        kernel[:] = feature_mat @ feature_mat.T
        if isinstance(kernel, np.memmap):
            kernel.flush()
        norms = np.sum(feature_mat * feature_mat, axis=1)
        self._matrix_memmap_path = matrix_memmap_path
        self._seq_memmap_path = None
        self.kernels = None
        self.norms_list = None
        return kernel, norms

    def distances_from(self, target_id: str) -> List[Tuple[str, float]]:
        """
        Compute Euclidean distances from the target sample to all others using
        kernel entries: ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x^T y
        Returns list of (sample_id, distance) in the original order.

        Args:
            target_id: sample id to compare against.

        Returns:
            List of (id, distance) including the target with distance 0.
        """
        ids = self.ids
        kernel = self.matrix
        norms = self.norms

        if target_id not in ids:
            raise ValueError(f"Target id {target_id} not found in kernel samples.")

        idx = ids.index(target_id)
        k_row = kernel[idx]
        dist_sq = norms[idx] + norms - 2 * k_row
        dist_sq = np.clip(dist_sq, a_min=0.0, a_max=None)
        dists = np.sqrt(dist_sq)
        return list(zip(ids, dists.tolist()))

    def best_match(self, target_id: str, top_n: Optional[int] = 1) -> Optional[List[Tuple[str, float]]]:
        """
        Return the closest sample(s) to target_id based on Euclidean distance
        derived from the kernel matrix.

        Args:
            target_id: id to match against.
            top_n: number of closest samples to return (excluding the target itself).
                Defaults to 1 for backward compatibility.

        Returns:
            List of (id, distance) tuples sorted by ascending distance.
            If only one closest sample is requested, returns a list of length 1.
            Returns None if there are no other samples.
        """
        distances = self.distances_from(target_id)
        # exclude the target itself
        filtered = [(sid, d) for sid, d in distances if sid != target_id]
        if not filtered:
            return None

        # sort by distance ascending
        filtered.sort(key=lambda x: x[1])

        if top_n is None or top_n <= 1:
            # return single closest as a list
            return [filtered[0]]
        else:
            # return up to top_n closest
            return filtered[:top_n]


    def cleanup(self):
        """Delete memmap files (always)."""
        for path in (self._matrix_memmap_path, self._seq_memmap_path):
            if path:
                try:
                    os.remove(path)
                    logger.info("Removed memmap %s", path)
                except OSError:
                    pass
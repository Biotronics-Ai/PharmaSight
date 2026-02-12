<p align="center">
  <img src="images/banner.png" alt="Biotronics Ai">
</p>

# *PharmaSight*

An open-source asynchronous proteomic and molecular comparison toolkit built by Biotronics AI. It supports multi-format parsing, efficient memory management with lazy-loading, and local similarity matching for pharma, drug repurposing/repositioning and molecular analysis workflows.

**What it delivers in practice?**

- **Multi-format molecular comparison within seconds:** Parse 14+ file formats (PDB, MMCIF, FASTA, MZML, SMILES, InChI, and more), stream in batches, eliminate weak candidates fast, then score the top matches with kernel-based similarity search.
- **Memory-efficient batch processing:** Lazy-load sequences from memory-mapped files, process in configurable batches (default 25), and maintain stable RAM usage even with 20,000+ samples.
- **Unified extraction pipeline:** Single-pass `extract_all()` method across all parsers normalizes sequence and metadata extraction, reducing code duplication and improving performance.
- **Multi-format molecular handling:** Normalize heterogeneous sequence formats (genomic, proteomic, cheminformatic) to be compatible and directly comparable.

## Features

**Supported Data File Formats**:

* **Sequence:** FASTA, FASTQ, GenBank formats.
* **Structure:** PDB (Protein Data Bank), MMCIF (mmCIF).
* **Mass Spectrometry:** MZML, MZXML, MZIdentML, MZTab.
* **Genomics:** PepXML.
* **Chemistry:** SMILES, InChI, SDF, MOL files.
* **Network:** EdgeList, BioPAX RDF.

**Core Features**:

- `MolSample` (lazy-loading with memmap read-only mode)
- Unified `extract_all()` extraction across 14 parser classes
- Batch processing with memory monitoring (psutil integration)
- `KernelMatrix` for fast similarity search and best-match discovery
- Async/concurrent file parsing with garbage collection between batches
- Flexible metadata extraction with normalized output format

**Memory-aware architecture**:

- Lazy-loading: Store only file paths initially, load sequences on-demand
- Memory monitoring: Track RSS memory before/after each batch
- Batch processing: Process 25 files concurrently, then gc.collect()
- Memmap buffers: Read-only access (`mmap_mode='r'`) for minimal overhead

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install pharmasight
```

Dependencies are defined in `pyproject.toml` / `requirements.txt`.

## Project layout

- `models.py` — MolSample class, lazy-loading with memmap support.
- `parsers.py` — 14 file-format parsers with unified `extract_all()` method.
- `kernel.py` — KernelMatrix for similarity search and matching.
- `base.py` — base components and utility helpers.

## Quick start

### 1) Unified extraction across multiple formats

All parsers implement the same `extract_all()` interface for consistent, single-pass extraction:

```python
from pharmasight.parsers import FASTAParser, PDBStructureParser

# FASTA example
fasta_parser = FASTAParser()
result = fasta_parser.extract_all(raw_fasta_data)
sequence = result["sequence"]
metadata = result["metadata"]

# PDB example
pdb_parser = PDBStructureParser()
result = pdb_parser.extract_all(raw_pdb_data)
structure_vectors = result["sequence"]
pdb_metadata = result["metadata"]
```

_Real life_: Parse genomic sequences (FASTA), protein structures (PDB), and mass spectrometry data (MZML) in a single, unified pipeline without format-specific extraction logic.

### 2) Batch processing with memory monitoring

Load and parse large sample collections efficiently with automatic memory management:

```python
from pharmasight.extract import AsyncExtractor
from pharmasight.models import MolSample

extractor = AsyncExtractor(
    batch_size=25,
    memmap_dir="mem_map",
    logs_dir="logs"
)

samples = await extractor.extract_from_directory(
    directory_path="data_samples/trcc_pdb",
    recursive=True
)

# Samples are lazy-loaded; sequences stored as memmap file paths
for sample in samples:
    # Access sequence only when needed (lazy-load)
    seq = sample.sequence
    print(f"{sample.sample_id}: {seq.shape}")
```

_Real life_: Process 19,000+ molecular structures without exhausting RAM; batch operations with memory delta logging ensure stable performance.

### 3) Two-directory similarity workflow

Build a reference kernel from one directory, then search against samples in another:

```python
from pharmasight.kernel import KernelMatrix
from pharmasight.extract import AsyncExtractor

# Phase 1: Build kernel from reference directory
extractor = AsyncExtractor(batch_size=25, memmap_dir="mem_map")
kernel_samples = await extractor.extract_from_directory("data_samples/trcc_pdb")
kernel = KernelMatrix(kernel_samples, memmap_dir="mem_map", logs_dir="logs")

# Phase 2: Search target samples in kernel
target_samples = await extractor.extract_from_directory("data_samples/trcc")
results = {}
for target in target_samples:
    best_matches = kernel.best_match(target.sample_id, top_k=5)
    results[target.sample_id] = best_matches
```

_Real life_: Compare query molecules against a pharmaceutical database; find top-5 closest structural matches within seconds.

### 4) Kernel matrix with lazy-loaded sequences

Use KernelMatrix for fast similarity search across large sample pools without loading everything into RAM:

```python
from pharmasight.kernel import KernelMatrix

kernel = KernelMatrix(
    samples=samples,
    memmap_dir="mem_map",
    logs_dir="logs"
)

# Find top-5 most similar samples to a query
best_matches = kernel.best_match("query_sample_id", top_k=5)
print(f"Top matches: {best_matches}")

kernel.cleanup()
```

⚠️ **Memory note**: Kernel matrix builds are based on sequence length and sample count. Set `memmap_dir` for disk buffering and monitor memory usage via `psutil`. Ensure sequences are normalized to consistent dimensions (padded/truncated to base_length).

### 5) Direct MolSample creation with lazy-loading

Create molecular samples manually for custom workflows:

```python
import numpy as np
from pharmasight.models import MolSample

# From memmap file
sample = MolSample(
    sample_id="PDB_1ABC",
    memmap_path="mem_map/AF-Q4CKA0-F1-model_v6.npy",  # lazy-load on access
    metadata={"format": "pdb", "organism": "human"}
)

# Sequence loads from disk only when accessed
vec = sample.sequence  # Loads memmap here
print(f"Shape: {vec.shape}, Size: {vec.nbytes/1024/1024:.2f} MB")

# Clear cache if needed
sample.clear_cache()
```

_Real life_: Build custom pipelines where sequences are accessed on-demand; avoid unnecessary disk I/O and memory consumption.

### 6) Fast iteration: memmap-only kernel building

Skip expensive parsing and build kernel directly from pre-parsed memmaps:

```python
from pharmasight import test_memmap_kernel

# Assumes mem_map/ directory contains .npy files
result = test_memmap_kernel.main()
# Loads existing memmaps, builds kernel, saves results in seconds
```

_Real life_: Iterate rapidly on similarity search logic without re-parsing 19,000+ files (saves 30+ minutes per test cycle).

### 7) Mixed format handling with automatic normalization

Compare samples across different file formats seamlessly:

```python
from pharmasight.extract import AsyncExtractor

extractor = AsyncExtractor(batch_size=25, memmap_dir="mem_map")

# Mix of FASTA, PDB, MZML, SMILES formats
mixed_samples = await extractor.extract_from_directory(
    "data_samples/mixed",
    recursive=True
)

# All sequences normalized to vectors; ready for kernel/comparison
print(f"Loaded {len(mixed_samples)} samples across mixed formats")
for s in mixed_samples[:3]:
    print(f"  {s.sample_id} ({s.metadata.get('format')}): {s.sequence.shape}")
```

_Real life_: Pharma research combining genomic (FASTA), structural (PDB), and proteomic (MS) data in one analysis pipeline.

## Key components

| Component                | Purpose                | Key Features                                                   |
| ------------------------ | ---------------------- | -------------------------------------------------------------- |
| **MolSample**      | Sample container       | Lazy-loading, memmap support, metadata, cache management       |
| **14 Parsers**     | File format extraction | Unified `extract_all()`, async-friendly, normalized output   |
| **AsyncExtractor** | Batch processing       | Batch size 25, memory monitoring, gc.collect() between batches |
| **KernelMatrix**   | Similarity search      | O(n²) kernel, best_match(), top_k filtering                   |

## Memory & performance tips

- **Always use `memmap_dir`**: Offload sequence vectors to disk for 19,000+ samples.
- **Batch size tuning**: Default 25 files per batch; reduce if RAM-constrained, increase for better throughput.
- **Lazy-loading by default**: Sequences load on-demand; avoid premature full loads.
- **Memory monitoring**: Logs show memory deltas per batch; watch for unexpected growth.
- **Cleanup explicitly**: Call `kernel.cleanup()` and `sample.clear_cache()` when done.

## Testing

- `test.py` — Full pipeline: parse all data, build kernel, search targets.
- `test2.py` — Two-directory workflow: kernel_dir (reference) + target_dir (queries).
- `test_memmap_kernel.py` — Fast iteration: build kernel from existing memmaps only.

## Contributing

We value contributions! Areas of interest:

- New file format parsers (return `{"sequence": ..., "metadata": ...}` from `extract_all()`).
- Performance optimizations for KernelMatrix (vectorization, distributed compute).
- Additional similarity metrics beyond cosine.
- Visualization tools for similarity results.
- Documentation and usage examples.
- Additional testing for mass-spectrometry formats
- Increasing efficiency for feature-dimensional vectorization features of file format-specific parsers.

## Known limitations

- Variable-length sequences padded/truncated to base_length (first sample's length).
- KernelMatrix requires sequences of identical dimensionality.
- Memory usage is O(n²) for n samples in kernel; disk buffering via memmap mitigates but doesn't eliminate.
- Lazy-loading depends on memmap file availability; memmaps are read-only.

<p align="center">
  <a href="https://biotronics.ai">
    <img src="images/logo.png" alt="Biotronics Ai Logo" width="180">
  </a>
</p>

import gc
from time import time
import aiofiles
import numpy as np
from typing import List, Dict
import asyncio
from base import BaseParser
import networkx as nx
import os
from Bio import SeqIO
import json


class FASTAParser(BaseParser):
    """
    FASTA parser with dynamic feature extraction and graph embedding.
    Streams sequences, computes k-mer and global features, and produces
    feature-dimension vectors aligned across sequences.
    """

    async def _read_file(self, filepath: str) -> List[str]:
        return [filepath]

    def _extract_sequence(self, raw_data: List[str]) -> dict:
        filepath = raw_data[0]
        sequences = []
        with open(filepath, "r") as fh:
            seq = []
            for line in fh:
                if line.startswith(">"):
                    if seq:
                        sequences.append(''.join(seq))
                        seq = []
                    continue
                seq.append(line.strip())
            if seq:
                sequences.append(''.join(seq))

        feature_store = {}

        for idx, seq in enumerate(sequences):
            self._log_state("PROCESS_SEQUENCE", f"Processing sequence {idx}")

            # ---------------------------
            # Global features
            # ---------------------------
            seq_len = len(seq)
            if seq_len == 0:
                continue
            feature_store.setdefault("seq_length", []).append(seq_len)
            counts = {}
            for nt in seq:
                counts[nt] = counts.get(nt, 0) + 1
            for nt, count in counts.items():
                feature_store.setdefault(f"nt_count_{nt}", []).append(count / seq_len)  # normalize

            # ---------------------------
            # k-mer features (k=3 for example)
            # ---------------------------
            k = 3
            kmer_counts = {}
            for i in range(len(seq) - k + 1):
                kmer = seq[i:i+k]
                kmer_counts[kmer] = kmer_counts.get(kmer, 0) + 1
            for kmer, count in kmer_counts.items():
                feature_store.setdefault(f"kmer_{kmer}", []).append(count / (seq_len - k + 1))

            # ---------------------------
            # Graph embedding
            # ---------------------------
            try:
                G = nx.Graph()
                for i, nt in enumerate(seq):
                    G.add_node(i, feat=nt)
                    if i > 0:
                        G.add_edge(i-1, i)  # sequential adjacency
                # Node degrees
                degrees = np.array([d for n, d in G.degree()])
                # Node type histogram
                nt_list = sorted(list(set(seq)))
                nt_hist = np.array([seq.count(nt) for nt in nt_list], dtype=np.float32)
                graph_embedding = np.concatenate([degrees, nt_hist])
                feature_store.setdefault("graph_embedding", []).append(graph_embedding)
            except Exception as e:
                self._log_state("GRAPH_EMBED_FAIL", f"Sequence {idx} graph embedding failed: {e}")
                feature_store.setdefault("graph_embedding", []).append(np.zeros(seq_len + len(set(seq)), dtype=np.float32))

        # ---------------------------
        # Vectorization
        # ---------------------------
        vectors = {}
        for key, values in feature_store.items():
            try:
                if key == "graph_embedding":
                    max_len = max(len(v) for v in values)
                    padded = np.array([np.pad(v, (0, max_len - len(v)), 'constant') for v in values], dtype=np.float32)
                    vectors[key] = padded
                else:
                    vectors[key] = np.array(values, dtype=np.float32)
            except Exception:
                vectors[key] = np.frombuffer(" ".join(map(str, values)).encode(), dtype=np.uint8)

        self._log_state("EXTRACT_SEQUENCE_DONE", f"Extracted {len(vectors)} feature vectors including graph embedding")
        return vectors

    def _extract_metadata(self, raw_data: List[str]) -> dict:
        filepath = raw_data[0]
        headers = []
        sequences = []
        with open(filepath, "r") as fh:
            seq = []
            for line in fh:
                if line.startswith(">"):
                    headers.append(line.strip()[1:])
                    if seq:
                        sequences.append(''.join(seq))
                        seq = []
                    continue
                seq.append(line.strip())
            if seq:
                sequences.append(''.join(seq))
        return {"headers": headers, "num_sequences": len(sequences)}





from Bio.PDB import PDBParser as BioPDBParser


class PDBStructureParser(BaseParser):
    """
    Parser for Protein Data Bank (PDB) files that:
    - Reads the file using a bioinformatics library (Bio.PDB).
    - Traverses the full structural hierarchy (model → chain → residue → atom).
    - Dynamically extracts every numeric attribute available on each atom object.
    - Vectorizes all extracted data into NumPy tensors.
    - Logs every major and minor step in the parsing pipeline.
    """
    def __init__(self, memmap_dir: str):
        self.memmap_dir = memmap_dir

    async def _read_file(self, filepath: str):
        """
        Reads a PDB file from disk and returns a Bio.PDB Structure object.

        Parameters
        ----------
        filepath : str
            Absolute or relative path to the .pdb file.

        Returns
        -------
        structure : Bio.PDB.Structure.Structure
            Parsed structural object containing models, chains, residues, and atoms.

        Logging
        -------
        - READ_INIT: File reading started.
        - READ_SUCCESS: File successfully parsed.
        - READ_FAILED: File reading failed (includes error message).
        """
        self._log_state("READ_INIT", f"PDB file read started: {filepath}")
        try:
            parser = BioPDBParser(QUIET=True)
            structure = parser.get_structure("structure", filepath)
            self._log_state("READ_SUCCESS", f"PDB structure loaded: {filepath}")
            return structure
        except Exception as e:
            self._log_state("READ_FAILED", f"PDB read failed: {filepath}", error=str(e))
            raise

    def _atom_to_vector(self, atoms, memmap_path):
        """
        Converts a list of atoms into a NumPy array by first writing atom
        numeric attributes to a memmap file and then reading them all at once.
        Deletes the memmap file after reading to free disk space.

        Parameters
        ----------
        atoms : list of Bio.PDB.Atom.Atom
            Atom objects obtained from Bio.PDB traversal.
        memmap_path : str
            Path to save temporary memory-mapped array of atom features.

        Returns
        -------
        vectors : np.ndarray, shape = (n_atoms, n_features)
            2D float32 array representing all numeric attributes for all atoms.
        """
        self._log_state("VECTORIZE_ATOMS_START", f"Vectorizing {len(atoms)} atoms to memmap: {memmap_path}")

        atom_dicts = []
        for atom in atoms:
            atom_data = {}
            for attr in dir(atom):
                if attr.startswith("_"):
                    continue
                try:
                    val = getattr(atom, attr)
                    if isinstance(val, (int, float)):
                        atom_data[attr] = float(val)
                    elif isinstance(val, (list, tuple, np.ndarray)):
                        flat_vals = [float(v) for v in val if isinstance(v, (int, float))]
                        atom_data[attr] = flat_vals
                except Exception:
                    continue
            atom_dicts.append(atom_data)

        max_len = max(
            sum(len(v) if isinstance(v, list) else 1 for v in d.values())
            for d in atom_dicts
        ) if atom_dicts else 0

        n_atoms = len(atom_dicts)
        memmap_array = np.memmap(memmap_path, dtype='float32', mode='w+', shape=(n_atoms, max_len))
        
        for i, atom_d in enumerate(atom_dicts):
            flat_vals = []
            for v in atom_d.values():
                if isinstance(v, list):
                    flat_vals.extend(v)
                else:
                    flat_vals.append(v)
            if len(flat_vals) < max_len:
                flat_vals.extend([0.0]*(max_len - len(flat_vals)))
            memmap_array[i, :] = np.array(flat_vals, dtype=np.float32)

        del atom_dicts
        gc.collect()

        vectors = np.array(memmap_array[:], dtype=np.float32)

        # İşlem bitince memmap dosyasını sil
        memmap_array._mmap.close()
        del memmap_array
        if os.path.exists(memmap_path):
            os.remove(memmap_path)
        gc.collect()

        self._log_state("VECTORIZE_ATOMS_DONE", f"All atoms vectorized: shape={vectors.shape}")
        return vectors



    def extract_all(self, raw_data, memmap_path=None, memmap_dir=None):
        """
        Unified extraction: returns both sequence and metadata in a single call.
        """
        # --- Sequence extraction (was _extract_sequence) ---
        self._log_state("EXTRACT_SEQUENCE_START", "Starting atom-level vector extraction")
        all_atoms = [atom for model in raw_data for chain in model for residue in chain for atom in residue]
        self._log_state("EXTRACT_SEQUENCE_COUNTS", f"Total atoms: {len(all_atoms)}")
        if not all_atoms:
            self._log_state("EXTRACT_SEQUENCE_EMPTY", "No atom vectors extracted — returning empty matrix")
            sequence = np.empty((0, 0), dtype=np.float32)
        else:
            temp_memmap = False
            if memmap_path is None:
                memmap_path = os.path.join(self.memmap_dir, f"temp_atoms_{time.time()}.npy")
                temp_memmap = True
            sequence = self._atom_to_vector(all_atoms, memmap_path=memmap_path)
            if temp_memmap and os.path.exists(memmap_path):
                os.remove(memmap_path)
            self._log_state("EXTRACT_SEQUENCE_DONE", f"Final tensor shape: {sequence.shape}")

        # --- Metadata extraction (was _extract_metadata) ---
        self._log_state("EXTRACT_METADATA_START", "Starting dynamic metadata extraction from structure object")
        metadata = {}
        try:
            for attr in dir(raw_data):
                if attr.startswith("_"):
                    continue
                try:
                    val = getattr(raw_data, attr)
                    if isinstance(val, (int, float, str)):
                        metadata[attr] = val
                        self._log_state("METADATA_FIELD_ADDED", f"{attr}={val}")
                    elif hasattr(val, "__len__") and not isinstance(val, str):
                        metadata[attr] = len(val)
                        self._log_state("METADATA_FIELD_ADDED", f"{attr}_len={len(val)}")
                except Exception as e:
                    self._log_state("METADATA_FIELD_SKIP", f"Skipping metadata field: {attr}", error=str(e))
                    continue
            self._log_state("EXTRACT_METADATA_DONE", f"Metadata fields extracted: {len(metadata)}")
        except Exception as e:
            self._log_state("EXTRACT_METADATA_FAILED", "Metadata extraction failed", error=str(e))
            metadata = {}

        return {"sequence": sequence, "metadata": metadata}





from Bio.PDB import MMCIFParser as BioMMCIFParser


class MMCIFStructureParser(BaseParser):
    """
    Parser for mmCIF (.cif) protein structure files that:
    - Reads files using a bioinformatics library (Bio.PDB MMCIFParser).
    - Traverses the full structural hierarchy (model → chain → residue → atom).
    - Dynamically extracts all numeric attributes from atom objects.
    - Vectorizes extracted values into NumPy tensors.
    - Logs every step of the parsing and vectorization pipeline.
    """

    async def _read_file(self, filepath: str):
        """
        Reads an mmCIF file from disk and returns a Bio.PDB Structure object.

        Parameters
        ----------
        filepath : str
            Absolute or relative path to the .cif file.

        Returns
        -------
        structure : Bio.PDB.Structure.Structure
            Parsed structure object containing models, chains, residues, and atoms.

        Logging
        -------
        - READ_INIT: File reading started.
        - READ_SUCCESS: File successfully parsed.
        - READ_FAILED: File reading failed (includes error message).
        """
        self._log_state("READ_INIT", f"mmCIF file read started: {filepath}")
        try:
            parser = BioMMCIFParser(QUIET=True)
            structure = parser.get_structure("structure", filepath)
            self._log_state("READ_SUCCESS", f"mmCIF structure loaded: {filepath}")
            return structure
        except Exception as e:
            self._log_state("READ_FAILED", f"mmCIF read failed: {filepath}", error=str(e))
            raise

    def _atom_to_vector(self, atom):
        """
        Converts a single atom object into a NumPy vector by dynamically extracting
        all numeric attributes.

        Parameters
        ----------
        atom : Bio.PDB.Atom.Atom
            Atom object obtained from Bio.PDB traversal.

        Returns
        -------
        vector : np.ndarray, shape = (n_features,)
            One-dimensional float32 vector representing all numeric attributes found
            on the atom object.

        Logging
        -------
        - VECTORIZE_ATOM_START: Atom vectorization started.
        - VECTORIZE_ATOM_ATTR_SKIP: Attribute skipped due to error or incompatibility.
        - VECTORIZE_ATOM_DONE: Atom successfully vectorized (includes vector length).
        """
        self._log_state(
            "VECTORIZE_ATOM_START",
            f"Vectorizing atom: {atom.get_full_id() if hasattr(atom, 'get_full_id') else str(atom)}"
        )

        values = []
        for attr in dir(atom):
            if attr.startswith("_"):
                continue
            try:
                val = getattr(atom, attr)
                if isinstance(val, (int, float)):
                    values.append(val)
                elif isinstance(val, (list, tuple, np.ndarray)):
                    for v in val:
                        if isinstance(v, (int, float)):
                            values.append(v)
            except Exception as e:
                self._log_state(
                    "VECTORIZE_ATOM_ATTR_SKIP",
                    f"Skipping atom attribute: {attr}",
                    error=str(e)
                )
                continue

        vector = np.array(values, dtype=np.float32)
        self._log_state("VECTORIZE_ATOM_DONE", f"Atom vectorized: length={vector.shape[0]}")
        return vector

    def extract_all(self, raw_data, **kwargs):
        """Unified extraction: returns both sequence and metadata in a single call."""
        # --- Sequence extraction ---
        self._log_state("EXTRACT_SEQUENCE_START", "Starting atom-level vector extraction")
        vectors = []
        atom_count = 0
        model_count = 0
        chain_count = 0
        residue_count = 0
        try:
            for model in raw_data:
                model_count += 1
                self._log_state("MODEL_ITERATE", f"Processing model index: {model.id}")
                for chain in model:
                    chain_count += 1
                    self._log_state("CHAIN_ITERATE", f"Processing chain ID: {chain.id}")
                    for residue in chain:
                        residue_count += 1
                        self._log_state("RESIDUE_ITERATE", f"Processing residue: {residue.get_full_id()}")
                        for atom in residue:
                            atom_count += 1
                            self._log_state("ATOM_ITERATE", f"Processing atom: {atom.get_name()}")
                            vector = self._atom_to_vector(atom)
                            vectors.append(vector)
            self._log_state("EXTRACT_SEQUENCE_COUNTS", f"Models={model_count}, Chains={chain_count}, Residues={residue_count}, Atoms={atom_count}")
        except Exception as e:
            self._log_state("EXTRACT_SEQUENCE_FAILED", "Failed during structural traversal", error=str(e))
            raise
        if not vectors:
            self._log_state("EXTRACT_SEQUENCE_EMPTY", "No atom vectors extracted — returning empty matrix")
            matrix = np.empty((0, 0), dtype=np.float32)
        else:
            self._log_state("SEQUENCE_VECTOR_SHAPING", "Normalizing atom vectors to uniform matrix shape")
            try:
                max_len = max(v.shape[0] for v in vectors)
                self._log_state("SEQUENCE_VECTOR_MAXLEN", f"Max atom vector length: {max_len}")
                matrix = np.zeros((len(vectors), max_len), dtype=np.float32)
                for i, v in enumerate(vectors):
                    matrix[i, :v.shape[0]] = v
                    if i % 1000 == 0 and i > 0:
                        self._log_state("SEQUENCE_VECTOR_PROGRESS", f"Vectorized {i}/{len(vectors)} atoms")
                self._log_state("EXTRACT_SEQUENCE_DONE", f"Final tensor shape: {matrix.shape}")
            except Exception as e:
                self._log_state("SEQUENCE_VECTOR_FAILED", "Failed during tensor assembly", error=str(e))
                raise
        
        # --- Metadata extraction ---
        self._log_state("EXTRACT_METADATA_START", "Starting dynamic metadata extraction from structure object")
        metadata = {}
        try:
            for attr in dir(raw_data):
                if attr.startswith("_"):
                    continue
                try:
                    val = getattr(raw_data, attr)
                    if isinstance(val, (int, float, str)):
                        metadata[attr] = val
                        self._log_state("METADATA_FIELD_ADDED", f"{attr}={val}")
                    elif hasattr(val, "__len__") and not isinstance(val, str):
                        metadata[attr] = len(val)
                        self._log_state("METADATA_FIELD_ADDED", f"{attr}_len={len(val)}")
                except Exception as e:
                    self._log_state("METADATA_FIELD_SKIP", f"Skipping metadata field: {attr}", error=str(e))
                    continue
            self._log_state("EXTRACT_METADATA_DONE", f"Metadata fields extracted: {len(metadata)}")
        except Exception as e:
            self._log_state("EXTRACT_METADATA_FAILED", "Metadata extraction failed", error=str(e))
            metadata = {}
        
        return {"sequence": matrix, "metadata": metadata}



from pyteomics import mzml


class MZMLParser(BaseParser):
    """mzML parser that dynamically extracts all numeric and vector-like fields."""

    async def _read_file(self, filepath: str):
        self._log_state("READ_START", f"Opening mzML file: {filepath}")
        reader = mzml.read(filepath)
        self._log_state("READ_COMPLETE", "mzML file successfully opened")
        return reader

    def extract_all(self, raw_data, **kwargs):
        self._log_state("EXTRACT_START", "Beginning dynamic feature extraction from spectra")
        feature_store = {}
        spectrum_count = 0
        for spectrum in raw_data:
            self._log_state("SPECTRUM_START", f"Processing spectrum index {spectrum_count}")
            for key, value in spectrum.items():
                try:
                    if key not in feature_store:
                        feature_store[key] = []
                    if isinstance(value, (list, tuple, np.ndarray)):
                        vec = np.array(value, dtype=np.float32)
                        feature_store[key].append(vec)
                        self._log_state("FEATURE_VECTOR", f"Spectrum {spectrum_count} | Feature '{key}' vector shape {vec.shape}")
                    elif isinstance(value, (int, float)):
                        vec = np.array([value], dtype=np.float32)
                        feature_store[key].append(vec)
                        self._log_state("FEATURE_SCALAR", f"Spectrum {spectrum_count} | Feature '{key}' scalar vectorized")
                    else:
                        self._log_state("FEATURE_SKIP", f"Spectrum {spectrum_count} | Feature '{key}' non-numeric, skipped")
                except Exception as e:
                    self._log_state("FEATURE_ERROR", f"Spectrum {spectrum_count} | Feature '{key}' extraction failed", error=str(e))
            spectrum_count += 1
        
        if not feature_store:
            self._log_state("EXTRACT_EMPTY", "No numeric features extracted from mzML file")
            sequence = {}
        else:
            self._log_state("EXTRACT_FEATURES_COMPLETE", f"Extracted {len(feature_store)} feature dimensions from {spectrum_count} spectra")
            normalized_features = {}
            for feature_name, vectors in feature_store.items():
                try:
                    max_len = max(v.shape[0] for v in vectors)
                    matrix = np.zeros((len(vectors), max_len), dtype=np.float32)
                    for i, v in enumerate(vectors):
                        matrix[i, :v.shape[0]] = v
                    normalized_features[feature_name] = matrix
                    self._log_state("FEATURE_NORMALIZED", f"Feature '{feature_name}' normalized to shape {matrix.shape}")
                except Exception as e:
                    normalized_features[feature_name] = vectors
                    self._log_state("FEATURE_RAGGED", f"Feature '{feature_name}' kept as ragged list", error=str(e))
            self._log_state("EXTRACT_COMPLETE", "All features extracted and normalized")
            sequence = normalized_features
        
        self._log_state("METADATA_START", "Beginning dynamic metadata extraction")
        metadata = {}
        for attr in dir(raw_data):
            if attr.startswith("_"):
                continue
            try:
                value = getattr(raw_data, attr)
                if isinstance(value, (int, float, str)):
                    metadata[attr] = value
                elif hasattr(value, "__len__") and not isinstance(value, str):
                    metadata[attr] = len(value)
            except Exception:
                continue
        self._log_state("METADATA_COMPLETE", f"Metadata extraction completed with {len(metadata)} fields")
        return {"sequence": sequence, "metadata": metadata}



from pyteomics import mzxml


class MZXMLParser(BaseParser):
    """mzXML parser that dynamically extracts all numeric and vector-like fields."""

    async def _read_file(self, filepath: str):
        self._log_state("READ_START", f"Opening mzXML file: {filepath}")
        reader = mzxml.read(filepath)
        self._log_state("READ_COMPLETE", "mzXML file successfully opened")
        return reader

    def extract_all(self, raw_data, **kwargs):
        self._log_state("EXTRACT_START", "Beginning dynamic feature extraction from spectra")
        feature_store = {}
        spectrum_count = 0
        for spectrum in raw_data:
            self._log_state("SPECTRUM_START", f"Processing spectrum index {spectrum_count}")
            for key, value in spectrum.items():
                try:
                    if key not in feature_store:
                        feature_store[key] = []
                    if isinstance(value, (list, tuple, np.ndarray)):
                        vec = np.array(value, dtype=np.float32)
                        feature_store[key].append(vec)
                        self._log_state("FEATURE_VECTOR", f"Spectrum {spectrum_count} | Feature '{key}' vector shape {vec.shape}")
                    elif isinstance(value, (int, float)):
                        vec = np.array([value], dtype=np.float32)
                        feature_store[key].append(vec)
                        self._log_state("FEATURE_SCALAR", f"Spectrum {spectrum_count} | Feature '{key}' scalar vectorized")
                    else:
                        self._log_state("FEATURE_SKIP", f"Spectrum {spectrum_count} | Feature '{key}' non-numeric, skipped")
                except Exception as e:
                    self._log_state("FEATURE_ERROR", f"Spectrum {spectrum_count} | Feature '{key}' extraction failed", error=str(e))
            spectrum_count += 1
        
        if not feature_store:
            self._log_state("EXTRACT_EMPTY", "No numeric features extracted from mzXML file")
            sequence = {}
        else:
            self._log_state("EXTRACT_FEATURES_COMPLETE", f"Extracted {len(feature_store)} feature dimensions from {spectrum_count} spectra")
            normalized_features = {}
            for feature_name, vectors in feature_store.items():
                try:
                    max_len = max(v.shape[0] for v in vectors)
                    matrix = np.zeros((len(vectors), max_len), dtype=np.float32)
                    for i, v in enumerate(vectors):
                        matrix[i, :v.shape[0]] = v
                    normalized_features[feature_name] = matrix
                    self._log_state("FEATURE_NORMALIZED", f"Feature '{feature_name}' normalized to shape {matrix.shape}")
                except Exception as e:
                    normalized_features[feature_name] = vectors
                    self._log_state("FEATURE_RAGGED", f"Feature '{feature_name}' kept as ragged list", error=str(e))
            self._log_state("EXTRACT_COMPLETE", "All features extracted and normalized")
            sequence = normalized_features
        
        self._log_state("METADATA_START", "Beginning dynamic metadata extraction")
        metadata = {}
        for attr in dir(raw_data):
            if attr.startswith("_"):
                continue
            try:
                value = getattr(raw_data, attr)
                if isinstance(value, (int, float, str)):
                    metadata[attr] = value
                elif hasattr(value, "__len__") and not isinstance(value, str):
                    metadata[attr] = len(value)
            except Exception:
                continue
        self._log_state("METADATA_COMPLETE", f"Metadata extraction completed with {len(metadata)} fields")
        return {"sequence": sequence, "metadata": metadata}



from pyteomics import mzid


class MZIdentMLParser(BaseParser):
    """
    mzIdentML parser that dynamically extracts *all* numeric and vector-like
    fields from identification results as independent feature-dimension vectors.

    Output from read():
        {
            "sequence": {
                feature_name: np.ndarray (shape: [n_entries, variable_length])
                OR list[np.ndarray] if ragged,
                ...
            },
            "n_positions": [],
            "metadata": dict
        }
    """

    async def _read_file(self, filepath: str):
        """
        Reads mzIdentML file using pyteomics.

        Args:
            filepath (str): Path to mzIdentML file.

        Returns:
            Generator yielding identification dictionaries.
        """
        self._log_state("READ_START", f"Opening mzIdentML file: {filepath}")
        reader = mzid.read(filepath)
        self._log_state("READ_COMPLETE", "mzIdentML file successfully opened")
        return reader

    def extract_all(self, raw_data, **kwargs):
        self._log_state("EXTRACT_START", "Beginning dynamic feature extraction from mzIdentML entries")
        feature_store = {}
        entry_count = 0
        for entry in raw_data:
            self._log_state("ENTRY_START", f"Processing identification entry {entry_count}")
            for key, value in entry.items():
                try:
                    if key not in feature_store:
                        feature_store[key] = []
                    if isinstance(value, (list, tuple, np.ndarray)):
                        vec = np.array(value, dtype=np.float32)
                        feature_store[key].append(vec)
                        self._log_state("FEATURE_VECTOR", f"Entry {entry_count} | Feature '{key}' vector shape {vec.shape}")
                    elif isinstance(value, (int, float)):
                        vec = np.array([value], dtype=np.float32)
                        feature_store[key].append(vec)
                        self._log_state("FEATURE_SCALAR", f"Entry {entry_count} | Feature '{key}' scalar vectorized")
                    else:
                        self._log_state("FEATURE_SKIP", f"Entry {entry_count} | Feature '{key}' non-numeric, skipped")
                except Exception as e:
                    self._log_state("FEATURE_ERROR", f"Entry {entry_count} | Feature '{key}' extraction failed", error=str(e))
            entry_count += 1
        
        if not feature_store:
            self._log_state("EXTRACT_EMPTY", "No numeric features extracted from mzIdentML file")
            sequence = {}
        else:
            self._log_state("EXTRACT_FEATURES_COMPLETE", f"Extracted {len(feature_store)} feature dimensions from {entry_count} entries")
            normalized_features = {}
            for feature_name, vectors in feature_store.items():
                try:
                    max_len = max(v.shape[0] for v in vectors)
                    matrix = np.zeros((len(vectors), max_len), dtype=np.float32)
                    for i, v in enumerate(vectors):
                        matrix[i, :v.shape[0]] = v
                    normalized_features[feature_name] = matrix
                    self._log_state("FEATURE_NORMALIZED", f"Feature '{feature_name}' normalized to shape {matrix.shape}")
                except Exception as e:
                    normalized_features[feature_name] = vectors
                    self._log_state("FEATURE_RAGGED", f"Feature '{feature_name}' kept as ragged list", error=str(e))
            self._log_state("EXTRACT_COMPLETE", "All features extracted and normalized")
            sequence = normalized_features
        
        self._log_state("METADATA_START", "Beginning dynamic metadata extraction")
        metadata = {}
        for attr in dir(raw_data):
            if attr.startswith("_"):
                continue
            try:
                value = getattr(raw_data, attr)
                if isinstance(value, (int, float, str)):
                    metadata[attr] = value
                elif hasattr(value, "__len__") and not isinstance(value, str):
                    metadata[attr] = len(value)
            except Exception:
                continue
        self._log_state("METADATA_COMPLETE", f"Metadata extraction completed with {len(metadata)} fields")
        return {"sequence": sequence, "metadata": metadata}



from pyteomics import mztab


class MZTabParser(BaseParser):
    """mzTab parser that dynamically extracts all numeric and vector-like fields from all sections."""

    async def _read_file(self, filepath: str):
        self._log_state("READ_START", f"Opening mzTab file: {filepath}")
        reader = mztab.read(filepath)
        self._log_state("READ_COMPLETE", "mzTab file successfully opened")
        return reader

    def extract_all(self, raw_data, **kwargs):
        self._log_state("EXTRACT_START", "Beginning dynamic feature extraction from mzTab sections")
        feature_store = {}
        entry_count = 0
        sections = {"PSM": raw_data.get("PSM", []), "PEPTIDE": raw_data.get("PEP", []), "PROTEIN": raw_data.get("PRT", []), "SMALLMOLECULE": raw_data.get("SMF", [])}
        for section_name, rows in sections.items():
            self._log_state("SECTION_START", f"Processing section: {section_name}")
            for row_idx, row in enumerate(rows):
                self._log_state("ROW_START", f"Processing {section_name} row {row_idx}")
                for key, value in row.items():
                    feature_key = f"{section_name}.{key}"
                    try:
                        if feature_key not in feature_store:
                            feature_store[feature_key] = []
                        if isinstance(value, (list, tuple, np.ndarray)):
                            vec = np.array(value, dtype=np.float32)
                            feature_store[feature_key].append(vec)
                            self._log_state("FEATURE_VECTOR", f"{section_name} row {row_idx} | Feature '{feature_key}' vector shape {vec.shape}")
                        elif isinstance(value, (int, float)):
                            vec = np.array([value], dtype=np.float32)
                            feature_store[feature_key].append(vec)
                            self._log_state("FEATURE_SCALAR", f"{section_name} row {row_idx} | Feature '{feature_key}' scalar vectorized")
                        else:
                            self._log_state("FEATURE_SKIP", f"{section_name} row {row_idx} | Feature '{feature_key}' non-numeric, skipped")
                    except Exception as e:
                        self._log_state("FEATURE_ERROR", f"{section_name} row {row_idx} | Feature '{feature_key}' extraction failed", error=str(e))
                entry_count += 1
        
        if not feature_store:
            self._log_state("EXTRACT_EMPTY", "No numeric features extracted from mzTab file")
            sequence = {}
        else:
            self._log_state("EXTRACT_FEATURES_COMPLETE", f"Extracted {len(feature_store)} feature dimensions from {entry_count} total rows")
            normalized_features = {}
            for feature_name, vectors in feature_store.items():
                try:
                    max_len = max(v.shape[0] for v in vectors)
                    matrix = np.zeros((len(vectors), max_len), dtype=np.float32)
                    for i, v in enumerate(vectors):
                        matrix[i, :v.shape[0]] = v
                    normalized_features[feature_name] = matrix
                    self._log_state("FEATURE_NORMALIZED", f"Feature '{feature_name}' normalized to shape {matrix.shape}")
                except Exception as e:
                    normalized_features[feature_name] = vectors
                    self._log_state("FEATURE_RAGGED", f"Feature '{feature_name}' kept as ragged list", error=str(e))
            self._log_state("EXTRACT_COMPLETE", "All mzTab features extracted and normalized")
            sequence = normalized_features
        
        self._log_state("METADATA_START", "Beginning dynamic metadata extraction")
        metadata = {}
        header = raw_data.get("MTD", {})
        for key, value in header.items():
            try:
                if isinstance(value, (int, float, str)):
                    metadata[key] = value
                elif hasattr(value, "__len__") and not isinstance(value, str):
                    metadata[key] = len(value)
            except Exception:
                continue
        self._log_state("METADATA_COMPLETE", f"Metadata extraction completed with {len(metadata)} fields")
        return {"sequence": sequence, "metadata": metadata}



from pyteomics import pepxml


class PepXMLParser(BaseParser):
    """
    pepXML parser that dynamically extracts *all* numeric and vector-like
    fields from every spectrum_query and search_hit as independent
    feature-dimension vectors.

    Output from read():
        {
            "sequence": {
                feature_name: np.ndarray (shape: [n_entries, variable_length])
                OR list[np.ndarray] if ragged,
                ...
            },
            "n_positions": [],
            "metadata": dict
        }
    """

    async def _read_file(self, filepath: str):
        """
        Reads pepXML file using pyteomics.

        Args:
            filepath (str): Path to pepXML file.

        Returns:
            Iterator of spectrum_query entries.
        """
        self._log_state("READ_START", f"Opening pepXML file: {filepath}")
        reader = pepxml.read(filepath)
        self._log_state("READ_COMPLETE", "pepXML file successfully opened")
        return reader

    def extract_all(self, raw_data, **kwargs):
        """
        Unified extraction of features and metadata from pepXML in a single pass.
        
        Dynamically extracts every numeric field from spectrum_query, nested
        search_hit, and analysis_result entries as feature vectors.
        Also extracts metadata from run_summary.

        Args:
            raw_data: pepXML reader iterator.
            **kwargs: Additional arguments (unused).

        Returns:
            dict with keys:
                "sequence": dict[str, np.ndarray] - normalized feature matrices
                "metadata": dict - extracted metadata from run_summary
        """
        self._log_state("EXTRACT_START", "Beginning unified feature and metadata extraction from pepXML")

        feature_store = {}
        entry_count = 0

        # Phase 1: Extract features through single iteration
        for query_idx, spectrum_query in enumerate(raw_data):
            self._log_state("QUERY_START", f"Processing spectrum_query {query_idx}")

            # Process top-level spectrum_query fields
            for key, value in spectrum_query.items():
                feature_key = f"SPECTRUM_QUERY.{key}"

                try:
                    if feature_key not in feature_store:
                        feature_store[feature_key] = []

                    if isinstance(value, (list, tuple, np.ndarray)):
                        vec = np.array(value, dtype=np.float32)
                        feature_store[feature_key].append(vec)
                        self._log_state(
                            "FEATURE_VECTOR",
                            f"spectrum_query {query_idx} | Feature '{feature_key}' vector shape {vec.shape}"
                        )

                    elif isinstance(value, (int, float)):
                        vec = np.array([value], dtype=np.float32)
                        feature_store[feature_key].append(vec)
                        self._log_state(
                            "FEATURE_SCALAR",
                            f"spectrum_query {query_idx} | Feature '{feature_key}' scalar vectorized"
                        )

                    else:
                        self._log_state(
                            "FEATURE_SKIP",
                            f"spectrum_query {query_idx} | Feature '{feature_key}' non-numeric, skipped"
                        )

                except Exception as e:
                    self._log_state(
                        "FEATURE_ERROR",
                        f"spectrum_query {query_idx} | Feature '{feature_key}' extraction failed",
                        error=str(e)
                    )

            # Process nested search_hit entries
            search_hits = spectrum_query.get("search_hit", [])
            for hit_idx, hit in enumerate(search_hits):
                self._log_state(
                    "HIT_START",
                    f"Processing search_hit {hit_idx} in spectrum_query {query_idx}"
                )

                for key, value in hit.items():
                    feature_key = f"SEARCH_HIT.{key}"

                    try:
                        if feature_key not in feature_store:
                            feature_store[feature_key] = []

                        if isinstance(value, (list, tuple, np.ndarray)):
                            vec = np.array(value, dtype=np.float32)
                            feature_store[feature_key].append(vec)
                            self._log_state(
                                "FEATURE_VECTOR",
                                f"search_hit {hit_idx} | Feature '{feature_key}' vector shape {vec.shape}"
                            )

                        elif isinstance(value, (int, float)):
                            vec = np.array([value], dtype=np.float32)
                            feature_store[feature_key].append(vec)
                            self._log_state(
                                "FEATURE_SCALAR",
                                f"search_hit {hit_idx} | Feature '{feature_key}' scalar vectorized"
                            )

                        else:
                            self._log_state(
                                "FEATURE_SKIP",
                                f"search_hit {hit_idx} | Feature '{feature_key}' non-numeric, skipped"
                            )

                    except Exception as e:
                        self._log_state(
                            "FEATURE_ERROR",
                            f"search_hit {hit_idx} | Feature '{feature_key}' extraction failed",
                            error=str(e)
                        )

                # Process nested analysis_result fields (e.g., PeptideProphet, iProphet, etc.)
                analysis_results = hit.get("analysis_result", [])
                for ar_idx, analysis_result in enumerate(analysis_results):
                    analysis_name = analysis_result.get("analysis", "UNKNOWN_ANALYSIS")

                    self._log_state(
                        "ANALYSIS_START",
                        f"Processing analysis_result '{analysis_name}' for search_hit {hit_idx}"
                    )

                    for key, value in analysis_result.items():
                        feature_key = f"ANALYSIS.{analysis_name}.{key}"

                        try:
                            if feature_key not in feature_store:
                                feature_store[feature_key] = []

                            if isinstance(value, (list, tuple, np.ndarray)):
                                vec = np.array(value, dtype=np.float32)
                                feature_store[feature_key].append(vec)
                                self._log_state(
                                    "FEATURE_VECTOR",
                                    f"analysis_result '{analysis_name}' | Feature '{feature_key}' vector shape {vec.shape}"
                                )

                            elif isinstance(value, (int, float)):
                                vec = np.array([value], dtype=np.float32)
                                feature_store[feature_key].append(vec)
                                self._log_state(
                                    "FEATURE_SCALAR",
                                    f"analysis_result '{analysis_name}' | Feature '{feature_key}' scalar vectorized"
                                )

                            else:
                                self._log_state(
                                    "FEATURE_SKIP",
                                    f"analysis_result '{analysis_name}' | Feature '{feature_key}' non-numeric, skipped"
                                )

                        except Exception as e:
                            self._log_state(
                                "FEATURE_ERROR",
                                f"analysis_result '{analysis_name}' | Feature '{feature_key}' extraction failed",
                                error=str(e)
                            )

                entry_count += 1

        if not feature_store:
            self._log_state("EXTRACT_EMPTY", "No numeric features extracted from pepXML file")
            normalized_features = {}
        else:
            self._log_state(
                "EXTRACT_FEATURES_COMPLETE",
                f"Extracted {len(feature_store)} feature dimensions from {entry_count} spectrum queries"
            )

            # Normalize feature dimensions (pad to uniform length per feature)
            normalized_features = {}

            for feature_name, vectors in feature_store.items():
                try:
                    max_len = max(v.shape[0] for v in vectors)
                    matrix = np.zeros((len(vectors), max_len), dtype=np.float32)

                    for i, v in enumerate(vectors):
                        matrix[i, :v.shape[0]] = v

                    normalized_features[feature_name] = matrix
                    self._log_state(
                        "FEATURE_NORMALIZED",
                        f"Feature '{feature_name}' normalized to shape {matrix.shape}"
                    )

                except Exception as e:
                    normalized_features[feature_name] = vectors
                    self._log_state(
                        "FEATURE_RAGGED",
                        f"Feature '{feature_name}' kept as ragged list",
                        error=str(e)
                    )

            self._log_state("EXTRACT_COMPLETE", "All pepXML features extracted and normalized")

        # Phase 2: Extract metadata from run_summary
        self._log_state("METADATA_START", "Beginning dynamic metadata extraction")

        metadata = {}
        try:
            if hasattr(raw_data, "run_summary"):
                run_summary = raw_data.run_summary
                for key, value in run_summary.items():
                    if isinstance(value, (int, float, str)):
                        metadata[key] = value
                    elif hasattr(value, "__len__") and not isinstance(value, str):
                        metadata[key] = len(value)
        except Exception as e:
            self._log_state("METADATA_SKIP", "Failed to extract some metadata", error=str(e))

        self._log_state(
            "METADATA_COMPLETE",
            f"Metadata extraction completed with {len(metadata)} fields"
        )

        return {"sequence": normalized_features, "metadata": metadata}





import pandas as pd


class EdgeListParser(BaseParser):
    """
    Parser for simple PPI edge-list files (TSV/CSV).

    Extracts node pairs, weights, and all additional columns as separate
    feature-dimension vectors.

    Output (sequence):
        Dict[str, np.ndarray] of feature vectors.

    Output (metadata):
        Dict[str, np.ndarray] dynamically extracted from all columns.
    """

    async def _read_file(self, filepath: str):
        self._log_state("READ_FILE_START", f"Opening edge-list file: {filepath}")
        df = pd.read_csv(filepath, sep=None, engine="python")
        self._log_state("READ_FILE_DONE", f"Loaded edge list with {len(df)} edges")
        return df

    def extract_all(self, raw_data, **kwargs):
        """
        Unified extraction of edge-list features and metadata in a single pass.

        Args:
            raw_data: pandas DataFrame with edge-list data.
            **kwargs: Additional arguments (unused).

        Returns:
            dict with keys:
                "sequence": dict[str, np.ndarray] - feature vectors from all columns
                "metadata": dict[str, np.ndarray] - metadata versions of same features
        """
        self._log_state("EXTRACT_START", "Extracting edge-list features and metadata")

        # Phase 1: Extract sequence features
        vectors = {}
        for col in raw_data.columns:
            self._log_state("PROCESS_COLUMN", f"Processing column: {col}")
            series = raw_data[col]

            if pd.api.types.is_numeric_dtype(series):
                vectors[col] = series.to_numpy(dtype=np.float32)
            else:
                vectors[col] = np.frombuffer(" ".join(series.astype(str)).encode(), dtype=np.uint8)

        self._log_state("EXTRACT_SEQUENCE_DONE", f"Extracted {len(vectors)} feature vectors")

        # Phase 2: Extract metadata (same columns with metadata prefix)
        self._log_state("EXTRACT_METADATA_START", "Extracting metadata from same columns")

        metadata_store = {}
        for col in raw_data.columns:
            self._log_state("PROCESS_METADATA_COLUMN", f"Processing metadata for column: {col}")
            series = raw_data[col]

            meta_key = f"meta_{col}"
            if pd.api.types.is_numeric_dtype(series):
                metadata_store[meta_key] = series.to_numpy(dtype=np.float32)
            else:
                metadata_store[meta_key] = np.frombuffer(" ".join(series.astype(str)).encode(), dtype=np.uint8)

        self._log_state("EXTRACT_METADATA_DONE", f"Extracted {len(metadata_store)} metadata dimensions")

        return {"sequence": vectors, "metadata": metadata_store}




from rdflib import Graph


class BioPAXRDFParser(BaseParser):
    """
    Parser for BioPAX Level 2/3 files in RDF/XML or OWL format using rdflib.

    This parser reads the RDF graph, dynamically extracts all triples,
    and converts every distinct predicate and object datatype into separate
    feature-dimension vectors suitable for kernel-based computation.

    All numeric values are vectorized as float arrays.
    All categorical/textual values are encoded as byte arrays.

    Output (sequence):
        Dict[str, np.ndarray]
            - Each key corresponds to a unique RDF predicate or datatype category.
            - Each value is a vector aligned across RDF triples or entities.

    Output (metadata):
        Dict[str, Any]
            - Includes graph size, number of triples, namespaces, and class counts.
    """

    async def _read_file(self, filepath: str):
        self._log_state("READ_FILE_START", f"Opening BioPAX RDF file: {filepath}")
        graph = Graph()
        graph.parse(filepath)
        self._log_state("READ_FILE_DONE", f"Loaded RDF graph with {len(graph)} triples")
        return graph

    def extract_all(self, raw_data, **kwargs):
        """
        Unified extraction of RDF triples features and metadata in a single pass.

        Args:
            raw_data: rdflib Graph object.
            **kwargs: Additional arguments (unused).

        Returns:
            dict with keys:
                "sequence": dict[str, np.ndarray] - vectorized predicates
                "metadata": dict[str, np.ndarray] - metadata by item type
        """
        self._log_state("EXTRACT_START", "Extracting RDF triples into features and metadata")

        feature_store = {}
        metadata = {}
        triple_index = 0

        # Phase 1: Extract features and metadata in single pass through triples
        for subj, pred, obj in raw_data:
            self._log_state("PROCESS_TRIPLE", f"Processing triple {triple_index}")
            triple_index += 1

            # Feature extraction by predicate
            pred_key = str(pred)

            if pred_key not in feature_store:
                feature_store[pred_key] = []

            # Process object value dynamically
            if obj.is_literal:
                try:
                    val = obj.toPython()
                    if isinstance(val, (int, float)):
                        feature_store[pred_key].append(val)
                    elif isinstance(val, str):
                        feature_store[pred_key].append(val)
                    elif hasattr(val, "__len__"):
                        feature_store[pred_key].append(len(val))
                    else:
                        feature_store[pred_key].append(np.nan)
                except Exception:
                    feature_store[pred_key].append(np.nan)
            else:
                # URIRef or BNode → encode as string
                feature_store[pred_key].append(str(obj))

            # Metadata extraction by item type (same loop iteration)
            for item in (subj, pred, obj):
                key = str(type(item))
                if key not in metadata:
                    metadata[key] = []

                if item.is_literal:
                    try:
                        val = item.toPython()
                        metadata[key].append(val)
                    except Exception:
                        metadata[key].append(str(item))
                else:
                    metadata[key].append(str(item))

        # Phase 2: Vectorize features
        self._log_state("VECTORIZE_START", "Vectorizing RDF predicate dimensions")

        vectors = {}
        for key, values in feature_store.items():
            self._log_state("VECTORIZE_FEATURE", f"Vectorizing predicate: {key}")

            if all(isinstance(v, (int, float, np.number)) or v is None for v in values):
                vectors[key] = np.array(values, dtype=np.float32)
            else:
                # Categorical/textual → byte encoding
                joined = " ".join(map(str, values))
                vectors[key] = np.frombuffer(joined.encode(), dtype=np.uint8)

        self._log_state("EXTRACT_SEQUENCE_DONE", f"Extracted {len(vectors)} feature vectors")

        # Phase 3: Vectorize metadata
        self._log_state("VECTORIZE_METADATA_START", "Vectorizing metadata dimensions")

        metadata_vectors = {}
        for key, values in metadata.items():
            if all(isinstance(v, (int, float, np.number)) or v is None for v in values):
                metadata_vectors[key] = np.array(values, dtype=np.float32)
            else:
                metadata_vectors[key] = np.frombuffer(" ".join(map(str, values)).encode(), dtype=np.uint8)

        self._log_state("EXTRACT_METADATA_DONE", f"Extracted {len(metadata_vectors)} metadata dimensions")

        return {"sequence": vectors, "metadata": metadata_vectors}



from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors


class SMILESParser(BaseParser):
    """
    Parser for SMILES files.

    Reads SMILES strings and dynamically extracts:
      - Atom-level numeric features
      - Bond-level numeric features
      - Molecular descriptors
      - Multiple fingerprint representations

    All extracted information is returned as separate feature-dimension vectors.

    Output (sequence):
        Dict[str, np.ndarray] where each key corresponds to a dynamically
        discovered feature dimension and each value is a vector aligned
        across molecules.

    Output (metadata):
        Dict[str, np.ndarray] dynamically extracted from molecule-level
        and file-level properties, with no hard-coded schema.
    """

    async def _read_file(self, filepath: str):
        """
        Reads a SMILES file line by line.

        Args:
            filepath (str): Path to the SMILES file.

        Returns:
            List[str]: List of SMILES strings.
        """
        self._log_state("READ_FILE_START", f"Opening SMILES file: {filepath}")
        with open(filepath, "r") as f:
            smiles_list = [line.strip().split()[0] for line in f if line.strip()]
        self._log_state("READ_FILE_DONE", f"Loaded {len(smiles_list)} SMILES entries")
        return smiles_list

    def extract_all(self, raw_data, **kwargs):
        """
        Unified extraction of molecule features and metadata in a single pass.

        Extracts all numeric, categorical, and structural information
        dynamically from RDKit molecule objects, including graph embeddings
        and molecule-level properties.

        Args:
            raw_data (List[str]): List of SMILES strings.
            **kwargs: Additional arguments (unused).

        Returns:
            dict with keys:
                "sequence": dict[str, np.ndarray] - feature vectors from atoms, bonds, descriptors, fingerprints
                "metadata": dict[str, np.ndarray] - molecule-level properties
        """
        self._log_state("EXTRACT_START", "Extracting SMILES features and metadata")

        feature_store = {}
        metadata = {}

        # Phase 1: Extract features and metadata in single pass
        for idx, smiles in enumerate(raw_data):
            self._log_state("PROCESS_MOLECULE_START", f"Processing molecule {idx}")
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                self._log_state("PROCESS_MOLECULE_SKIP", f"Invalid SMILES at index {idx}")
                continue

            # ---------------------------
            # Atom-level dynamic features
            # ---------------------------
            self._log_state("ATOM_FEATURE_START", f"Extracting atom-level features for molecule {idx}")
            for atom in mol.GetAtoms():
                for attr in dir(atom):
                    if attr.startswith("_"):
                        continue
                    try:
                        val = getattr(atom, attr)
                        if callable(val):
                            val = val()
                        if isinstance(val, (int, float, np.number)):
                            feature_store.setdefault(f"atom_{attr}", []).append(val)
                    except Exception:
                        continue
            self._log_state("ATOM_FEATURE_DONE", f"Atom-level features extracted for molecule {idx}")

            # ---------------------------
            # Bond-level dynamic features
            # ---------------------------
            self._log_state("BOND_FEATURE_START", f"Extracting bond-level features for molecule {idx}")
            for bond in mol.GetBonds():
                for attr in dir(bond):
                    if attr.startswith("_"):
                        continue
                    try:
                        val = getattr(bond, attr)
                        if callable(val):
                            val = val()
                        if isinstance(val, (int, float, np.number)):
                            feature_store.setdefault(f"bond_{attr}", []).append(val)
                    except Exception:
                        continue
            self._log_state("BOND_FEATURE_DONE", f"Bond-level features extracted for molecule {idx}")

            # ---------------------------
            # Molecular descriptors
            # ---------------------------
            self._log_state("DESCRIPTOR_START", f"Extracting molecular descriptors for molecule {idx}")
            for name, func in Descriptors._descList:
                try:
                    val = func(mol)
                    if isinstance(val, (int, float, np.number)):
                        feature_store.setdefault(f"desc_{name}", []).append(val)
                except Exception:
                    continue
            self._log_state("DESCRIPTOR_DONE", f"Molecular descriptors extracted for molecule {idx}")

            # ---------------------------
            # Fingerprints (multiple types)
            # ---------------------------
            self._log_state("FINGERPRINT_START", f"Extracting fingerprints for molecule {idx}")

            # Morgan / ECFP
            morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=4096)
            for i, bit in enumerate(morgan_fp):
                feature_store.setdefault(f"fp_morgan_bit_{i}", []).append(bit)

            # RDKit fingerprint
            rdkit_fp = Chem.RDKFingerprint(mol)
            for i, bit in enumerate(rdkit_fp):
                feature_store.setdefault(f"fp_rdkit_bit_{i}", []).append(bit)

            # MACCS keys
            maccs_fp = AllChem.GetMACCSKeysFingerprint(mol)
            for i, bit in enumerate(maccs_fp):
                feature_store.setdefault(f"fp_maccs_bit_{i}", []).append(bit)

            self._log_state("FINGERPRINT_DONE", f"Fingerprints extracted for molecule {idx}")

            # ---------------------------
            # Graph Embedding
            # ---------------------------
            self._log_state("GRAPH_EMBEDDING_START", f"Extracting graph embeddings for molecule {idx}")
            try:
                G = nx.Graph()

                # Add nodes with atom indices
                for atom in mol.GetAtoms():
                    G.add_node(atom.GetIdx(), atomic_num=atom.GetAtomicNum())

                # Add edges
                for bond in mol.GetBonds():
                    G.add_edge(
                        bond.GetBeginAtomIdx(),
                        bond.GetEndAtomIdx(),
                        bond_type=bond.GetBondTypeAsDouble()
                    )

                # Simple graph embedding: degree + atomic_num vector per node, flattened
                node_features = []
                for node in G.nodes(data=True):
                    deg = G.degree[node[0]]
                    atomic_num = node[1]["atomic_num"]
                    node_features.extend([deg, atomic_num])

                feature_store.setdefault("graph_embedding", []).append(node_features)
            except Exception as e:
                self._log_state("GRAPH_EMBEDDING_FAIL", f"Failed graph embedding for molecule {idx}: {e}")
            self._log_state("GRAPH_EMBEDDING_DONE", f"Graph embedding extracted for molecule {idx}")

            # ---------------------------
            # Metadata extraction (molecule-level attributes)
            # ---------------------------
            self._log_state("METADATA_ATTR_START", f"Extracting molecule attributes for metadata {idx}")
            for attr in dir(mol):
                if attr.startswith("_"):
                    continue
                try:
                    val = getattr(mol, attr)
                    if callable(val):
                        val = val()
                    if isinstance(val, (int, float, np.number)):
                        metadata.setdefault(f"mol_{attr}", []).append(val)
                    elif isinstance(val, str):
                        metadata.setdefault(f"mol_{attr}", []).append(val)
                except Exception:
                    continue
            self._log_state("METADATA_ATTR_DONE", f"Molecule attributes extracted for metadata {idx}")

            self._log_state("PROCESS_MOLECULE_DONE", f"Finished processing molecule {idx}")

        # Phase 2: Vectorize features
        self._log_state("VECTORIZE_START", "Vectorizing SMILES feature dimensions")
        vectors = {}
        for key, values in feature_store.items():
            try:
                vectors[key] = np.array(values, dtype=np.float32)
            except Exception:
                vectors[key] = np.frombuffer(" ".join(map(str, values)).encode(), dtype=np.uint8)

        self._log_state("EXTRACT_SEQUENCE_DONE", f"Extracted {len(vectors)} feature vectors")

        # Phase 3: Vectorize metadata
        self._log_state("VECTORIZE_METADATA_START", "Vectorizing metadata dimensions")
        for key, values in metadata.items():
            try:
                metadata[key] = np.array(values, dtype=np.float32)
            except Exception:
                metadata[key] = np.frombuffer(" ".join(map(str, values)).encode(), dtype=np.uint8)

        self._log_state("EXTRACT_METADATA_DONE", f"Extracted {len(metadata)} metadata dimensions")

        return {"sequence": vectors, "metadata": metadata}




class InChIParser(BaseParser):
    """
    Parser for InChI files.

    Reads InChI strings and dynamically extracts:
      - Atom-level numeric features
      - Bond-level numeric features
      - Molecular descriptors
      - Multiple fingerprint representations

    All extracted information is returned as separate feature-dimension vectors.

    Output (sequence):
        Dict[str, np.ndarray] where each key corresponds to a dynamically
        discovered feature dimension and each value is a vector aligned
        across molecules.

    Output (metadata):
        Dict[str, np.ndarray] dynamically extracted from molecule-level
        and file-level properties, with no hard-coded schema.
    """

    async def _read_file(self, filepath: str):
        """
        Reads an InChI file line by line.

        Args:
            filepath (str): Path to the InChI file.

        Returns:
            List[str]: List of InChI strings.
        """
        self._log_state("READ_FILE_START", f"Opening InChI file: {filepath}")
        with open(filepath, "r") as f:
            inchi_list = [line.strip() for line in f if line.strip()]
        self._log_state("READ_FILE_DONE", f"Loaded {len(inchi_list)} InChI entries")
        return inchi_list

    def extract_all(self, raw_data, **kwargs):
        """
        Unified extraction of molecule features and metadata from InChI strings.

        Extracts all numeric, categorical, and structural information
        dynamically from RDKit molecule objects, including graph embeddings
        and molecule-level properties.

        Args:
            raw_data (List[str]): List of InChI strings.
            **kwargs: Additional arguments (unused).

        Returns:
            dict with keys:
                "sequence": dict[str, np.ndarray] - feature vectors from atoms, bonds, descriptors, fingerprints
                "metadata": dict[str, np.ndarray] - molecule-level properties
        """
        self._log_state("EXTRACT_START", "Extracting InChI features and metadata")

        feature_store = {}
        metadata = {}

        # Phase 1: Extract features and metadata in single pass
        for idx, inchi in enumerate(raw_data):
            self._log_state("PROCESS_MOLECULE_START", f"Processing molecule {idx}")
            mol = Chem.MolFromInchi(inchi)
            if mol is None:
                self._log_state("PROCESS_MOLECULE_SKIP", f"Invalid InChI at index {idx}")
                continue

            # ---------------------------
            # Atom-level dynamic features
            # ---------------------------
            self._log_state("ATOM_FEATURE_START", f"Extracting atom-level features for molecule {idx}")
            for atom in mol.GetAtoms():
                for attr in dir(atom):
                    if attr.startswith("_"):
                        continue
                    try:
                        val = getattr(atom, attr)
                        if callable(val):
                            val = val()
                        if isinstance(val, (int, float, np.number)):
                            feature_store.setdefault(f"atom_{attr}", []).append(val)
                    except Exception:
                        continue
            self._log_state("ATOM_FEATURE_DONE", f"Atom-level features extracted for molecule {idx}")

            # ---------------------------
            # Bond-level dynamic features
            # ---------------------------
            self._log_state("BOND_FEATURE_START", f"Extracting bond-level features for molecule {idx}")
            for bond in mol.GetBonds():
                for attr in dir(bond):
                    if attr.startswith("_"):
                        continue
                    try:
                        val = getattr(bond, attr)
                        if callable(val):
                            val = val()
                        if isinstance(val, (int, float, np.number)):
                            feature_store.setdefault(f"bond_{attr}", []).append(val)
                    except Exception:
                        continue
            self._log_state("BOND_FEATURE_DONE", f"Bond-level features extracted for molecule {idx}")

            # ---------------------------
            # Molecular descriptors
            # ---------------------------
            self._log_state("DESCRIPTOR_START", f"Extracting molecular descriptors for molecule {idx}")
            for name, func in Descriptors._descList:
                try:
                    val = func(mol)
                    if isinstance(val, (int, float, np.number)):
                        feature_store.setdefault(f"desc_{name}", []).append(val)
                except Exception:
                    continue
            self._log_state("DESCRIPTOR_DONE", f"Molecular descriptors extracted for molecule {idx}")

            # ---------------------------
            # Fingerprints (multiple types)
            # ---------------------------
            self._log_state("FINGERPRINT_START", f"Extracting fingerprints for molecule {idx}")

            # Morgan / ECFP
            morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=4096)
            for i, bit in enumerate(morgan_fp):
                feature_store.setdefault(f"fp_morgan_bit_{i}", []).append(bit)

            # RDKit fingerprint
            rdkit_fp = Chem.RDKFingerprint(mol)
            for i, bit in enumerate(rdkit_fp):
                feature_store.setdefault(f"fp_rdkit_bit_{i}", []).append(bit)

            # MACCS keys
            maccs_fp = AllChem.GetMACCSKeysFingerprint(mol)
            for i, bit in enumerate(maccs_fp):
                feature_store.setdefault(f"fp_maccs_bit_{i}", []).append(bit)

            self._log_state("FINGERPRINT_DONE", f"Fingerprints extracted for molecule {idx}")

            # ---------------------------
            # Graph Embedding
            # ---------------------------
            self._log_state("GRAPH_EMBEDDING_START", f"Extracting graph embeddings for molecule {idx}")
            try:
                G = nx.Graph()
                for atom in mol.GetAtoms():
                    G.add_node(atom.GetIdx(), atomic_num=atom.GetAtomicNum())
                for bond in mol.GetBonds():
                    G.add_edge(
                        bond.GetBeginAtomIdx(),
                        bond.GetEndAtomIdx(),
                        bond_type=bond.GetBondTypeAsDouble()
                    )

                node_features = []
                for node in G.nodes(data=True):
                    deg = G.degree[node[0]]
                    atomic_num = node[1]["atomic_num"]
                    node_features.extend([deg, atomic_num])

                feature_store.setdefault("graph_embedding", []).append(node_features)
            except Exception as e:
                self._log_state("GRAPH_EMBEDDING_FAIL", f"Failed graph embedding for molecule {idx}: {e}")
            self._log_state("GRAPH_EMBEDDING_DONE", f"Graph embedding extracted for molecule {idx}")

            # ---------------------------
            # Metadata extraction (molecule-level attributes)
            # ---------------------------
            self._log_state("METADATA_ATTR_START", f"Extracting molecule attributes for metadata {idx}")
            for attr in dir(mol):
                if attr.startswith("_"):
                    continue
                try:
                    val = getattr(mol, attr)
                    if callable(val):
                        val = val()
                    if isinstance(val, (int, float, np.number)):
                        metadata.setdefault(f"mol_{attr}", []).append(val)
                    elif isinstance(val, str):
                        metadata.setdefault(f"mol_{attr}", []).append(val)
                except Exception:
                    continue
            self._log_state("METADATA_ATTR_DONE", f"Molecule attributes extracted for metadata {idx}")

            self._log_state("PROCESS_MOLECULE_DONE", f"Finished processing molecule {idx}")

        # Phase 2: Vectorize features
        self._log_state("VECTORIZE_START", "Vectorizing InChI feature dimensions")
        vectors = {}
        for key, values in feature_store.items():
            try:
                vectors[key] = np.array(values, dtype=np.float32)
            except Exception:
                vectors[key] = np.frombuffer(" ".join(map(str, values)).encode(), dtype=np.uint8)

        self._log_state("EXTRACT_SEQUENCE_DONE", f"Extracted {len(vectors)} feature vectors")

        # Phase 3: Vectorize metadata
        self._log_state("VECTORIZE_METADATA_START", "Vectorizing metadata dimensions")
        for key, values in metadata.items():
            try:
                metadata[key] = np.array(values, dtype=np.float32)
            except Exception:
                metadata[key] = np.frombuffer(" ".join(map(str, values)).encode(), dtype=np.uint8)

        self._log_state("EXTRACT_METADATA_DONE", f"Extracted {len(metadata)} metadata dimensions")

        return {"sequence": vectors, "metadata": metadata}




class SDFParser(BaseParser):
    """
    Parser for SDF files.

    Extracts molecular descriptors, 3D coordinates, dynamic properties,
    and fingerprint-based features as aligned multi-dimensional vectors.

    Output (sequence):
        Dict[str, np.ndarray] of feature matrices.

    Output (metadata):
        File-level and molecule-level metadata.
    """

    async def _read_file(self, filepath: str):
        self._log_state("READ_FILE_START", f"Opening SDF file: {filepath}")
        supplier = Chem.SDMolSupplier(filepath)
        mols = [mol for mol in supplier if mol is not None]
        self._log_state("READ_FILE_DONE", f"Loaded {len(mols)} molecules from SDF")
        return mols

    def extract_all(self, raw_data, **kwargs):
        """
        Unified extraction of SDF molecule features and metadata in a single pass.

        Args:
            raw_data: List of RDKit molecule objects from SDMolSupplier.
            **kwargs: Additional arguments (unused).

        Returns:
            dict with keys:
                "sequence": dict[str, np.ndarray] - descriptors, fingerprints, coordinates, graph embeddings
                "metadata": dict[str, np.ndarray] - molecule-level properties and characteristics
        """
        self._log_state("EXTRACT_START", "Extracting SDF features and metadata")

        descriptors = []
        fingerprints = []
        coordinates = []
        properties = {}
        graph_embeddings = []
        raw_metadata = []
        all_metadata_keys = set()

        # Phase 1: Extract features and metadata in single pass through molecules
        for idx, mol in enumerate(raw_data):
            self._log_state("PROCESS_MOLECULE", f"Processing molecule {idx}")

            mol_meta = {}

            # ------- Molecular descriptors -------
            desc_vector = []
            for name, func in Descriptors._descList:
                try:
                    val = func(mol)
                    desc_vector.append(float(val))
                except Exception:
                    desc_vector.append(0.0)
            descriptors.append(desc_vector)

            # ------ Fingerprints (Morgan) ------
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=4096)
            fingerprints.append(np.array(fp, dtype=np.float32))

            # ------- 3D Coordinates (flattened per molecule) -------
            if mol.GetNumConformers() > 0:
                conf = mol.GetConformer()
                mol_coords = []
                for i in range(mol.GetNumAtoms()):
                    pos = conf.GetAtomPosition(i)
                    mol_coords.extend([pos.x, pos.y, pos.z])
                coordinates.append(mol_coords)
            else:
                coordinates.append([])

            # --- Dynamic properties (features) ---
            for prop in mol.GetPropNames():
                try:
                    val = mol.GetProp(prop)
                    try:
                        val_num = float(val)
                    except ValueError:
                        val_num = float(len(val))
                    properties.setdefault(prop, []).append(val_num)
                except Exception:
                    continue

            # ------- Graph Embedding -------
            self._log_state("GRAPH_EMBEDDING_START", f"Extracting graph embedding for molecule {idx}")
            try:
                G = nx.Graph()
                for atom in mol.GetAtoms():
                    G.add_node(atom.GetIdx(), atomic_num=atom.GetAtomicNum())
                for bond in mol.GetBonds():
                    G.add_edge(
                        bond.GetBeginAtomIdx(),
                        bond.GetEndAtomIdx(),
                        bond_type=bond.GetBondTypeAsDouble()
                    )

                node_features = []
                for node in G.nodes(data=True):
                    deg = G.degree[node[0]]
                    atomic_num = node[1]["atomic_num"]
                    node_features.extend([deg, atomic_num])

                graph_embeddings.append(node_features)
            except Exception as e:
                self._log_state("GRAPH_EMBEDDING_FAIL", f"Failed graph embedding for molecule {idx}: {e}")
                graph_embeddings.append([])

            self._log_state("GRAPH_EMBEDDING_DONE", f"Graph embedding extracted for molecule {idx}")

            # ------- Metadata extraction (same iteration) -------
            self._log_state("METADATA_START", f"Extracting metadata for molecule {idx}")

            # 3D coordinates presence
            mol_meta["3d_coordinates"] = mol.GetNumConformers() > 0

            # Molecule properties
            for prop in mol.GetPropNames():
                mol_meta[f"prop_{prop}"] = mol.GetProp(prop)

            # Descriptors
            for name, func in Descriptors._descList:
                try:
                    val = func(mol)
                    mol_meta[f"desc_{name}"] = val
                except Exception:
                    continue

            raw_metadata.append(mol_meta)
            all_metadata_keys.update(mol_meta.keys())
            self._log_state("METADATA_DONE", f"Metadata extracted for molecule {idx}")

        # Phase 2: Vectorize features
        self._log_state("VECTORIZE_START", "Vectorizing SDF feature dimensions")
        vectors = {
            "descriptors": np.array(descriptors, dtype=np.float32),
            "fingerprint_morgan_2048": np.array(fingerprints, dtype=np.float32),
            "coordinates": np.array(coordinates, dtype=np.float32),
            "graph_embedding": np.array(graph_embeddings, dtype=np.float32),
        }

        for prop, values in properties.items():
            vectors[f"property_{prop}"] = np.array(values, dtype=np.float32)

        self._log_state("EXTRACT_SEQUENCE_DONE", f"Extracted {len(vectors)} feature matrices including graph embeddings")

        # Phase 3: Vectorize metadata
        self._log_state("VECTORIZE_METADATA_START", "Vectorizing metadata dimensions")
        metadata_vectors = {}
        for key in all_metadata_keys:
            vec = []
            for mol_meta in raw_metadata:
                val = mol_meta.get(key, np.nan)  # NaN if not present
                try:
                    val = float(val)
                except Exception:
                    val = float(len(str(val)))  # compute length for non-numeric
                vec.append(val)
            metadata_vectors[key] = np.array(vec, dtype=np.float32)

        self._log_state("EXTRACT_METADATA_DONE", f"Extracted {len(metadata_vectors)} metadata feature vectors")

        return {"sequence": vectors, "metadata": metadata_vectors}




class MOLParser(BaseParser):
    """
    Parser for MOL files.

    Reads MOL records and dynamically extracts atom-level, bond-level,
    coordinate, descriptor, and fingerprint-based features as separate
    feature-dimension vectors.

    Output (sequence):
        Dict[str, np.ndarray] of feature vectors.

    Output (metadata):
        File-level and molecule-level metadata.
    """

    async def _read_file(self, filepath: str):
        self._log_state("READ_FILE_START", f"Opening MOL file: {filepath}")
        mol = Chem.MolFromMolFile(filepath, sanitize=True)
        mols = [mol] if mol is not None else []
        self._log_state("READ_FILE_DONE", f"Loaded {len(mols)} molecule from MOL file")
        return mols

    def extract_all(self, raw_data, **kwargs):
        """
        Unified extraction of MOL molecule features and metadata in a single pass.

        Args:
            raw_data: List of RDKit molecule objects.
            **kwargs: Additional arguments (unused).

        Returns:
            dict with keys:
                "sequence": dict[str, np.ndarray] - atom/bond features, coordinates, descriptors, fingerprints, graph embeddings
                "metadata": dict[str, np.ndarray] - molecule-level properties and characteristics
        """
        self._log_state("EXTRACT_START", "Extracting MOL features and metadata")

        feature_store = {}
        raw_metadata = []
        all_metadata_keys = set()

        # Phase 1: Extract features and metadata in single pass
        for idx, mol in enumerate(raw_data):
            self._log_state("PROCESS_MOLECULE", f"Processing molecule {idx}")

            mol_meta = {}

            # ---------------------------
            # Atom-level features
            # ---------------------------
            for atom in mol.GetAtoms():
                for attr in dir(atom):
                    if attr.startswith("_"):
                        continue
                    try:
                        val = getattr(atom, attr)
                        if callable(val):
                            val = val()
                        if isinstance(val, (int, float)):
                            feature_store.setdefault(f"atom_{attr}", []).append(val)
                    except Exception:
                        continue

            # ---------------------------
            # Bond-level features
            # ---------------------------
            for bond in mol.GetBonds():
                for attr in dir(bond):
                    if attr.startswith("_"):
                        continue
                    try:
                        val = getattr(bond, attr)
                        if callable(val):
                            val = val()
                        if isinstance(val, (int, float)):
                            feature_store.setdefault(f"bond_{attr}", []).append(val)
                    except Exception:
                        continue

            # ---------------------------
            # Coordinates
            # ---------------------------
            if mol.GetNumConformers() > 0:
                conf = mol.GetConformer()
                for i in range(mol.GetNumAtoms()):
                    pos = conf.GetAtomPosition(i)
                    feature_store.setdefault("coord_x", []).append(pos.x)
                    feature_store.setdefault("coord_y", []).append(pos.y)
                    feature_store.setdefault("coord_z", []).append(pos.z)

            # ---------------------------
            # Molecular descriptors
            # ---------------------------
            for name, func in Descriptors._descList:
                try:
                    val = func(mol)
                    if isinstance(val, (int, float)):
                        feature_store.setdefault(f"desc_{name}", []).append(val)
                except Exception:
                    continue

            # ---------------------------
            # Fingerprints
            # ---------------------------
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=4096)
            for i, bit in enumerate(fp):
                feature_store.setdefault(f"fp_bit_{i}", []).append(bit)

            # ---------------------------
            # Graph embedding
            # ---------------------------
            try:
                from rdkit.Chem.rdmolops import GetAdjacencyMatrix

                # Molekülden adjacency matrix al
                adj = GetAdjacencyMatrix(mol)
                G = nx.from_numpy_array(adj)

                # Basit bir node feature olarak atom türleri kullan
                atom_types = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
                for i, atom_num in enumerate(atom_types):
                    G.nodes[i]['feat'] = atom_num

                # Graph embedding için basit bir yöntem: degree + atom type histogram
                degrees = np.array([d for n, d in G.degree()])
                atom_hist = np.histogram(atom_types, bins=range(1, 119))[0]  # H-Z=118
                graph_embedding = np.concatenate([degrees, atom_hist])
                feature_store.setdefault("graph_embedding", []).append(graph_embedding)
            except Exception as e:
                self._log_state("GRAPH_EMBED_FAIL", f"Molecule {idx} graph embedding failed: {e}")
                feature_store.setdefault("graph_embedding", []).append(np.zeros(118 + mol.GetNumAtoms(), dtype=np.float32))

            # ------- Metadata extraction (same iteration) -------
            self._log_state("METADATA_START", f"Extracting metadata for molecule {idx}")

            # 3D coordinates presence
            mol_meta["3d_coordinates"] = mol.GetNumConformers() > 0

            # Molecule properties
            for prop in mol.GetPropNames():
                mol_meta[f"prop_{prop}"] = mol.GetProp(prop)

            # Descriptors
            for name, func in Descriptors._descList:
                try:
                    val = func(mol)
                    mol_meta[f"desc_{name}"] = val
                except Exception:
                    continue

            raw_metadata.append(mol_meta)
            all_metadata_keys.update(mol_meta.keys())
            self._log_state("METADATA_DONE", f"Metadata extracted for molecule {idx}")

        # Phase 2: Vectorize features
        self._log_state("VECTORIZE_START", "Vectorizing MOL feature dimensions")
        vectors = {}
        for key, values in feature_store.items():
            try:
                if key == "graph_embedding":
                    # graph embedding list of arrays -> pad to same length
                    max_len = max(len(v) for v in values)
                    padded = np.array([np.pad(v, (0, max_len - len(v)), 'constant') for v in values], dtype=np.float32)
                    vectors[key] = padded
                else:
                    vectors[key] = np.array(values, dtype=np.float32)
            except Exception:
                vectors[key] = np.frombuffer(" ".join(map(str, values)).encode(), dtype=np.uint8)

        self._log_state("EXTRACT_SEQUENCE_DONE", f"Extracted {len(vectors)} feature vectors (including graph embedding)")

        # Phase 3: Vectorize metadata
        self._log_state("VECTORIZE_METADATA_START", "Vectorizing metadata dimensions")
        metadata_vectors = {}
        for key in all_metadata_keys:
            vec = []
            for mol_meta in raw_metadata:
                val = mol_meta.get(key, np.nan)  # olmayan alanlara NaN koy
                try:
                    val = float(val)
                except Exception:
                    val = float(len(str(val)))  # stringler varsa uzunluğunu say
                vec.append(val)
            metadata_vectors[key] = np.array(vec, dtype=np.float32)

        self._log_state("EXTRACT_METADATA_DONE", f"Extracted {len(metadata_vectors)} metadata feature vectors")

        return {"sequence": vectors, "metadata": metadata_vectors}


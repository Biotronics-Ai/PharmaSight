import os
import asyncio
import math
import tempfile
from math import sqrt
from typing import Dict, List, Optional, Tuple
from base import BaseParser, BaseComponent
from parsers import ( 
    FASTAParser, 
    SMILESParser, 
    SDFParser, 
    PDBStructureParser, 
    MOLParser, 
    MMCIFStructureParser,
    MZMLParser,
    MZXMLParser,
    MZTabParser,
    #MzDataParser,
    MZIdentMLParser,
    PepXMLParser,
    #PSIMITABParser,
    EdgeListParser,
    BioPAXRDFParser,
    InChIParser,
)

import numpy as np



class MolSample:
    """
        Represents the molecular or genomic sample vectorized presentation.

        Holds parser outputs in a unified structure:
            - features: Dict of all feature-dimension vectors (atom, bond, descriptor, fingerprints, graph embeddings)
            - metadata: Dict of all molecule/file-level metadata vectors
            - n_positions: optional positional information (e.g., atom indices, sequence positions)
            - memmap_path: optional path if vectors are stored externally to save RAM
    """
    def __init__(
        self,
        sample_id: str,
        features: Dict[str, np.ndarray],        
        file_format: str,
        n_positions: Optional[Dict[str, np.ndarray]] = None,        
        metadata: Optional[Dict[str, np.ndarray]] = None,
        memmap_path: Optional[str] = None,
    ):
        self.id = sample_id
        self.features = features           
        self.n_positions = n_positions or []
        self.format = file_format
        self.metadata = metadata or {}
        self.memmap_path = memmap_path
        self._sequence_cache = None  # Cache for loaded memmap

    @property
    def sequence(self) -> np.ndarray:
        """
        Lazily loads and returns the sequence vector.
        
        If features["sequence"] is a string path, loads the memmap from disk.
        Otherwise returns the array directly.
        
        Returns:
            np.ndarray: The sequence vector (possibly memory-mapped).
        """
        # Return cached if already loaded
        if self._sequence_cache is not None:
            return self._sequence_cache
        
        seq_data = self.features.get("sequence")
        
        if seq_data is None:
            raise ValueError(f"Sample {self.id} has no sequence data in features")
        
        # If it's a string path, load memmap
        if isinstance(seq_data, str):
            if not os.path.exists(seq_data):
                raise FileNotFoundError(f"Memmap file not found: {seq_data}")
            
            # Load as read-only memmap to save memory
            self._sequence_cache = np.load(seq_data, mmap_mode='r')
            return self._sequence_cache
        
        # It's already an array
        self._sequence_cache = seq_data
        return self._sequence_cache
    
    def clear_cache(self):
        """Clear the cached sequence to free memory."""
        self._sequence_cache = None  



class ParserFactory(BaseComponent):
    """
    Factory class to return the appropriate parser instance based on file extension.

    This class manages pre-defined parser classes for different bioinformatics,
    cheminformatics, and structural biology file formats, and returns an
    instance of the correct parser for the given file path.

    Supported formats:
        - Genomic / Sequence: FASTA (.fa, .fasta)
        - Molecular / Chemical: SMILES (.smiles, .smi), InChI (.inchi, .stdinchi),
          SDF (.sdf), MOL (.mol, .mdl)
        - Structural Biology: PDB (.pdb), MMCIF (.mmcif)
        - Proteomics / Metabolomics: mzML (.mzml), mzXML (.mzxml), PepXML (.pepxml),
          MzIdentML (.mzidentml), MzTab (.mztab), mzData (.mzdata), PSIMITAB (.psimitab)
        - Network / Pathway / RDF: EdgeList (.edgelist), BioPAX RDF (.biopax)

    Usage:
        parser = ParserFactory.get_parser("example.fasta")
        vectors = parser.extract_sequence(...)
        metadata = parser.extract_metadata(...)

    Methods:
        get_parser(filepath: str) -> BaseParser
            Returns an instance of the parser class corresponding to the file extension.

            Args:
                filepath (str): Full path or name of the input file.

            Returns:
                BaseParser: An instance of the parser class for the file format.

            Raises:
                ValueError: If the file extension is unsupported.

    Notes:
        - ParserFactory only returns the parser instance; actual file reading
          and feature extraction are performed through the returned parser.
        - All parsers inherit from BaseParser, ensuring that common methods
          such as extract_sequence and extract_metadata are available.
        - To add support for a new file format, simply add the file extension
          and corresponding parser class to the _PARSER_MAP dictionary.
    """

    # Tüm desteklenen parserlar
    _PARSER_MAP = {
        # Genomic / sequence
        "fa": FASTAParser,
        "fasta": FASTAParser,

        # Molecules / chemical formats
        "smiles": SMILESParser,
        "smi": SMILESParser,
        "inchi": InChIParser,
        "stdinchi": InChIParser,
        "sdf": SDFParser,
        "mol": MOLParser,
        "mdl": MOLParser,

        # Structural biology
        "pdb": PDBStructureParser,
        "mmcif": MMCIFStructureParser,

        # Mass spectrometry proteomics/metabolomics
        "mzml": MZMLParser,
        "mzxml": MZXMLParser,
        "pepxml": PepXMLParser,
        "mzidentml": MZIdentMLParser,
        "mztab": MZTabParser,
        #"mzdata": MzDataParser,
        #"psimitab": PSIMITABParser,

        # Network / pathway formats
        "edgelist": EdgeListParser,
        "biopax": BioPAXRDFParser,
    }

    @staticmethod
    def get_parser(filepath: str, memmap_dir: str = None) -> BaseParser:
        ext = filepath.lower().split('.')[-1]
        cls = ParserFactory._PARSER_MAP.get(ext)
        if not cls:
            raise ValueError(f"Unsupported file format: {filepath}")
        return cls(memmap_dir=memmap_dir)




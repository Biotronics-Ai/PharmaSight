from pyteomics import mzdata


class MzDataParser(BaseParser):
    """
    mzData parser that dynamically extracts *all* numeric and vector-like
    fields from every spectrum and nested structures as independent
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
        Reads mzData file using pyteomics.

        Args:
            filepath (str): Path to mzData file.

        Returns:
            mzData reader object (iterator over spectra).
        """
        self._log_state("READ_START", f"Opening mzData file: {filepath}")
        reader = mzdata.read(filepath)
        self._log_state("READ_COMPLETE", "mzData file successfully opened")
        return reader

    def _extract_sequence(self, raw_data):
        """
        Dynamically extracts every numeric field from spectra and nested
        structures (precursor, instrument, acquisition, peak list, etc.)
        and stores each field as a separate feature-dimension vector.

        Args:
            raw_data: mzData reader iterator.

        Returns:
            dict[str, np.ndarray or list[np.ndarray]]:
                Mapping from feature name → vector across spectra.
        """
        self._log_state("EXTRACT_START", "Beginning dynamic feature extraction from mzData")

        feature_store = {}
        spectrum_count = 0

        for spec_idx, spectrum in enumerate(raw_data):
            self._log_state("SPECTRUM_START", f"Processing spectrum {spec_idx}")

            # Process top-level spectrum fields
            for key, value in spectrum.items():
                feature_key = f"SPECTRUM.{key}"

                try:
                    if feature_key not in feature_store:
                        feature_store[feature_key] = []

                    if isinstance(value, (list, tuple, np.ndarray)):
                        vec = np.array(value, dtype=np.float32)
                        feature_store[feature_key].append(vec)
                        self._log_state(
                            "FEATURE_VECTOR",
                            f"spectrum {spec_idx} | Feature '{feature_key}' vector shape {vec.shape}"
                        )

                    elif isinstance(value, (int, float)):
                        vec = np.array([value], dtype=np.float32)
                        feature_store[feature_key].append(vec)
                        self._log_state(
                            "FEATURE_SCALAR",
                            f"spectrum {spec_idx} | Feature '{feature_key}' scalar vectorized"
                        )

                    else:
                        self._log_state(
                            "FEATURE_SKIP",
                            f"spectrum {spec_idx} | Feature '{feature_key}' non-numeric, skipped"
                        )

                except Exception as e:
                    self._log_state(
                        "FEATURE_ERROR",
                        f"spectrum {spec_idx} | Feature '{feature_key}' extraction failed",
                        error=str(e)
                    )

            # Process precursor information
            precursor = spectrum.get("precursor")
            if precursor:
                self._log_state("PRECURSOR_START", f"Processing precursor for spectrum {spec_idx}")
                for key, value in precursor.items():
                    feature_key = f"PRECURSOR.{key}"

                    try:
                        if feature_key not in feature_store:
                            feature_store[feature_key] = []

                        if isinstance(value, (list, tuple, np.ndarray)):
                            vec = np.array(value, dtype=np.float32)
                            feature_store[feature_key].append(vec)
                            self._log_state(
                                "FEATURE_VECTOR",
                                f"precursor | Feature '{feature_key}' vector shape {vec.shape}"
                            )

                        elif isinstance(value, (int, float)):
                            vec = np.array([value], dtype=np.float32)
                            feature_store[feature_key].append(vec)
                            self._log_state(
                                "FEATURE_SCALAR",
                                f"precursor | Feature '{feature_key}' scalar vectorized"
                            )

                        else:
                            self._log_state(
                                "FEATURE_SKIP",
                                f"precursor | Feature '{feature_key}' non-numeric, skipped"
                            )

                    except Exception as e:
                        self._log_state(
                            "FEATURE_ERROR",
                            f"precursor | Feature '{feature_key}' extraction failed",
                            error=str(e)
                        )

            # Process peak list (m/z and intensity arrays, etc.)
            mz_array = spectrum.get("m/z array")
            intensity_array = spectrum.get("intensity array")

            if mz_array is not None:
                feature_key = "PEAKS.MZ"
                try:
                    if feature_key not in feature_store:
                        feature_store[feature_key] = []
                    vec = np.array(mz_array, dtype=np.float32)
                    feature_store[feature_key].append(vec)
                    self._log_state(
                        "FEATURE_VECTOR",
                        f"spectrum {spec_idx} | Feature '{feature_key}' vector shape {vec.shape}"
                    )
                except Exception as e:
                    self._log_state(
                        "FEATURE_ERROR",
                        f"spectrum {spec_idx} | Feature '{feature_key}' extraction failed",
                        error=str(e)
                    )

            if intensity_array is not None:
                feature_key = "PEAKS.INTENSITY"
                try:
                    if feature_key not in feature_store:
                        feature_store[feature_key] = []
                    vec = np.array(intensity_array, dtype=np.float32)
                    feature_store[feature_key].append(vec)
                    self._log_state(
                        "FEATURE_VECTOR",
                        f"spectrum {spec_idx} | Feature '{feature_key}' vector shape {vec.shape}"
                    )
                except Exception as e:
                    self._log_state(
                        "FEATURE_ERROR",
                        f"spectrum {spec_idx} | Feature '{feature_key}' extraction failed",
                        error=str(e)
                    )

            spectrum_count += 1

        if not feature_store:
            self._log_state("EXTRACT_EMPTY", "No numeric features extracted from mzData file")
            return {}

        self._log_state(
            "EXTRACT_FEATURES_COMPLETE",
            f"Extracted {len(feature_store)} feature dimensions from {spectrum_count} spectra"
        )

        # Normalize each feature to a uniform 2D matrix
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

        self._log_state("EXTRACT_COMPLETE", "All mzData features extracted and normalized")
        return normalized_features

    def _extract_metadata(self, raw_data):
        """
        Dynamically extracts run-level and instrument-level metadata from mzData.
        """
        self._log_state("METADATA_START", "Beginning dynamic metadata extraction")

        metadata = {}
        try:
            if hasattr(raw_data, "file_description"):
                for key, value in raw_data.file_description.items():
                    if isinstance(value, (int, float, str)):
                        metadata[key] = value
                    elif hasattr(value, "__len__") and not isinstance(value, str):
                        metadata[key] = len(value)

            if hasattr(raw_data, "instrument_configuration"):
                for key, value in raw_data.instrument_configuration.items():
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

        return metadata





from pyteomics import psimi


class PSIMITABParser(BaseParser):
    """
    Parser for PSI-MI TAB format (.mitab) files.

    Reads interaction records dynamically and extracts all available numeric
    and categorical fields as separate feature-dimension vectors.

    Output (sequence):
        Dict[str, np.ndarray] where each key is a feature name and each value
        is a vector aligned across interactions.

    Output (metadata):
        Dict[str, np.ndarray] dynamically extracted from all record fields.
    """

    async def _read_file(self, filepath: str):
        self._log_state("READ_FILE_START", f"Opening PSI-MI TAB file: {filepath}")
        data = list(psimi.read(filepath))
        self._log_state("READ_FILE_DONE", f"Loaded {len(data)} interaction records")
        return data

    def _extract_sequence(self, raw_data):
        self._log_state("EXTRACT_SEQUENCE_START", "Extracting interaction features")

        feature_store = {}

        for idx, record in enumerate(raw_data):
            self._log_state("PROCESS_RECORD", f"Processing interaction {idx}")
            for key, value in record.items():
                if key not in feature_store:
                    feature_store[key] = []

                if isinstance(value, (int, float)):
                    feature_store[key].append(value)
                elif isinstance(value, str):
                    feature_store[key].append(value)
                elif isinstance(value, (list, tuple)):
                    feature_store[key].append(len(value))
                else:
                    feature_store[key].append(np.nan)

        self._log_state("VECTORIZE_START", "Vectorizing feature dimensions")
        vectors = {}
        for key, values in feature_store.items():
            if all(isinstance(v, (int, float, np.number)) or v is None for v in values):
                vectors[key] = np.array(values, dtype=np.float32)
            else:
                vectors[key] = np.frombuffer(" ".join(map(str, values)).encode(), dtype=np.uint8)

        self._log_state("EXTRACT_SEQUENCE_DONE", f"Extracted {len(vectors)} feature vectors")
        return vectors

    def _extract_metadata(self, raw_data):
        self._log_state("EXTRACT_METADATA_START", "Dynamically extracting PSI-MI TAB metadata")

        metadata_store = {}

        for idx, record in enumerate(raw_data):
            self._log_state("PROCESS_METADATA_RECORD", f"Processing metadata from interaction {idx}")
            for key, value in record.items():
                meta_key = f"meta_{key}"
                if meta_key not in metadata_store:
                    metadata_store[meta_key] = []

                if isinstance(value, (int, float)):
                    metadata_store[meta_key].append(value)
                elif isinstance(value, str):
                    metadata_store[meta_key].append(value)
                elif isinstance(value, (list, tuple)):
                    metadata_store[meta_key].append(len(value))
                else:
                    metadata_store[meta_key].append(np.nan)

        self._log_state("VECTORIZE_METADATA_START", "Vectorizing metadata dimensions")
        metadata = {}
        for key, values in metadata_store.items():
            if all(isinstance(v, (int, float, np.number)) or v is None for v in values):
                metadata[key] = np.array(values, dtype=np.float32)
            else:
                metadata[key] = np.frombuffer(" ".join(map(str, values)).encode(), dtype=np.uint8)

        self._log_state("EXTRACT_METADATA_DONE", f"Dynamically extracted {len(metadata)} metadata dimensions")
        return metadata





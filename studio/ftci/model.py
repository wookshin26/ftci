"""Portable coefficient-MPS models and a NumPy-only prediction archive."""
from __future__ import annotations

import io
import json
import math
from pathlib import Path
from typing import Mapping, Sequence
import zipfile

import numpy as np

from .basis import Basis


_MODEL_SCHEMA = "ftci.model"
_MODEL_VERSION = 1
_MAX_ARCHIVE_BYTES = 128 * 1024 * 1024
_MAX_MANIFEST_BYTES = 2 * 1024 * 1024
_MAX_PREDICTOR_BYTES = 512 * 1024
_MAX_SITES = 512


def _jsonable(value):
    """Convert numerical diagnostics to strict JSON, representing NaN as null."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("Metadata object keys must be strings.")
        return {key: _jsonable(item) for key, item in value.items()}
    raise ValueError(f"Metadata contains an unsupported value of type {type(value).__name__}.")


def _validate_npz(data: bytes, core_specs: list[dict]) -> None:
    """Check nested NPY headers before NumPy can allocate their declared arrays."""
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            members = archive.infolist()
            expected = {f"core_{k}.npy" for k in range(len(core_specs))}
            if len(members) != len(expected) or {member.filename for member in members} != expected:
                raise ValueError("Coefficient archive has unexpected or duplicate members.")
            if sum(member.file_size for member in members) > _MAX_ARCHIVE_BYTES:
                raise ValueError("Coefficient archive exceeds the 128 MiB loading limit.")
            for index, spec in enumerate(core_specs):
                member = archive.getinfo(f"core_{index}.npy")
                with archive.open(member) as stream:
                    version = np.lib.format.read_magic(stream)
                    if version == (1, 0):
                        shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
                    elif version == (2, 0):
                        shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
                    else:
                        raise ValueError("Unsupported coefficient array format.")
                    if list(shape) != spec["shape"] or dtype != np.dtype("float64"):
                        raise ValueError("Coefficient array header disagrees with its model manifest.")
                    nbytes = math.prod(shape) * dtype.itemsize
                    if nbytes > _MAX_ARCHIVE_BYTES or member.file_size != stream.tell() + nbytes:
                        raise ValueError("Coefficient array size is invalid or exceeds the loading limit.")
    except (zipfile.BadZipFile, EOFError, OSError, TypeError, OverflowError,
            RuntimeError, NotImplementedError) as exc:
        raise ValueError("Invalid coefficient array archive.") from exc


def _read_archive(path) -> tuple[dict, bytes]:
    try:
        if Path(path).stat().st_size > _MAX_ARCHIVE_BYTES:
            raise ValueError("Model file exceeds the 128 MiB loading limit.")
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            names = [member.filename for member in members]
            if len(names) != len(set(names)) or not {"manifest.json", "cores.npz"}.issubset(names):
                raise ValueError("Model archive is missing files or contains duplicate members.")
            if set(names) - {"manifest.json", "cores.npz", "predict.py"}:
                raise ValueError("Model archive contains unexpected files.")
            if sum(member.file_size for member in members) > _MAX_ARCHIVE_BYTES:
                raise ValueError("Model archive exceeds the 128 MiB loading limit.")
            if archive.getinfo("manifest.json").file_size > _MAX_MANIFEST_BYTES:
                raise ValueError("Model manifest is too large.")
            if "predict.py" in names and archive.getinfo("predict.py").file_size > _MAX_PREDICTOR_BYTES:
                raise ValueError("Bundled predictor is too large.")
            manifest = json.loads(archive.read("manifest.json"))
            data = archive.read("cores.npz")
    except (zipfile.BadZipFile, json.JSONDecodeError, UnicodeError, EOFError,
            OSError, RuntimeError, NotImplementedError, RecursionError) as exc:
        raise ValueError("Could not read a valid FTCI model archive.") from exc
    if not isinstance(manifest, dict):
        raise ValueError("Model manifest must be an object.")
    if (manifest.get("schema") != _MODEL_SCHEMA
            or type(manifest.get("version")) is not int
            or manifest.get("version") != _MODEL_VERSION):
        raise ValueError("Unsupported FTCI model format or version.")
    specs = manifest.get("cores")
    if not isinstance(specs, list) or not 1 <= len(specs) <= _MAX_SITES:
        raise ValueError("Model manifest has an invalid core list.")
    total_bytes = 0
    for spec in specs:
        if not isinstance(spec, dict) or set(spec) != {"shape", "dtype"}:
            raise ValueError("Invalid core specification.")
        shape = spec.get("shape")
        if (not isinstance(shape, list) or len(shape) != 3
                or any(isinstance(d, bool) or not isinstance(d, int) or d < 1 or d > 65536 for d in shape)
                or spec.get("dtype") != "float64"):
            raise ValueError("Invalid coefficient core dimensions or type.")
        total_bytes += math.prod(shape) * 8
    if total_bytes > _MAX_ARCHIVE_BYTES:
        raise ValueError("Coefficient arrays exceed the 128 MiB loading limit.")
    _validate_npz(data, specs)
    return manifest, data


def _standalone_source() -> str:
    """Bundle the exact expression, basis and model implementation, without imports.

    This keeps exported predictions on the same evaluation/validation path as
    the SDK; generated code does not execute any source stored in a model.
    """
    if globals().get("_PORTABLE_PREDICTOR", False):
        return Path(__file__).read_text(encoding="utf-8")
    directory = Path(__file__).resolve().parent
    sections = ["#!/usr/bin/env python3\nfrom __future__ import annotations\n_PORTABLE_PREDICTOR = True\n"]
    for filename in ("expressions.py", "basis.py", "model.py"):
        source = (directory / filename).read_text(encoding="utf-8")
        lines = [line for line in source.splitlines()
                 if not line.startswith("from __future__ import ") and not line.startswith("from .")]
        sections.append("\n".join(lines))
    sections.append('''
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Predict with an FTCI model using only NumPy.")
    parser.add_argument("model", help="Path to a .ftci model archive")
    parser.add_argument("input", help='JSON input vector, e.g. "[0.2, 0.4]"')
    parser.add_argument("--batch", action="store_true", help="Input is an array of input vectors")
    args = parser.parse_args()
    try:
        learned = Model.load(args.model)
        point = json.loads(args.input)
        prediction = learned.predict(point).tolist() if args.batch else learned(point)
        print(json.dumps(prediction, allow_nan=False))
    except (ValueError, TypeError, IndexError) as error:
        parser.exit(2, f"Prediction failed: {error}\\n")
''')
    return "\n\n".join(sections) + "\n"


class Model:
    """A scalar real function represented by dictionaries and coefficient MPS.

    Core indices follow ``(left bond, basis entry, right bond)``; coefficient
    indices are zero-based. Evaluation contracts cores without constructing
    the exponentially large coefficient tensor or product feature vector.
    """

    def __init__(self, basis: Sequence[Basis], cores: Sequence[np.ndarray],
                 domain: Sequence[Sequence[float]], metadata: Mapping | None = None,
                 name: str = "Untitled"):
        if not isinstance(basis, (list, tuple)) or not 1 <= len(basis) <= _MAX_SITES:
            raise ValueError("A model needs between 1 and 512 local dictionaries.")
        if not isinstance(cores, (list, tuple)) or len(basis) != len(cores):
            raise ValueError("There must be one coefficient core per dictionary.")
        if not all(isinstance(dictionary, Basis) for dictionary in basis):
            raise ValueError("All dictionaries must be Basis objects.")
        if not isinstance(domain, (list, tuple)) or not 1 <= len(domain) <= 4096:
            raise ValueError("Domain must specify between 1 and 4096 coordinate intervals.")
        checked_domain = []
        for interval in domain:
            if not isinstance(interval, (list, tuple, np.ndarray)) or len(interval) != 2 or np.iscomplexobj(interval):
                raise ValueError("Each domain interval must contain two finite real endpoints.")
            lower, upper = (float(endpoint) for endpoint in interval)
            if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
                raise ValueError("Domain intervals must have finite increasing endpoints.")
            checked_domain.append((lower, upper))
        if any(dictionary.input_indices and dictionary.input_indices[-1] >= len(checked_domain) for dictionary in basis):
            raise ValueError("A dictionary references a coordinate outside the model input dimension.")
        checked_cores = []
        left_bond = 1
        total_bytes = 0
        for index, (raw, dictionary) in enumerate(zip(cores, basis)):
            if np.iscomplexobj(raw):
                raise ValueError("Coefficient cores must be real arrays.")
            core = np.asarray(raw, dtype=np.float64)
            if (core.ndim != 3 or any(size < 1 or size > 65536 for size in core.shape)
                    or core.shape[0] != left_bond or core.shape[1] != dictionary.dimension):
                raise ValueError(f"Core {index} must have shape ({left_bond}, {dictionary.dimension}, right_bond).")
            if not np.all(np.isfinite(core)):
                raise ValueError(f"Core {index} contains non-finite coefficients.")
            checked_cores.append(core)
            left_bond = core.shape[2]
            total_bytes += core.nbytes
        if left_bond != 1:
            raise ValueError("The final coefficient core must have right bond dimension 1.")
        if total_bytes > _MAX_ARCHIVE_BYTES:
            raise ValueError("Model coefficient arrays exceed the 128 MiB limit.")
        if not isinstance(name, str):
            raise ValueError("Model name must be a string.")
        if metadata is not None and not isinstance(metadata, Mapping):
            raise ValueError("Model metadata must be an object.")
        self.basis = list(basis)
        self.cores = checked_cores
        self.domain = checked_domain
        self.metadata = _jsonable(dict(metadata or {}))
        self.name = name

    @property
    def input_dim(self) -> int:
        return len(self.domain)

    @property
    def bond_dimensions(self) -> list[int]:
        return [core.shape[2] for core in self.cores[:-1]]

    def __call__(self, x: Sequence[float] | np.ndarray) -> float:
        if np.iscomplexobj(x):
            raise ValueError("Prediction inputs must be real.")
        values = np.asarray(x, dtype=float)
        if values.ndim == 0 and self.input_dim == 1:
            values = values.reshape(1)
        if values.shape != (self.input_dim,) or not np.all(np.isfinite(values)):
            raise ValueError(f"Prediction requires a vector of {self.input_dim} finite real input values.")
        state = np.ones(1)
        with np.errstate(over="ignore", invalid="ignore"):
            for dictionary, core in zip(self.basis, self.cores):
                matrix = np.einsum("aib,i->ab", core, dictionary(values), optimize=True)
                state = state @ matrix
        prediction = float(state[0])
        if not np.isfinite(prediction):
            raise ValueError("Model contraction produced a non-finite prediction.")
        return prediction

    def predict(self, inputs: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
        if np.iscomplexobj(inputs):
            raise ValueError("Prediction inputs must be real.")
        values = np.asarray(inputs, dtype=float)
        if values.ndim == 1 and self.input_dim == 1:
            values = values[:, None]
        if values.ndim != 2 or values.shape[1] != self.input_dim:
            raise ValueError(f"Batch prediction requires shape (number_of_points, {self.input_dim}).")
        return np.array([self(row) for row in values], dtype=float)

    def coefficient(self, indices: Sequence[int]) -> float:
        if len(indices) != len(self.cores):
            raise ValueError("Supply one zero-based coefficient index per dictionary.")
        state = np.ones(1)
        for index, core in zip(indices, self.cores):
            if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
                raise ValueError("Coefficient indices must be integers.")
            if index < 0 or index >= core.shape[1]:
                raise IndexError("Coefficient index is outside its dictionary.")
            state = state @ core[:, index, :]
        result = float(state[0])
        if not np.isfinite(result):
            raise ValueError("Coefficient contraction produced a non-finite value.")
        return result

    def summary(self) -> dict:
        dictionaries = []
        for dictionary in self.basis:
            dictionaries.append({
                "name": dictionary.name,
                "dimension": dictionary.dimension,
                "input_indices": list(dictionary.input_indices),
                "expressions": list(dictionary.expressions) if dictionary.portable else None,
                "portable": dictionary.portable,
            })
        return _jsonable({
            "name": self.name, "input_dim": self.input_dim,
            "domain": self.domain, "basis": dictionaries,
            "core_shapes": [list(core.shape) for core in self.cores],
            "bond_dimensions": self.bond_dimensions,
            "parameter_count": sum(core.size for core in self.cores),
            "full_coefficient_count": math.prod(dictionary.dimension for dictionary in self.basis),
            "portable": all(dictionary.portable for dictionary in self.basis),
            "metadata": self.metadata,
        })

    def save(self, path) -> Path:
        # Serializing a callable's Python closure would defeat portable and
        # inspectable exports. to_spec raises a useful error for that case.
        manifest = {
            "schema": _MODEL_SCHEMA, "version": _MODEL_VERSION,
            "name": self.name, "input_dim": self.input_dim,
            "domain": self.domain,
            "basis": [dictionary.to_spec() for dictionary in self.basis],
            "cores": [{"shape": list(core.shape), "dtype": "float64"} for core in self.cores],
            "index_convention": "zero-based; core axes: left_bond,basis_entry,right_bond",
            "metadata": self.metadata,
        }
        manifest_data = json.dumps(_jsonable(manifest), ensure_ascii=False, allow_nan=False, indent=2).encode("utf-8")
        if len(manifest_data) > _MAX_MANIFEST_BYTES:
            raise ValueError("Model manifest exceeds the 2 MiB limit.")
        tensor_data = io.BytesIO()
        np.savez_compressed(tensor_data, **{f"core_{k}": core for k, core in enumerate(self.cores)})
        destination = Path(path)
        with zipfile.ZipFile(destination, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("manifest.json", manifest_data)
            archive.writestr("cores.npz", tensor_data.getvalue())
            archive.writestr("predict.py", _standalone_source())
        return destination

    @classmethod
    def load(cls, path) -> "Model":
        manifest, tensor_data = _read_archive(path)
        if not isinstance(manifest.get("basis"), list) or len(manifest["basis"]) != len(manifest["cores"]):
            raise ValueError("Manifest basis and coefficient core counts differ.")
        dictionaries = [Basis.from_spec(spec) for spec in manifest["basis"]]
        with np.load(io.BytesIO(tensor_data), allow_pickle=False) as arrays:
            cores = [arrays[f"core_{k}"] for k in range(len(dictionaries))]
        result = cls(dictionaries, cores, manifest.get("domain"),
                     metadata=manifest.get("metadata"), name=manifest.get("name", "Untitled"))
        input_dim = manifest.get("input_dim")
        if isinstance(input_dim, bool) or not isinstance(input_dim, int) or input_dim != result.input_dim:
            raise ValueError("Manifest input dimension does not match its domain.")
        return result

    def __repr__(self) -> str:
        return f"Model(name={self.name!r}, input_dim={self.input_dim}, sites={len(self.cores)}, bonds={self.bond_dimensions})"

"""Quantum-function inputs and independent encoding-angle oracles.

The coefficient tensor is learned by querying each encoding *occurrence* at
an independent angle. Original preprocessing is used only for prediction.
The built-in simulator and the optional Qiskit adapter use little-endian
qubits: in ``"ZI"``, Z acts on qubit 1 and I acts on qubit 0.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
import re
from typing import Callable, Mapping, Sequence

import numpy as np

from .basis import Basis
from .expressions import Expression


@dataclass
class QuantumProblem:
    auxiliary_oracle: Callable
    auxiliary_basis: list[Basis]
    auxiliary_domain: list[tuple[float, float]]
    basis: list[Basis]
    domain: list[tuple[float, float]]
    target: Callable
    name: str
    metadata: dict
    # Period of each auxiliary dictionary when its frequencies are commensurate.
    auxiliary_periods: list | None = None


def _domain(value):
    try:
        result = [(float(lo), float(hi)) for lo, hi in value]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("domain must contain one [lower, upper] pair per input") from exc
    if not result or any(not (math.isfinite(lo) and math.isfinite(hi) and lo < hi)
                         for lo, hi in result):
        raise ValueError("domain must contain finite, increasing input intervals")
    return result


def _input(value, size, label="input"):
    if np.iscomplexobj(value):
        raise ValueError(f"{label} must be real")
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a finite real vector") from exc
    if array.ndim == 0 and size == 1:
        array = array.reshape(1)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"{label} must contain {size} finite real values")
    return array


def _real(value, label="oracle output"):
    array = np.asarray(value)
    if array.ndim != 0:
        raise ValueError(f"{label} must be a real scalar")
    if np.iscomplexobj(array) and abs(array.imag.item()) > 1e-10:
        raise ValueError(f"{label} must be real")
    result = float(array.real)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def generator_frequencies(generator, *, spectrum=False, atol=1e-10, rtol=1e-9):
    """Return distinct positive eigenvalue gaps of a Hermitian generator.

    Pass ``spectrum=True`` to supply its eigenvalues directly. Near-duplicate
    gaps are merged using the declared numerical tolerances. An identity
    generator has no positive gaps and produces a constant dictionary.
    """
    if not (math.isfinite(atol) and math.isfinite(rtol) and atol >= 0 and rtol >= 0):
        raise ValueError("frequency tolerances must be finite and nonnegative")
    try:
        array = np.asarray(generator, dtype=complex)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("generator must be a Hermitian matrix or a real spectrum") from exc
    if not np.isfinite(array).all():
        raise ValueError("generator must contain finite numbers")
    if spectrum:
        if array.ndim != 1 or not 1 <= array.size <= 4096:
            raise ValueError("a spectrum must be a nonempty vector with at most 4096 entries")
        if np.any(np.abs(array.imag) > atol):
            raise ValueError("the eigenvalues of a Hermitian generator must be real")
        eigenvalues = np.sort(array.real)
    else:
        if (array.ndim != 2 or array.shape[0] != array.shape[1]
                or not 1 <= array.shape[0] <= 1024):
            raise ValueError("generator must be a square Hermitian matrix of size at most 1024")
        if not np.allclose(array, array.conj().T, atol=atol, rtol=rtol):
            raise ValueError("generator must be Hermitian")
        eigenvalues = np.linalg.eigvalsh(array)
    candidates = sorted(float(eigenvalues[j] - eigenvalues[i])
                        for i in range(len(eigenvalues))
                        for j in range(i + 1, len(eigenvalues))
                        if eigenvalues[j] - eigenvalues[i] > atol)
    result = []
    for gap in candidates:
        if not result or not math.isclose(gap, result[-1], abs_tol=atol, rel_tol=rtol):
            result.append(gap)
    return result


def _normalization_denominator(size):
    """Represent sqrt(size) exactly, extracting its integer square factors.

    ``size`` is the known integer D+1 from the dictionary definition. This
    does not attempt to infer fractions or radicals from numerical inputs.
    """
    factor, radicand = 1, size
    candidate = 2
    while candidate * candidate <= radicand:
        square = candidate * candidate
        while radicand % square == 0:
            factor *= candidate
            radicand //= square
        candidate += 1
    if radicand == 1:
        return str(factor)
    if factor == 1:
        return f"sqrt({radicand})"
    return f"({factor}*sqrt({radicand}))"


def _dictionary(frequencies, preprocessing, name):
    denominator = _normalization_denominator(len(frequencies) + 1)
    suffix = "" if denominator == "1" else f"/{denominator}"
    expressions = [f"1{suffix}"]
    for frequency in frequencies:
        argument = f"({frequency!r})*({preprocessing})"
        expressions.extend((f"cos({argument}){suffix}", f"sin({argument}){suffix}"))
    return Basis(expressions, name=name)


def _auxiliary_interval(frequencies):
    if not frequencies:
        return (0.0, 2 * math.pi + 1.0)
    gaps = [frequencies[0]] + [b - a for a, b in zip(frequencies, frequencies[1:])]
    width = 2 * math.pi / min(gaps) + 1.0
    if not math.isfinite(width):
        raise ValueError("generator frequencies require a non-finite auxiliary interval")
    return (0.0, width)


REPRESENTATIONS = ("canonical", "merged", "custom")
_MAX_SITE_DIMENSION = 1024
_MAX_MERGED_FREQUENCIES = 512


def merged_frequencies(frequency_sets, *, atol=1e-10, rtol=1e-9):
    """Positive frequencies of gates that are driven by one shared angle.

    Each generator contributes the signed set {0, +w, -w}; a product of
    gates sharing an angle contains every sum of one member per gate.
    """
    def distinct(values):
        result = []
        for value in sorted(values):
            if not result or not math.isclose(value, result[-1], abs_tol=atol, rel_tol=rtol):
                result.append(value)
        return result

    signed = [0.0]
    for omega in frequency_sets:
        steps = [0.0] + [sign * float(w) for w in omega for sign in (1.0, -1.0)]
        signed = distinct(a + b for a in signed for b in steps)
        if len(signed) > 2 * _MAX_MERGED_FREQUENCIES + 1:
            raise ValueError(
                f"Merging these encoding gates needs more than {_MAX_MERGED_FREQUENCIES} frequencies "
                "in one dictionary. Use the canonical representation instead.")
    return [value for value in signed if value > atol]


def _merge_groups(transforms, domain, *, samples=16, atol=1e-10, rtol=1e-9):
    """Group encoding occurrences whose preprocessing functions agree.

    Equality is tested numerically at fixed pseudo-random inputs of the
    domain, so equivalent spellings such as ``2*x0`` and ``x0+x0`` merge.
    A preprocessing that cannot be evaluated there stays on its own site.
    """
    bounds = np.asarray(domain, dtype=float)
    rng = np.random.default_rng(20260519)
    points = np.vstack([bounds.mean(axis=1),
                        rng.uniform(bounds[:, 0], bounds[:, 1], size=(samples, len(bounds)))])
    signatures = []
    for phi in transforms:
        try:
            values = np.array([_real(phi(x), "preprocessing output") for x in points])
        except (ValueError, ArithmeticError, TypeError):
            values = None
        signatures.append(values)
    groups, representatives = [], []
    for index, values in enumerate(signatures):
        match = None
        if values is not None:
            for position, other in enumerate(representatives):
                if other is not None and np.allclose(values, other, atol=atol, rtol=rtol):
                    match = position
                    break
        if match is None:
            groups.append([index])
            representatives.append(values)
        else:
            groups[match].append(index)
    return groups


def _period(frequencies, *, tolerance=1e-9, max_multiple=4096):
    """Common period of commensurate frequencies, or None."""
    if not frequencies:
        return None
    scale = max(frequencies)
    step = frequencies[0]
    for value in frequencies[1:]:
        a, b = max(step, value), min(step, value)
        while b > tolerance * scale:
            remainder = math.fmod(a, b)
            if remainder < tolerance * scale or b - remainder < tolerance * scale:
                remainder = 0.0
            a, b = b, remainder
        step = a
    multiples = [value / step for value in frequencies]
    if max(multiples) > max_multiple or any(abs(m - round(m)) > 1e-6 for m in multiples):
        return None
    return 2 * math.pi / step


def _site_name(members):
    if len(members) == 1:
        return f"Encoding {members[0] + 1}"
    shown = ", ".join(str(m + 1) for m in members[:6])
    return f"Encodings {shown}" + (f", … ({len(members)} gates)" if len(members) > 6 else "")


def from_generators(
    generators: Sequence,
    preprocessing: Sequence[str | Callable],
    auxiliary_oracle: Callable,
    domain: Sequence[Sequence[float]],
    *,
    spectra: bool = False,
    name: str = "Quantum function",
    preprocessing_expressions: Sequence[str | None] | None = None,
    frequency_atol: float = 1e-10,
    frequency_rtol: float = 1e-9,
    representation: str = "canonical",
    sites: Sequence[Sequence[int]] | None = None,
) -> QuantumProblem:
    """Describe hidden quantum hardware using generators and an angle oracle.

    ``auxiliary_oracle(a)`` must independently control every listed encoding
    angle. Generators and preprocessing alone do not specify coefficients.
    Callable preprocessing is supported locally; provide equivalent
    ``preprocessing_expressions`` for portable model export.

    The same function has several tensor-network representations.
    ``representation="canonical"`` uses one dictionary per encoding
    occurrence. ``"merged"`` gives occurrences that share a preprocessing
    function one dictionary and one auxiliary angle; its frequencies are
    the sums of the merged generators' frequencies. ``"custom"`` takes
    ``sites``, an ordered partition of the encoding occurrences: each inner
    list is one dictionary and their order is the order along the chain.
    Occurrences of one site that share a preprocessing function are merged;
    different preprocessing functions on one site form a tensor-product
    dictionary, each keeping its own training angle.
    """
    if representation not in REPRESENTATIONS:
        raise ValueError("representation must be 'canonical', 'merged' or 'custom'")
    if (sites is not None) != (representation == "custom"):
        raise ValueError("provide sites exactly when the representation is 'custom'")
    domain = _domain(domain)
    generators = list(generators)
    preprocessing = list(preprocessing)
    if not generators or len(generators) != len(preprocessing):
        raise ValueError("provide one generator and one preprocessing function per encoding occurrence")
    if not callable(auxiliary_oracle):
        raise ValueError("an auxiliary-angle oracle is required to learn coefficients")
    if preprocessing_expressions is None:
        preprocessing_expressions = [None] * len(preprocessing)
    if len(preprocessing_expressions) != len(preprocessing):
        raise ValueError("preprocessing_expressions must match preprocessing")
    frequencies = [generator_frequencies(g, spectrum=spectra,
                                          atol=frequency_atol, rtol=frequency_rtol)
                   for g in generators]
    transforms, texts = [], []
    for index, (phi, portable) in enumerate(zip(preprocessing, preprocessing_expressions)):
        if isinstance(phi, str):
            expression = Expression(phi)
            text = phi
        elif callable(phi):
            expression = Expression(portable) if portable is not None else None
            text = portable
        else:
            raise ValueError("preprocessing must contain expression strings or callable functions")
        if expression is not None and any(i >= len(domain) for i in expression.input_indices):
            raise ValueError(f"preprocessing at encoding {index} refers to an input outside domain")
        transforms.append(expression if isinstance(phi, str) else phi)
        texts.append(text)
    equal = _merge_groups(transforms, domain, atol=frequency_atol, rtol=frequency_rtol)
    family = {member: position for position, members in enumerate(equal) for member in members}
    if representation == "canonical":
        groups = [[index] for index in range(len(generators))]
    elif representation == "merged":
        groups = equal
    else:
        groups = _custom_groups(sites, len(generators))

    def class_dictionary(omega, phi, text, variable=None):
        """Expressions (when portable) and evaluator of one merged class."""
        expressions = None
        if variable is not None:
            expressions = list(_dictionary(omega, variable, "").expressions)
        elif text is not None:
            expressions = list(_dictionary(omega, text, "").expressions)
        if isinstance(phi, str) or variable is not None:
            return expressions, None
        def evaluate(x, phi=phi, omega=tuple(omega)):
            angle = _real(phi(x), "preprocessing output")
            vector = [1.0]
            for frequency in omega:
                vector.extend((math.cos(frequency * angle), math.sin(frequency * angle)))
            return np.asarray(vector) / math.sqrt(len(omega) + 1)
        return expressions, evaluate

    def product(parts):
        return [parts[0][i] for i in range(len(parts[0]))] if len(parts) == 1 else [
            "*".join(f"({factor})" for factor in factors) for factors in itertools.product(*parts)]

    auxiliary_of = np.empty(len(generators), dtype=int)
    site_of = np.empty(len(generators), dtype=int)
    basis, auxiliary_basis, auxiliary_domain, periods, site_metadata = [], [], [], [], []
    for position, members in enumerate(groups):
        site_of[members] = position
        classes = {}
        for member in members:
            classes.setdefault(family[member], []).append(member)
        label = _site_name(members)
        described, portable, evaluators, auxiliary_parts, dimension = [], [], [], [], 1
        for merged in classes.values():
            omega = (frequencies[merged[0]] if len(merged) == 1 else
                     merged_frequencies([frequencies[m] for m in merged],
                                        atol=frequency_atol, rtol=frequency_rtol))
            coordinate = len(auxiliary_domain)
            auxiliary_of[merged] = coordinate
            auxiliary_domain.append(_auxiliary_interval(omega))
            expressions, evaluate = class_dictionary(omega, preprocessing[merged[0]], texts[merged[0]])
            portable.append(expressions)
            evaluators.append((evaluate, expressions))
            auxiliary_parts.append(class_dictionary(omega, None, None, variable=f"x{coordinate}")[0])
            dimension *= 2 * len(omega) + 1
            described.append({"encoding_indices": list(merged), "preprocessing": texts[merged[0]],
                              "frequencies": omega, "dimension": 2 * len(omega) + 1,
                              "auxiliary_index": coordinate})
        if dimension > _MAX_SITE_DIMENSION:
            raise ValueError(
                f"Joining {label.lower()} needs a dictionary of {dimension} functions; "
                f"the limit is {_MAX_SITE_DIMENSION}. Split this dictionary.")
        expressions = product(portable) if all(part is not None for part in portable) else None
        if all(evaluate is None for evaluate, _ in evaluators):
            prediction_basis = Basis(expressions, name=label)
        else:
            parts = [(evaluate, None if evaluate is not None else Basis(text))
                     for evaluate, text in evaluators]
            def evaluate_site(x, parts=tuple(parts)):
                vector = np.ones(1)
                for evaluate, fixed in parts:
                    vector = np.kron(vector, evaluate(x) if evaluate is not None else fixed(x))
                return vector
            dependencies = (Basis(expressions).input_indices if expressions is not None
                            else tuple(range(len(domain))))
            prediction_basis = Basis.from_callable(
                evaluate_site, input_indices=dependencies, dimension=dimension,
                name=label, expressions=expressions)
        basis.append(prediction_basis)
        auxiliary_basis.append(Basis(product(auxiliary_parts),
                                     name=f"Auxiliary angle {position + 1}" if len(described) == 1
                                     else f"Auxiliary angles of site {position + 1}"))
        single = described[0] if len(described) == 1 else None
        periods.append(_period(single["frequencies"]) if single else None)
        site_metadata.append({"site_index": position, "encoding_indices": list(members),
                              "preprocessing": single["preprocessing"] if single else None,
                              "frequencies": single["frequencies"] if single else [],
                              "dimension": dimension, "classes": described})
    metadata = [{"encoding_index": index, "site_index": int(site_of[index]),
                 "auxiliary_index": int(auxiliary_of[index]),
                 "preprocessing": texts[index], "frequencies": frequencies[index],
                 "dimension": 2 * len(frequencies[index]) + 1}
                for index in range(len(generators))]

    def occurrence_query(angles):
        return _real(auxiliary_oracle(_input(angles, len(generators), "auxiliary angles")))

    def query(a):
        # One auxiliary angle drives every encoding occurrence merged to it.
        return occurrence_query(_input(a, len(auxiliary_domain), "auxiliary angles")[auxiliary_of])

    def target(x):
        x = _input(x, len(domain))
        return occurrence_query([_real(phi(x), "preprocessing output") for phi in transforms])

    coordinates = {"canonical": "independent encoding occurrences",
                   "merged": "one auxiliary angle per preprocessing function",
                   "custom": "one auxiliary angle per preprocessing function on each site"}
    return QuantumProblem(
        auxiliary_oracle=query, auxiliary_basis=auxiliary_basis,
        auxiliary_domain=auxiliary_domain,
        basis=basis, domain=domain, target=target, name=name,
        auxiliary_periods=periods,
        metadata={"kind": "quantum", "backend": "external-angle-oracle", "encodings": metadata,
                  "representation": representation, "sites": site_metadata,
                  "frequency_atol": frequency_atol, "frequency_rtol": frequency_rtol,
                  "dictionary_normalization": "1/sqrt(number_of_positive_frequencies+1)",
                  "training_coordinates": coordinates[representation]},
    )


def _custom_groups(sites, count):
    """Validate an ordered partition of the encoding occurrences."""
    try:
        groups = [[int(index) for index in members] for members in sites]
        exact = all(not isinstance(index, bool) and index == int(index)
                    for members in sites for index in members)
    except (TypeError, ValueError) as exc:
        raise ValueError("sites must be a list of lists of encoding indices") from exc
    listed = sorted(index for members in groups for index in members)
    if not exact or not groups or any(not members for members in groups) or listed != list(range(count)):
        raise ValueError(
            f"sites must place each of the {count} encoding occurrences on exactly one non-empty site")
    return groups


_I = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.diag([1, -1]).astype(complex)
_FIXED = {"i": _I, "id": _I, "x": _X, "y": _Y, "z": _Z,
          "h": np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2),
          "s": np.diag([1, 1j]), "sdg": np.diag([1, -1j]),
          "t": np.diag([1, np.exp(1j * math.pi / 4)]),
          "tdg": np.diag([1, np.exp(-1j * math.pi / 4)]),
          "cx": np.array([[1, 0, 0, 0], [0, 1, 0, 0],
                          [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex),
          "cz": np.diag([1, 1, 1, -1]).astype(complex),
          "swap": np.array([[1, 0, 0, 0], [0, 0, 1, 0],
                            [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)}
_ROTATIONS = {"rx": _X, "ry": _Y, "rz": _Z,
              "rxx": np.kron(_X, _X), "ryy": np.kron(_Y, _Y), "rzz": np.kron(_Z, _Z)}
_PHASE = {"p", "phase"}


def _gate_matrix(gate, angle=None):
    if gate in _ROTATIONS:
        pauli = _ROTATIONS[gate]
        return math.cos(angle / 2) * np.eye(len(pauli)) - 1j * math.sin(angle / 2) * pauli
    if gate in _PHASE:
        return np.diag([1.0, np.exp(1j * angle)])
    return _FIXED[gate]


def _apply_operator(state, matrix, qubits, n_qubits):
    axes = [n_qubits - 1 - q for q in qubits]
    moved = np.moveaxis(state.reshape((2,) * n_qubits), axes, range(len(axes)))
    updated = (matrix @ moved.reshape(2 ** len(axes), -1)).reshape(moved.shape)
    return np.moveaxis(updated, range(len(axes)), axes).reshape(-1)


def _pauli_label(observable, n_qubits):
    if not isinstance(observable, str):
        raise ValueError("observable must be a Pauli string such as 'ZI' (leftmost is highest qubit)")
    observable = observable.upper().strip()
    if len(observable) != n_qubits or set(observable) - set("IXYZ"):
        raise ValueError(f"observable must contain {n_qubits} letters from I, X, Y, Z")
    return observable


def _pauli_expectation(state, observable, n_qubits):
    transformed = state
    for position, letter in enumerate(observable):
        if letter != "I":
            transformed = _apply_operator(transformed, {"X": _X, "Y": _Y, "Z": _Z}[letter],
                                          [n_qubits - 1 - position], n_qubits)
    return _real(np.vdot(state, transformed), "expectation value")


def _initial_state(value, n_qubits):
    if value is None:
        value = "0" * n_qubits
    if isinstance(value, str):
        if len(value) != n_qubits or set(value) - {"0", "1"}:
            raise ValueError("initial_state bitstring must contain one 0 or 1 per qubit")
        state = np.zeros(2 ** n_qubits, dtype=complex)
        state[int(value, 2)] = 1
        return state
    state = np.asarray(value, dtype=complex)
    if (state.shape != (2 ** n_qubits,) or not np.isfinite(state).all()
            or not np.isclose(np.vdot(state, state).real, 1.0, atol=1e-10, rtol=1e-10)):
        raise ValueError("initial_state must be a normalized finite statevector of length 2**n_qubits")
    return state.copy()


def from_circuit(spec: Mapping, *, name: str | None = None,
                 representation: str = "canonical",
                 sites: Sequence[Sequence[int]] | None = None) -> QuantumProblem:
    """Build an exact, small statevector oracle without installing Qiskit.

    A gate is e.g. ``{'gate': 'ry', 'qubits': [0], 'angle': 'x0**2'}``.
    Standard rotations use exp(-i angle P/2). Each dynamic gate occurrence
    receives its own auxiliary angle, including gates sharing the same x.
    """
    if not isinstance(spec, Mapping):
        raise ValueError("circuit must be a specification object")
    n_qubits = spec.get("n_qubits")
    if isinstance(n_qubits, bool) or not isinstance(n_qubits, int) or not 1 <= n_qubits <= 10:
        raise ValueError("the exact demo simulator supports 1 to 10 qubits")
    domain = _domain(spec.get("domain", []))
    observable = _pauli_label(spec.get("observable", "Z" * n_qubits), n_qubits)
    state0 = _initial_state(spec.get("initial_state"), n_qubits)
    descriptors = spec.get("gates")
    if not isinstance(descriptors, (list, tuple)) or len(descriptors) > 1000:
        raise ValueError("gates must be a list containing at most 1000 gates")
    gates, preprocessing, encodings, spectra = [], [], [], []
    for index, descriptor in enumerate(descriptors):
        if not isinstance(descriptor, Mapping):
            raise ValueError(f"gate {index} must be an object")
        gate = str(descriptor.get("gate", "")).lower()
        if gate == "cnot":
            gate = "cx"
        if gate not in _FIXED and gate not in _ROTATIONS and gate not in _PHASE:
            raise ValueError(f"unsupported gate '{gate}' at position {index}; provide a supported decomposition")
        qubits = descriptor.get("qubits", [])
        expected = 2 if gate in {"cx", "cz", "swap", "rxx", "ryy", "rzz"} else 1
        if (not isinstance(qubits, (list, tuple)) or len(qubits) != expected
                or any(isinstance(q, bool) or not isinstance(q, int) or not 0 <= q < n_qubits for q in qubits)
                or len(set(qubits)) != expected):
            raise ValueError(f"gate {index} requires {expected} distinct, valid zero-based qubit indices")
        angle, auxiliary_index = None, None
        if gate in _ROTATIONS or gate in _PHASE:
            if "angle" not in descriptor:
                raise ValueError(f"gate {index} requires an angle")
            raw_angle = descriptor["angle"]
            if not isinstance(raw_angle, (str, float, int)) or isinstance(raw_angle, bool):
                raise ValueError(f"gate {index} angle must be a number or expression string")
            angle = Expression(str(raw_angle))
            if any(i >= len(domain) for i in angle.input_indices):
                raise ValueError(f"gate {index} angle refers to an input outside domain")
            if angle.input_indices:
                auxiliary_index = len(preprocessing)
                preprocessing.append(str(raw_angle))
                spectra.append([-1.0, 0.0] if gate in _PHASE else [-0.5, 0.5])
                encodings.append({"gate_position": index, "gate": gate, "qubits": list(qubits),
                                  "generator": "-diag(0,1)" if gate in _PHASE else gate[1:].upper() + "/2"})
            else:
                angle = _real(angle(np.zeros(len(domain))), "constant gate angle")
        elif "angle" in descriptor:
            raise ValueError(f"fixed gate '{gate}' does not take an angle")
        gates.append((gate, tuple(qubits), angle, auxiliary_index))
    dynamic_count = len(preprocessing)
    def simulate(x=None, auxiliary=None):
        state = state0.copy()
        for gate, qubits, angle, auxiliary_index in gates:
            if auxiliary_index is not None:
                theta = float(auxiliary[auxiliary_index]) if auxiliary is not None else _real(angle(x))
            else:
                theta = angle
            state = _apply_operator(state, _gate_matrix(gate, theta), qubits, n_qubits)
        return _pauli_expectation(state, observable, n_qubits)
    if dynamic_count == 0:
        preprocessing = ["0"]
        spectra = [[0.0]]
    def auxiliary(a):
        return simulate(auxiliary=a)
    problem = from_generators(spectra, preprocessing, auxiliary, domain, spectra=True,
                              name=name or str(spec.get("name", "Quantum circuit")),
                              representation=representation, sites=sites)
    # Compare the original circuit directly in original input coordinates.
    problem.target = lambda x: simulate(x=_input(x, len(domain)))
    for item, detail in zip(problem.metadata["encodings"], encodings):
        item.update(detail)
    if encodings:
        for site in problem.metadata["sites"]:
            site["gate_positions"] = [encodings[i]["gate_position"] for i in site["encoding_indices"]]
    problem.metadata.update({"backend": "numpy-statevector", "n_qubits": n_qubits,
                             "observable": observable, "encoding_occurrences": dynamic_count,
                             "qubit_order": "little-endian; leftmost Pauli acts on highest-index qubit"})
    return problem


def from_qiskit(
    circuit,
    input_parameters: Sequence,
    domain: Sequence[Sequence[float]],
    observable: str,
    *,
    fixed_parameters: Mapping | None = None,
    name: str = "Qiskit quantum function",
    representation: str = "canonical",
    sites: Sequence[Sequence[int]] | None = None,
) -> QuantumProblem:
    """Adapt a symbolic Qiskit circuit to independent encoding occurrences.

    This optional exact backend supports at most 10 qubits, a zero initial
    state, unitary circuits, and RX/RY/RZ/RXX/RYY/RZZ/P input-dependent gates.
    Arbitrary fixed unitary gates may remain in the circuit. Mid-circuit
    measurements, resets, and dynamic control flow require another oracle.
    """
    try:
        from qiskit.circuit import Parameter
        from qiskit.quantum_info import Pauli, Statevector
    except ImportError as exc:
        raise ImportError("Qiskit adapter requires the optional dependency: pip install 'ftci-studio[quantum]'") from exc
    domain = _domain(domain)
    input_parameters = list(input_parameters)
    if len(input_parameters) != len(domain) or len(set(input_parameters)) != len(input_parameters):
        raise ValueError("provide one distinct Qiskit input Parameter per domain interval")
    if not all(isinstance(parameter, Parameter) for parameter in input_parameters):
        raise ValueError("input_parameters must be Qiskit Parameter objects")
    if not 1 <= circuit.num_qubits <= 10:
        raise ValueError("the exact Qiskit adapter supports 1 to 10 qubits")
    observable = _pauli_label(observable, circuit.num_qubits)
    input_set = set(input_parameters)
    fixed_parameters = dict(fixed_parameters or {})
    if any(key in input_set or (isinstance(key, str) and key in {p.name for p in input_parameters})
           for key in fixed_parameters):
        raise ValueError("an input parameter cannot also be fixed")
    bound = circuit.assign_parameters(fixed_parameters, inplace=False)
    # Final measurements are irrelevant when evaluating a chosen observable.
    bound = bound.remove_final_measurements(inplace=False)
    unknown = set(bound.parameters) - input_set
    if unknown:
        raise ValueError("bind all non-input parameters first: " + ", ".join(sorted(str(p) for p in unknown)))
    lifted = bound.copy_empty_like()
    lifted.global_phase = 0.0  # Global phase does not affect expectation values.
    auxiliary_parameters, preprocessing, spectra, encodings = [], [], [], []
    temp_prefix = "ftci_input_"
    while any(temp_prefix in p.name for p in input_parameters):
        temp_prefix = "ftci_" + temp_prefix
    temp_parameters = {p: Parameter(f"{temp_prefix}{i}") for i, p in enumerate(input_parameters)}
    supported = set(_ROTATIONS) | {"p"}
    for position, instruction in enumerate(bound.data):
        operation = instruction.operation
        if operation.name in {"measure", "reset"} or getattr(operation, "blocks", ()):
            raise ValueError("the exact Qiskit adapter needs a unitary circuit without mid-circuit measurement or control flow")
        parameters = set().union(*(set(getattr(value, "parameters", ())) for value in operation.params))
        if parameters:
            if operation.name not in supported or len(operation.params) != 1:
                raise ValueError(f"input-dependent gate '{operation.name}' at position {position} is unsupported; decompose it or supply generators and an auxiliary-angle oracle")
            expression = operation.params[0]
            renamed = expression.subs({p: q for p, q in temp_parameters.items() if p in parameters})
            text = str(renamed)
            for i in range(len(input_parameters)):
                text = re.sub(rf"\b{temp_prefix}{i}\b", f"x{i}", text)
            text = re.sub(r"\bAbs\b", "abs", text)
            # Validate expression portability before any expensive simulation.
            Expression(text)
            auxiliary_parameter = Parameter(f"ftci_aux_{len(auxiliary_parameters)}")
            auxiliary_parameters.append(auxiliary_parameter)
            preprocessing.append(text)
            spectra.append([-1.0, 0.0] if operation.name == "p" else [-0.5, 0.5])
            copied = operation.copy()
            copied.params = [auxiliary_parameter]
            lifted.append(copied, instruction.qubits, instruction.clbits)
            encodings.append({"gate_position": position, "gate": operation.name,
                              "qubits": [bound.find_bit(q).index for q in instruction.qubits],
                              "generator": "-diag(0,1)" if operation.name == "p" else operation.name[1:].upper() + "/2"})
        else:
            lifted.append(operation.copy(), instruction.qubits, instruction.clbits)
    operator = Pauli(observable)
    dynamic_count = len(auxiliary_parameters)
    def auxiliary(a):
        values = {parameter: float(value) for parameter, value in zip(auxiliary_parameters, a)}
        instance = lifted.assign_parameters(values, inplace=False)
        return _real(Statevector.from_instruction(instance).expectation_value(operator))
    if not preprocessing:
        preprocessing, spectra = ["0"], [[0.0]]
    problem = from_generators(spectra, preprocessing, auxiliary, domain, spectra=True, name=name,
                              representation=representation, sites=sites)
    def target(x):
        x = _input(x, len(domain))
        instance = bound.assign_parameters(dict(zip(input_parameters, x)), inplace=False, strict=False)
        return _real(Statevector.from_instruction(instance).expectation_value(operator))
    problem.target = target
    # Fail early for fixed instructions which Statevector cannot execute.
    target([(lo + hi) / 2 for lo, hi in domain])
    for item, detail in zip(problem.metadata["encodings"], encodings):
        item.update(detail)
    if encodings:
        for site in problem.metadata["sites"]:
            site["gate_positions"] = [encodings[i]["gate_position"] for i in site["encoding_indices"]]
    problem.metadata.update({"backend": "qiskit-statevector", "n_qubits": circuit.num_qubits,
                             "observable": observable, "encoding_occurrences": dynamic_count,
                             "input_parameters": [p.name for p in input_parameters],
                             "qubit_order": "little-endian; leftmost Pauli acts on highest-index qubit"})
    return problem


__all__ = ["QuantumProblem", "REPRESENTATIONS", "generator_frequencies", "merged_frequencies",
           "from_generators", "from_circuit", "from_qiskit"]

"""FTCI orchestration: active TCI queries, local inverses, and holdout checks.

The tensor cross interpolation is performed by TensorCrossInterpolation.jl.
NumPy applies each local inverse and evaluates the portable coefficient MPS.
No full product tensor is constructed.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
import math
import os
import queue
import shutil
import subprocess
import threading
import time
from typing import Callable

import numpy as np

from .basis import Basis
from .model import Model


class FitCancelled(RuntimeError):
    pass


class QueryBudgetExceeded(RuntimeError):
    pass


@dataclass
class FunctionProblem:
    target: Callable
    basis: list[Basis]
    domain: list[tuple[float, float]]
    name: str = "Function model"
    metadata: dict = field(default_factory=dict)


def julia_project() -> Path:
    configured = os.environ.get("FTCI_JULIA_PROJECT")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parent / "julia"


def engine_info() -> dict:
    executable = os.environ.get("FTCI_JULIA", "julia")
    return {
        "name": "TensorCrossInterpolation.jl",
        "engine_available": shutil.which(executable) is not None,
        "julia_project": str(julia_project()),
    }


def _bounds(domain, *, max_dimensions=32):
    a = np.asarray(domain, dtype=float)
    if a.ndim != 2 or a.shape[1] != 2 or not 1 <= len(a) <= max_dimensions:
        raise ValueError(f"Domain must contain 1–{max_dimensions} [lower, upper] pairs.")
    if not np.isfinite(a).all() or np.any(a[:, 1] <= a[:, 0]):
        raise ValueError("Every input needs finite bounds with lower < upper.")
    return a


NODE_RULES = ("auto", "equispaced", "random")


def _site_design(basis, domain, midpoint, points, rule, rng, period, pinv_cutoff, guard):
    """Choose auxiliary inputs for one dictionary and report their conditioning.

    ``random`` draws uniform points on the site's input block. ``equispaced``
    uses evenly spaced points: over one period for a trigonometric dictionary
    with commensurate frequencies, where the design rows become orthogonal,
    and at interval midpoints otherwise. ``auto`` keeps whichever of the two
    amplifies errors least.
    """
    q, indices = basis.dimension, list(basis.input_indices)

    def measure(local):
        matrix = np.empty((q, points))
        for j in range(points):
            guard()
            x = midpoint.copy()
            x[indices] = local[j]
            matrix[:, j] = basis(x)
        singular = np.linalg.svd(matrix, compute_uv=False)
        rank = int(np.sum(singular > pinv_cutoff * singular[0])) if singular[0] else 0
        return matrix, singular, rank

    candidates, rank = {}, 0
    if rule in ("auto", "random") or len(indices) != 1:
        for _ in range(8):
            local = rng.uniform(domain[indices, 0], domain[indices, 1], size=(points, len(indices)))
            matrix, singular, rank = measure(local)
            if rank == q:
                candidates["random"] = (local, matrix, singular)
                break
    if rule in ("auto", "equispaced") and len(indices) == 1:
        lower, upper = domain[indices[0]]
        steps = np.arange(points)
        local = (steps * period / points if period else
                 lower + (steps + 0.5) * (upper - lower) / points).reshape(points, 1)
        matrix, singular, equispaced_rank = measure(local)
        if equispaced_rank == q:
            candidates["equispaced"] = (local, matrix, singular)
        rank = max(rank, equispaced_rank)
    if not candidates:
        hint = (" Evenly spaced inputs repeat values of this dictionary; use random or automatic nodes."
                if rule == "equispaced" and len(indices) == 1 else
                " Remove redundant functions or widen the domain.")
        raise ValueError(
            f"Dictionary {basis.name or '(unnamed)'} has no full-row-rank sampling design on the given "
            f"domain (rank {rank}, dimension {q})." + hint)
    chosen = min(candidates, key=lambda name: 1 / candidates[name][2][-1])
    local, matrix, singular = candidates[chosen]
    return local, matrix, {
        "dimension": q, "auxiliary_points": points, "rank": q, "node_rule": chosen,
        "smallest_singular_value": float(singular[-1]),
        "condition_number": float(singular[0] / singular[-1]),
        "inverse_norm": float(1 / singular[-1]), "input_indices": indices}


def _build_designs(bases, domain, periods, *, oversampling, node_rule, rng, pinv_cutoff, guard):
    midpoint = domain.mean(axis=1)
    nodes, matrices, diagnostics = [], [], []
    for k, basis in enumerate(bases):
        guard()
        q = basis.dimension
        points = max(q, math.ceil(oversampling * q))
        if q < 1 or points > 2048:
            raise ValueError("Each dictionary must have 1–1024 functions and at most 2048 auxiliary points.")
        if not basis.name:
            basis = _named(basis, f"{k + 1}")
        local, matrix, report = _site_design(basis, domain, midpoint, points, node_rule, rng,
                                             periods[k], pinv_cutoff, guard)
        nodes.append(local)
        matrices.append(matrix)
        diagnostics.append({"site": k, **report})
    return nodes, matrices, diagnostics


class _named:
    """A dictionary shown under a fallback name in error messages."""

    def __init__(self, basis, name):
        self._basis, self.name = basis, name
        self.dimension, self.input_indices = basis.dimension, basis.input_indices

    def __call__(self, x):
        return self._basis(x)


def _training_frame(problem):
    is_quantum = hasattr(problem, "auxiliary_oracle")
    max_sites = 64 if is_quantum else 32
    original_domain = _bounds(problem.domain)
    domain = _bounds(problem.auxiliary_domain, max_dimensions=max_sites) if is_quantum else original_domain
    bases = list(problem.auxiliary_basis if is_quantum else problem.basis)
    if not 1 <= len(bases) <= max_sites:
        raise ValueError(f"Provide 1–{max_sites} basis dictionaries.")
    owners = {}
    for k, basis in enumerate(bases):
        for index in basis.input_indices:
            if index >= len(domain):
                raise ValueError(f"Dictionary {k + 1} refers to x{index}, outside the input domain.")
            if index in owners:
                raise ValueError(
                    f"Dictionaries {owners[index] + 1} and {k + 1} both depend on x{index}. "
                    "Direct FTCI needs disjoint input blocks. Merge these dictionaries, "
                    "or use a quantum adapter that can independently query encoding angles. "
                    "General overlapping-basis MPS regression is not implemented in this version."
                )
            owners[index] = k
    periods = list(getattr(problem, "auxiliary_periods", None) or [None] * len(bases))
    return is_quantum, original_domain, domain, bases, periods


def _check_design_options(oversampling, node_rule, pinv_cutoff):
    if not np.isfinite(oversampling) or not 1 <= oversampling <= 16:
        raise ValueError("Oversampling must be between 1 and 16.")
    if node_rule not in NODE_RULES:
        raise ValueError("Nodes must be automatic, equispaced or random.")
    if not 0 < pinv_cutoff < 1:
        raise ValueError("Invalid pseudoinverse cutoff.")


def _amplification(diagnostics):
    return float(sum(math.log10(item["inverse_norm"]) for item in diagnostics))


def design_report(problem, *, oversampling=2, node_rule="auto", seed=0, pinv_cutoff=1e-12) -> dict:
    """Conditioning of the local sampling designs, without querying the function.

    The designs depend only on the dictionaries and the auxiliary inputs.
    ``log10_inversion_amplification`` is log10 of the product of the inverse
    norms, the factor bounding how much a sampled-tensor error can grow when
    it is transformed into a coefficient error.
    """
    oversampling = float(oversampling)
    _check_design_options(oversampling, node_rule, pinv_cutoff)
    _, _, domain, bases, periods = _training_frame(problem)
    _, _, diagnostics = _build_designs(
        bases, domain, periods, oversampling=oversampling, node_rule=node_rule,
        rng=np.random.default_rng(int(seed)), pinv_cutoff=pinv_cutoff, guard=lambda: None)
    return {"designs": diagnostics, "node_rule": node_rule, "oversampling": oversampling,
            "log10_inversion_amplification": _amplification(diagnostics)}


def _run_tci(dims, oracle, initialpivot, *, tolerance, max_bond, seed,
             cancel, timeout_seconds, notify):
    executable = os.environ.get("FTCI_JULIA", "julia")
    if not shutil.which(executable):
        raise RuntimeError("Julia is required for learning. Install Julia and run the README setup; saved models need only NumPy.")
    command = [executable, f"--project={julia_project()}", "--startup-file=no",
               str(Path(__file__).parent / "julia" / "worker.jl")]
    proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, bufsize=1)
    output = queue.Queue()
    stderr = deque(maxlen=30)

    def read_output():
        for line in proc.stdout:
            output.put(line.rstrip("\n"))
        output.put(None)

    def read_errors():
        for line in proc.stderr:
            stderr.append(line.rstrip())

    readers = [threading.Thread(target=read_output, daemon=True),
               threading.Thread(target=read_errors, daemon=True)]
    for reader in readers:
        reader.start()
    started = time.monotonic()
    cores, errors, ranks = [], [], []
    ready, done = False, False
    try:
        while True:
            if cancel and cancel():
                raise FitCancelled("Learning was cancelled.")
            if time.monotonic() - started > timeout_seconds:
                raise TimeoutError(f"Learning exceeded the {timeout_seconds:g} second time limit.")
            try:
                line = output.get(timeout=0.2)
            except queue.Empty:
                continue
            if line is None:
                break
            if line == "READY":
                ready = True
                header = [",".join(map(str, dims)), str(tolerance), str(max_bond),
                          str(seed), ",".join(str(i + 1) for i in initialpivot)]
                proc.stdin.write("\n".join(header) + "\n")
                proc.stdin.flush()
                notify("learning", "Tensor cross interpolation is querying the function.")
            elif line.startswith("Q\t"):
                indices = tuple(int(i) - 1 for i in line[2:].split(","))
                value = oracle(indices)
                proc.stdin.write(repr(value) + "\n")
                proc.stdin.flush()
            elif line.startswith("CORE\t"):
                _, shape, values = line.split("\t", 2)
                shape = tuple(int(i) for i in shape.split(","))
                if len(shape) != 3 or math.prod(shape) > 20_000_000:
                    raise RuntimeError("The TCI engine returned an invalid core size.")
                core = np.fromstring(values, sep=",")
                cores.append(core.reshape(shape, order="F"))
            elif line.startswith("ERRORS\t"):
                errors = [float(v) for v in line[7:].split(",") if v]
            elif line.startswith("RANKS\t"):
                ranks = [int(v) for v in line[6:].split(",") if v]
            elif line.startswith("ERROR\t"):
                raise RuntimeError("Julia TCI: " + line[6:])
            elif line == "DONE":
                done = True
                break
        proc.wait(timeout=10)
        if not ready or not done or proc.returncode or len(cores) != len(dims):
            details = "\n".join(stderr)[-5000:]
            raise RuntimeError("Julia TCI did not finish. Check the Julia project setup.\n" + details)
        return cores, errors, ranks
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=3)
        for stream in (proc.stdin, proc.stdout, proc.stderr):
            stream.close()
        for reader in readers:
            reader.join(timeout=1)


def fit(problem, *, max_bond=32, tolerance=1e-8, oversampling=2,
        query_budget=20_000, validation_samples=100, seed=0,
        preview_samples=0, progress=None, cancel=None, timeout_seconds=600,
        pinv_cutoff=1e-12, node_rule="auto", shots=None) -> Model:
    """Learn a coefficient MPS from a FunctionProblem or QuantumProblem.

    ``node_rule`` chooses the auxiliary inputs of each dictionary: ``random``,
    ``equispaced``, or ``auto`` for the better conditioned of the two.
    ``shots`` simulates a finite-shot oracle: every training query returns
    its value plus Gaussian noise of variance 1/shots. Validation always
    compares against the exact target.

    ``query_budget`` covers actual oracle evaluations during training, holdout
    validation, and optional chart generation. Cache hits do not consume it.
    Exceeding a budget fails explicitly, rather than labelling an unvalidated
    approximation successful. ``tolerance`` is a TCI pivot criterion, not an
    asserted continuous-function error.
    """
    max_bond, query_budget, validation_samples = map(int, (max_bond, query_budget, validation_samples))
    preview_samples, seed = int(preview_samples), int(seed)
    tolerance, oversampling = float(tolerance), float(oversampling)
    if not 1 <= max_bond <= 512 or not 0 < tolerance < 1:
        raise ValueError("Use max_bond in [1,512] and tolerance strictly between 0 and 1.")
    _check_design_options(oversampling, node_rule, pinv_cutoff)
    if shots is not None:
        if isinstance(shots, bool) or int(shots) != shots or not 1 <= shots <= 10 ** 12:
            raise ValueError("Shots must be a whole number between 1 and 10^12, or omitted for exact queries.")
        shots = int(shots)
    if not 1 <= validation_samples <= 5000 or not 0 <= preview_samples <= 201:
        raise ValueError("Use 1–5000 validation points and at most 201 preview points.")
    if query_budget <= validation_samples + preview_samples:
        raise ValueError("Query budget must leave room for both training and validation.")
    if timeout_seconds <= 0:
        raise ValueError("Invalid timeout.")

    started = time.monotonic()
    rng = np.random.default_rng(seed)
    noise_rng = np.random.default_rng([seed, 1])
    is_quantum, original_domain, domain, bases, periods = _training_frame(problem)
    prediction_bases = list(problem.basis)
    target = problem.auxiliary_oracle if is_quantum else problem.target

    counts = {"training": 0, "validation": 0, "preview": 0}
    reserved = validation_samples + preview_samples
    training_budget = query_budget - reserved

    def notify(stage, message, **extra):
        if progress:
            progress({"stage": stage, "message": message,
                      "queries": sum(counts.values()), "query_counts": dict(counts),
                      "query_budget": query_budget, **extra})

    def guard():
        if cancel and cancel():
            raise FitCancelled("Learning was cancelled.")
        if time.monotonic() - started > timeout_seconds:
            raise TimeoutError("The learning time limit was reached.")

    def evaluate(fn, x, phase):
        guard()
        if sum(counts.values()) >= query_budget or (phase == "training" and counts[phase] >= training_budget):
            raise QueryBudgetExceeded(
                f"Training reached its query budget ({training_budget} calls; {reserved} reserved for validation/preview). "
                "Increase the budget or reduce the dictionary sizes / maximum bond dimension."
            )
        counts[phase] += 1
        value = fn(np.asarray(x, dtype=float))
        array = np.asarray(value)
        if array.ndim != 0 or np.iscomplexobj(array) or not np.isfinite(array):
            raise ValueError("The oracle must return one finite real number per input.")
        if counts[phase] % 25 == 0:
            notify("learning" if phase == "training" else phase,
                   f"Evaluated {counts[phase]} {phase} inputs.")
        return float(array)

    notify("design", "Checking dictionaries and constructing stable local sampling matrices.")
    midpoint = domain.mean(axis=1)
    nodes, matrices, diagnostics = _build_designs(
        bases, domain, periods, oversampling=oversampling, node_rule=node_rule,
        rng=rng, pinv_cutoff=pinv_cutoff, guard=guard)
    inverse_designs = [np.linalg.pinv(matrix, rcond=pinv_cutoff) for matrix in matrices]

    cache = {}

    def grid_oracle(key):
        key = tuple(key)
        if key not in cache:
            x = midpoint.copy()
            for k, index in enumerate(key):
                x[list(bases[k].input_indices)] = nodes[k][index]
            value = evaluate(target, x, "training")
            if shots:
                # One finite-shot estimate per distinct query; repeats reuse it.
                value += float(noise_rng.normal(0.0, 1.0 / math.sqrt(shots)))
            cache[key] = value
        return cache[key]

    dims = [len(local) for local in nodes]
    notify("starting", "Starting the Julia tensor cross interpolation engine.")
    zero_candidate = False
    # L=1 is just one sampled vector: no cross interpolation is required.
    if len(bases) == 1:
        cores_f = [np.array([grid_oracle((j,)) for j in range(dims[0])]).reshape(1, dims[0], 1)]
        pivot_errors, rank_history = [], [1]
        algorithm = "FTCI (single-dictionary direct sampling)"
    else:
        candidates = [tuple(0 for _ in dims)] + [tuple(int(rng.integers(d)) for d in dims) for _ in range(min(12, training_budget - 1))]
        initialpivot = max(candidates, key=lambda key: abs(grid_oracle(key)))
        if grid_oracle(initialpivot) == 0:
            # TCI cannot initialize a zero pivot. Probe additional grid entries,
            # then return an explicitly empirical zero candidate if all vanish.
            for _ in range(min(64, training_budget - counts["training"])):
                key = tuple(int(rng.integers(d)) for d in dims)
                if grid_oracle(key) != 0:
                    initialpivot = key
                    break
        zero_candidate = grid_oracle(initialpivot) == 0
        if zero_candidate:
            cores_f = [np.zeros((1, d, 1)) for d in dims]
            pivot_errors, rank_history = [], [1]
            algorithm = "FTCI zero-candidate fallback (zero sampled grid values; empirical holdout validation)"
        else:
            cores_f, pivot_errors, rank_history = _run_tci(
                dims, grid_oracle, initialpivot, tolerance=tolerance, max_bond=max_bond,
                seed=seed, cancel=cancel, timeout_seconds=max(1, timeout_seconds - (time.monotonic() - started)),
                notify=notify,
            )
            algorithm = "FTCI / TensorCrossInterpolation.jl two-site full-pivot TCI"

    notify("coefficients", "Transforming sample-space cores into coefficient MPS cores.")
    cores = [np.einsum("amb,mq->aqb", core, inverse)
             for core, inverse in zip(cores_f, inverse_designs)]
    finite_errors = [float(e) if np.isfinite(e) else None for e in pivot_errors]
    last_error = finite_errors[-1] if finite_errors else None
    metadata = dict(getattr(problem, "metadata", {}) or {})
    metadata.update({
        "algorithm": algorithm, "mode": "quantum" if is_quantum else "function",
        "seed": seed, "max_bond": max_bond, "tci_tolerance": tolerance,
        "oversampling": oversampling, "pinv_cutoff": pinv_cutoff,
        "designs": diagnostics, "auxiliary_domain": domain.tolist(),
        "node_rule": node_rule, "shots": shots,
        "log10_inversion_amplification": _amplification(diagnostics),
        "oracle_noise": (f"Simulated shot noise: Gaussian, variance 1/{shots} per distinct training query"
                         if shots else "Exact queries"),
        "pivot_errors": finite_errors, "rank_history": rank_history,
        "tci_tolerance_met": None if last_error is None else last_error <= tolerance,
        "query_budget": query_budget,
        "zero_candidate": zero_candidate,
        "assumption": "Target is representable (or approximable) in the supplied tensor-product dictionary. Holdout error is empirical.",
    })
    model = Model(prediction_bases, cores, original_domain.tolist(), metadata=metadata, name=problem.name)
    # Kept in memory for the nodal view; not part of the exported model.
    model.sample_cores = [np.asarray(core, dtype=float) for core in cores_f]
    model.sample_nodes = [np.asarray(local, dtype=float) for local in nodes]
    model.sample_inputs = [list(basis.input_indices) for basis in bases]
    notify("validation", "Evaluating independent holdout inputs in the original input domain.")
    xs = rng.uniform(original_domain[:, 0], original_domain[:, 1], size=(validation_samples, len(original_domain)))
    truth = np.array([evaluate(problem.target, x, "validation") for x in xs])
    predictions = np.asarray(model.predict(xs))
    residual = predictions - truth
    truth_norm = float(np.linalg.norm(truth))
    validation = {
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "relative_l2": float(np.linalg.norm(residual) / truth_norm) if truth_norm > 1e-14 else None,
        "max_absolute_error": float(np.max(np.abs(residual))),
        "n_samples": validation_samples, "distribution": "Uniform on the original input domain",
    }
    model.metadata["validation"] = validation
    if preview_samples:
        preview_x = np.linspace(*original_domain[0], preview_samples)
        fixed = original_domain.mean(axis=1)
        points = np.tile(fixed, (preview_samples, 1))
        points[:, 0] = preview_x
        model.metadata["preview"] = {
            "x": preview_x.tolist(),
            "truth": [evaluate(problem.target, x, "preview") for x in points],
            "prediction": np.asarray(model.predict(points)).tolist(),
            "varying_input": 0, "fixed_inputs": fixed.tolist(),
        }
    model.metadata.update({"query_counts": dict(counts), "queries": sum(counts.values()),
                           "seconds": time.monotonic() - started,
                           "termination": ("Zero candidate from sampled grid values; holdout validation is empirical."
                                           if zero_candidate else "completed"
                                           if last_error is None or last_error <= tolerance
                                           else "TCI stopped before the requested pivot tolerance; inspect holdout error.")})
    notify("complete", "Model and independent validation are ready.")
    return model

"""Local, same-origin web application for real FTCI jobs.

Run ``python -m ftci.server``. This development runner binds to loopback;
public hosting requires a separately deployed, authenticated compute service.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import mimetypes
import os
from pathlib import Path
import re
import threading
import time
from urllib.parse import parse_qs, urlsplit
import uuid

import numpy as np

from .basis import Basis
from .diagnostics import coefficient_structure
from .engine import NODE_RULES, FitCancelled, FunctionProblem, design_report, engine_info, fit
from .expressions import Expression
from .quantum import REPRESENTATIONS, from_circuit
from .circuit_source import parse_source, serialize_source, source_info
from .views import VIEWS, basis_view


def _heisenberg_example():
    """Trotterized Heisenberg chain: every coupling drives several gates."""
    n_qubits, steps = 3, 2
    domain = [[-1, 1] for _ in range(4)]
    # The Néel state, then K first-order Trotter steps of exp(-i H(J) t), t = 1.
    gates = [{"gate": "x", "qubits": [q]} for q in range(0, n_qubits, 2)]
    for _ in range(steps):
        for coupling, axis in enumerate(("rxx", "ryy", "rzz")):
            for parity in (0, 1):
                gates.extend({"gate": axis, "qubits": [i, i + 1], "angle": f"2*x{coupling}/{steps}"}
                             for i in range(parity, n_qubits - 1, 2))
        gates.extend({"gate": "rz", "qubits": [q], "angle": f"2*x3/{steps}"} for q in range(n_qubits))
    return {
        "id": "heisenberg-trotter", "name": "Heisenberg chain · Trotter dynamics",
        "description": "Numerics, Hamiltonian dynamics. Four couplings drive eighteen gates. Merging gates that share a coupling gives four dictionaries instead of eighteen.",
        "mode": "quantum", "target": "", "basis": [], "domain": domain,
        "representation": "merged",
        "circuit": {"n_qubits": n_qubits, "gates": gates, "observable": "IIZ", "domain": domain},
        "options": {"max_bond": 16, "tolerance": 1e-8, "oversampling": 2,
                    "query_budget": 100000, "validation_samples": 200, "seed": 7},
    }


def _random_rotations(rng, qubits):
    gates = []
    for q in qubits:
        for axis in ("rz", "ry", "rz"):
            gates.append({"gate": axis, "qubits": [q], "angle": float(rng.uniform(0, 2 * np.pi))})
    return gates


def _pauli_readout_example():
    """Every Pauli-string expectation value of a shallow-circuit state."""
    n_qubits = 6
    rng = np.random.default_rng(1905)
    gates = _random_rotations(rng, range(n_qubits))
    gates += [{"gate": "cx", "qubits": [q, q + 1]} for q in range(0, n_qubits - 1, 2)]
    gates += _random_rotations(rng, range(n_qubits))
    gates += [{"gate": "cx", "qubits": [q, q + 1]} for q in range(1, n_qubits - 1, 2)]
    # exp(-i x (X+Y+Z)/sqrt(3)) = U RZ(2x) U^dagger, where U turns the z axis into (1,1,1)/sqrt(3).
    tilt = float(np.arccos(1 / np.sqrt(3)))
    for q in range(n_qubits):
        gates += [{"gate": "rz", "qubits": [q], "angle": -np.pi / 4},
                  {"gate": "ry", "qubits": [q], "angle": -tilt},
                  {"gate": "rz", "qubits": [q], "angle": f"2*x{q}"},
                  {"gate": "ry", "qubits": [q], "angle": tilt},
                  {"gate": "rz", "qubits": [q], "angle": np.pi / 4}]
    domain = [[0, np.pi] for _ in range(n_qubits)]
    return {
        "id": "pauli-readout", "name": "Pauli expectations of a shallow-circuit state",
        "description": "Numerics, global observable. Each qubit is rotated about (X+Y+Z)/√3 before measuring Z, so x_k = 0, 2π/3, 4π/3 reads out Z, X, Y. With equispaced nodes and no oversampling the sampling-grid view of the coefficients lists the 3⁶ Pauli-string expectation values; its grid angles 0, 2π/3, 4π/3 stand for Z, Y, X.",
        "mode": "quantum", "target": "", "basis": [], "domain": domain,
        "circuit": {"n_qubits": n_qubits, "gates": gates, "observable": "Z" * n_qubits, "domain": domain},
        "options": {"max_bond": 16, "tolerance": 1e-8, "oversampling": 1, "node_rule": "equispaced",
                    "query_budget": 50000, "validation_samples": 200, "seed": 7},
    }


def _encoding_first_example():
    """Input rotations followed by a fixed unitary W, measured on one qubit."""
    n_qubits = 6
    rng = np.random.default_rng(314)
    gates = []
    for q in range(n_qubits):
        gates += [{"gate": "h", "qubits": [q]}, {"gate": "rz", "qubits": [q], "angle": f"x{q}"}]
    for layer in range(6):
        gates += _random_rotations(rng, range(n_qubits))
        gates += [{"gate": "cz", "qubits": [q, q + 1]} for q in range(layer % 2, n_qubits - 1, 2)]
    gates += _random_rotations(rng, range(n_qubits))
    domain = [[-np.pi, np.pi] for _ in range(n_qubits)]
    return {
        "id": "encoding-first", "name": "Encoding layer, then a fixed circuit W",
        "description": "Section II C. The coefficient tensor holds the Pauli coordinates of the Heisenberg-evolved observable W†OW, restricted to I, X, Y on each qubit, so the spreading in Diagnostics is its Pauli spreading.",
        "mode": "quantum", "target": "", "basis": [], "domain": domain,
        "circuit": {"n_qubits": n_qubits, "gates": gates, "observable": "I" * (n_qubits - 1) + "Z",
                    "domain": domain},
        "options": {"max_bond": 16, "tolerance": 1e-8, "oversampling": 1, "node_rule": "auto",
                    "query_budget": 50000, "validation_samples": 200, "seed": 7},
    }


def _convolutional_example():
    """A small translation-invariant convolutional circuit used as a QML model."""
    n_qubits = 8
    rng = np.random.default_rng(88)
    gates = []
    for q in range(n_qubits):
        gates += [{"gate": "h", "qubits": [q]}, {"gate": "rz", "qubits": [q], "angle": f"x{q}"}]
    active = list(range(n_qubits))
    while len(active) > 1:
        # All blocks of one layer share their parameters.
        local = rng.uniform(0, 2 * np.pi, size=4)
        coupling = rng.uniform(0, 2 * np.pi, size=3)
        pooling = float(rng.uniform(0, 2 * np.pi))
        pairs = list(zip(active[0::2], active[1::2])) + list(zip(active[1::2], active[2::2]))
        for a, b in pairs:
            gates += [{"gate": "ry", "qubits": [a], "angle": float(local[0])},
                      {"gate": "ry", "qubits": [b], "angle": float(local[1])},
                      {"gate": "rzz", "qubits": [a, b], "angle": float(coupling[0])},
                      {"gate": "ryy", "qubits": [a, b], "angle": float(coupling[1])},
                      {"gate": "rxx", "qubits": [a, b], "angle": float(coupling[2])},
                      {"gate": "rz", "qubits": [a], "angle": float(local[2])},
                      {"gate": "rz", "qubits": [b], "angle": float(local[3])}]
        for a, b in zip(active[0::2], active[1::2]):
            gates += [{"gate": "cz", "qubits": [b, a]}, {"gate": "ry", "qubits": [a], "angle": pooling}]
        active = active[0::2]
    domain = [[-np.pi, np.pi] for _ in range(n_qubits)]
    return {
        "id": "convolutional", "name": "Convolutional circuit as a QML model",
        "description": "Numerics, QML model. Eight inputs are encoded by Z rotations, followed by convolution and pooling layers with shared parameters down to one measured qubit. Learning it gives a classical surrogate of the model. It needs bond dimension 32: a sweep over the maximum bond shows the error falling as the cap rises.",
        "mode": "quantum", "target": "", "basis": [], "domain": domain,
        "circuit": {"n_qubits": n_qubits, "gates": gates, "observable": "I" * (n_qubits - 1) + "Z",
                    "domain": domain},
        "options": {"max_bond": 32, "tolerance": 1e-8, "oversampling": 1, "node_rule": "auto",
                    "query_budget": 100000, "validation_samples": 200, "seed": 7},
    }


def examples():
    defaults = {"max_bond": 12, "tolerance": 1e-8, "oversampling": 2,
                "query_budget": 10000, "validation_samples": 100, "seed": 7}
    fourier = lambda k: {"name": f"Fourier · x{k}", "expressions": [
        "1/sqrt(2)", f"cos(x{k})/sqrt(2)", f"sin(x{k})/sqrt(2)"]}
    return [
        {"id": "coupled-fourier", "name": "A coupled Fourier function",
         "description": "Three input variables, local Fourier dictionaries, and a compact coefficient MPS.",
         "mode": "function", "target": "cos(x0+x1) + 0.4*sin(x1)*cos(x2) + 0.2",
         "basis": [fourier(k) for k in range(3)],
         "domain": [[-np.pi, np.pi]] * 3, "options": defaults},
        {"id": "chirp", "name": "The quadratic chirp",
         "description": "cos(3x²), represented in a merged dictionary with preprocessing x².",
         "mode": "function", "target": "cos(3*x0**2)",
         "basis": [{"name": "Merged chirp dictionary", "expressions": ["0.5"] +
                    [f"0.5*{trig}({k}*x0**2)" for k in (1, 2, 3) for trig in ("cos", "sin")]}],
         "domain": [[-2, 2]], "options": defaults},
        {"id": "polynomial", "name": "A custom polynomial dictionary",
         "description": "General dictionaries are supported: learn a polynomial in two input blocks.",
         "mode": "function", "target": "1 + x0*x1 + 0.3*x0**2 - 0.7*x1**3",
         "basis": [{"name": f"Polynomial · x{k}", "expressions": ["1", f"x{k}", f"x{k}**2", f"x{k}**3"]} for k in range(2)],
         "domain": [[-1, 1], [-1, 1]], "options": defaults},
        _encoding_first_example(),
        _heisenberg_example(),
        _pauli_readout_example(),
        _convolutional_example(),
    ]


def inspect_circuit(spec):
    """Validate an editable circuit and describe its actual encoding occurrences.

    This does not query a simulator or run FTCI. It uses the same adapter as
    learning, so displayed generators, dependencies and dictionaries agree
    with the circuit submitted to the compute worker.
    """
    problem = from_circuit(spec)
    encodings = []
    for item in problem.metadata["encodings"]:
        # Fixed circuits have an internal constant dictionary, not an encoding.
        if "gate_position" not in item:
            continue
        index = item["encoding_index"]
        dictionary = problem.basis[index]
        encodings.append({**item, "input_indices": list(dictionary.input_indices),
                          "inputs": [f"x{i}" for i in dictionary.input_indices],
                          "basis": list(dictionary.expressions)})
    return {
        "n_qubits": problem.metadata["n_qubits"],
        "observable": problem.metadata["observable"],
        "encoding_occurrences": len(encodings), "encodings": encodings,
        "basis": [problem.basis[e["encoding_index"]].to_spec() for e in encodings],
        "auxiliary_domain": [problem.auxiliary_domain[e["encoding_index"]] for e in encodings],
        "qubit_order": problem.metadata["qubit_order"],
        "representations": {name: _describe_representation(spec, name, problem)
                            for name in ("canonical", "merged")},
    }


def describe_custom_representation(spec, sites):
    """Describe a user-arranged grouping and ordering of the encoding gates."""
    return _describe_representation(spec, "custom", None, sites)


def _describe_representation(spec, name, canonical, sites=None):
    """Sites, dictionaries and sizes of one tensor-network representation.

    Every representation describes the same function; they differ in how
    encoding gates are grouped into dictionaries, and therefore in chain
    length, local dimensions and learning cost.
    """
    try:
        problem = (canonical if name == "canonical" else
                   from_circuit(spec, representation=name, sites=sites))
    except ValueError as error:
        return {"available": False, "reason": str(error)}
    sites = []
    for site in problem.metadata["sites"]:
        # Fixed circuits have an internal constant dictionary, not an encoding.
        if "gate_positions" not in site:
            continue
        dictionary = problem.basis[site["site_index"]]
        sites.append({**site, "name": dictionary.name,
                      "input_indices": list(dictionary.input_indices),
                      "basis": list(dictionary.expressions),
                      "auxiliary_domain": list(problem.auxiliary_domain[site["site_index"]])})
    dimensions = [site["dimension"] for site in sites]
    return {"available": True, "length": len(sites), "dimensions": dimensions,
            "max_dimension": max(dimensions, default=1),
            "log10_basis_functions": sum(math.log10(d) for d in dimensions),
            "sites": sites}


def problem_from_spec(spec):
    if not isinstance(spec, dict):
        raise ValueError("The problem must be a JSON object.")
    mode = spec.get("mode", "function")
    name = str(spec.get("name", "Untitled model"))[:150]
    domain = spec.get("domain")
    if mode == "quantum":
        source_spec = spec.get("circuit_source")
        if source_spec is not None:
            if not isinstance(source_spec, dict):
                raise ValueError("Circuit source must contain a format and source text.")
            circuit = parse_source(source_spec.get("format"), source_spec.get("source"), default_domain=domain)
        else:
            circuit = deepcopy(spec.get("circuit"))
            if not isinstance(circuit, dict):
                raise ValueError("Provide circuit source code or a circuit specification.")
            circuit["domain"] = domain
        representation = spec.get("representation", "canonical")
        if representation not in REPRESENTATIONS:
            raise ValueError("Choose the canonical, merged or custom representation.")
        problem = from_circuit(circuit, name=name, representation=representation,
                               sites=spec.get("sites") if representation == "custom" else None)
        problem.metadata["circuit_spec"] = deepcopy(circuit)
        if source_spec is not None:
            problem.metadata["circuit_source"] = deepcopy(source_spec)
    elif mode == "function":
        expression = Expression(str(spec.get("target", "")))
        basis_spec = spec.get("basis", [])
        if not isinstance(basis_spec, list) or not 1 <= len(basis_spec) <= 12:
            raise ValueError("Provide 1–12 dictionaries.")
        if any(not isinstance(b, dict) or "expressions" not in b for b in basis_spec):
            raise ValueError("Each dictionary must be an object containing expressions.")
        bases = [Basis(b["expressions"], name=str(b.get("name", f"Dictionary {k+1}")))
                 for k, b in enumerate(basis_spec)]
        if any(b.dimension > 64 for b in bases):
            raise ValueError("The web runner supports at most 64 functions per dictionary.")
        if not isinstance(domain, list) or not 1 <= len(domain) <= 12:
            raise ValueError("Provide bounds for 1–12 input variables.")
        if any(i >= len(domain) for i in expression.input_indices):
            raise ValueError("The target expression refers to an input outside the domain.")
        problem = FunctionProblem(expression, bases, domain, name=name,
                                  metadata={"target_expression": str(spec["target"])})
    else:
        raise ValueError("Choose function or quantum mode.")
    allowed = {"max_bond", "tolerance", "oversampling", "query_budget", "validation_samples", "seed",
               "node_rule", "shots"}
    supplied_options = spec.get("options", {})
    if not isinstance(supplied_options, dict):
        raise ValueError("Learning options must be a JSON object.")
    options = {k: v for k, v in supplied_options.items() if k in allowed}
    if options.get("node_rule", "auto") not in NODE_RULES:
        raise ValueError("Nodes must be automatic, equispaced or random.")
    shots = options.get("shots")
    if shots is not None and (isinstance(shots, bool) or not isinstance(shots, int) or not 1 <= shots <= 10 ** 12):
        raise ValueError("Shots must be a whole number between 1 and 10^12, or empty for exact queries.")
    if not 1 <= int(options.get("max_bond", 32)) <= 64:
        raise ValueError("The web runner supports maximum bond dimensions up to 64.")
    if not 1 <= int(options.get("query_budget", 20000)) <= 100000:
        raise ValueError("The web runner query budget must be in [1,100000].")
    if not 1 <= int(options.get("validation_samples", 100)) <= 1000:
        raise ValueError("The web runner supports 1–1000 validation points.")
    return problem, options


def slice_values(model, target, axes, fixed=None, *, points=None, with_target=False):
    """Evaluate a learned model on a line or a plane through its input domain.

    ``axes`` names one or two inputs to vary over their domain; every other
    input keeps its value from ``fixed`` (domain midpoints by default). The
    prediction costs no queries. With ``with_target`` the target is queried
    once per point, after and outside the learning budget.
    """
    domain = np.asarray(model.domain, dtype=float)
    dimension = len(domain)
    if (not isinstance(axes, (list, tuple)) or not 1 <= len(axes) <= 2 or len(set(axes)) != len(axes)
            or any(isinstance(a, bool) or not isinstance(a, int) or not 0 <= a < dimension for a in axes)):
        raise ValueError("Choose one or two different inputs to vary.")
    if fixed is None:
        fixed = domain.mean(axis=1)
    try:
        fixed = np.asarray(fixed, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("Fixed input values must be numbers.") from error
    if fixed.shape != (dimension,) or not np.isfinite(fixed).all():
        raise ValueError(f"Provide one finite value for each of the {dimension} inputs.")
    limit, default = (201, 61) if len(axes) == 1 else (41, 31)
    points = default if points is None else points
    if isinstance(points, bool) or not isinstance(points, int) or not 2 <= points <= limit:
        raise ValueError(f"Use between 2 and {limit} points per axis.")
    grids = [np.linspace(*domain[a], points) for a in axes]
    inputs = np.tile(fixed, (points ** len(axes), 1))
    if len(axes) == 1:
        inputs[:, axes[0]] = grids[0]
    else:
        # Rows follow the second axis, columns the first.
        inputs[:, axes[0]] = np.tile(grids[0], points)
        inputs[:, axes[1]] = np.repeat(grids[1], points)
    shape = (points,) * len(axes)
    prediction = np.asarray(model.predict(inputs), dtype=float).reshape(shape)
    truth = None
    if with_target:
        if target is None:
            raise ValueError("The target function of this run is no longer available.")
        truth = np.array([float(target(x)) for x in inputs]).reshape(shape)
        if not np.isfinite(truth).all():
            raise ValueError("The target returned a non-finite value on this slice.")
    return {"axes": list(axes), "x": grids[0].tolist(),
            "y": grids[1].tolist() if len(axes) == 2 else None,
            "prediction": prediction.tolist(),
            "truth": None if truth is None else truth.tolist(),
            "fixed": fixed.tolist(), "target_queries": inputs.shape[0] if with_target else 0}


class Studio:
    def __init__(self, model_dir):
        self.model_dir = Path(model_dir).resolve()
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.jobs = {}
        self.lock = threading.RLock()
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ftci-fit")

    def submit(self, spec):
        problem, options = problem_from_spec(spec)
        with self.lock:
            if sum(j["status"] in ("queued", "running") for j in self.jobs.values()) >= 4:
                raise ValueError("Four jobs are already queued. Wait for a job or cancel it first.")
            while len(self.jobs) >= 32:
                oldest = next((key for key, j in self.jobs.items() if j["status"] not in ("queued", "running")), None)
                if oldest is None:
                    break
                old = self.jobs.pop(oldest)
                if old.get("path"):
                    Path(old["path"]).unlink(missing_ok=True)
            identifier = uuid.uuid4().hex
            job = {"id": identifier, "status": "queued", "progress": {
                "stage": "queued", "message": "Waiting for the local compute worker.", "queries": 0},
                "cancel": threading.Event(), "created": time.time()}
            self.jobs[identifier] = job
        self.pool.submit(self._run, job, problem, options)
        return {"id": identifier}

    def _run(self, job, problem, options):
        def update(event):
            with self.lock:
                job["progress"] = event
        try:
            if job["cancel"].is_set():
                raise FitCancelled("Learning was cancelled.")
            with self.lock:
                job["status"] = "running"
            model = fit(problem, **options, preview_samples=61,
                        progress=update, cancel=job["cancel"].is_set)
            path = self.model_dir / f'{job["id"]}.ftci'
            model.save(path)
            result = model.summary()
            result["preview"] = model.metadata.get("preview")
            result["validation"] = model.metadata["validation"]
            try:
                # Quantum dictionaries list the constant function first.
                result["structure"] = coefficient_structure(
                    model.cores, seed=int(options.get("seed", 0)),
                    constant_index=0 if model.metadata.get("mode") == "quantum" else None)
            except (ValueError, np.linalg.LinAlgError) as error:
                result["structure"] = {"error": str(error)}
            with self.lock:
                job.update(status="complete", model=model, path=path, result=result,
                           target=problem.target)
        except FitCancelled as error:
            with self.lock:
                job.update(status="cancelled", error=str(error))
        except Exception as error:
            with self.lock:
                job.update(status="failed", error=str(error))

    def view(self, identifier, name):
        """Cores, labels and structure of a finished model in one coefficient basis."""
        with self.lock:
            job = self.jobs.get(identifier)
            if job is None:
                raise KeyError("Job not found. Restarted servers do not retain the job list; downloaded models remain usable.")
            if job["status"] != "complete":
                raise ValueError("The model is not ready yet.")
            cache = job.setdefault("views", {})
            if name not in cache:
                model = job["model"]
                data = basis_view(model, name)
                quantum = model.metadata.get("mode") == "quantum"
                data["structure"] = coefficient_structure(
                    data["cores"], seed=int(model.metadata.get("seed", 0)),
                    constant_index=0 if quantum and name != "nodal" else None)
                cache[name] = data
            return cache[name]

    def public_job(self, identifier):
        with self.lock:
            job = self.jobs.get(identifier)
            if job is None:
                raise KeyError("Job not found. Restarted servers do not retain the job list; downloaded models remain usable.")
            return {k: deepcopy(job[k]) for k in ("id", "status", "progress", "created", "error", "result") if k in job}


def forwarded_host(port, environ=None):
    """Host name under which GitHub Codespaces forwards ``port``, or ``None`` elsewhere.

    The Codespaces proxy normally rewrites Host and Origin to the local
    address, so the server stays bound to the loopback interface. The
    forwarded name is accepted as well in case a request keeps it.
    """
    environ = os.environ if environ is None else environ
    name = environ.get("CODESPACE_NAME")
    domain = environ.get("GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN")
    if environ.get("CODESPACES") != "true" or not name or not domain:
        return None
    return f"{name}-{port}.{domain}".lower()


def make_handler(studio):
    class Handler(BaseHTTPRequestHandler):
        server_version = "FTCIStudio/0.1"

        def log_message(self, fmt, *args):
            # Polling logs obscure meaningful job output.
            if args and str(args[0]).startswith("POST"):
                super().log_message(fmt, *args)

        def send_bytes(self, data, content_type, status=200, disposition=None):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'")
            if disposition:
                self.send_header("Content-Disposition", disposition)
            self.end_headers()
            self.wfile.write(data)

        def json(self, data, status=200):
            self.send_bytes(json.dumps(data, allow_nan=False).encode(), "application/json; charset=utf-8", status)

        def read_json(self):
            if self.headers.get_content_type() != "application/json":
                raise ValueError("Send application/json.")
            origin = self.headers.get("Origin")
            host = self.headers.get("Host")
            allowed = {f"http://{host}"}
            forwarded = forwarded_host(self.server.server_port)
            if forwarded:
                allowed |= {f"https://{host}", f"https://{forwarded}"}
            if origin and origin not in allowed:
                raise ValueError("Requests must come from this local FTCI Studio window.")
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 1_048_576:
                raise ValueError("Request must be between 1 byte and 1 MiB.")
            return json.loads(self.rfile.read(length), parse_constant=lambda token: (_ for _ in ()).throw(ValueError(f"Invalid number: {token}")))

        def check_local_host(self):
            try:
                host = urlsplit("//" + self.headers.get("Host", ""))
                port = self.server.server_port
                local = host.hostname in {"127.0.0.1", "localhost"} and host.port == port
                forwarded = (host.hostname is not None and host.hostname == forwarded_host(port)
                             and host.port in (None, 443))
                if not (local or forwarded):
                    raise ValueError("Use this computer's local FTCI Studio address.")
            except ValueError as error:
                raise ValueError("Use this computer's local FTCI Studio address.") from error

        def job(self, identifier):
            studio.public_job(identifier)
            return studio.jobs[identifier]

        def do_GET(self):
            try:
                self.check_local_host()
                path = urlsplit(self.path).path
                if path == "/api/health":
                    return self.json({"status": "ok", **engine_info()})
                if path == "/api/examples":
                    return self.json(examples())
                match = re.fullmatch(r"/api/jobs/([a-f0-9]{32})(?:/(download|view|cores/(\d+)))?", path)
                if match:
                    identifier, action, core_index = match.groups()
                    if action is None:
                        return self.json(studio.public_job(identifier))
                    basis = parse_qs(urlsplit(self.path).query).get("basis", ["dictionary"])[0]
                    if basis not in VIEWS:
                        raise ValueError("Choose the dictionary, orthonormal or nodal view.")
                    if action == "view":
                        data = studio.view(identifier, basis)
                        return self.json({
                            "view": basis, "description": data["description"],
                            "dictionaries": data["dictionaries"], "structure": data["structure"],
                            "core_shapes": [list(core.shape) for core in data["cores"]]})
                    job = self.job(identifier)
                    if job["status"] != "complete":
                        raise ValueError("The model is not ready yet.")
                    if action == "download":
                        return self.send_bytes(job["path"].read_bytes(), "application/zip",
                                               disposition='attachment; filename="model.ftci"')
                    k = int(core_index)
                    if not 0 <= k < len(job["model"].cores):
                        raise ValueError("Core index is out of range.")
                    core = studio.view(identifier, basis)["cores"][k]
                    return self.json({"shape": list(core.shape), "data": core.tolist(),
                                      "indices": "left bond, basis index, right bond"})
                if path.startswith("/api/"):
                    return self.json({"error": "Unknown API route."}, 404)
                relative = "index.html" if path == "/" else path.lstrip("/")
                root = Path(__file__).parent / "web"
                file = (root / relative).resolve()
                if root.resolve() not in file.parents or not file.is_file():
                    return self.json({"error": "File not found."}, 404)
                content_type = mimetypes.guess_type(str(file))[0] or "application/octet-stream"
                return self.send_bytes(file.read_bytes(), content_type)
            except KeyError as error:
                return self.json({"error": str(error)}, 404)
            except (ValueError, TypeError, IndexError) as error:
                return self.json({"error": str(error)}, 400)
            except (BrokenPipeError, ConnectionResetError):
                return

        def do_POST(self):
            try:
                self.check_local_host()
                body = self.read_json()
                path = urlsplit(self.path).path
                if path == "/api/circuit/source":
                    if not isinstance(body, dict):
                        raise ValueError("Provide the source format and text.")
                    circuit = parse_source(body.get("format"), body.get("source"), default_domain=body.get("default_domain"))
                    return self.json({"circuit": circuit, "inspection": inspect_circuit(circuit),
                                      "source_info": source_info(body.get("format"), body.get("source"))})
                if path == "/api/circuit/serialize":
                    if not isinstance(body, dict):
                        raise ValueError("Provide the source format and circuit.")
                    source = serialize_source(body.get("format"), body.get("circuit"), previous_source=body.get("previous_source"))
                    circuit = parse_source(body.get("format"), source, default_domain=body.get("circuit", {}).get("domain"))
                    return self.json({"source": source, "circuit": circuit, "inspection": inspect_circuit(circuit),
                                      "source_info": source_info(body.get("format"), source)})
                if path == "/api/circuit/inspect":
                    return self.json(inspect_circuit(body))
                if path == "/api/circuit/representation":
                    if not isinstance(body, dict) or not isinstance(body.get("circuit"), dict):
                        raise ValueError("Provide the circuit and its sites.")
                    return self.json(describe_custom_representation(body["circuit"], body.get("sites")))
                if path == "/api/design":
                    problem, options = problem_from_spec(body)
                    return self.json(design_report(
                        problem, oversampling=options.get("oversampling", 2),
                        node_rule=options.get("node_rule", "auto"), seed=options.get("seed", 0)))
                if path == "/api/jobs":
                    return self.json(studio.submit(body), 202)
                match = re.fullmatch(r"/api/jobs/([a-f0-9]{32})/(predict|cancel|slice)", path)
                if not match:
                    return self.json({"error": "Unknown API route."}, 404)
                identifier, action = match.groups()
                job = self.job(identifier)
                if action == "cancel":
                    job["cancel"].set()
                    return self.json({"id": identifier, "status": job["status"]})
                if job["status"] != "complete":
                    raise ValueError("The model is not ready yet.")
                if action == "slice":
                    if not isinstance(body, dict):
                        raise ValueError("Provide the inputs to vary.")
                    return self.json(slice_values(
                        job["model"], job.get("target"), body.get("axes"), body.get("fixed"),
                        points=body.get("points"), with_target=body.get("target") is True))
                x = np.asarray(body["x"], dtype=float)
                model = job["model"]
                value = float(model(x))
                return self.json({"value": value, "basis_values": [b(x).tolist() for b in model.basis]})
            except KeyError as error:
                return self.json({"error": str(error)}, 404)
            except (ValueError, TypeError, IndexError, OverflowError) as error:
                return self.json({"error": str(error)}, 400)
            except (BrokenPipeError, ConnectionResetError):
                return

    return Handler


def main():
    parser = argparse.ArgumentParser(description="Run the local FTCI Studio web workspace.")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--model-dir", default=".ftci_jobs")
    args = parser.parse_args()
    studio = Studio(args.model_dir)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(studio))
    print(f"FTCI Studio: http://127.0.0.1:{args.port}", flush=True)
    print(f"Compute: {engine_info()['name']} | models: {studio.model_dir}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        for job in studio.jobs.values():
            job["cancel"].set()
        server.server_close()
        studio.pool.shutdown(wait=True, cancel_futures=True)


if __name__ == "__main__":
    main()

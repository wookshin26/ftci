"""Custom site arrangements, coefficient-basis views and the paper examples."""

from email.message import Message
import io
import itertools
import json
import math
import shutil
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from ftci import Basis, FunctionProblem, Model, fit
from ftci.expressions import Expression
from ftci.quantum import _apply_operator, _gate_matrix, _I, _X, _Y, _Z, from_circuit
from ftci.server import (Studio, describe_custom_representation, examples, make_handler,
                         problem_from_spec)
from ftci.views import basis_view

JULIA = shutil.which("julia")


def circuit():
    # Encodings: 0 ry(2*x0), 1 rx(x0+x0), 2 rz(x1), 3 rzz(2*x0), 4 ry(x0*x1)
    return {"n_qubits": 2, "observable": "ZZ", "domain": [[-1, 1], [-2, 2]], "gates": [
        {"gate": "h", "qubits": [0]},
        {"gate": "ry", "qubits": [0], "angle": "2*x0"},
        {"gate": "cx", "qubits": [0, 1]},
        {"gate": "rx", "qubits": [1], "angle": "x0+x0"},
        {"gate": "rz", "qubits": [0], "angle": "x1"},
        {"gate": "rzz", "qubits": [0, 1], "angle": "2*x0"},
        {"gate": "ry", "qubits": [1], "angle": "x0*x1"},
    ]}


def contract(cores, vectors):
    state = np.ones(1)
    for core, vector in zip(cores, vectors):
        state = state @ np.einsum("aqb,q->ab", core, vector)
    return float(state[0])


def evaluate(view, x):
    return contract(view["cores"], [np.array([Expression(text)(x) for text in dictionary["expressions"]])
                                    for dictionary in view["dictionaries"]])


def pauli(letters):
    matrix = np.eye(1)
    for letter in reversed(letters):  # little-endian: the last qubit is the leftmost factor
        matrix = np.kron(matrix, {"I": _I, "X": _X, "Y": _Y, "Z": _Z}[letter])
    return matrix


def unitary(gates, n_qubits):
    columns = []
    for column in np.eye(2 ** n_qubits, dtype=complex):
        for gate in gates:
            column = _apply_operator(column, _gate_matrix(gate["gate"], gate.get("angle")),
                                     gate["qubits"], n_qubits)
        columns.append(column)
    return np.stack(columns, axis=1)


class CustomRepresentationTests(unittest.TestCase):
    def test_sites_keep_their_order_and_combine_merged_groups(self):
        problem = from_circuit(circuit(), representation="custom", sites=[[4], [2, 0, 3], [1]])
        self.assertEqual([basis.dimension for basis in problem.basis], [3, 15, 3])
        self.assertEqual([basis.input_indices for basis in problem.auxiliary_basis], [(0,), (1, 2), (3,)])
        joined = problem.metadata["sites"][1]
        self.assertEqual([group["encoding_indices"] for group in joined["classes"]], [[2], [0, 3]])
        self.assertEqual([group["frequencies"] for group in joined["classes"]], [[1.0], [1.0, 2.0]])
        self.assertIsNone(joined["preprocessing"])
        self.assertEqual(problem.auxiliary_periods[1], None)
        self.assertTrue(all(basis.portable for basis in problem.basis))
        self.assertEqual(problem.basis[1].input_indices, (0, 1))

    def test_training_angles_reach_the_right_gates(self):
        canonical = from_circuit(circuit())
        custom = from_circuit(circuit(), representation="custom", sites=[[4], [2, 0, 3], [1]])
        a = [0.3, 0.7, -0.4, 1.1]
        self.assertAlmostEqual(custom.auxiliary_oracle(a),
                               canonical.auxiliary_oracle([a[2], a[3], a[1], a[2], a[0]]), places=12)
        x = [0.2, -0.6]
        self.assertAlmostEqual(custom.target(x), canonical.target(x), places=12)

    def test_presets_are_special_cases_of_a_custom_arrangement(self):
        merged = from_circuit(circuit(), representation="merged")
        same = from_circuit(circuit(), representation="custom", sites=[[0, 1, 3], [2], [4]])
        self.assertEqual([basis.expressions for basis in same.basis],
                         [basis.expressions for basis in merged.basis])
        singles = from_circuit(circuit(), representation="custom", sites=[[k] for k in range(5)])
        self.assertEqual([basis.expressions for basis in singles.basis],
                         [basis.expressions for basis in from_circuit(circuit()).basis])

    def test_invalid_arrangements_are_rejected(self):
        for sites in ([[0, 1]], [[0], [1], [2], [3], [4], [4]], [[0, 1, 2, 3, 4], []], "abc", [[0.5, 1, 2, 3, 4]]):
            with self.assertRaisesRegex(ValueError, "sites must"):
                from_circuit(circuit(), representation="custom", sites=sites)
        with self.assertRaisesRegex(ValueError, "exactly when"):
            from_circuit(circuit(), representation="custom")
        with self.assertRaisesRegex(ValueError, "exactly when"):
            from_circuit(circuit(), representation="merged", sites=[[0, 1, 2, 3, 4]])

    def test_oversized_joined_dictionary_is_refused(self):
        spec = {"n_qubits": 1, "observable": "Z", "domain": [[-1, 1]] * 7,
                "gates": [{"gate": "ry", "qubits": [0], "angle": f"x{k}"} for k in range(7)]}
        with self.assertRaisesRegex(ValueError, "2187 functions"):
            from_circuit(spec, representation="custom", sites=[list(range(7))])

    def test_server_describes_and_accepts_custom_arrangements(self):
        description = describe_custom_representation(circuit(), [[4], [2, 0, 3], [1]])
        self.assertEqual((description["length"], description["dimensions"]), (3, [3, 15, 3]))
        self.assertEqual(description["sites"][1]["gate_positions"], [4, 1, 5])
        self.assertFalse(describe_custom_representation(circuit(), [[0]])["available"])
        spec = {"mode": "quantum", "name": "Custom", "domain": [[-1, 1], [-2, 2]], "circuit": circuit(),
                "representation": "custom", "sites": [[4], [2, 0, 3], [1]]}
        problem, _ = problem_from_spec(spec)
        self.assertEqual(problem.metadata["representation"], "custom")
        del spec["sites"]
        with self.assertRaisesRegex(ValueError, "exactly when"):
            problem_from_spec(spec)

    @unittest.skipUnless(JULIA, "Julia is not installed")
    def test_custom_arrangement_learns_the_same_function(self):
        model = fit(from_circuit(circuit(), representation="custom", sites=[[4], [2, 0, 3], [1]]),
                    max_bond=12, query_budget=20000, validation_samples=60, seed=3)
        self.assertLess(model.metadata["validation"]["rmse"], 1e-9)
        self.assertEqual([core.shape[1] for core in model.cores], [3, 15, 3])
        self.assertEqual([site["node_rule"] for site in model.metadata["designs"]],
                         ["equispaced", "random", "equispaced"])


@unittest.skipUnless(JULIA, "Julia is not installed")
class BasisViewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.target = staticmethod(lambda x: 1 + x[0] * x[1] + 0.3 * x[0] ** 2 - 0.7 * x[1] ** 3)
        cls.function = fit(FunctionProblem(
            cls.target, [Basis(["1", f"x{k}", f"x{k}**2", f"x{k}**3"]) for k in range(2)],
            [(-1, 1), (-1, 1)]), max_bond=6, query_budget=2000, validation_samples=20, seed=1)
        cls.quantum = fit(from_circuit(circuit(), representation="custom", sites=[[4], [2, 0, 3], [1]]),
                          max_bond=12, query_budget=20000, validation_samples=20, seed=3)

    def test_orthonormal_function_view_is_the_same_function_with_parseval(self):
        view = basis_view(self.function, "orthonormal")
        # Orthonormalizing monomials in order gives the normalized Legendre polynomials.
        self.assertEqual(view["dictionaries"][0]["expressions"][:2], ["1*(1)", "1.73205*(x0)"])
        x = np.array([0.3, -0.6])
        self.assertAlmostEqual(evaluate(view, x), self.function(x), places=4)
        tensor = np.einsum("aib,bjc->ij", *view["cores"])
        nodes, weights = np.polynomial.legendre.leggauss(12)
        mean_square = sum(wa * wb / 4 * self.target([a, b]) ** 2
                          for a, wa in zip(nodes, weights) for b, wb in zip(nodes, weights))
        self.assertAlmostEqual(float(np.sum(tensor ** 2)), mean_square, places=10)
        self.assertEqual([core.shape for core in view["cores"]],
                         [core.shape for core in self.function.cores])

    def test_nodal_view_holds_function_values_on_the_grid(self):
        view = basis_view(self.function, "nodal")
        nodes = self.function.sample_nodes
        for j0, j1 in ((0, 0), (2, 5), (7, 3)):
            entry = float(view["cores"][0][0, j0, :] @ view["cores"][1][:, j1, 0])
            self.assertAlmostEqual(entry, self.target([nodes[0][j0][0], nodes[1][j1][0]]), places=10)
        self.assertTrue(view["dictionaries"][0]["labels"][0].startswith("x0 = "))
        self.assertIsNone(view["dictionaries"][0]["expressions"])

    def test_orthonormal_quantum_view_rescales_without_changing_the_function(self):
        view = basis_view(self.quantum, "orthonormal")
        x = np.array([0.4, -1.0])
        self.assertAlmostEqual(evaluate(view, x), self.quantum(x), places=10)
        self.assertEqual(view["dictionaries"][0]["expressions"][:2], ["1", "sqrt(2)*cos((1.0)*(x0*x1))"])
        self.assertEqual(view["dictionaries"][1]["dimension"], 15)
        labels = basis_view(self.quantum, "nodal")["dictionaries"][1]["labels"]
        self.assertRegex(labels[0], r"^a2 = .+, a3 = .+$")

    def test_dictionary_view_and_invalid_requests(self):
        view = basis_view(self.function, "dictionary")
        np.testing.assert_array_equal(view["cores"][0], self.function.cores[0])
        with self.assertRaisesRegex(ValueError, "Choose the dictionary"):
            basis_view(self.function, "pauli")
        with tempfile.TemporaryDirectory() as directory:
            restored = Model.load(self.function.save(f"{directory}/model.ftci"))
        with self.assertRaisesRegex(ValueError, "learned in this session"):
            basis_view(restored, "nodal")
        self.assertAlmostEqual(evaluate(basis_view(restored, "orthonormal"), np.array([0.1, 0.2])),
                               restored([0.1, 0.2]), places=4)

    def test_dependent_dictionary_has_no_orthonormal_form(self):
        dependent = Model([Basis(["1", "x0", "2*x0"])], [np.ones((1, 3, 1))], [[-1, 1]],
                          metadata={"mode": "function"})
        with self.assertRaisesRegex(ValueError, "not linearly independent"):
            basis_view(dependent, "orthonormal")

    def test_server_serves_every_view_of_a_finished_job(self):
        spec = {"mode": "quantum", "name": "Views", "domain": [[-1, 1], [-2, 2]], "circuit": circuit(),
                "representation": "merged",
                "options": {"max_bond": 8, "query_budget": 5000, "validation_samples": 20, "oversampling": 1}}
        with tempfile.TemporaryDirectory() as directory:
            studio = Studio(directory)
            try:
                identifier = studio.submit(spec)["id"]
                studio.pool.shutdown(wait=True)
                self.assertEqual(studio.public_job(identifier)["status"], "complete")

                def get(path):
                    handler_class = make_handler(studio)
                    handler = handler_class.__new__(handler_class)
                    handler.requestline, handler.request_version = f"GET {path} HTTP/1.1", "HTTP/1.1"
                    handler.command, handler.path = "GET", path
                    handler.client_address = ("127.0.0.1", 0)
                    handler.server = SimpleNamespace(server_port=8765)
                    handler.headers = Message()
                    handler.headers["Host"] = "127.0.0.1:8765"
                    handler.rfile, handler.wfile = io.BytesIO(), io.BytesIO()
                    handler.log_message = lambda *a: None
                    handler.do_GET()
                    header, data = handler.wfile.getvalue().split(b"\r\n\r\n", 1)
                    return int(header.split()[1]), json.loads(data)

                shapes = {}
                for name in ("dictionary", "orthonormal", "nodal"):
                    status, view = get(f"/api/jobs/{identifier}/view?basis={name}")
                    self.assertEqual(status, 200)
                    shapes[name] = view["core_shapes"]
                    self.assertEqual(len(view["structure"]["bonds"]), 2)
                    status, core = get(f"/api/jobs/{identifier}/cores/0?basis={name}")
                    self.assertEqual((status, core["shape"]), (200, view["core_shapes"][0]))
                self.assertEqual(shapes["dictionary"], shapes["orthonormal"])
                self.assertEqual([shape[1] for shape in shapes["nodal"]], [7, 3, 3])
                self.assertEqual(get(f"/api/jobs/{identifier}/view?basis=pauli")[0], 400)
                self.assertEqual(get(f"/api/jobs/{identifier}/cores/0")[1]["shape"], shapes["dictionary"][0])
            finally:
                studio.pool.shutdown(wait=True, cancel_futures=True)


class ExampleTests(unittest.TestCase):
    def test_every_example_is_a_valid_problem_with_a_description(self):
        listed = examples()
        json.dumps(listed, allow_nan=False)
        self.assertEqual(len({example["id"] for example in listed}), len(listed))
        for example in listed:
            problem, _ = problem_from_spec(example)
            self.assertTrue(problem.basis)
            self.assertTrue(example.get("description"), example["id"])

    @unittest.skipUnless(JULIA, "Julia is not installed")
    def test_encoding_first_coefficients_are_pauli_coordinates(self):
        example = next(item for item in examples() if item["id"] == "encoding-first")
        problem, options = problem_from_spec(example)
        model = fit(problem, **options)
        n = example["circuit"]["n_qubits"]
        w = unitary(example["circuit"]["gates"][2 * n:], n)
        evolved = w.conj().T @ pauli("Z" + "I" * (n - 1)) @ w
        for index in itertools.product(range(3), repeat=n):
            coordinate = np.trace(pauli(["IXY"[i] for i in index]) @ evolved).real / 2 ** (n / 2)
            self.assertAlmostEqual(model.coefficient(index), coordinate, places=10)

    @unittest.skipUnless(JULIA, "Julia is not installed")
    def test_pauli_readout_grid_values_are_expectation_values(self):
        example = next(item for item in examples() if item["id"] == "pauli-readout")
        problem, options = problem_from_spec(example)
        model = fit(problem, **options)
        n = example["circuit"]["n_qubits"]
        np.testing.assert_allclose(model.sample_nodes[0][:, 0], [0, 2 * math.pi / 3, 4 * math.pi / 3])
        state = unitary(example["circuit"]["gates"][:-5 * n], n)[:, 0]
        cores = basis_view(model, "nodal")["cores"]
        rng = np.random.default_rng(0)
        for index in rng.integers(0, 3, size=(40, n)):
            value = np.ones(1)
            for core, i in zip(cores, index):
                value = value @ core[:, i, :]
            expectation = np.vdot(state, pauli(["ZYX"[i] for i in index]) @ state).real
            self.assertAlmostEqual(float(value[0]), expectation, places=10)


if __name__ == "__main__":
    unittest.main()

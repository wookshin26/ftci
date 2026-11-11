"""Sampling designs, simulated shot noise and coefficient-tensor diagnostics."""

from email.message import Message
import io
import json
import math
import shutil
from types import SimpleNamespace
import unittest

import numpy as np

from ftci import Basis, FunctionProblem, fit
from ftci.diagnostics import coefficient_structure
from ftci.engine import design_report
from ftci.quantum import _period, from_circuit
from ftci.server import Studio, examples, make_handler, problem_from_spec


def merged_circuit(gates=3):
    return {"n_qubits": 1, "observable": "Z", "domain": [[-1, 1], [-1, 1]],
            "gates": [{"gate": "h", "qubits": [0]}]
            + [{"gate": "rz" if k % 2 else "ry", "qubits": [0], "angle": "x0"} for k in range(gates)]
            + [{"gate": "rx", "qubits": [0], "angle": "x1"}]}


def fourier_problem():
    basis = [Basis(["1", f"cos(x{k})", f"sin(x{k})"]) for k in range(2)]
    return FunctionProblem(lambda x: math.cos(x[0] + x[1]) + 0.2, basis,
                           [(-math.pi, math.pi)] * 2)


class NodeRuleTests(unittest.TestCase):
    def test_commensurate_frequencies_have_a_common_period(self):
        self.assertAlmostEqual(_period([1.0, 2.0, 3.0]), 2 * math.pi)
        self.assertAlmostEqual(_period([0.5, 1.5]), 4 * math.pi)
        self.assertAlmostEqual(_period([3.0, 5.0]), 2 * math.pi)
        self.assertIsNone(_period([1.0, math.sqrt(2)]))
        self.assertIsNone(_period([]))

    def test_equispaced_nodes_make_trigonometric_designs_orthogonal(self):
        problem = from_circuit(merged_circuit(), representation="merged")
        self.assertEqual([basis.dimension for basis in problem.basis], [7, 3])
        equispaced = design_report(problem, oversampling=1, node_rule="equispaced")
        for site in equispaced["designs"]:
            self.assertEqual(site["node_rule"], "equispaced")
            self.assertEqual(site["auxiliary_points"], site["dimension"])
            self.assertAlmostEqual(site["condition_number"], math.sqrt(2), places=9)
        random = design_report(problem, oversampling=1, node_rule="random", seed=3)
        self.assertEqual({site["node_rule"] for site in random["designs"]}, {"random"})
        self.assertGreater(random["log10_inversion_amplification"],
                           equispaced["log10_inversion_amplification"] + 1)

    def test_automatic_nodes_are_never_worse_than_either_rule(self):
        for problem in (from_circuit(merged_circuit(), representation="merged"), fourier_problem()):
            reports = {rule: design_report(problem, oversampling=2, node_rule=rule, seed=5)
                       for rule in ("auto", "equispaced", "random")}
            self.assertLessEqual(reports["auto"]["log10_inversion_amplification"],
                                 min(reports["equispaced"]["log10_inversion_amplification"],
                                     reports["random"]["log10_inversion_amplification"]) + 1e-12)

    def test_design_report_needs_no_function_queries(self):
        def forbidden(x):
            raise AssertionError("the target must not be queried")
        problem = FunctionProblem(forbidden, [Basis(["1", "x0", "x0**2"])], [(-1, 1)])
        report = design_report(problem, oversampling=2)
        self.assertEqual(len(report["designs"]), 1)
        json.dumps(report, allow_nan=False)

    def test_repeating_equispaced_inputs_are_reported_and_avoided_automatically(self):
        even = FunctionProblem(lambda x: x[0] ** 4, [Basis(["1", "x0**2", "x0**4", "x0**6"])], [(-1, 1)])
        with self.assertRaisesRegex(ValueError, "Evenly spaced inputs repeat"):
            design_report(even, oversampling=1, node_rule="equispaced")
        self.assertEqual(design_report(even, oversampling=1, node_rule="auto")["designs"][0]["node_rule"],
                         "random")

    def test_multi_input_dictionary_keeps_random_nodes(self):
        joint = FunctionProblem(lambda x: x[0] * x[1], [Basis(["1", "x0", "x1", "x0*x1"])],
                                [(-1, 1), (-1, 1)])
        self.assertEqual(design_report(joint, node_rule="equispaced")["designs"][0]["node_rule"], "random")

    def test_invalid_design_options_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Nodes must be"):
            design_report(fourier_problem(), node_rule="chebyshev")
        with self.assertRaisesRegex(ValueError, "Oversampling"):
            design_report(fourier_problem(), oversampling=0.5)


@unittest.skipUnless(shutil.which("julia"), "Julia is not installed")
class NoisyLearningTests(unittest.TestCase):
    def test_equispaced_nodes_learn_without_oversampling(self):
        model = fit(from_circuit(merged_circuit(), representation="merged"), max_bond=8, oversampling=1,
                    node_rule="equispaced", query_budget=2000, validation_samples=40, seed=2)
        self.assertLess(model.metadata["validation"]["rmse"], 1e-10)
        self.assertEqual(model.metadata["node_rule"], "equispaced")
        self.assertIsNone(model.metadata["shots"])
        self.assertLess(model.metadata["log10_inversion_amplification"], 1)

    def test_shot_noise_perturbs_training_but_not_validation(self):
        exact = fit(fourier_problem(), max_bond=4, query_budget=2000, validation_samples=50, seed=4)
        noisy = [fit(fourier_problem(), max_bond=4, query_budget=2000, validation_samples=50,
                     seed=4, shots=10 ** 4) for _ in range(2)]
        self.assertLess(exact.metadata["validation"]["rmse"], 1e-10)
        self.assertEqual(noisy[0].metadata["validation"]["rmse"], noisy[1].metadata["validation"]["rmse"])
        self.assertGreater(noisy[0].metadata["validation"]["rmse"], 1e-4)
        self.assertLess(noisy[0].metadata["validation"]["rmse"], 0.2)
        self.assertEqual(noisy[0].metadata["shots"], 10 ** 4)
        self.assertIn("variance 1/10000", noisy[0].metadata["oracle_noise"])

    def test_invalid_shots_are_rejected(self):
        for shots in (0, 2.5, True, -3):
            with self.assertRaisesRegex(ValueError, "Shots must be"):
                fit(fourier_problem(), shots=shots)


class CoefficientStructureTests(unittest.TestCase):
    def test_uniform_tensor_is_fully_spread_but_unentangled(self):
        result = coefficient_structure([np.ones((1, 3, 1)) for _ in range(5)], constant_index=0)
        self.assertAlmostEqual(result["maximum_bond_entropy"], 0, places=10)
        self.assertAlmostEqual(result["spreading"]["shannon"], 5 * math.log(3))
        self.assertAlmostEqual(result["spreading"]["participation"], 3 ** 5)
        self.assertAlmostEqual(sum(result["support_weights"]), 1)
        self.assertAlmostEqual(result["support_weights"][5], (2 / 3) ** 5)

    def test_single_coefficient_has_no_spreading(self):
        cores = [np.zeros((1, 3, 1)) for _ in range(4)]
        for core in cores:
            core[0, 1, 0] = -2.0
        result = coefficient_structure(cores, constant_index=0)
        self.assertAlmostEqual(result["norm"], 16.0)
        self.assertAlmostEqual(result["spreading"]["shannon"], 0)
        self.assertEqual(result["support_weights"], [0, 0, 0, 0, 1])

    def test_entanglement_never_exceeds_spreading_and_sampling_agrees(self):
        rng = np.random.default_rng(11)
        bonds = [1, 3, 4, 4, 4, 3, 1]
        cores = [rng.normal(size=(bonds[k], 3, bonds[k + 1])) for k in range(6)]
        exact = coefficient_structure(cores, constant_index=0)
        self.assertEqual(exact["spreading"]["method"], "exact")
        self.assertEqual([bond["dimension"] for bond in exact["bonds"]], [3, 4, 4, 4, 3])
        for bond in exact["bonds"]:
            self.assertLessEqual(bond["entropy"], exact["spreading"]["shannon"])
            self.assertLessEqual(bond["renyi2"], exact["spreading"]["renyi2"] + 1e-12)
            self.assertAlmostEqual(sum(v * v for v in bond["schmidt_values"]), 1)
        sampled = coefficient_structure(cores, constant_index=0, exact_limit=1, samples=20000, seed=1)
        self.assertEqual(sampled["spreading"]["method"], "sampled")
        self.assertAlmostEqual(sampled["spreading"]["shannon"], exact["spreading"]["shannon"], delta=0.06)
        self.assertAlmostEqual(sampled["spreading"]["renyi2"], exact["spreading"]["renyi2"], delta=0.1)
        np.testing.assert_allclose(sampled["support_weights"], exact["support_weights"], atol=0.02)

    def test_gauge_changes_leave_the_structure_unchanged(self):
        rng = np.random.default_rng(5)
        cores = [rng.normal(size=shape) for shape in ((1, 3, 2), (2, 3, 2), (2, 3, 1))]
        gauge = rng.normal(size=(2, 2))
        changed = [np.einsum("aqb,bc->aqc", cores[0], gauge),
                   np.einsum("ab,bqc->aqc", np.linalg.inv(gauge), cores[1]), cores[2]]
        first, second = coefficient_structure(cores), coefficient_structure(changed)
        self.assertAlmostEqual(first["norm"], second["norm"])
        self.assertAlmostEqual(first["bonds"][0]["entropy"], second["bonds"][0]["entropy"])
        self.assertIsNone(first["support_weights"])

    def test_zero_tensor_and_invalid_cores(self):
        self.assertTrue(coefficient_structure([np.zeros((1, 2, 1))])["zero"])
        with self.assertRaises(ValueError):
            coefficient_structure([np.zeros((2, 2))])


class SamplingServerTests(unittest.TestCase):
    def post(self, path, body):
        handler_class = make_handler(SimpleNamespace())
        handler = handler_class.__new__(handler_class)
        handler.requestline = f"POST {path} HTTP/1.1"
        handler.request_version, handler.command, handler.path = "HTTP/1.1", "POST", path
        handler.client_address = ("127.0.0.1", 0)
        handler.server = SimpleNamespace(server_port=8765)
        handler.headers = Message()
        handler.headers["Host"] = "127.0.0.1:8765"
        raw = json.dumps(body).encode()
        handler.headers["Content-Type"] = "application/json"
        handler.headers["Content-Length"] = str(len(raw))
        handler.rfile, handler.wfile = io.BytesIO(raw), io.BytesIO()
        handler.log_message = lambda *a: None
        handler.do_POST()
        header, data = handler.wfile.getvalue().split(b"\r\n\r\n", 1)
        return int(header.split()[1]), json.loads(data)

    def quantum_spec(self, **options):
        return {"mode": "quantum", "name": "Merged", "domain": [[-1, 1], [-1, 1]],
                "circuit": merged_circuit(), "representation": "merged",
                "options": {"max_bond": 8, "query_budget": 2000, "validation_samples": 20, **options}}

    def test_design_endpoint_reports_conditioning_before_learning(self):
        status, report = self.post("/api/design", self.quantum_spec(oversampling=1, node_rule="auto"))
        self.assertEqual(status, 200)
        self.assertEqual([site["node_rule"] for site in report["designs"]], ["equispaced", "equispaced"])
        self.assertLess(report["log10_inversion_amplification"], 1)
        status, report = self.post("/api/design", examples()[0])
        self.assertEqual((status, len(report["designs"])), (200, 3))
        status, report = self.post("/api/design", self.quantum_spec(node_rule="lattice"))
        self.assertEqual(status, 400)
        self.assertIn("Nodes must be", report["error"])

    def test_sampling_options_are_validated(self):
        _, options = problem_from_spec(self.quantum_spec(node_rule="random", shots=500))
        self.assertEqual((options["node_rule"], options["shots"]), ("random", 500))
        for shots in (0, 1.5, "100", True):
            with self.assertRaisesRegex(ValueError, "Shots must be"):
                problem_from_spec(self.quantum_spec(shots=shots))

    @unittest.skipUnless(shutil.which("julia"), "Julia is not installed")
    def test_completed_job_carries_structure_diagnostics(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            studio = Studio(directory)
            try:
                identifier = studio.submit(self.quantum_spec(oversampling=1))["id"]
                studio.pool.shutdown(wait=True)
                job = studio.public_job(identifier)
            finally:
                studio.pool.shutdown(wait=True, cancel_futures=True)
        self.assertEqual(job["status"], "complete", job.get("error"))
        structure = job["result"]["structure"]
        self.assertEqual(len(structure["bonds"]), 1)
        self.assertEqual(len(structure["support_weights"]), 3)
        self.assertEqual(job["result"]["metadata"]["designs"][0]["node_rule"], "equispaced")
        json.dumps(job, allow_nan=False)


if __name__ == "__main__":
    unittest.main()

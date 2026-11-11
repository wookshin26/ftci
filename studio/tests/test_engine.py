import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ftci import Basis, FunctionProblem, Model, fit
from ftci.engine import FitCancelled, QueryBudgetExceeded, engine_info
from ftci.quantum import from_circuit, from_generators


class DesignTests(unittest.TestCase):
    def test_overlapping_inputs_fail_explicitly(self):
        problem = FunctionProblem(lambda x: np.cos(3*x[0]**2),
                                  [Basis(["1", "cos(x0)", "sin(x0)"]),
                                   Basis(["1", "cos(2*x0)", "sin(2*x0)"])], [[-1, 1]])
        with self.assertRaisesRegex(ValueError, "both depend on x0"):
            fit(problem, validation_samples=5)

    def test_redundant_basis_rejected(self):
        problem = FunctionProblem(lambda x: x[0], [Basis(["1", "x0", "2*x0"])], [[-1, 1]])
        with self.assertRaisesRegex(ValueError, "full-row-rank"):
            fit(problem, validation_samples=5)

    def test_scalar_chirp_and_out_of_basis_error(self):
        target = lambda x: np.cos(3*x[0]**2)
        exact = FunctionProblem(target, [Basis(["1", "cos(3*x0**2)", "sin(3*x0**2)"])], [[-2, 2]])
        learned = fit(exact, validation_samples=25, preview_samples=9, query_budget=100, seed=4)
        self.assertLess(learned.metadata["validation"]["rmse"], 1e-12)
        self.assertEqual(learned.metadata["queries"], sum(learned.metadata["query_counts"].values()))
        self.assertLessEqual(learned.metadata["queries"], 100)
        insufficient = FunctionProblem(target, [Basis(["1", "cos(x0)", "sin(x0)"])], [[-2, 2]])
        approximate = fit(insufficient, validation_samples=40, query_budget=100)
        self.assertGreater(approximate.metadata["validation"]["rmse"], 0.1)

    def test_early_cancellation_and_invalid_oracle(self):
        problem = FunctionProblem(lambda x: x[0], [Basis(["1", "x0"])], [[-1, 1]])
        with self.assertRaises(FitCancelled):
            fit(problem, cancel=lambda: True, validation_samples=5)
        problem.target = lambda x: float("nan")
        with self.assertRaisesRegex(ValueError, "finite real"):
            fit(problem, validation_samples=5)

    def test_zero_multisite_candidate(self):
        problem = FunctionProblem(lambda x: 0., [Basis(["1", "x0"]), Basis(["1", "x1"])], [[-1, 1]]*2)
        model = fit(problem, validation_samples=10, query_budget=200)
        self.assertEqual(model([0.2, 0.7]), 0.)
        self.assertTrue(model.metadata["zero_candidate"])
        self.assertIsNone(model.metadata["tci_tolerance_met"])
        self.assertEqual(model.metadata["validation"]["rmse"], 0.)


@unittest.skipUnless(engine_info()["engine_available"], "Julia is not installed")
class ActualTCITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.problem = FunctionProblem(
            lambda x: np.cos(x[0]+x[1]) + 0.4*np.sin(x[1])*np.cos(x[2]) + 0.2,
            [Basis(["1/sqrt(2)", f"cos(x{k})/sqrt(2)", f"sin(x{k})/sqrt(2)"])
             for k in range(3)], [[-math.pi, math.pi]] * 3)
        cls.model = fit(cls.problem, max_bond=8, validation_samples=30, preview_samples=11,
                        query_budget=1500, seed=12)

    def test_real_tci_continuous_recovery(self):
        self.assertIn("TensorCrossInterpolation.jl", self.model.metadata["algorithm"])
        self.assertLess(self.model.metadata["validation"]["relative_l2"], 1e-10)
        self.assertTrue(all(r <= 8 for r in self.model.bond_dimensions))
        self.assertEqual(self.model.metadata["query_counts"]["validation"], 30)
        self.assertEqual(self.model.metadata["query_counts"]["preview"], 11)
        self.assertLessEqual(self.model.metadata["queries"], 1500)

    def test_learned_model_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "learned.ftci"
            self.model.save(path)
            restored = Model.load(path)
            x = [0.17, -0.39, 1.12]
            self.assertAlmostEqual(restored(x), self.problem.target(np.asarray(x)), places=10)

    def test_shared_input_quantum_end_to_end(self):
        problem = from_circuit({"n_qubits": 1, "observable": "Z", "domain": [[-2, 2]],
                                "gates": [{"gate": "ry", "qubits": [0], "angle": "x0"},
                                          {"gate": "ry", "qubits": [0], "angle": "2*x0**2"}]})
        model = fit(problem, max_bond=6, query_budget=1000, validation_samples=30, seed=5)
        xs = np.array([[-1.3], [-0.2], [0.7], [1.4]])
        np.testing.assert_allclose(model.predict(xs), np.cos(xs[:,0]+2*xs[:,0]**2), atol=1e-10)
        self.assertLess(model.metadata["validation"]["rmse"], 1e-10)
        self.assertEqual(len(model.cores), 2)

    def test_manual_hidden_oracle(self):
        problem = from_generators([[-0.5, 0.5], [-1, 1]], ["x0", "x0**2"],
                                  lambda a: np.cos(a[0])*np.cos(2*a[1]), [[-1, 1]], spectra=True)
        model = fit(problem, validation_samples=10, query_budget=500, seed=3)
        self.assertAlmostEqual(model([0.37]), np.cos(0.37)*np.cos(2*0.37**2), places=10)

    def test_quantum_model_with_42_encoding_occurrences_roundtrips(self):
        # A single qubit with 42 RY rotations shares one physical input while
        # the oracle exposes all 42 independent auxiliary angles during fit.
        sites = 42
        problem = from_generators([[-0.5, 0.5]] * sites, ["x0"] * sites,
                                  lambda angles: np.cos(angles.sum()), [[-1, 1]], spectra=True)
        model = fit(problem, max_bond=2, query_budget=20_000, validation_samples=12, seed=2)
        self.assertEqual(len(model.cores), sites)
        self.assertLess(model.metadata["validation"]["rmse"], 1e-9)
        with tempfile.TemporaryDirectory() as directory:
            path = model.save(Path(directory) / "large-quantum.ftci")
            restored = Model.load(path)
            self.assertEqual(len(restored.cores), sites)
            self.assertAlmostEqual(restored([0.37]), np.cos(sites * 0.37), places=8)

    def test_budget_is_hard_limit(self):
        calls = []
        problem = FunctionProblem(lambda x: calls.append(x.copy()) or np.cos(x.sum()),
                                  [Basis(["1", "cos(x0)", "sin(x0)"]),
                                   Basis(["1", "cos(x1)", "sin(x1)"])], [[-1, 1]]*2)
        with self.assertRaises(QueryBudgetExceeded):
            fit(problem, query_budget=12, validation_samples=5)
        self.assertLessEqual(len(calls), 7)


if __name__ == "__main__":
    unittest.main()

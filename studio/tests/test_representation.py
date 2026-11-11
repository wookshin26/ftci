"""The same quantum function in canonical and merged tensor-network form."""

import math
import shutil
import unittest

import numpy as np

from ftci import fit
from ftci.quantum import from_circuit, from_generators, merged_frequencies
from ftci.server import inspect_circuit, problem_from_spec


def shared_angle_circuit():
    return {"n_qubits": 2, "observable": "ZZ", "domain": [[-1, 1], [-2, 2]], "gates": [
        {"gate": "h", "qubits": [0]},
        {"gate": "ry", "qubits": [0], "angle": "2*x0"},
        {"gate": "cx", "qubits": [0, 1]},
        {"gate": "rx", "qubits": [1], "angle": "x0+x0"},
        {"gate": "rz", "qubits": [0], "angle": "x1"},
        {"gate": "rzz", "qubits": [0, 1], "angle": "2*x0"},
        {"gate": "ry", "qubits": [1], "angle": "x0*x1"},
    ]}


class MergedFrequencyTests(unittest.TestCase):
    def test_gates_sharing_an_angle_add_their_frequencies(self):
        self.assertEqual(merged_frequencies([[1.0], [1.0], [1.0]]), [1.0, 2.0, 3.0])
        self.assertEqual(merged_frequencies([[0.5], [1.0]]), [0.5, 1.0, 1.5])
        self.assertEqual(merged_frequencies([[1.0, 3.0]]), [1.0, 3.0])
        self.assertEqual(merged_frequencies([[]]), [])

    def test_oversized_merged_dictionary_is_refused(self):
        incommensurate = [[math.sqrt(prime)] for prime in (2, 3, 5, 7, 11, 13, 17)]
        with self.assertRaisesRegex(ValueError, "canonical representation"):
            merged_frequencies(incommensurate)


class MergedRepresentationTests(unittest.TestCase):
    def test_equal_preprocessing_shares_one_site_whatever_its_spelling(self):
        canonical = from_circuit(shared_angle_circuit())
        merged = from_circuit(shared_angle_circuit(), representation="merged")
        self.assertEqual(len(canonical.basis), 5)
        self.assertEqual([site["encoding_indices"] for site in merged.metadata["sites"]],
                         [[0, 1, 3], [2], [4]])
        self.assertEqual([site["gate_positions"] for site in merged.metadata["sites"]],
                         [[1, 3, 5], [4], [6]])
        self.assertEqual([basis.dimension for basis in merged.basis], [7, 3, 3])
        self.assertEqual(merged.metadata["sites"][0]["frequencies"], [1.0, 2.0, 3.0])
        self.assertEqual([item["site_index"] for item in merged.metadata["encodings"]], [0, 0, 1, 0, 2])
        self.assertEqual(merged.basis[0].input_indices, (0,))
        self.assertEqual(merged.basis[2].input_indices, (0, 1))
        self.assertEqual(merged.metadata["representation"], "merged")
        self.assertEqual(canonical.metadata["representation"], "canonical")

    def test_one_training_angle_drives_every_merged_gate(self):
        canonical = from_circuit(shared_angle_circuit())
        merged = from_circuit(shared_angle_circuit(), representation="merged")
        a, b, c = 0.37, -1.2, 2.1
        self.assertAlmostEqual(merged.auxiliary_oracle([a, b, c]),
                               canonical.auxiliary_oracle([a, a, b, a, c]), places=12)
        x = np.array([0.3, -0.8])
        self.assertAlmostEqual(merged.target(x), canonical.target(x), places=12)
        with self.assertRaises(ValueError):
            merged.auxiliary_oracle([a, a, b, a, c])

    def test_callable_preprocessing_ties_by_value(self):
        calls = []
        def oracle(angles):
            calls.append(tuple(angles))
            return math.cos(angles[0] + angles[1]) * math.cos(angles[2])
        problem = from_generators(
            [[-0.5, 0.5]] * 3, [lambda x: 3 * x[0], "3*x0", lambda x: x[1]], oracle,
            [[-1, 1], [-1, 1]], spectra=True, representation="merged",
            preprocessing_expressions=["3*x0", None, "x1"])
        self.assertEqual([site["encoding_indices"] for site in problem.metadata["sites"]], [[0, 1], [2]])
        problem.auxiliary_oracle([0.25, 0.5])
        self.assertEqual(calls[-1], (0.25, 0.25, 0.5))
        self.assertTrue(all(basis.portable for basis in problem.basis))

    def test_unknown_representation_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "canonical"):
            from_circuit(shared_angle_circuit(), representation="tied")

    @unittest.skipUnless(shutil.which("julia"), "Julia is not installed")
    def test_both_representations_learn_the_same_function(self):
        models = {name: fit(from_circuit(shared_angle_circuit(), representation=name),
                            max_bond=12, query_budget=20000, validation_samples=60, seed=3)
                  for name in ("canonical", "merged")}
        for model in models.values():
            self.assertLess(model.metadata["validation"]["rmse"], 1e-9)
        self.assertEqual(len(models["merged"].cores), 3)
        self.assertEqual(len(models["canonical"].cores), 5)
        x = [0.42, -1.3]
        self.assertAlmostEqual(models["merged"](x), models["canonical"](x), places=9)


class RepresentationServerTests(unittest.TestCase):
    def test_inspection_describes_every_representation(self):
        inspected = inspect_circuit(shared_angle_circuit())
        self.assertEqual(len(inspected["encodings"]), 5)
        canonical, merged = (inspected["representations"][name] for name in ("canonical", "merged"))
        self.assertEqual((canonical["length"], canonical["max_dimension"]), (5, 3))
        self.assertEqual((merged["length"], merged["dimensions"]), (3, [7, 3, 3]))
        self.assertAlmostEqual(10 ** merged["log10_basis_functions"], 63)
        self.assertEqual(merged["sites"][0]["gate_positions"], [1, 3, 5])
        self.assertEqual(len(merged["sites"][0]["basis"]), 7)
        self.assertEqual(merged["sites"][0]["input_indices"], [0])

    def test_fixed_circuit_has_no_sites_in_either_representation(self):
        spec = {"n_qubits": 1, "observable": "Z", "domain": [[0, 1]],
                "gates": [{"gate": "h", "qubits": [0]}]}
        for description in inspect_circuit(spec)["representations"].values():
            self.assertEqual((description["length"], description["sites"]), (0, []))

    def test_job_specification_selects_the_representation(self):
        spec = {"mode": "quantum", "name": "Shared", "domain": [[-1, 1], [-2, 2]],
                "circuit": shared_angle_circuit(), "representation": "merged"}
        problem, _ = problem_from_spec(spec)
        self.assertEqual(len(problem.basis), 3)
        self.assertEqual(problem.metadata["representation"], "merged")
        del spec["representation"]
        self.assertEqual(len(problem_from_spec(spec)[0].basis), 5)
        spec["representation"] = "quantics"
        with self.assertRaisesRegex(ValueError, "canonical, merged or custom"):
            problem_from_spec(spec)


if __name__ == "__main__":
    unittest.main()

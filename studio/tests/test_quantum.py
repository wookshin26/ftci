import importlib.util
import math
import unittest

import numpy as np

from ftci.quantum import from_circuit, from_generators, from_qiskit, generator_frequencies


class QuantumAdapterTests(unittest.TestCase):
    def test_generator_gaps_and_normalization(self):
        x_half = np.array([[0, 0.5], [0.5, 0]])
        self.assertEqual(generator_frequencies(x_half), [1.0])
        self.assertEqual(generator_frequencies([-0.5, -0.5, 0.5], spectrum=True), [1.0])
        problem = from_generators([x_half], ["x0**2"], lambda a: math.cos(a[0]), [[-1, 1]])
        np.testing.assert_allclose(problem.basis[0]([0.3]),
                                   np.array([1, math.cos(0.09), math.sin(0.09)]) / math.sqrt(2))
        self.assertAlmostEqual(problem.target([0.3]), math.cos(0.09))
        self.assertGreater(problem.auxiliary_domain[0][1], 2 * math.pi)

    def test_degenerate_generator_is_constant(self):
        problem = from_generators([np.eye(2)], ["x0"], lambda a: 0.25, [[-1, 1]])
        self.assertEqual(generator_frequencies(np.eye(2)), [])
        np.testing.assert_equal(problem.basis[0]([0.3]), [1.0])
        self.assertAlmostEqual(problem.target([0.3]), 0.25)

    def test_shared_input_is_lifted_per_occurrence(self):
        problem = from_circuit({"n_qubits": 1, "domain": [[-1, 1]], "observable": "Z",
                                "gates": [{"gate": "ry", "qubits": [0], "angle": "x0"},
                                          {"gate": "ry", "qubits": [0], "angle": "2*x0**2"}]})
        self.assertEqual(len(problem.basis), 2)
        self.assertEqual(problem.basis[0].input_indices, (0,))
        self.assertEqual(problem.basis[1].input_indices, (0,))
        self.assertEqual(problem.auxiliary_basis[0].input_indices, (0,))
        self.assertEqual(problem.auxiliary_basis[1].input_indices, (1,))
        self.assertAlmostEqual(problem.target([0.37]), math.cos(0.37 + 2 * 0.37**2))
        self.assertAlmostEqual(problem.auxiliary_oracle([0.2, 1.3]), math.cos(1.5))

    def test_little_endian_and_entanglement(self):
        problem = from_circuit({"n_qubits": 2, "domain": [[-1, 1]], "observable": "IZ",
                                "gates": [{"gate": "x", "qubits": [0]}]})
        self.assertAlmostEqual(problem.target([0]), -1)
        problem = from_circuit({"n_qubits": 2, "domain": [[-1, 1]], "observable": "ZI",
                                "gates": [{"gate": "x", "qubits": [0]}]})
        self.assertAlmostEqual(problem.target([0]), 1)
        bell = from_circuit({"n_qubits": 2, "domain": [[-1, 1]], "observable": "XX",
                             "gates": [{"gate": "h", "qubits": [0]},
                                       {"gate": "cx", "qubits": [0, 1]}]})
        self.assertAlmostEqual(bell.target([0]), 1)
        self.assertEqual(bell.metadata["encoding_occurrences"], 0)
        self.assertEqual(len(bell.basis), 1)

    def test_two_qubit_rotation_frequency(self):
        problem = from_circuit({"n_qubits": 2, "domain": [[-1, 1]], "observable": "IZ",
                                "gates": [{"gate": "rxx", "qubits": [0, 1], "angle": "x0"}]})
        for value in [0.0, 0.37, 1.0]:
            self.assertAlmostEqual(problem.target([value]), math.cos(value))

    def test_invalid_quantum_inputs_fail_before_simulation(self):
        with self.assertRaisesRegex(ValueError, "Hermitian"):
            generator_frequencies([[0, 1], [0, 0]])
        with self.assertRaisesRegex(ValueError, "finite"):
            generator_frequencies([np.nan, 1], spectrum=True)
        with self.assertRaisesRegex(ValueError, "oracle"):
            from_generators([[0, 1]], ["x0"], None, [[-1, 1]], spectra=True)
        with self.assertRaisesRegex(ValueError, "outside domain"):
            from_generators([[0, 1]], ["x1"], lambda a: 0, [[-1, 1]], spectra=True)
        with self.assertRaisesRegex(ValueError, "unsupported gate"):
            from_circuit({"n_qubits": 1, "domain": [[-1, 1]], "gates": [{"gate": "u", "qubits": [0]}]})
        with self.assertRaisesRegex(ValueError, "normalized"):
            from_circuit({"n_qubits": 1, "domain": [[-1, 1]], "gates": [], "initial_state": [1, 1]})

    def test_callable_preprocessing(self):
        problem = from_generators([[0, 1]], [lambda x: x[0] ** 2],
                                  lambda a: math.cos(a[0]), [[-1, 1]], spectra=True)
        self.assertAlmostEqual(problem.target([0.3]), math.cos(0.09))
        with self.assertRaises(ValueError):
            problem.basis[0].to_spec()
        portable = from_generators([[0, 1]], [lambda x: x[0] ** 2],
                                   lambda a: math.cos(a[0]), [[-1, 1], [-2, 2]], spectra=True,
                                   preprocessing_expressions=["x0**2"])
        self.assertEqual(len(portable.basis[0].to_spec()["expressions"]), 3)
        self.assertEqual(portable.basis[0].input_indices, (0,))


@unittest.skipUnless(importlib.util.find_spec("qiskit"), "optional Qiskit dependency not installed")
class QiskitAdapterTests(unittest.TestCase):
    def test_occurrence_lifting_and_fixed_parameters(self):
        from qiskit import QuantumCircuit
        from qiskit.circuit import Parameter
        x, theta = Parameter("feature"), Parameter("theta")
        circuit = QuantumCircuit(1)
        circuit.ry(x, 0)
        circuit.ry(2 * x**2, 0)
        circuit.ry(theta, 0)
        problem = from_qiskit(circuit, [x], [[-1, 1]], "Z", fixed_parameters={theta: 0.4})
        self.assertEqual(len(problem.basis), 2)
        self.assertAlmostEqual(problem.auxiliary_oracle([0.2, 1.3]), math.cos(1.9))
        self.assertAlmostEqual(problem.target([0.37]), math.cos(0.37 + 2 * 0.37**2 + 0.4))
        self.assertEqual(len(circuit.parameters), 2)  # Input circuit remains untouched.

    def test_qiskit_rejects_unsupported_encoding(self):
        from qiskit import QuantumCircuit
        from qiskit.circuit import Parameter
        x = Parameter("x")
        circuit = QuantumCircuit(2)
        circuit.crx(x, 0, 1)
        with self.assertRaisesRegex(ValueError, "unsupported"):
            from_qiskit(circuit, [x], [[-1, 1]], "ZZ")


if __name__ == "__main__":
    unittest.main()

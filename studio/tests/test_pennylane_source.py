"""The source reader never imports PennyLane or executes uploaded programs."""
from copy import deepcopy
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from ftci.pennylane_source import parse_pennylane, serialize_pennylane
from ftci.quantum import from_circuit


SOURCE = '''"""Keep this research note, including its Korean 설명."""
import pennylane as qml
from pennylane import numpy as np

dev = qml.device("default.qubit", wires=2)
offset = 0.25  # Calibration retained during visual edits.

@qml.qnode(dev, interface="autograd")
def signal(x):
    # Preparation and encoding notes must survive.
    qml.Hadamard(wires=0)
    qml.RZ(
        x[0],  # This encoding comment stays.
        wires=0,
    )
    qml.RY(np.sin(x[1]) + offset, wires=1)
    qml.CNOT(wires=[0, 1])  # Entangler.
    return qml.expval(qml.PauliY(0) @ qml.PauliY(wires=1))

domain = [[-np.pi, np.pi], [-1, 1]]
name = "Laboratory signal"
'''


def assert_same_function(case, a, b):
    case.assertEqual(a["n_qubits"], b["n_qubits"])
    case.assertEqual(a["domain"], b["domain"])
    case.assertEqual(a["observable"], b["observable"])
    original, edited = from_circuit(a), from_circuit(b)
    for x in np.random.default_rng(9).uniform(-0.8, 0.8, (8, len(a["domain"]))):
        case.assertAlmostEqual(original.target(x), edited.target(x), places=13)


class PennyLaneSourceTests(unittest.TestCase):
    def test_vector_inputs_tensor_observable_and_independent_matrix_equivalence(self):
        spec = parse_pennylane(SOURCE)
        self.assertEqual(spec["observable"], "YY")
        self.assertEqual(spec["domain"], [[-math.pi, math.pi], [-1.0, 1.0]])
        self.assertEqual([gate["gate"] for gate in spec["gates"]], ["h", "rz", "ry", "cx"])
        problem = from_circuit(spec)
        # Independent matrices in PennyLane's wire_order=[0,1] convention.
        identity = np.eye(2)
        h = np.array([[1, 1], [1, -1]]) / math.sqrt(2)
        y = np.array([[0, -1j], [1j, 0]])
        cnot = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
        for x in [[0.2, -0.4], [0.7, 0.3], [-0.6, 0.8]]:
            angle = math.sin(x[1]) + 0.25
            rz = np.diag([np.exp(-1j * x[0] / 2), np.exp(1j * x[0] / 2)])
            ry = np.array([[math.cos(angle / 2), -math.sin(angle / 2)],
                           [math.sin(angle / 2), math.cos(angle / 2)]])
            state = cnot @ np.kron(identity, ry) @ np.kron(rz, identity) @ np.kron(h, identity) @ np.array([1, 0, 0, 0])
            expected = float(np.vdot(state, np.kron(y, y) @ state).real)
            self.assertAlmostEqual(problem.target(x), expected, places=14)
            self.assertAlmostEqual(expected, -math.cos(x[0]) * math.cos(angle), places=14)

    def test_scalar_and_multiple_args_math_aliases_and_pauli_order(self):
        source = '''import pennylane as qp
import math as m
from math import cos as cosine
dev = qp.device("default.qubit", wires=3)
weights = [0.2, -0.7]
@qp.qnode(dev)
def circuit(a, b):
    theta = m.sin(a) + weights[0]
    qp.RX(theta, wires=0)
    qp.RY(cosine(b), 2)
    qp.PhaseShift(phi=m.pi/3, wires=1)
    return qp.expval(qp.X(2) @ qp.Z(0))
domain = [[-1, 1], [-2, 2]]
'''
        spec = parse_pennylane(source)
        self.assertEqual(spec["observable"], "XIZ")
        value = [0.3, 0.7]
        self.assertAlmostEqual(from_circuit(spec).target(value),
                               math.sin(math.cos(value[1])) * math.cos(math.sin(value[0]) + 0.2), places=14)

    def test_scalar_input_and_inline_device(self):
        source = '''import pennylane as qml
@qml.qnode(qml.device("default.qubit", wires=1))
def circuit(angle):
    qml.RY(angle**2, wires=0)
    return qml.expval(qml.PauliZ(0))
'''
        spec = parse_pennylane(source, [[-2, 2]])
        self.assertAlmostEqual(from_circuit(spec).target([0.7]), math.cos(0.49), places=14)

    def test_direct_imports_fixed_ising_gates_and_adjoint(self):
        source = '''from pennylane import device, qnode, Hadamard, IsingXX, S, adjoint, PauliZ, expval
dev = device("default.qubit", 2)
@qnode(dev)
def circuit(angle):
    Hadamard(0)
    adjoint(S)(wires=0)
    IsingXX(angle, wires=[0,1])
    return expval(PauliZ(wires=1))
'''
        spec = parse_pennylane(source)
        self.assertEqual([gate["gate"] for gate in spec["gates"]], ["h", "sdg", "rxx"])
        self.assertAlmostEqual(from_circuit(spec).target([0.4]), math.cos(0.4), places=14)
        edited = deepcopy(spec)
        edited["gates"][2]["angle"] = "2*x0"
        text = serialize_pennylane(edited, source)
        self.assertIn("from pennylane import device", text)
        self.assertIn("import pennylane as _ftci_qml", text)
        assert_same_function(self, edited, parse_pennylane(text))

    def test_fixed_state_preparation_maps_wire_order(self):
        basis = '''import pennylane as qml
dev = qml.device("default.qubit", wires=2)
@qml.qnode(dev)
def circuit(x):
    qml.BasisState([1,0], wires=[0,1])
    qml.RY(x, wires=1)
    return qml.expval(qml.PauliZ(0))
'''
        spec = parse_pennylane(basis)
        self.assertEqual(spec["initial_state"], "01")
        self.assertEqual(spec["observable"], "IZ")
        self.assertAlmostEqual(from_circuit(spec).target([0.4]), -1)
        vector = basis.replace("qml.BasisState([1,0], wires=[0,1])", "qml.StatePrep([0,1,0,0], wires=[0,1])")
        self.assertEqual(parse_pennylane(vector)["initial_state"], [0.0, 0.0, 1.0, 0.0])

    def test_canonical_roundtrip_and_initial_state(self):
        spec = parse_pennylane(SOURCE)
        spec["initial_state"] = "10"
        generated = serialize_pennylane(spec)
        self.assertIn("@qml.qnode(dev)", generated)
        self.assertIn("qml.BasisState([0, 1]", generated)
        assert_same_function(self, spec, parse_pennylane(generated))
        spec["initial_state"] = [0.0, 1.0, 0.0, 0.0]
        assert_same_function(self, spec, parse_pennylane(serialize_pennylane(spec)))

    def test_visual_edits_preserve_source_comments_and_scalar_assignments(self):
        spec = parse_pennylane(SOURCE)
        spec["gates"][1]["angle"] = "2*x0**2"
        spec["gates"].insert(2, {"gate": "rx", "qubits": [1], "angle": "sqrt(1+x1**2)"})
        del spec["gates"][0]
        spec["observable"] = "ZI"
        spec["domain"] = [[-2, 2], [-1, 1]]
        text = serialize_pennylane(spec, previous_source=SOURCE)
        self.assertIn('"""Keep this research note, including its Korean 설명."""', text)
        self.assertIn("offset = 0.25  # Calibration retained during visual edits.", text)
        self.assertIn("# This encoding comment stays.", text)
        self.assertIn("# Preparation and encoding notes must survive.", text)
        self.assertIn("# Entangler.", text)
        self.assertIn('@qml.qnode(dev, interface="autograd")', text)
        self.assertIn("def signal(x):", text)
        self.assertNotIn("qml.Hadamard(wires=0)", text)
        assert_same_function(self, spec, parse_pennylane(text))

    def test_deletions_leave_comments_and_same_line_statements_intact(self):
        source = '''import pennylane as qml
dev = qml.device("default.qubit", wires=1)
@qml.qnode(dev)
def circuit(x):
    qml.Hadamard(0); qml.RY(x, 0)  # Keep the complete line's note.
    return qml.expval(qml.Z(0))
'''
        spec = parse_pennylane(source)
        del spec["gates"][0]
        text = serialize_pennylane(spec, source)
        self.assertIn("pass; qml.RY(x, 0)", text)
        self.assertIn("# Keep the complete line's note.", text)
        assert_same_function(self, spec, parse_pennylane(text))

    def test_empty_circuit_can_add_gate_and_observable_without_losing_comments(self):
        source = '''import pennylane as qml
dev = qml.device("default.qubit", wires=1)
# An empty circuit is still a useful starting point.
@qml.qnode(dev)
def circuit(x):
    return qml.expval(qml.Identity(0))  # Keep this return comment.
'''
        spec = parse_pennylane(source)
        spec["gates"] = [{"gate": "ry", "qubits": [0], "angle": "sin(x0)"}]
        spec["observable"] = "Z"
        spec["initial_state"] = "1"
        text = serialize_pennylane(spec, source)
        self.assertIn("# An empty circuit is still a useful starting point.", text)
        self.assertIn("# Keep this return comment.", text)
        assert_same_function(self, spec, parse_pennylane(text))

    def test_vector_dimension_edit_and_scalar_declaration_rejection(self):
        spec = parse_pennylane(SOURCE)
        spec["domain"].append([-1, 1])
        spec["gates"][1]["angle"] = "x2"
        text = serialize_pennylane(spec, SOURCE)
        self.assertIn("x[2]", text)
        assert_same_function(self, spec, parse_pennylane(text))
        scalar = "import pennylane as qml\ndev=qml.device('default.qubit',wires=1)\n@qml.qnode(dev)\ndef f(a):\n    qml.RX(a,0)\n    return qml.expval(qml.Z(0))\n"
        spec = parse_pennylane(scalar)
        spec["domain"].append([-1, 1])
        spec["gates"][0]["angle"] = "x1"
        with self.assertRaisesRegex(ValueError, "Declare qnode input"):
            serialize_pennylane(spec, scalar)

    def test_metadata_conflicts_and_lossy_conversions_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "metadata must match"):
            parse_pennylane(SOURCE + "observable = 'ZI'\n")
        spec = parse_pennylane(SOURCE)
        spec["extra_note"] = "Do not lose this."
        with self.assertRaisesRegex(ValueError, "extra JSON metadata"):
            serialize_pennylane(spec)
        with self.assertRaisesRegex(ValueError, "state preparation"):
            parse_pennylane(SOURCE + "initial_state = '01'\n")
        # An explicit zero state is equivalent to native PennyLane defaults.
        self.assertEqual(parse_pennylane(SOURCE + "initial_state = [1,0,0,0]\n")["initial_state"], [1,0,0,0])

    def test_python_scope_and_name_collisions_are_rejected(self):
        for source in [SOURCE.replace("def signal(x):", "def qml(x):"),
                       SOURCE + "signal = 0.2\n", SOURCE + "import math as signal\n",
                       SOURCE.replace("def signal(x):", "def signal(x, x):"),
                       SOURCE.replace("def signal(x):", "def signal(x: __import__('os').system('echo unsafe')):"),
                       SOURCE.replace("    qml.Hadamard(wires=0)", "    qml.RY(offset*x[0], 0)\n    offset = 0.3")]:
            with self.subTest(source=source):
                with self.assertRaisesRegex(ValueError, "PennyLane line"):
                    parse_pennylane(source)

    def test_unsupported_and_malicious_statements_fail_with_line_numbers(self):
        replacements = [
            ("qml.Hadamard(wires=0)", "for i in range(2):\n        qml.Hadamard(wires=i)"),
            ("qml.Hadamard(wires=0)", "qml.CRX(x[0], wires=[0,1])"),
            ("return qml.expval(qml.PauliY(0) @ qml.PauliY(wires=1))", "return qml.probs(wires=[0,1])"),
            ("return qml.expval(qml.PauliY(0) @ qml.PauliY(wires=1))", "return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))"),
            ("return qml.expval(qml.PauliY(0) @ qml.PauliY(wires=1))", "return qml.expval(qml.PauliZ(0) @ qml.PauliX(0))"),
            ("return qml.expval(qml.PauliY(0) @ qml.PauliY(wires=1))", "return qml.expval(qml.Hermitian([[1,0],[0,-1]], wires=0))"),
            ("qml.RZ(\n        x[0],  # This encoding comment stays.\n        wires=0,\n    )", "qml.RX(__import__('os').system('echo unsafe'), wires=0)"),
            ("import pennylane as qml", "import os"),
        ]
        for before, after in replacements:
            with self.subTest(code=after):
                with self.assertRaisesRegex(ValueError, "PennyLane line"):
                    parse_pennylane(SOURCE.replace(before, after))
        with tempfile.TemporaryDirectory() as temporary:
            sentinel = Path(temporary) / "should-not-exist"
            malicious = SOURCE + f"open({str(sentinel)!r}, 'w').write('executed')\n"
            with self.assertRaisesRegex(ValueError, "PennyLane line"):
                parse_pennylane(malicious)
            self.assertFalse(sentinel.exists())

    def test_parser_never_imports_pennylane_and_checks_source_bounds(self):
        original = __import__
        def guarded_import(name, *args, **kwargs):
            if name == "pennylane" or name.startswith("pennylane."):
                raise AssertionError("The source reader tried to import PennyLane.")
            return original(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=guarded_import):
            spec = parse_pennylane(SOURCE)
            self.assertEqual(spec["n_qubits"], 2)
        for source in ["", "#" * 128001, "\x00"]:
            with self.subTest(source_size=len(source)):
                with self.assertRaises(ValueError):
                    parse_pennylane(source)


if __name__ == "__main__":
    unittest.main()

"""Package-independent Python upload and source-preserving edits."""
from copy import deepcopy
import tempfile
from pathlib import Path
import unittest

import numpy as np

from ftci.circuit_source import parse_source, serialize_source, source_info
from ftci.quantum import from_circuit


SPEC = {"n_qubits": 2, "domain": [[-1, 1], [-2, 2]], "observable": "ZZ",
        "gates": [{"gate": "h", "qubits": [0]},
                  {"gate": "ry", "qubits": [1], "angle": "x0+x1**2"},
                  {"gate": "cx", "qubits": [0, 1]}]}


class PythonSourceTests(unittest.TestCase):
    def test_any_package_context_is_not_imported_or_executed(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "must-not-exist"
            source = ("import a_quantum_package_that_is_not_installed\n"
                      f"open({str(marker)!r}, 'w').write('unsafe')\n"
                      "raise RuntimeError('not executed')\n"
                      f"ftci_circuit = {SPEC!r}\n")
            parsed = parse_source("python", source)
            self.assertEqual(parsed, SPEC)
            self.assertFalse(marker.exists())
            self.assertEqual(source_info("python", source)["kind"], "portable")
            np.testing.assert_allclose(from_circuit(parsed).target([0.2, 0.3]),
                                       from_circuit(SPEC).target([0.2, 0.3]))

    def test_visual_edits_preserve_surrounding_python_and_comments(self):
        source = ("# 원본 회로\nimport another_package as qp\n"
                  "ftci_circuit = {\n    # encoding note\n" + repr(SPEC)[1:] +
                  "  # keep after declaration\n\n"
                  "def build_native_circuit():\n    return qp.build()\n")
        wanted = deepcopy(SPEC)
        wanted["gates"][1]["angle"] = "2*x0**2+x1"
        edited = serialize_source("python", wanted, previous_source=source)
        self.assertIn("# 원본 회로\nimport another_package as qp", edited)
        self.assertIn("# encoding note", edited)
        self.assertIn("# keep after declaration", edited)
        self.assertTrue(edited.endswith("def build_native_circuit():\n    return qp.build()\n"))
        self.assertEqual(parse_source("python", edited), wanted)

    def test_neutral_python_roundtrip_preserves_extra_metadata(self):
        spec = deepcopy(SPEC)
        spec["project"] = {"owner": "researcher", "enabled": True, "notes": None}
        text = serialize_source("python", spec)
        self.assertIn("ftci_circuit =", text)
        self.assertNotIn("qiskit", text)
        self.assertEqual(parse_source("python", text), spec)
        annotated = text.replace("ftci_circuit =", "ftci_circuit: dict =")
        self.assertEqual(parse_source("python", annotated), spec)

    def test_dynamic_ambiguous_or_mutated_specifications_are_rejected(self):
        sources = ["ftci_circuit = build_circuit()", "ftci_circuit = dict(n_qubits=1)",
                   f"ftci_circuit = {SPEC!r}\nftci_circuit = {SPEC!r}",
                   f"def factory():\n    ftci_circuit = {SPEC!r}",
                   f"ftci_circuit = {SPEC!r}\nftci_circuit['gates'].append({{'gate':'x','qubits':[0]}})",
                   f"ftci_circuit = {SPEC!r}\nftci_circuit['observable'] = 'XX'",
                   f"ftci_circuit = alias = {SPEC!r}",
                   f"ftci_circuit = {SPEC!r}\nalias = ftci_circuit\nalias['gates'].clear()",
                   f"ftci_circuit = {SPEC!r}\nmutate(ftci_circuit)",
                   f"ftci_circuit = {SPEC!r}\ndel ftci_circuit",
                   f"ftci_circuit = {SPEC!r}\nfrom some_package import ftci_circuit",
                   f"ftci_circuit = {SPEC!r}\ndef ftci_circuit(): pass"]
        for source in sources:
            with self.subTest(source=source), self.assertRaises(ValueError):
                parse_source("python", source)

    def test_unknown_python_has_an_actionable_bridge_error(self):
        with self.assertRaisesRegex(ValueError, "ftci_circuit"):
            parse_source("python", "import some_package\ncircuit = some_package.make()\n")

    def test_qiskit_is_auto_detected_and_still_roundtrips(self):
        source = "from qiskit import QuantumCircuit as QC\nqc=QC(1)\nqc.h(0)\nqc.ry(0.2,0)\n"
        spec = parse_source("python", source)
        self.assertEqual(source_info("python", source)["kind"], "qiskit")
        wanted = deepcopy(spec)
        wanted["gates"][1]["angle"] = "0.7"
        edited = serialize_source("python", wanted, source)
        self.assertTrue(edited.startswith("from qiskit import QuantumCircuit as QC"))
        self.assertEqual(parse_source("python", edited), wanted)

    def test_native_examples_are_detected_and_visual_edits_preserve_dialect(self):
        examples = Path(__file__).resolve().parents[1] / "examples"
        for kind in ("cirq", "pennylane"):
            with self.subTest(kind=kind):
                source = (examples / f"circuit_{kind}.py").read_text()
                parsed = parse_source("python", source)
                self.assertEqual(source_info("python", source)["kind"], kind)
                self.assertEqual(parsed, parse_source(kind, source))
                desired = deepcopy(parsed)
                angle_gate = next(gate for gate in desired["gates"] if "angle" in gate)
                angle_gate["angle"] = "2*x0**2+x1"
                edited = serialize_source("python", desired, previous_source=source)
                self.assertEqual(source_info("python", edited)["kind"], kind)
                actual = from_circuit(parse_source("python", edited))
                expected = from_circuit(desired)
                for point in ([0.2, 0.3], [-0.4, 0.7]):
                    self.assertAlmostEqual(actual.target(point), expected.target(point), places=13)

    def test_mixed_native_packages_require_an_explicit_specification(self):
        source = ("import qiskit\nimport cirq\na=qiskit.QuantumCircuit(1)\n"
                  "b=cirq.Circuit(cirq.X(cirq.LineQubit(0)))\n")
        with self.assertRaisesRegex(ValueError, "Multiple Python"):
            parse_source("python", source)


if __name__ == "__main__":
    unittest.main()

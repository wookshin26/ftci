from copy import deepcopy
import importlib.util
import unittest

import numpy as np

from ftci.circuit_source import parse_source, serialize_source
from ftci.quantum import from_circuit


SOURCE = """# My circuit and preprocessing stay in code.
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
import numpy as np

x = ParameterVector('features', 2)
scale = 2
qc = QuantumCircuit(2)
qc.h(0)  # prepare q0
qc.ry(x[0], 0)
# A nonlinear encoding on the other qubit.
qc.rz(scale*x[0]**2+x[1], 1)  # preserve this comment
qc.cx(0, 1)
qc.rx(np.sin(x[0])+x[1], 0)

observable = 'ZZ'
domain = [[-np.pi, np.pi], [-1, 1]]
"""


def assert_same_function(test, a, b):
    qa, qb = from_circuit(a), from_circuit(b)
    rng = np.random.default_rng(43)
    for x in rng.uniform(-0.8, 0.8, (12, len(a["domain"]))):
        test.assertAlmostEqual(qa.target(x), qb.target(x), places=12)


class CircuitSourceTests(unittest.TestCase):
    def test_qiskit_shared_nonlinear_inputs(self):
        circuit = parse_source("qiskit", SOURCE)
        self.assertEqual(circuit["n_qubits"], 2)
        self.assertEqual(circuit["gates"][2]["gate"], "rz")
        problem = from_circuit(circuit)
        self.assertEqual(problem.basis[1].input_indices, (0, 1))
        self.assertEqual(problem.basis[0].input_indices, (0,))
        self.assertEqual(circuit["domain"][0], [-np.pi, np.pi])

    def test_json_roundtrip_and_metadata(self):
        spec = parse_source("qiskit", SOURCE)
        spec["note"] = {"researcher": "keep this metadata"}
        text = serialize_source("json", spec)
        restored = parse_source("json", text)
        self.assertEqual(restored, spec)
        with self.assertRaisesRegex(ValueError, "extra JSON metadata"):
            serialize_source("qiskit", spec)

    def test_canonical_qiskit_roundtrip(self):
        original = parse_source("qiskit", SOURCE)
        original["name"] = "A named circuit"
        text = serialize_source("qiskit", original)
        restored = parse_source("qiskit", text)
        assert_same_function(self, original, restored)
        self.assertEqual(original["domain"], restored["domain"])
        self.assertEqual(original["name"], restored["name"])

    def test_visual_edit_preserves_code_and_comments(self):
        circuit = parse_source("qiskit", SOURCE)
        circuit["gates"][2]["angle"] = "3*x0**2-x1"
        text = serialize_source("qiskit", circuit, previous_source=SOURCE)
        for retained in ("# My circuit", "scale = 2", "# prepare q0", "# A nonlinear encoding", "# preserve this comment", "ParameterVector('features', 2)"):
            self.assertIn(retained, text)
        restored = parse_source("qiskit", text)
        assert_same_function(self, circuit, restored)
        self.assertNotEqual(from_circuit(restored).target([0.4, 0.3]), from_circuit(parse_source("qiskit", SOURCE)).target([0.4, 0.3]))

    def test_insert_delete_reorder_remains_valid_source(self):
        original = parse_source("qiskit", SOURCE)
        for operation in ("insert", "delete", "reverse", "empty"):
            with self.subTest(operation=operation):
                circuit = deepcopy(original)
                if operation == "insert":
                    circuit["gates"].insert(2, {"gate": "rzz", "angle": "x0*x1", "qubits": [1, 0]})
                elif operation == "delete":
                    circuit["gates"].pop(1)
                elif operation == "reverse":
                    circuit["gates"].reverse()
                else:
                    circuit["gates"] = []
                text = serialize_source("qiskit", circuit, previous_source=SOURCE)
                restored = parse_source("qiskit", text)
                assert_same_function(self, circuit, restored)
                self.assertIn("# preserve this comment", text)

    def test_semicolon_statements_and_empty_circuit(self):
        source = "from qiskit import QuantumCircuit\nqc = QuantumCircuit(2)\nqc.h(0); qc.x(1)  # keep\nobservable='ZZ'\n"
        circuit = parse_source("qiskit", source)
        circuit["gates"] = [{"gate": "z", "qubits": [1]}]
        text = serialize_source("qiskit", circuit, previous_source=source)
        assert_same_function(self, circuit, parse_source("qiskit", text))
        self.assertIn("# keep", text)
        empty = parse_source("qiskit", "from qiskit import QuantumCircuit\nqc=QuantumCircuit(1)\n")
        empty["gates"] = [{"gate": "ry", "qubits": [0], "angle": "x0"}]
        # No Parameter was declared in this original source: do not invent a
        # reference and silently replace the user's code.
        with self.assertRaisesRegex(ValueError, "Declare input"):
            serialize_source("qiskit", empty, previous_source="from qiskit import QuantumCircuit\nqc=QuantumCircuit(1)\n")

    def test_qubits_domain_and_initial_state_updates(self):
        source = SOURCE + "initial_state = '00'\n"
        circuit = parse_source("qiskit", source)
        circuit["n_qubits"] = 3
        circuit["observable"] = "IZZ"
        circuit["initial_state"] = "000"
        circuit["domain"] = [[-2, 2], [-3, 3], [-4, 4]]
        text = serialize_source("qiskit", circuit, previous_source=source)
        restored = parse_source("qiskit", text)
        self.assertIn("ParameterVector('features', 3)", text)
        self.assertEqual(restored["n_qubits"], 3)
        self.assertEqual(restored["observable"], "IZZ")
        self.assertEqual(restored["domain"], circuit["domain"])
        assert_same_function(self, circuit, restored)

    def test_scalar_parameters_and_fixed_bindings(self):
        source = """from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
feature = Parameter('feature')
theta = Parameter('theta')
qc = QuantumCircuit(1)
qc.ry(feature, 0)
qc.rx(theta, 0)
qc.assign_parameters({theta: 0.7}, inplace=True)
observable='Z'
domain=[[-1,1]]
"""
        circuit = parse_source("qiskit", source)
        problem = from_circuit(circuit)
        self.assertEqual(problem.metadata["encoding_occurrences"], 1)
        self.assertAlmostEqual(problem.target([0.4]), np.cos(0.4)*np.cos(0.7))
        circuit["gates"][0]["angle"] = "2*x0"
        text = serialize_source("qiskit", circuit, previous_source=source)
        self.assertIn("Parameter('feature')", text)
        assert_same_function(self, circuit, parse_source("qiskit", text))
        circuit["gates"][1]["angle"] = "0.9"
        with self.assertRaisesRegex(ValueError, "no longer used"):
            serialize_source("qiskit", circuit, previous_source=source)

    def test_aliases_parameter_methods_and_fixed_scalars(self):
        source = """import qiskit as q
from qiskit.circuit import ParameterVector as PV
from math import pi
v=PV('v',2)
fixed=pi/3
qc=q.QuantumCircuit(2)
qc.ry(v[0].sin()+v[1],0)
qc.rz(fixed,1)
qc.cnot(0,1)
"""
        circuit = parse_source("qiskit", source)
        self.assertEqual(circuit["gates"][2]["gate"], "cx")
        self.assertEqual(from_circuit(circuit).metadata["encoding_occurrences"], 1)

    def test_unsupported_python_is_never_executed(self):
        for source in (
            "import os\nos.system('echo unsafe')",
            "from qiskit import QuantumCircuit\nqc=QuantumCircuit(1)\nopen('/tmp/ftci-unsafe','w')",
            "from qiskit import QuantumCircuit\nqc=QuantumCircuit(1)\nfor i in range(4): qc.x(0)",
            "from qiskit import QuantumCircuit\nqc=QuantumCircuit(1)\nqc.measure_all()",
            "from qiskit import QuantumCircuit\nqc=QuantumCircuit(1)\nqc.rx(__import__('os').system('echo unsafe'),0)",
            "from qiskit import QuantumCircuit\nqc=QuantumCircuit(1)\nqc.rx(x0,0)",
            "from qiskit import QuantumCircuit\nqc=QuantumCircuit(1)\nqc.x(True)",
            "from qiskit import QuantumCircuit\nfrom qiskit.circuit import Parameter\np=Parameter('p')\nqc=QuantumCircuit(1)\nqc.ry(p,0)\nqc.assign_parameters({p:0.7},inplace=True)\nqc.rx(p,0)",
        ):
            with self.subTest(source=source):
                with self.assertRaises(ValueError):
                    parse_source("qiskit", source)
        with self.assertRaisesRegex(ValueError, "JSON line"):
            parse_source("json", '{"n_qubits":')
        with self.assertRaises(ValueError):
            parse_source("json", '{"n_qubits":1,"gates":[],"domain":[[0,Infinity]]}')
        with self.assertRaises(ValueError):
            parse_source("json", "["*2000+"0"+"]"*2000)
        with self.assertRaises(ValueError):
            parse_source("qiskit", "\x00")

    @unittest.skipUnless(importlib.util.find_spec("qiskit"), "Qiskit is optional")
    def test_trusted_generated_source_builds_real_qiskit(self):
        # Only this fixed test fixture's generated code is executed. The
        # product parser/serializer never executes user-provided source.
        from qiskit.quantum_info import Pauli, Statevector
        spec = {"n_qubits": 2, "domain": [[-1, 1], [-1, 1]], "observable": "ZZ",
                "gates": [{"gate": "h", "qubits": [0]},
                          {"gate": "ry", "qubits": [0], "angle": "2*x0**2+x1"},
                          {"gate": "cx", "qubits": [0, 1]},
                          {"gate": "rx", "qubits": [1], "angle": "sin(x0)+x1"}]}
        generated = serialize_source("qiskit", spec)
        namespace = {}
        exec(compile(generated, "<trusted-generated-test-fixture>", "exec"), namespace)
        circuit, parameters = namespace["qc"], namespace["x"]
        oracle = from_circuit(spec).target
        for point in ([0.2, -0.3], [0.7, 0.1]):
            bound = circuit.assign_parameters({parameters[i]: point[i] for i in range(2)})
            actual = Statevector.from_instruction(bound).expectation_value(Pauli(spec["observable"])).real
            self.assertAlmostEqual(actual, oracle(point), places=12)
        unsupported = deepcopy(spec)
        unsupported["gates"][-1]["angle"] = "sqrt(x0)"
        with self.assertRaisesRegex(ValueError, "symbolic inputs"):
            serialize_source("qiskit", unsupported)
        with self.assertRaisesRegex(ValueError, "symbolic inputs"):
            parse_source("qiskit", "from qiskit import QuantumCircuit\nfrom qiskit.circuit import Parameter\nimport numpy as np\na=Parameter('a')\nqc=QuantumCircuit(1)\nqc.rx(np.sqrt(a),0)")


if __name__ == "__main__":
    unittest.main()

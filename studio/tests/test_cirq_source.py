from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest

import numpy as np

from ftci.cirq_source import parse_cirq, serialize_cirq
from ftci.quantum import from_circuit


SOURCE = """# Keep my native circuit source.
import cirq as cq
import sympy as sy

x = sy.symbols('feature0:2')
q = cq.LineQubit.range(2)
scale = 2
phase = scale*x[0]**2+x[1]
circuit = cq.Circuit(
    cq.H(q[0]),  # preparation
    cq.ry(x[0])(q[0]),
    cq.CNOT(q[0], q[1]),
)
# The shared input occurs in two different preprocessing functions.
circuit.append(cq.rz(phase).on(q[1]))  # retain this comment
circuit.append(cq.rx(sy.sin(x[0])+x[1])(q[0]))
domain = [[-sy.pi, sy.pi], [-1, 1]]
observable = 'ZZ'
initial_state = '00'
name = 'My native source'
"""


def assert_same_function(test, a, b):
    qa, qb = from_circuit(a), from_circuit(b)
    test.assertEqual(a["domain"], b["domain"])
    rng = np.random.default_rng(743)
    for x in rng.uniform(-0.8, 0.8, (16, len(a["domain"]))):
        test.assertAlmostEqual(qa.target(x), qb.target(x), places=12)


class CirqSourceTests(unittest.TestCase):
    def test_shared_nonlinear_inputs_and_function(self):
        circuit = parse_cirq(SOURCE)
        expected = {"n_qubits": 2, "domain": [[-np.pi, np.pi], [-1, 1]], "observable": "ZZ",
                    "initial_state": "00", "gates": [{"gate": "h", "qubits": [0]},
                        {"gate": "ry", "qubits": [0], "angle": "x0"},
                        {"gate": "cx", "qubits": [0, 1]},
                        {"gate": "rz", "qubits": [1], "angle": "2*x0**2+x1"},
                        {"gate": "rx", "qubits": [0], "angle": "sin(x0)+x1"}]}
        self.assertEqual(circuit["n_qubits"], 2)
        self.assertEqual(from_circuit(circuit).basis[1].input_indices, (0, 1))
        assert_same_function(self, expected, circuit)

    def test_symbol_declaration_order_and_repeated_names(self):
        source = """from cirq import Circuit, LineQubit, ry, rz
from sympy import Symbol, symbols, sin
b = Symbol('not_x0')
a = Symbol('not_x1')
again = Symbol('not_x0')
qubit = LineQubit(0)
c = Circuit(ry(b)(qubit), rz(a+sin(again))(qubit))
"""
        circuit = parse_cirq(source)
        self.assertEqual(len(circuit["domain"]), 2)
        self.assertEqual(circuit["gates"][0]["angle"], "x0")
        self.assertEqual(from_circuit(circuit).basis[1].input_indices, (0, 1))
        self.assertIn("x1", circuit["gates"][1]["angle"])
        with self.assertRaisesRegex(ValueError, "domain"):
            parse_cirq(source, [[-1, 1]])

    def test_tuple_declarations_radians_and_fixed_gates(self):
        source = """import cirq
from sympy import symbols
from math import pi
a, b = symbols('a b')
left, right = cirq.LineQubit.range(2)
constant = pi/3
circuit = cirq.Circuit([cirq.H.on(left), cirq.X(right), cirq.Y(left), cirq.Z(right), cirq.S(left), cirq.T(right)])
circuit.append([cirq.CZ(left, right), [cirq.SWAP(left, right), cirq.rx(rads=constant)(left), cirq.ry(a+b).on(right)]])
observable = 'ZI'
"""
        circuit = parse_cirq(source)
        self.assertEqual([g["gate"] for g in circuit["gates"]], ["h", "x", "y", "z", "s", "t", "cz", "swap", "rx", "ry"])
        self.assertEqual(from_circuit(circuit).metadata["encoding_occurrences"], 1)
        simple = parse_cirq("import cirq\nq=cirq.LineQubit(0)\nc=cirq.Circuit(cirq.ry(0.8)(q))\n")
        self.assertAlmostEqual(from_circuit(simple).target([0]), np.cos(0.8), places=12)

    def test_real_symbols_size_variables_and_richer_preprocessing(self):
        source = """import cirq
import sympy as sp
n_qubits=2
q=cirq.LineQubit.range(n_qubits)
left=q[0]
right=q[1]
x=sp.symbols('x0:2', real=True)
c=cirq.Circuit(cirq.ry(sp.sqrt(x[0]**2+1)+sp.Abs(x[1]))(left), cirq.rx(sp.Min(x[0],x[1]))(right))
"""
        spec = parse_cirq(source)
        self.assertEqual(spec["n_qubits"], 2)
        assert_same_function(self, spec, parse_cirq(serialize_cirq(spec)))
        spec["gates"][0]["angle"] = "log1p(x0**2)+expm1(x1)"
        assert_same_function(self, spec, parse_cirq(serialize_cirq(spec)))

    def test_canonical_roundtrip_and_example(self):
        spec = parse_cirq(SOURCE)
        generated = serialize_cirq(spec)
        restored = parse_cirq(generated)
        assert_same_function(self, spec, restored)
        self.assertEqual(spec["name"], restored["name"])
        example = Path(__file__).resolve().parents[1] / "examples" / "circuit_cirq.py"
        self.assertEqual(parse_cirq(example.read_text())["n_qubits"], 2)

    def test_visual_edit_preserves_native_source_and_comments(self):
        spec = parse_cirq(SOURCE)
        spec["gates"][3]["angle"] = "3*x0**2-x1"
        updated = serialize_cirq(spec, SOURCE)
        for fragment in ("# Keep my native", "import cirq as cq", "import sympy as sy", "scale = 2", "phase = scale*x[0]**2+x[1]", "# preparation", "# retain this comment", "# The shared input", "sy.symbols('feature0:2')"):
            self.assertIn(fragment, updated)
        assert_same_function(self, spec, parse_cirq(updated))

    def test_insert_delete_reorder_and_empty(self):
        original = parse_cirq(SOURCE)
        for change in ("insert", "delete", "reverse", "empty"):
            with self.subTest(change=change):
                spec = deepcopy(original)
                if change == "insert":
                    spec["gates"].insert(2, {"gate": "rx", "qubits": [1], "angle": "x0*x1"})
                elif change == "delete":
                    spec["gates"].pop(0)
                elif change == "reverse":
                    spec["gates"].reverse()
                else:
                    spec["gates"] = []
                updated = serialize_cirq(spec, SOURCE)
                assert_same_function(self, spec, parse_cirq(updated))
                self.assertIn("# retain this comment", updated)

    def test_metadata_and_vector_sizes_update(self):
        spec = parse_cirq(SOURCE)
        spec["n_qubits"] = 3
        spec["domain"] = [[-2, 2], [-3, 3], [-4, 4]]
        spec["initial_state"] = "000"
        spec["observable"] = "IZZ"
        spec["gates"].append({"gate": "ry", "qubits": [2], "angle": "x2"})
        updated = serialize_cirq(spec, SOURCE)
        self.assertIn("sy.symbols('feature0:3')", updated)
        self.assertIn("cq.LineQubit.range(3)", updated)
        assert_same_function(self, spec, parse_cirq(updated))

    def test_from_imports_and_internal_comments_survive(self):
        source = """from cirq import Circuit, LineQubit, ry
from sympy import Symbol
feature=Symbol('feature')
q=LineQubit(0)
native=Circuit(ry(
    feature  # angle commentary
)(q))
"""
        spec = parse_cirq(source)
        spec["gates"][0]["angle"] = "sin(x0)"
        updated = serialize_cirq(spec, source)
        self.assertIn("# angle commentary", updated)
        self.assertIn("from cirq import Circuit, LineQubit, ry", updated)
        self.assertIn("from sympy import Symbol", updated)
        assert_same_function(self, spec, parse_cirq(updated))
        source = "from cirq import Circuit, LineQubit, H\nimport numpy as _ftci_cirq\nq=LineQubit(0)\nnative=Circuit(H(q))\nlater_alias=q\n"
        spec = parse_cirq(source)
        spec["gates"][0]["gate"] = "x"
        updated = serialize_cirq(spec, source)
        self.assertIn("later_alias=q", updated)
        assert_same_function(self, spec, parse_cirq(updated))

    def test_ambiguous_reused_operations_rejected_for_graphic_edit(self):
        source = "import cirq\nq=cirq.LineQubit(0)\nop=cirq.H(q)\ncircuit=cirq.Circuit(op, op)\n"
        spec = parse_cirq(source)
        self.assertEqual(len(spec["gates"]), 2)
        spec["gates"][0]["gate"] = "x"
        with self.assertRaisesRegex(ValueError, "reuses operation"):
            serialize_cirq(spec, source)
        with self.assertRaisesRegex(ValueError, "extra JSON metadata"):
            serialize_cirq({**spec, "note": "preserve me"})
        spec["gates"] = [{"gate": "rxx", "qubits": [0, 1], "angle": "x0"}]
        spec["n_qubits"] = 2
        spec["observable"] = "ZZ"
        with self.assertRaisesRegex(ValueError, "outside the native Cirq"):
            serialize_cirq(spec)

    def test_untrusted_or_unsupported_python_is_never_executed(self):
        prefix = "import cirq\nimport sympy as sp\nx=sp.Symbol('x')\nq=cirq.LineQubit(0)\ncircuit=cirq.Circuit()\n"
        for suffix in (
            "import os\nos.system('echo unsafe')", "open('/tmp/ftci-cirq-unsafe','w')",
            "for i in range(2): circuit.append(cirq.H(q))", "def function(): return 1",
            "circuit.append(cirq.measure(q))", "circuit.append(cirq.X(q)**x)",
            "circuit.append(cirq.rx(x)(q).controlled_by(q))", "circuit.append(cirq.XPowGate(exponent=x)(q))",
            "circuit.append(cirq.rx(__import__('os').system('echo unsafe'))(q))",
            "circuit.append(cirq.rx(1e999)(q))", "q = cirq.LineQubit(1)",
            "circuit.append(cirq.CNOT(q,q))", "circuit.append(cirq.H(True))",
            "import numpy as np\ncircuit.append(cirq.rx(np.sin(x))(q))",
            "circuit.append(cirq.rx(sp.e)(q))", "import math\ncircuit.append(cirq.rx(math.arcsin(0.2))(q))",
            "circuit.append(cirq.rx(undefined)(q))",
        ):
            with self.subTest(suffix=suffix):
                with self.assertRaises(ValueError):
                    parse_cirq(prefix+suffix)
        for source in ("\0", "["*3000+"0"+"]"*3000, "#"*128001):
            with self.assertRaises(ValueError):
                parse_cirq(source)

    @unittest.skipUnless(importlib.util.find_spec("cirq") and importlib.util.find_spec("sympy"), "Cirq and SymPy are optional")
    def test_trusted_generated_code_matches_real_cirq(self):
        # Product code never executes uploaded text. This fixed test fixture
        # alone verifies generated syntax against the optional actual SDK.
        import cirq
        spec = {"n_qubits": 2, "domain": [[-1, 1], [-1, 1]], "observable": "XZ",
                "gates": [{"gate": "h", "qubits": [0]},
                          {"gate": "ry", "qubits": [0], "angle": "2*x0**2+x1"},
                          {"gate": "cx", "qubits": [0, 1]},
                          {"gate": "rx", "qubits": [1], "angle": "sin(x0)+x1"}]}
        namespace = {}
        exec(compile(serialize_cirq(spec), "<trusted-generated-cirq-fixture>", "exec"), namespace)
        observable = cirq.PauliString({namespace["q"][0]: cirq.Z, namespace["q"][1]: cirq.X})
        simulator = cirq.Simulator(dtype=np.complex128)
        for point in ([0.3, -0.2], [0.7, 0.6]):
            # Reverse qubit_order because FTCI strings are little endian.
            actual = simulator.simulate_expectation_values(namespace["circuit"], observables=[observable],
                        param_resolver={namespace["x"][i]: point[i] for i in range(2)},
                        qubit_order=list(reversed(namespace["q"])))[0].real
            self.assertAlmostEqual(actual, from_circuit(spec).target(point), places=12)


if __name__ == "__main__":
    unittest.main()

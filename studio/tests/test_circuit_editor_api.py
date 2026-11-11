"""Behavioral checks for diagram-edited circuits and live encoding inspection."""

from copy import deepcopy
from email.message import Message
import io
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace
import unittest

import numpy as np

from ftci import fit
from ftci.engine import engine_info
from ftci.expressions import Expression
from ftci.quantum import from_circuit
from ftci.server import inspect_circuit, make_handler


def edited_circuit():
    return {
        "n_qubits": 1,
        "observable": "Z",
        "domain": [[-1.5, 1.5], [-1, 1]],
        "gates": [
            {"gate": "ry", "qubits": [0], "angle": "x0"},
            {"gate": "ry", "qubits": [0], "angle": "2*x0**2+x1"},
        ],
    }


class CircuitInspectionTests(unittest.TestCase):
    def test_edited_preprocessing_maps_inputs_and_normalized_basis(self):
        inspected = inspect_circuit(edited_circuit())
        self.assertEqual(inspected["encoding_occurrences"], 2)
        first, second = inspected["encodings"]
        self.assertEqual([first["gate_position"], second["gate_position"]], [0, 1])
        self.assertEqual([first["encoding_index"], second["encoding_index"]], [0, 1])
        self.assertEqual(first["input_indices"], [0])
        self.assertEqual(second["input_indices"], [0, 1])
        self.assertEqual(second["preprocessing"], "2*x0**2+x1")
        self.assertEqual(second["frequencies"], [1.0])
        self.assertEqual(second["dimension"], 3)
        x = [0.4, -0.2]
        values = np.array([Expression(text)(x) for text in second["basis"]])
        phi = 2*x[0]**2+x[1]
        np.testing.assert_allclose(values, np.array([1, np.cos(phi), np.sin(phi)])/np.sqrt(2))
        self.assertAlmostEqual(float(values @ values), 1.)

    def test_shared_inputs_keep_independent_encoding_occurrences(self):
        spec = edited_circuit()
        spec["gates"][1]["angle"] = "x0"
        inspected = inspect_circuit(spec)
        self.assertEqual([e["input_indices"] for e in inspected["encodings"]], [[0], [0]])
        problem = from_circuit(spec)
        self.assertAlmostEqual(problem.auxiliary_oracle([0.3, -0.7]), np.cos(0.3-0.7))
        self.assertAlmostEqual(problem.target([0.2, 0.8]), np.cos(0.4))

    def test_fixed_rotations_excluded_and_occurrence_positions_preserved(self):
        spec = edited_circuit()
        spec["gates"].insert(1, {"gate": "rz", "qubits": [0], "angle": "pi/7"})
        inspected = inspect_circuit(spec)
        self.assertEqual(inspected["encoding_occurrences"], 2)
        self.assertEqual([e["gate_position"] for e in inspected["encodings"]], [0, 2])
        self.assertEqual([e["encoding_index"] for e in inspected["encodings"]], [0, 1])
        spec["gates"] = [{"gate": "ry", "qubits": [0], "angle": 0.7}]
        fixed = inspect_circuit(spec)
        self.assertEqual(fixed["encoding_occurrences"], 0)
        self.assertEqual(fixed["encodings"], [])

    def test_phase_generator_and_control_target_order(self):
        spec = edited_circuit()
        spec["gates"] = [{"gate": "p", "qubits": [0], "angle": "x1"}]
        phase = inspect_circuit(spec)["encodings"][0]
        self.assertEqual(phase["input_indices"], [1])
        self.assertEqual(phase["frequencies"], [1.0])
        controlled = {"n_qubits": 2, "observable": "ZI", "domain": [[-1, 1]],
                      "gates": [{"gate": "x", "qubits": [1]},
                                {"gate": "cx", "qubits": [1, 0]}]}
        self.assertAlmostEqual(from_circuit(controlled).target([0.2]), -1.)
        controlled["observable"] = "IZ"
        self.assertAlmostEqual(from_circuit(controlled).target([0.2]), -1.)

    def test_invalid_edits_rejected_without_executing_expressions(self):
        for replacement in (
            {"gate": "ry", "qubits": [0], "angle": "x2"},
            {"gate": "ry", "qubits": [1], "angle": "x0"},
            {"gate": "unsupported", "qubits": [0]},
            {"gate": "ry", "qubits": [0], "angle": "sin("},
            {"gate": "ry", "qubits": [0], "angle": "__import__('os').getcwd()"},
        ):
            with self.subTest(replacement=replacement):
                spec = edited_circuit()
                spec["gates"][0] = replacement
                with self.assertRaises(ValueError):
                    inspect_circuit(spec)
        for invalid in (None, [], {"n_qubits": 1, "domain": [[-1, 1]], "gates": "bad"}):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    inspect_circuit(invalid)

    def test_inspection_is_json_safe_and_does_not_mutate_editor_spec(self):
        spec = edited_circuit()
        original = deepcopy(spec)
        inspected = inspect_circuit(spec)
        json.dumps(inspected, allow_nan=False)
        self.assertEqual(spec, original)

    def test_live_inspection_endpoint_returns_the_same_mapping(self):
        handler_type = make_handler(SimpleNamespace())
        handler = handler_type.__new__(handler_type)
        handler.requestline = "POST /api/circuit/inspect HTTP/1.1"
        handler.request_version = "HTTP/1.1"
        handler.command = "POST"
        handler.path = "/api/circuit/inspect"
        handler.client_address = ("127.0.0.1", 0)
        handler.server = SimpleNamespace(server_port=8765)
        handler.headers = Message()
        handler.headers["Host"] = "127.0.0.1:8765"
        handler.headers["Content-Type"] = "application/json"
        body = json.dumps(edited_circuit()).encode()
        handler.headers["Content-Length"] = str(len(body))
        handler.rfile, handler.wfile = io.BytesIO(body), io.BytesIO()
        handler.log_message = lambda *args: None
        handler.do_POST()
        header, payload = handler.wfile.getvalue().split(b"\r\n\r\n", 1)
        self.assertEqual(int(header.split()[1]), 200)
        result = json.loads(payload)
        self.assertEqual(result["encodings"][1]["input_indices"], [0, 1])


@unittest.skipUnless(engine_info()["engine_available"], "Julia is not installed")
class EditedCircuitLearningTests(unittest.TestCase):
    def test_visual_spec_edit_reaches_actual_ftci_training(self):
        # This is the same JSON the editor submits; changing the preprocessing
        # must change the learned function in the original input coordinates.
        problem = from_circuit(edited_circuit())
        model = fit(problem, max_bond=6, query_budget=1000, validation_samples=20, seed=14)
        xs = np.array([[0.2, 0.1], [-0.7, 0.3], [1.1, -0.4]])
        expected = np.cos(xs[:, 0] + 2*xs[:, 0]**2 + xs[:, 1])
        np.testing.assert_allclose(model.predict(xs), expected, atol=1e-10)
        self.assertLess(model.metadata["validation"]["rmse"], 1e-10)


@unittest.skipUnless(shutil.which("node"), "Node is not installed")
class CircuitEditorStateTests(unittest.TestCase):
    def run_node(self, assertions):
        source = Path(__file__).resolve().parents[1] / "ftci" / "web" / "circuit-editor.js"
        script = "const assert = require('node:assert/strict'); const editor = require(process.argv[1]);\n" + assertions
        result = subprocess.run(["node", "-e", script, str(source)], text=True,
                                capture_output=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_dependency_mapping_and_expression_validation(self):
        self.run_node("""
            assert.deepEqual(editor.dependencies('2*x0**2+x1'), [0, 1]);
            assert.deepEqual(editor.dependencies('sin(x2)+x10*x0'), [0, 2, 10]);
            assert.deepEqual(editor.validateExpression('2*x0**2+x1', 2), [0, 1]);
            assert.throws(() => editor.validateExpression('x2', 2));
            assert.throws(() => editor.validateExpression('sin(', 2));
            assert.throws(() => editor.validateExpression('__import__("os")', 2));
            assert.throws(() => editor.validateExpression('x0.constructor', 2));
        """)

    def test_gate_edits_respect_encoding_and_qubit_semantics(self):
        self.run_node("""
            const descriptor = editor.validateDraft({gate:'ry',qubits:[0],angle:'2*x0**2+x1',mode:'encoding'}, 2, 2);
            assert.equal(descriptor.gate, 'ry');
            assert.equal(descriptor.angle, '2*x0**2+x1');
            assert.deepEqual(descriptor.qubits, [0]);
            assert.throws(() => editor.validateDraft({gate:'ry',qubits:[0],angle:'x0',mode:'fixed'}, 2, 2));
            assert.throws(() => editor.validateDraft({gate:'ry',qubits:[0],angle:'0.7',mode:'encoding'}, 2, 2));
            assert.throws(() => editor.validateDraft({gate:'cx',qubits:[0,0],mode:'fixed'}, 2, 2));
            const cx = editor.validateDraft({gate:'cx',qubits:[1,0],mode:'fixed'}, 2, 2);
            assert.deepEqual(cx.qubits, [1,0]);
            const spec = {n_qubits:2, observable:'ZI', domain:[[-1,1],[-2,2]], gates:[descriptor,cx]};
            const checked = editor.validateCircuit(spec, spec.domain);
            assert.equal(checked.gates[0].angle, '2*x0**2+x1');
            assert.deepEqual(checked.gates[1].qubits, [1,0]);
            assert.throws(() => editor.validateCircuit({...spec, observable:''}, spec.domain));
            for (const invalid of [null, true, '0']) {
                assert.throws(() => editor.validateDraft({gate:'h',qubits:[invalid],mode:'fixed'}, 2, 2));
            }
        """)

    def test_gates_on_different_wires_share_a_column(self):
        self.run_node("""
            const g = (gate, ...qubits) => ({gate, qubits});
            const layers = gates => editor.layoutColumns(gates);
            // Parallel gates line up, whether or not their angles depend on inputs.
            assert.deepEqual(layers([g('h',0), g('h',1), {gate:'rz',qubits:[2],angle:'x0'}]), {columns:[0,0,0], count:1});
            assert.deepEqual(layers([g('x',0), g('x',2), g('rxx',0,1), g('rxx',1,2), g('h',0), g('cx',0,2), g('h',1)]).columns,
                             [0,0,1,2,2,3,4]);
            // A two-qubit gate occupies every wire it crosses.
            assert.deepEqual(layers([g('cx',0,2), g('h',1)]).columns, [0,1]);
            assert.deepEqual(layers([g('cx',2,0), g('h',3)]).columns, [0,0]);
            assert.deepEqual(layers([]), {columns:[], count:0});
        """)

    def test_inserting_at_a_wire_position_keeps_the_circuit_order_valid(self):
        self.run_node("""
            const g = (gate, ...qubits) => ({gate, qubits});
            const text = gates => gates.map(x => x.gate + x.qubits.join('')).join(' ');
            const base = [g('x',0), g('x',2), g('rxx',0,1), g('rxx',1,2)];
            // Between the gates on its wire: the list gains one entry and nothing else moves.
            let placed = editor.insertAtGap(base, g('z',2), 1);
            assert.equal(text(placed.gates), 'x0 x2 rxx01 z2 rxx12');
            assert.equal(placed.index, 3);
            assert.deepEqual(editor.layoutColumns(placed.gates).columns, [0,0,1,1,2]);
            // After the last gate on a wire the new gate goes to the end of the list.
            placed = editor.insertAtGap(base, g('h',0), 2);
            assert.equal(text(placed.gates), 'x0 x2 rxx01 rxx12 h0');
            // Before everything on its wire.
            assert.equal(text(editor.insertAtGap(base, g('h',2), 0).gates), 'x0 h2 x2 rxx01 rxx12');
            assert.equal(base.length, 4);
            // No single list position separates the two groups, so commuting gates are reordered.
            const tangled = [g('h',0), g('x',0), g('h',1)];
            placed = editor.insertAtGap(tangled, g('cx',0,1), 1);
            assert.equal(text(placed.gates), 'h0 h1 cx01 x0');
            assert.deepEqual(editor.layoutColumns(placed.gates).columns, [0,0,1,2]);
            assert.equal(text(editor.insertAtGap([], g('h',1), 0).gates), 'h1');
        """)

    def test_moving_and_placing_gates_choose_sensible_qubits(self):
        self.run_node("""
            const g = (gate, ...qubits) => ({gate, qubits});
            const text = gates => gates.map(x => x.gate + x.qubits.join('')).join(' ');
            assert.deepEqual(editor.placeQubits(1, 2, 3), [2]);
            assert.deepEqual(editor.placeQubits(2, 0, 3), [0, 1]);
            assert.deepEqual(editor.placeQubits(2, 2, 3), [2, 1]);
            assert.deepEqual(editor.placeQubits(2, 1, 3, -1), [1, 0]);
            assert.deepEqual(editor.placeQubits(2, 0, 4, 3), [0, 3]);
            assert.throws(() => editor.placeQubits(2, 0, 1));
            const base = [g('x',0), g('x',2), g('rxx',0,1), g('rxx',1,2), g('h',0)];
            // Drag H from the end of wire 0 to the start of wire 2.
            let moved = editor.moveToGap(base, 4, 0, 2, 3);
            assert.equal(text(moved.gates), 'x0 h2 x2 rxx01 rxx12');
            assert.equal(moved.index, 1);
            // Dropping a gate where it already is leaves the circuit as it was.
            assert.equal(text(editor.moveToGap(base, 4, 3, 0, 3).gates), text(base));
            // A two-qubit gate keeps its orientation and its parameters.
            const angled = [g('h',0), {gate:'rzz', qubits:[2,1], angle:'x0'}];
            moved = editor.moveToGap(angled, 1, 0, 1, 3);
            assert.deepEqual(moved.gates[moved.index], {gate:'rzz', qubits:[1,0], angle:'x0'});
            assert.equal(angled[1].qubits[0], 2);
        """)

    def test_undo_redo_keep_independent_circuit_snapshots(self):
        self.run_node("""
            const original = {n_qubits:1,observable:'Z',domain:[[-1,1]],gates:[{gate:'ry',qubits:[0],angle:'x0'}]};
            const history = new editor.CircuitHistory(original);
            original.gates[0].angle = 'x0+100';
            assert.equal(history.current.gates[0].angle, 'x0');
            const changed = JSON.parse(JSON.stringify(history.current));
            changed.gates[0].angle = '2*x0**2';
            history.commit(changed);
            changed.gates[0].angle = 'x0+200';
            assert.equal(history.current.gates[0].angle, '2*x0**2');
            assert.equal(history.undo(), true);
            assert.equal(history.current.gates[0].angle, 'x0');
            assert.equal(history.redo(), true);
            assert.equal(history.current.gates[0].angle, '2*x0**2');
            history.undo();
            const branch = JSON.parse(JSON.stringify(history.current));
            branch.gates.push({gate:'h',qubits:[0]});
            history.commit(branch);
            assert.equal(history.redo(), false);
            assert.equal(history.current.gates.length, 2);
        """)


if __name__ == "__main__":
    unittest.main()

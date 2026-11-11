"""Behavioral checks for coefficient-core matrix slices in the browser."""

from pathlib import Path
import shutil
import subprocess
import unittest


HARNESS = r"""
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const context = vm.createContext({assert});
vm.runInContext(`
class Element {
  constructor(tag = 'div') {
    this.tagName = tag;
    this.children = [];
    this.textContent = '';
    this.value = '';
    this.hidden = false;
    this.dataset = {};
    this.classList = {toggle() {}};
  }
  append(...nodes) {
    for (const node of nodes) {
      if (this.tagName === 'select' && !this.children.length) this.value = String(node.value);
      this.children.push(node);
    }
  }
  replaceChildren(...nodes) {
    this.children = [];
    if (this.tagName === 'select') this.value = '';
    this.append(...nodes);
  }
  setAttribute() {}
  querySelectorAll() { return []; }
}
const fields = {};
globalThis.document = {
  getElementById: id => fields[id] ||= new Element(id.endsWith('-select') ? 'select' : 'div'),
  createElement: tag => new Element(tag),
};
`, context);
const app = fs.readFileSync('ftci/web/app.js', 'utf8');
vm.runInContext(app.slice(0, app.lastIndexOf('initialize().catch')), context);
vm.runInContext(`
function matrixView() {
  const table = $('core-table-container').children[0];
  assert.equal(table.tagName, 'table');
  const head = table.children.find(node => node.tagName === 'thead').children[0];
  const body = table.children.find(node => node.tagName === 'tbody');
  return {
    columns: head.children.slice(1).map(cell => cell.textContent),
    rows: body.children.map(row => row.children[0].textContent),
    values: body.children.map(row => row.children.slice(1).map(cell => cell.textContent)),
  };
}
`, context);
async function run() {
"""


@unittest.skipUnless(shutil.which("node"), "Node is not installed")
class TensorInspectorUITests(unittest.TestCase):
    def node(self, body):
        script = HARNESS + "await vm.runInContext(`" + body + "`, context);\n}\n"
        script += "run().catch(error => { console.error(error); process.exitCode = 1; });"
        project = Path(__file__).resolve().parents[1]
        completed = subprocess.run([shutil.which("node"), "-e", script], cwd=project,
                                   text=True, capture_output=True, timeout=15)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)

    def test_basis_index_selects_matrix_with_left_rows_and_right_columns(self):
        self.node(r"""
(async () => {
  const core = {shape: [2, 3, 4], data: [
    [[101, 102, 103, 104], [111, 112, 113, 114], [121, 122, 123, 124]],
    [[201, 202, 203, 204], [211, 212, 213, 214], [221, 222, 223, 224]],
  ]};
  state.result = {core_shapes: [core.shape]};
  state.jobId = 'asymmetric-core';
  api = async () => core;
  await selectCore(0);
  const selector = $('basis-index-select');
  assert.deepEqual(selector.children.map(option => String(option.value)), ['0', '1', '2']);
  assert.deepEqual(matrixView(), {
    columns: ['0', '1', '2', '3'], rows: ['0', '1'],
    values: [['101', '102', '103', '104'], ['201', '202', '203', '204']],
  });
  selector.value = '2';
  renderCoreTable();
  assert.deepEqual(matrixView(), {
    columns: ['0', '1', '2', '3'], rows: ['0', '1'],
    values: [['121', '122', '123', '124'], ['221', '222', '223', '224']],
  });
  selector.value = '0';
  renderCoreTable();
  assert.deepEqual(matrixView().values, [['101', '102', '103', '104'], ['201', '202', '203', '204']]);
})()
""")

    def test_unit_boundary_bonds_keep_a_one_by_one_matrix(self):
        self.node(r"""
(async () => {
  const core = {shape: [1, 3, 1], values: [[[10], [20], [30]]]};
  state.result = {core_shapes: [core.shape]};
  state.jobId = 'boundary-core';
  api = async () => core;
  await selectCore(0);
  $('basis-index-select').value = '2';
  renderCoreTable();
  assert.deepEqual(matrixView(), {columns: ['0'], rows: ['0'], values: [['30']]});
})()
""")

    def test_right_bond_matrix_is_not_truncated_to_twelve_columns(self):
        self.node(r"""
(async () => {
  const core = {shape: [2, 1, 13], data: [
    [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]],
    [[14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]],
  ]};
  state.result = {core_shapes: [core.shape]};
  state.jobId = 'wide-core';
  api = async () => core;
  await selectCore(0);
  assert.deepEqual(matrixView(), {
    columns: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'],
    rows: ['0', '1'],
    values: [
      ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13'],
      ['14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26'],
    ],
  });
})()
""")


if __name__ == "__main__":
    unittest.main()

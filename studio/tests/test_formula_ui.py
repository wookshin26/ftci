"""Meaning and safety checks for offline MathML basis formatting."""

from pathlib import Path
import shutil
import subprocess
import unittest


HARNESS = r"""
const assert = require('node:assert/strict');
const formula = require('./ftci/web/formula.js');
const NS = 'http://www.w3.org/1998/Math/MathML';
class Node {
  constructor(name, document) { this.localName = name; this.ownerDocument = document; this.children = []; this.attributes = {}; this.text = ''; }
  appendChild(child) { this.children.push(child); return child; }
  replaceChildren(...children) { this.children = children; this.text = ''; }
  setAttribute(name, value) { this.attributes[name] = value; }
  get textContent() { return this.text + this.children.map(child => child.textContent).join(''); }
  set textContent(value) { this.text = String(value); this.children = []; }
  set innerHTML(value) { throw new Error('Raw HTML must never be used.'); }
}
const document = {createElementNS(namespace, name) { assert.equal(namespace, NS); return new Node(name, document); }};
const container = new Node('div', document);
function render(source) {
  assert.equal(formula.render(container, source), true, source);
  const math = container.children[0];
  assert.equal(math.localName, 'math');
  assert.equal(math.attributes['aria-label'], source);
  assert.equal(math.attributes.display, 'inline');
  return math.children[0];
}
function tags(node) { return node.children.map(child => child.localName); }
function descendants(node, name) { return [node, ...node.children.flatMap(child => descendants(child, name))].filter(child => child.localName === name); }
function coordinate(node, index) {
  assert.equal(node.localName, 'msub');
  assert.deepEqual(node.children.map(child => child.textContent), ['x', String(index)]);
}
function fenced(node) {
  assert.equal(node.localName, 'mrow');
  assert.equal(node.children[0].textContent, '(');
  assert.equal(node.children.at(-1).textContent, ')');
  return node.children[1];
}
"""


@unittest.skipUnless(shutil.which("node"), "Node is not installed")
class FormulaUITests(unittest.TestCase):
    def node(self, body):
        project = Path(__file__).resolve().parents[1]
        completed = subprocess.run([shutil.which("node"), "-e", HARNESS + body],
                                   cwd=project, text=True, capture_output=True, timeout=15)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)

    def test_python_power_associativity_and_unary_precedence(self):
        self.node(r"""
let tree = render('-x0**2');
assert.deepEqual(tags(tree), ['mo', 'msup']);
assert.equal(tree.children[0].textContent, '−');
coordinate(tree.children[1].children[0], 0);
tree = render('(-x0)**2');
assert.equal(tree.localName, 'msup');
const negativeBase = fenced(tree.children[0]);
assert.equal(negativeBase.children[0].textContent, '−');
coordinate(negativeBase.children[1], 0);
tree = render('x0**-2');
assert.equal(tree.localName, 'msup');
assert.deepEqual(tree.children[1].children.map(child => child.textContent), ['−', '2']);
tree = render('x0**x1**2');
coordinate(tree.children[0], 0);
assert.equal(tree.children[1].localName, 'msup');
coordinate(tree.children[1].children[0], 1);
tree = render('(x0**x1)**2');
assert.equal(fenced(tree.children[0]).localName, 'msup');
tree = render('(2e-3)**2');
const scientificBase = fenced(tree.children[0]);
assert.deepEqual(tags(scientificBase), ['mn', 'mo', 'msup']);
assert.equal(scientificBase.children[1].textContent, '×');
assert.equal(tree.children[1].textContent, '2');
""")

    def test_fraction_nesting_and_nonassociative_subtraction(self):
        self.node(r"""
let tree = render('(x0+x1)/(1+x2**2)');
assert.equal(tree.localName, 'mfrac');
assert.equal(tree.children[0].children[1].textContent, '+');
assert.equal(tree.children[1].children[2].localName, 'msup');
tree = render('x0/(x1/x2)');
assert.equal(tree.localName, 'mfrac');
coordinate(tree.children[0], 0);
assert.equal(tree.children[1].localName, 'mfrac');
tree = render('(x0/x1)/x2');
assert.equal(tree.children[0].localName, 'mfrac');
coordinate(tree.children[1], 2);
tree = render('x0-(x1-x2)');
const right = fenced(tree.children[2]);
assert.equal(tree.children[1].textContent, '−');
assert.equal(right.children[1].textContent, '−');
coordinate(right.children[0], 1); coordinate(right.children[2], 2);
""")

    def test_named_functions_roots_and_two_argument_order(self):
        self.node(r"""
const unary = ['sin','cos','tan','exp','log','sqrt','abs','tanh','sinh','cosh','asin','acos','atan','arcsin','arccos','arctan','log1p','expm1'];
for (const name of unary) render(name + '(x0)');
const root = render('sqrt(1+x12**2)');
assert.equal(root.localName, 'msqrt');
coordinate(descendants(root, 'msup')[0].children[0], 12);
const absolute = render('abs(x0-x1)');
assert.equal(absolute.children[0].textContent, '|');
assert.equal(absolute.children.at(-1).textContent, '|');
for (const name of ['atan2','minimum','maximum']) {
  const call = render(name + '(x1,x0)');
  assert.equal(call.children[0].attributes.mathvariant, 'normal');
  const argumentsRow = fenced(call.children[2]);
  coordinate(argumentsRow.children[0], 1);
  assert.equal(argumentsRow.children[1].textContent, ',');
  coordinate(argumentsRow.children[2], 0);
}
assert.equal(render('atan2(x0,x1)').children[0].textContent, 'atan2');
for (const name of ['log1p','expm1']) {
  const multiplied = render('2*' + name + '(x0)');
  assert.equal(multiplied.children[2].children[0].textContent, name);
  const powered = render(name + '(x0)**2');
  assert.equal(powered.children[0].children[0].textContent, name);
}
""")

    def test_constants_and_coordinates_preserve_numeric_values(self):
        self.node(r"""
coordinate(render('x4095'), 4095);
const exact = '0.7071067811865475';
const decimal = render(exact);
assert.equal(decimal.localName, 'mn'); assert.equal(decimal.textContent, exact);
const pi = render('pi'); assert.equal(pi.textContent, 'π');
const e = render('e'); assert.equal(e.textContent, 'e');
const scientific = render('1.25e-6');
assert.equal(scientific.children[0].textContent, '1.25');
assert.equal(scientific.children[1].textContent, '×');
const power = scientific.children[2];
assert.equal(power.localName, 'msup');
assert.equal(power.children[0].textContent, '10');
assert.equal(power.children[1].textContent, '−6');
for (const source of ['.5','2.','2e+3','1_000.25','0xff','0b101','0o17']) render(source);
assert.equal(render('0xff').textContent, '255');
render('  cos(x0)  ');
""")

    def test_invalid_or_unrecognized_input_falls_back_to_exact_text(self):
        self.node(r"""
globalThis.executed = false;
for (const source of [
  '<script>globalThis.executed=true</script>', '__import__("os")',
  'x0.constructor', 'sin(x0);globalThis.executed=true', 'x0[0]',
  'unknown(x0)', 'sin(x0,x1)', 'atan2(x0)', 'sqrt()',
  'True', 'x01', 'x4096', '1e309', '01', 'x0//2', 'x0%2', 'x0 +',
]) {
  render('x0**2');
  assert.equal(formula.render(container, source), false, source);
  assert.equal(container.textContent, source);
  assert.equal(container.children.length, 0);
}
assert.equal(globalThis.executed, false);
assert.equal(formula.render(null, 'x0'), false);
""")

    def test_complexity_limits_fall_back_without_throwing(self):
        self.node(r"""
function sum(leaves) { if (leaves === 1) return 'x0'; const left = Math.floor(leaves/2); return '(' + sum(left) + '+' + sum(leaves-left) + ')'; }
render(sum(128)); // 255 AST nodes, with shallow balanced depth.
render('-'.repeat(48) + 'x0');
for (const source of [sum(129), '-'.repeat(49) + 'x0', 'x0' + ' '.repeat(4095), '('.repeat(1000) + 'x0' + ')'.repeat(1000)]) {
  assert.equal(formula.render(container, source), false);
  assert.equal(container.textContent, source);
}
""")


if __name__ == "__main__":
    unittest.main()

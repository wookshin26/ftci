"""A safe, source-preserving subset of native Cirq circuit construction.

Uploaded Python is inspected as syntax, never executed or imported. Inputs are
SymPy symbols in declaration order (including unused symbols); repeated Symbol
names refer to the same input. LineQubit(k) maps to diagram wire k. FTCI metadata
uses little-endian Pauli strings: the rightmost observable letter acts on q0.

Supported code is straight-line, single-assignment Python: Cirq/SymPy imports,
LineQubit declarations, Symbol/symbols declarations, scalar preprocessing,
Circuit operation trees and append calls. Rotations use radians, not Cirq's
half-turn power-gate convention. Measurements, controls, powers, loops and
arbitrary Python calls are rejected explicitly.
"""
from __future__ import annotations

import ast
from copy import deepcopy
from dataclasses import dataclass
import io
import json
import math
import re
import tokenize

from .expressions import Expression, _FUNCTIONS
from .quantum import from_circuit


_MAX_SOURCE = 128_000
_FIXED = {"I": "id", "H": "h", "X": "x", "Y": "y", "Z": "z", "S": "s", "T": "t",
          "CNOT": "cx", "CX": "cx", "CZ": "cz", "SWAP": "swap"}
_ROTATIONS = {"rx": "rx", "ry": "ry", "rz": "rz", "Rx": "rx", "Ry": "ry", "Rz": "rz"}
_EXPORT = {value: key for key, value in _FIXED.items()}
_EXPORT["cx"] = "CNOT"
_ALIASES = {"cnot": "cx", "i": "id"}
_SYMPY_FUNCTIONS = {name: name for name in _FUNCTIONS if name not in {"log1p", "expm1", "arcsin", "arccos", "arctan", "minimum", "maximum", "abs"}}
_SYMPY_FUNCTIONS.update({"Abs": "abs", "Min": "minimum", "Max": "maximum"})
_MATH_FUNCTIONS = set(_FUNCTIONS) - {"abs", "arcsin", "arccos", "arctan", "minimum", "maximum"}
_SYMPY_EXPORT = {"abs": "Abs", "minimum": "Min", "maximum": "Max", "arcsin": "asin", "arccos": "acos", "arctan": "atan"}
_COORD = re.compile(r"x(0|[1-9][0-9]*)\Z")
_SYMBOL_RANGE = re.compile(r"([A-Za-z_]\w*?)([0-9]*):([0-9]+)\Z")
_METADATA = {"domain", "observable", "initial_state", "name", "n_qubits"}


@dataclass(frozen=True)
class _Scalar:
    text: str


@dataclass(frozen=True)
class _Qubit:
    index: int


@dataclass
class _Operation:
    gate: dict
    node: ast.AST


def _text(source):
    if not isinstance(source, str) or not source.strip():
        raise ValueError("Cirq source must be nonempty text.")
    if len(source) > _MAX_SOURCE:
        raise ValueError("Cirq source is limited to 128000 characters.")
    return source


class _CirqReader:
    def __init__(self, source, default_domain=None):
        self.source, self.default_domain = _text(source), default_domain
        try:
            self.tree = ast.parse(source)
        except (SyntaxError, RecursionError, ValueError) as error:
            raise ValueError(f"Cirq line {getattr(error, 'lineno', '?')}: {getattr(error, 'msg', str(error))}") from error
        if sum(1 for _ in ast.walk(self.tree)) > 15000 or len(self.tree.body) > 3000:
            raise ValueError("Cirq source is too complex for the circuit editor.")
        self.modules, self.imported, self.env = {}, {}, {}
        self.symbols, self.symbol_refs, self.qubit_refs = [], [], {}
        self.symbol_assumptions = {}
        self.symbol_sequences, self.qubit_sequences = [], []
        self.metadata, self.metadata_nodes = {}, {}
        self.gates, self.gate_nodes = [], []
        self.circuit, self.constructor_statement = None, None
        self.last_operation_statement = None
        self._expr_count = 0

    def fail(self, node, message):
        raise ValueError(f"Cirq line {getattr(node, 'lineno', '?')}: {message}")

    def qualified(self, node):
        if isinstance(node, ast.Name):
            return self.imported.get(node.id)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            module = self.modules.get(node.value.id)
            if module:
                return module, node.attr
        if (isinstance(node, ast.Attribute) and node.attr == "range"
                and self.qualified(node.value) == ("cirq", "LineQubit")):
            return "cirq", "LineQubit.range"
        return None

    def expr(self, node, depth=0):
        self._expr_count += 1
        if depth > 45 or self._expr_count > 15000:
            self.fail(node, "Expression is too complex.")
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            try:
                finite = math.isfinite(float(node.value))
            except (OverflowError, ValueError):
                finite = False
            if not finite:
                self.fail(node, "Constants must be finite and within the supported numeric range.")
            return repr(node.value)
        if isinstance(node, ast.Name) and isinstance(self.env.get(node.id), _Scalar):
            return self.env[node.id].text
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            sequence = self.env.get(node.value.id)
            index = self.integer(node.slice)
            if not isinstance(sequence, list) or not 0 <= index < len(sequence) or not isinstance(sequence[index], _Scalar):
                self.fail(node, "Use an in-range index of a declared symbol sequence.")
            return sequence[index].text
        qualified = self.qualified(node)
        if qualified and ((qualified[0] == "sympy" and qualified[1] in {"pi", "E"})
                          or (qualified[0] in {"numpy", "math"} and qualified[1] in {"pi", "e"})):
            return "pi" if qualified[1] == "pi" else "e"
        if isinstance(node, ast.UnaryOp) and type(node.op) in (ast.UAdd, ast.USub):
            return f"({'+' if isinstance(node.op, ast.UAdd) else '-'}{self.expr(node.operand, depth+1)})"
        if isinstance(node, ast.BinOp) and type(node.op) in (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow):
            operator = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.Pow: "**"}[type(node.op)]
            result = f"({self.expr(node.left, depth+1)}{operator}{self.expr(node.right, depth+1)})"
            if len(result) > 4096:
                self.fail(node, "Expanded preprocessing exceeds 4096 characters.")
            return result
        if isinstance(node, ast.Call) and not node.keywords:
            qualified = self.qualified(node.func)
            module, name = qualified if qualified else (None, None)
            if isinstance(node.func, ast.Name) and node.func.id == "abs" and node.func.id not in self.env:
                module, name = "sympy", "Abs"
            function = (_SYMPY_FUNCTIONS.get(name) if module == "sympy" else
                        name if (module == "numpy" and name in _FUNCTIONS) or (module == "math" and name in _MATH_FUNCTIONS) else None)
            if function and len(node.args) == _FUNCTIONS[function][1]:
                args = [self.expr(arg, depth+1) for arg in node.args]
                if module != "sympy" and any(Expression(arg).input_indices for arg in args):
                    self.fail(node, "Use SymPy functions for symbolic preprocessing; NumPy/math functions do not accept native SymPy input symbols.")
                return f"{function}({','.join(args)})"
        self.fail(node, "Use declared SymPy inputs, scalar arithmetic and supported math functions; arbitrary Python is not executed.")

    def number(self, node):
        expression = Expression(self.expr(node))
        if expression.input_indices:
            self.fail(node, "This value must be a fixed number, not an input symbol.")
        try:
            return expression([])
        except ValueError as error:
            self.fail(node, str(error))

    def integer(self, node):
        value = self.number(node)
        if value != int(value):
            self.fail(node, "Use a whole-number qubit index or size.")
        return int(value)

    def literal(self, node):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, (ast.List, ast.Tuple)):
            return [self.literal(item) for item in node.elts]
        return self.number(node)

    def import_statement(self, node):
        if isinstance(node, ast.Import):
            for item in node.names:
                if item.name not in {"cirq", "sympy", "numpy", "math"}:
                    self.fail(node, "Only cirq, sympy, numpy and math imports are accepted; imports are not executed.")
                name = item.asname or item.name
                self.reserve(name, node)
                self.modules[name] = item.name
            return
        if node.level or node.module not in {"cirq", "sympy", "numpy", "math"}:
            self.fail(node, "This import is outside the editable Cirq subset.")
        for item in node.names:
            if ((node.module == "cirq" and item.name in {"Circuit", "LineQubit", *_FIXED, *_ROTATIONS})
                    or (node.module == "sympy" and item.name in {"Symbol", "symbols", "pi", "E", *_SYMPY_FUNCTIONS})
                    or (node.module == "numpy" and item.name in {"pi", "e", *_FUNCTIONS})
                    or (node.module == "math" and item.name in {"pi", "e", *_MATH_FUNCTIONS})):
                name = item.asname or item.name
                self.reserve(name, node)
                self.imported[name] = node.module, item.name
            else:
                self.fail(node, f"Unsupported import {item.name!r}.")

    def reserve(self, name, node):
        if name in self.env or name in self.modules or name in self.imported or name == self.circuit:
            self.fail(node, f"{name!r} is already declared; the editable subset uses single assignment. Edit its original declaration.")

    def symbol(self, name, node, real=False):
        if not isinstance(name, str) or not name or len(name) > 80:
            self.fail(node, "Symbol names must be nonempty strings of at most 80 characters.")
        if name not in self.symbols:
            if len(self.symbols) >= 32:
                self.fail(node, "The circuit editor supports at most 32 input symbols.")
            self.symbols.append(name)
            self.symbol_refs.append(None)
            self.symbol_assumptions[name] = real
        elif self.symbol_assumptions[name] != real:
            self.fail(node, "Repeated Symbol names must use the same real assumption in the editable subset.")
        return _Scalar(f"x{self.symbols.index(name)}")

    def symbol_value(self, node, constructor):
        if (len(node.args) != 1 or not isinstance(node.args[0], ast.Constant)
                or not isinstance(node.args[0].value, str)
                or len(node.keywords) > 1 or any(keyword.arg != "real" or not isinstance(keyword.value, ast.Constant)
                                                or keyword.value.value is not True for keyword in node.keywords)):
            self.fail(node, "Use Symbol('name') or symbols('a b') / symbols('x0:2') with literal names and optional real=True.")
        names = node.args[0].value
        real = bool(node.keywords)
        if constructor == "Symbol":
            return self.symbol(names, node, real)
        match = _SYMBOL_RANGE.fullmatch(names)
        if match:
            prefix, start, stop = match.groups()
            start, stop = int(start or 0), int(stop)
            if not 0 <= start < stop <= 32:
                self.fail(node, "Symbol ranges must contain 1–32 inputs.")
            return [self.symbol(f"{prefix}{i}", node, real) for i in range(start, stop)]
        if ":" in names or not re.fullmatch(r"[\w\s,]+", names):
            self.fail(node, "Use simple whitespace/comma-separated symbol names or a range such as 'x0:2'.")
        labels = [part for part in re.split(r"[,\s]+", names.strip()) if part]
        if not labels or len(labels) > 32:
            self.fail(node, "Declare 1–32 symbols.")
        values = [self.symbol(label, node, real) for label in labels]
        return values[0] if len(values) == 1 and not names.strip().endswith(",") else values

    def qubit(self, node):
        if isinstance(node, ast.Name) and isinstance(self.env.get(node.id), _Qubit):
            return self.env[node.id]
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            values = self.env.get(node.value.id)
            index = self.integer(node.slice)
            if isinstance(values, list) and 0 <= index < len(values) and isinstance(values[index], _Qubit):
                return values[index]
        if isinstance(node, ast.Call) and self.qualified(node.func) == ("cirq", "LineQubit"):
            if len(node.args) != 1 or node.keywords:
                self.fail(node, "LineQubit needs one fixed integer coordinate.")
            index = self.integer(node.args[0])
            if not 0 <= index < 10:
                self.fail(node, "LineQubit coordinates must be between 0 and 9; coordinate k maps to diagram wire k.")
            return _Qubit(index)
        self.fail(node, "Use a declared LineQubit or an in-range qubit sequence index.")

    def operation(self, node):
        if not isinstance(node, ast.Call):
            self.fail(node, "Use supported Cirq gate operations; power gates and controlled operations are not accepted.")
        gate_function, qubit_args = node.func, node.args
        if node.keywords:
            self.fail(node, "Gate operations require positional qubit arguments.")
        if isinstance(gate_function, ast.Attribute) and gate_function.attr == "on":
            gate_function = gate_function.value
        qualified = self.qualified(gate_function)
        gate = None
        if qualified and qualified[0] == "cirq" and qualified[1] in _FIXED:
            gate = {"gate": _FIXED[qualified[1]]}
        elif isinstance(gate_function, ast.Call):
            rotation = self.qualified(gate_function.func)
            if rotation and rotation[0] == "cirq" and rotation[1] in _ROTATIONS:
                if len(gate_function.args) == 1 and not gate_function.keywords:
                    angle_node = gate_function.args[0]
                elif (not gate_function.args and len(gate_function.keywords) == 1
                      and gate_function.keywords[0].arg == "rads"):
                    angle_node = gate_function.keywords[0].value
                else:
                    self.fail(gate_function, "rx/ry/rz need exactly one angle in radians (optionally rads=...).")
                angle = self.expr(angle_node)
                try:
                    expression = Expression(angle)
                    if not expression.input_indices:
                        expression([])
                except ValueError as error:
                    self.fail(angle_node, str(error))
                gate = {"gate": _ROTATIONS[rotation[1]], "angle": angle}
        if gate is None:
            self.fail(node, "Unsupported Cirq operation. Use I/H/X/Y/Z/S/T/CNOT/CZ/SWAP or rx/ry/rz in radians; powers, measurements, channels and controls require an SDK adapter.")
        arity = 2 if gate["gate"] in {"cx", "cz", "swap"} else 1
        if len(qubit_args) != arity:
            self.fail(node, f"{gate['gate']} requires {arity} qubit argument(s).")
        gate["qubits"] = []
        for argument in qubit_args:
            qubit = self.qubit(argument)
            gate["qubits"].append(qubit.index)
            self.qubit_refs.setdefault(qubit.index, ast.get_source_segment(self.source, argument))
        if len(set(gate["qubits"])) != arity:
            self.fail(node, "A gate cannot target the same qubit twice.")
        return _Operation(gate, node)

    def value(self, node):
        if isinstance(node, ast.Name) and node.id in self.env:
            return self.env[node.id]
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            values = self.env.get(node.value.id)
            index = self.integer(node.slice)
            if isinstance(values, list) and 0 <= index < len(values):
                return values[index]
            self.fail(node, "Use an in-range index of a declared sequence.")
        if isinstance(node, (ast.List, ast.Tuple)):
            return [self.value(item) for item in node.elts]
        if isinstance(node, ast.Call):
            qualified = self.qualified(node.func)
            if qualified and qualified[0] == "sympy" and qualified[1] in {"Symbol", "symbols"}:
                return self.symbol_value(node, qualified[1])
            if qualified == ("cirq", "LineQubit.range"):
                if node.keywords or not 1 <= len(node.args) <= 3:
                    self.fail(node, "LineQubit.range accepts 1–3 fixed integer arguments.")
                args = [self.integer(arg) for arg in node.args]
                try:
                    indices = range(*args)
                except ValueError as error:
                    self.fail(node, str(error))
                if not indices or len(indices) > 10 or min(indices) < 0 or max(indices) > 9:
                    self.fail(node, "Declare 1–10 line qubits with coordinates from 0 to 9.")
                return [_Qubit(i) for i in indices]
            if qualified == ("cirq", "LineQubit"):
                return self.qubit(node)
            if (qualified and qualified[0] == "cirq") or isinstance(node.func, ast.Call) or (isinstance(node.func, ast.Attribute) and node.func.attr == "on"):
                return self.operation(node)
        return _Scalar(self.expr(node))

    def bind(self, target, value, value_node):
        if isinstance(target, ast.Name):
            self.reserve(target.id, target)
            self.env[target.id] = value
            self.record_refs(value, target.id)
            if isinstance(value, list) and isinstance(value_node, ast.Call):
                constructor = self.qualified(value_node.func)
                if constructor == ("cirq", "LineQubit.range") and len(value_node.args) == 1:
                    self.qubit_sequences.append((target.id, value, value_node.args[0]))
                if constructor == ("sympy", "symbols") and _SYMBOL_RANGE.fullmatch(value_node.args[0].value):
                    self.symbol_sequences.append((target.id, value, value_node.args[0]))
            return
        if isinstance(target, (ast.Tuple, ast.List)) and isinstance(value, list) and len(target.elts) == len(value):
            for item, item_value in zip(target.elts, value):
                self.bind(item, item_value, value_node)
            return
        self.fail(target, "Use one variable or a matching tuple of variable names in assignments.")

    def record_refs(self, value, ref):
        if isinstance(value, _Scalar) and _COORD.fullmatch(value.text):
            index = int(value.text[1:])
            if self.symbol_refs[index] is None:
                self.symbol_refs[index] = ref
        elif isinstance(value, _Qubit):
            # Keep the earliest usable reference; a later alias may not exist
            # yet at the earlier gate expression that a visual edit rewrites.
            self.qubit_refs.setdefault(value.index, ref)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                self.record_refs(item, f"{ref}[{index}]")

    def add_operations(self, node):
        if isinstance(node, ast.Name) and node.id in self.env:
            value = self.env[node.id]
        elif isinstance(node, (ast.List, ast.Tuple)):
            for item in node.elts:
                self.add_operations(item)
            return
        else:
            value = self.operation(node)

        def flatten(item):
            if isinstance(item, _Operation):
                self.gates.append(deepcopy(item.gate))
                self.gate_nodes.append(item.node)
            elif isinstance(item, list):
                for child in item:
                    flatten(child)
            else:
                self.fail(node, "Circuit/append arguments must contain supported gate operations.")
        flatten(value)

    def read(self):
        for statement in self.tree.body:
            if isinstance(statement, (ast.Import, ast.ImportFrom)):
                self.import_statement(statement)
            elif isinstance(statement, ast.Pass) or (isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant) and isinstance(statement.value.value, str)):
                continue
            elif isinstance(statement, ast.Assign) and len(statement.targets) == 1:
                target, value_node = statement.targets[0], statement.value
                if isinstance(target, ast.Name) and target.id in _METADATA:
                    if target.id in self.metadata:
                        self.fail(target, f"Declare {target.id} metadata only once.")
                    self.metadata[target.id] = self.integer(value_node) if target.id == "n_qubits" else self.literal(value_node)
                    self.metadata_nodes[target.id] = value_node
                    if target.id == "n_qubits":
                        self.reserve(target.id, target)
                        self.env[target.id] = _Scalar(repr(self.metadata[target.id]))
                elif isinstance(value_node, ast.Call) and self.qualified(value_node.func) == ("cirq", "Circuit"):
                    if self.circuit is not None or not isinstance(target, ast.Name) or value_node.keywords:
                        self.fail(statement, "Declare one Circuit with positional operations and a simple variable name.")
                    self.reserve(target.id, target)
                    self.circuit, self.constructor_statement = target.id, statement
                    for argument in value_node.args:
                        self.add_operations(argument)
                    self.last_operation_statement = statement
                else:
                    self.bind(target, self.value(value_node), value_node)
            elif (isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call)
                  and isinstance(statement.value.func, ast.Attribute) and statement.value.func.attr == "append"
                  and isinstance(statement.value.func.value, ast.Name) and statement.value.func.value.id == self.circuit):
                call = statement.value
                if len(call.args) != 1 or call.keywords:
                    self.fail(call, "append accepts one operation or literal operation tree; strategy keywords are outside this subset.")
                self.add_operations(call.args[0])
                self.last_operation_statement = statement
            else:
                self.fail(statement, "Use straight-line Circuit construction and append calls. Loops, definitions, mutation and arbitrary Python calls are not executed.")
        if self.circuit is None:
            raise ValueError("Cirq source must declare a circuit = cirq.Circuit(...).")
        n_qubits = self.metadata.get("n_qubits", max(self.qubit_refs, default=-1) + 1)
        if not 1 <= n_qubits <= 10 or any(index >= n_qubits for index in self.qubit_refs):
            raise ValueError("Declare 1–10 LineQubits; n_qubits must include every declared coordinate.")
        domain = self.metadata.get("domain", self.default_domain)
        if domain is None:
            domain = [[-math.pi, math.pi] for _ in range(max(1, len(self.symbols)))]
        if not isinstance(domain, (list, tuple)) or len(domain) < len(self.symbols):
            raise ValueError(f"Cirq declares {len(self.symbols)} input symbols; domain must provide at least that many coordinate intervals. Inputs follow symbol declaration order.")
        result = {"n_qubits": n_qubits, "gates": self.gates, "domain": deepcopy(domain),
                  "observable": self.metadata.get("observable", "Z" * n_qubits)}
        result.update({key: self.metadata[key] for key in ("initial_state", "name") if key in self.metadata})
        problem = from_circuit(result)
        result["domain"] = [list(pair) for pair in problem.domain]
        result["observable"] = problem.metadata["observable"]
        return result


def parse_cirq(source, default_domain=None):
    """Safely parse native Cirq text into the simulator's circuit descriptor."""
    try:
        return _CirqReader(source, default_domain).read()
    except (RecursionError, OverflowError) as error:
        raise ValueError("Cirq source is too deeply nested or contains an oversized number.") from error


def _angle_source(text, refs, sympy_alias):
    expression = Expression(str(text))
    if expression.input_indices and max(expression.input_indices) >= len(refs):
        raise ValueError("Declare additional SymPy input symbols in the Cirq code before connecting them in the diagram.")

    class NativeNames(ast.NodeTransformer):
        def visit_Name(self, node):
            if _COORD.fullmatch(node.id):
                ref = refs[int(node.id[1:])]
                if ref is None:
                    raise ValueError("Declare input symbols in the Cirq code before editing their connections.")
                return ast.copy_location(ast.parse(ref, mode="eval").body, node)
            if node.id in {"pi", "e"}:
                return ast.copy_location(ast.Attribute(value=ast.Name(id=sympy_alias, ctx=ast.Load()), attr="pi" if node.id == "pi" else "E", ctx=ast.Load()), node)
            return node

        def visit_Call(self, node):
            name = node.func.id
            args = [self.visit(argument) for argument in node.args]
            def call(function, arguments):
                return ast.Call(func=ast.Attribute(value=ast.Name(id=sympy_alias, ctx=ast.Load()), attr=function, ctx=ast.Load()), args=arguments, keywords=[])
            if name == "log1p":
                result = call("log", [ast.BinOp(left=ast.Constant(1), op=ast.Add(), right=args[0])])
            elif name == "expm1":
                result = ast.BinOp(left=call("exp", args), op=ast.Sub(), right=ast.Constant(1))
            else:
                result = call(_SYMPY_EXPORT.get(name, name), args)
            return ast.copy_location(result, node)

    tree = ast.parse(str(text), mode="eval")
    return ast.unparse(ast.fix_missing_locations(NativeNames().visit(tree))).strip()


def _operation_source(gate, refs, qubits, cirq_alias, sympy_alias):
    name = _ALIASES.get(gate["gate"], gate["gate"])
    if name not in {*_EXPORT, "rx", "ry", "rz"}:
        raise ValueError(f"Gate {name!r} is outside the native Cirq text subset. Keep JSON/Qiskit format or edit a supported Cirq operation directly.")
    try:
        arguments = ", ".join(qubits[q] for q in gate["qubits"])
    except KeyError as error:
        raise ValueError("Declare additional LineQubits in the Cirq code before adding diagram wires.") from error
    if name in _ROTATIONS:
        angle = _angle_source(gate["angle"], refs, sympy_alias)
        return f"{cirq_alias}.{name}({angle})({arguments})"
    return f"{cirq_alias}.{_EXPORT[name]}({arguments})"


def _key(gate):
    angle = None
    if "angle" in gate:
        # Numeric int/float spelling and alias spellings do not change angles.
        class Canonical(ast.NodeTransformer):
            def visit_Constant(self, node):
                return ast.copy_location(ast.Constant(float(node.value)), node)
            def visit_Call(self, node):
                self.generic_visit(node)
                node.func.id = {"arcsin": "asin", "arccos": "acos", "arctan": "atan"}.get(node.func.id, node.func.id)
                if node.func.id == "log1p":
                    return ast.Call(func=ast.Name(id="log", ctx=ast.Load()), args=[ast.BinOp(left=ast.Constant(1.0), op=ast.Add(), right=node.args[0])], keywords=[])
                if node.func.id == "expm1":
                    return ast.BinOp(left=ast.Call(func=ast.Name(id="exp", ctx=ast.Load()), args=node.args, keywords=[]), op=ast.Sub(), right=ast.Constant(1.0))
                return node
        angle = ast.dump(Canonical().visit(ast.parse(str(gate["angle"]), mode="eval")), include_attributes=False)
    return _ALIASES.get(gate["gate"], gate["gate"]), tuple(gate["qubits"]), angle


def _same(a, b):
    return (a["n_qubits"] == b["n_qubits"] and [_key(g) for g in a["gates"]] == [_key(g) for g in b["gates"]]
            and a["domain"] == b["domain"] and a["observable"] == b["observable"]
            and a.get("initial_state", "0"*a["n_qubits"]) == b.get("initial_state", "0"*b["n_qubits"])
            and a.get("name") == b.get("name"))


def _span(source, node):
    lines = source.splitlines(keepends=True)
    start = sum(map(len, lines[:node.lineno-1])) + len(lines[node.lineno-1].encode()[:node.col_offset].decode())
    end = sum(map(len, lines[:node.end_lineno-1])) + len(lines[node.end_lineno-1].encode()[:node.end_col_offset].decode())
    return start, end


def _patch(circuit, source):
    reader = _CirqReader(source, circuit["domain"])
    old = reader.read()
    edits, prefixes, suffixes = [], [], []
    refs, qubits = list(reader.symbol_refs), dict(reader.qubit_refs)
    if len(circuit["domain"]) != len(refs) and reader.symbol_sequences:
        if len(reader.symbol_sequences) == 1:
            name, values, node = reader.symbol_sequences[0]
            if len(values) == len(refs) and [v.text for v in values] == [f"x{i}" for i in range(len(refs))]:
                match = _SYMBOL_RANGE.fullmatch(node.value)
                prefix, start, _ = match.groups()
                start = int(start or 0)
                edits.append((*_span(source, node), repr(f"{prefix}{start}:{start+len(circuit['domain'])}")))
                refs = [f"{name}[{i}]" for i in range(len(circuit["domain"]))]
    if len(circuit["domain"]) < len(refs):
        raise ValueError("Edit the SymPy declarations before removing Cirq input coordinates.")
    if circuit["n_qubits"] != old["n_qubits"]:
        if len(reader.qubit_sequences) == 1:
            name, values, node = reader.qubit_sequences[0]
            if [v.index for v in values] == list(range(old["n_qubits"])):
                edits.append((*_span(source, node), str(circuit["n_qubits"])))
                qubits = {i: f"{name}[{i}]" for i in range(circuit["n_qubits"])}
        if set(qubits) != set(range(circuit["n_qubits"])):
            raise ValueError("Edit the LineQubit declarations in Cirq code before changing the number of diagram wires.")
    used_names = ({node.id for node in ast.walk(reader.tree) if isinstance(node, ast.Name)}
                  | set(reader.modules) | set(reader.imported) | set(reader.env))
    def module_alias(module):
        existing = next((name for name, value in reader.modules.items() if value == module), None)
        if existing:
            return existing
        candidate = f"_ftci_{module}"
        while candidate in used_names:
            candidate += "_"
        used_names.add(candidate)
        return candidate
    cirq_alias, sympy_alias = module_alias("cirq"), module_alias("sympy")
    spans = [_span(source, node) for node in reader.gate_nodes]
    if len(spans) != len(set(spans)):
        raise ValueError("This Cirq source reuses operation variables. Edit those operations in code; the diagram cannot rewrite shared operation definitions safely.")
    generated = []
    for index, (gate, span) in enumerate(zip(old["gates"], spans)):
        if index < len(circuit["gates"]):
            desired = circuit["gates"][index]
            if _key(gate) == _key(desired):
                continue
            replacement = _operation_source(desired, refs, qubits, cirq_alias, sympy_alias)
        else:
            replacement = "[]"  # Empty operation tree works inside lists, Circuit and append.
        generated.append(replacement)
        edits.append((*span, replacement))
    if len(circuit["gates"]) > len(spans):
        extra = [_operation_source(gate, refs, qubits, cirq_alias, sympy_alias) for gate in circuit["gates"][len(spans):]]
        generated.extend(extra)
        # Append after all source statements; declarations retain their meaning.
        suffixes.extend(f"{reader.circuit}.append({operation})" for operation in extra)
    for key in ("domain", "observable", "initial_state", "name", "n_qubits"):
        desired = circuit.get(key, "0"*circuit["n_qubits"] if key == "initial_state" else None)
        previous = old.get(key, "0"*old["n_qubits"] if key == "initial_state" else None)
        if desired == previous:
            continue
        if desired is None:
            raise ValueError(f"Remove {key} metadata in Cirq code directly before this visual edit.")
        if key in reader.metadata_nodes:
            edits.append((*_span(source, reader.metadata_nodes[key]), repr(desired)))
        elif key != "n_qubits":
            suffixes.append(f"{key} = {desired!r}")
    for alias, module in ((cirq_alias, "cirq"), (sympy_alias, "sympy")):
        if alias not in reader.modules and any(f"{alias}." in value for value in generated):
            prefixes.append(f"import {module} as {alias}")
    for index, (start, end, replacement) in enumerate(edits):
        try:
            comments = [token.string for token in tokenize.generate_tokens(io.StringIO(source[start:end]).readline) if token.type == tokenize.COMMENT]
        except (tokenize.TokenError, IndentationError):
            comments = []
        if comments:
            edits[index] = (start, end, "\n".join(comments) + "\n" + replacement)
    output = source
    for start, end, replacement in sorted(edits, reverse=True):
        output = output[:start] + replacement + output[end:]
    if prefixes:
        output = "\n".join(prefixes) + "\n" + output
    if suffixes:
        output = output.rstrip() + "\n" + "\n".join(suffixes) + "\n"
    check = parse_cirq(output, circuit["domain"])
    if not _same(circuit, check):
        raise ValueError("This visual edit cannot preserve the native Cirq source semantics. Edit the code directly instead.")
    return output


def serialize_cirq(circuit, previous_source=None):
    """Write native Cirq code, patching existing code without executing it.

    Visual edits retain declarations, scalar preprocessing variables, imports
    and comments. Reused operation definitions or ambiguous rewrites produce an
    actionable error; source is never replaced silently.
    """
    try:
        circuit = deepcopy(circuit)
        json.dumps(circuit, allow_nan=False)
        problem = from_circuit(circuit)
        circuit["domain"] = [list(pair) for pair in problem.domain]
        circuit["observable"] = problem.metadata["observable"]
    except (RecursionError, OverflowError, TypeError) as error:
        raise ValueError("Circuit data must contain finite, JSON-compatible values.") from error
    if (set(circuit) - {"n_qubits", "gates", "domain", "observable", "initial_state", "name"}
            or any(set(gate) - {"gate", "qubits", "angle"} for gate in circuit["gates"])):
        raise ValueError("This circuit has extra JSON metadata. Keep JSON format to preserve it.")
    if previous_source is not None:
        return _patch(circuit, _text(previous_source))
    dimension = len(circuit["domain"])
    refs = [f"x[{i}]" for i in range(dimension)]
    qubits = {i: f"q[{i}]" for i in range(circuit["n_qubits"])}
    lines = ["import cirq", "import sympy as sp", "", "# Inputs follow Symbol declaration order; q[k] is diagram wire k.",
             f"x = sp.symbols('x0:{dimension}')", f"q = cirq.LineQubit.range({circuit['n_qubits']})", "circuit = cirq.Circuit()"]
    lines.extend(f"circuit.append({_operation_source(gate, refs, qubits, 'cirq', 'sp')})" for gate in circuit["gates"])
    lines.extend(["", "# FTCI metadata: rightmost Pauli letter acts on q0.", f"domain = {circuit['domain']!r}", f"observable = {circuit['observable']!r}"])
    lines.extend(f"{key} = {circuit[key]!r}" for key in ("initial_state", "name") if key in circuit)
    output = "\n".join(lines) + "\n"
    if not _same(circuit, parse_cirq(output)):
        raise ValueError("This circuit's preprocessing cannot be represented faithfully in the native Cirq text subset. Keep JSON format.")
    return output

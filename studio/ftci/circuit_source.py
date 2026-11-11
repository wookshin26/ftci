"""Safe, editable JSON and Python sources for the circuit workspace.

Uploaded Python is parsed, never executed. Native Qiskit, Cirq and PennyLane
subsets are auto-detected. Any package can instead provide a standalone
``ftci_circuit`` literal specification; surrounding Python remains context.
Native source edits preserve supported construction and parameter semantics.
"""
from __future__ import annotations

import ast
from copy import deepcopy
import difflib
import io
import json
import math
import pprint
import re
import tokenize

from .expressions import Expression, _FUNCTIONS
from .quantum import from_circuit


_MAX_SOURCE = 128_000
_ALIASES = {"cnot": "cx", "phase": "p", "i": "id"}
_ANGLE_GATES = {"rx", "ry", "rz", "rxx", "ryy", "rzz", "p"}
_GATES = _ANGLE_GATES | {"id", "h", "x", "y", "z", "s", "sdg", "t", "tdg", "cx", "cz", "swap"}
_TWO_QUBIT = {"rxx", "ryy", "rzz", "cx", "cz", "swap"}
_COORD = re.compile(r"x(0|[1-9][0-9]*)\Z")
_QISKIT_FUNCTIONS = {"sin", "cos", "tan", "exp", "log", "abs", "asin", "acos", "atan", "arcsin", "arccos", "arctan"}


def _format(value):
    if not isinstance(value, str) or value.lower() not in {"json", "python", "qiskit", "cirq", "pennylane"}:
        raise ValueError("Choose JSON or Python source format.")
    return value.lower()


def _text(source):
    if not isinstance(source, str) or not source.strip():
        raise ValueError("Circuit source must be nonempty text.")
    if len(source) > _MAX_SOURCE:
        raise ValueError("Circuit source is limited to 128000 characters.")
    return source


def _validated(spec, default_domain=None):
    if not isinstance(spec, dict):
        raise ValueError("The circuit source must describe an object.")
    try:
        result = deepcopy(spec)
    except RecursionError as error:
        raise ValueError("Circuit data is nested too deeply.") from error
    if "domain" not in result:
        result["domain"] = deepcopy(default_domain or [[-math.pi, math.pi]])
    # Strict JSON compatibility also excludes non-finite values in extra fields.
    try:
        json.dumps(result, allow_nan=False)
    except (ValueError, TypeError, OverflowError, RecursionError) as error:
        raise ValueError("Circuit data must contain finite, JSON-compatible values.") from error
    problem = from_circuit(result)
    result["domain"] = [list(pair) for pair in problem.domain]
    result["observable"] = problem.metadata["observable"]
    return result


def parse_source(format, source, default_domain=None):
    """Read authoritative code into a validated, simulator-compatible circuit.

    JSON and the package-neutral Python specification may include additional
    application metadata. Native readers accept their documented static
    subsets; unsupported statements produce line-number errors.
    """
    format, source = _format(format), _text(source)
    if format == "json":
        try:
            spec = json.loads(source, parse_constant=lambda token: (_ for _ in ()).throw(ValueError(f"Invalid number {token}.")))
        except json.JSONDecodeError as error:
            raise ValueError(f"JSON line {error.lineno}, column {error.colno}: {error.msg}") from error
        except (RecursionError, OverflowError) as error:
            raise ValueError("Circuit JSON is too deeply nested or contains an oversized number.") from error
        return _validated(spec, default_domain)
    kind = _python_kind(source) if format == "python" else format
    if kind == "portable":
        node = _portable_node(_python_tree(source))
        try:
            spec = ast.literal_eval(node)
        except (ValueError, TypeError, SyntaxError, RecursionError, OverflowError) as error:
            raise ValueError(f"Python line {node.lineno}: ftci_circuit must be a literal dictionary; put angle expressions in strings.") from error
        return _validated(spec, default_domain)
    if kind == "cirq":
        from .cirq_source import parse_cirq
        return _validated(parse_cirq(source, default_domain), default_domain)
    if kind == "pennylane":
        from .pennylane_source import parse_pennylane
        return _validated(parse_pennylane(source, default_domain), default_domain)
    return _QiskitReader(source, default_domain).read()


def _python_tree(source):
    try:
        tree = ast.parse(_text(source))
    except (SyntaxError, RecursionError, ValueError) as error:
        raise ValueError(f"Python line {getattr(error, 'lineno', '?')}: {getattr(error, 'msg', 'invalid syntax')}") from error
    if sum(1 for _ in ast.walk(tree)) > 15000 or len(tree.body) > 3000:
        raise ValueError("Python source is too complex for the circuit editor.")
    return tree


def _portable_node(tree):
    declarations = []
    targets = []
    for statement in tree.body:
        if isinstance(statement, ast.Assign):
            names = statement.targets
        elif isinstance(statement, ast.AnnAssign):
            names = [statement.target]
        else:
            continue
        if any(isinstance(target, ast.Name) and target.id == "ftci_circuit" for target in names):
            if len(names) != 1 or statement.value is None:
                raise ValueError(f"Python line {statement.lineno}: use one literal ftci_circuit declaration.")
            declarations.append(statement.value)
            targets.append(names[0])
    writes = [node for node in ast.walk(tree) if isinstance(node, ast.Name)
              and node.id == "ftci_circuit" and isinstance(node.ctx, ast.Store)]
    if not declarations and not writes:
        return None
    if len(declarations) != 1 or len(writes) != 1 or writes[0] not in targets:
        raise ValueError("Declare ftci_circuit exactly once as a top-level literal dictionary.")
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == "ftci_circuit" and node not in targets:
            raise ValueError(f"Python line {node.lineno}: ftci_circuit is a standalone static declaration; edit its literal directly instead of referencing or mutating it.")
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == "ftci_circuit":
            raise ValueError(f"Python line {node.lineno}: reserve ftci_circuit for the static circuit declaration.")
        if isinstance(node, ast.alias) and (node.asname or node.name.split(".")[0]) == "ftci_circuit":
            raise ValueError("Reserve ftci_circuit for the static circuit declaration, not an import name.")

    return declarations[0]


def _python_kind(source):
    tree = _python_tree(source)
    if _portable_node(tree) is not None:
        return "portable"
    imports = {}
    for node in tree.body:
        if isinstance(node, ast.Import):
            for item in node.names:
                imports[item.asname or item.name.split(".")[0]] = item.name if item.asname else item.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
            for item in node.names:
                imports[item.asname or item.name] = f"{node.module}.{item.name}"

    def qualified(node):
        if isinstance(node, ast.Name):
            return imports.get(node.id, node.id)
        if isinstance(node, ast.Attribute):
            base = qualified(node.value)
            return f"{base}.{node.attr}" if base else None
        return None

    constructors = {"qiskit.QuantumCircuit": "qiskit", "cirq.Circuit": "cirq",
                    "pennylane.qnode": "pennylane", "pennylane.device": "pennylane"}
    kinds = {constructors[name] for node in ast.walk(tree) if isinstance(node, ast.Call)
             and (name := qualified(node.func)) in constructors}
    if len(kinds) > 1:
        raise ValueError("Multiple Python circuit packages were detected. Provide one circuit or add a top-level ftci_circuit specification.")
    if not kinds:
        raise ValueError("No supported Python circuit definition found. Use Qiskit, Cirq or PennyLane circuit construction, or add a literal ftci_circuit dictionary for another package.")
    return kinds.pop()


def source_info(format, source):
    """Describe the source contract without importing or executing its code."""
    format = _format(format)
    kind = _python_kind(source) if format == "python" else format
    labels = {"json": "JSON", "qiskit": "Qiskit", "cirq": "Cirq", "pennylane": "PennyLane", "portable": "FTCI specification"}
    return {"kind": kind, "label": labels[kind],
            "notice": "Learning source: ftci_circuit literal" if kind == "portable" else ""}


def _portable_source(circuit, previous_source=None):
    value = pprint.pformat(circuit, sort_dicts=False, width=88)
    if previous_source is None:
        return "# Package-independent FTCI circuit specification\nftci_circuit = " + value + "\n"
    node = _portable_node(_python_tree(previous_source))
    if node is None:
        raise ValueError("No literal ftci_circuit declaration is available for this edit.")
    start, end = _span(previous_source, _offsets(previous_source), node)
    try:
        comments = [token.string for token in tokenize.generate_tokens(io.StringIO(previous_source[start:end]).readline)
                    if token.type == tokenize.COMMENT]
    except (tokenize.TokenError, IndentationError):
        comments = []
    if comments:
        value = "{\n" + "".join("    " + comment + "\n" for comment in comments) + value[1:]
    result = previous_source[:start] + value + previous_source[end:]
    if parse_source("python", result, circuit["domain"]) != circuit:
        raise ValueError("The visual edit could not preserve the Python circuit specification.")
    return result


class _QiskitReader:
    def __init__(self, source, default_domain=None):
        self.source, self.default_domain = source, default_domain
        try:
            self.tree = ast.parse(source)
        except (SyntaxError, RecursionError) as error:
            raise ValueError(f"Qiskit line {getattr(error, 'lineno', '?')}: {getattr(error, 'msg', 'invalid syntax')}") from error
        if sum(1 for _ in ast.walk(self.tree)) > 15000 or len(self.tree.body) > 3000:
            raise ValueError("Qiskit source is too complex for the circuit editor.")
        self.env, self.modules, self.constructors, self.functions = {}, {}, {}, {}
        self.symbols, self.vectors, self.bindings = [], [], {}
        self.gates, self.gate_nodes = [], []
        self.metadata, self.metadata_nodes = {}, {}
        self.qc = None
        self.constructor_node = None
        self.qubit_node = None
        self.active_refs = []
        self.binding_call_seen = False
        self._expr_count = 0

    def fail(self, node, message):
        raise ValueError(f"Qiskit line {getattr(node, 'lineno', '?')}: {message}")

    def _qualified(self, node):
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            base = self._qualified(node.value)
            return f"{base}.{node.attr}" if base else None
        return None

    def _constructor(self, node):
        if isinstance(node, ast.Name):
            return self.constructors.get(node.id)
        qualified = self._qualified(node)
        if qualified:
            root, _, suffix = qualified.partition(".")
            if self.modules.get(root) == "qiskit" and suffix in {"QuantumCircuit", "circuit.Parameter", "circuit.ParameterVector"}:
                return suffix.rsplit(".", 1)[-1]
        return None

    def _expr(self, node, depth=0):
        self._expr_count += 1
        if depth > 45 or self._expr_count > 15000:
            self.fail(node, "Expression is too complex.")
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            try:
                value = float(node.value)
            except OverflowError as error:
                self.fail(node, "Constants must be finite and within the supported numeric range.")
            if not math.isfinite(value):
                self.fail(node, "Constants must be finite.")
            return repr(node.value)
        if isinstance(node, ast.Name):
            value = self.env.get(node.id)
            if isinstance(value, str):
                return value
            if node.id in {"pi", "e"}:
                return node.id
            self.fail(node, f"Unknown scalar {node.id!r}; declare input Parameters before using them.")
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            vector = self.env.get(node.value.id)
            index = self._integer(node.slice)
            if not isinstance(vector, list) or not 0 <= index < len(vector):
                self.fail(node, "ParameterVector index is outside its declared range.")
            return vector[index]
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if self.modules.get(node.value.id) in {"numpy", "math"} and node.attr in {"pi", "e"}:
                return node.attr
        if isinstance(node, ast.UnaryOp) and type(node.op) in (ast.UAdd, ast.USub):
            return f"({'+' if isinstance(node.op, ast.UAdd) else '-'}{self._expr(node.operand, depth+1)})"
        if isinstance(node, ast.BinOp) and type(node.op) in (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow):
            operator = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.Pow: "**"}[type(node.op)]
            value = f"({self._expr(node.left, depth+1)}{operator}{self._expr(node.right, depth+1)})"
            if len(value) > 4096:
                self.fail(node, "Expanded expression exceeds 4096 characters.")
            return value
        if isinstance(node, ast.Call) and not node.keywords:
            function = None
            if isinstance(node.func, ast.Name):
                function = self.functions.get(node.func.id)
                if node.func.id in _FUNCTIONS:
                    function = node.func.id
            elif isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and self.modules.get(node.func.value.id) in {"numpy", "math"}:
                function = node.func.attr
            elif isinstance(node.func, ast.Attribute) and node.func.attr in _FUNCTIONS and not node.args:
                # Qiskit ParameterExpression methods, e.g. x[0].sin().
                function = node.func.attr
                if _FUNCTIONS[function][1] == 1:
                    return f"{function}({self._expr(node.func.value, depth+1)})"
            if function in _FUNCTIONS and len(node.args) == _FUNCTIONS[function][1]:
                return f"{function}({','.join(self._expr(arg, depth+1) for arg in node.args)})"
        self.fail(node, "Use scalar arithmetic and supported math functions; arbitrary Python expressions are not executed.")

    def _number(self, node):
        expression = Expression(self._expr(node))
        if expression.input_indices:
            self.fail(node, "This value must be a fixed number, not an input parameter.")
        return expression([])

    def _integer(self, node):
        if isinstance(node, ast.Constant) and isinstance(node.value, bool):
            self.fail(node, "Use an integer, not a boolean.")
        value = self._number(node)
        if value != int(value):
            self.fail(node, "Use a whole-number index or size.")
        return int(value)

    def _literal(self, node):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, (ast.List, ast.Tuple)):
            return [self._literal(item) for item in node.elts]
        return self._number(node)

    def _import(self, node):
        if isinstance(node, ast.Import):
            for item in node.names:
                if item.name not in {"qiskit", "numpy", "math"}:
                    self.fail(node, "Only qiskit, numpy, and math imports are accepted; imports are not executed.")
                name = item.asname or item.name
                self.modules[name] = item.name
            return
        module = node.module
        if node.level or module not in {"qiskit", "qiskit.circuit", "numpy", "math"}:
            self.fail(node, "This import is outside the editable Qiskit subset.")
        for item in node.names:
            name = item.asname or item.name
            if module == "qiskit" and item.name == "QuantumCircuit":
                self.constructors[name] = item.name
            elif module == "qiskit.circuit" and item.name in {"Parameter", "ParameterVector"}:
                self.constructors[name] = item.name
            elif module in {"numpy", "math"} and item.name in {"pi", "e"}:
                self.env[name] = item.name
            elif module in {"numpy", "math"} and item.name in _FUNCTIONS:
                self.functions[name] = item.name
            else:
                self.fail(node, f"Unsupported import {item.name!r}.")

    def _bind(self, call, inplace):
        if self.binding_call_seen:
            self.fail(call, "Use one assign_parameters call after the final gate in the editable subset.")
        if len(call.args) != 1 or not isinstance(call.args[0], ast.Dict):
            self.fail(call, "assign_parameters needs a literal parameter-to-number dictionary.")
        if any(keyword.arg not in {"inplace", "strict"} or not isinstance(keyword.value, ast.Constant) or not isinstance(keyword.value.value, bool) for keyword in call.keywords):
            self.fail(call, "Only literal inplace/strict=True/False keywords are supported.")
        supplied = next((kw.value.value for kw in call.keywords if kw.arg == "inplace"), False)
        if inplace and not supplied:
            self.fail(call, "An unassigned assign_parameters call needs inplace=True.")
        if not inplace and supplied:
            self.fail(call, "Do not assign the result of inplace=True.")
        strict = next((kw.value.value for kw in call.keywords if kw.arg == "strict"), True)
        used = {i for gate in self.gates if "angle" in gate for i in Expression(gate["angle"]).input_indices}
        if strict:
            for key in call.args[0].keys:
                parameter = self._expr(key)
                if _COORD.fullmatch(parameter) and int(parameter[1:]) not in used:
                    self.fail(key, "assign_parameters includes a parameter no longer used by this circuit. Remove that binding or use strict=False in the code.")
        self._bindings_dict(call.args[0])
        self.binding_call_seen = True

    def _bindings_dict(self, node):
        if not isinstance(node, ast.Dict):
            self.fail(node, "fixed_parameters must be a literal parameter-to-number dictionary.")
        for key, value in zip(node.keys, node.values):
            parameter = self._expr(key)
            if not _COORD.fullmatch(parameter):
                self.fail(key, "Bind an individual declared parameter.")
            self.bindings[int(parameter[1:])] = repr(self._number(value))

    def read(self):
        for statement in self.tree.body:
            if isinstance(statement, (ast.Import, ast.ImportFrom)):
                self._import(statement)
                continue
            if isinstance(statement, ast.Pass):
                continue
            if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant) and isinstance(statement.value.value, str):
                continue  # A module docstring is inert text.
            if isinstance(statement, ast.Assign):
                if len(statement.targets) != 1 or not isinstance(statement.targets[0], ast.Name):
                    self.fail(statement, "Use simple named assignments.")
                name, value = statement.targets[0].id, statement.value
                if name.startswith("__"):
                    self.fail(statement, "Double-underscore names are not accepted.")
                constructor = self._constructor(value.func) if isinstance(value, ast.Call) else None
                if constructor == "QuantumCircuit":
                    if self.qc is not None:
                        self.fail(statement, "Declare one circuit only.")
                    if len(value.args) != 1 or value.keywords:
                        self.fail(value, "Use QuantumCircuit(number_of_qubits) without classical registers.")
                    self.n_qubits = self._integer(value.args[0])
                    self.qc = name
                    self.constructor_node, self.qubit_node = value, value.args[0]
                elif constructor in {"Parameter", "ParameterVector"}:
                    if name in self.env or name in self.modules:
                        self.fail(statement, "Do not overwrite a declared parameter or module alias.")
                    if value.keywords or len(value.args) != (1 if constructor == "Parameter" else 2) or not isinstance(value.args[0], ast.Constant) or not isinstance(value.args[0].value, str):
                        self.fail(value, "Use Parameter('name') or ParameterVector('name', size).")
                    if constructor == "Parameter":
                        index = len(self.symbols)
                        self.symbols.append(name)
                        self.env[name] = f"x{index}"
                    else:
                        size = self._integer(value.args[1])
                        if not 1 <= size <= 32:
                            self.fail(value, "ParameterVector size must be in [1,32].")
                        start = len(self.symbols)
                        refs = [f"{name}[{i}]" for i in range(size)]
                        self.symbols.extend(refs)
                        self.env[name] = [f"x{start+i}" for i in range(size)]
                        self.vectors.append((name, start, size, value.args[1]))
                elif name in {"domain", "observable", "initial_state", "name"}:
                    self.metadata[name] = self._literal(value)
                    self.metadata_nodes[name] = value
                elif name == "fixed_parameters":
                    self._bindings_dict(value)
                elif name == self.qc and isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute) and isinstance(value.func.value, ast.Name) and value.func.value.id == self.qc and value.func.attr == "assign_parameters":
                    self._bind(value, inplace=False)
                elif name in self.modules or name in self.constructors or name == self.qc or name in self.env and isinstance(self.env[name], list):
                    self.fail(statement, "Do not overwrite a circuit, constructor, module, or ParameterVector.")
                else:
                    self.env[name] = self._expr(value)
                continue
            if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
                call = statement.value
                if not isinstance(call.func, ast.Attribute) or not isinstance(call.func.value, ast.Name) or call.func.value.id != self.qc:
                    self.fail(statement, "Only supported calls on the declared quantum circuit are accepted.")
                if call.func.attr == "assign_parameters":
                    self._bind(call, inplace=True)
                    continue
                gate = _ALIASES.get(call.func.attr, call.func.attr)
                if self.binding_call_seen:
                    self.fail(statement, "Place assign_parameters after the final gate; interleaved binding is outside the editable subset.")
                if gate not in _GATES:
                    self.fail(statement, f"Gate or method {call.func.attr!r} is outside the editable subset; use the Python SDK for richer circuits.")
                arity = 2 if gate in _TWO_QUBIT else 1
                nargs = arity + (gate in _ANGLE_GATES)
                if call.keywords or len(call.args) != nargs:
                    self.fail(call, f"{gate} requires {nargs} positional arguments.")
                args = list(call.args)
                descriptor = {"gate": gate}
                if gate in _ANGLE_GATES:
                    descriptor["angle"] = self._expr(args.pop(0))
                descriptor["qubits"] = [self._integer(arg) for arg in args]
                self.gates.append(descriptor)
                self.gate_nodes.append(statement)
                continue
            self.fail(statement, "Only imports, declarations, scalar assignments, and supported gate calls are editable. Loops, functions, and arbitrary Python are not executed.")
        if self.qc is None:
            raise ValueError("Declare a circuit with QuantumCircuit(number_of_qubits).")
        active = [i for i in range(len(self.symbols)) if i not in self.bindings]
        self.active_refs = [self.symbols[i] for i in active]
        remap = {i: f"x{new}" for new, i in enumerate(active)}
        remap.update(self.bindings)
        gates = deepcopy(self.gates)
        for gate_index, gate in enumerate(gates):
            if "angle" in gate:
                gate["angle"] = re.sub(r"\bx(\d+)\b", lambda match: f"({remap[int(match[1])]})", gate["angle"])
                Expression(gate["angle"])
                for node in ast.walk(ast.parse(gate["angle"], mode="eval")):
                    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                            and node.func.id not in _QISKIT_FUNCTIONS
                            and any(Expression(ast.unparse(arg)).input_indices for arg in node.args)):
                        self.fail(self.gate_nodes[gate_index], f"{node.func.id} with symbolic inputs is outside the editable Qiskit subset. Use supported ParameterExpression operations or keep JSON format.")
        domain = self.metadata.get("domain", self.default_domain)
        if domain is None:
            domain = [[-math.pi, math.pi] for _ in range(max(1, len(active)))]
        if len(domain) < len(active):
            raise ValueError(f"Qiskit declares {len(active)} unbound inputs but domain has only {len(domain)} coordinates. Bind fixed parameters or add input bounds.")
        self.result = _validated({"n_qubits": self.n_qubits, "gates": gates,
                                  "observable": self.metadata.get("observable", "Z"*self.n_qubits),
                                  "domain": domain,
                                  **({"name": self.metadata["name"]} if "name" in self.metadata else {}),
                                  **({"initial_state": self.metadata["initial_state"]} if "initial_state" in self.metadata else {})})
        return self.result


def _angle_source(expression, refs, numpy_alias="np"):
    tree = ast.parse(str(expression), mode="eval")
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id not in _QISKIT_FUNCTIONS:
            if any(Expression(ast.unparse(arg)).input_indices for arg in node.args):
                raise ValueError(f"{node.func.id} with symbolic inputs is outside the editable Qiskit subset. Keep JSON format or write a supported Qiskit expression.")

    class Names(ast.NodeTransformer):
        def visit_Name(self, node):
            coordinate = _COORD.fullmatch(node.id)
            if coordinate:
                index = int(coordinate[1])
                if index >= len(refs):
                    raise ValueError(f"Declare input x{index} in the Qiskit code before connecting it in the diagram.")
                return ast.copy_location(ast.parse(refs[index], mode="eval").body, node)
            if node.id in {"pi", "e"}:
                return ast.copy_location(ast.Constant(getattr(math, node.id)), node)
            return node

        def visit_Call(self, node):
            self.generic_visit(node)
            if isinstance(node.func, ast.Name) and node.func.id in _FUNCTIONS:
                node.func = ast.Attribute(value=ast.Name(id=numpy_alias, ctx=ast.Load()), attr=node.func.id, ctx=ast.Load())
            return node

    return ast.unparse(ast.fix_missing_locations(Names().visit(tree))).strip()


def _gate_source(gate, circuit_name, refs, numpy_alias="np"):
    name = _ALIASES.get(gate["gate"], gate["gate"])
    args = [_angle_source(gate["angle"], refs, numpy_alias)] if name in _ANGLE_GATES else []
    args += [str(q) for q in gate["qubits"]]
    return f"{circuit_name}.{name}({', '.join(args)})"


def _offsets(source):
    result, total = [], 0
    for line in source.splitlines(keepends=True):
        result.append(total)
        total += len(line)
    result.append(total)
    return result


def _span(source, offsets, node):
    # Python AST columns count UTF-8 bytes; source slices count code points.
    lines = source.splitlines(keepends=True)
    start = offsets[node.lineno-1] + len(lines[node.lineno-1].encode()[:node.col_offset].decode())
    end = offsets[node.end_lineno-1] + len(lines[node.end_lineno-1].encode()[:node.end_col_offset].decode())
    return start, end


def _gate_key(gate):
    angle = ast.dump(ast.parse(str(gate["angle"]), mode="eval"), include_attributes=False) if "angle" in gate else None
    return _ALIASES.get(gate["gate"], gate["gate"]), tuple(gate["qubits"]), angle


def _patch_qiskit(circuit, previous_source):
    reader = _QiskitReader(_text(previous_source), circuit["domain"])
    old = reader.read()
    refs = list(reader.active_refs)
    offsets = _offsets(previous_source)
    edits = []
    if len(circuit["domain"]) != len(refs):
        if len(reader.vectors) == 1 and not reader.bindings and reader.vectors[0][1] == 0 and reader.vectors[0][2] == len(reader.symbols):
            vector, _, _, node = reader.vectors[0]
            refs = [f"{vector}[{i}]" for i in range(len(circuit["domain"]))]
            edits.append((*_span(previous_source, offsets, node), str(len(refs))))
        elif len(circuit["domain"]) < len(refs):
            raise ValueError("Change the Parameter declarations in Qiskit code before removing input coordinates.")
    numpy_alias = next((name for name, module in reader.modules.items() if module == "numpy"), None)
    prefix = ""
    if numpy_alias is None:
        numpy_alias = "_ftci_np"
        names = {n.id for n in ast.walk(reader.tree) if isinstance(n, ast.Name)}
        while numpy_alias in names:
            numpy_alias += "_"
        # Added only if a rewritten gate actually uses a function.
    spans = [_span(previous_source, offsets, node) for node in reader.gate_nodes]
    old_keys, new_keys = list(map(_gate_key, old["gates"])), list(map(_gate_key, circuit["gates"]))
    matcher = difflib.SequenceMatcher(a=old_keys, b=new_keys, autojunk=False)
    rewritten = []
    for operation, i1, i2, j1, j2 in matcher.get_opcodes():
        if operation == "equal":
            continue
        paired = min(i2-i1, j2-j1)
        for delta in range(paired):
            text = _gate_source(circuit["gates"][j1+delta], reader.qc, refs, numpy_alias)
            rewritten.append(text)
            edits.append((*spans[i1+delta], text))
        for index in range(i1+paired, i2):
            # A no-op preserves semicolon-separated statements and all comments.
            edits.append((*spans[index], "pass"))
        if j2-j1 > paired:
            text = "\n".join(_gate_source(gate, reader.qc, refs, numpy_alias) for gate in circuit["gates"][j1+paired:j2])
            rewritten.append(text)
            if i1+paired < len(spans):
                at = spans[i1+paired][0]
                edits.append((at, at, text + "\n"))
            elif spans:
                at = spans[-1][1]
                edits.append((at, at, "\n" + text))
            else:
                # Insert immediately after construction, before metadata.
                construction = next(n for n in reader.tree.body if isinstance(n, ast.Assign) and n.value is reader.constructor_node)
                at = _span(previous_source, offsets, construction)[1]
                edits.append((at, at, "\n" + text))
    if old["n_qubits"] != circuit["n_qubits"]:
        edits.append((*_span(previous_source, offsets, reader.qubit_node), str(circuit["n_qubits"])))
    suffix = []
    for key in ("domain", "observable", "initial_state", "name"):
        desired = circuit.get(key, "0"*circuit["n_qubits"] if key == "initial_state" else None)
        previous = old.get(key, "0"*old["n_qubits"] if key == "initial_state" else None)
        if desired == previous:
            continue
        if key in reader.metadata_nodes:
            edits.append((*_span(previous_source, offsets, reader.metadata_nodes[key]), repr(desired)))
        else:
            suffix.append(f"{key} = {desired!r}")
    if any(f"{numpy_alias}." in text for text in rewritten) and numpy_alias not in reader.modules:
        prefix = f"import numpy as {numpy_alias}\n"
    # Retain comments inside a replaced multiline gate, not only trailing ones.
    for index, (start, end, replacement) in enumerate(edits):
        segment = previous_source[start:end]
        if "#" in segment:
            try:
                comments = [token.string for token in tokenize.generate_tokens(io.StringIO(segment).readline) if token.type == tokenize.COMMENT]
            except (tokenize.TokenError, IndentationError):
                comments = []
            if comments:
                edits[index] = (start, end, "\n".join(comments) + "\n" + replacement)
    output = previous_source
    for start, end, replacement in sorted(edits, key=lambda edit: (edit[0], edit[1]), reverse=True):
        output = output[:start] + replacement + output[end:]
    output = prefix + output
    if suffix:
        output = output.rstrip() + "\n" + "\n".join(suffix) + "\n"
    check = parse_source("qiskit", output, circuit["domain"])
    if ([ _gate_key(g) for g in check["gates"] ] != new_keys or check["n_qubits"] != circuit["n_qubits"]
            or check["domain"] != circuit["domain"] or check["observable"] != circuit["observable"]
            or check.get("initial_state", "0"*check["n_qubits"]) != circuit.get("initial_state", "0"*circuit["n_qubits"])):
        raise ValueError("This visual edit cannot be written back without changing the Qiskit source semantics. Edit the code directly instead.")
    return output


def serialize_source(format, circuit, previous_source=None):
    """Write a circuit into source, preserving supported Qiskit code on edits.

    The source patcher changes gate-call spans and changed metadata only. It
    leaves imports, declarations, comments and unrelated scalar assignments
    in place. Deleted gate calls become ``pass`` to preserve same-line code.
    Rewrites are reparsed and checked before being returned.
    """
    format = _format(format)
    circuit = _validated(circuit)
    if format == "json":
        return json.dumps(circuit, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    kind = (_python_kind(previous_source) if previous_source is not None else "portable") if format == "python" else format
    if kind == "portable":
        return _portable_source(circuit, previous_source)
    if kind == "cirq":
        from .cirq_source import serialize_cirq
        return serialize_cirq(circuit, previous_source)
    if kind == "pennylane":
        from .pennylane_source import serialize_pennylane
        return serialize_pennylane(circuit, previous_source)
    if any(set(gate) - {"gate", "qubits", "angle"} for gate in circuit["gates"]):
        raise ValueError("This circuit has extra per-gate JSON metadata. Keep JSON format to preserve it.")
    if previous_source is not None:
        return _patch_qiskit(circuit, previous_source)
    # Unsupported metadata is never silently dropped in a format conversion.
    if set(circuit) - {"n_qubits", "gates", "domain", "observable", "initial_state", "name"}:
        raise ValueError("This circuit has extra JSON metadata. Keep JSON format to preserve it.")
    refs = [f"x[{i}]" for i in range(len(circuit["domain"]))]
    lines = ["from qiskit import QuantumCircuit", "from qiskit.circuit import ParameterVector", "import numpy as np", "",
             f"x = ParameterVector('x', {len(refs)})", f"qc = QuantumCircuit({circuit['n_qubits']})"]
    lines.extend(_gate_source(gate, "qc", refs) for gate in circuit["gates"])
    lines += ["", "# FTCI input and expectation-value metadata", f"domain = {circuit['domain']!r}", f"observable = {circuit['observable']!r}"]
    if "initial_state" in circuit:
        lines.append(f"initial_state = {circuit['initial_state']!r}")
    if "name" in circuit:
        lines.append(f"name = {circuit['name']!r}")
    result = "\n".join(lines) + "\n"
    parse_source("qiskit", result)
    return result

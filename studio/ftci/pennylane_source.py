"""Read and edit a bounded PennyLane source subset without executing Python.

Supported sources contain one integer-wire device and one ``@qml.qnode``
function. The function is straight-line scalar preprocessing, supported gates,
and one Pauli expectation value. Other programs can use the Python SDK instead.
No PennyLane import is required to read or write source text.
"""
from __future__ import annotations

import ast
from copy import deepcopy
import difflib
import io
import json
import math
import re
import tokenize

from .expressions import Expression, _FUNCTIONS
from .quantum import from_circuit


_MAX_SOURCE = 128_000
_COORDINATE = re.compile(r"x(0|[1-9][0-9]*)\Z")
_GATE_NAMES = {
    "Identity": "id", "I": "id", "Hadamard": "h", "H": "h",
    "PauliX": "x", "X": "x", "PauliY": "y", "Y": "y", "PauliZ": "z", "Z": "z",
    "S": "s", "T": "t", "RX": "rx", "RY": "ry", "RZ": "rz", "PhaseShift": "p",
    "CNOT": "cx", "CX": "cx", "CZ": "cz", "SWAP": "swap",
    "IsingXX": "rxx", "IsingYY": "ryy", "IsingZZ": "rzz",
}
_SOURCE_NAMES = {"id": "Identity", "h": "Hadamard", "x": "PauliX", "y": "PauliY",
                 "z": "PauliZ", "s": "S", "t": "T", "rx": "RX", "ry": "RY", "rz": "RZ",
                 "p": "PhaseShift", "cx": "CNOT", "cz": "CZ", "swap": "SWAP",
                 "rxx": "IsingXX", "ryy": "IsingYY", "rzz": "IsingZZ"}
_ALIASES = {"i": "id", "cnot": "cx", "phase": "p"}
_ANGLE_GATES = {"rx", "ry", "rz", "p", "rxx", "ryy", "rzz"}
_TWO_QUBIT = {"cx", "cz", "swap", "rxx", "ryy", "rzz"}
_PAULIS = {"Identity": "I", "I": "I", "PauliX": "X", "X": "X",
           "PauliY": "Y", "Y": "Y", "PauliZ": "Z", "Z": "Z"}
_METADATA = {"domain", "observable", "initial_state", "name"}


def _text(source):
    if not isinstance(source, str) or not source.strip():
        raise ValueError("PennyLane source must be nonempty text.")
    if len(source) > _MAX_SOURCE:
        raise ValueError("PennyLane source is limited to 128000 characters.")
    return source


def _validated(spec, default_domain=None):
    if not isinstance(spec, dict):
        raise ValueError("The circuit must be a specification object.")
    try:
        result = deepcopy(spec)
        if "domain" not in result:
            result["domain"] = deepcopy(default_domain or [[-math.pi, math.pi]])
        json.dumps(result, allow_nan=False)
    except (RecursionError, TypeError, ValueError, OverflowError) as error:
        raise ValueError("Circuit data must be finite and JSON-compatible.") from error
    problem = from_circuit(result)
    result["domain"] = [list(pair) for pair in problem.domain]
    result["observable"] = problem.metadata["observable"]
    if "name" in result and not isinstance(result["name"], str):
        raise ValueError("Circuit name must be text.")
    return result


class _PennyLaneReader:
    def __init__(self, source, default_domain=None):
        self.source = _text(source)
        self.default_domain = default_domain
        try:
            self.tree = ast.parse(self.source)
        except (SyntaxError, RecursionError, ValueError) as error:
            raise ValueError(f"PennyLane line {getattr(error, 'lineno', '?')}: {getattr(error, 'msg', 'invalid syntax')}") from error
        if sum(1 for _ in ast.walk(self.tree)) > 15000 or len(self.tree.body) > 3000:
            raise ValueError("PennyLane source is too complex for the circuit editor.")
        self.modules, self.imports, self.env = {}, {}, {}
        self.metadata, self.metadata_nodes = {}, {}
        self.device_name = None
        self.device_node = self.wires_node = self.function = self.return_node = None
        self.gates, self.gate_nodes = [], []
        self.preparation_node = None
        self.prepared_state = None
        self.input_refs = []
        self.vector_argument = None
        self.n_qubits = None
        self._expr_count = 0

    def fail(self, node, message):
        raise ValueError(f"PennyLane line {getattr(node, 'lineno', '?')}: {message}")

    def _qualified(self, node):
        if isinstance(node, ast.Name):
            return self.imports.get(node.id, self.modules.get(node.id))
        if isinstance(node, ast.Attribute):
            base = self._qualified(node.value)
            return f"{base}.{node.attr}" if base else None
        return None

    def _qml_name(self, node):
        value = self._qualified(node)
        if value and value.startswith("pennylane."):
            suffix = value[len("pennylane."):]
            if "." not in suffix:
                return suffix
        return None

    def _expr(self, node, depth=0):
        self._expr_count += 1
        if depth > 45 or self._expr_count > 15000:
            self.fail(node, "Expression is too complex.")
        if isinstance(node, ast.Constant) and type(node.value) in {int, float}:
            try:
                finite = math.isfinite(float(node.value))
            except (OverflowError, ValueError):
                finite = False
            if not finite:
                self.fail(node, "Constants must be finite numbers.")
            return repr(node.value)
        if isinstance(node, ast.Name):
            value = self.env.get(node.id)
            if isinstance(value, str):
                return value
            if node.id in {"pi", "e"}:
                return node.id
            self.fail(node, f"Unknown scalar {node.id!r}; use a qnode input or a fixed scalar assignment.")
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            vector = self.env.get(node.value.id)
            index = self._integer(node.slice)
            if not isinstance(vector, list) or not 0 <= index < len(vector):
                self.fail(node, "Vector index is outside the declared or inferred input range.")
            return vector[index]
        qualified = self._qualified(node)
        if qualified in {f"{module}.{constant}" for module in ("numpy", "math", "pennylane.numpy", "pennylane.math") for constant in ("pi", "e")}:
            return qualified.rsplit(".", 1)[-1]
        if isinstance(node, ast.UnaryOp) and type(node.op) in {ast.UAdd, ast.USub}:
            return f"({'+' if isinstance(node.op, ast.UAdd) else '-'}{self._expr(node.operand, depth+1)})"
        if isinstance(node, ast.BinOp) and type(node.op) in {ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow}:
            operator = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.Pow: "**"}[type(node.op)]
            expression = f"({self._expr(node.left, depth+1)}{operator}{self._expr(node.right, depth+1)})"
            if len(expression) > 4096:
                self.fail(node, "Expanded expression exceeds 4096 characters.")
            return expression
        if isinstance(node, ast.Call) and not node.keywords:
            qualified = self._qualified(node.func)
            function = None
            if qualified and qualified.rsplit(".", 1)[0] in {"numpy", "math", "pennylane.numpy", "pennylane.math"}:
                function = qualified.rsplit(".", 1)[-1]
            elif isinstance(node.func, ast.Name) and node.func.id == "abs":
                function = "abs"
            if function in _FUNCTIONS and len(node.args) == _FUNCTIONS[function][1]:
                return f"{function}({','.join(self._expr(arg, depth+1) for arg in node.args)})"
        self.fail(node, "Use scalar arithmetic and supported math/NumPy functions; uploaded Python is never executed.")

    def _number(self, node):
        try:
            expression = Expression(self._expr(node))
            if expression.input_indices:
                self.fail(node, "This value must be fixed, not input-dependent.")
            return expression([])
        except ValueError as error:
            if str(error).startswith("PennyLane line"):
                raise
            self.fail(node, str(error))

    def _integer(self, node):
        value = self._number(node)
        if value != int(value):
            self.fail(node, "Use a whole-number size or wire index.")
        return int(value)

    def _literal(self, node):
        if isinstance(node, ast.Constant) and (isinstance(node.value, (str, bool)) or node.value is None):
            return node.value
        if isinstance(node, (ast.List, ast.Tuple)):
            return [self._literal(item) for item in node.elts]
        if isinstance(node, ast.Call) and self._qualified(node.func) in {"numpy.array", "pennylane.numpy.array"}:
            if len(node.args) == 1 and all(k.arg == "requires_grad" and isinstance(k.value, ast.Constant) and isinstance(k.value.value, bool) for k in node.keywords):
                return self._literal(node.args[0])
        return self._number(node)

    def _import(self, statement):
        if isinstance(statement, ast.Import):
            for item in statement.names:
                if item.name not in {"pennylane", "numpy", "math"}:
                    self.fail(statement, "Only pennylane, numpy, and math imports are accepted; imports are not executed.")
                alias = item.asname or item.name
                if (alias in self.modules or alias in self.imports or alias in self.env
                        or self.function is not None and alias == self.function.name):
                    self.fail(statement, "Do not overwrite an imported or declared name.")
                self.modules[alias] = item.name
            return
        if statement.module == "__future__" and not statement.level and all(item.name == "annotations" and item.asname is None for item in statement.names):
            return
        if statement.level or statement.module not in {"pennylane", "pennylane.numpy", "pennylane.math", "numpy", "math"}:
            self.fail(statement, "This import is outside the editable PennyLane subset.")
        for item in statement.names:
            alias = item.asname or item.name
            if (alias in self.modules or alias in self.imports or alias in self.env
                    or self.function is not None and alias == self.function.name):
                self.fail(statement, "Do not overwrite an imported or declared name.")
            if statement.module == "pennylane":
                if item.name == "numpy":
                    self.modules[alias] = "pennylane.numpy"
                elif item.name in set(_GATE_NAMES) | {"device", "qnode", "expval", "BasisState", "StatePrep", "adjoint"}:
                    self.imports[alias] = f"pennylane.{item.name}"
                else:
                    self.fail(statement, f"Unsupported PennyLane import {item.name!r}.")
            elif item.name in _FUNCTIONS or item.name in {"pi", "e"}:
                self.imports[alias] = f"{statement.module}.{item.name}"
                if item.name in {"pi", "e"}:
                    self.env[alias] = item.name
            else:
                self.fail(statement, f"Unsupported math import {item.name!r}.")

    def _device(self, call):
        if self.device_node is not None:
            self.fail(call, "Declare one PennyLane device only.")
        keywords = {keyword.arg: keyword.value for keyword in call.keywords}
        if (len(keywords) != len(call.keywords) or set(keywords) - {"wires", "shots"}
                or not 1 <= len(call.args) <= 2):
            self.fail(call, "Use qml.device('default.qubit', wires=integer) without custom device options.")
        if not isinstance(call.args[0], ast.Constant) or call.args[0].value not in {"default.qubit", "lightning.qubit"}:
            self.fail(call, "The editable subset supports default.qubit and lightning.qubit unitary circuits.")
        if len(call.args) == 2 and "wires" in keywords:
            self.fail(call, "Specify the device wire count once.")
        wires = call.args[1] if len(call.args) == 2 else keywords.get("wires")
        if wires is None:
            self.fail(call, "Give an explicit integer device wire count.")
        if "shots" in keywords and not (isinstance(keywords["shots"], ast.Constant) and keywords["shots"].value is None):
            self.fail(call, "Use shots=None for an exact expectation-value source; configure sampling in the learning backend.")
        self.n_qubits = self._integer(wires)
        if not 1 <= self.n_qubits <= 10:
            self.fail(wires, "The exact circuit workspace supports 1 to 10 qubits.")
        self.device_node, self.wires_node = call, wires

    def _qnode(self, function):
        if self.function is not None:
            self.fail(function, "Keep one qnode function in the editable source.")
        if (function.name in self.modules or function.name in self.imports
                or function.name in self.env or function.name == self.device_name
                or function.name.startswith("__")):
            self.fail(function, "Use a distinct qnode name that does not overwrite an import, device or scalar declaration.")
        if len(function.decorator_list) != 1:
            self.fail(function, "Use one @qml.qnode(device) decorator without additional transforms.")
        decorator = function.decorator_list[0]
        if not isinstance(decorator, ast.Call) or self._qml_name(decorator.func) != "qnode" or len(decorator.args) != 1:
            self.fail(function, "Use @qml.qnode(device) for the one editable circuit function.")
        options = {keyword.arg: keyword.value for keyword in decorator.keywords}
        allowed = {"interface", "diff_method", "max_diff", "cache", "cachesize", "grad_on_execution"}
        if len(options) != len(decorator.keywords) or set(options) - allowed:
            self.fail(decorator, "This qnode option changes execution outside the editable subset.")
        for node in options.values():
            self._literal(node)
        device = decorator.args[0]
        if isinstance(device, ast.Call) and self._qml_name(device.func) == "device":
            self._device(device)
        elif not isinstance(device, ast.Name) or device.id != self.device_name:
            self.fail(device, "Use the declared PennyLane device in the qnode decorator.")
        args = function.args
        if args.vararg or args.kwarg or args.kwonlyargs or args.defaults or args.kw_defaults:
            self.fail(function, "Use positional qnode inputs without defaults or *args; put fixed values in scalar assignments.")
        for argument in [*args.posonlyargs, *args.args]:
            if argument.annotation is not None:
                self._annotation(argument.annotation)
        if function.returns is not None:
            self._annotation(function.returns)
        self.function = function

    def _annotation(self, node):
        if isinstance(node, ast.Constant) and (node.value is None or isinstance(node.value, str)):
            return
        if isinstance(node, ast.Name) and node.id in {"float", "int", "list", "tuple"}:
            return
        if self._qualified(node) in {"numpy.ndarray", "pennylane.numpy.ndarray"}:
            return
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id in {"list", "tuple"}:
            items = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
            for item in items:
                self._annotation(item)
            return
        self.fail(node, "Use inert scalar/vector type annotations; annotation expressions are not executed.")

    def _assignment(self, statement, local=False):
        if len(statement.targets) != 1 or not isinstance(statement.targets[0], ast.Name):
            self.fail(statement, "Use simple named scalar assignments.")
        name, value = statement.targets[0].id, statement.value
        if (name.startswith("__") or name in self.modules or name in self.imports or name == self.device_name
                or self.function is not None and name == self.function.name):
            self.fail(statement, "Do not overwrite imported names, devices, or double-underscore names.")
        if not local and name in _METADATA:
            self.metadata[name] = self._literal(value)
            self.metadata_nodes[name] = value
        elif isinstance(value, (ast.List, ast.Tuple)):
            self.env[name] = [self._expr(item) for item in value.elts]
        else:
            self.env[name] = self._expr(value)

    def _inputs(self):
        args = [*self.function.args.posonlyargs, *self.function.args.args]
        names = [arg.arg for arg in args]
        if len(names) != len(set(names)):
            self.fail(self.function, "Qnode input argument names must be distinct.")
        if any(name in self.modules or name in self.imports or name.startswith("__") for name in names):
            self.fail(self.function, "Qnode argument names must not shadow module imports or special names.")
        parents = {child: node for node in ast.walk(self.function) for child in ast.iter_child_nodes(node)}
        vector_sizes, scalar_uses = {}, set()
        for node in ast.walk(self.function):
            if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id in names:
                index = self._integer(node.slice)
                if not 0 <= index < 32:
                    self.fail(node, "Input vector indices must be fixed integers in [0,31].")
                vector_sizes[node.value.id] = max(vector_sizes.get(node.value.id, 0), index + 1)
            elif isinstance(node, ast.Name) and node.id in names:
                parent = parents.get(node)
                if not (isinstance(parent, ast.Subscript) and parent.value is node):
                    scalar_uses.add(node.id)
        if set(vector_sizes) & scalar_uses:
            self.fail(self.function, "Use each qnode argument consistently as a scalar or an indexed vector.")
        domain = self.metadata.get("domain", self.default_domain)
        minimum = sum(vector_sizes.get(name, 1) for name in names)
        if domain is None:
            domain = [[-math.pi, math.pi] for _ in range(max(1, minimum))]
        if not isinstance(domain, (list, tuple)) or len(domain) < minimum:
            self.fail(self.function, f"The qnode needs at least {minimum} input bounds; add domain metadata or bind fixed values outside the qnode.")
        if len(vector_sizes) == 1:
            vector = next(iter(vector_sizes))
            vector_sizes[vector] += len(domain) - minimum
            if len(args) == 1:
                self.vector_argument = vector
        elif len(vector_sizes) > 1 and len(domain) != minimum:
            self.fail(self.function, "Multiple vector arguments require exactly their inferred number of input bounds.")
        coordinate = 0
        for name in names:
            if name in vector_sizes:
                size = vector_sizes[name]
                self.env[name] = [f"x{i}" for i in range(coordinate, coordinate + size)]
                self.input_refs.extend(f"{name}[{i}]" for i in range(size))
                coordinate += size
            else:
                self.env[name] = f"x{coordinate}"
                self.input_refs.append(name)
                coordinate += 1
        self.domain = domain
        self.argument_names = set(names)

    def _call_parts(self, call, angle=False):
        keywords = {keyword.arg: keyword.value for keyword in call.keywords}
        if len(keywords) != len(call.keywords) or set(keywords) - ({"wires", "phi"} if angle else {"wires"}):
            self.fail(call, "Only an angle and explicit wires are supported for this operation.")
        args = list(call.args)
        phi = None
        if angle:
            if args:
                if "phi" in keywords:
                    self.fail(call, "Specify the rotation angle once.")
                phi = args.pop(0)
            else:
                phi = keywords.pop("phi", None)
            if phi is None:
                self.fail(call, "This rotation needs an angle.")
        if args:
            if len(args) != 1 or "wires" in keywords:
                self.fail(call, "Specify wires once, as one integer or a wire list.")
            wires = args[0]
        else:
            wires = keywords.get("wires")
        if wires is None:
            self.fail(call, "An explicit wire or wire list is required.")
        values = list(wires.elts) if isinstance(wires, (ast.List, ast.Tuple)) else [wires]
        indices = [self._integer(value) for value in values]
        if len(set(indices)) != len(indices) or any(not 0 <= wire < self.n_qubits for wire in indices):
            self.fail(wires, "Use distinct wire indices within the declared device range.")
        return phi, indices

    def _gate_name(self, node):
        name = self._qml_name(node)
        if name in _GATE_NAMES:
            return _GATE_NAMES[name]
        if isinstance(node, ast.Call) and self._qml_name(node.func) == "adjoint" and len(node.args) == 1 and not node.keywords:
            name = self._qml_name(node.args[0])
            if name in {"S", "T"}:
                return name.lower() + "dg"
        return None

    def _observable(self, node):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
            left, right = self._observable(node.left), self._observable(node.right)
            if set(left) & set(right):
                self.fail(node, "Use a Pauli tensor product on distinct wires.")
            return {**left, **right}
        if isinstance(node, ast.Call) and self._qml_name(node.func) in _PAULIS:
            _, wires = self._call_parts(node)
            if len(wires) != 1:
                self.fail(node, "Each Pauli observable acts on one wire.")
            return {wires[0]: _PAULIS[self._qml_name(node.func)]}
        self.fail(node, "Return one expval of a Pauli observable or distinct-wire tensor product using @.")

    def _preparation(self, call, statement):
        if self.preparation_node is not None or self.gates:
            self.fail(call, "Use at most one fixed state preparation before the first gate.")
        keywords = {keyword.arg: keyword.value for keyword in call.keywords}
        if len(keywords) != len(call.keywords) or set(keywords) - {"wires"} or len(call.args) not in {1, 2}:
            self.fail(call, "Use a fixed literal state and an explicit all-wire list for state preparation.")
        if len(call.args) == 2 and "wires" in keywords:
            self.fail(call, "Specify preparation wires once.")
        wires_node = call.args[1] if len(call.args) == 2 else keywords.get("wires")
        if not isinstance(wires_node, (ast.List, ast.Tuple)):
            self.fail(call, "State preparation needs every wire in an explicit list.")
        wires = [self._integer(item) for item in wires_node.elts]
        if sorted(wires) != list(range(self.n_qubits)):
            self.fail(call, "State preparation must include each device wire exactly once.")
        values = self._literal(call.args[0])
        if not isinstance(values, list):
            self.fail(call, "State preparation must use a fixed literal vector.")
        if self._qml_name(call.func) == "BasisState":
            if len(values) != self.n_qubits or any(value not in {0, 1} or isinstance(value, bool) for value in values):
                self.fail(call, "BasisState needs one fixed 0 or 1 per wire.")
            bits = ["0"] * self.n_qubits
            for wire, value in zip(wires, values):
                bits[self.n_qubits - 1 - wire] = str(int(value))
            self.prepared_state = "".join(bits)
        else:
            if len(values) != 2 ** self.n_qubits or any(not isinstance(value, (int, float)) or isinstance(value, bool) for value in values):
                self.fail(call, "StatePrep needs a fixed real vector of length 2**n_qubits.")
            output = [0.0] * len(values)
            for index, value in enumerate(values):
                destination = sum(((index >> (self.n_qubits - 1 - position)) & 1) << wire for position, wire in enumerate(wires))
                output[destination] = value
            self.prepared_state = output
        self.preparation_node = statement

    def _body(self):
        self._inputs()
        # Python treats every assigned name as local throughout the function.
        # Remove same-named globals before interpreting the first statement, so
        # a read before local assignment is rejected instead of using a global.
        local_names = {target.id for statement in self.function.body if isinstance(statement, ast.Assign)
                       for target in statement.targets if isinstance(target, ast.Name)}
        for name in local_names - self.argument_names:
            self.env.pop(name, None)
        for statement in self.function.body:
            if self.return_node is not None:
                self.fail(statement, "Place the single expectation-value return at the end of the qnode.")
            if isinstance(statement, ast.Pass):
                continue
            if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant) and isinstance(statement.value.value, str):
                continue
            if isinstance(statement, ast.Assign):
                if any(isinstance(target, ast.Name) and target.id in self.argument_names for target in statement.targets):
                    self.fail(statement, "Do not overwrite qnode arguments; use another name for preprocessing.")
                self._assignment(statement, local=True)
                continue
            if isinstance(statement, ast.Return):
                call = statement.value
                if not isinstance(call, ast.Call) or self._qml_name(call.func) != "expval" or len(call.args) != 1 or call.keywords:
                    self.fail(statement, "Return exactly one qml.expval(Pauli observable); probabilities, samples, states and multiple measurements are unsupported.")
                factors = self._observable(call.args[0])
                self.observable = "".join(factors.get(wire, "I") for wire in reversed(range(self.n_qubits)))
                self.return_node = statement
                continue
            if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
                call = statement.value
                if self._qml_name(call.func) in {"BasisState", "StatePrep"}:
                    self._preparation(call, statement)
                    continue
                gate = self._gate_name(call.func)
                if gate is None:
                    self.fail(statement, "This operation is outside the editable PennyLane gate subset. Decompose it or use the Python SDK.")
                phi, wires = self._call_parts(call, angle=gate in _ANGLE_GATES)
                arity = 2 if gate in _TWO_QUBIT else 1
                if len(wires) != arity:
                    self.fail(call, f"This gate needs {arity} wire(s).")
                descriptor = {"gate": gate, "qubits": wires}
                if phi is not None:
                    descriptor["angle"] = self._expr(phi)
                    try:
                        Expression(descriptor["angle"])
                    except ValueError as error:
                        self.fail(phi, str(error))
                self.gates.append(descriptor)
                self.gate_nodes.append(statement)
                continue
            self.fail(statement, "Only scalar preprocessing, supported gates and one Pauli expectation return are editable. Loops, conditionals and arbitrary Python are not executed.")
        if self.return_node is None:
            self.fail(self.function, "The qnode must return one qml.expval(Pauli observable).")

    def read(self):
        for statement in self.tree.body:
            if isinstance(statement, (ast.Import, ast.ImportFrom)):
                self._import(statement)
            elif isinstance(statement, ast.Pass):
                continue
            elif isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant) and isinstance(statement.value.value, str):
                continue
            elif isinstance(statement, ast.FunctionDef):
                self._qnode(statement)
            elif isinstance(statement, ast.Assign):
                if (len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name)
                        and isinstance(statement.value, ast.Call) and self._qml_name(statement.value.func) == "device"):
                    name = statement.targets[0].id
                    if (name in self.modules or name in self.imports or name in self.env
                            or self.function is not None and name == self.function.name):
                        self.fail(statement, "Use a distinct name for the PennyLane device.")
                    self._device(statement.value)
                    self.device_name = name
                else:
                    self._assignment(statement)
            else:
                self.fail(statement, "Use imports, fixed scalar declarations, one device and one @qml.qnode function. Unsupported Python is not executed.")
        if self.function is None or self.device_node is None:
            raise ValueError("PennyLane source needs one device and one @qml.qnode(device) function.")
        self._body()
        if "observable" in self.metadata and self.metadata["observable"] != self.observable:
            self.fail(self.metadata_nodes["observable"], "Observable metadata must match the qnode's returned Pauli expectation.")
        initial_state = self.metadata.get("initial_state", self.prepared_state)
        result = {"n_qubits": self.n_qubits, "gates": self.gates, "domain": self.domain, "observable": self.observable}
        if initial_state is not None:
            result["initial_state"] = initial_state
        if "name" in self.metadata:
            result["name"] = self.metadata["name"]
        self.result = _validated(result)
        if "initial_state" in self.metadata:
            actual = {"n_qubits": self.n_qubits}
            if self.prepared_state is not None:
                actual["initial_state"] = self.prepared_state
            if _initial_key(self.result) != _initial_key(actual):
                self.fail(self.metadata_nodes["initial_state"], "initial_state metadata must match a qnode state preparation; add qml.BasisState or qml.StatePrep before the gates.")
        return self.result


def parse_pennylane(source, default_domain=None):
    """Translate supported PennyLane Python text into a canonical circuit dict."""
    return _PennyLaneReader(source, default_domain).read()


def _expression_source(expression, refs, numpy_alias):
    tree = ast.parse(str(expression), mode="eval")
    Expression(str(expression))
    class Names(ast.NodeTransformer):
        def visit_Name(self, node):
            coordinate = _COORDINATE.fullmatch(node.id)
            if coordinate:
                index = int(coordinate[1])
                if index >= len(refs):
                    raise ValueError(f"Declare qnode input x{index} before connecting it in the diagram.")
                return ast.copy_location(ast.parse(refs[index], mode="eval").body, node)
            if node.id in {"pi", "e"}:
                return ast.copy_location(ast.Attribute(value=ast.Name(id=numpy_alias, ctx=ast.Load()), attr=node.id, ctx=ast.Load()), node)
            return node
        def visit_Call(self, node):
            self.generic_visit(node)
            if isinstance(node.func, ast.Name) and node.func.id in _FUNCTIONS:
                node.func = ast.Attribute(value=ast.Name(id=numpy_alias, ctx=ast.Load()), attr=node.func.id, ctx=ast.Load())
            return node
    return ast.unparse(ast.fix_missing_locations(Names().visit(tree))).strip()


def _gate_source(gate, qml_alias, refs, numpy_alias):
    name = _ALIASES.get(gate["gate"], gate["gate"])
    if name in {"sdg", "tdg"}:
        constructor = f"{qml_alias}.adjoint({qml_alias}.{name[0].upper()})"
    elif name in _SOURCE_NAMES:
        constructor = f"{qml_alias}.{_SOURCE_NAMES[name]}"
    else:
        raise ValueError(f"Gate {gate['gate']!r} has no supported PennyLane representation.")
    args = [_expression_source(gate["angle"], refs, numpy_alias)] if name in _ANGLE_GATES else []
    wires = gate["qubits"][0] if len(gate["qubits"]) == 1 else gate["qubits"]
    args.append(f"wires={wires!r}")
    return f"{constructor}({', '.join(args)})"


def _observable_source(observable, qml_alias):
    factors = [f"{qml_alias}.Pauli{letter}(wires={wire})" for wire, letter in enumerate(reversed(observable)) if letter != "I"]
    return " @ ".join(factors) if factors else f"{qml_alias}.Identity(wires=0)"


def _preparation_source(initial_state, n_qubits, qml_alias):
    if initial_state is None or initial_state == "0" * n_qubits:
        return None
    wires = list(range(n_qubits))
    if isinstance(initial_state, str):
        return f"{qml_alias}.BasisState({[int(bit) for bit in reversed(initial_state)]!r}, wires={wires!r})"
    values = []
    for index in range(2 ** n_qubits):
        native = int(f"{index:0{n_qubits}b}"[::-1], 2)
        values.append(initial_state[native])
    return f"{qml_alias}.StatePrep({values!r}, wires={wires!r})"


def _offsets(source):
    offsets, total = [], 0
    for line in source.splitlines(keepends=True):
        offsets.append(total)
        total += len(line)
    offsets.append(total)
    return offsets


def _span(source, offsets, node):
    lines = source.splitlines(keepends=True)
    start = offsets[node.lineno-1] + len(lines[node.lineno-1].encode()[:node.col_offset].decode())
    end = offsets[node.end_lineno-1] + len(lines[node.end_lineno-1].encode()[:node.end_col_offset].decode())
    return start, end


def _gate_key(gate):
    angle = ast.dump(ast.parse(str(gate["angle"]), mode="eval"), include_attributes=False) if "angle" in gate else None
    return _ALIASES.get(gate["gate"], gate["gate"]), tuple(gate["qubits"]), angle


def _initial_key(circuit):
    state = circuit.get("initial_state", "0" * circuit["n_qubits"])
    if isinstance(state, str):
        vector = [0.0] * (2 ** circuit["n_qubits"])
        vector[int(state, 2)] = 1.0
        return vector
    return state


def _equivalent(a, b):
    return (a["n_qubits"] == b["n_qubits"] and a["domain"] == b["domain"]
            and a["observable"] == b["observable"] and a.get("name") == b.get("name")
            and list(map(_gate_key, a["gates"])) == list(map(_gate_key, b["gates"]))
            and _initial_key(a) == _initial_key(b))


def _patch(circuit, previous_source):
    reader = _PennyLaneReader(previous_source, circuit["domain"])
    old = reader.read()
    refs = list(reader.input_refs)
    if reader.vector_argument:
        refs = [f"{reader.vector_argument}[{i}]" for i in range(len(circuit["domain"]))]
    elif len(circuit["domain"]) < len(refs):
        raise ValueError("Change the PennyLane qnode argument declarations before removing input coordinates.")
    offsets = _offsets(previous_source)
    lines = previous_source.splitlines(keepends=True)
    indent = re.match(r"[ \t]*", lines[reader.return_node.lineno-1]).group()
    names = {node.id for node in ast.walk(reader.tree) if isinstance(node, ast.Name)}
    qml_alias = next((name for name, module in reader.modules.items() if module == "pennylane"), None)
    numpy_alias = next((name for name, module in reader.modules.items() if module in {"numpy", "pennylane.numpy"}), None)
    prefix = []
    if qml_alias is None:
        qml_alias = "_ftci_qml"
        while qml_alias in names:
            qml_alias += "_"
        prefix.append(f"import pennylane as {qml_alias}")
    if numpy_alias is None:
        numpy_alias = "_ftci_np"
        while numpy_alias in names:
            numpy_alias += "_"
    edits, rewritten = [], []
    spans = [_span(previous_source, offsets, node) for node in reader.gate_nodes]
    old_keys, new_keys = list(map(_gate_key, old["gates"])), list(map(_gate_key, circuit["gates"]))
    matcher = difflib.SequenceMatcher(a=old_keys, b=new_keys, autojunk=False)
    for operation, i1, i2, j1, j2 in matcher.get_opcodes():
        if operation == "equal":
            continue
        paired = min(i2-i1, j2-j1)
        for delta in range(paired):
            text = _gate_source(circuit["gates"][j1+delta], qml_alias, refs, numpy_alias)
            rewritten.append(text)
            edits.append((*spans[i1+delta], text))
        for index in range(i1+paired, i2):
            edits.append((*spans[index], "pass"))
        if j2-j1 > paired:
            text = ("\n" + indent).join(_gate_source(gate, qml_alias, refs, numpy_alias) for gate in circuit["gates"][j1+paired:j2])
            rewritten.append(text)
            at = spans[i1+paired][0] if i1+paired < len(spans) else _span(previous_source, offsets, reader.return_node)[0]
            edits.append((at, at, text + "\n" + indent))
    if old["n_qubits"] != circuit["n_qubits"]:
        edits.append((*_span(previous_source, offsets, reader.wires_node), str(circuit["n_qubits"])))
    if old["observable"] != circuit["observable"]:
        replacement = f"return {qml_alias}.expval({_observable_source(circuit['observable'], qml_alias)})"
        edits.append((*_span(previous_source, offsets, reader.return_node), replacement))
    if _initial_key(old) != _initial_key(circuit) or (old["n_qubits"] != circuit["n_qubits"] and reader.preparation_node is not None):
        preparation = _preparation_source(circuit.get("initial_state"), circuit["n_qubits"], qml_alias)
        if reader.preparation_node is not None:
            edits.append((*_span(previous_source, offsets, reader.preparation_node), preparation or "pass"))
        elif preparation:
            node = reader.gate_nodes[0] if reader.gate_nodes else reader.return_node
            at = _span(previous_source, offsets, node)[0]
            edits.append((at, at, preparation + "\n" + indent))
    suffix = []
    for key in ("domain", "observable", "initial_state", "name"):
        if key == "initial_state":
            desired = circuit.get(key, "0" * circuit["n_qubits"])
            previous = old.get(key, "0" * old["n_qubits"])
        else:
            desired, previous = circuit.get(key), old.get(key)
        if key in reader.metadata_nodes and desired != previous:
            edits.append((*_span(previous_source, offsets, reader.metadata_nodes[key]), repr(desired)))
        elif key not in reader.metadata_nodes and (key == "domain" or key == "name" and desired is not None):
            suffix.append(f"{key} = {desired!r}")
    if any(f"{numpy_alias}." in text for text in rewritten) and numpy_alias not in reader.modules:
        prefix.append(f"from pennylane import numpy as {numpy_alias}")
    for index, (start, end, replacement) in enumerate(edits):
        segment = previous_source[start:end]
        if "#" in segment:
            try:
                comments = [token.string for token in tokenize.generate_tokens(io.StringIO(segment).readline) if token.type == tokenize.COMMENT]
            except (tokenize.TokenError, IndentationError):
                comments = []
            if comments:
                edits[index] = (start, end, ("\n" + indent).join(comments) + "\n" + indent + replacement)
    output = previous_source
    for start, end, replacement in sorted(edits, key=lambda edit: (edit[0], edit[1]), reverse=True):
        output = output[:start] + replacement + output[end:]
    if prefix:
        # Keep module docstrings and any future imports in their original positions.
        top = 0
        for statement in reader.tree.body:
            if isinstance(statement, (ast.Import, ast.ImportFrom)) or isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant) and isinstance(statement.value.value, str):
                top = _span(previous_source, offsets, statement)[1]
            else:
                break
        # Earlier gate/metadata edits do not affect the import region offsets.
        output = output[:top] + "\n" + "\n".join(prefix) + "\n" + output[top:]
    if suffix:
        output = output.rstrip() + "\n\n" + "\n".join(suffix) + "\n"
    check = parse_pennylane(output, circuit["domain"])
    if not _equivalent(check, circuit):
        raise ValueError("This visual edit cannot be written back without changing PennyLane source semantics. Edit the code directly instead.")
    return output


def serialize_pennylane(circuit, previous_source=None):
    """Write a qnode, or preserve existing source while patching visual edits.

    An existing uploaded program is never replaced wholesale. Gate-call spans,
    the returned observable, state preparation and changed metadata are edited;
    the result is parsed again and checked before it is returned.
    """
    circuit = _validated(circuit)
    if set(circuit) - {"n_qubits", "gates", "domain", "observable", "initial_state", "name"}:
        raise ValueError("This circuit has extra JSON metadata. Keep JSON format to preserve it.")
    if any(set(gate) - {"gate", "qubits", "angle"} for gate in circuit["gates"]):
        raise ValueError("This circuit has extra per-gate metadata. Keep JSON format to preserve it.")
    if previous_source is not None:
        return _patch(circuit, _text(previous_source))
    refs = [f"x{i}" for i in range(len(circuit["domain"]))]
    lines = ["import pennylane as qml", "from pennylane import numpy as np", "",
             f"dev = qml.device('default.qubit', wires={circuit['n_qubits']})", "",
             "@qml.qnode(dev)", f"def circuit({', '.join(refs)}):"]
    preparation = _preparation_source(circuit.get("initial_state"), circuit["n_qubits"], "qml")
    if preparation:
        lines.append("    " + preparation)
    lines.extend("    " + _gate_source(gate, "qml", refs, "np") for gate in circuit["gates"])
    lines.append(f"    return qml.expval({_observable_source(circuit['observable'], 'qml')})")
    lines.extend(["", "# FTCI input metadata", f"domain = {circuit['domain']!r}"])
    if "name" in circuit:
        lines.append(f"name = {circuit['name']!r}")
    result = "\n".join(lines) + "\n"
    if not _equivalent(parse_pennylane(result), circuit):
        raise ValueError("This circuit cannot be represented faithfully in the supported PennyLane subset.")
    return result


__all__ = ["parse_pennylane", "serialize_pennylane"]

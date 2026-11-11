(function (global) {
  "use strict";

  const NS = "http://www.w3.org/1998/Math/MathML";
  const ARITY = Object.freeze({
    sin: 1, cos: 1, tan: 1, exp: 1, log: 1, sqrt: 1, abs: 1,
    tanh: 1, sinh: 1, cosh: 1, asin: 1, acos: 1, atan: 1,
    atan2: 2, arcsin: 1, arccos: 1, arctan: 1, log1p: 1,
    expm1: 1, minimum: 2, maximum: 2,
  });
  const NAMES = Object.freeze({asin: "arcsin", acos: "arccos", atan: "arctan", minimum: "min", maximum: "max"});
  const NUMBER = /^(?:0[xX]_?[0-9a-fA-F](?:_?[0-9a-fA-F])*|0[oO]_?[0-7](?:_?[0-7])*|0[bB]_?[01](?:_?[01])*|(?:(?:[0-9](?:_?[0-9])*)\.(?:[0-9](?:_?[0-9])*)?|\.[0-9](?:_?[0-9])*|[0-9](?:_?[0-9])*)(?:[eE][+-]?[0-9](?:_?[0-9])*)?)/;

  class Parser {
    constructor(source) {
      if (typeof source !== "string" || !source.trim() || source.length > 4096) throw new Error("Invalid expression length.");
      this.source = source;
      this.position = 0;
      this.nodes = 0;
      this.next();
    }

    next() {
      while (/\s/.test(this.source[this.position] || "") && this.position < this.source.length) this.position++;
      const rest = this.source.slice(this.position);
      if (!rest) { this.token = {type: "end"}; return; }
      const numeric = NUMBER.exec(rest);
      if (numeric) {
        const raw = numeric[0], clean = raw.replace(/_/g, "");
        if (!Number.isFinite(Number(clean)) || (/^0[0-9]+$/.test(clean) && /[1-9]/.test(clean))) throw new Error("Invalid numeric constant.");
        this.token = {type: "number", value: clean};
        this.position += raw.length;
        return;
      }
      const name = /^[A-Za-z_][A-Za-z_0-9]*/.exec(rest);
      if (name) {
        this.token = {type: "name", value: name[0]};
        this.position += name[0].length;
        return;
      }
      const operation = /^(?:\*\*|[+*/(),-])/.exec(rest);
      if (!operation) throw new Error("Unsupported expression token.");
      this.token = {type: operation[0]};
      this.position += operation[0].length;
    }

    take(type) {
      if (this.token.type !== type) return false;
      this.next();
      return true;
    }

    require(type) {
      if (!this.take(type)) throw new Error("Incomplete expression.");
    }

    node(type, details) {
      if (++this.nodes > 256) throw new Error("Expression has too many nodes.");
      return {type, ...details};
    }

    parse() {
      const tree = this.expression(0);
      if (this.token.type !== "end") throw new Error("Unexpected expression suffix.");
      function validateDepth(node, depth) {
        if (depth > 48) throw new Error("Expression is too deep.");
        for (const child of node.args || (node.type === "binary" ? [node.left, node.right] : node.type === "unary" ? [node.operand] : [])) validateDepth(child, depth + 1);
      }
      validateDepth(tree, 0);
      return tree;
    }

    expression(depth) {
      let left = this.term(depth);
      while (this.token.type === "+" || this.token.type === "-") {
        const op = this.token.type; this.next();
        left = this.node("binary", {op, left, right: this.term(depth)});
      }
      return left;
    }

    term(depth) {
      let left = this.factor(depth);
      while (this.token.type === "*" || this.token.type === "/") {
        const op = this.token.type; this.next();
        left = this.node("binary", {op, left, right: this.factor(depth)});
      }
      return left;
    }

    // Python: unary signs bind below a power's base and above its exponent.
    factor(depth) {
      if (depth > 64) throw new Error("Expression is too deeply nested.");
      if (this.token.type === "+" || this.token.type === "-") {
        const op = this.token.type; this.next();
        return this.node("unary", {op, operand: this.factor(depth + 1)});
      }
      const left = this.atom(depth);
      if (!this.take("**")) return left;
      return this.node("binary", {op: "**", left, right: this.factor(depth + 1)});
    }

    atom(depth) {
      if (this.token.type === "number") {
        const value = this.token.value; this.next();
        return this.node("number", {value});
      }
      if (this.take("(")) {
        const tree = this.expression(depth + 1);
        this.require(")");
        return tree;
      }
      if (this.token.type !== "name") throw new Error("Expected a number, coordinate, or function.");
      const name = this.token.value; this.next();
      if (this.take("(")) {
        if (!Object.prototype.hasOwnProperty.call(ARITY, name)) throw new Error("Unknown function.");
        const args = [];
        if (this.token.type !== ")") {
          args.push(this.expression(depth + 1));
          while (this.take(",")) {
            if (this.token.type === ")") break;
            args.push(this.expression(depth + 1));
          }
        }
        this.require(")");
        if (args.length !== ARITY[name]) throw new Error("Incorrect function arity.");
        return this.node("call", {name, args});
      }
      if (name === "pi" || name === "e") return this.node("constant", {name});
      if (/^x(?:0|[1-9][0-9]*)$/.test(name) && Number(name.slice(1)) < 4096) return this.node("coordinate", {index: name.slice(1)});
      throw new Error("Unknown expression name.");
    }
  }

  function typeset(document, tree) {
    function math(tag, children = [], text) {
      const node = document.createElementNS(NS, tag);
      if (text !== undefined) node.textContent = text;
      for (const child of children) node.appendChild(child);
      return node;
    }
    const symbol = value => math("mo", [], value);
    const row = children => math("mrow", children);
    const fence = value => row([symbol("("), value, symbol(")")]);
    function precedence(node) {
      if (node.type === "number" && !/^0[xob]/i.test(node.value) && /[eE]/.test(node.value)) return 20;
      if (node.type === "unary") return 30;
      if (node.type !== "binary" || node.op === "/") return 50;
      return {"+": 10, "-": 10, "*": 20, "**": 40}[node.op];
    }
    function operand(node, minimum, fenceEqual = false) {
      const value = visit(node), level = precedence(node);
      return level < minimum || (fenceEqual && level === minimum) ? fence(value) : value;
    }
    function numeric(value) {
      if (/^0[xob]/i.test(value)) return math("mn", [], BigInt(value).toString());
      const parts = value.split(/[eE]/);
      if (parts.length === 1) return math("mn", [], value);
      const exponent = parts[1].replace(/^\+/, "");
      const power = exponent[0] === "-" ? row([symbol("−"), math("mn", [], exponent.slice(1))]) : math("mn", [], exponent);
      return row([math("mn", [], parts[0]), symbol("×"), math("msup", [math("mn", [], "10"), power])]);
    }
    function visit(node) {
      if (node.type === "number") return numeric(node.value);
      if (node.type === "coordinate") return math("msub", [math("mi", [], "x"), math("mn", [], node.index)]);
      if (node.type === "constant") {
        const value = math("mi", [], node.name === "pi" ? "π" : "e");
        value.setAttribute("mathvariant", "normal");
        return value;
      }
      if (node.type === "unary") return row([symbol(node.op === "-" ? "−" : "+"), operand(node.operand, 30, true)]);
      if (node.type === "binary") {
        if (node.op === "/") return math("mfrac", [visit(node.left), visit(node.right)]);
        if (node.op === "**") return math("msup", [operand(node.left, 40, true), visit(node.right)]);
        const level = node.op === "*" ? 20 : 10;
        return row([operand(node.left, level), symbol(node.op === "*" ? "·" : node.op === "-" ? "−" : "+"), operand(node.right, level, true)]);
      }
      if (node.name === "sqrt") return math("msqrt", [visit(node.args[0])]);
      if (node.name === "abs") return row([symbol("|"), visit(node.args[0]), symbol("|")]);
      const name = math("mi", [], NAMES[node.name] || node.name);
      name.setAttribute("mathvariant", "normal");
      const args = [];
      node.args.forEach((arg, index) => { if (index) args.push(symbol(",")); args.push(visit(arg)); });
      return row([name, symbol("⁡"), fence(row(args))]);
    }
    return visit(tree);
  }

  function render(container, expression) {
    let source = "";
    try {
      source = typeof expression === "string" ? expression : String(expression ?? "");
      if (typeof expression !== "string") throw new Error("An expression must be text.");
      const tree = new Parser(expression).parse();
      const document = container.ownerDocument || global.document;
      const formula = document.createElementNS(NS, "math");
      formula.setAttribute("display", "inline");
      formula.setAttribute("displaystyle", "true");
      formula.setAttribute("aria-label", source);
      formula.appendChild(typeset(document, tree));
      container.replaceChildren(formula);
      return true;
    } catch {
      try { container.textContent = source; } catch { /* An absent container cannot display fallback text. */ }
      return false;
    }
  }

  const api = Object.freeze({render});
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  global.FTCIFormula = api;
})(typeof window !== "undefined" ? window : globalThis);

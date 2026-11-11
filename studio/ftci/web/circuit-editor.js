(function (global) {
  "use strict";

  const GATES = {
    h: { label: "H", description: "Hadamard", arity: 1 },
    x: { label: "X", description: "Pauli X", arity: 1 },
    y: { label: "Y", description: "Pauli Y", arity: 1 },
    z: { label: "Z", description: "Pauli Z", arity: 1 },
    rx: { label: "RX", description: "X rotation", arity: 1, angle: true },
    ry: { label: "RY", description: "Y rotation", arity: 1, angle: true },
    rz: { label: "RZ", description: "Z rotation", arity: 1, angle: true },
    cx: { label: "CX", description: "Controlled X", arity: 2, control: true },
    cz: { label: "CZ", description: "Controlled Z", arity: 2, control: true },
    rxx: { label: "RXX", description: "XX rotation", arity: 2, angle: true },
    ryy: { label: "RYY", description: "YY rotation", arity: 2, angle: true },
    rzz: { label: "RZZ", description: "ZZ rotation", arity: 2, angle: true },
    p: { label: "P", description: "Phase rotation", arity: 1, angle: true },
    swap: { label: "SWAP", description: "Swap qubits", arity: 2 },
    s: { label: "S", description: "S phase", arity: 1 },
    sdg: { label: "S†", description: "Inverse S", arity: 1 },
    t: { label: "T", description: "T phase", arity: 1 },
    tdg: { label: "T†", description: "Inverse T", arity: 1 },
    i: { label: "I", description: "Identity", arity: 1 },
  };
  const FUNCTIONS = { sin: 1, cos: 1, tan: 1, exp: 1, log: 1, sqrt: 1, abs: 1,
    tanh: 1, sinh: 1, cosh: 1, asin: 1, acos: 1, atan: 1, atan2: 2,
    arcsin: 1, arccos: 1, arctan: 1, log1p: 1, expm1: 1, minimum: 2, maximum: 2 };
  const COLORS = ["#bd513c", "#397991", "#92703d", "#7265a0", "#4e8060", "#a3547c"];
  const copy = (value) => JSON.parse(JSON.stringify(value));
  const gateName = (name) => ({ cnot: "cx", phase: "p", id: "i" }[String(name).toLowerCase()] || String(name).toLowerCase());
  const pretty = (text) => String(text).replace(/\bx(\d+)\b/g, (_, digits) => `x${digits.replace(/\d/g, d => "₀₁₂₃₄₅₆₇₈₉"[d])}`).replace(/\*\*2\b/g, "²").replace(/\*\*3\b/g, "³").replace(/\*/g, "·");
  const dependencies = (text) => [...new Set((String(text).match(/\bx(?:0|[1-9][0-9]*)\b/g) || []).map(name => Number(name.slice(1))))].sort((a, b) => a - b);

  // Syntax validation only. Expressions are never evaluated in the browser.
  // The engine performs its own authoritative safe-expression validation.
  function validateExpression(text, inputCount) {
    if (typeof text !== "string" || !text.trim()) throw new Error("Enter an angle or preprocessing expression.");
    if (text.length > 4096) throw new Error("The expression is too long (4096 characters maximum).");
    const tokens = [];
    const pattern = /\s+|(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|[A-Za-z_][A-Za-z_0-9]*|\*\*|[+\-*/(),]/gy;
    let cursor = 0;
    while (cursor < text.length) {
      pattern.lastIndex = cursor;
      const match = pattern.exec(text);
      if (!match) throw new Error(`Unsupported expression character near “${text.slice(cursor, cursor + 10)}”. Use ** for powers.`);
      cursor = pattern.lastIndex;
      if (!/^\s+$/.test(match[0])) tokens.push(match[0]);
    }
    let at = 0, nodes = 0;
    const used = new Set();
    function parse(minimum = 0, depth = 0) {
      if (++nodes > 256 || depth > 48) throw new Error("The expression is too complex.");
      let token = tokens[at++];
      if (token === "+" || token === "-") parse(25, depth + 1);
      else if (token === "(") {
        parse(0, depth + 1);
        if (tokens[at++] !== ")") throw new Error("Close the expression's parentheses.");
      } else if (token && /^(?:\d|\.)/.test(token)) {
        if (!Number.isFinite(Number(token))) throw new Error("Numeric constants must be finite.");
      } else if (token && /^(x(?:0|[1-9][0-9]*))$/.test(token)) {
        const index = Number(token.slice(1));
        if (index >= inputCount) throw new Error(`${token} is not an available input. Add its input domain first.`);
        used.add(index);
      } else if (token === "pi" || token === "e") {
        // Named scalar constants.
      } else if (Object.prototype.hasOwnProperty.call(FUNCTIONS, token)) {
        if (tokens[at++] !== "(") throw new Error(`Write ${token}(…) with parentheses.`);
        for (let arg = 0; arg < FUNCTIONS[token]; arg++) {
          if (arg && tokens[at++] !== ",") throw new Error(`${token} needs ${FUNCTIONS[token]} arguments.`);
          parse(0, depth + 1);
        }
        if (tokens[at++] !== ")") throw new Error(`${token} needs ${FUNCTIONS[token]} argument(s) and a closing parenthesis.`);
      } else throw new Error(token ? `Unknown expression name or token “${token}”.` : "The expression is incomplete.");
      while (at < tokens.length) {
        const op = tokens[at];
        const precedence = { "+": 10, "-": 10, "*": 20, "/": 20, "**": 30 }[op];
        if (precedence === undefined || precedence < minimum) break;
        at++;
        parse(op === "**" ? precedence : precedence + 1, depth + 1);
      }
    }
    parse();
    if (at !== tokens.length) throw new Error(`Unexpected token “${tokens[at]}” in the expression.`);
    return [...used].sort((a, b) => a - b);
  }

  function validateDraft(draft, nQubits, inputCount) {
    const name = gateName(draft.gate);
    const info = GATES[name];
    if (!info) throw new Error("Choose a supported gate.");
    const qubits = draft.qubits;
    if (!Array.isArray(qubits) || qubits.length !== info.arity || qubits.some(q => typeof q !== "number" || !Number.isInteger(q) || q < 0 || q >= nQubits) || new Set(qubits).size !== info.arity) {
      throw new Error(info.arity === 2 ? "Choose two distinct qubits within this circuit." : "Choose a valid target qubit.");
    }
    const descriptor = { gate: name, qubits };
    if (info.angle) {
      const angle = String(draft.angle ?? "").trim();
      const used = validateExpression(angle, inputCount);
      if (draft.mode === "fixed" && used.length) throw new Error("A fixed angle cannot refer to inputs. Choose Input encoding to use x0, x1, ….");
      if (draft.mode === "encoding" && !used.length) throw new Error("An input encoding needs an input such as x0. Choose Fixed angle for a constant rotation.");
      descriptor.angle = angle;
    }
    return descriptor;
  }

  function validateCircuit(spec, domain = spec && spec.domain) {
    if (!spec || typeof spec !== "object" || Array.isArray(spec)) throw new Error("The circuit must be an object.");
    const n = spec.n_qubits;
    if (!Number.isInteger(n) || n < 1 || n > 10) throw new Error("Choose between 1 and 10 qubits.");
    if (!Array.isArray(domain) || !domain.length || domain.some(pair => !Array.isArray(pair) || pair.length !== 2 || !Number.isFinite(Number(pair[0])) || !Number.isFinite(Number(pair[1])) || Number(pair[0]) >= Number(pair[1]))) throw new Error("Set finite, increasing input domains.");
    if (!Array.isArray(spec.gates) || spec.gates.length > 1000) throw new Error("A circuit can contain at most 1000 gates.");
    const observable = String(spec.observable === undefined ? "Z".repeat(n) : spec.observable).trim().toUpperCase();
    if (observable.length !== n || /[^IXYZ]/.test(observable)) throw new Error(`The observable needs ${n} letters from I, X, Y, Z.`);
    const normalized = { ...copy(spec), n_qubits: n, observable, domain: copy(domain), gates: [] };
    spec.gates.forEach((gate, index) => {
      if (!gate || typeof gate !== "object") throw new Error(`Gate ${index + 1} is invalid.`);
      const name = gateName(gate.gate);
      if (!GATES[name]) throw new Error(`Gate ${index + 1} is not supported by this editor.`);
      if (!GATES[name].angle && Object.prototype.hasOwnProperty.call(gate, "angle")) throw new Error(`Gate ${index + 1} (${GATES[name].label}) does not take an angle.`);
      try {
        const checked = validateDraft({ ...gate, gate: name, mode: dependencies(gate.angle).length ? "encoding" : "fixed" }, n, domain.length);
        normalized.gates.push({ ...copy(gate), ...checked });
      } catch (error) { throw new Error(`Gate ${index + 1}: ${error.message}`); }
    });
    if (typeof spec.initial_state === "string" && (spec.initial_state.length !== n || /[^01]/.test(spec.initial_state))) throw new Error(`Initial state needs ${n} bits, e.g. ${"0".repeat(n)}.`);
    return normalized;
  }

  class CircuitHistory {
    constructor(initial) { this.current = copy(initial); this.past = []; this.future = []; }
    commit(next) {
      if (JSON.stringify(next) === JSON.stringify(this.current)) return false;
      this.past.push(copy(this.current));
      if (this.past.length > 80) this.past.shift();
      this.current = copy(next); this.future = [];
      return true;
    }
    undo() {
      if (!this.past.length) return false;
      this.future.push(copy(this.current)); this.current = this.past.pop(); return true;
    }
    redo() {
      if (!this.future.length) return false;
      this.past.push(copy(this.current)); this.current = this.future.pop(); return true;
    }
  }

  // Wires a gate occupies in the drawing: everything between its lowest and highest qubit.
  function gateSpan(gate) { return [Math.min(...gate.qubits), Math.max(...gate.qubits)]; }
  function spansOverlap(a, b) { const [lo, hi] = gateSpan(a), [low, high] = gateSpan(b); return lo <= high && low <= hi; }

  // Left-aligned layers: every gate sits in the first column after the gates before it on its wires,
  // so gates on different wires share a column.
  function layoutColumns(gates) {
    const frontier = [], columns = [];
    let count = 0;
    for (const gate of gates) {
      const [lo, hi] = gateSpan(gate);
      let column = 0;
      for (let q = lo; q <= hi; q++) column = Math.max(column, frontier[q] || 0);
      for (let q = lo; q <= hi; q++) frontier[q] = column + 1;
      columns.push(column);
      count = Math.max(count, column + 1);
    }
    return { columns, count };
  }

  // Insert a gate so that it follows the gates in columns before `gap` on its wires and precedes the
  // others. Gates on other wires commute with it, so the list is only reordered when no single
  // position separates the two groups.
  function insertAtGap(gates, gate, gap) {
    const { columns } = layoutColumns(gates);
    let lastBefore = -1, firstAfter = gates.length;
    gates.forEach((other, index) => {
      if (!spansOverlap(gate, other)) return;
      if (columns[index] < gap) lastBefore = Math.max(lastBefore, index);
      else firstAfter = Math.min(firstAfter, index);
    });
    if (lastBefore < firstAfter) {
      const next = gates.slice();
      next.splice(firstAfter, 0, gate);
      return { gates: next, index: firstAfter };
    }
    const before = gates.filter((_, index) => columns[index] < gap);
    return { gates: [...before, gate, ...gates.filter((_, index) => columns[index] >= gap)], index: before.length };
  }

  // Qubits of a gate placed with its first qubit on `qubit`.
  function placeQubits(arity, qubit, nQubits, offset = 1) {
    if (arity === 1) return [qubit];
    if (nQubits < 2) throw new Error("This gate needs two qubits. Add a qubit first.");
    for (const step of [offset, -offset, 1, -1]) {
      if (qubit + step >= 0 && qubit + step < nQubits) return [qubit, qubit + step];
    }
    throw new Error("This gate does not fit on the circuit.");
  }

  // Move a gate to a slot of the current drawing, given as the gap before a column and a wire.
  function moveToGap(gates, index, gap, qubit, nQubits) {
    const { columns } = layoutColumns(gates);
    const moving = gates[index];
    let anchor = null;
    gates.forEach((other, position) => {
      const [lo, hi] = gateSpan(other);
      if (position === index || qubit < lo || qubit > hi || columns[position] >= gap) return;
      if (anchor === null || columns[position] > columns[anchor]) anchor = position;
    });
    const rest = gates.filter((_, position) => position !== index);
    const restColumns = layoutColumns(rest).columns;
    const restGap = anchor === null ? 0 : restColumns[anchor > index ? anchor - 1 : anchor] + 1;
    const offset = moving.qubits.length === 2 ? moving.qubits[1] - moving.qubits[0] : 1;
    const moved = { ...moving, qubits: placeQubits(moving.qubits.length, qubit, nQubits, offset) };
    return insertAtGap(rest, moved, restGap);
  }

  const PALETTE = [["Fixed", ["h", "x", "y", "z", "s", "sdg", "t", "tdg"]], ["Rotation", ["rx", "ry", "rz", "p"]],
    ["Two-qubit", ["cx", "cz", "swap", "rxx", "ryy", "rzz"]]];
  const PITCH = 92, TOP = 100;
  const wireY = (qubit) => TOP + qubit * PITCH;

  const helpers = { GATES, dependencies, validateExpression, validateDraft, validateCircuit, CircuitHistory,
    layoutColumns, insertAtGap, moveToGap, placeQubits };
  if (typeof module !== "undefined" && module.exports) module.exports = helpers;
  if (typeof document === "undefined") return;

  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }
  function button(text, className, callback, label) {
    const node = el("button", className, text); node.type = "button";
    if (label) node.setAttribute("aria-label", label);
    if (callback) node.addEventListener("click", callback);
    return node;
  }
  function select(options, label) {
    const node = el("select"); node.setAttribute("aria-label", label);
    options.forEach(([value, text]) => { const option = el("option", "", text); option.value = value; node.append(option); });
    return node;
  }
  function field(label, input, help) {
    const wrap = el("div", "ce-field"); const lab = el("label", "", label);
    const id = `ce-field-${++fieldCounter}`;
    input.id = id; lab.htmlFor = id; wrap.append(lab, input);
    if (help) wrap.append(el("p", "ce-help", help));
    return wrap;
  }
  let fieldCounter = 0;

  class FTCICircuitEditor {
    constructor(container, { onChange = () => {}, onInspect = () => {} } = {}) {
      this.root = container;
      this.root.classList.add("ce-editor");
      this.onChange = onChange; this.onInspect = onInspect;
      this.domain = [[-Math.PI, Math.PI]];
      this.disabled = false; this.selected = null; this.activeInput = null;
      this.armed = null; this.pendingSlot = null; this.dragSource = null;
      this.draftDirty = false; this.settingsDirty = false; this.inspection = null; this.inspectionError = "";
      this._build();
      this.setCircuit({ n_qubits: 2, gates: [], observable: "ZZ", domain: this.domain });
    }

    _build() {
      const heading = el("div", "ce-heading");
      const title = el("div"); title.append(el("h2", "", "Circuit"));
      const history = el("div", "ce-history");
      this.undoButton = button("↶ Undo", "ce-tool", () => this._history("undo"), "Undo circuit edit");
      this.redoButton = button("↷ Redo", "ce-tool", () => this._history("redo"), "Redo circuit edit");
      history.append(this.undoButton, this.redoButton); heading.append(title, history);
      this.legend = el("div", "ce-legend"); this.legend.setAttribute("aria-label", "Circuit input connections");
      const body = el("div", "ce-body");
      const drawing = el("div", "ce-drawing");
      this.scroll = el("div", "ce-scroll"); this.scroll.tabIndex = 0; this.scroll.setAttribute("aria-label", "Editable quantum circuit, scroll horizontally for all gates");
      this.track = el("div", "ce-track"); this.scroll.append(this.track);
      this.palette = el("div", "ce-palette"); this.palette.setAttribute("aria-label", "Gates to add");
      for (const [title, names] of PALETTE) {
        const group = el("div", "ce-palette-group");
        group.append(el("span", "ce-palette-label", title));
        for (const name of names) {
          const item = button(GATES[name].label, "ce-palette-gate", () => this._arm(name), `Add ${GATES[name].description}`);
          item.title = GATES[name].description; item.dataset.gate = name; item.draggable = true;
          item.addEventListener("dragstart", event => this._dragStart(event, { kind: "new", gate: name }));
          item.addEventListener("dragend", () => this._dragEnd());
          group.append(item);
        }
        this.palette.append(group);
      }
      this.paletteHint = el("p", "ce-palette-hint"); this.paletteHint.setAttribute("aria-live", "polite");
      this.palette.append(this.paletteHint);
      this.root.addEventListener("keydown", event => this._keydown(event));
      this.settings = el("details", "ce-settings");
      this.settings.append(el("summary", "", "Qubits, initial state & measurement"));
      this.settingsBody = el("div", "ce-settings-body"); this.settings.append(this.settingsBody);
      this.mapping = el("div", "ce-encoding-map");
      drawing.append(this.palette, this.scroll, this.mapping, this.settings);
      this.inspector = el("aside", "ce-inspector"); this.inspector.setAttribute("aria-label", "Selected gate settings");
      body.append(drawing, this.inspector);
      this.error = el("div", "ce-error"); this.error.setAttribute("role", "alert"); this.error.hidden = true;
      this.validation = el("div", "ce-validation"); this.validation.setAttribute("aria-live", "polite");
      this.root.replaceChildren(heading, this.legend, body, this.error, this.validation);
    }

    setCircuit(spec, { resetHistory = true } = {}) {
      const domain = spec.domain || this.domain;
      const normalized = validateCircuit(spec, domain);
      this.domain = copy(domain);
      if (resetHistory || !this.store) this.store = new CircuitHistory(normalized);
      else this.store.commit(normalized);
      this.selected = normalized.gates.length ? 0 : null;
      this.pendingSlot = null;
      this.draftDirty = false; this.settingsDirty = false; this.inspection = null;
      this._showError(""); this._render();
    }

    setDomain(domain) {
      this.domain = copy(domain);
      this._renderLegend(); this._renderTrack();
      if (this.draft) this._updateDraftPreview();
    }

    getCircuit() {
      if (this.settingsDirty) throw new Error("Apply or discard the pending visual circuit settings first.");
      if (this.draftDirty) throw new Error("Apply or discard the pending visual gate changes first.");
      if (this.inspectionError) throw new Error(`Correct the circuit before learning: ${this.inspectionError}`);
      return validateCircuit({ ...copy(this.store.current), domain: copy(this.domain) }, this.domain);
    }

    setDisabled(disabled) {
      this.disabled = Boolean(disabled);
      for (const node of this.root.querySelectorAll("button,input,select,textarea")) node.disabled = this.disabled;
      if (!this.disabled) this._syncButtons();
      this.root.classList.toggle("ce-disabled", this.disabled);
    }

    // Marks each encoding gate with the dictionary it belongs to: {gateIndex: {label, color}}.
    setSiteMarks(marks) {
      this.siteMarks = marks || {};
      this._renderTrack();
    }

    setInspection(data, error) {
      this.inspection = error ? null : data;
      this.inspectionError = error ? String(error.message || error) : "";
      this._renderValidation(); this._renderDictionary();
    }

    _showError(message) { this.error.textContent = message; this.error.hidden = !message; }
    _beforeNavigate() {
      if (this.disabled) return false;
      if (this.draftDirty || this.settingsDirty) {
        this._showError("Apply or discard pending changes first.");
        return false;
      }
      this._showError(""); return true;
    }
    _emit() {
      this.inspection = null; this.inspectionError = ""; this.siteMarks = {};
      this._renderValidation();
      const spec = { ...copy(this.store.current), domain: copy(this.domain) };
      this.onChange(spec); this.onInspect(spec);
    }
    _commit(spec, selected = this.selected) {
      const normalized = validateCircuit(spec, this.domain);
      this.store.commit(normalized); this.selected = selected;
      this.draftDirty = false; this.settingsDirty = false;
      this.inspection = null; this.inspectionError = ""; this.siteMarks = {};
      this._showError(""); this._render(); this._emit();
    }
    _history(direction) {
      if (!this._beforeNavigate()) return;
      if (!this.store[direction]()) return;
      this.selected = this.store.current.gates.length ? Math.min(this.selected || 0, this.store.current.gates.length - 1) : null;
      this.inspection = null; this.inspectionError = ""; this.siteMarks = {};
      this._render(); this._emit();
    }
    _syncButtons() {
      if (this.disabled) return;
      this.undoButton.disabled = !this.store.past.length;
      this.redoButton.disabled = !this.store.future.length;
      if (this.earlierButton) this.earlierButton.disabled = !this.draft || this._neighbour(-1) === null;
      if (this.laterButton) this.laterButton.disabled = !this.draft || this._neighbour(1) === null;
    }
    _render() {
      this._renderPalette(); this._renderLegend(); this._renderTrack(); this._renderSettings(); this._renderInspector(); this._renderValidation(); this._syncButtons();
      if (this.disabled) this.setDisabled(true);
    }

    _encodingInfo() {
      let occurrence = 0;
      return this.store.current.gates.map((gate, index) => {
        const inputs = dependencies(gate.angle);
        return { gate, index, inputs, encoding: inputs.length ? ++occurrence : null };
      });
    }
    _chip(index, text = `x${index}`) {
      const chip = el("span", "ce-input-chip", text);
      chip.style.setProperty("--input-color", COLORS[index % COLORS.length]);
      return chip;
    }
    _renderLegend() {
      this.legend.replaceChildren();
      const info = this._encodingInfo();
      for (let index = 0; index < this.domain.length; index++) {
        const connections = info.filter(item => item.inputs.includes(index));
        const row = el("div", "ce-input-connection");
        const inputButton = button(`x${index}`, "ce-input-filter", () => {
          this.activeInput = this.activeInput === index ? null : index;
          this._renderLegend(); this._renderTrack();
        }, `Highlight gates using x${index}`);
        inputButton.style.setProperty("--input-color", COLORS[index % COLORS.length]);
        inputButton.setAttribute("aria-pressed", String(this.activeInput === index));
        row.append(inputButton, el("span", "ce-connection-arrow", "→"));
        if (connections.length) {
          for (const connection of connections) {
            const link = button(`E${connection.encoding} · ${GATES[gateName(connection.gate.gate)].label}`, "ce-encoding-link", () => this._select(connection.index), `Select encoding E${connection.encoding} using x${index}`);
            row.append(link);
          }
        } else row.append(el("span", "ce-unused", "not connected"));
        this.legend.append(row);
      }
      if (this.activeInput !== null && this.activeInput >= this.domain.length) this.activeInput = null;
    }

    _renderTrack() {
      const spec = this.store.current, n = spec.n_qubits;
      const height = wireY(n - 1) + 52;
      const labels = el("div", "ce-wire-labels"); labels.style.height = `${height}px`;
      labels.append(el("span", "ce-step-label", "QUBITS"));
      for (let q = 0; q < n; q++) {
        const row = el("div", "ce-wire-label"); row.style.top = `${wireY(q)}px`;
        row.append(el("span", "", `q${q}`), el("small", "", spec.observable[n - 1 - q]));
        labels.append(row);
      }
      this.track.replaceChildren(labels);
      const items = this._encodingInfo();
      const { columns, count } = layoutColumns(spec.gates);
      this.columns = columns;
      for (let c = 0; c <= count; c++) {
        const gap = el("div", "ce-gap-column"); gap.style.height = `${height}px`;
        for (let q = 0; q < n; q++) gap.append(this._wire(q), this._slot({ gap: c, qubit: q }, "ce-gap-add"));
        this.track.append(gap);
        if (c === count) break;
        const members = items.filter(item => columns[item.index] === c);
        const column = el("div", `ce-column${members.some(item => item.encoding) ? " ce-encoding-column" : ""}`);
        column.style.height = `${height}px`;
        column.append(el("span", "ce-step-label ce-column-number", String(c + 1).padStart(2, "0")));
        const covered = new Set();
        for (const item of members) { const [lo, hi] = gateSpan(item.gate); for (let q = lo; q <= hi; q++) covered.add(q); }
        for (let q = 0; q < n; q++) {
          column.append(this._wire(q));
          if (!covered.has(q)) column.append(this._slot({ gap: c, qubit: q }, "ce-wire-insert"));
        }
        for (const item of members) this._renderGate(column, item);
        this.track.append(column);
      }
      if (!spec.gates.length) this.track.append(el("p", "ce-empty-circuit", "No gates yet. Choose a gate above, then click a + on a wire."));
      this.root.classList.toggle("ce-armed", Boolean(this.armed || this.pendingSlot));
      this._renderMapping();
    }

    _wire(qubit) {
      const wire = el("span", "ce-wire"); wire.style.top = `${wireY(qubit)}px`;
      return wire;
    }

    // A place on a wire where a gate can be added, moved or dropped.
    _slot(slot, className) {
      const pending = this.pendingSlot && this.pendingSlot.gap === slot.gap && this.pendingSlot.qubit === slot.qubit;
      const node = button("+", `${className}${pending ? " ce-slot-pending" : ""}`, () => this._slotClicked(slot), `Add a gate on q${slot.qubit}, before column ${slot.gap + 1}`);
      node.style.top = `${wireY(slot.qubit)}px`;
      node.setAttribute("aria-pressed", String(Boolean(pending)));
      node.addEventListener("dragover", event => { if (this.dragSource) { event.preventDefault(); node.classList.add("ce-slot-over"); } });
      node.addEventListener("dragleave", () => node.classList.remove("ce-slot-over"));
      node.addEventListener("drop", event => { event.preventDefault(); node.classList.remove("ce-slot-over"); this._drop(slot); });
      return node;
    }

    _renderGate(column, item) {
      const { gate, index, inputs, encoding } = item;
      const name = gateName(gate.gate), info = GATES[name];
      const qs = gate.qubits, min = Math.min(...qs), max = Math.max(...qs);
      const muted = this.activeInput !== null && !inputs.includes(this.activeInput);
      const mark = encoding && this.siteMarks ? this.siteMarks[index] : null;
      if (encoding || info.angle) {
        const note = el("div", `ce-gate-note${muted ? " ce-muted-gate" : ""}`); note.style.top = `${wireY(min) - 67}px`;
        if (encoding) {
          const badges = el("div", "ce-gate-badges");
          badges.append(el("span", "ce-encoding-badge", `E${encoding}`));
          if (mark) {
            const site = el("span", "ce-site-badge", mark.label);
            site.style.setProperty("--site-color", mark.color);
            badges.append(site);
          }
          inputs.forEach(input => badges.append(this._chip(input)));
          note.append(badges, el("div", "ce-angle-expression", `φ = ${pretty(gate.angle)}`));
        } else note.append(el("div", "ce-fixed-expression", pretty(gate.angle)));
        column.append(note);
      }
      const gateButton = button("", `ce-gate${encoding ? " ce-encoding-gate" : ""}${qs.length === 2 ? " ce-multigate" : ""}${muted ? " ce-muted-gate" : ""}`, () => this._select(index),
        `Gate ${index + 1}: ${info.label} on ${qs.map(q => `q${q}`).join(info.control ? " control, " : ", ")}${encoding ? `; encoding E${encoding}; ${gate.angle}; inputs ${inputs.map(i => `x${i}`).join(", ")}` : info.angle ? `; fixed angle ${gate.angle}` : ""}`);
      gateButton.setAttribute("aria-pressed", String(this.selected === index));
      gateButton.title = `${info.description}${info.angle ? ` (${gate.angle})` : ""} · ${qs.map(q => `q${q}`).join(", ")} · drag to move, Delete to remove`;
      gateButton.dataset.index = index;
      gateButton.style.top = `${wireY(min) - 21}px`;
      gateButton.style.height = `${(max - min) * PITCH + 42}px`;
      if (mark) { gateButton.classList.add("ce-site-gate"); gateButton.style.setProperty("--site-color", mark.color); }
      gateButton.draggable = true;
      gateButton.addEventListener("dragstart", event => this._dragStart(event, { kind: "move", index }));
      gateButton.addEventListener("dragend", () => this._dragEnd());
      if (qs.length === 2) gateButton.append(el("span", "ce-vertical-connector"));
      qs.forEach((q, position) => {
        let symbol = info.label, className = "ce-gate-box";
        if (name === "cx") { symbol = position === 0 ? "●" : "⊕"; className = position === 0 ? "ce-control-dot" : "ce-target-circle"; }
        if (name === "cz") { symbol = "●"; className = "ce-control-dot"; }
        if (name === "swap") { symbol = "×"; className = "ce-swap-symbol"; }
        const body = el("span", className, symbol); body.style.top = `${(q - min) * PITCH}px`; gateButton.append(body);
      });
      column.append(gateButton);
      if (this.selected === index) {
        const remove = button("×", "ce-gate-remove", () => this._delete(), `Delete gate ${index + 1}`);
        remove.title = "Delete this gate";
        remove.style.top = `${wireY(min) - 21}px`;
        column.append(remove);
      }
    }

    _renderPalette() {
      for (const item of this.palette.querySelectorAll(".ce-palette-gate")) item.setAttribute("aria-pressed", String(item.dataset.gate === this.armed));
      this.paletteHint.textContent = this.armed ? `Click a + on a wire to place ${GATES[this.armed].label}. It stays selected for more; press Esc when done.`
        : this.pendingSlot ? `Choose a gate for q${this.pendingSlot.qubit}.`
        : "Click a gate, then a + on a wire, or drag it there. Gates on different wires line up in one column.";
      this.root.classList.toggle("ce-armed", Boolean(this.armed || this.pendingSlot));
    }

    // Arm a palette gate, or place it at the slot chosen first.
    _arm(name) {
      if (this.disabled) return;
      if (this.pendingSlot) {
        const slot = this.pendingSlot; this.pendingSlot = null;
        this._place(name, slot);
        return;
      }
      this.armed = this.armed === name ? null : name;
      this._renderPalette();
    }

    _slotClicked(slot) {
      if (this.disabled) return;
      if (this.armed) { this._place(this.armed, slot); return; }
      const same = this.pendingSlot && this.pendingSlot.gap === slot.gap && this.pendingSlot.qubit === slot.qubit;
      this.pendingSlot = same ? null : slot;
      this._renderTrack(); this._renderPalette();
    }

    _place(name, slot) {
      if (!this._beforeNavigate()) return;
      try {
        const spec = copy(this.store.current), info = GATES[name];
        const gate = { gate: name, qubits: placeQubits(info.arity, slot.qubit, spec.n_qubits) };
        if (info.angle) gate.angle = "x0";
        const placed = insertAtGap(spec.gates, gate, slot.gap);
        spec.gates = placed.gates;
        this.pendingSlot = null;
        this._commit(spec, placed.index);
      } catch (error) { this._showError(error.message); this._renderTrack(); this._renderPalette(); }
    }

    _dragStart(event, source) {
      if (this.disabled) { event.preventDefault(); return; }
      this.dragSource = source;
      if (event.dataTransfer) { event.dataTransfer.effectAllowed = "copyMove"; event.dataTransfer.setData("text/plain", source.kind); }
      this.root.classList.add("ce-armed");
    }
    _dragEnd() {
      this.dragSource = null;
      this.root.classList.toggle("ce-armed", Boolean(this.armed || this.pendingSlot));
    }
    _drop(slot) {
      const source = this.dragSource;
      this._dragEnd();
      if (!source) return;
      if (source.kind === "new") { this._place(source.gate, slot); return; }
      if (!this._beforeNavigate()) return;
      try {
        const spec = copy(this.store.current);
        const moved = moveToGap(spec.gates, source.index, slot.gap, slot.qubit, spec.n_qubits);
        spec.gates = moved.gates;
        this._commit(spec, moved.index);
      } catch (error) { this._showError(error.message); }
    }

    _keydown(event) {
      if (event.key === "Escape" && (this.armed || this.pendingSlot)) {
        this.armed = null; this.pendingSlot = null;
        this._renderTrack(); this._renderPalette();
        return;
      }
      const gate = event.target && event.target.closest ? event.target.closest(".ce-gate") : null;
      if (gate && (event.key === "Delete" || event.key === "Backspace")) {
        event.preventDefault();
        if (this.selected !== Number(gate.dataset.index)) { if (!this._beforeNavigate()) return; this.selected = Number(gate.dataset.index); }
        this._delete();
      }
    }

    // The nearest gate before or after the selected one that shares a wire with it.
    _neighbour(direction) {
      const gates = this.store.current.gates, index = this.selected;
      if (index === null || !gates[index]) return null;
      for (let other = index + direction; other >= 0 && other < gates.length; other += direction) {
        if (spansOverlap(gates[index], gates[other])) return other;
      }
      return null;
    }

    _renderMapping() {
      if (!this.mapping) return;
      const encodings = this._encodingInfo().filter(item => item.encoding);
      this.mapping.replaceChildren(el("p", "eyebrow small", "ENCODINGS"));
      if (!encodings.length) {
        this.mapping.append(el("p", "ce-help", "No encodings"));
        return;
      }
      const table = el("table", "ce-mapping-table");
      const head = el("thead"), header = el("tr");
      ["Encoding", "Inputs", "Preprocessing φ(x)", "Gate"].forEach(label => header.append(el("th", "", label)));
      head.append(header); table.append(head);
      const body = el("tbody");
      for (const item of encodings) {
        const row = el("tr", this.selected === item.index ? "ce-mapping-selected" : "");
        if (this.activeInput !== null && !item.inputs.includes(this.activeInput)) row.classList.add("ce-mapping-muted");
        const encodingCell = el("td");
        encodingCell.append(button(`E${item.encoding}`, "ce-map-select", () => this._select(item.index), `Edit encoding E${item.encoding}`));
        const inputCell = el("td", "ce-mapping-inputs"); item.inputs.forEach(index => inputCell.append(this._chip(index)));
        const phiCell = el("td", "ce-mapping-expression", pretty(item.gate.angle)); phiCell.title = item.gate.angle;
        const gateCell = el("td", "ce-mapping-gate", `${GATES[gateName(item.gate.gate)].label} · ${item.gate.qubits.map(q => `q${q}`).join(", ")}`);
        row.append(encodingCell, inputCell, phiCell, gateCell); body.append(row);
      }
      table.append(body); this.mapping.append(table);
    }

    _select(index) {
      if (!this._beforeNavigate()) return;
      this.selected = index; this._renderTrack(); this._renderInspector(); this._syncButtons();
      // Keep keyboard focus on the gate so Delete removes it.
      const active = this.track.querySelector('.ce-gate[aria-pressed="true"]');
      if (active && active.focus) active.focus({ preventScroll: true });
    }

    _renderInspector(preserveDraft = false) {
      this.earlierButton = null; this.laterButton = null; this.dictionary = null;
      this.inputButtons = null; this.dependencyPreview = null;
      if (!preserveDraft) {
        if (this.selected === null) this.draft = null;
        else {
          const gate = copy(this.store.current.gates[this.selected]);
          this.draft = { ...gate, kind: "edit", index: this.selected,
            mode: dependencies(gate.angle).length ? "encoding" : "fixed" };
        }
      }
      this.inspector.replaceChildren();
      if (!this.draft) {
        this.inspector.append(el("h3", "", "Select a gate"));
        const steps = el("ul", "ce-howto");
        for (const text of ["Add: click a gate in the palette, then a + on a wire. Dragging it there works too.",
          "Move: drag a gate to another +, or use the move buttons here.",
          "Remove: select a gate and press Delete, or click its ×."]) steps.append(el("li", "", text));
        this.inspector.append(steps);
        return;
      }
      const total = this.store.current.gates.length;
      this.inspector.append(el("p", "eyebrow small", `GATE ${this.draft.index + 1} OF ${total} · COLUMN ${(this.columns ? this.columns[this.draft.index] : 0) + 1}`),
        el("h3", "", "Gate settings"));
      this.gateSelect = select(Object.entries(GATES).map(([value, info]) => [value, `${info.label} · ${info.description}`]), "Gate operation");
      this.gateSelect.value = gateName(this.draft.gate);
      this.inspector.append(field("Operation", this.gateSelect));
      this.qubitFields = el("div", "ce-qubit-fields"); this.inspector.append(this.qubitFields);
      this.angleFields = el("div", "ce-angle-fields"); this.inspector.append(this.angleFields);
      this.draftStatus = el("p", "ce-draft-status"); this.draftStatus.setAttribute("aria-live", "polite");
      this.inspector.append(this.draftStatus);
      const applyRow = el("div", "ce-apply-row");
      this.applyButton = button("Apply gate", "ce-apply", () => this._applyGate());
      this.discardButton = button("Discard", "ce-tool", () => {
        this.draftDirty = false; this._showError("");
        this._renderTrack(); this._renderInspector(); this._syncButtons();
      }, "Discard gate changes");
      applyRow.append(this.applyButton, this.discardButton); this.inspector.append(applyRow);
      if (this.draft.kind === "edit") {
        const actions = el("div", "ce-gate-actions");
        this.earlierButton = button("← Move left", "ce-tool", () => this._move(-1), "Move the selected gate before the previous gate on its wires");
        this.laterButton = button("Move right →", "ce-tool", () => this._move(1), "Move the selected gate after the next gate on its wires");
        actions.append(this.earlierButton, this.laterButton,
          button("Delete gate", "ce-delete", () => this._delete(), "Delete selected gate"));
        this.inspector.append(actions);
      }
      this.dictionary = el("div", "ce-basis-preview"); this.inspector.append(this.dictionary);
      this.gateSelect.addEventListener("change", () => {
        this._readDraftFields();
        this.draft.gate = this.gateSelect.value;
        const arity = GATES[this.draft.gate].arity;
        const first = this.draft.qubits[0] || 0;
        this.draft.qubits = arity === 1 ? [first] : [first, (first + 1) % this.store.current.n_qubits];
        if (GATES[this.draft.gate].angle && this.draft.angle === undefined) { this.draft.angle = "x0"; this.draft.mode = "encoding"; }
        this.draftDirty = true; this._renderDraftFields(); this._updateDraftPreview();
      });
      this._renderDraftFields(); this._updateDraftPreview(); this._renderDictionary(); this._syncButtons();
    }

    _renderDraftFields() {
      const info = GATES[gateName(this.draft.gate)];
      const choices = Array.from({ length: this.store.current.n_qubits }, (_, q) => [String(q), `q${q}`]);
      this.qubitSelect = select(choices, info.control ? "Control qubit" : "Target qubit");
      this.qubitSelect.value = String(this.draft.qubits[0]);
      this.qubitFields.replaceChildren(field(info.control ? "Control qubit" : info.arity === 2 ? "First qubit" : "Target qubit", this.qubitSelect));
      if (info.arity === 2) {
        this.secondSelect = select(choices, info.control ? "Target qubit" : "Second qubit");
        this.secondSelect.value = String(this.draft.qubits[1]);
        this.qubitFields.append(field(info.control ? "Target qubit" : "Second qubit", this.secondSelect));
      } else this.secondSelect = null;
      this.qubitSelect.addEventListener("change", () => this._draftInput());
      if (this.secondSelect) this.secondSelect.addEventListener("change", () => this._draftInput());
      this.angleFields.replaceChildren();
      this.modeSelect = null; this.angleInput = null; this.inputButtons = null; this.dependencyPreview = null;
      if (info.angle) {
        this.modeSelect = select([["encoding", "Input encoding"], ["fixed", "Fixed angle"]], "Angle source");
        this.modeSelect.value = this.draft.mode;
        this.angleFields.append(field("Angle source", this.modeSelect));
        this.angleInput = el("textarea", "ce-expression-input"); this.angleInput.rows = 2; this.angleInput.spellcheck = false;
        this.angleInput.placeholder = "2*x0**2+x1";
        this.angleInput.value = String(this.draft.angle ?? (this.draft.mode === "fixed" ? "0.0" : "x0"));
        this.angleFields.append(field("Preprocessing / angle expression", this.angleInput));
        this.inputButtons = el("div", "ce-insert-inputs"); this.angleFields.append(this.inputButtons);
        this.dependencyPreview = el("div", "ce-dependency-preview"); this.angleFields.append(this.dependencyPreview);
        this._renderInputButtons();
        this.modeSelect.addEventListener("change", () => this._draftInput());
        this.angleInput.addEventListener("input", () => this._draftInput());
      }
    }
    _renderInputButtons() {
      if (!this.inputButtons) return;
      this.inputButtons.replaceChildren();
      for (let index = 0; index < this.domain.length; index++) {
        const insert = button(`+ x${index}`, "ce-input-insert", () => {
          const input = this.angleInput;
          input.setRangeText(`x${index}`, input.selectionStart, input.selectionEnd, "end");
          input.focus(); this._draftInput();
        }, `Insert x${index} into preprocessing`);
        insert.style.setProperty("--input-color", COLORS[index % COLORS.length]);
        this.inputButtons.append(insert);
      }
    }
    _readDraftFields() {
      if (!this.draft) return;
      const info = GATES[gateName(this.draft.gate)];
      if (this.qubitSelect) this.draft.qubits = [Number(this.qubitSelect.value), ...(info.arity === 2 ? [Number(this.secondSelect.value)] : [])];
      if (this.modeSelect) this.draft.mode = this.modeSelect.value;
      if (this.angleInput) this.draft.angle = this.angleInput.value;
    }
    _draftInput() { this._readDraftFields(); this.draftDirty = true; this._updateDraftPreview(); this._renderDictionary(); }
    _updateDraftPreview() {
      if (!this.draft || !this.draftStatus) return;
      if (this.inputButtons && this.inputButtons.children.length !== this.domain.length) this._renderInputButtons();
      this.draftStatus.className = "ce-draft-status";
      try {
        validateDraft(this.draft, this.store.current.n_qubits, this.domain.length);
        this.draftStatus.textContent = this.draftDirty ? "Unapplied changes" : "";
        this.draftStatus.classList.toggle("ce-unapplied", this.draftDirty);
      } catch (error) {
        this.draftStatus.textContent = error.message; this.draftStatus.classList.add("ce-invalid");
      }
      if (this.dependencyPreview && GATES[gateName(this.draft.gate)].angle) {
        this.dependencyPreview.replaceChildren(el("span", "ce-phi-preview", `φ = ${pretty(this.draft.angle || "…")}`));
        const used = dependencies(this.draft.angle);
        const mapping = el("div", "ce-draft-mapping");
        if (!used.length) mapping.append(el("span", "ce-help", "Constant"));
        else {
          used.forEach((index, position) => { if (position) mapping.append(el("span", "", "+")); mapping.append(this._chip(index)); });
          mapping.append(el("span", "", "→ φ →"), el("strong", "", GATES[gateName(this.draft.gate)].label));
        }
        this.dependencyPreview.append(mapping);
      }
    }
    _applyGate() {
      if (this.disabled) return;
      try {
        this._readDraftFields();
        const checked = validateDraft(this.draft, this.store.current.n_qubits, this.domain.length);
        const spec = copy(this.store.current), index = this.draft.index;
        spec.gates[index] = { ...spec.gates[index], ...checked };
        if (!GATES[checked.gate].angle) delete spec.gates[index].angle;
        this._commit(spec, index);
      } catch (error) { this._showError(error.message); }
    }
    _move(direction) {
      if (!this._beforeNavigate()) return;
      // Swapping with a gate on another wire would change nothing, so step past the nearest one sharing a wire.
      const index = this.selected, next = this._neighbour(direction);
      if (next === null) return;
      const spec = copy(this.store.current);
      const [gate] = spec.gates.splice(index, 1);
      spec.gates.splice(next, 0, gate);
      this._commit(spec, next);
    }
    _delete() {
      if (!this._beforeNavigate()) return;
      const spec = copy(this.store.current); spec.gates.splice(this.selected, 1);
      this._commit(spec, spec.gates.length ? Math.min(this.selected, spec.gates.length - 1) : null);
    }

    _renderSettings() {
      const spec = this.store.current;
      this.qubitCount = select(Array.from({ length: 10 }, (_, index) => [String(index + 1), String(index + 1)]), "Number of circuit qubits");
      this.qubitCount.value = String(spec.n_qubits);
      this.observableInput = el("input"); this.observableInput.type = "text"; this.observableInput.value = spec.observable;
      this.observableInput.setAttribute("aria-label", "Measured Pauli observable"); this.observableInput.maxLength = 10; this.observableInput.spellcheck = false;
      this.observableInput.title = "Leftmost letter acts on the highest-index qubit.";
      this.initialInput = el("input"); this.initialInput.type = "text";
      const customState = Array.isArray(spec.initial_state);
      this.initialInput.value = customState ? "Custom statevector" : spec.initial_state || "0".repeat(spec.n_qubits);
      this.initialInput.readOnly = customState;
      this.initialInput.setAttribute("aria-label", "Initial state bitstring");
      this.initialInput.title = "Leftmost bit is the highest-index qubit.";
      this.settingsBody.replaceChildren(field("Qubit count", this.qubitCount),
        field("Pauli string (high → low)", this.observableInput),
        field("Initial state (high → low)", this.initialInput));
      const actions = el("div", "ce-circuit-actions");
      actions.append(button("Apply circuit settings", "ce-tool ce-tool-strong", () => {
        if (this.draftDirty) { this._showError("Apply or discard the gate changes first."); return; }
        try {
          const next = copy(this.store.current); next.n_qubits = Number(this.qubitCount.value);
          next.observable = this.observableInput.value.trim().toUpperCase();
          if (!Array.isArray(next.initial_state)) next.initial_state = this.initialInput.value.trim();
          else if (next.n_qubits !== this.store.current.n_qubits) throw new Error("Reset the custom initial state before changing the qubit count.");
          this._commit(next, this.selected);
        } catch (error) { this._showError(error.message); }
      }), button("Discard settings", "ce-tool", () => { this.settingsDirty = false; this._showError(""); this._renderSettings(); }));
      if (customState) actions.append(button("Reset initial state", "ce-tool", () => {
        if (!this._beforeNavigate()) return;
        const next = copy(this.store.current); next.initial_state = "0".repeat(next.n_qubits); this._commit(next);
      }));
      this.settingsStatus = el("p", "ce-settings-status"); this.settingsStatus.setAttribute("aria-live", "polite");
      this.settingsBody.append(actions, this.settingsStatus);
      for (const input of [this.qubitCount, this.observableInput, this.initialInput]) input.addEventListener("input", () => {
        this.settingsDirty = true; this.settingsStatus.textContent = "Unapplied settings";
      });
      this.qubitCount.addEventListener("change", () => {
        // Keep the editable strings coherent with a requested wire count.
        // Gates on removed wires are never silently deleted.
        const n = Number(this.qubitCount.value), observable = this.observableInput.value;
        this.observableInput.value = observable.length < n ? observable.padStart(n, "I") : observable.slice(observable.length - n);
        if (!customState) {
          const bits = this.initialInput.value;
          this.initialInput.value = bits.length < n ? bits.padStart(n, "0") : bits.slice(bits.length - n);
        }
        this.settingsDirty = true;
        this.settingsStatus.textContent = "Unapplied settings · remove gates from deleted qubits first.";
      });
    }

    _renderDictionary() {
      if (!this.dictionary) return;
      this.dictionary.replaceChildren();
      if (!this.draft || this.draft.kind !== "edit" || this.draftDirty) return;
      const data = this.inspection || {};
      const encodings = data.encodings || (data.metadata && data.metadata.encodings) || [];
      const detail = encodings.find(item => item.gate_position === this.selected);
      if (!detail) return;
      if (detail.generator) this.dictionary.append(el("p", "ce-generator", `G = ${detail.generator}${detail.frequencies ? ` · ω = ${detail.frequencies.join(", ")}` : ""}`));
      const basis = detail.basis || detail.dictionary || (data.basis && data.basis[detail.encoding_index]);
      const expressions = basis && (basis.expressions || basis);
      if (Array.isArray(expressions)) {
        const disclosure = el("details", "ce-dictionary-details");
        const functions = el("div", "ce-generated-basis");
        expressions.forEach((expression, index) => {
          const row = el("div", "ce-dictionary-function");
          const formula = el("span", "basis-formula");
          if (global.FTCIFormula) global.FTCIFormula.render(formula, expression);
          else formula.textContent = pretty(expression);
          row.append(el("span", "ce-dictionary-index", String(index)), formula);
          functions.append(row);
        });
        disclosure.append(el("summary", "", `Dictionary · ${expressions.length} functions`), functions);
        this.dictionary.append(disclosure);
      }
      else if (detail.frequencies) this.dictionary.append(el("p", "ce-help", `ω = ${detail.frequencies.join(", ")} · ${detail.dimension} functions`));
    }
    _renderValidation() {
      if (this.inspectionError) {
        this.validation.className = "ce-validation ce-validation-error";
        this.validation.textContent = this.inspectionError;
      } else if (this.inspection) {
        const count = this._encodingInfo().filter(item => item.encoding).length;
        this.validation.className = "ce-validation ce-validation-ready";
        this.validation.textContent = `✓ ${count} encoding${count === 1 ? "" : "s"}`;
      } else {
        this.validation.className = "ce-validation";
        this.validation.textContent = "";
      }
    }
  }

  global.FTCICircuitEditor = FTCICircuitEditor;
  global.FTCICircuitHelpers = helpers;
})(typeof window !== "undefined" ? window : globalThis);

"use strict";

const $ = (id) => document.getElementById(id);
const state = { mode: "function", examples: [], jobId: null, busy: false, result: null, request: null, core: null, selectedCore: 0, polling: null, engineAvailable: false, circuitEditor: null, circuitSource: null, circuitSnapshot: null, suppressCircuitSync: false, circuitValid: null, circuitError: "", modeDrafts: {}, representation: "canonical", inspection: null, runs: [], runCounter: 0, designTimer: null, designToken: 0, customSites: null, customData: null, customSignature: "", customToken: 0, view: "dictionary", viewData: null, sweep: null, sweepCounter: 0, sliceTimer: null, sliceToken: 0 };
const SWEEP_AXES = {
  max_bond: { label: "Maximum bond", short: "bond", defaults: "2, 4, 8, 16", min: 1, max: 64 },
  oversampling: { label: "Oversampling", short: "oversampling", defaults: "1, 2, 3", min: 1, max: 10 },
  shots: { label: "Shots per query", short: "shots", defaults: "100, 10000, 1000000", min: 1, max: 1e12 },
  representation: { label: "Representation", short: "", defaults: "" },
};
const VIEW_LABELS = { dictionary: "Dictionary (as learned)", orthonormal: "Orthonormal", nodal: "Sampling grid (nodal)" };
const VIEW_HELP = "The same function can be read in other bases of each dictionary's function space. Bond dimensions and predictions do not change.";
const REPRESENTATION_LABELS = { canonical: "Canonical", merged: "Merged by preprocessing", custom: "Custom" };
const SITE_COLORS = ["#a8402b", "#2f6f8f", "#557864", "#b07a1e", "#6b4e9b", "#2a8a7a", "#a34a7c", "#5f6b2e"];

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: { ...(options.body ? { "Content-Type": "application/json" } : {}), ...options.headers },
    cache: "no-store",
  });
  let data;
  try { data = await response.json(); } catch { throw new Error(`The engine returned an unreadable response (${response.status}).`); }
  if (!response.ok) {
    const detail = data.error || data.detail || data.message || `Request failed (${response.status}).`;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return data;
}

function showError(id, message = "") {
  $(id).textContent = message;
  $(id).hidden = !message;
}

function number(value, digits = 5) {
  if (value === null || value === undefined || !Number.isFinite(Number(value))) return "—";
  const n = Number(value);
  if (n === 0) return "0";
  if (Math.abs(n) < 0.001 || Math.abs(n) >= 100000) return n.toExponential(3);
  return Number(n.toPrecision(digits)).toString();
}

function integer(value, fallback) {
  const n = Number(value);
  if (!Number.isInteger(n)) throw new Error(`${fallback} must be a whole number.`);
  return n;
}

function numeric(value, fallback) {
  const n = Number(value);
  if (String(value).trim() === "" || !Number.isFinite(n)) throw new Error(`${fallback} must be a finite number.`);
  return n;
}

function setStatus(status, title, message) {
  $("run-status").className = `run-status ${status}`;
  $("run-title").textContent = title;
  $("run-message").textContent = message || "";
  $("run-message").hidden = !message;
  $("run-icon").textContent = { running: "·", queued: "·", complete: "✓", failed: "!", cancelled: "−" }[status] || "○";
}

function setBusy(busy) {
  state.busy = busy;
  refreshLearnButton();
  $("learn-button").firstElementChild.textContent = busy ? "Learning…" : "Learn function";
  $("cancel-button").hidden = !busy;
  $("cancel-button").disabled = false;
  // Keep the requested problem visible while the active model is being learned.
  for (const input of $("learning-form").querySelectorAll("input, textarea, select, button:not(#learn-button)")) input.disabled = busy;
  if (!busy && $("domain-inputs").children.length === 1) $("domain-inputs").querySelector(".remove-button").disabled = true;
  state.circuitSource?.setDisabled(busy);
  if (!busy) renderRepresentation();
  renderRuns();
  if (!busy) scheduleDesign();
}

function refreshLearnButton() {
  $("learn-button").disabled = state.busy || !state.engineAvailable || (state.mode === "quantum" && state.circuitValid !== true);
}

function setMode(mode, chooseExample = true) {
  state.mode = mode;
  for (const button of document.querySelectorAll(".mode-button[data-mode]")) {
    const selected = button.dataset.mode === mode;
    button.classList.toggle("active", selected);
    button.setAttribute("aria-pressed", String(selected));
  }
  $("function-settings").hidden = mode !== "function";
  $("basis-settings").hidden = mode !== "function";
  $("quantum-representation").hidden = mode !== "quantum";
  $("shots-field").hidden = mode !== "quantum";
  for (const option of $("sweep-axis").querySelectorAll("option[data-quantum]")) option.hidden = option.disabled = mode !== "quantum";
  if (mode !== "quantum" && ["shots", "representation"].includes($("sweep-axis").value)) $("sweep-axis").value = "";
  renderSweepControls();
  $("quantum-settings").hidden = mode !== "quantum";
  $("quantum-editor").hidden = mode !== "quantum";
  document.body.classList.toggle("quantum-mode", mode === "quantum");
  renderExampleSelect();
  if (chooseExample) {
    const example = state.examples.find((item) => item.mode === mode);
    if (example) loadExample(example);
  }
  refreshLearnButton();
}

function renderExampleSelect() {
  const select = $("example-select");
  const previous = select.value;
  select.replaceChildren();
  const examples = state.examples.filter((item) => item.mode === state.mode);
  for (const [i, example] of examples.entries()) {
    const option = element("option", null, example.name);
    option.value = example.id || String(i);
    select.append(option);
  }
  if (examples.some((item, i) => (item.id || String(i)) === previous)) select.value = previous;
  if (!examples.length) {
    const option = element("option", null, "Custom problem");
    option.value = "custom";
    select.append(option);
  }
}

function loadExample(example) {
  if (state.busy) return;
  state.mode = example.mode || "function";
  setMode(state.mode, false);
  const filtered = state.examples.filter((item) => item.mode === state.mode);
  $("example-select").value = example.id || String(filtered.indexOf(example));
  $("model-name").value = example.name || "My function";
  $("example-description").textContent = example.description || "";
  $("target-expression").value = example.target || "";
  state.suppressCircuitSync = true;
  try { renderDomain(example.domain || [[-Math.PI, Math.PI]]); }
  finally { state.suppressCircuitSync = false; }
  renderBasisEditors(example.basis || []);
  if (example.circuit) {
    state.representation = example.representation === "merged" ? "merged" : "canonical";
    state.customSites = null; state.customData = null;
    const circuit = { ...example.circuit, domain: readDomain() };
    $("circuit-file-name").hidden = true;
    state.circuitSource.setCircuit(circuit, {format: $("circuit-format").value || "json"});
  }
  const ids = { max_bond: "max-bond", tolerance: "tolerance", oversampling: "oversampling", query_budget: "query-budget", validation_samples: "validation-samples", seed: "seed", node_rule: "node-rule", shots: "shots" };
  $("node-rule").value = "auto";
  $("shots").value = "";
  for (const [key, id] of Object.entries(ids)) {
    if (example.options && example.options[key] !== undefined) $(id).value = example.options[key];
  }
  showError("form-error");
  scheduleDesign();
}

function rememberMode() {
  state.modeDrafts[state.mode] = {
    name: $("model-name").value, target: $("target-expression").value,
    example: $("example-select").value,
    domain: [...$("domain-inputs").children].map(row => [row.querySelector('[data-bound="min"]').value, row.querySelector('[data-bound="max"]').value]),
    basis: [...$("basis-editors").children].map(editor => ({name:editor.querySelector('[data-field="name"]').value, expressions:editor.querySelector('[data-field="expressions"]').value.split("\n")})),
    options: Object.fromEntries(["max-bond", "query-budget", "tolerance", "oversampling", "validation-samples", "seed", "node-rule", "shots"].map(id => [id, $(id).value])),
  };
}

function switchProblemMode(mode) {
  if (state.busy || state.mode === mode) return;
  rememberMode();
  const saved = state.modeDrafts[mode];
  if (!saved) { setMode(mode); return; }
  setMode(mode, false);
  $("model-name").value = saved.name;
  $("target-expression").value = saved.target;
  $("example-select").value = saved.example;
  state.suppressCircuitSync = true;
  try {
    // A pending source edit may finish while the general-function mode is open.
    const domain = mode === "quantum" && state.circuitValid === true
      ? state.circuitSnapshot?.circuit?.domain || saved.domain : saved.domain;
    renderDomain(domain);
  }
  finally { state.suppressCircuitSync = false; }
  renderBasisEditors(saved.basis);
  Object.entries(saved.options).forEach(([id, value]) => { $(id).value = value; });
  showError("form-error");
  scheduleDesign();
}

function renderDomain(domain) {
  const root = $("domain-inputs");
  root.replaceChildren();
  domain.forEach((range, index) => {
    const row = element("div", "domain-row");
    row.append(element("span", null, `x${index}`));
    const min = element("input");
    min.type = "number"; min.step = "any"; min.value = range[0]; min.required = true;
    min.setAttribute("aria-label", `Lower bound for x${index}`);
    min.dataset.bound = "min";
    const max = element("input");
    max.type = "number"; max.step = "any"; max.value = range[1]; max.required = true;
    max.setAttribute("aria-label", `Upper bound for x${index}`);
    max.dataset.bound = "max";
    const remove = element("button", "remove-button", "×");
    remove.type = "button"; remove.title = `Remove input x${index}`;
    remove.setAttribute("aria-label", `Remove input x${index}`);
    remove.disabled = domain.length === 1;
    remove.addEventListener("click", () => {
      try {
        const current = readDomain();
        if (state.mode === "quantum") {
          if (index !== current.length - 1) throw new Error("Remove quantum inputs from the end to preserve the gate-to-input mapping.");
          const used = state.circuitSource.getCircuit().gates.some(gate => (String(gate.angle ?? "").match(/\bx\d+\b/g) || []).includes(`x${index}`));
          if (used) throw new Error(`x${index} is connected to an encoding gate. Change that gate's preprocessing before removing this input.`);
        }
        current.splice(index, 1);
        renderDomain(current);
      } catch (error) { showError("form-error", error.message); }
    });
    row.append(min, element("span", null, "to"), max, remove);
    root.append(row);
  });
  if (state.mode === "quantum" && state.circuitEditor && !state.suppressCircuitSync) syncCircuitDomain();
}

function readDomain() {
  return [...$("domain-inputs").children].map((row, index) => {
    const lo = numeric(row.querySelector('[data-bound="min"]').value, `x${index} lower bound`);
    const hi = numeric(row.querySelector('[data-bound="max"]').value, `x${index} upper bound`);
    if (lo >= hi) throw new Error(`The lower bound of x${index} must be smaller than its upper bound.`);
    return [lo, hi];
  });
}

function renderBasisEditors(basis) {
  $("basis-editors").replaceChildren();
  basis.forEach((item) => addBasisEditor(item));
}

function addBasisEditor(spec = {}) {
  const editor = element("div", "basis-editor");
  const head = element("div", "basis-editor-head");
  head.append(element("span", "site-number", `Φ${$("basis-editors").children.length + 1}`));
  const name = element("input");
  name.type = "text"; name.value = spec.name || "Local dictionary";
  name.maxLength = 100; name.setAttribute("aria-label", "Dictionary name");
  name.dataset.field = "name";
  const remove = element("button", "remove-button", "×");
  remove.type = "button"; remove.setAttribute("aria-label", "Remove this dictionary");
  remove.addEventListener("click", () => {
    editor.remove();
    [...$("basis-editors").children].forEach((node, index) => { node.querySelector(".site-number").textContent = `Φ${index + 1}`; });
  });
  head.append(name, remove);
  const expressions = element("textarea", "expression-input");
  expressions.rows = 4; expressions.spellcheck = false; expressions.dataset.field = "expressions";
  expressions.setAttribute("aria-label", "Basis functions, one expression per line");
  expressions.value = (spec.expressions || ["1", "cos(x0)", "sin(x0)"]).join("\n");
  const dependencies = element("span", "dependency-label");
  function updateDependencies() {
    const variables = [...new Set(expressions.value.match(/\bx\d+\b/g) || [])].sort((a, b) => Number(a.slice(1)) - Number(b.slice(1)));
    const dimension = expressions.value.split("\n").filter((line) => line.trim()).length;
    dependencies.textContent = `${dimension} functions · ${variables.length ? `depends on ${variables.join(", ")}` : "constant dictionary"}`;
  }
  expressions.addEventListener("input", updateDependencies);
  updateDependencies();
  editor.append(head, expressions, dependencies);
  $("basis-editors").append(editor);
}

function readBasis() {
  const basis = [...$("basis-editors").children].map((editor) => ({
    name: editor.querySelector('[data-field="name"]').value.trim() || "Local dictionary",
    expressions: editor.querySelector('[data-field="expressions"]').value.split("\n").map((line) => line.trim()).filter(Boolean),
  }));
  if (!basis.length) throw new Error("Add at least one basis dictionary.");
  if (basis.some((item) => !item.expressions.length)) throw new Error("Every dictionary needs at least one basis function.");
  return basis;
}

function markCircuitChanged() {
  if (state.result && state.request?.mode === "quantum") {
    setStatus("idle", "Circuit updated", "Previous model · learn again to update");
  }
}

function circuitSourceChanged(snapshot) {
  state.circuitSnapshot = snapshot;
  state.circuitValid = snapshot.valid && !snapshot.pending;
  state.circuitError = snapshot.error || "";
  if (state.circuitValid) {
    state.inspection = snapshot.inspection || null;
    if (state.customSites && state.customSignature !== encodingSignature()) {
      // The arrangement refers to encoding gates that no longer exist in this form.
      state.customSites = null; state.customData = null;
      if (state.representation === "custom") state.representation = "canonical";
    } else if (state.customSites) updateCustomRepresentation(state.customSites);
  }
  renderRepresentation();
  if (snapshot.valid && !snapshot.pending && snapshot.circuit && state.mode === "quantum") {
    state.suppressCircuitSync = true;
    try { renderDomain(snapshot.circuit.domain); }
    finally { state.suppressCircuitSync = false; }
  }
  markCircuitChanged();
  refreshLearnButton();
}

function representationData(name = state.representation) {
  if (name === "custom") return state.customData;
  return state.inspection?.representations?.[name] || null;
}

function encodingSignature() {
  return JSON.stringify((state.inspection?.encodings || []).map(item => [item.gate_position, item.gate, item.qubits, item.preprocessing]));
}

// A custom arrangement is an ordered partition of the encoding gates; the server derives its dictionaries.
async function updateCustomRepresentation(sites) {
  const token = ++state.customToken;
  try {
    const data = await api("/api/circuit/representation", { method: "POST", body: JSON.stringify({ circuit: state.circuitSource.getCircuit(), sites }) });
    if (token !== state.customToken) return;
    if (!data.available) throw new Error(data.reason);
    state.customSites = sites;
    state.customData = data;
    state.customSignature = encodingSignature();
    showError("form-error");
    markCircuitChanged();
  } catch (error) {
    if (token !== state.customToken) return;
    showError("form-error", error.message);
  }
  renderRepresentation();
}

function arrangeSites(action, index) {
  if (state.busy || !state.customSites) return;
  const sites = state.customSites.map(members => [...members]);
  if (action === "up" && index > 0) [sites[index - 1], sites[index]] = [sites[index], sites[index - 1]];
  else if (action === "down" && index < sites.length - 1) [sites[index + 1], sites[index]] = [sites[index], sites[index + 1]];
  else if (action === "join" && index < sites.length - 1) sites.splice(index, 2, [...sites[index], ...sites[index + 1]]);
  else if (action === "split" && sites[index].length > 1) sites.splice(index, 1, ...sites[index].map(member => [member]));
  else return;
  updateCustomRepresentation(sites);
}

// Gates sharing a preprocessing function keep one colour in every representation.
function siteColor(site) {
  const merged = representationData("merged");
  const first = site.encoding_indices[0];
  const shared = merged?.available ? merged.sites.findIndex(item => item.encoding_indices.includes(first)) : -1;
  return SITE_COLORS[(shared >= 0 ? shared : first) % SITE_COLORS.length];
}

// Entries of the two-site blocks one TCI sweep evaluates when every bond is full.
function sweepEstimate(dimensions, maxBond, oversampling) {
  const length = dimensions.length;
  if (!length) return 0;
  const points = dimensions.map(q => Math.max(q, Math.ceil(oversampling * q)));
  const grid = points.reduce((product, m) => Math.min(product * m, 1e18), 1);
  if (length === 1) return points[0];
  const bonds = [1];
  for (let k = 1; k < length; k++) {
    const left = dimensions.slice(0, k).reduce((product, q) => Math.min(product * q, 1e18), 1);
    const right = dimensions.slice(k).reduce((product, q) => Math.min(product * q, 1e18), 1);
    bonds.push(Math.min(maxBond, left, right));
  }
  bonds.push(1);
  let total = 0;
  for (let k = 0; k < length - 1; k++) total += bonds[k] * points[k] * points[k + 1] * bonds[k + 2];
  return Math.min(total, grid);
}

function basisFunctionCount(dimensions, log10) {
  if (log10 < 15) return dimensions.reduce((product, q) => product * q, 1).toLocaleString();
  const exponent = Math.floor(log10);
  const superscript = String(exponent).replace(/\d/g, digit => "⁰¹²³⁴⁵⁶⁷⁸⁹"[digit]);
  return `≈ ${(10 ** (log10 - exponent)).toFixed(1)} × 10${superscript}`;
}

function compactList(values, limit = 8) {
  if (values.length <= limit) return values.join(", ");
  return `${values.slice(0, limit - 2).join(", ")}, …, ${values[values.length - 1]} (${values.length})`;
}

function renderRepresentation() {
  if (!$("quantum-representation")) return;
  const canonical = representationData("canonical");
  const merged = representationData("merged");
  if (state.representation === "merged" && merged && !merged.available) state.representation = "canonical";
  for (const button of document.querySelectorAll(".representation-button")) {
    const name = button.dataset.representation;
    const data = representationData(name);
    const selected = name === state.representation;
    button.classList.toggle("active", selected);
    button.setAttribute("aria-pressed", String(selected));
    const ready = name === "custom" ? Boolean(canonical?.available && canonical.length) : Boolean(data?.available);
    button.disabled = state.busy || !ready;
    button.title = data && !data.available ? data.reason : "";
    const short = { canonical: "Canonical", merged: "Merged", custom: "Custom" }[name];
    button.textContent = data?.available ? `${short} · ${data.length}` : short;
    button.setAttribute("aria-label", data?.available ? `${REPRESENTATION_LABELS[name]}, ${data.length} ${data.length === 1 ? "dictionary" : "dictionaries"}` : REPRESENTATION_LABELS[name]);
    if (!button.title) button.title = REPRESENTATION_LABELS[name];
  }
  const current = representationData();
  const note = $("representation-note");
  const sites = $("representation-sites");
  if (!current?.available) {
    note.textContent = "Dictionaries appear once the circuit is valid.";
    $("representation-summary").replaceChildren();
    sites.replaceChildren();
    sites.hidden = true;
    state.circuitEditor?.setSiteMarks({});
    return;
  }
  const custom = state.representation === "custom";
  let text = state.representation === "merged"
    ? "Gates that share a preprocessing function are merged into one dictionary with one training angle. Its frequencies are sums of the merged gates' frequencies."
    : custom ? "Reorder the dictionaries along the chain, join neighbours or split them. Gates on one dictionary that share a preprocessing function are merged; gates with different preprocessing functions form a tensor-product dictionary."
    : "One dictionary per encoding gate. Training varies every gate angle independently.";
  if (!current.length) text = "This circuit has no input-dependent gates, so the function is constant.";
  else if (merged && !merged.available) text += ` Merging is unavailable: ${merged.reason}`;
  else if (!custom && merged && canonical && merged.length === canonical.length) text += " No gates share a preprocessing function here, so both choices give the same dictionaries.";
  note.textContent = text;
  renderRepresentationSummary();
  sites.replaceChildren();
  sites.hidden = !current.length;
  const marks = {};
  if (current.length) {
    const table = element("table", "site-table");
    const head = element("thead"), heading = element("tr");
    for (const label of ["Dictionary", "Preprocessing φ(x)", "Encoding gates", "Frequencies", "Dimension", ...(custom ? ["Arrange"] : [])]) {
      const cell = element("th", null, label); cell.scope = "col"; heading.append(cell);
    }
    head.append(heading);
    const body = element("tbody");
    current.sites.forEach((site, index) => {
      const color = siteColor(site);
      const row = element("tr");
      const label = element("th"); label.scope = "row";
      const name = element("span", "site-label");
      const swatch = element("span", "site-swatch"); swatch.style.setProperty("--site-color", color);
      name.append(swatch, element("span", null, `Φ${index + 1}`));
      label.append(name);
      const groups = site.classes?.length ? site.classes : [site];
      const preprocessing = element("td", "site-factors");
      groups.forEach((group, position) => {
        if (position) preprocessing.append(element("span", "site-product", "⊗"));
        const formula = element("span", "basis-formula");
        renderBasisFormula(formula, group.preprocessing ?? "");
        preprocessing.append(formula);
      });
      const gates = element("td", "numeric", compactList(site.encoding_indices.map(i => `E${i + 1}`)));
      const frequencies = element("td", "numeric", groups.map(group => group.frequencies.length ? compactList(group.frequencies.map(value => number(value, 6))) : "0").join("  ⊗  "));
      row.append(label, preprocessing, gates, frequencies, element("td", "numeric", String(site.dimension)));
      if (custom) {
        const actions = element("td", "site-actions");
        const add = (text, action, enabled, description) => {
          const control = element("button", "text-button", text);
          control.type = "button"; control.disabled = state.busy || !enabled;
          control.setAttribute("aria-label", `${description} Φ${index + 1}`);
          control.title = description;
          control.addEventListener("click", () => arrangeSites(action, index));
          actions.append(control);
        };
        add("↑", "up", index > 0, "Move earlier:");
        add("↓", "down", index < current.sites.length - 1, "Move later:");
        add("Join next", "join", index < current.sites.length - 1, "Join with the next dictionary:");
        add("Split", "split", site.encoding_indices.length > 1, "Split into one dictionary per gate:");
        row.append(actions);
      }
      body.append(row);
      for (const position of site.gate_positions || []) marks[position] = { label: `Φ${index + 1}`, color };
    });
    table.append(head, body);
    sites.append(table);
  }
  state.circuitEditor?.setSiteMarks(marks);
  scheduleDesign();
}

function renderRepresentationSummary() {
  const root = $("representation-summary");
  if (!root) return;
  const current = representationData();
  root.replaceChildren();
  if (!current?.available || !current.length) return;
  const maxBond = Math.max(1, Number($("max-bond").value) || 1);
  const oversampling = Math.max(1, Number($("oversampling").value) || 1);
  const encodings = state.inspection?.encoding_occurrences ?? current.length;
  const items = [
    ["Dictionaries", String(current.length), `for ${encodings} encoding ${encodings === 1 ? "gate" : "gates"}`],
    ["Local dimension", String(current.max_dimension), "largest dictionary"],
    ["Basis functions", basisFunctionCount(current.dimensions, current.log10_basis_functions), "in the tensor product"],
    ["Queries per sweep", `≲ ${Math.round(sweepEstimate(current.dimensions, maxBond, oversampling)).toLocaleString()}`, `estimate at bond ${maxBond}`],
  ];
  for (const [label, value, detail] of items) {
    const item = element("div");
    const data = element("dd", null, value);
    data.append(element("small", null, detail));
    item.append(element("dt", null, label), data);
    root.append(item);
  }
}

function chooseRepresentation(name) {
  if (state.busy || state.representation === name) return;
  if (name === "custom") {
    // Start from the arrangement currently shown.
    const start = state.customSites || representationData()?.sites?.map(site => [...site.encoding_indices]);
    if (!start?.length) return;
    state.representation = "custom";
    updateCustomRepresentation(start);
    return;
  }
  if (!representationData(name)?.available) return;
  state.representation = name;
  renderRepresentation();
  markCircuitChanged();
}

async function syncCircuitDomain() {
  if (!state.circuitSource || state.suppressCircuitSync || state.mode !== "quantum") return;
  try {
    if (!await state.circuitSource.setDomain(readDomain())) throw new Error("Input bounds could not be synchronized. Check the circuit source before learning.");
    showError("form-error");
  } catch (error) { showError("form-error", error.message); }
}

function readRequest() {
  const domain = readDomain();
  const options = {
    max_bond: integer($("max-bond").value, "Maximum bond"),
    query_budget: integer($("query-budget").value, "Query budget"),
    tolerance: numeric($("tolerance").value, "TCI tolerance"),
    oversampling: integer($("oversampling").value, "Oversampling"),
    validation_samples: integer($("validation-samples").value, "Validation points"),
    seed: integer($("seed").value, "Random seed"),
    node_rule: $("node-rule").value,
  };
  if (state.mode === "quantum" && $("shots").value.trim() !== "") {
    options.shots = integer($("shots").value, "Shots per query");
    if (options.shots < 1) throw new Error("Shots per query must be at least 1.");
  }
  const request = { mode: state.mode, name: $("model-name").value.trim() || "My function", domain, options };
  if (state.mode === "function") {
    request.target = $("target-expression").value.trim();
    if (!request.target) throw new Error("Enter a target function.");
    request.basis = readBasis();
  } else {
    if (state.circuitValid !== true) throw new Error(state.circuitError || "The circuit is being checked. Wait a moment before learning.");
    request.circuit = state.circuitSource.getCircuit();
    if (JSON.stringify(domain) !== JSON.stringify(request.circuit.domain)) throw new Error("Input bounds are still being synchronized with the source code.");
    request.domain = request.circuit.domain;
    request.circuit_source = state.circuitSource.getSource();
    request.representation = state.representation;
    if (state.representation === "custom") {
      if (!state.customSites || state.customSignature !== encodingSignature()) throw new Error("The custom arrangement is being updated. Wait a moment before learning.");
      request.sites = state.customSites;
    }
  }
  return request;
}

// 10^x written as a plain number when short, and as a power of ten otherwise.
function powerOfTen(log10) {
  if (!Number.isFinite(log10)) return "—";
  if (Math.abs(log10) < 4) return number(10 ** log10, 3);
  const exponent = Math.floor(log10);
  const superscript = String(Math.abs(exponent)).replace(/\d/g, digit => "⁰¹²³⁴⁵⁶⁷⁸⁹"[digit]);
  return `${(10 ** (log10 - exponent)).toFixed(1)} × 10${exponent < 0 ? "⁻" : ""}${superscript}`;
}

function nodeSummary(designs) {
  const sites = designs.filter(item => item.dimension > 1);
  const equispaced = sites.filter(item => item.node_rule === "equispaced").length;
  if (!sites.length) return "no sampled dictionaries";
  if (equispaced === sites.length) return "equispaced nodes";
  if (!equispaced) return "random nodes";
  return `equispaced nodes on ${equispaced} of ${sites.length} dictionaries`;
}

function worstDesign(designs) {
  return designs.reduce((worst, item) => !worst || item.inverse_norm > worst.inverse_norm ? item : worst, null);
}

// The sampling design depends only on dictionaries and nodes, so its conditioning is known before learning.
function scheduleDesign() {
  if (!$("design-readout")) return;
  clearTimeout(state.designTimer);
  state.designTimer = setTimeout(updateDesign, 350);
}

async function updateDesign() {
  const readout = $("design-readout");
  if (state.busy) return;
  const token = ++state.designToken;
  let request;
  try { request = readRequest(); }
  catch { readout.textContent = ""; readout.className = "design-readout"; return; }
  try {
    const design = await api("/api/design", { method: "POST", body: JSON.stringify(request) });
    if (token !== state.designToken) return;
    const worst = worstDesign(design.designs);
    const log10 = design.log10_inversion_amplification;
    const parts = [`Inversion amplification ∏‖V⁺‖ ≈ ${powerOfTen(log10)}`];
    if (worst && design.designs.length > 1) parts.push(`worst dictionary Φ${worst.site + 1}, condition number ${number(worst.condition_number, 3)}`);
    else if (worst) parts.push(`condition number ${number(worst.condition_number, 3)}`);
    parts.push(nodeSummary(design.designs));
    if (request.options.shots) parts.push(`noise per query σ ≈ ${number(1 / Math.sqrt(request.options.shots), 2)}`);
    let text = `${parts.join(" · ")}.`;
    if (log10 > 3) text += " Errors in the sampled values can grow by this factor: raise oversampling or use equispaced nodes.";
    readout.textContent = text;
    readout.className = `design-readout${log10 > 3 ? " warning" : ""}`;
  } catch (error) {
    if (token !== state.designToken) return;
    readout.textContent = error.message;
    readout.className = "design-readout warning";
  }
}

let chartObserver;

function resetResults() {
  chartObserver?.disconnect();
  state.result = null;
  state.core = null;
  $("result-title").textContent = state.request.name;
  $("download-button").hidden = true;
  $("chart-legend").hidden = true;
  $("chart-container").replaceChildren();
  const empty = element("div", "empty-state");
  empty.append(element("h3", null, "Learning…"));
  $("chart-container").append(empty);
  $("chart-caption").textContent = "Function preview";
  $("chart-note").textContent = "";
  $("metric-error").textContent = "—";
  $("metric-queries").textContent = "—";
  $("metric-bonds").textContent = "—";
  $("metric-validation").textContent = "Independent validation";
  $("predict-inputs").replaceChildren();
  $("predict-button").disabled = true;
  $("prediction-value").textContent = "—";
  $("basis-values").hidden = true;
  showError("prediction-error");
  $("tensor-network").replaceChildren(element("p", "muted", "No model yet"));
  $("core-inspector").hidden = true;
  $("dictionary-inspector").hidden = true;
  $("model-facts").replaceChildren();
  $("metadata-details").hidden = true;
  $("error-budget").hidden = true;
  $("coefficient-structure").hidden = true;
  $("slice-controls").hidden = true;
  state.sliceToken += 1;
  state.view = "dictionary"; state.viewData = null;
  $("basis-view").value = "dictionary"; $("basis-view").disabled = true;
  $("basis-view-note").textContent = VIEW_HELP;
}

function sweepRepresentations() {
  const names = ["canonical", "merged"].filter(name => representationData(name)?.available);
  if (state.customSites && state.customSignature === encodingSignature()) names.push("custom");
  return names;
}

function renderSweepControls() {
  const axis = $("sweep-axis").value;
  const field = $("sweep-values-field");
  field.hidden = !axis || axis === "representation";
  if (axis && axis !== "representation" && $("sweep-values").dataset.axis !== axis) {
    $("sweep-values").value = SWEEP_AXES[axis].defaults;
    $("sweep-values").dataset.axis = axis;
  }
  $("sweep-note").textContent = !axis ? ""
    : axis === "representation" ? `Learns the same circuit once per representation: ${sweepRepresentations().map(name => REPRESENTATION_LABELS[name]).join(", ")}.`
    : "One run per value, with every other setting as above. Up to 8 values, separated by commas.";
}

function readSweep() {
  const axis = $("sweep-axis").value;
  if (!axis) return null;
  if (axis === "representation") {
    const values = sweepRepresentations();
    if (values.length < 2) throw new Error("A representation sweep needs at least two available representations.");
    return { axis, values };
  }
  const { label, min, max } = SWEEP_AXES[axis];
  const values = $("sweep-values").value.split(/[\s,;]+/).filter(Boolean).map(text => integer(text, `${label} sweep values`));
  if (values.length < 2 || values.length > 8) throw new Error("Enter between 2 and 8 sweep values.");
  if (new Set(values).size !== values.length) throw new Error("Sweep values must be distinct.");
  if (values.some(value => value < min || value > max)) throw new Error(`${label} sweep values must lie between ${min} and ${max.toLocaleString()}.`);
  return { axis, values };
}

function sweepLabel(axis, value) {
  return axis === "representation" ? REPRESENTATION_LABELS[value] : `${SWEEP_AXES[axis].short} ${Number(value).toLocaleString()}`;
}

function sweepRequest(sweep, index) {
  const request = JSON.parse(JSON.stringify(sweep.base));
  const value = sweep.values[index];
  if (sweep.axis === "representation") {
    request.representation = value;
    if (value === "custom") request.sites = state.customSites; else delete request.sites;
  } else request.options[sweep.axis] = value;
  request.name = `${sweep.base.name} · ${sweepLabel(sweep.axis, value)}`.slice(0, 150);
  return request;
}

async function submitRun(request) {
  state.request = request;
  resetResults();
  const step = state.sweep ? `Run ${state.sweep.index + 1} of ${state.sweep.values.length} · ` : "";
  setStatus("running", `${step}Preparing learning…`, "");
  const response = await api("/api/jobs", { method: "POST", body: JSON.stringify(request) });
  state.jobId = response.id;
  if (!state.jobId) throw new Error("The engine did not return a job identifier.");
  pollJob();
}

// The results sit beside or below a long form; bring them into view when a run starts.
function showResultsPanel() {
  const panel = document.querySelector(".results-panel");
  if (!panel?.getBoundingClientRect) return;
  const box = panel.getBoundingClientRect();
  if (box.top >= 0 && box.top < window.innerHeight * 0.6) return;
  const calm = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
  panel.scrollIntoView({ block: "start", behavior: calm ? "auto" : "smooth" });
}

async function startLearning(event) {
  event.preventDefault();
  if (state.busy) return;
  showError("form-error");
  try {
    const base = readRequest();
    const sweep = readSweep();
    state.sweep = sweep ? { ...sweep, id: ++state.sweepCounter, index: 0, base, failures: 0 } : null;
    setBusy(true);
    showResultsPanel();
    await submitRun(state.sweep ? sweepRequest(state.sweep, 0) : base);
  } catch (error) {
    state.sweep = null;
    setBusy(false);
    showError("form-error", error.message);
    setStatus("failed", "The run could not start", error.message);
  }
}

// After a run of a sweep ends, start the next value; returns false when nothing is left.
async function continueSweep() {
  const sweep = state.sweep;
  if (!sweep || sweep.index + 1 >= sweep.values.length) return false;
  sweep.index += 1;
  try { await submitRun(sweepRequest(sweep, sweep.index)); return true; }
  catch (error) {
    sweep.failures += 1;
    recordFailedRun(sweepRequest(sweep, sweep.index), error.message);
    return continueSweep();
  }
}

function finishSweep() {
  const sweep = state.sweep;
  state.sweep = null;
  setBusy(false);
  if (!sweep) return;
  const done = sweep.values.length - sweep.failures;
  setStatus(done ? "complete" : "failed", `Sweep over ${SWEEP_AXES[sweep.axis].label.toLowerCase()} finished`, `${done} of ${sweep.values.length} runs completed · see Runs`);
  if (done) setTab("runs");
}

async function pollJob() {
  clearTimeout(state.polling);
  if (!state.jobId || !state.busy) return;
  try {
    const job = await api(`/api/jobs/${encodeURIComponent(state.jobId)}`);
    const progress = job.progress || {};
    const queries = progress.query_counts?.training ?? progress.training_queries ?? progress.query_count;
    if (queries !== undefined) $("metric-queries").textContent = Number(queries).toLocaleString();
    if (job.status === "complete") {
      state.result = job.result;
      if (!state.result) throw new Error("The completed run did not contain a model result.");
      recordRun(state.jobId, state.request, state.result);
      if (state.sweep) {
        // Keep the last finished model so it can be shown when the sweep ends.
        const result = state.result, jobId = state.jobId, request = state.request;
        if (await continueSweep()) return;
        state.jobId = jobId; state.request = request; state.result = result;
        renderResult(result);
        finishSweep();
        return;
      }
      setBusy(false);
      renderResult(state.result);
      setStatus("complete", "Model ready", "");
      return;
    }
    if (job.status === "failed") {
      const message = job.error || "The learning engine could not finish this run.";
      if (state.sweep) {
        state.sweep.failures += 1;
        recordFailedRun(state.request, message);
        if (await continueSweep()) return;
        renderEndedState("Sweep finished", "");
        finishSweep();
        return;
      }
      setBusy(false);
      setStatus("failed", "Learning stopped", message);
      showError("form-error", message);
      renderEndedState("Learning stopped", "");
      return;
    }
    if (job.status === "cancelled") {
      state.sweep = null;
      setBusy(false);
      setStatus("cancelled", "Learning cancelled", "");
      renderEndedState("Learning cancelled", "");
      return;
    }
    const titles = { sampling: "Sampling the target function", training: "Learning the coefficient structure", tci: "Learning the coefficient structure", validating: "Checking fresh inputs", validation: "Checking fresh inputs", converting: "Recovering the coefficient MPS", preparing: "Preparing dictionaries and sample points", queued: "Waiting for the engine", starting: "Starting the learning engine", preview: "Preparing your function preview" };
    const step = state.sweep ? `Run ${state.sweep.index + 1} of ${state.sweep.values.length} · ` : "";
    const title = step + (titles[progress.stage] || (job.status === "queued" ? "Waiting for the engine" : "Learning your function"));
    const message = progress.queries !== undefined ? `${Number(progress.queries).toLocaleString()} evaluations` : "";
    setStatus("running", title, message);
    state.polling = setTimeout(pollJob, 550);
  } catch (error) {
    state.sweep = null;
    setBusy(false);
    showError("form-error", `Could not read the learning run: ${error.message}`);
    setStatus("failed", "Connection interrupted", "The engine may still be working. Refresh this page after checking the local server.");
  }
}

function renderEndedState(title, text) {
  const empty = element("div", "empty-state");
  empty.append(element("h3", null, title), element("p", null, text));
  $("chart-container").replaceChildren(empty);
}

function modelValidation(result) { return result.validation || result.metadata?.validation || {}; }

function renderResult(result) {
  const metadata = result.metadata || {};
  const validation = modelValidation(result);
  const queries = metadata.query_counts?.training ?? metadata.training_queries ?? metadata.oracle_queries ?? metadata.queries ?? metadata.query_count ?? result.training_queries ?? result.queries;
  $("result-title").textContent = result.name || state.request.name;
  $("metric-error").textContent = number(validation.relative_l2);
  $("metric-validation").textContent = validation.n_samples ? `${Number(validation.n_samples).toLocaleString()} fresh input points` : "Independent validation";
  if (queries !== undefined) $("metric-queries").textContent = Number(queries).toLocaleString();
  const bonds = result.bond_dimensions || [];
  $("metric-bonds").textContent = bonds.length ? bonds.join(" · ") : "1 site";
  const download = $("download-button");
  download.href = `/api/jobs/${encodeURIComponent(state.jobId)}/download`;
  download.download = `${(result.name || "model").replace(/[^a-z0-9_-]+/gi, "-")}.ftci`;
  download.hidden = false;
  renderChart(result.preview);
  renderSliceControls(result);
  renderPredictInputs(result.domain || state.request.domain);
  state.view = "dictionary"; state.viewData = null;
  $("basis-view").value = "dictionary"; $("basis-view").disabled = false;
  $("basis-view-note").textContent = VIEW_HELP;
  renderNetwork(result);
  renderErrorBudget(result);
  renderStructure(result);
  renderFacts(result);
}

function renderChart(preview) {
  if (!preview || !Array.isArray(preview.x) || !preview.x.length) {
    renderEndedState("Your model is ready.", "Use the input fields below to evaluate it. A one-dimensional preview is not available for this run.");
    return;
  }
  const points = preview.x.map((x, i) => ({ x: Number(x), truth: Number(preview.truth?.[i]), prediction: Number(preview.prediction?.[i]) })).filter((p) => Number.isFinite(p.x) && Number.isFinite(p.prediction));
  if (!points.length) return;
  points.sort((a, b) => a.x - b.x);
  let xmin = Math.min(...points.map((p) => p.x));
  let xmax = Math.max(...points.map((p) => p.x));
  if (xmax === xmin) xmax += 1;
  const ys = points.flatMap((p) => [p.truth, p.prediction]).filter(Number.isFinite);
  let ymin = Math.min(...ys), ymax = Math.max(...ys);
  const ypad = ymax === ymin ? Math.max(0.1, Math.abs(ymin) * 0.1) : (ymax - ymin) * 0.13;
  ymin -= ypad; ymax += ypad;
  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", "Target function and learned FTCI prediction over a one-dimensional input slice");
  const varying = preview.varying_input ?? 0;
  const makeTicks = (min, max) => Array.from({ length: 5 }, (_, i) => {
    const value = min + (max - min) * i / 4;
    return { value, label: number(value, 3) };
  });
  const xTicks = makeTicks(xmin, xmax), yTicks = makeTicks(ymin, ymax);
  const labelWidth = (ticks) => 8 * Math.max(...ticks.map(({ label }) => label.length));
  function s(tag, attrs, text) {
    const node = document.createElementNS(ns, tag);
    for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
    if (text !== undefined) node.textContent = text;
    svg.append(node);
  }
  let renderedWidth;
  function draw(width) {
    if (width <= 0 || width === renderedWidth) return;
    renderedWidth = width;
    const height = Math.min(350, Math.max(210, Math.round(width * .4)));
    const left = 14 + labelWidth(yTicks);
    const right = 12, top = 14, bottom = 48;
    const plotWidth = width - left - right;
    const xp = (x) => left + (x - xmin) / (xmax - xmin) * plotWidth;
    const yp = (y) => height - bottom - (y - ymin) / (ymax - ymin) * (height - bottom - top);
    const axisLabel = { class: "chart-axis-label", fill: "#5f685d", "font-size": 12, "font-family": "monospace" };
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    svg.replaceChildren();
    for (const { value, label } of yTicks) {
      const y = yp(value);
      s("line", { class: "chart-gridline", x1: left, x2: width - right, y1: y, y2: y, stroke: "#e7e9df", "stroke-width": 1 });
      s("text", { ...axisLabel, x: left - 9, y: y + 4, "text-anchor": "end" }, label);
    }
    const tickSpacing = 1.5 * labelWidth(xTicks) + 6;
    const step = plotWidth >= 4 * tickSpacing ? 1 : plotWidth >= 2 * tickSpacing ? 2 : 4;
    for (let i = 0; i < xTicks.length; i += step) {
      const { value, label } = xTicks[i];
      const anchor = i === 0 ? "start" : i === xTicks.length - 1 ? "end" : "middle";
      s("text", { ...axisLabel, x: xp(value), y: height - bottom + 20, "text-anchor": anchor }, label);
    }
    const path = (field) => points.filter((p) => Number.isFinite(p[field])).map((p, i) => `${i ? "L" : "M"}${xp(p.x).toFixed(2)},${yp(p[field]).toFixed(2)}`).join(" ");
    s("path", { d: path("truth"), fill: "none", stroke: "#68736a", "stroke-width": 2.5, "stroke-linejoin": "round", "stroke-linecap": "round" });
    s("path", { d: path("prediction"), fill: "none", stroke: "var(--accent)", "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round", "stroke-dasharray": "5 4" });
    s("text", { ...axisLabel, x: width - right, y: height - 6, "text-anchor": "end" }, `x${varying}`);
  }
  $("chart-container").replaceChildren(svg);
  // Match the view box to its rendered width so labels keep a readable size.
  draw(Math.round(svg.getBoundingClientRect().width) || 720);
  if (typeof ResizeObserver === "function") {
    chartObserver?.disconnect();
    chartObserver = new ResizeObserver(([entry]) => draw(Math.round(entry.contentRect.width)));
    chartObserver.observe(svg);
  }
  $("chart-legend").hidden = false;
  if ($("chart-legend").firstElementChild) $("chart-legend").firstElementChild.hidden = !points.some((p) => Number.isFinite(p.truth));
  $("chart-caption").textContent = `Input slice · x${varying}`;
  const fixed = preview.fixed_inputs || [];
  const other = fixed.map((value, index) => index === varying ? "" : `x${index} = ${number(value, 3)}`).filter(Boolean);
  if (preview.target_queries) other.push(`target curve from ${Number(preview.target_queries).toLocaleString()} extra queries`);
  $("chart-note").textContent = other.join(" · ");
}

// Which inputs the preview varies, and where the others are held.
function renderSliceControls(result) {
  const domain = result.domain || [];
  const root = $("slice-controls");
  root.hidden = domain.length < 2;
  if (root.hidden) return;
  const preview = result.preview || {};
  const xSelect = $("slice-x"), ySelect = $("slice-y");
  xSelect.replaceChildren();
  ySelect.replaceChildren();
  const none = element("option", null, "None · line plot"); none.value = "";
  ySelect.append(none);
  domain.forEach((_, index) => {
    for (const select of [xSelect, ySelect]) { const option = element("option", null, `x${index}`); option.value = index; select.append(option); }
  });
  xSelect.value = String(preview.varying_input ?? 0);
  ySelect.value = "";
  $("slice-target").checked = true;
  const fixed = $("slice-fixed");
  fixed.replaceChildren();
  domain.forEach((range, index) => {
    const label = element("label", null, `x${index} =`);
    const input = element("input");
    input.type = "number"; input.step = "any"; input.dataset.input = index;
    input.value = number(preview.fixed_inputs?.[index] ?? (range[0] + range[1]) / 2, 6);
    input.setAttribute("aria-label", `Fixed value of x${index}`);
    label.append(input);
    fixed.append(label);
  });
  syncSliceControls();
}

function sliceAxes() {
  const x = Number($("slice-x").value);
  return $("slice-y").value === "" ? [x] : [x, Number($("slice-y").value)];
}

function syncSliceControls() {
  const x = $("slice-x").value;
  for (const option of $("slice-y").children) option.disabled = option.value !== "" && option.value === x;
  if ($("slice-y").value === x) $("slice-y").value = "";
  const axes = sliceAxes();
  for (const label of $("slice-fixed").children) label.hidden = axes.includes(Number(label.querySelector("input").dataset.input));
  const queries = axes.length === 1 ? 61 : 31 * 31;
  $("slice-target-label").textContent = `Compare with the target · ${queries.toLocaleString()} queries`;
}

function scheduleSlice() {
  clearTimeout(state.sliceTimer);
  state.sliceTimer = setTimeout(updateSlice, 300);
}

async function updateSlice() {
  const result = state.result, jobId = state.jobId;
  if (!result || !jobId || state.busy) return;
  const token = ++state.sliceToken;
  const axes = sliceAxes();
  const inputs = [...$("slice-fixed").querySelectorAll("input")];
  if (inputs.some((input, index) => !axes.includes(index) && (input.value.trim() === "" || !Number.isFinite(Number(input.value))))) {
    $("chart-note").textContent = "Enter a number for every input that is held fixed.";
    return;
  }
  const fixed = inputs.map((input, index) => axes.includes(index) ? (result.domain[index][0] + result.domain[index][1]) / 2 : Number(input.value));
  try {
    const data = await api(`/api/jobs/${encodeURIComponent(jobId)}/slice`, { method: "POST", body: JSON.stringify({ axes, fixed, target: $("slice-target").checked }) });
    if (token !== state.sliceToken || state.result !== result) return;
    if (axes.length === 1) renderChart({ x: data.x, truth: data.truth, prediction: data.prediction, varying_input: axes[0], fixed_inputs: data.fixed, target_queries: data.target_queries });
    else renderHeatmaps(data);
  } catch (error) {
    if (token === state.sliceToken) $("chart-note").textContent = error.message;
  }
}

function mixColor(from, to, t) {
  const channel = (hex, i) => parseInt(hex.slice(1 + 2 * i, 3 + 2 * i), 16);
  return `rgb(${[0, 1, 2].map(i => Math.round(channel(from, i) + (channel(to, i) - channel(from, i)) * t)).join(",")})`;
}

// Signed values use two hues around a neutral zero; one-signed values use one hue from light to dark.
function heatmapScale(low, high) {
  if (low < 0 && high > 0) {
    const extent = Math.max(-low, high);
    return { low: -extent, high: extent, color: value => mixColor("#eceae3", value < 0 ? "#2f6f8f" : "#a8402b", Math.min(1, Math.abs(value) / extent)) };
  }
  const span = high - low || 1;
  return { low, high, color: value => mixColor("#f5ebe6", "#8a2f1e", Math.min(1, Math.max(0, (value - low) / span))) };
}

function renderHeatmaps(data) {
  chartObserver?.disconnect();
  const [xAxis, yAxis] = data.axes;
  const panels = data.truth ? [["Target", data.truth], ["Learned", data.prediction]] : [["Learned", data.prediction]];
  const values = panels.flatMap(([, grid]) => grid.flat());
  const scale = heatmapScale(Math.min(...values), Math.max(...values));
  const ns = "http://www.w3.org/2000/svg";
  const make = (parent, tag, attributes, text) => {
    const node = document.createElementNS(ns, tag);
    for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
    if (text !== undefined) node.textContent = text;
    parent.append(node);
    return node;
  };
  const ink = { fill: "#5f685d", "font-size": 11, "font-family": "monospace" };
  const root = element("div", "heatmap-grid");
  for (const [title, grid] of panels) {
    const figure = element("figure", "heatmap");
    figure.append(element("figcaption", null, title));
    const size = 280, left = 46, top = 6, bottom = 34, right = 8;
    const svg = document.createElementNS(ns, "svg");
    svg.setAttribute("viewBox", `0 0 ${left + size + right} ${top + size + bottom}`);
    svg.setAttribute("role", "img");
    svg.setAttribute("aria-label", `${title} function over x${xAxis} and x${yAxis}, values from ${number(Math.min(...grid.flat()), 3)} to ${number(Math.max(...grid.flat()), 3)}`);
    const nx = data.x.length, ny = data.y.length, w = size / nx, h = size / ny;
    grid.forEach((row, j) => row.forEach((value, i) => {
      const cell = make(svg, "rect", { x: (left + i * w).toFixed(2), y: (top + size - (j + 1) * h).toFixed(2), width: (w + 0.4).toFixed(2), height: (h + 0.4).toFixed(2), fill: scale.color(value) });
      make(cell, "title", {}, `x${xAxis} = ${number(data.x[i], 3)}, x${yAxis} = ${number(data.y[j], 3)}: ${number(value, 4)}`);
    }));
    make(svg, "rect", { x: left, y: top, width: size, height: size, fill: "none", stroke: "#c7c2b7", "stroke-width": 1 });
    for (const t of [0, 0.5, 1]) {
      make(svg, "text", { ...ink, x: left + t * size, y: top + size + 15, "text-anchor": t === 0 ? "start" : t === 1 ? "end" : "middle" }, number(data.x[0] + t * (data.x[nx - 1] - data.x[0]), 3));
      make(svg, "text", { ...ink, x: left - 6, y: top + size - t * size + (t === 0 ? 0 : t === 1 ? 9 : 4), "text-anchor": "end" }, number(data.y[0] + t * (data.y[ny - 1] - data.y[0]), 3));
    }
    make(svg, "text", { ...ink, x: left + size, y: top + size + 30, "text-anchor": "end" }, `x${xAxis}`);
    make(svg, "text", { ...ink, x: 4, y: top + 9 + 14, "text-anchor": "start" }, `x${yAxis}`);
    figure.append(svg);
    root.append(figure);
  }
  const legend = element("div", "heatmap-scale");
  const bar = document.createElementNS(ns, "svg");
  bar.setAttribute("viewBox", "0 0 240 12");
  bar.setAttribute("aria-hidden", "true");
  for (let step = 0; step < 60; step++) make(bar, "rect", { x: step * 4, y: 0, width: 4.3, height: 12, fill: scale.color(scale.low + (step + 0.5) / 60 * (scale.high - scale.low)) });
  legend.append(element("span", null, number(scale.low, 3)), bar, element("span", null, number(scale.high, 3)));
  $("chart-container").replaceChildren(root, legend);
  $("chart-legend").hidden = true;
  $("chart-caption").textContent = `Input slice · x${xAxis} × x${yAxis}`;
  const notes = data.fixed.map((value, index) => data.axes.includes(index) ? "" : `x${index} = ${number(value, 3)}`).filter(Boolean);
  if (data.truth) {
    const difference = Math.max(...data.truth.flatMap((row, j) => row.map((value, i) => Math.abs(value - data.prediction[j][i]))));
    notes.push(`largest difference on this grid ${number(difference, 3)}`, `target from ${Number(data.target_queries).toLocaleString()} extra queries`);
  }
  $("chart-note").textContent = notes.join(" · ");
}

function renderPredictInputs(domain) {
  const root = $("predict-inputs");
  root.replaceChildren();
  domain.forEach((range, i) => {
    const wrapper = element("div");
    const label = element("label", null, `x${i}`);
    const input = element("input");
    input.id = `prediction-x${i}`; input.type = "number"; input.step = "any";
    input.value = number((range[0] + range[1]) / 2, 6); input.required = true;
    input.title = `Training domain: ${number(range[0])} to ${number(range[1])}`;
    label.htmlFor = input.id;
    wrapper.append(label, input); root.append(wrapper);
  });
  $("predict-button").disabled = false;
}

async function predict(event) {
  event.preventDefault();
  if (!state.result) return;
  showError("prediction-error");
  $("predict-button").disabled = true;
  try {
    const x = [...$("predict-inputs").querySelectorAll("input")].map((input, i) => numeric(input.value, `x${i}`));
    const response = await api(`/api/jobs/${encodeURIComponent(state.jobId)}/predict`, { method: "POST", body: JSON.stringify({ x }) });
    $("prediction-value").textContent = number(response.value, 10);
    const root = $("basis-values");
    root.replaceChildren();
    (response.basis_values || []).forEach((values, i) => {
      const row = element("div", "basis-value-row");
      row.append(element("span", null, `Φ${i + 1}(x)`), element("span", null, `[${values.map((v) => number(v, 5)).join(", ")}]`));
      root.append(row);
    });
    root.hidden = !root.children.length;
  } catch (error) { showError("prediction-error", error.message); }
  finally { $("predict-button").disabled = false; }
}

// Dictionaries and core shapes of the coefficient basis currently shown.
function viewActive() { return state.view !== "dictionary" && Boolean(state.viewData); }
function activeShapes() { return viewActive() ? state.viewData.core_shapes : state.result?.core_shapes || []; }
function activeDictionary(index) {
  const learned = state.result?.basis?.[index] || {};
  if (!viewActive()) return { ...learned, formulas: true };
  const shown = state.viewData.dictionaries[index] || {};
  return { ...learned, dimension: shown.dimension, expressions: shown.expressions || shown.labels, formulas: Boolean(shown.expressions), portable: true };
}

async function setView(name) {
  if (!state.result || !state.jobId || state.busy) return;
  const result = state.result;
  if (name === "dictionary") { state.view = name; state.viewData = null; }
  else {
    try {
      const data = await api(`/api/jobs/${encodeURIComponent(state.jobId)}/view?basis=${encodeURIComponent(name)}`);
      if (state.result !== result) return;
      state.view = name; state.viewData = data;
    } catch (error) {
      $("basis-view").value = state.view;
      $("basis-view-note").textContent = error.message;
      return;
    }
  }
  $("basis-view-note").textContent = state.viewData?.description || VIEW_HELP;
  renderNetwork(result);
  renderStructure(result);
}

function renderNetwork(result) {
  const root = $("tensor-network");
  root.replaceChildren();
  state.selectedTensorNode = null;
  state.core = null;
  const shapes = activeShapes();
  shapes.forEach((shape, index) => {
    const site = element("div", "tensor-site");
    const button = element("button", "tensor-core-node");
    button.type = "button";
    button.setAttribute("aria-label", `Inspect coefficient core ${index + 1}, shape ${shape.join(" by ")}`);
    button.setAttribute("aria-controls", "coefficient-inspector");
    button.setAttribute("aria-pressed", "false");
    button.dataset.index = index;
    button.dataset.kind = "core";
    button.append(element("span", "tensor-symbol", `A${index + 1}`), element("span", "tensor-size", shape.join(" × ")));
    button.addEventListener("click", () => selectCore(index));
    const leg = element("span", "tensor-physical-leg");
    leg.setAttribute("aria-hidden", "true");
    const dictionary = element("button", "tensor-dictionary-node");
    dictionary.type = "button";
    dictionary.dataset.index = index;
    dictionary.dataset.kind = "dictionary";
    dictionary.setAttribute("aria-label", `Inspect function dictionary ${index + 1}, dimension ${result.basis?.[index]?.dimension || shape[1]}`);
    dictionary.setAttribute("aria-controls", "dictionary-inspector");
    dictionary.setAttribute("aria-pressed", "false");
    const triangle = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    triangle.setAttribute("viewBox", "0 0 90 76");
    triangle.setAttribute("class", "tensor-dictionary-triangle");
    triangle.setAttribute("aria-hidden", "true");
    triangle.setAttribute("focusable", "false");
    const polygon = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
    polygon.setAttribute("points", "7,5 83,5 45,69");
    triangle.append(polygon);
    dictionary.append(triangle, element("span", "tensor-dictionary-symbol", `Φ${index + 1}(x)`));
    dictionary.addEventListener("click", () => selectDictionary(index));
    site.append(button, leg, dictionary);
    root.append(site);
    if (index < shapes.length - 1) {
      const link = element("span", "tensor-link");
      link.dataset.bond = shape[2];
      link.setAttribute("aria-hidden", "true");
      root.append(link);
    }
  });
  if (shapes.length) selectCore(0);
  else $("core-inspector").hidden = true;
}

function selectTensorNode(kind, index) {
  state.selectedTensorNode = `${kind}:${index}`;
  for (const button of $("tensor-network").querySelectorAll("button[data-kind]")) {
    const selected = button.dataset.kind === kind && Number(button.dataset.index) === index;
    button.classList.toggle("selected", selected);
    button.setAttribute("aria-pressed", String(selected));
  }
  $("core-inspector").hidden = false;
  $("coefficient-inspector").hidden = kind !== "core";
  $("dictionary-inspector").hidden = kind !== "dictionary";
}

function selectDictionary(index) {
  const result = state.result;
  if (!result?.core_shapes?.[index]) return;
  selectTensorNode("dictionary", index);
  const dictionary = activeDictionary(index);
  const expressions = dictionary.expressions || [];
  const dimension = dictionary.dimension || activeShapes()[index][1];
  const indices = Array.isArray(dictionary.input_indices) ? dictionary.input_indices : null;
  const domain = result.domain || state.request?.domain || [];
  $("core-title").textContent = `Function dictionary Φ${index + 1}(x)`;
  $("core-shape").textContent = `Dimension ${dimension}`;
  const facts = [];
  if (viewActive()) facts.push(["Basis", VIEW_LABELS[state.view]]);
  if (dictionary.name) facts.push(["Name", dictionary.name]);
  facts.push(["Inputs", indices === null ? "Not recorded" : indices.length ? indices.map(i => `x${i}`).join(", ") : "Constant"]);
  if (indices?.length) {
    facts.push(["Input domain", indices.map(i => domain[i] ? `x${i} ∈ [${number(domain[i][0], 5)}, ${number(domain[i][1], 5)}]` : `x${i}`).join(" · ")]);
  }
  const site = result.metadata?.sites?.[index];
  const members = (site?.encoding_indices || [index]).map(i => result.metadata?.encodings?.[i]).filter(Boolean);
  const encoding = site || members[0];
  const generators = [...new Set(members.map(item => item.generator).filter(Boolean))];
  if (members.length > 1) facts.push(["Merged encoding gates", compactList(site.encoding_indices.map(i => `E${i + 1}`), 12)]);
  if (generators.length) facts.push([generators.length === 1 ? "Generator" : "Generators", generators.join(", ")]);
  if (site?.classes?.length > 1) {
    facts.push(["Structure", `Tensor product of ${site.classes.length} merged groups, each with its own training angle`]);
    facts.push(["Preprocessing", site.classes.map(group => group.preprocessing ?? "callable").join("  ⊗  ")]);
    facts.push(["Frequencies", site.classes.map(group => group.frequencies.length ? group.frequencies.map(value => number(value, 6)).join(", ") : "0").join("  ⊗  ")]);
  } else {
    if (encoding?.preprocessing !== undefined && encoding.preprocessing !== null) facts.push(["Preprocessing", encoding.preprocessing]);
    if (Array.isArray(encoding?.frequencies)) facts.push(["Frequencies", encoding.frequencies.length ? encoding.frequencies.map(value => number(value, 6)).join(", ") : "0"]);
  }
  const factsRoot = $("dictionary-facts");
  factsRoot.replaceChildren();
  for (const [label, value] of facts) {
    const row = element("div");
    row.append(element("dt", null, label), element("dd", null, value));
    factsRoot.append(row);
  }
  const functionsRoot = $("dictionary-functions");
  functionsRoot.replaceChildren();
  if (!expressions.length) {
    functionsRoot.append(element("p", "field-help", dictionary.portable === false ? "This dictionary uses a local callable." : "Basis expressions are unavailable."));
    return;
  }
  const table = element("table", "dictionary-table");
  const head = element("thead");
  const heading = element("tr");
  for (const label of ["Index", "Basis function"]) {
    const cell = element("th", null, label);
    cell.scope = "col";
    heading.append(cell);
  }
  head.append(heading);
  const body = element("tbody");
  expressions.forEach((expression, basisIndex) => {
    const row = element("tr");
    const label = element("th", null, String(basisIndex));
    label.scope = "row";
    const value = element("td");
    const formula = element("span", dictionary.formulas ? "basis-formula" : "basis-label");
    if (dictionary.formulas) renderBasisFormula(formula, expression);
    else formula.textContent = expression;
    value.append(formula);
    row.append(label, value);
    body.append(row);
  });
  table.append(head, body);
  functionsRoot.append(table);
}

function renderBasisFormula(container, expression) {
  if (typeof FTCIFormula !== "undefined") FTCIFormula.render(container, expression);
  else container.textContent = expression;
}

function renderCoreBasisExpression() {
  const index = Number($("basis-index-select").value || 0);
  const dictionary = activeDictionary(state.selectedCore);
  const expression = dictionary.expressions?.[index];
  const display = $("core-basis-expression");
  if (dictionary.formulas) renderBasisFormula(display, expression ?? "");
  else display.textContent = expression ?? "";
  display.hidden = !expression;
}

async function selectCore(index) {
  const result = state.result;
  if (!result?.core_shapes?.[index]) return;
  const jobId = state.jobId;
  state.selectedCore = index;
  selectTensorNode("core", index);
  state.core = null;
  $("core-title").textContent = `Coefficient core A${index + 1}`;
  const shape = activeShapes()[index];
  $("core-shape").textContent = shape.join(" × ");
  $("core-table-container").replaceChildren(element("p", "field-help", "Loading coefficient values…"));
  $("core-table-note").textContent = "";
  $("basis-index-select").replaceChildren();
  for (let i = 0; i < shape[1]; i++) {
    const option = element("option", null, String(i)); option.value = i; $("basis-index-select").append(option);
  }
  renderCoreBasisExpression();
  try {
    const core = await api(`/api/jobs/${encodeURIComponent(jobId)}/cores/${index}${viewActive() ? `?basis=${encodeURIComponent(state.view)}` : ""}`);
    if (state.selectedTensorNode !== `core:${index}` || state.result !== result || state.jobId !== jobId) return;
    state.core = core;
    renderCoreTable();
  } catch (error) {
    if (state.selectedTensorNode !== `core:${index}` || state.result !== result || state.jobId !== jobId) return;
    state.core = null;
    $("core-table-container").replaceChildren(element("p", "error-message", error.message));
    $("core-table-note").textContent = "Full arrays are in the model export.";
  }
}

function renderCoreTable() {
  if (!state.selectedTensorNode?.startsWith("core:")) return;
  renderCoreBasisExpression();
  if (!state.core) return;
  const core = state.core;
  const data = core.data || core.values;
  const shape = core.shape;
  const basis = Number($("basis-index-select").value || 0);
  if (!data || !Number.isInteger(basis) || basis < 0 || basis >= shape[1] || !Array.isArray(data[0]?.[basis])) {
    $("core-table-container").replaceChildren(element("p", "field-help", "Core values are unavailable."));
    return;
  }
  const rows = shape[0], cols = shape[2];
  const table = element("table", "core-table");
  table.setAttribute("aria-label", `Left × right bond matrix, basis index ${basis}`);
  const thead = element("thead");
  const header = element("tr"); header.append(element("th", null, "left ↓ / right →"));
  for (let c = 0; c < cols; c++) header.append(element("th", null, String(c)));
  thead.append(header); table.append(thead);
  const tbody = element("tbody");
  for (let r = 0; r < rows; r++) {
    const row = element("tr"); row.append(element("td", null, String(r)));
    for (let c = 0; c < cols; c++) row.append(element("td", null, number(data[r]?.[basis]?.[c], 6)));
    tbody.append(row);
  }
  table.append(tbody);
  $("core-table-container").replaceChildren(table);
  $("core-table-note").textContent = "Zero-based indices.";
}

function samplingLabel(metadata) {
  const designs = metadata.designs || [];
  const rule = nodeSummary(designs).replace(" nodes", "");
  const parts = [`${rule} ×${number(metadata.oversampling ?? 1, 3)}`];
  if (metadata.shots) parts.push(`${Number(metadata.shots).toLocaleString()} shots`);
  return parts.join(" · ");
}

function budgetCard(title, value, lines) {
  const card = element("div", "budget-card");
  card.append(element("span", "budget-title", title), element("strong", null, value));
  for (const line of lines.filter(Boolean)) card.append(element("small", null, line));
  return card;
}

// The three error sources of quantum-function learning, followed by the measured outcome.
function renderErrorBudget(result) {
  const metadata = result.metadata || {};
  const validation = modelValidation(result);
  const designs = metadata.designs || [];
  const section = $("error-budget");
  section.hidden = !designs.length;
  if (!designs.length) return;
  const training = metadata.query_counts?.training;
  const sigma = metadata.shots ? 1 / Math.sqrt(metadata.shots) : 0;
  const errors = (metadata.pivot_errors || []).filter(value => value !== null);
  const pivot = errors.length ? errors[errors.length - 1] : null;
  const bond = Math.max(1, ...(result.bond_dimensions || []));
  const cap = metadata.max_bond;
  const capped = cap !== undefined && bond >= cap;
  const log10 = metadata.log10_inversion_amplification;
  const worst = worstDesign(designs);
  const cards = $("budget-cards");
  cards.replaceChildren(
    budgetCard("1 · Oracle", metadata.shots ? `σ ≈ ${number(sigma, 2)}` : "Exact", [
      metadata.shots ? `${Number(metadata.shots).toLocaleString()} simulated shots per query` : "No shot noise",
      training !== undefined ? `${Number(training).toLocaleString()} training queries` : "",
    ]),
    budgetCard("2 · TCI compression", pivot === null ? "—" : number(pivot, 3), [
      pivot === null ? "No pivot error recorded" : `last pivot error, tolerance ${number(metadata.tci_tolerance, 3)} ${metadata.tci_tolerance_met ? "met" : "not met"}`,
      result.bond_dimensions?.length ? `bond ${bond} of cap ${cap}${capped ? ", cap reached" : ""}` : "single dictionary, no compression",
    ]),
    budgetCard("3 · Inversion", `× ${powerOfTen(log10)}`, [
      "amplification bound ∏‖V⁺‖",
      worst ? `worst Φ${worst.site + 1}: condition number ${number(worst.condition_number, 3)}` : "",
      `${nodeSummary(designs)}, oversampling ${number(metadata.oversampling ?? 1, 3)}`,
    ]),
    budgetCard("Function error", number(validation.relative_l2), [
      "relative L² on fresh inputs",
      validation.n_samples ? `${Number(validation.n_samples).toLocaleString()} points, exact target` : "",
    ]),
  );
  const notes = [];
  if (metadata.zero_candidate) notes.push("Every sampled grid value was zero, so the model is the zero function.");
  if (metadata.shots) notes.push(`Shot noise puts a floor of about ${number(sigma, 2)} under the pivot error, so the TCI tolerance is not expected to be met.`);
  if (pivot !== null && !metadata.tci_tolerance_met && capped && !metadata.shots) notes.push("TCI stopped at the bond cap before reaching its tolerance. A larger maximum bond can lower the error.");
  if (pivot !== null && !metadata.tci_tolerance_met && !capped && !metadata.shots) notes.push("TCI stopped before reaching its tolerance although the bond cap was not reached.");
  if (log10 > 3) notes.push(`The inversion step can amplify errors of the sampled tensor by up to ${powerOfTen(log10)}. More oversampling or equispaced nodes lower this factor.`);
  if (!notes.length) notes.push("TCI reached its tolerance and the inversion step is well conditioned.");
  $("budget-notes").replaceChildren(...notes.map(text => element("li", null, text)));
}

function structureItem(label, value, detail) {
  const item = element("div");
  const data = element("dd", null, value);
  data.append(element("small", null, detail));
  item.append(element("dt", null, label), data);
  return item;
}

// How the learned coefficients are organized: entanglement across cuts and spreading over the basis.
function renderStructure(result) {
  const structure = viewActive() ? state.viewData.structure : result.structure;
  $("structure-basis").textContent = `Coefficient basis: ${VIEW_LABELS[state.view]}. Choose another one in the Tensor network view.`;
  const section = $("coefficient-structure");
  const root = $("structure-content");
  section.hidden = !structure || Boolean(structure.error);
  if (section.hidden) return;
  root.replaceChildren();
  if (structure.zero) { root.append(element("p", "muted", "The coefficient tensor is zero.")); return; }
  const spreading = structure.spreading;
  const total = Math.exp(spreading.maximum);
  const maxBond = Math.max(1, ...structure.bonds.map(bond => bond.dimension));
  const summary = element("dl", "representation-summary structure-summary");
  summary.append(
    structureItem("Entanglement", `${number(structure.maximum_bond_entropy, 3)} nats`, `largest over cuts, bond dimension ${maxBond}`),
    structureItem("Spreading", `${number(spreading.shannon, 3)} nats`, `of at most ${number(spreading.maximum, 3)}`),
    structureItem("Effective basis functions", number(spreading.participation, 3), `of ${total < 1e15 ? Math.round(total).toLocaleString() : powerOfTen(spreading.maximum / Math.LN10)} (participation ratio)`),
  );
  root.append(summary);
  root.append(element("p", "field-help", `${spreading.method === "exact" ? "Computed exactly" : `Spreading estimated from ${Number(spreading.samples).toLocaleString()} sampled coefficients`}. Entanglement limits the bond dimension an MPS needs; spreading limits what a sparse ansatz could capture. Both refer to the squared entries of the normalized coefficient tensor and depend on the chosen dictionaries.`));
  if (structure.bonds.length) {
    const container = element("div", "site-table-container");
    const table = element("table", "site-table");
    const head = element("thead"), heading = element("tr");
    for (const label of ["Cut", "Bond dimension", "Entanglement entropy", "Schmidt weights"]) { const cell = element("th", null, label); cell.scope = "col"; heading.append(cell); }
    head.append(heading);
    const body = element("tbody");
    for (const bond of structure.bonds) {
      const row = element("tr");
      const label = element("th", null, `Φ${bond.bond} | Φ${bond.bond + 1}`); label.scope = "row";
      const bar = element("div", "schmidt-bar");
      const weights = bond.schmidt_values.map(value => value * value);
      bar.setAttribute("role", "img");
      bar.setAttribute("aria-label", `Largest Schmidt weights ${weights.slice(0, 4).map(value => number(value, 2)).join(", ")}`);
      weights.forEach((weight, index) => {
        if (weight < 0.002) return;
        const segment = element("span", index % 2 ? "odd" : "");
        segment.style.flexGrow = String(weight);
        segment.title = `Schmidt weight ${number(weight, 3)}`;
        bar.append(segment);
      });
      const barCell = element("td"); barCell.append(bar);
      row.append(label, element("td", "numeric", String(bond.dimension)), element("td", "numeric", `${number(bond.entropy, 3)} nats`), barCell);
      body.append(row);
    }
    table.append(head, body); container.append(table); root.append(container);
  }
  const support = structure.support_weights;
  if (Array.isArray(support) && support.length > 1) {
    root.append(element("h4", "structure-heading", "Coefficient weight by support size"));
    root.append(element("p", "field-help", "Share of the squared coefficients on basis functions that are non-constant on exactly k dictionaries. A low-degree truncation keeps only small k."));
    const container = element("div", "site-table-container");
    const table = element("table", "site-table support-table");
    const body = element("tbody");
    const largest = Math.max(...support);
    support.forEach((weight, size) => {
      const row = element("tr");
      const label = element("th", null, `k = ${size}`); label.scope = "row";
      const bar = element("div", "support-bar");
      const fill = element("span");
      fill.style.width = `${largest > 0 ? 100 * weight / largest : 0}%`;
      bar.append(fill);
      const barCell = element("td"); barCell.append(bar);
      row.append(label, element("td", "numeric", `${(100 * weight).toFixed(weight < 0.001 && weight > 0 ? 3 : 1)}%`), barCell);
      body.append(row);
    });
    table.append(body); container.append(table); root.append(container);
  }
}

function renderFacts(result) {
  const metadata = result.metadata || {};
  const validation = modelValidation(result);
  const facts = [
    ["Input dimensions", String((result.domain || state.request.domain).length)],
    ["Dictionary dimensions", (result.basis || []).map((basis) => basis.dimension || basis.expressions?.length).join(" × ")],
    ["Core axis order", "Left bond · basis index · right bond"],
    ["Validation RMSE", number(validation.rmse, 7)],
    ["Maximum absolute error", number(validation.max_absolute_error, 7)],
    ["Learning method", metadata.method || metadata.algorithm || "Function tensor cross interpolation"],
    ["Termination", metadata.termination_reason || metadata.stop_reason || metadata.termination || "Run completed"],
    ["Representation", metadata.representation
      ? `${REPRESENTATION_LABELS[metadata.representation] || metadata.representation} · ${(result.basis || []).length} dictionaries for ${(metadata.encodings || []).length} encoding gates`
      : "Function dictionaries + coefficient MPS"],
  ];
  if (metadata.query_counts) facts.splice(5, 0, ["Query accounting", `${metadata.query_counts.training || 0} training + ${metadata.query_counts.validation || 0} validation + ${metadata.query_counts.preview || 0} preview = ${metadata.queries ?? Object.values(metadata.query_counts).reduce((sum, value) => sum + value, 0)} total`]);
  if (metadata.tci_tolerance_met !== undefined && metadata.tci_tolerance_met !== null) facts.push(["Internal TCI tolerance", metadata.tci_tolerance_met ? "Met" : "Not met"]);
  if (metadata.seconds !== undefined || metadata.elapsed_seconds !== undefined) facts.splice(6, 0, ["Elapsed time", `${number(metadata.seconds ?? metadata.elapsed_seconds, 4)} seconds`]);
  const root = $("model-facts"); root.replaceChildren();
  for (const [label, value] of facts) { const row = element("div"); row.append(element("dt", null, label), element("dd", null, value || "—")); root.append(row); }
  $("model-metadata").textContent = JSON.stringify(metadata, null, 2);
  $("metadata-details").hidden = false;
}

function recordRun(jobId, request, result) {
  const metadata = result.metadata || {};
  const dimensions = (result.basis || []).map(basis => basis.dimension || basis.expressions?.length || 1);
  state.runs.unshift({
    number: ++state.runCounter, jobId, request, result,
    name: result.name || request.name,
    representation: request.mode === "quantum" ? REPRESENTATION_LABELS[metadata.representation || "canonical"] : "Custom dictionaries",
    sites: dimensions.length, maxDimension: Math.max(1, ...dimensions),
    bond: Math.max(1, ...(result.bond_dimensions || [])), bondCap: request.options.max_bond,
    queries: metadata.query_counts?.training, error: modelValidation(result).relative_l2,
    sampling: samplingLabel(metadata),
    sweep: state.sweep ? { id: state.sweep.id, axis: state.sweep.axis, value: state.sweep.values[state.sweep.index] } : null,
  });
  renderRuns();
}

function recordFailedRun(request, message) {
  state.runs.unshift({
    number: ++state.runCounter, failed: true, message, request, name: request.name,
    representation: request.mode === "quantum" ? REPRESENTATION_LABELS[request.representation || "canonical"] : "Custom dictionaries",
    sweep: state.sweep ? { id: state.sweep.id, axis: state.sweep.axis, value: state.sweep.values[state.sweep.index] } : null,
  });
  renderRuns();
}

// Error and query count along the latest sweep, on the swept values in the order given.
function renderSweepCharts() {
  const root = $("sweep-charts");
  const latest = state.runs.find(run => run.sweep)?.sweep;
  const runs = latest ? state.runs.filter(run => run.sweep?.id === latest.id).reverse() : [];
  const done = runs.filter(run => !run.failed);
  root.hidden = done.length < 2;
  root.replaceChildren();
  if (root.hidden) return;
  const labels = runs.map(run => latest.axis === "representation" ? REPRESENTATION_LABELS[run.sweep.value] : Number(run.sweep.value).toLocaleString());
  root.append(element("h3", "sweep-title", `Sweep over ${SWEEP_AXES[latest.axis].label.toLowerCase()}`));
  const grid = element("div", "sweep-grid");
  grid.append(
    sweepChart("Relative L² error", labels, runs.map(run => run.failed || !(run.error > 0) ? null : run.error), true),
    sweepChart("Training queries", labels, runs.map(run => run.failed || run.queries === undefined ? null : run.queries), false),
  );
  root.append(grid);
  const failed = runs.length - done.length;
  if (failed) root.append(element("p", "field-help", `${failed} ${failed === 1 ? "run" : "runs"} did not finish and ${failed === 1 ? "is" : "are"} left out of the charts.`));
}

function sweepChart(title, labels, values, logarithmic) {
  const figure = element("figure", "sweep-chart");
  figure.append(element("figcaption", null, logarithmic ? `${title} · log scale` : title));
  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg");
  const width = 360, height = 190, left = 62, right = 14, top = 12, bottom = 34;
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.setAttribute("role", "img");
  const shown = values.map((value, index) => ({ value, index })).filter(point => point.value !== null);
  svg.setAttribute("aria-label", `${title}: ${shown.map(point => `${labels[point.index]}: ${number(point.value, 3)}`).join("; ")}`);
  const scale = value => logarithmic ? Math.log10(value) : value;
  let low = Math.min(...shown.map(point => scale(point.value))), high = Math.max(...shown.map(point => scale(point.value)));
  if (logarithmic) { low = Math.floor(low); high = Math.ceil(high); if (high === low) high += 1; }
  else { low = 0; if (high <= 0) high = 1; }
  const add = (tag, attributes, text) => {
    const node = document.createElementNS(ns, tag);
    for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
    if (text !== undefined) node.textContent = text;
    svg.append(node);
    return node;
  };
  const xp = index => labels.length === 1 ? left + (width - left - right) / 2 : left + index * (width - left - right) / (labels.length - 1);
  const yp = value => height - bottom - (scale(value) - low) / (high - low) * (height - bottom - top);
  const text = { fill: "#5f685d", "font-size": 11, "font-family": "monospace" };
  const ticks = logarithmic
    ? Array.from({ length: high - low + 1 }, (_, i) => low + i).filter((_, i, all) => i % Math.ceil(all.length / 5) === 0)
    : [0, high / 2, high];
  for (const tick of ticks) {
    const y = height - bottom - (tick - low) / (high - low) * (height - bottom - top);
    add("line", { x1: left, x2: width - right, y1: y, y2: y, stroke: "#e7e9df", "stroke-width": 1 });
    add("text", { ...text, x: left - 8, y: y + 4, "text-anchor": "end" }, logarithmic ? `1e${tick}` : Math.round(tick).toLocaleString());
  }
  labels.forEach((label, index) => add("text", { ...text, x: xp(index), y: height - bottom + 18, "text-anchor": index === 0 && labels.length > 1 ? "start" : index === labels.length - 1 && labels.length > 1 ? "end" : "middle" }, label.length > 12 ? `${label.slice(0, 11)}…` : label));
  if (shown.length > 1) add("path", { d: shown.map((point, i) => `${i ? "L" : "M"}${xp(point.index).toFixed(1)},${yp(point.value).toFixed(1)}`).join(" "), fill: "none", stroke: "var(--accent)", "stroke-width": 2, "stroke-linejoin": "round" });
  for (const point of shown) {
    const dot = add("circle", { cx: xp(point.index), cy: yp(point.value), r: 4, fill: "var(--surface)", stroke: "var(--accent)", "stroke-width": 2 });
    add("title", {}, `${labels[point.index]}: ${number(point.value, 4)}`);
    dot.append(svg.lastChild);
  }
  figure.append(svg);
  return figure;
}

function renderRuns() {
  const root = $("runs-container");
  if (!root) return;
  renderSweepCharts();
  if (!state.runs.length) { root.replaceChildren(element("p", "muted", "No runs yet")); return; }
  const table = element("table", "site-table runs-table");
  const head = element("thead"), heading = element("tr");
  for (const label of ["Run", "Representation", "Sampling", "Dictionaries", "Max dimension", "Max bond", "Training queries", "Relative L² error"]) {
    const cell = element("th", null, label); cell.scope = "col"; heading.append(cell);
  }
  head.append(heading);
  const body = element("tbody");
  for (const run of state.runs) {
    const row = element("tr", !run.failed && run.jobId === state.jobId && state.result === run.result ? "current" : "");
    const label = element("th"); label.scope = "row";
    if (run.failed) {
      label.textContent = `#${run.number} · ${run.name}`;
      const reason = element("td", "run-failure", `Did not finish: ${run.message}`);
      reason.colSpan = 6;
      row.append(label, element("td", null, run.representation), reason);
      body.append(row);
      continue;
    }
    const view = element("button", "text-button", `#${run.number} · ${run.name}`);
    view.type = "button"; view.disabled = state.busy;
    view.setAttribute("aria-label", `Show run ${run.number}, ${run.name}`);
    view.addEventListener("click", () => showRun(run));
    label.append(view);
    row.append(label, element("td", null, run.representation), element("td", null, run.sampling),
      element("td", "numeric", String(run.sites)), element("td", "numeric", String(run.maxDimension)),
      element("td", "numeric", `${run.bond} of ${run.bondCap}`),
      element("td", "numeric", run.queries === undefined ? "—" : Number(run.queries).toLocaleString()),
      element("td", "numeric", number(run.error)));
    body.append(row);
  }
  table.append(head, body);
  root.replaceChildren(table, element("p", "field-help", "Runs stay available until the local server restarts. Select one to show it in the other views."));
}

function showRun(run) {
  if (state.busy) return;
  state.jobId = run.jobId;
  state.request = run.request;
  state.result = run.result;
  renderResult(run.result);
  setStatus("complete", "Model ready", `Run #${run.number}`);
  renderRuns();
  setTab("prediction");
}

function setTab(name, focus = false) {
  for (const tab of document.querySelectorAll("[data-tab]")) {
    const active = tab.dataset.tab === name;
    tab.setAttribute("aria-selected", String(active)); tab.tabIndex = active ? 0 : -1;
    $(`panel-${tab.dataset.tab}`).hidden = !active;
    if (active && focus) tab.focus();
  }
}

async function initialize() {
  state.circuitEditor = new FTCICircuitEditor($("quantum-editor"), {onChange: spec => state.circuitSource?.handleGraphicChange(spec)});
  state.circuitSource = new FTCICircuitSource({
    textarea: $("circuit-json"), formatSelect: $("circuit-format"),
    status: $("circuit-source-status"), error: $("circuit-source-error"),
    editor: state.circuitEditor, onChange: circuitSourceChanged,
  });
  $("learning-form").addEventListener("submit", startLearning);
  $("predict-form").addEventListener("submit", predict);
  $("basis-index-select").addEventListener("change", renderCoreTable);
  $("basis-view").addEventListener("change", () => setView($("basis-view").value));
  $("sweep-axis").addEventListener("change", renderSweepControls);
  $("slice-x").addEventListener("change", () => { syncSliceControls(); scheduleSlice(); });
  $("slice-y").addEventListener("change", () => {
    // A plane costs many more target queries than a line, so comparing is opt-in there.
    $("slice-target").checked = $("slice-y").value === "";
    syncSliceControls(); scheduleSlice();
  });
  $("slice-target").addEventListener("change", scheduleSlice);
  $("slice-fixed").addEventListener("input", scheduleSlice);
  $("add-input").addEventListener("click", () => {
    try { const domain = readDomain(); if (domain.length >= 12) throw new Error("The web workspace supports up to 12 inputs."); domain.push([-Math.PI, Math.PI]); renderDomain(domain); }
    catch (error) { showError("form-error", error.message); }
  });
  $("add-basis").addEventListener("click", () => {
    const index = Math.min($("basis-editors").children.length, $("domain-inputs").children.length - 1);
    addBasisEditor({ name: `Input x${index}`, expressions: ["1", `cos(x${index})`, `sin(x${index})`] });
  });
  for (const button of document.querySelectorAll(".mode-button[data-mode]")) button.addEventListener("click", () => switchProblemMode(button.dataset.mode));
  for (const button of document.querySelectorAll(".representation-button")) button.addEventListener("click", () => chooseRepresentation(button.dataset.representation));
  for (const id of ["max-bond", "oversampling"]) $(id).addEventListener("input", renderRepresentationSummary);
  for (const type of ["input", "change"]) $("learning-form").addEventListener(type, scheduleDesign);
  $("example-select").addEventListener("change", () => {
    const list = state.examples.filter((item) => item.mode === state.mode);
    const example = list.find((item, i) => (item.id || String(i)) === $("example-select").value);
    if (example) loadExample(example);
  });
  $("upload-circuit-button").addEventListener("click", () => $("circuit-file").click());
  $("circuit-file").addEventListener("change", async () => {
    const file = $("circuit-file").files?.[0];
    if (!file || state.busy) return;
    try {
      await state.circuitSource.loadFile(file);
      if (state.circuitSource.loadedFileName === file.name) {
        $("circuit-file-name").textContent = file.name;
        $("circuit-file-name").hidden = false;
      }
    } catch (error) { showError("form-error", error.message); }
    finally { $("circuit-file").value = ""; }
  });
  $("domain-inputs").addEventListener("change", syncCircuitDomain);
  for (const tab of document.querySelectorAll("[data-tab]")) {
    tab.addEventListener("click", () => setTab(tab.dataset.tab));
    tab.addEventListener("keydown", (event) => {
      const tabs = ["prediction", "tensors", "details", "runs"];
      const current = tabs.indexOf(tab.dataset.tab);
      let next;
      if (event.key === "ArrowRight") next = (current + 1) % tabs.length;
      if (event.key === "ArrowLeft") next = (current + tabs.length - 1) % tabs.length;
      if (event.key === "Home") next = 0;
      if (event.key === "End") next = tabs.length - 1;
      if (next !== undefined) { event.preventDefault(); setTab(tabs[next], true); }
    });
  }
  $("cancel-button").addEventListener("click", async () => {
    if (!state.jobId) return;
    $("cancel-button").disabled = true;
    try {
      await api(`/api/jobs/${encodeURIComponent(state.jobId)}/cancel`, { method: "POST", body: "{}" });
      setStatus("running", "Stopping…", "");
    } catch (error) { showError("form-error", error.message); $("cancel-button").disabled = false; }
  });
  renderDomain([[-Math.PI, Math.PI], [-Math.PI, Math.PI]]);
  renderBasisEditors([{ name: "Input x0", expressions: ["1", "cos(x0)", "sin(x0)"] }, { name: "Input x1", expressions: ["1", "cos(x1)", "sin(x1)"] }]);
  $("learn-button").disabled = true;
  const results = await Promise.allSettled([api("/api/health"), api("/api/examples")]);
  if (results[0].status === "fulfilled") {
    const health = results[0].value;
    state.engineAvailable = health.engine_available !== false;
    $("engine-state").className = `engine-state ${state.engineAvailable ? "ready" : "error"}`;
    $("engine-label").textContent = state.engineAvailable ? "Local engine ready" : "Julia engine unavailable";
    if (!state.engineAvailable) showError("form-error", health.message || "The Julia engine is not available. Check the setup instructions before learning.");
  } else {
    $("engine-state").className = "engine-state error";
    $("engine-label").textContent = "Engine disconnected";
    showError("form-error", "Could not connect to the local learning engine. Start FTCI Studio with its Python server, then reload this page.");
  }
  refreshLearnButton();
  if (results[1].status === "fulfilled") {
    state.examples = Array.isArray(results[1].value) ? results[1].value : results[1].value.examples || [];
    setMode("function");
  } else {
    renderExampleSelect();
    showError("form-error", "Examples unavailable");
  }
}

initialize().catch((error) => {
  showError("form-error", error.message);
  setStatus("failed", "Workspace could not initialize", "Check the local server and reload this page.");
});

(function (global) {
  "use strict";

  const clone = value => value == null ? value : JSON.parse(JSON.stringify(value));
  const FORMATS = new Set(["json", "python", "qiskit"]);
  const MAX_FILE_BYTES = 96 * 1024;
  const MAX_REQUEST_BYTES = 1024 * 1024;

  async function requestJSON(path, body) {
    const response = await fetch(path, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body), cache: "no-store",
    });
    let result;
    try { result = await response.json(); }
    catch { throw new Error(`The circuit service returned an unreadable response (${response.status}).`); }
    if (!response.ok) throw new Error(result.error || "The circuit source could not be processed.");
    return result;
  }

  /** Keep editable source authoritative, with a synchronized optional diagram.
   *
   * All public mutations return Promise<boolean>. A false result means the
   * current source was retained and cannot yet be used for learning. Pending
   * and invalid source disables graph edits, so a graphic action cannot erase
   * a partially typed source document. Request revisions suppress late results.
   */
  class FTCICircuitSource {
    constructor({ textarea, formatSelect, status, error, editor, onChange = () => {},
                  request = requestJSON, debounceMs = 250 }) {
      if (!textarea || !formatSelect || !editor) throw new Error("Circuit source needs a text editor, format selector, and diagram.");
      this.textarea = textarea;
      this.formatSelect = formatSelect;
      this.statusElement = status;
      this.errorElement = error;
      this.editor = editor;
      this.onChange = onChange;
      this.request = request;
      this.debounceMs = debounceMs;
      this.format = FORMATS.has(formatSelect.value) ? formatSelect.value : "json";
      this._selectFormat();
      this.circuit = null;
      this.inspection = null;
      this.sourceInfo = null;
      this.valid = false;
      this.pending = false;
      this.error = "";
      this.disabled = false;
      this.origin = "source";
      this.revision = 0;
      this.timer = null;
      this.activeTask = null;
      this.validatedSource = null;
      this.validatedFormat = null;
      this.defaultDomain = null;
      this.loadedFileName = null;
      this.textarea.addEventListener("input", () => {
        if (this.disabled) return;
        const source = this.textarea.value, format = this.format;
        const revision = this._begin("source");
        this.timer = setTimeout(() => {
          this.timer = null;
          this.activeTask = this._parse(revision, format, source, "source");
        }, this.debounceMs);
      });
      this.formatSelect.addEventListener("change", () => {
        const requested = this.formatSelect.value;
        this._selectFormat();
        this.setFormat(requested);
      });
      // Construction does not emit: the caller can finish wiring the form.
      this._render();
    }

    getSource() { return { format: this.format, source: this.textarea.value }; }

    getCircuit() {
      if (this.pending) throw new Error("The circuit code is being checked. Wait a moment before learning.");
      if (!this.valid || !this.circuit || this.textarea.value !== this.validatedSource || this.format !== this.validatedFormat) {
        throw new Error(this.error || "Correct the circuit code before learning.");
      }
      // Unapplied helper drafts do not change the authoritative circuit code.
      return clone(this.circuit);
    }

    setDisabled(disabled) { this.disabled = Boolean(disabled); this._render(); }

    async setCircuit(spec, { format = "json" } = {}) {
      this._checkFormat(format);
      this.defaultDomain = clone(spec.domain || this.defaultDomain);
      if (format === "json") {
        this.format = format;
        this._selectFormat();
        this.textarea.value = JSON.stringify(spec, null, 2);
        return this.refresh("example");
      }
      return this._serialize(spec, { format, origin: "example", previousSource: null, updateDiagram: true });
    }

    loadFile(file) {
      const match = file && typeof file.name === "string" && /\.(json|py)$/i.exec(file.name);
      if (!match || typeof file.text !== "function") {
        this._nonDestructiveError("Choose a .json or .py circuit file.");
        return Promise.resolve(false);
      }
      if (!Number.isInteger(file.size) || file.size < 0 || file.size > MAX_FILE_BYTES) {
        this._nonDestructiveError("Circuit files must be no larger than 96 KiB.");
        return Promise.resolve(false);
      }
      if (this.disabled) return Promise.resolve(false);
      const format = match[1].toLowerCase() === "json" ? "json" : "python";
      const before = { format: this.format, source: this.textarea.value, pending: this.pending };
      const revision = this._begin("file");
      this.activeTask = (async () => {
        let source;
        try {
          source = await file.text();
          if (typeof source !== "string") throw new Error("The selected file could not be read as text.");
          const encoder = new TextEncoder();
          if (encoder.encode(source).length > MAX_FILE_BYTES) throw new Error("Circuit files must be no larger than 96 KiB.");
          const body = { format, source, default_domain: clone(this.defaultDomain) };
          if (encoder.encode(JSON.stringify(body)).length > MAX_REQUEST_BYTES) throw new Error("This file exceeds the circuit request size after text encoding. Use a smaller file.");
        } catch (error) {
          if (!this._current(revision, before.format, before.source)) return false;
          const message = `File was not loaded: ${error.message || error}`;
          if (before.pending) {
            // Loading invalidates old responses immediately. If reading fails,
            // resume validation of the retained draft instead of stranding it.
            const recovery = this.refresh("source");
            const recoveryRevision = this.revision;
            await recovery;
            if (this._current(recoveryRevision, before.format, before.source)) this._nonDestructiveError(message + (this.valid || !this.error ? "" : ` ${this.error}`));
          } else this._fail(new Error(message), "file");
          return false;
        }
        if (!this._current(revision, before.format, before.source)) return false;
        // A readable but invalid file remains visible as its exact source,
        // with parser errors. It never silently falls back to the old model.
        this.format = format;
        this._selectFormat();
        this.textarea.value = source;
        this.loadedFileName = file.name;
        return this._parse(revision, format, source, "file");
      })();
      return this.activeTask;
    }

    refresh(origin = "source") {
      const revision = this._begin(origin);
      this.activeTask = this._parse(revision, this.format, this.textarea.value, origin);
      return this.activeTask;
    }

    async setFormat(format) {
      try { this._checkFormat(format); }
      catch (error) { this._nonDestructiveError(error.message); return false; }
      this._selectFormat();
      if (format === this.format || (format === "python" && this.format === "qiskit")) return this._ensureCurrent();
      if (this.disabled) return false;
      if (!await this._ensureCurrent()) return false;
      try { this.editor.getCircuit(); }
      catch (error) { this._nonDestructiveError(error.message); return false; }
      return this._serialize(this.circuit, { format, origin: "format", previousSource: null, updateDiagram: true });
    }

    async setDomain(domain) {
      if (this.disabled) return false;
      if (!await this._ensureCurrent()) return false;
      try { this.editor.getCircuit(); }
      catch (error) { this._nonDestructiveError(error.message); return false; }
      const circuit = { ...clone(this.circuit), domain: clone(domain) };
      if (JSON.stringify(circuit.domain) === JSON.stringify(this.circuit.domain)) return true;
      return this._serialize(circuit, { format: this.format, origin: "domain", updateDiagram: true });
    }

    async handleGraphicChange(circuit) {
      // The diagram commits before issuing its callback. If the code cannot
      // accept that edit, put the last accepted code circuit back in the guide.
      if (this.pending) return false;
      if (this.disabled || !this.valid || this.textarea.value !== this.validatedSource) {
        this._restoreDiagram();
        this._nonDestructiveError("Finish correcting the circuit code before editing its diagram.");
        return false;
      }
      return this._serialize(circuit, { format: this.format, origin: "graphic", updateDiagram: false });
    }

    _checkFormat(format) {
      if (!FORMATS.has(format)) throw new Error("Choose JSON or Python code.");
    }

    _selectFormat() {
      // Existing SDK callers may still identify a source as Qiskit. The
      // language menu groups all supported package sources under Python.
      this.formatSelect.value = this.format === "qiskit" ? "python" : this.format;
    }

    async _ensureCurrent() {
      if (this.valid && !this.pending && this.textarea.value === this.validatedSource && this.format === this.validatedFormat) return true;
      // A domain/format action can follow a just-committed graphic edit before
      // serialization finishes. Await that edit rather than reparsing the old
      // source and accidentally dropping the newly committed gate.
      if (this.pending && this.activeTask && !this.timer) {
        await this.activeTask;
        if (this.valid && !this.pending && this.textarea.value === this.validatedSource && this.format === this.validatedFormat) return true;
        if (!this.pending && this.error) return false;
      }
      return this.refresh("source");
    }

    _begin(origin) {
      clearTimeout(this.timer);
      this.timer = null;
      this.activeTask = null;
      const revision = ++this.revision;
      this.origin = origin;
      this.valid = false;
      this.pending = true;
      this.error = "";
      this._render();
      this._emit();
      return revision;
    }

    async _parse(revision, format, source, origin) {
      try {
        const result = await this.request("/api/circuit/source", {
          format, source, default_domain: clone(this.defaultDomain),
        });
        if (!this._current(revision, format, source)) return false;
        this._accept(result, { format, source, origin, replaceSource: false, updateDiagram: true });
        return true;
      } catch (error) {
        if (!this._current(revision, format, source)) return false;
        this._fail(error, origin);
        return false;
      }
    }

    _serialize(circuit, { format, origin, previousSource = undefined, updateDiagram }) {
      const baseFormat = this.format, baseSource = this.textarea.value;
      const previous = previousSource === undefined && format === baseFormat ? baseSource : previousSource;
      const revision = this._begin(origin);
      this.activeTask = (async () => {
        try {
          const result = await this.request("/api/circuit/serialize", {
            format, circuit: clone(circuit), previous_source: previous == null ? null : previous,
          });
          if (!this._current(revision, baseFormat, baseSource)) return false;
          if (typeof result.source !== "string") throw new Error("The circuit service did not return synchronized source text.");
          this._accept(result, { format, source: result.source, origin, replaceSource: true, updateDiagram });
          return true;
        } catch (error) {
          if (!this._current(revision, baseFormat, baseSource)) return false;
          if (origin === "graphic") this._restoreDiagram();
          this._fail(error, origin);
          return false;
        }
      })();
      return this.activeTask;
    }

    _current(revision, format, source) {
      return revision === this.revision && format === this.format && source === this.textarea.value;
    }

    _accept(result, { format, source, origin, replaceSource, updateDiagram }) {
      if (!result.circuit || !Array.isArray(result.circuit.gates)) throw new Error("The circuit service returned no circuit definition.");
      this.editor.setInspection(null);
      if (updateDiagram) this.editor.setCircuit(clone(result.circuit), { resetHistory: origin !== "domain" });
      else this.editor.setDomain(clone(result.circuit.domain));
      this.editor.setInspection(clone(result.inspection || null));
      this.circuit = clone(result.circuit);
      this.inspection = clone(result.inspection || null);
      this.sourceInfo = clone(result.source_info || null);
      this.defaultDomain = clone(this.circuit.domain);
      this.format = format;
      this._selectFormat();
      if (replaceSource) this.textarea.value = source;
      this.validatedSource = source;
      this.validatedFormat = format;
      this.origin = origin;
      this.valid = true;
      this.pending = false;
      this.error = "";
      this._render();
      this._emit();
    }

    _restoreDiagram() {
      if (!this.circuit) return;
      this.editor.setInspection(null);
      this.editor.setCircuit(clone(this.circuit));
      this.editor.setInspection(clone(this.inspection));
    }

    _fail(error, origin) {
      // Failed visual/domain/format transformations have not changed source.
      // Keep its previously validated definition usable after rolling back the
      // guide, while an actual invalid source draft remains untrainable.
      this.valid = origin !== "source" && origin !== "example" && this.circuit != null
        && this.format === this.validatedFormat && this.textarea.value === this.validatedSource;
      this.pending = false;
      const prefix = { graphic: "The graphic edit was not applied. ", domain: "Input bounds were not changed. ", format: "The source format was not changed. " }[origin] || "";
      this.error = prefix + String(error.message || error);
      this.origin = origin;
      this._selectFormat();
      this._render();
      this._emit();
    }

    _nonDestructiveError(message) {
      this.error = message;
      this._render();
      this._emit();
    }

    _render() {
      const rewriting = this.pending && this.origin !== "source";
      this.textarea.disabled = this.disabled || rewriting;
      this.formatSelect.disabled = this.disabled || rewriting;
      this.editor.setDisabled(this.disabled || this.pending || !this.valid);
      if (this.statusElement) {
        this.statusElement.className = `circuit-source-status ${this.pending ? "is-pending" : this.valid ? "is-valid" : "is-invalid"}`;
        const count = this.inspection && this.inspection.encoding_occurrences;
        const info = this.sourceInfo;
        const label = this.format === "json" ? "Synchronized"
          : info && typeof info.label === "string" && info.label ? info.label
            : this.format === "qiskit" ? "Qiskit" : "Python";
        const notice = this.format !== "json" && info && info.kind === "portable"
          && typeof info.notice === "string" && info.notice ? ` · ${info.notice}` : "";
        this.statusElement.textContent = this.pending
          ? "Checking…"
          : this.valid
            ? `${label}${count == null ? "" : ` · ${count} encoding${count === 1 ? "" : "s"}`}${notice}`
            : this.circuit
              ? "Code error · last valid preview"
              : "Enter circuit code";
      }
      if (this.errorElement) {
        this.errorElement.textContent = this.error;
        this.errorElement.hidden = !this.error;
      }
    }

    _emit() {
      this.onChange({ circuit: clone(this.circuit), inspection: clone(this.inspection),
        valid: this.valid, pending: this.pending, error: this.error,
        format: this.format, source: this.textarea.value, origin: this.origin,
        sourceInfo: clone(this.sourceInfo),
        fileName: this.loadedFileName });
    }
  }

  if (typeof module !== "undefined" && module.exports) module.exports = { FTCICircuitSource };
  global.FTCICircuitSource = FTCICircuitSource;
})(typeof window !== "undefined" ? window : globalThis);

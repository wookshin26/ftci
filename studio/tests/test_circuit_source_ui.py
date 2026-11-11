"""Behavioral synchronization tests for the browser's authoritative source."""

from pathlib import Path
import shutil
import subprocess
import unittest


HARNESS = r"""
const assert = require('node:assert/strict');
const {FTCICircuitSource} = require('./ftci/web/circuit-source.js');
const clone = value => JSON.parse(JSON.stringify(value));
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
class Field {
  constructor(value='') { this.value=value; this.disabled=false; this.hidden=false; this.textContent=''; this.className=''; this.events={}; }
  addEventListener(name, callback) { this.events[name]=callback; }
  input(value) { this.value=value; this.events.input(); }
}
class Guide {
  constructor() { this.circuit=null; this.dirty=false; this.calls=[]; }
  setDisabled(value) { this.disabled=value; }
  setInspection(value) { this.inspection=value; }
  setCircuit(value, options) { this.circuit=clone(value); this.calls.push(options); this.dirty=false; }
  setDomain(value) { this.circuit.domain=clone(value); }
  getCircuit() { if(this.dirty) throw new Error('Apply the gate draft first.'); return clone(this.circuit); }
}
const original = () => ({n_qubits:1,observable:'Z',domain:[[-1,1]],gates:[{gate:'ry',qubits:[0],angle:'x0'}]});
function result(circuit, source, sourceInfo) { return {circuit:clone(circuit),source,source_info:sourceInfo,inspection:{encoding_occurrences:circuit.gates.filter(g=>String(g.angle).includes('x')).length}}; }
function parse(format, source) { return JSON.parse(format==='json' ? source : source.slice('QISKIT\n'.length)); }
function backend(path, body) {
  const info=body.format==='json'?{label:'JSON',kind:'json',notice:''}:{label:'Qiskit',kind:'qiskit',notice:''};
  if(path.endsWith('/source')) return Promise.resolve(result(parse(body.format,body.source),undefined,info));
  const source=(body.format==='json'?'':'QISKIT\n')+JSON.stringify(body.circuit,null,2);
  return Promise.resolve(result(body.circuit,source,info));
}
function setup(request=backend) {
  const fields={textarea:new Field(),formatSelect:new Field('json'),status:new Field(),error:new Field(),editor:new Guide()};
  const events=[];
  const source=new FTCICircuitSource({...fields,request,debounceMs:1,onChange:event=>events.push(event)});
  return {...fields,source,events};
}
async function run() {
"""


@unittest.skipUnless(shutil.which("node"), "Node is not installed")
class CircuitSourceSynchronizationTests(unittest.TestCase):
    def node(self, body):
        script = HARNESS + body + "\n}\nrun().catch(error=>{console.error(error);process.exitCode=1;});"
        project = Path(__file__).resolve().parents[1]
        completed = subprocess.run([shutil.which("node"), "-e", script], cwd=project,
                                   text=True, capture_output=True, timeout=15)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)

    def test_typing_updates_preview_without_reformatting_the_code(self):
        self.node(r"""
const {source,textarea,editor,events}=setup();
assert.equal(events.length,0); // Construction never races the form setup.
await source.setCircuit(original());
const next=original(); next.gates[0].angle='2*x0**2';
const raw='  '+JSON.stringify(next,null,4)+'\n';
textarea.input(raw);
assert.equal(editor.disabled,true);
assert.throws(()=>source.getCircuit(),/being checked/);
await sleep(8); await source.activeTask;
assert.equal(textarea.value,raw);
assert.equal(editor.circuit.gates[0].angle,'2*x0**2');
assert.equal(source.getCircuit().gates[0].angle,'2*x0**2');
assert.equal(editor.disabled,false);
editor.dirty=true;
assert.equal(source.getCircuit().gates[0].angle,'2*x0**2'); // Helper drafts do not override valid code.
""")

    def test_graphic_edits_write_source_and_preserve_graph_history(self):
        self.node(r"""
const {source,textarea,editor}=setup();
await source.setCircuit(original());
const next=original();next.gates.push({gate:'rz',qubits:[0],angle:'x0**2'});
editor.circuit=clone(next);
const calls=editor.calls.length;
assert.equal(await source.handleGraphicChange(next),true);
assert.equal(JSON.parse(textarea.value).gates.length,2);
assert.equal(source.getCircuit().gates.length,2);
assert.equal(editor.calls.length,calls); // No resetting committed graph undo.
editor.circuit.gates=[];
assert.equal(source.getCircuit().gates.length,2); // Code is the learned object.
""")

    def test_invalid_draft_is_retained_and_cannot_switch_or_learn(self):
        self.node(r"""
const {source,textarea,formatSelect,editor}=setup();
await source.setCircuit(original());
const raw='{"n_qubits":';
textarea.value=raw;
assert.equal(await source.refresh(),false);
assert.equal(textarea.value,raw);
assert.equal(editor.circuit.gates[0].angle,'x0');
assert.equal(editor.disabled,true);
assert.throws(()=>source.getCircuit());
assert.equal(await source.setFormat('python'),false);
assert.equal(source.format,'json');assert.equal(formatSelect.value,'json');
assert.equal(textarea.value,raw);
source.setDisabled(true);source.setDisabled(false);
assert.equal(editor.disabled,true);assert.equal(textarea.disabled,false);
""")

    def test_late_parse_response_cannot_restore_older_code(self):
        self.node(r"""
const pending=[];
const {source,textarea,editor}=setup((path,body)=>new Promise(resolve=>pending.push({body,resolve})));
textarea.value=JSON.stringify(original());
const old=source.refresh();
const next=original();next.gates[0].angle='3*x0';
textarea.value=JSON.stringify(next);
const fresh=source.refresh();
pending[1].resolve(result(next));await fresh;
pending[0].resolve(result(original()));await old;
assert.equal(source.valid,true);assert.equal(source.getCircuit().gates[0].angle,'3*x0');
assert.equal(editor.circuit.gates[0].angle,'3*x0');
""")

    def test_domain_edit_waits_for_graphic_serialization(self):
        self.node(r"""
let deferred=null,first=true;
const {source,editor,textarea}=setup((path,body)=>{
  if(path.endsWith('/serialize') && first) {first=false;return new Promise(resolve=>deferred={body,resolve});}
  return backend(path,body);
});
await source.setCircuit(original());
const next=original();next.gates[0].angle='4*x0';editor.circuit=clone(next);
const graphic=source.handleGraphicChange(next);
const bounds=source.setDomain([[-2,2]]);
await sleep(1);
assert.equal(deferred.body.circuit.gates[0].angle,'4*x0');
deferred.resolve(result(next,JSON.stringify(next)));
assert.equal(await graphic,true);assert.equal(await bounds,true);
const learned=source.getCircuit();
assert.equal(learned.gates[0].angle,'4*x0');assert.deepEqual(learned.domain,[[-2,2]]);
assert.deepEqual(JSON.parse(textarea.value),learned);
""")

    def test_format_switch_translates_current_model_and_roundtrips(self):
        self.node(r"""
const {source,textarea,editor,formatSelect}=setup();
await source.setCircuit(original());
assert.equal(await source.setFormat('python'),true);
assert.equal(formatSelect.value,'python');
const next=original();next.gates[0].angle='sin(x0)';
textarea.value='QISKIT\n'+JSON.stringify(next);
await source.refresh();
assert.equal(editor.circuit.gates[0].angle,'sin(x0)');
assert.equal(await source.setFormat('json'),true);
assert.equal(JSON.parse(textarea.value).gates[0].angle,'sin(x0)');
assert.equal(source.format,'json');
""")

    def test_failed_graphic_rewrite_keeps_the_valid_source(self):
        self.node(r"""
const {source,textarea,editor,error}=setup((path,body)=>path.endsWith('/serialize')?Promise.reject(new Error('Cannot patch this source.')):backend(path,body));
await source.setCircuit(original());
const before=textarea.value,next=original();next.gates[0].angle='9*x0';editor.circuit=clone(next);
assert.equal(await source.handleGraphicChange(next),false);
assert.equal(textarea.value,before);assert.equal(editor.circuit.gates[0].angle,'x0');
assert.equal(source.getCircuit().gates[0].angle,'x0');
assert.match(error.textContent,/not applied/);
""")

    def test_file_imports_keep_exact_json_or_python_source(self):
        self.node(r"""
const seen=[];
const {source,textarea,formatSelect,editor}=setup((path,body)=>{
  seen.push(body);
  if(path.endsWith('/source')&&body.format==='python') return Promise.resolve(result(original(),undefined,{label:'Qiskit',kind:'qiskit',notice:''}));
  return backend(path,body);
});
await source.setCircuit(original());
const next=original();next.gates[0].angle='2*x0';
const json='  '+JSON.stringify(next,null,3)+'\n';
assert.equal(await source.loadFile({name:'Circuit.JSON',size:Buffer.byteLength(json),text:async()=>json}),true);
assert.equal(textarea.value,json);assert.equal(source.getSource().format,'json');
assert.equal(source.loadedFileName,'Circuit.JSON');
assert.equal(editor.circuit.gates[0].angle,'2*x0');
const python='# preserve this comment\nfrom qiskit import QuantumCircuit\nqc = QuantumCircuit(1)\nqc.ry(x0, 0)\n';
assert.equal(await source.loadFile({name:'circuit.py',size:Buffer.byteLength(python),text:async()=>python}),true);
assert.equal(textarea.value,python);assert.equal(formatSelect.value,'python');
assert.equal(source.loadedFileName,'circuit.py');
assert.equal(seen.at(-1).source,python);assert.equal(seen.at(-1).format,'python');
assert.equal(source.getCircuit().gates[0].angle,'x0');
""")

    def test_damaged_file_stays_visible_and_cannot_train(self):
        self.node(r"""
const {source,textarea,editor}=setup();await source.setCircuit(original());
for(const [name,raw,format] of [['broken.json','{"gates":','json'],['broken.py','not valid circuit code','python']]) {
  assert.equal(await source.loadFile({name,size:Buffer.byteLength(raw),text:async()=>raw}),false);
  assert.equal(textarea.value,raw);assert.equal(source.getSource().format,format);
  assert.equal(source.loadedFileName,name);
  assert.equal(editor.circuit.gates[0].angle,'x0');assert.equal(editor.disabled,true);
  assert.throws(()=>source.getCircuit());
}
""")

    def test_unknown_package_carrier_keeps_surrounding_python_during_edits(self):
        self.node(r"""
const prefix='# Keep my original program\nimport custom_quantum_package as quantum\ndef make_native():\n    return quantum.build()\n\nftci_circuit = ';
const suffix='\n# Native execution stays in my own environment.\n';
const info={label:'FTCI specification',kind:'portable',notice:'Learning source: ftci_circuit'};
let accepted=prefix+JSON.stringify(original())+suffix;
const {source,textarea,formatSelect,editor,status,events}=setup((path,body)=>{
  if(body.format==='json') return backend(path,body);
  assert.equal(body.format,'python');
  if(path.endsWith('/source')) {
    assert.equal(body.source,accepted);
    return Promise.resolve(result(JSON.parse(body.source.slice(prefix.length,-suffix.length)),undefined,info));
  }
  assert.equal(body.previous_source,accepted);
  accepted=prefix+JSON.stringify(body.circuit)+suffix;
  return Promise.resolve(result(body.circuit,accepted,info));
});
assert.equal(await source.loadFile({name:'my_package.py',size:Buffer.byteLength(accepted),text:async()=>accepted}),true);
assert.equal(textarea.value,accepted);assert.equal(formatSelect.value,'python');
assert.equal(status.textContent,'FTCI specification · 1 encoding · Learning source: ftci_circuit');
assert.equal(events.at(-1).sourceInfo.kind,'portable');
const edited=source.getCircuit();edited.gates[0].angle='3*x0**2';editor.circuit=clone(edited);
assert.equal(await source.handleGraphicChange(edited),true);
assert.equal(await source.setDomain([[-2,2]]),true);
assert.ok(textarea.value.startsWith(prefix));assert.ok(textarea.value.endsWith(suffix));
assert.equal(source.getCircuit().gates[0].angle,'3*x0**2');
assert.deepEqual(editor.circuit.domain,[[-2,2]]);
assert.equal(await source.setFormat('json'),true);
assert.equal(JSON.parse(textarea.value).gates[0].angle,'3*x0**2');
assert.equal(status.textContent,'Synchronized · 1 encoding');
assert.equal(source.sourceInfo.kind,'json');
""")

    def test_python_native_detection_status_comes_from_backend(self):
        self.node(r"""
for(const [kind,label] of [['qiskit','Qiskit'],['cirq','Cirq'],['pennylane','PennyLane']]) {
  const raw='# '+label+' circuit\nexact_source = "keep me"\n';
  const info={label,kind,notice:''};
  const {source,textarea,status,formatSelect,editor}=setup((path,body)=>{
    assert.equal(path,'/api/circuit/source');assert.equal(body.format,'python');assert.equal(body.source,raw);
    return Promise.resolve(result(original(),undefined,info));
  });
  assert.equal(await source.loadFile({name:kind+'.py',size:Buffer.byteLength(raw),text:async()=>raw}),true);
  assert.equal(textarea.value,raw);assert.equal(formatSelect.value,'python');
  assert.equal(status.textContent,label+' · 1 encoding');
  assert.equal(source.getSource().format,'python');assert.equal(editor.disabled,false);
}
""")

    def test_legacy_qiskit_format_uses_the_python_language_menu(self):
        self.node(r"""
const seen=[];
const {source,formatSelect,status}=setup((path,body)=>{seen.push(body);return backend(path,body);});
assert.equal(await source.setCircuit(original(),{format:'qiskit'}),true);
assert.equal(source.getSource().format,'qiskit');
assert.equal(formatSelect.value,'python');assert.equal(status.textContent,'Qiskit · 1 encoding');
const calls=seen.length;
assert.equal(await source.setFormat('python'),true);
assert.equal(seen.length,calls); // A menu alias never rewrites supported legacy source.
assert.equal(await source.setFormat('json'),true);
assert.equal(formatSelect.value,'json');
""")

    def test_language_menu_and_upload_copy_are_package_neutral(self):
        html = (Path(__file__).resolve().parents[1] / "ftci/web/index.html").read_text()
        self.assertIn('<option value="python">Python (auto)</option>', html)
        self.assertNotIn('<option value="qiskit">', html)
        self.assertIn('title="JSON or Python (.py)"', html)

    def test_unsupported_oversized_or_unreadable_file_preserves_draft(self):
        self.node(r"""
const {source,textarea,error,editor}=setup();await source.setCircuit(original());
const loaded=JSON.stringify(original());
await source.loadFile({name:'original.json',size:Buffer.byteLength(loaded),text:async()=>loaded});
const before=textarea.value,inspection=JSON.stringify(editor.inspection);let read=false;
assert.equal(await source.loadFile({name:'circuit.txt',size:10,text:async()=>{read=true;return 'bad';}}),false);
assert.equal(read,false);assert.equal(textarea.value,before);
assert.equal(source.loadedFileName,'original.json');
assert.equal(await source.loadFile({name:'large.py',size:96*1024+1,text:async()=>{read=true;return 'bad';}}),false);
assert.equal(read,false);assert.equal(textarea.value,before);
assert.equal(source.loadedFileName,'original.json');
assert.equal(await source.loadFile({name:'lost.json',size:10,text:async()=>{throw new Error('File is unavailable.');}}),false);
assert.equal(textarea.value,before);assert.match(error.textContent,/not loaded/);
assert.equal(source.loadedFileName,'original.json');assert.equal(source.valid,true);
assert.equal(editor.disabled,false);assert.equal(JSON.stringify(editor.inspection),inspection);
assert.equal(source.getCircuit().gates[0].angle,'x0');
const escaped='"'.repeat(70000);
assert.equal(await source.loadFile({name:'escaped.json',size:70000,text:async()=>escaped}),false);
assert.equal(textarea.value,escaped);assert.equal(source.loadedFileName,'escaped.json');
assert.doesNotMatch(error.textContent,/request size/); // The escaped request fits the 1 MiB API cap.
""")

    def test_file_load_invalidates_older_parse_before_read_finishes(self):
        self.node(r"""
let oldRequest=null,resolveFile=null,delay=false;
const {source,textarea,editor}=setup((path,body)=>{
  if(delay&&path.endsWith('/source')&&body.source.includes('2*x0')) return new Promise(resolve=>oldRequest={body,resolve});
  return backend(path,body);
});
await source.setCircuit(original());delay=true;
const older=original();older.gates[0].angle='2*x0';textarea.value=JSON.stringify(older);
const stale=source.refresh();
const newer=original();newer.gates[0].angle='3*x0';const raw=JSON.stringify(newer);
const loaded=source.loadFile({name:'new.json',size:Buffer.byteLength(raw),text:()=>new Promise(resolve=>resolveFile=resolve)});
oldRequest.resolve(result(older));await stale;
assert.equal(editor.circuit.gates[0].angle,'x0');
resolveFile(raw);assert.equal(await loaded,true);
assert.equal(textarea.value,raw);assert.equal(editor.circuit.gates[0].angle,'3*x0');
""")

    def test_failed_file_read_resumes_validation_of_retained_pending_draft(self):
        self.node(r"""
let staleResolve=null,first=false;
const {source,textarea,error}=setup((path,body)=>{
  if(first&&path.endsWith('/source')) {first=false;return new Promise(resolve=>staleResolve=resolve);}
  return backend(path,body);
});
await source.setCircuit(original());
const next=original();next.gates[0].angle='5*x0';const raw=JSON.stringify(next);textarea.value=raw;
first=true;const stale=source.refresh();
assert.equal(await source.loadFile({name:'unreadable.py',size:10,text:async()=>{throw new Error('Read failed.');}}),false);
assert.equal(textarea.value,raw);assert.equal(source.getCircuit().gates[0].angle,'5*x0');
assert.match(error.textContent,/not loaded/);
staleResolve(result(original()));await stale;
assert.equal(source.getCircuit().gates[0].angle,'5*x0');
""")


@unittest.skipUnless(shutil.which("node"), "Node is not installed")
class AppModeSynchronizationTests(unittest.TestCase):
    def test_quantum_mode_uses_source_bounds_validated_while_away(self):
        script = r"""
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync('ftci/web/app.js', 'utf8');
const context = vm.createContext({assert});
vm.runInContext(source.slice(0, source.lastIndexOf('initialize().catch')), context);
vm.runInContext(`
const fields = Object.fromEntries(['model-name','target-expression','example-select','basis-editors','domain-inputs','max-bond','query-budget','tolerance','oversampling','validation-samples','seed','node-rule','shots'].map(id => [id, {value: '1', children: []}]));
globalThis.document = {getElementById: id => fields[id]};
// Rendering is stubbed; mode restoration, source callbacks, and request assembly
// use the real app behavior. Values match browser number inputs.
renderDomain = domain => { fields['domain-inputs'].children = domain.map(range => ({querySelector: selector => ({value: String(range[selector.includes('min') ? 0 : 1])})})); };
setMode = mode => { state.mode = mode; };
renderBasisEditors = () => {};
showError = () => {};
markCircuitChanged = () => {};
refreshLearnButton = () => {};
renderDomain([[-1, 1]]);
state.mode = 'quantum';
state.circuitValid = false;
const circuit = {domain: [[-2, 2]], gates: [{gate: 'ry', qubits: [0], angle: 'x0'}]};
state.circuitSource = {getCircuit: () => circuit, getSource: () => ({format: 'json', source: JSON.stringify(circuit)})};
switchProblemMode('function');
renderDomain([[10, 20]]);
// Validation completes after leaving Quantum, so its sidebar is not rendered.
circuitSourceChanged({valid: true, pending: false, circuit});
assert.deepEqual(readDomain(), [[10, 20]]);
switchProblemMode('quantum');
assert.deepEqual(readRequest().domain, [[-2, 2]]);
switchProblemMode('function');
assert.deepEqual(readDomain(), [[10, 20]]);
`, context);
"""
        project = Path(__file__).resolve().parents[1]
        completed = subprocess.run([shutil.which("node"), "-e", script], cwd=project,
                                   text=True, capture_output=True, timeout=15)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)


if __name__ == "__main__":
    unittest.main()

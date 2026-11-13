# FTCI Studio

A working research preview for learning continuous, real scalar functions as
local function dictionaries contracted with a coefficient MPS. The web app
submits **real TensorCrossInterpolation.jl jobs**; predictions use the learned
coefficient cores and do not call the training oracle.

## Install and start

Requires Python 3.10+, NumPy, Julia 1.10+, and TensorCrossInterpolation.jl 0.9.
Qiskit is optional and required only for importing Qiskit circuits.

Open a terminal in this folder and run:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[quantum]"
julia --project=ftci/julia -e 'using Pkg; Pkg.instantiate()'
.venv/bin/python -m ftci.server
```

Then open <http://127.0.0.1:8765>. Once installed, the last command alone
starts the server, and `Launch FTCI Studio.command` runs it on macOS.

The Julia executable and environment can be selected explicitly with
`FTCI_JULIA` and `FTCI_JULIA_PROJECT`. The SDK and exported predictor require
only NumPy for inference; Julia and Qiskit are not prediction dependencies.

Choose an example, edit the target and dictionaries, and click **Learn
function**. Results include independent validation, a function preview, new
input evaluation, basis values, MPS core slices, and model download. Quantum
examples accept an editable circuit specification and automatically construct
dictionaries and independently controlled encoding-angle queries.

Inside **Define the problem → Quantum circuit**, the source code defines the
function to be learned. Choose **JSON** or **Python (auto)**, type the circuit, or use
**Upload code** to load a `.json` or `.py` file (up to 96 KiB). Typing updates
the validated circuit and dictionaries automatically; incomplete code stays
visible, with its error, while the last valid diagram remains a disabled
reference. The original uploaded file is never modified.

The optional **Circuit diagram** panel lives inside the same
problem definition. Click a gate to change its operation, qubits, and angle
source. Choose **Input encoding**, enter a preprocessing expression such as
`2*x0**2+x1`, and apply the gate. Applied changes update the source code; the
server reparses that source when creating a learning job. Each encoding
displays its expression and input labels directly above the wires. The
input-to-encoding map highlights all gates using an input. The selected
encoding also shows the generator and dictionary computed by the same
backend adapter used during learning.

**Representation** is the second step in both problem types. The same
function has many tensor-network representations, and this step chooses
one. For a general function it is the dictionary editor. For a quantum
circuit it offers three ways to group encoding gates into dictionaries:

- **Canonical**: one dictionary per encoding gate, so the chain has as many
  sites as there are encoding gates. Training varies every gate angle
  independently.
- **Merged by preprocessing**: gates whose preprocessing functions agree are
  merged into one dictionary with one training angle. The chain has one
  site per distinct preprocessing function, and each dictionary holds the
  sums of its gates' frequencies: `m` Pauli rotations give frequencies
  `1, …, m` and dimension `2m + 1`.
- **Custom**: starts from the arrangement shown and lets you move
  dictionaries along the chain, join a dictionary with the next one, or
  split one into single gates. Gates on one dictionary that share a
  preprocessing function are merged; different preprocessing functions form a
  tensor-product dictionary and keep separate training angles. Joining every
  gate into one dictionary is plain regression in the full basis. Editing the
  circuit's gates resets a custom arrangement.

The step lists each dictionary with its preprocessing, gates, frequencies
and dimension, and shows chain length, largest dictionary, number of basis
functions and an estimated query count per TCI sweep before anything is
learned. The circuit diagram marks every encoding gate with its dictionary;
gates sharing a preprocessing function keep one colour in both choices.
The **Runs** result view keeps every completed run of the session in one
table, so the same circuit learned in both representations can be compared
by sampling, chain length, bond dimension, queries and error.

**Learning**, the third step, has two parts. *Sampling grid* decides where
the function is queried: the oversampling factor (auxiliary inputs per
dictionary function), the node rule, and for quantum circuits an optional
number of shots per query. *Compression* holds the TCI settings. Because the
sampling design depends only on dictionaries and nodes, the step shows the
inversion amplification `∏‖V⁺‖` and the worst condition number before any
query is made.

- **Equispaced** nodes are evenly spaced: over one period for a
  trigonometric dictionary with commensurate frequencies, where the design
  rows are orthogonal (condition number √2 without oversampling), and at
  interval midpoints otherwise.
- **Random** nodes are uniform on the site's interval and need oversampling.
- **Automatic** keeps, per dictionary, whichever of the two amplifies less.
- **Shots per query** simulates a finite-shot oracle by adding Gaussian
  noise of variance `1/shots` to each distinct training query. Validation
  always uses the exact circuit.

The **Prediction** result view plots the learned function on a slice of the
input domain. For models with several inputs you choose which input runs
along the horizontal axis, optionally a second input for a plane, and the
values at which the remaining inputs are held. A line shows target and
learned curves; a plane shows the learned function as a heat map, with the
target beside it on request. Signed values use two hues around a neutral
zero. Evaluating the learned function is free. Comparing with the target
queries it again, 61 times for a line and 961 for a plane, outside the
learning budget; the plot states how many queries were used.

A **Sweep** under the same step repeats the run for several values of one
setting: maximum bond, oversampling, shots per query, or representation.
Runs are submitted one after another. The Runs view then plots the relative
error (log scale) and the training queries along the swept values, above the
table of all runs; runs that fail, for example on the query budget, are
listed with their reason and left out of the plots.

The **Tensor network** result view has a *Coefficient basis* selector. The
learned function can be read in other bases of each dictionary's function
space without new queries, and without changing predictions or bond
dimensions:

- **Dictionary**: the dictionaries the model was learned and is exported with.
- **Orthonormal**: basis functions with unit mean square under the uniform
  measure on the coordinates the dictionaries are independent in. For a
  general function these are the inputs, and each dictionary is
  orthonormalized in its given order (monomials become Legendre
  polynomials). For a circuit they are the training angles, where the
  trigonometric dictionaries only need rescaling. Squared coefficients are
  then shares of the mean square.
- **Sampling grid (nodal)**: coefficients are the function values on the
  grid, the tensor that cross interpolation approximated.

The Diagnostics view follows the selected basis.

The **Diagnostics** result view starts with an error budget in the order
errors arise: oracle noise, TCI compression (last pivot error, tolerance,
bond cap), inversion amplification, and the measured function error. Below
it, *Coefficient tensor* reports the entanglement entropy and Schmidt
weights across every cut, the spreading of the coefficients over the
tensor-product basis with the resulting effective number of basis
functions, and for quantum circuits the coefficient weight by support size.

Measured on the Heisenberg example (seed 7, bond cap 16, exact queries; all
runs reach a relative error below 2e-14):

| Representation | Nodes | Oversampling | Training queries |
| --- | --- | --- | --- |
| Merged | automatic (equispaced) | 1 | 4,310 |
| Merged | automatic (equispaced) | 2 | 17,787 |
| Merged | random | 2 | 27,059 |
| Canonical | automatic (equispaced) | 1 | 16,060 |
| Canonical | automatic (equispaced) | 2 | 87,583 |
| Canonical | random | 2 | 84,768 |

With 10,000 shots, merged, bond cap 9: equispaced nodes without oversampling
give a relative error of 0.020 from 9,477 queries; random nodes need
oversampling 2 for 0.072 from 75,910 queries, and without oversampling
their amplification of about 4e10 makes the result useless.

The diagram draws gates in left-aligned columns: a gate sits in the first
column after the gates before it on its wires, so gates on different wires
are drawn side by side. To add a gate, click it in the palette and then a
`+` on a wire, or drag it there; a clicked palette gate stays selected for
repeated placement until Esc. Clicking a `+` first and the gate second
works too. A two-qubit gate takes the clicked wire and the next one, and
its qubits can be changed in the inspector. To move a gate, drag it to
another `+` or use the move buttons, which step past the nearest gate
sharing a wire. To remove one, select it and press Delete or click its `×`.
Inserting between gates on one wire adds one entry to the gate list;
mutually commuting gates are reordered only when a two-qubit gate cannot
otherwise be placed. Undo/redo is available for committed circuit edits. Circuit
settings include qubit count, initial computational state, and the measured
Pauli string. Unapplied visual drafts do not change the authoritative code.
Mode switches preserve the source draft. Invalid source prevents learning;
an old diagram cannot silently replace it.

Python uploads automatically detect supported Qiskit, Cirq and PennyLane
construction. The status line identifies the detected package. These readers
do not require the uploaded package to be installed and do not execute the
uploaded file. Each native reader accepts a static subset; arbitrary loops,
custom operations and external execution are reported instead of guessed.

Qiskit supports straight-line `QuantumCircuit` construction,
`Parameter`/`ParameterVector`, constants, supported gate calls, parameter
expressions, fixed parameter bindings, and `domain`/`observable`/`initial_state`
metadata. It parses this subset without executing uploaded Python. Visual
edits preserve supported source imports, declarations, aliases, and comments;
an edit that cannot round-trip safely reports an error. Loops, arbitrary
Python, unsupported gates, and unsupported symbolic functions are reported.
Use the Python SDK for richer circuit-building programs. Cirq supports native
`LineQubit` declarations, SymPy input symbols, `Circuit`/`append` operation
trees, standard fixed gates and radian `rx`/`ry`/`rz` rotations. PennyLane
supports one integer-wire device and one `@qml.qnode` function, scalar or
indexed-vector inputs, standard/rotation/Ising gates, and a single Pauli
expectation value. Measurement wire order is translated to the workspace's
Pauli-string convention (rightmost letter acts on q0). Source-preserving
visual edits are reparsed and checked; an edit outside a reader's supported
subset leaves the original source intact.

For **any other Python package**, add one top-level literal declaration:

```python
ftci_circuit = {
    "n_qubits": 2,
    "domain": [[-1, 1], [-1, 1]],
    "observable": "ZZ",
    "gates": [
        {"gate": "h", "qubits": [0]},
        {"gate": "ry", "qubits": [1], "angle": "2*x0**2+x1"},
        {"gate": "cx", "qubits": [0, 1]},
    ],
}
```

This is an explicit package-independent **learning specification**, not an
automatic interpretation of the surrounding native program. FTCI uses this
literal for the diagram, basis extraction and learning; imports and other
Python remain preserved context. The reserved `ftci_circuit` name must be
declared once and must not be referenced, reassigned or mutated elsewhere.
Angles are expression strings. Visual edits update this literal while
preserving the surrounding source and comments. Dynamically constructed
circuits or package-specific gates that cannot be represented by the shared
gate set need an adapter or the existing `from_generators` Python SDK path.

The quantum examples are the ones of the manuscript: an encoding layer followed by a
fixed circuit (its coefficients are the Pauli coordinates of the
Heisenberg-evolved observable, checked to 1e-10 in the tests), Trotterized
Heisenberg dynamics, Pauli expectation values of a shallow-circuit state
(read off the nodal view), and a convolutional circuit as a QML model. Each
example shows a one-line description under the selector.

Uploadable examples: `examples/circuit_source.py` (Qiskit),
`examples/circuit_cirq.py`, `examples/circuit_pennylane.py`, and
`examples/circuit_python.py` (package-independent specification).

This first version runs on this computer. It is not a public cloud service.
Its loopback-only server is a local development runner; the library and model
format are independent of that server.

## Learn an ordinary function

```python
import numpy as np
from ftci import Basis, FunctionProblem, Model, fit

problem = FunctionProblem(
    target=lambda x: np.cos(x[0] + x[1]) + 0.2,
    basis=[Basis(["1", "cos(x0)", "sin(x0)"]),
           Basis(["1", "cos(x1)", "sin(x1)"])],
    domain=[(-np.pi, np.pi), (-np.pi, np.pi)],
    name="Cosine addition",
)
model = fit(problem, max_bond=8, query_budget=2000)   # node_rule="auto", shots=None
print(model([0.2, 0.4]))
print(model.basis[0]([0.2, 0.4]))
print(model.cores[0])                       # (left bond, basis index, right bond)
print(model.coefficient([1, 1]))            # zero-based indices
model.save("cosine.ftci")
restored = Model.load("cosine.ftci")
```

Two helpers need no learning run. `design_report` returns the conditioning
of the sampling design without querying the target, and
`coefficient_structure` analyses any coefficient MPS:

```python
from ftci.engine import design_report
from ftci.diagnostics import coefficient_structure

report = design_report(problem, oversampling=1, node_rule="auto")
print(report["log10_inversion_amplification"], report["designs"][0]["condition_number"])
structure = coefficient_structure(model.cores)   # constant_index=0 adds support weights
print(structure["maximum_bond_entropy"], structure["spreading"]["participation"])
```

`basis_view` returns the cores and basis-function labels of a model in
another coefficient basis:

```python
from ftci.views import basis_view

view = basis_view(model, "orthonormal")   # or "dictionary", "nodal"
print(view["dictionaries"][0]["expressions"], view["cores"][0].shape)
```

The nodal view needs the sampling grid, which is kept in memory after `fit`
and is not part of the exported file. Expressions of an orthonormalized
general dictionary are printed with six significant digits.

Entropies are in nats and refer to the squared entries of the normalized
coefficient tensor, so they depend on the chosen dictionaries. Tensors with
at most 200,000 entries are analysed exactly; larger ones are sampled.

General dictionaries may contain trigonometric functions, polynomials, or
other supported scalar expressions. Coordinate names are `x0`, `x1`, etc.;
use `**` for powers. Expressions are parsed with a small allowlisted
interpreter and never passed to Python `eval`.

For an arbitrary local vector function:

```python
basis = Basis.from_callable(
    lambda x: np.array([1., np.cos(x[0]), np.sin(x[0])]),
    input_indices=[0], dimension=3,
    expressions=["1", "cos(x0)", "sin(x0)"],
)
```

Portable expressions are optional for local fitting and mandatory for
portable export. Their equivalence to the callable and the declared input
dependencies are the caller's responsibility.

## Learn a quantum function

`ftci.quantum` provides three entry points:

- `from_qiskit(circuit, input_parameters, domain, observable,
  fixed_parameters=...)`: a parameterized Qiskit circuit with fixed trained
  parameters. Standard RX/RY/RZ/RXX/RYY/RZZ and phase encodings are supported.
  Initial state is the computational zero state; prepare other states using
  fixed circuit gates. The current adapter uses an exact local statevector.
- `from_circuit(spec)`: a lightweight circuit dictionary with gates, angles,
  input domain, and Pauli observable; no Qiskit installation is needed.
- `from_generators(generators, preprocessing, auxiliary_oracle, domain,
  spectra=False)`: hidden circuits or external hardware. The oracle must
  independently control each encoding occurrence. Generators alone determine
  the dictionary, not the coefficient tensor.

Each returns a `QuantumProblem` accepted by the same `fit` function. Training
replaces each encoding occurrence's angle by an independent auxiliary
parameter. Predictions restore the original preprocessing functions.

All three accept `representation="canonical"` (default) or
`representation="merged"`:

```python
problem = from_circuit(spec, representation="merged")
print(problem.metadata["sites"])     # encoding gates, frequencies, dimension per site
```

A custom arrangement is an ordered partition of the encoding occurrences:

```python
problem = from_circuit(spec, representation="custom", sites=[[4], [2, 0, 3], [1]])
```

Preprocessing functions are compared numerically at fixed sample inputs, so
`2*x0` and `x0+x0` are merged. Only equal functions are merged; proportional or
shifted ones stay on separate sites. With `from_generators`, the supplied
oracle still receives one angle per encoding occurrence. Merging is refused
when a dictionary would need more than 512 frequencies, which happens for
many gates with incommensurate spectra. A merged model fixes the shared
angles, so it cannot be re-evaluated with different preprocessing per gate;
a canonical model can.

Qubit indices are zero-based. Pauli strings use Qiskit's little-endian
convention: in `"ZI"`, Z acts on qubit 1. Built-in simulation is limited to
small circuits (at most 10 qubits). Unsupported input-dependent gates fail
explicitly rather than guessing a generator. See runnable examples in
`examples/learn_quantum.py` and `examples/hidden_quantum_oracle.py`.

## Portable model format

A `.ftci` file is a ZIP archive containing:

- `manifest.json`: versioned schema, dictionary expressions and order,
  input domain, core shapes and index convention, and diagnostics.
- `cores.npz`: real float64 coefficient MPS cores; no Python pickle.
- `predict.py`: a standalone NumPy-only loader and predictor.

After extracting only `predict.py` beside the archive:

```sh
python predict.py cosine.ftci '[0.2, 0.4]'
python predict.py cosine.ftci '[[0.2, 0.4], [0.5, 0.7]]' --batch
```

`Model.load` reads the manifest and arrays; it never executes bundled code.
It validates schema version, array headers, bonds, shapes, finiteness, and
archive size. The coefficient representation depends on the chosen
dictionaries and their normalization; MPS cores also have gauge freedom.

## Scope and numerical meaning

- Direct general-function FTCI requires disjoint dictionary input blocks.
  Overlapping dependencies are rejected with an explanation. A merged
  single dictionary can handle small cases; generic overlapping-basis MPS
  regression is not implemented. The chirp example uses a merged dictionary.
- Arbitrary function input does not imply exact representability in an
  arbitrary finite dictionary. The independent holdout error reports the
  resulting approximation quality. Basis insufficiency, low rank, and
  conditioning can all limit accuracy.
- A one-site problem is sampled directly and transformed by its local
  pseudoinverse. Multi-site problems use the actual two-site, full-pivot
  Julia TCI algorithm. The full product tensor is never constructed.
- All-zero pilot values require an explicitly labelled zero-candidate
  fallback because TCI cannot start at a zero pivot. Independent validation
  remains empirical and is not proof of an identically zero function.
- `tolerance` is the TCI pivot criterion, relative to the largest sampled
  value and separate from continuous-function holdout error. Diagnostics
  include design ranks, singular values, condition numbers, the node rule
  used per dictionary, and whether that pivot criterion was met.
- The inversion amplification is an upper bound on how much an error of the
  sampled tensor can grow in the coefficients. Observed errors are usually
  far below it.
- `query_budget` counts actual oracle evaluations during training,
  validation, and optional preview generation. Reused training values are
  cached. Exceeding the training allocation fails explicitly; no unfinished
  model is presented as successful. This version does not resume jobs.
- Cancellation and time limits are cooperative between user callbacks;
  a single long-running Python callback cannot be interrupted safely.
- Shot noise is simulated, not measured: one Gaussian sample of variance
  `1/shots` per distinct training query, which the cache then reuses. With
  noise the sampled tensor has full rank, so TCI grows to the bond cap and
  can exhaust the query budget; lower the cap for noisy runs. Repeated-shot
  averaging, uncertainty estimation, and noise-specific stopping remain
  future work.
- The web UI accepts safe formulas, JSON circuits, and the documented Qiskit
  source subset, including uploaded code files. Arbitrary Python functions,
  Qiskit objects, and external device oracles use the Python SDK.

- Merging shortens the chain but does not by itself make a function
  learnable: a six-qubit Havlíček-type feature map, whose 42 encoding angles
  couple pairs of inputs, stayed at relative error 1 at bond 2 in both
  representations.
- Dictionaries joined across different preprocessing functions use random
  nodes; equispaced nodes exist only for dictionaries with one training
  coordinate. A joined dictionary may hold at most 1024 functions.
- Merging is by equal preprocessing functions only; proportional or shifted
  ones are not merged. A learned model is not converted from one arrangement
  to another: each arrangement is learned separately.

Public cloud workers, authentication, QASM import, and a general
overlapping-dictionary regression backend are not included yet.
These can use the same problem/model interfaces without changing the
exported predictor.

For a hardware cloud runner, the next interface should evaluate batches of
independent auxiliary angles and return expectation values, uncertainty,
actual shots, backend identity, job IDs, and execution metadata. The runner
should compile the lifted circuit once, map observables through its layout,
track adaptive query rounds, and enforce job/shot/QPU-time budgets. This is
distinct from the website hosting server. Credentials belong to the runner,
outside circuit code and exported models. No hardware account connection or
QPU submission is implemented by the current web runner.

## Validation

```sh
.venv/bin/python -m unittest discover -s tests -v
node --check ftci/web/app.js
```

Tests exercise actual Julia TCI recovery, original-input validation after
quantum angle lifting, arbitrary dictionaries, budget/cancellation paths,
Qiskit imports when installed, malformed inputs, model round trips, and an
isolated subprocess using the bundled predictor without FTCI or Julia.

## License

FTCI Studio is released under the Apache License, Version 2.0. See `LICENSE`
and `NOTICE`.

"""Learn a Qiskit circuit whose encoding gates share an external input."""
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from ftci import fit
from ftci.quantum import from_qiskit

x = Parameter("input")
theta = Parameter("trained_angle")
circuit = QuantumCircuit(2)
circuit.h(0)
circuit.ry(x, 0)
circuit.cx(0, 1)
circuit.rz(x*x, 1)
circuit.rx(theta, 0)

problem = from_qiskit(circuit, input_parameters=[x], domain=[[-np.pi, np.pi]],
                      observable="YY", fixed_parameters={theta: 0.7},
                      name="Shared-input circuit")
model = fit(problem, max_bond=8, validation_samples=100, query_budget=2000)
model.save("quantum.ftci")
print("Generated dictionaries:", [b.expressions for b in model.basis])
print("Prediction:", model([0.4]))
print("Validation:", model.metadata["validation"])

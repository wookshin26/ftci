from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
import numpy as np

x = ParameterVector("x", 2)
qc = QuantumCircuit(2)
qc.h(0)
qc.h(1)
qc.ry(x[0], 0)
# Nonlinear preprocessing remains part of the circuit source.
qc.rz(2*x[0]**2+x[1], 1)
qc.cx(0, 1)
qc.rx(x[0]+x[1], 0)
qc.ry(0.7, 1)

domain = [[-np.pi, np.pi], [-np.pi, np.pi]]
observable = "ZZ"

"""A source-import example: FTCI reads this program without executing it."""
import pennylane as qml
from pennylane import numpy as np

dev = qml.device("default.qubit", wires=2)
offset = 0.25


@qml.qnode(dev)
def circuit(x):
    qml.Hadamard(wires=0)
    qml.RZ(x[0], wires=0)
    qml.RY(np.sin(x[1]) + offset, wires=1)
    qml.CNOT(wires=[0, 1])
    return qml.expval(qml.PauliY(0) @ qml.PauliY(1))


domain = [[-np.pi, np.pi], [-1.0, 1.0]]
name = "PennyLane feature circuit"

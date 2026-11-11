"""Uploadable package-independent circuit specification.

Any package's code can surround this declaration. FTCI reads the literal
ftci_circuit below as its circuit definition; surrounding code is context,
not a second circuit that is automatically kept in sync.
"""

ftci_circuit = {
    "n_qubits": 2,
    "domain": [[-3.141592653589793, 3.141592653589793],
               [-3.141592653589793, 3.141592653589793]],
    "observable": "ZZ",
    "gates": [
        {"gate": "h", "qubits": [0]},
        {"gate": "ry", "qubits": [0], "angle": "x0"},
        {"gate": "rz", "qubits": [1], "angle": "2*x0**2"},
        {"gate": "cx", "qubits": [0, 1]},
        {"gate": "rx", "qubits": [0], "angle": "x0+x1"},
    ],
}

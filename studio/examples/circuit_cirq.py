# Native Cirq source: each input can feed several nonlinear encoding gates.
import cirq
import sympy as sp

# x[0] and x[1] map to FTCI inputs x0 and x1 in declaration order.
x = sp.symbols('x0:2')
q = cirq.LineQubit.range(2)
scale = 2
phase = scale*x[0]**2+x[1]

circuit = cirq.Circuit(
    cirq.H(q[0]),
    cirq.ry(x[0])(q[0]),
    cirq.CNOT(q[0], q[1]),
)
circuit.append(cirq.rz(phase)(q[1]))
circuit.append(cirq.rx(sp.sin(x[0])+x[1])(q[0]))

domain = [[-sp.pi, sp.pi], [-1, 1]]
# Little endian: rightmost letter acts on q0, leftmost on q1.
observable = 'ZZ'
initial_state = '00'
name = 'Shared-input nonlinear Cirq circuit'

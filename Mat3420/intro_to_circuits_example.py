import numpy as np
import qiskit as qk
import matplotlib.pyplot as plt
from qiskit.tools.visualization import circuit_drawer



n_quantum =  5
n_classic = 2

q_reg = qk.QuantumRegister(n_quantum)
#c_reg = qk.ClassicalRegister(n_classic)
circuit = qk.QuantumCircuit(q_reg)#, c_reg)

circuit.cx(q_reg[0], q_reg[4])
circuit.cx(q_reg[1], q_reg[4])
circuit.cx(q_reg[1], q_reg[3])
circuit.cx(q_reg[2], q_reg[3])
circuit.cx(q_reg[3], q_reg[4])

circuit.x(q_reg[3])
circuit.toffoli(q_reg[3], q_reg[4], q_reg[0])
circuit.x(q_reg[3])

circuit.x(q_reg[4])
circuit.toffoli(q_reg[3], q_reg[4], q_reg[2])
circuit.x(q_reg[4])

circuit.toffoli(q_reg[3], q_reg[4], q_reg[2])

print(circuit_drawer(circuit, output='latex_source'))

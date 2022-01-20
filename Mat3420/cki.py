import numpy as np
import qiskit as qk
import matplotlib.pyplot as plt
from qiskit.tools.visualization import circuit_drawer
from qiskit.quantum_info import Operator



circ = qk.QuantumCircuit(3)
circ.crz(-np.pi/2, 0, 1)

print(Operator(circ))

print(circ)
circ.draw('mpl')
plt.show()

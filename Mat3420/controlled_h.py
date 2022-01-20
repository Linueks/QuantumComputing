import numpy as np
import qiskit as qk
import matplotlib.pyplot as plt
from qiskit.circuit.library.standard_gates import HGate
from qiskit.quantum_info import Operator, Statevector
from qiskit.visualization import plot_bloch_multivector
from qiskit import BasicAer



backend = BasicAer.get_backend('unitary_simulator')



qr = qk.QuantumRegister(2)
qc = qk.QuantumCircuit(qr)

qc.ch(qr[0], qr[1])
print(qc)

job = qk.execute(qc, backend)
print(format_unitary(job.result().get_unitary(qc, decimals=3)))

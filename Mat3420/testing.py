import qiskit as qk

from qiskit import transpile, QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.providers.aer import QasmSimulator


from qiskit import Aer
my_backend = Aer.get_backend("qasm_simulator")


backend = QasmSimulator()

n = 1
q = QuantumRegister(n)
c = ClassicalRegister(n)

qc = QuantumCircuit(q,c)
qc.h(q[0])
qc.measure(q,c)
qc.x(q[0]).c_if(c,0)
qc.measure(q,c)
print(qc)


job = qk.execute(qc, backend=my_backend, shots=1024)
#job = my_backend.run(qc, shots=1024)
cif = job.result().get_counts(qc)
print(cif)

import numpy as np
import scipy as sp
import qiskit as qk
import qiskit.opflow as op
import matplotlib.pyplot as plt
import qiskit.quantum_info as qi
from classical_simulation import hamiltonian_xxx
from basic_approach import (trotter_step_zyxzyx, trotter_step_zzyyxx,
    build_circuit)
from qiskit.opflow import Zero, One, I, X, Y, Z

plt.style.use('seaborn-whitegrid')

# qiskit opflow seems like garbage...
# just gonna use numpy do the error calculation


XXs = (I^X^X) + (X^X^I)
YYs = (I^Y^Y) + (Y^Y^I)
ZZs = (I^Z^Z) + (Z^Z^I)

# Sum interactions
H = XXs + YYs + ZZs

H_numpy = H.to_matrix()


# the following could be done using just numpy methods tbh
diagonal = np.array(
    [
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 2, 0, 0, 0, 0, 0],
    [0, 0, 0, 2, 0, 0, 0, 0],
    [0, 0, 0, 0, 2, 0, 0, 0],
    [0, 0, 0, 0, 0, 2, 0, 0],
    [0, 0, 0, 0, 0, 0, -4, 0],
    [0, 0, 0, 0, 0, 0, 0, -4]
    ],
    dtype=complex,
)
vectors = np.array(
    [
    [0, 0,              1, 0, 0, 0,             0, 0],
    [-1/np.sqrt(2), 0,  0, 0, 0, 1/np.sqrt(3),  1/np.sqrt(6), 0],
    [0, 0,              0, 0, 0, 1/np.sqrt(3),  -2/np.sqrt(6), 0],
    [0, -1/np.sqrt(2),  0, 0, 1/np.sqrt(3), 0,  0, 1/np.sqrt(6)],
    [1/np.sqrt(2), 0,   0, 0, 0, 1/np.sqrt(3),  1/np.sqrt(6), 0],
    [0, 0,              0, 0, 1/np.sqrt(3), 0,  0, -2/np.sqrt(6)],
    [0, 1/np.sqrt(2),   0, 0, 1/np.sqrt(3), 0,  0, 1/np.sqrt(6)],
    [0, 0,              0, 1, 0, 0,             0, 0]
    ],
    dtype=complex,
)

vectors_inverse = np.array(
    [
    [0, -1/np.sqrt(2), 0, 0, 1/np.sqrt(2), 0, 0, 0],
    [0, 0, 0, -1/np.sqrt(2), 0, 0, 1/np.sqrt(2), 0],
    [1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 1/np.sqrt(3), 0, 1/np.sqrt(3), 1/np.sqrt(3), 0],
    [0, 1/np.sqrt(3), 1/np.sqrt(3), 0, 1/np.sqrt(3), 0, 0, 0],
    [0, 1/np.sqrt(6), -2/np.sqrt(6), 0, 1/np.sqrt(6), 0, 0, 0],
    [0, 0, 0, 1/np.sqrt(6), 0, -2/np.sqrt(6), 1/np.sqrt(6), 0]
    ],
    dtype=complex,
)

check_diagonalization = np.matmul(np.matmul(vectors, diagonal), vectors_inverse)
#print(H_numpy)
#print(np.round(check_diagonalization))


initial_state = One^One^Zero
initial_state = initial_state.to_matrix()

initial_state = np.array(
    [
    [0],
    [0],
    [0],
    [0],
    [0],
    [0],
    [1],
    [0],
    ]
)

# these give same result sooo, and theyre both just identities?
# it is correct that for pi, 2pi, ... npi the exact propagator will just be the
# identity.
propagator = np.matmul(np.matmul(vectors, sp.linalg.expm(-1j * np.pi * diagonal)), vectors_inverse)
propagator_2 = sp.linalg.expm(-1j * np.pi * H_numpy)

#print(np.round(propagator - propagator_2))


time = qk.circuit.Parameter('t')


circuit_zyxzyx, quantum_register = build_circuit(
    time,
    trotter_step_zyxzyx,
    trotter_steps=4,
    target_time=np.pi,
    draw_circuit=False,
    n_qubits=3,
    active_qubits=[0,1,2],
)

circuit_zzyyxx, quantum_register = build_circuit(
    time,
    trotter_step_zzyyxx,
    trotter_steps=4,
    target_time=np.pi,
    draw_circuit=False,
    n_qubits=3,
    active_qubits=[0,1,2],
)

#print(circuit_test)
circuit_operator = qi.Operator(circuit_test)

print(np.round(circuit_operator.data))
print(np.round(propagator_2))

#print(initial_state)
#print(np.round(np.matmul(propagator, initial_state), 4))

#print(np.round(np.matmul(propagator_2, initial_state), 4))
#print(np.round(np.matmul(np.matmul(vectors, sp.linalg.expm(diagonal)), vectors_inverse), 3))











if __name__=='__main__':
    pass

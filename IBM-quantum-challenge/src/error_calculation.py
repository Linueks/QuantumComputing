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
    ]
)
vectors = np.array(
    [
    [0, 0, 1, 0, 0, 0, 0, 0],
    [-1, 0, 0, 0, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 1, -2, 0],
    [0, -1, 0, 0, 1, 0, 0, 1],
    [1, 0, 0, 0, 0, 1, 1, 0],
    [0, 0, 0, 0, 1, 0, 0, -2],
    [0, 1, 0, 0, 1, 0, 0, 1],
    [0, 0, 0, 1, 0, 0, 0, 0]
    ]
)

vectors_inverse = np.array(
    [
    [0, -0.5, 0, 0, 0.5, 0, 0, 0],
    [0, 0, 0, -0.5, 0, 0, 0.5, 0],
    [1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 1/3, 0, 1/3, 1/3, 0],
    [0, 1/3, 1/3, 0, 1/3, 0, 0, 0],
    [0, 1/6, -1/3, 0, 1/6, 0, 0, 0],
    [0, 0, 0, 1/6, 0, -1/3, 1/6, 0]
    ]
)

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


propagator = np.matmul(np.matmul(vectors, sp.linalg.expm(-1j * np.pi * diagonal)), vectors_inverse)

propagator_2 = sp.linalg.expm(-1j * np.pi * H_numpy)

print(initial_state)
print(np.round(np.matmul(propagator, initial_state), 4))

print(np.round(np.matmul(propagator_2, initial_state), 4))
#print(np.round(np.matmul(np.matmul(vectors, sp.linalg.expm(diagonal)), vectors_inverse), 3))






if __name__=='__main__':
    pass

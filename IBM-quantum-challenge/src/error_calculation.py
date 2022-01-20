import numpy as np
import qiskit as qk
import qiskit.opflow as op
import matplotlib.pyplot as plt
from classical_simulation import hamiltonian_xxx
plt.style.use('seaborn-whitegrid')

# qiskit opflow seems like garbage...
# just gonna use numpy do the error calculation

pauli_x = op.X
pauli_y = op.Y
pauli_z = op.Z
#print(pauli_x.to_matrix())






if __name__=='__main__':
    pass

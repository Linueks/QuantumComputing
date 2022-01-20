import numpy as np




ket_0 = np.array([1, 0])
ket_1 = np.array([0, 1])


#a)
state = np.kron(ket_0, ket_0) + np.kron(ket_0, ket_1) + np.kron(ket_1, ket_1)
reduced_density = np.outer(ket_0, ket_0) + np.outer(ket_0, ket_0) + np.outer(ket_1, ket_1)
eigenvalues = np.linalg.eig(reduced_density)[0]
print(eigenvalues)


#b)
state2 = np.kron(ket_0, ket_0) - np.kron(ket_0, ket_1) - np.kron(ket_1, ket_0) + np.kron(ket_1, ket_1)
reduced_density2 = np.outer(ket_0, ket_0) + np.outer(ket_0, ket_0) + np.outer(ket_1, ket_1) + np.outer(ket_1, ket_1)

print(state2)
print(reduced_density2)
print(np.linalg.eig(reduced_density2)[0])
print(np.linalg.matrix_rank(reduced_density2))

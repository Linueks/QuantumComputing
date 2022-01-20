import numpy as np

e_1 = np.array([
    [0],
    [1/np.sqrt(2)],
    [-1/np.sqrt(2)],
    [0]
])

e_2 = np.array([
    [0],
    [1/np.sqrt(2)],
    [0],
    [-1/np.sqrt(2)]
])

V = np.array([
    [1, 0, 0, 0],
    [0, 1/2 - 1j/2, 1/2, 1j/2],
    [0, 0, 1/2+1j/2, 1/2-1j/2],
    [0, 1/2+1j/2, -1j/2, 1/2]
])

V_prime = np.array([
    [1, 0, 0, 0],
    [0, 1/2 - 1j/2, 0, 1/2+1j/2],
    [0, 1/2, 1/2+1j/2, -1j/2],
    [0, 1j/2, 1/2-1j/2, 1/2]
])


def calc_restriction(A, u_1, u_2):
    """
    Calculates the restriction of A to the space spanned by u_1 and u_2
    V|_k = [<u_1, Vu_1>, <u_1, Vu_2>]
           [<u_2, Vu_1>, <u_2, Vu_2>]
    """

    A_u1 = np.matmul(A, u_1)
    A_u2 = np.matmul(A, u_2)

    result = np.array([
        [np.vdot(A_u1, u_1), np.vdot(A_u2, u_1)],
        [np.vdot(A_u1, u_2), np.vdot(A_u2, u_2)]
    ])

    return result


restriction_V = calc_restriction(V, e_1, e_2)
restriction_V_prime = calc_restriction(V_prime, e_1, e_2)

commutator1 = np.matmul(restriction_V, restriction_V_prime) - np.matmul(restriction_V_prime, restriction_V)
commutator2 = np.matmul(restriction_V_prime, restriction_V) - np.matmul(restriction_V, restriction_V_prime)

#print(commutator1, commutator2)

eigenvalues1 = np.linalg.eig(V)
eigenvalues2 = np.linalg.eig(V_prime)

print(f"eig1: {eigenvalues1[0]}\n")
print(f"eig2: {eigenvalues2[0]}\n")
print(f"given: {(1 + 1j*np.sqrt(15))/4}")
print(f"given: {(1 - 1j*np.sqrt(15))/4}")

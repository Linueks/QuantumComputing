import sympy as sp



e_1 = sp.Matrix([
    [0],
    [1/sp.sqrt(2)],
    [-1/sp.sqrt(2)],
    [0]
])

e_2 = sp.Matrix([
    [0],
    [-1/sp.sqrt(6)],
    [-1/sp.sqrt(6)],
    [sp.sqrt(2/3)]
])

V = sp.Matrix([
    [1, 0, 0, 0],
    [0, 1/2 - 1j/2, 1/2, 1j/2],
    [0, 0, 1/2+1j/2, 1/2-1j/2],
    [0, 1/2-1j/2, 1/2, 1j/2]
])

V_prime = sp.Matrix([
    [1, 0, 0, 0],
    [0, 1/2 - 1j/2, 0, 1/2+1j/2],
    [0, -1j/2, 1/2+1j/2, 1/2],
    [0, 1/2, 1/2-1j/2, 1j/2]
])


print(e_1)
print(e_2)
print(V)
print(V_prime)

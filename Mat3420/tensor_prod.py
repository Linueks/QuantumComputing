import numpy as np



def bmatrix(a):
    """Returns a LaTeX bmatrix

    :a: numpy array
    :returns: LaTeX bmatrix as a string
    """
    if len(a.shape) > 2:
        raise ValueError('bmatrix can at most display two dimensions')
    lines = str(a).replace('[', '').replace(']', '').splitlines()
    rv = [r'\begin[bmatrix]']
    rv += ['  ' + ' & '.join(l.split()) + r'\\' for l in lines]
    rv +=  [r'\end[bmatrix]']
    return '\n'.join(rv)



h = np.array([[1/np.sqrt(2), 1/np.sqrt(2)],[1/np.sqrt(2), -1/np.sqrt(2)]])
i = np.eye(2)
ccx = np.array([[1,0,0,0,0,0,0,0],
                [0,1,0,0,0,0,0,0],
                [0,0,1,0,0,0,0,0],
                [0,0,0,1,0,0,0,0],
                [0,0,0,0,1,0,0,0],
                [0,0,0,0,0,1,0,0],
                [0,0,0,0,0,0,0,1],
                [0,0,0,0,0,0,1,0]])
k = np.array([[1, 0],[0, 1j]])
k_inv = np.array([[1, 0],[0, -1j]])



ii = np.kron(i, i)
iih = np.kron(ii, h)
one = np.matmul(iih, ccx)
two = np.matmul(one, iih)
iik_inv = np.kron(ii, k_inv)
three = np.matmul(two, iik_inv)
four = np.matmul(three, ccx)
iik = np.kron(ii, k)
five = np.matmul(four, iik)
six = np.matmul(five, ccx)
# looks pretty good


ck = np.array([[1,0,0,0],
               [0,1,0,0],
               [0,0,1,0],
               [0,0,0,1j]])

test = np.kron(ck, i)




#print(bmatrix(two))

print(six)

print(test)
epsilon = 1e-3
print(np.abs(six - test) < 1e-3)

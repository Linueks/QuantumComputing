import numpy as np


#Basic Gates -------------------------------------------------------------------
# all of these are hermitian
I = np.eye(2)
X = np.array([
            [0, 1],
            [1, 0]])
Y = np.array([
            [0, -1j],
            [1j, 0]])
Z = np.array([
            [1, 0],
            [0, -1]])
H = np.array([
            [1/np.sqrt(2), 1/np.sqrt(2)],
            [1/np.sqrt(2), -1/np.sqrt(2)]])
CNOT = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]])
CNOT2 = np.array([
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 1, 0, 0]])

SWAP = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]])

#Sqrt Gates --------------------------------------------------------------------
# these are not hermitian
sqrtX = np.array([
    [1 / 2 * (1 + 1j), 1 / 2 * (1 - 1j)],
    [1 / 2 * (1 - 1j), 1 / 2 * (1 + 1j)]])
sqrtY = np.array([
    [1/np.sqrt(2), 1/np.sqrt(2)],
    [-1/np.sqrt(2), 1/np.sqrt(2)]])
sqrtZ = np.array([
    [1, 0],
    [0, 1j]])


#Inverse Sqrt Gates ------------------------------------------------------------
inv_sqrtX = np.array([
    [1 / 2 * (1 - 1j), 1 / 2 * (1 + 1j)],
    [1 / 2 * (1 + 1j), 1 / 2 * (1 - 1j)]])
inv_sqrtY = np.array([
    [1/np.sqrt(2), -1/np.sqrt(2)],
    [1/np.sqrt(2), 1/np.sqrt(2)]])
inv_sqrtZ = np.array([
    [1, 0],
    [0, -1j]])


#print(np.matmul(np.matmul(np.kron(H, H), CNOT)))


def check_control_equality(sqrt, inv_sqrt):
    """
    DOES NOT WORK YET

    Function to check whether controlled gate equility holds
    Takes sqrt and inv sqrt of the gate you want to check it for.

    -----o------        ---|gate|---
         |         =         |
    ---|gate|---        -----o------

      number 1     =     number 2
    """
    I_tensor_sqrtgate = np.kron(I, sqrt)
    I_tensor_invsqrtgate = np.kron(I, inv_sqrt)
    step1 = np.matmul(CNOT, I_tensor_sqrtgate)
    step2 = np.matmul(step1, I_tensor_invsqrtgate)
    number_one = np.matmul(step2, CNOT)
    print(f'First tensor product:\n {I_tensor_invsqrtgate}\n ---> Step 2:\n {step1}\n ---> Second tensor product:\n {I_tensor_sqrtgate}\n ---> Step4:\n {step2}\n')
    print('-----------------------------------------------------------------------------')
    print(f'Circuit One: \n {number_one}\n\n')
    """
    print('Next Circuit ----------------------------------------------------------------')
    step1 = np.kron(sqrt, I)
    step2 = np.matmul(step1, CNOT)
    step3 = np.kron(inv_sqrt, I)
    step4 = np.matmul(step2, step3)
    number_two = np.matmul(step4, CNOT)
    print(f'Step 1:\n {step1}\n ---> Step 2:\n {step2}\n ---> Step3:\n {step3}\n ---> Step4:\n {step4}\n')
    print('-----------------------------------------------------------------------------')
    print(f'Circuit Two: \n {number_two}')
    """

    #print(number_one == number_two)


check_control_equality(Z, Z.T)

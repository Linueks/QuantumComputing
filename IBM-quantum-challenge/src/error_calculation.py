import numpy as np
import scipy as sp
import qiskit as qk
import qiskit.opflow as op
import matplotlib.pyplot as plt
import qiskit.quantum_info as qi
from classical_simulation import hamiltonian_xxx
from simulation import *
from qiskit.opflow import Zero, One, I, X, Y, Z
plt.style.use('seaborn-whitegrid')


def make_hamiltonian_matrix():
    xx = (I^X^X) + (X^X^I)
    yy = (I^Y^Y) + (Y^Y^I)
    zz = (I^Z^Z) + (Z^Z^I)
    hamiltonian = xx + yy + zz
    hamiltonian_numpy = hamiltonian.to_matrix()

    return hamiltonian_numpy



def manually_diagonalize_hamiltonian():
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
    return vectors, diagonal, vectors_inverse



def make_propagator(
    hamiltonian,
    use_diagonalized=True,
):
    # these give same result sooo, and theyre both just identities?
    # it is correct that for pi, 2pi, ... npi the exact propagator will just be the
    # identity.
    if use_diagonalized:
        vectors, diagonal, vectors_inverse = manually_diagonalize_hamiltonian()
        propagator = np.matmul(
            np.matmul(
                vectors,
                sp.linalg.expm(-1j * np.pi * diagonal)
            ),
            vectors_inverse
        )
        return propagator

    else:
        propagator = sp.linalg.expm(-1j * np.pi * hamiltonian)
        return propagator



#check_diagonalization = np.matmul(np.matmul(vectors, diagonal), vectors_inverse)
#print(H_numpy)
#print(np.round(check_diagonalization))



def calculate_error(
    initial_state,
    propagator,
    trotter_propagator,
):
    """
    This funcion should return the L2 norm squared between the approximate
    propagator applied to the initial state and the exact propagator applied to
    the initial state.
    """
    exact_propagation = np.matmul(propagator, initial_state)
    approximate_propagation = np.matmul(trotter_propagator, initial_state)
    error = np.linalg.norm(exact_propagation - approximate_propagation)**2

    return error



def calculate_error_evolution(
    initial_state,
    exact_propagator,
    trotter_step_function,
    trotter_steps_min=4,
    trotter_steps_max=16,
):
    """
    This function should loop over desired range of Trotter steps and return an
    array of error values from the exact propagation.
    """
    trotter_steps = trotter_steps_max - trotter_steps_min
    errors = np.zeros(trotter_steps)

    for steps in range(1, trotter_steps+1):
        quantum_circuit, quantum_register = build_circuit(
            time,
            trotter_step_function,
            trotter_steps=steps,
            target_time=np.pi,
            draw_circuit=False,
            n_qubits=3,
            active_qubits=[0,1,2],
            symmetry_protection=True,
            transpile_circuit=0,
        )

        circuit_operator = qi.Operator(quantum_circuit)
        circuit_operator = circuit_operator.data
        errors[steps-1] = calculate_error(
            initial_state,
            exact_propagator,
            circuit_operator,
        )

    return errors




#print(np.round(circuit_operator.data))
#print(np.round(propagator_2))

#print(initial_state)
#print(np.round(np.matmul(propagator, initial_state), 4))

#print(np.round(np.matmul(propagator_2, initial_state), 4))
#print(np.round(np.matmul(np.matmul(vectors, sp.linalg.expm(diagonal)), vectors_inverse), 3))


if __name__=='__main__':
    initial_state = One^One^Zero

    initial_state = initial_state.to_matrix()
    #print(initial_state)
    """
    initial_state = np.array(
        [0, 0, 0, 0, 0, 0, 1, 0],
        dtype=complex,
    )
    #"""
    """
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
        ],
        dtype=complex,
    )
    #"""
    #print(initial_state)
    time = qk.circuit.Parameter('t')
    trotter_steps_min = 1
    trotter_steps_max = 20

    hamiltonian = make_hamiltonian_matrix()
    exact_propagator = make_propagator(
        hamiltonian,
        use_diagonalized=False,
    )
    errors_zyxzyx = calculate_error_evolution(
        initial_state,
        exact_propagator,
        trotter_step_zyxzyx,
        trotter_steps_min=trotter_steps_min,
        trotter_steps_max=trotter_steps_max,
    )
    errors_zzyyxx = calculate_error_evolution(
        initial_state,
        exact_propagator,
        trotter_step_zzyyxx,
        trotter_steps_min=trotter_steps_min,
        trotter_steps_max=trotter_steps_max,
    )
    cancellations = calculate_error_evolution(
        initial_state,
        exact_propagator,
        trotter_step_actual_dot_product,
        trotter_steps_min=trotter_steps_min,
        trotter_steps_max=trotter_steps_max,
    )
    plt.plot(
        range(trotter_steps_min, trotter_steps_max),
        errors_zyxzyx,
        label='error zyxzyx',
    )
    plt.plot(
        range(trotter_steps_min, trotter_steps_max),
        errors_zzyyxx,
        label='error zzyyxx',
    )


    plt.plot(
        range(trotter_steps_min, trotter_steps_max),
        cancellations,
        label='trotter dot product',
    )

    plt.xlabel('Trotter Steps')
    plt.ylabel('Error')
    plt.title('L2 Squared Error of Trotter Approximation vs Exact Propagation')
    plt.legend()
    plt.show()

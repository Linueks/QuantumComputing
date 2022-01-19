import numpy as np
import qiskit as qk
import qiskit.opflow as opflow
import matplotlib.pyplot as plt



def heisenberg_chain():
    # define Heisenberg XXX hamiltonian opflow way
    # defining operators using qiskit opflow
    identity = opflow.I
    pauli_x = opflow.X
    pauli_y = opflow.Y
    pauli_z = opflow.Z

    x_interaction = (identity^pauli_x^pauli_x) + (pauli_x^pauli_x^identity)
    y_interaction = (identity^pauli_y^pauli_y) + (pauli_y^pauli_y^identity)
    z_interaction = (identity^pauli_z^pauli_z) + (pauli_z^pauli_z^identity)
    total_interaction = x_interaction + y_interaction + z_interaction

    return total_interaction



def propagator(time):
    # define the time evolution operator opflow way
    Hamiltonian = heisenberg_chain()
    time_evolution_unitary = (time * Hamiltonian).exp_i()

    return time_evolution_unitary



def classical_simulation(initial_state):
    # A copy paste from the notebook just to have it here

    time_points = np.linspace(0, np.pi, 100)
    probability_110 = np.zeros_like(time_points)

    for i, t in enumerate(time_points):
        probability_110[i] = np.abs((~initial_state @ propagator(float(t)) \
                                    @ initial_state).eval())**2

    plt.plot(time_points, probability_110)
    plt.xlabel('Time')
    plt.ylabel(r'Probability of state $|110\rangle$')
    plt.title(r'Evolution of state $|110\rangle$ under $H_{XXX}$')
    plt.show()



if __name__=='__main__':
    ket_zero = opflow.Zero
    ket_one = opflow.One
    initial_state = ket_one^ket_one^ket_zero
    classical_simulation(initial_state)
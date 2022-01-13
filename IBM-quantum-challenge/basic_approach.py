"""
Starting with implementation provided by IBM at:
https://github.com/qiskit-community/open-science-prize-2021/blob/main/ibmq-qsim-challenge.ipynb
"""
import numpy as np
import matplotlib.pyplot as plt
import qiskit as qk
import qiskit.opflow as opflow
import qiskit.ignis.verification.tomography as tomo
from qiskit.providers.aer import QasmSimulator
from qiskit.quantum_info import state_fidelity
#from qiskit.test.mock import FakeJakarta

# this is dumb...
import warnings
warnings.filterwarnings('ignore')





def heisenberg_chain():
    # define Heisenberg XXX hamiltonian opflow way
    # defining qubit states and operators using qiskit opflow
    ket_zero = opflow.Zero
    ket_one = opflow.One
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



def classical_simulation():
    # A copy paste from the notebook just to have it here

    time_points = np.linspace(0, np.pi, 100)
    initial_state = ket_one^ket_one^ket_zero
    probability_110 = np.zeros_like(time_points)

    for i, t in enumerate(time_points):
        probability_110[i] = np.abs((~initial_state @ propagator(float(t)) \
                                    @ initial_state).eval())**2

    plt.plot(time_points, probability_110)
    plt.xlabel('Time')
    plt.ylabel(r'Probability of state $|110\rangle$')
    plt.title(r'Evolution of state $|110\rangle$ under $H_{XXX}$')
    plt.show()



def construct_trotter_gate(time):
    # decomposition of propagator into quantum gates (copy pasta)
    # I'm not sure I like this way of programming using opflow. Feels like I don't
    # know what's happening behind the scenes. Ask Alessandro

    # build three components of single trotter step: xx, yy, zz (ugly code)
    xx_register = qk.QuantumRegister(2)
    xx_circuit = qk.QuantumCircuit(xx_register, name='xx')
    yy_register = qk.QuantumRegister(2)
    yy_circuit = qk.QuantumCircuit(yy_register, name='yy')
    zz_register = qk.QuantumRegister(2)
    zz_circuit = qk.QuantumCircuit(zz_register, name='zz')

    xx_circuit.ry(np.pi/2, [0,1])
    xx_circuit.cnot(0, 1)
    xx_circuit.rz(2*time, 1)
    xx_circuit.cnot(0, 1)
    xx_circuit.ry(-np.pi/2, [0,1])

    yy_circuit.rx(np.pi/2, [0,1])
    yy_circuit.cnot(0, 1)
    yy_circuit.rz(2*time, 1)
    yy_circuit.cnot(0, 1)
    yy_circuit.rx(-np.pi/2, [0,1])

    zz_circuit.cnot(0, 1)
    zz_circuit.rz(2*time, 1)
    zz_circuit.cnot(0, 1)

    # Convert custom quantum circuit into a gate
    xx = xx_circuit.to_instruction()
    yy = yy_circuit.to_instruction()
    zz = zz_circuit.to_instruction()

    # Combine subcircuits into a single multiqubit gate representing a single trotter step
    num_qubits = 3

    trot_register = qk.QuantumRegister(num_qubits)
    trot_circuit = qk.QuantumCircuit(trot_register, name='trot')

    for i in range(0, num_qubits-1):
        trot_circuit.append(zz, [trot_register[i], trot_register[i+1]])
        trot_circuit.append(yy, [trot_register[i], trot_register[i+1]])
        trot_circuit.append(xx, [trot_register[i], trot_register[i+1]])

    # Convert custom quantum circuit into a gate
    trotter_gate = trot_circuit.to_instruction()

    return trotter_gate



def trotterized_simulation(time,
                           target_time=np.pi,
                           trotter_steps=4,
                           draw_circuit=False):
    # generate the full circuit for the trotterized simulation
    # there are also some "fancy / ugly" things happening here
    quantum_register = qk.QuantumRegister(7)                                    # 7 qubits on Jakarta machine
    quantum_circuit = qk.QuantumCircuit(quantum_register)

    # set up initial state |110>
    quantum_circuit.x([3, 5])
    single_trotter_step = construct_trotter_gate(time)

    for n in range(trotter_steps):
        quantum_circuit.append(single_trotter_step,
                            [quantum_register[1],
                             quantum_register[3],
                             quantum_register[5]])

        quantum_circuit = quantum_circuit.bind_parameters(
                            {time: target_time/trotter_steps})
        final_circuit = tomo.state_tomography_circuits(quantum_circuit,
                                                    [quantum_register[1],
                                                     quantum_register[3],
                                                     quantum_register[5]])

    if draw_circuit:
        #print(final_circuit[-1].decompose())
        print(final_circuit[-1])

    return final_circuit



def execute_circuit(shots=8192, repetitions=8, backend=):
    # wrapper function to run jobs







if __name__=='__main__':
    #classical_simulation()
    time = qk.circuit.Parameter('t')

    # set up qiskit simulators
    jakarta_noiseless = QasmSimulator()
    """
    provider = qk.IBMQ.get_provider(hub='ibm-q-community',
                                    group='ibmquantumawards',
                                    project='open-science-22')
    jakarta_noisy = QasmSimulator.from_backend(provider.get_backend('ibmq_jakarta'))
    """



    trotterized_simulation(time, draw_circuit=True)

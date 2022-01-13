"""
Starting with implementation provided by IBM at:
https://github.com/qiskit-community/open-science-prize-2021/blob/main/ibmq-qsim-challenge.ipynb
"""
import numpy as np
import qiskit as qk
import time as time
import qiskit.opflow as opflow
import matplotlib.pyplot as plt
import qiskit.ignis.verification.tomography as tomo
from qiskit.providers.aer import QasmSimulator
from qiskit.quantum_info import state_fidelity
#from qiskit.test.mock import FakeJakarta

# this is dumb...
import warnings
warnings.filterwarnings('ignore')



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



def construct_trotter_gate(t):
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
    xx_circuit.rz(2*t, 1)
    xx_circuit.cnot(0, 1)
    xx_circuit.ry(-np.pi/2, [0,1])

    yy_circuit.rx(np.pi/2, [0,1])
    yy_circuit.cnot(0, 1)
    yy_circuit.rz(2*t, 1)
    yy_circuit.cnot(0, 1)
    yy_circuit.rx(-np.pi/2, [0,1])

    zz_circuit.cnot(0, 1)
    zz_circuit.rz(2*t, 1)
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



def trotterized_simulation(t,
                           target_time=np.pi,
                           trotter_steps=6,
                           draw_circuit=False):
    # generate the full circuit for the trotterized simulation
    # there are also some "fancy / ugly" things happening here
    quantum_register = qk.QuantumRegister(5)                                    # 7 qubits on Jakarta machine. 5 on Belem
    quantum_circuit = qk.QuantumCircuit(quantum_register)

    # set up initial state |110>
    quantum_circuit.x([3, 4])                                                   # Remember to switch back once access to Jakarta
    single_trotter_step = construct_trotter_gate(t)

    for n in range(trotter_steps):
        quantum_circuit.append(single_trotter_step,                             # Switch
                            [quantum_register[1],
                             quantum_register[3],
                             quantum_register[4]])

        quantum_circuit = quantum_circuit.bind_parameters(
                            {t: target_time/trotter_steps})
        final_circuit = tomo.state_tomography_circuits(quantum_circuit,         # Switch
                                                    [quantum_register[1],
                                                     quantum_register[3],
                                                     quantum_register[4]])

    if draw_circuit:
        #print(final_circuit[-1].decompose())
        print(final_circuit[-1])

    return final_circuit



def execute_circuit(circuit,
                    shots=8192,
                    repetitions=8,
                    backend=QasmSimulator()):
    # wrapper function to run jobs. Assumes QasmSimulator is imported as such.
    jobs = []
    for i in range(repetitions):
        job = qk.execute(circuit, backend, shots=shots)
        print(f'Job ID: {job.job_id()}')
        jobs.append(job)

    for job in jobs:
        qk.tools.monitor.job_monitor(job)

        # this thing seems stupid too
        try:
            if job.error_message() is not None:
                print(job.error_message())
        except:
            pass

    return jobs



def tomography_analysis(result, circuit, target_state):
    # assumes the target state is given as a qiskit opflow state
    target_state_matrix = target_state.to_matrix()
    tomography_fitter = tomo.StateTomographyFitter(result, circuit)
    rho_fit = tomography_fitter.fit(method='lstsq')
    fidelity = qk.quantum_info.state_fidelity(rho_fit, target_state_matrix)

    return fidelity



def run_simulation(backend):
    circuit = trotterized_simulation(time, draw_circuit=False)
    jobs = execute_circuit(circuit, backend=backend)
    fidelities = []

    for job in jobs:
        fidelity = tomography_analysis(job.result(), circuit, initial_state)
        fidelities.append(fidelity)

    print(f'state tomography fidelity = {np.mean(fidelities):.4f}',
           '\u00B1', f'{np.std(fidelities):.4f}')






if __name__=='__main__':
    ket_zero = opflow.Zero
    ket_one = opflow.One
    initial_state = ket_one^ket_one^ket_zero

    #classical_simulation(initial_state)
    time = qk.circuit.Parameter('t')

    # set up qiskit simulators
    jakarta_noiseless = QasmSimulator()

    provider = qk.IBMQ.load_account()
    provider = qk.IBMQ.get_provider(hub='ibm-q', group='open', project='main')
    #print(provider.backends())
    belem_backend = provider.get_backend('ibmq_belem')                          # has the same topology as Jakarta with qubits 1,3,4 corresponding to 1,3,5
    properties = belem_backend.properties()
    #print(properties)
    sim_noisy_belem = QasmSimulator.from_backend(belem_backend)
    #print(sim_noisy_belem)

    #jakarta = provider.get_backend('ibmq_jakarta')
    #properties = jakarta.properties()
    #print(properties)

    # Simulated backend based on ibmq_jakarta's device noise profile
    #sim_noisy_jakarta = QasmSimulator.from_backend(jakarta)
    run_simulation(sim_noisy_belem)

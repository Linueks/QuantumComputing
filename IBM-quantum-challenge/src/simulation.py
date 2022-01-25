"""
Starting with implementation provided by IBM at:
https://github.com/qiskit-community/open-science-prize-2021/blob/main/ibmq-qsim-challenge.ipynb
@linueks
"""
import numpy as np
import mitiq as mt
import time as time
import qiskit as qk
import qiskit.opflow as opflow
import matplotlib.pyplot as plt
import qiskit.ignis.verification.tomography as tomo
from qiskit.quantum_info import state_fidelity
import warnings
warnings.filterwarnings('ignore')
plt.style.use('seaborn-whitegrid')



def xx_subcircuit(
    time,
    quantum_register,
    qubit1,
    qubit2,
):
    """
    Subcircuit used in Trotter decomposition

    Inputs:
        time: qiskit.circuit.Parameter
        quantum_register: qiskit.circuit.QuantumRegister
        qubit1: int
        qubit2: int

    returns:
        quantum_circuit: qiskit.circuit.QuantumCircuit
    """

    # Like Alessandro says this might even be too flexible
    quantum_circuit = qk.QuantumCircuit(quantum_register)
    quantum_circuit.ry(np.pi/2, [qubit1, qubit2])
    quantum_circuit.cnot(qubit1, qubit2)
    quantum_circuit.rz(2*time, qubit2)
    quantum_circuit.cnot(qubit1, qubit2)
    quantum_circuit.ry(-np.pi/2, [qubit1, qubit2])
    quantum_circuit.barrier()

    return quantum_circuit



def yy_subcircuit(
    time,
    quantum_register,
    qubit1,
    qubit2,
):
    """
    Subcircuit used in Trotter decomposition

    Inputs:
        time: qiskit.circuit.Parameter
        quantum_register: qiskit.circuit.QuantumRegister
        qubit1: int
        qubit2: int

    returns:
        quantum_circuit: qiskit.circuit.QuantumCircuit
    """

    quantum_circuit = qk.QuantumCircuit(quantum_register)
    quantum_circuit.rx(np.pi/2, [qubit1, qubit2])
    quantum_circuit.cnot(qubit1, qubit2)
    quantum_circuit.rz(2*time, qubit2)
    quantum_circuit.cnot(qubit1, qubit2)
    quantum_circuit.rx(-np.pi/2, [qubit1, qubit2])
    quantum_circuit.barrier()

    return quantum_circuit



def zz_subcircuit(
    time,
    quantum_register,
    qubit1,
    qubit2,
):
    """
    Subcircuit used in Trotter decomposition

    Inputs:
        time: qiskit.circuit.Parameter
        quantum_register: qiskit.circuit.QuantumRegister
        qubit1: int
        qubit2: int

    returns:
        quantum_circuit: qiskit.circuit.QuantumCircuit
    """

    quantum_circuit = qk.QuantumCircuit(quantum_register)
    quantum_circuit.cnot(qubit1, qubit2)
    quantum_circuit.rz(2*time, qubit2)
    quantum_circuit.cnot(qubit1, qubit2)
    quantum_circuit.barrier()

    return quantum_circuit



def trotter_step_zyxzyx(
    time,
    quantum_register,
    active_qubits,
):
    """
    One Trotter step build out of xx, yy, zz subcircuits. This approach might
    prove worse than just hardcoding everything when it comes time to optimize.
    This is a three qubit operation, thus active qubits needs three positions


    Inputs:
        time: qiskit.circuit.Parameter
        quantum_register: qiskit.circuit.QuantumRegister
        active_qubits: list of qubit positions

    returns:
        quantum_circuit: qiskit.circuit.QuantumCircuit
    """
    qubit1, qubit2, qubit3 = active_qubits
    quantum_circuit = qk.QuantumCircuit(quantum_register)
    quantum_circuit += zz_subcircuit(time, quantum_register, qubit1, qubit2)
    quantum_circuit += yy_subcircuit(time, quantum_register, qubit1, qubit2)
    quantum_circuit += xx_subcircuit(time, quantum_register, qubit1, qubit2)
    quantum_circuit += zz_subcircuit(time, quantum_register, qubit2, qubit3)
    quantum_circuit += yy_subcircuit(time, quantum_register, qubit2, qubit3)
    quantum_circuit += xx_subcircuit(time, quantum_register, qubit2, qubit3)

    return quantum_circuit



def trotter_step_zzyyxx(
    time,
    quantum_register,
    active_qubits,
):
    """
    One Trotter step build out of xx, yy, zz subcircuits. This approach might
    prove worse than just hardcoding everything when it comes time to optimize.
    This is a three qubit operation, thus active qubits needs three positions

    Inputs:
        time: qiskit.circuit.Parameter
        quantum_register: qiskit.circuit.QuantumRegister
        active_qubits: list of qubit positions

    returns:
        quantum_circuit: qiskit.circuit.QuantumCircuit
    """
    qubit1, qubit2, qubit3 = active_qubits
    quantum_circuit = qk.QuantumCircuit(quantum_register)
    quantum_circuit += zz_subcircuit(time, quantum_register, qubit1, qubit2)
    quantum_circuit += zz_subcircuit(time, quantum_register, qubit2, qubit3)
    quantum_circuit += yy_subcircuit(time, quantum_register, qubit1, qubit2)
    quantum_circuit += yy_subcircuit(time, quantum_register, qubit2, qubit3)
    quantum_circuit += xx_subcircuit(time, quantum_register, qubit1, qubit2)
    quantum_circuit += xx_subcircuit(time, quantum_register, qubit2, qubit3)

    return quantum_circuit



def first_cancellations_zzyyxx(
    time,
    quantum_register,
    active_qubits,
):
    """
    Manually doing simplifications to the zzyyxx Trotterization
    """
    qubit1, qubit2, qubit3 = active_qubits
    quantum_circuit = qk.QuantumCircuit(quantum_register)
    # z part ---------------------------
    quantum_circuit.cnot(qubit2, qubit1)
    quantum_circuit.cnot(qubit2, qubit3)
    quantum_circuit.rz(2*time, qubit1)
    quantum_circuit.rz(2*time, qubit3)
    quantum_circuit.cnot(qubit2, qubit3)
    quantum_circuit.cnot(qubit2, qubit1)
    # y part ---------------------------
    quantum_circuit.barrier()
    quantum_circuit.rx(np.pi/2, active_qubits)
    quantum_circuit.cnot(qubit1, qubit2)
    quantum_circuit.rz(2*time, qubit2)
    quantum_circuit.cnot(qubit1, qubit2)
    quantum_circuit.cnot(qubit3, qubit2)
    quantum_circuit.rz(2*time, qubit2)
    quantum_circuit.cnot(qubit3, qubit2)
    quantum_circuit.rx(-np.pi/2, active_qubits)
    # x part ---------------------------
    quantum_circuit.barrier()
    quantum_circuit.ry(np.pi/2, active_qubits)
    quantum_circuit.cnot(qubit1, qubit2)
    quantum_circuit.rz(2*time, qubit2)
    quantum_circuit.cnot(qubit1, qubit2)
    quantum_circuit.cnot(qubit3, qubit2)
    quantum_circuit.rz(2*time, qubit2)
    quantum_circuit.cnot(qubit3, qubit2)
    quantum_circuit.ry(-np.pi/2, active_qubits)

    return quantum_circuit




def build_circuit(
    time,
    trotter_step_function,
    trotter_steps=4,
    target_time=np.pi,
    draw_circuit=False,
    n_qubits=7,
    active_qubits=[1, 3, 5],
    transpile_circuit=True,
):
    """
    Function to add specified amount of Trotter steps to a qiskit circuit and
    set correct simulation for each step given desired target time.

    Input:
        time: qiskit.circuit.Parameter
        trotter_step_function: Python Function returning qiskit.circuit
        trotter_steps: int
        target_time: float
        draw_circuit: bool
        show_all_tomography_circuits: bool
        n_qubits: int
        active_qubits: list of qubit positions

    Returns:
        quantum_circuit: qiskit.circuit.QuantumCircuit
        quantum_register: qiskit.circuit.QuantumRegister
    """
    quantum_register = qk.QuantumRegister(n_qubits)                       # 7 qubits on Jakarta machine. 5 on Belem
    quantum_circuit = qk.QuantumCircuit(quantum_register)

    if n_qubits == 7:
        # yalla fix so I can use the function with other qubit numbers.
        # set up initial state |110>
        quantum_circuit.x([3, 5])                                             # Remember to switch back once access to Jakarta
        quantum_circuit.barrier()

    for step in range(trotter_steps):
        single_trotter_step = trotter_step_function(
            time,
            quantum_register,
            active_qubits,
        )
        quantum_circuit += single_trotter_step
        quantum_circuit.barrier()

    if transpile_circuit:
        quantum_circuit = qk.compiler.transpile(
            quantum_circuit,
            basis_gates=['id', 'u1', 'u2', 'u3', 'cx'],
            optimization_level=3,
        )

    if draw_circuit:
        quantum_circuit.draw(output='mpl')
        plt.tight_layout()
        plt.show()

    #quantum_circuit.measure(measured_qubits, [2, 1, 0])
    quantum_circuit = quantum_circuit.bind_parameters(
        {time: target_time/trotter_steps}
    )

    return quantum_circuit, quantum_register



def generate_tomography_circuits(
    quantum_circuit,
    quantum_register,
    draw_all_tomography_circuits=False,
):
    """
    Generates 27 different quantum circuits to perform complete quantum state
    tomography.

    Inputs:
        quantum_circuit: qiskit.circuit.QuantumCircuit
        quantum_register: qiskit.circuit.QuantumRegister
        draw_all_tomography_circuits: bool

    Returns:
        tomography_circuits: list of qiskit.circuit.QuantumCircuit
    """
    measured_qubits = [
        quantum_register[1],
        quantum_register[3],
        quantum_register[5]
    ]                                                                           # Switch back when going to Jakarta
    tomography_circuits = tomo.state_tomography_circuits(
        quantum_circuit,
        measured_qubits
    )
    if draw_all_tomography_circuits:
        for i in range(len(tomography_circuits)):
            tomography_circuits[i].draw(output='mpl')
            plt.tight_layout()
        plt.show()

    return tomography_circuits



def execute_circuit(
    quantum_circuit,
    backend,
    shots,
):
    """
    Putting this into separate function in case I want to add more arguments
    later when optimizing. Looks a bit pointless as of now.
    """
    job = qk.execute(quantum_circuit, backend, shots=shots)

    return job



def calculate_fidelity(
    jobs,
    tomography_circuits,
    print_result=True,
):
    ket_zero = opflow.Zero
    ket_one = opflow.One
    final_state_target = ket_one^ket_one^ket_zero
    target_state_matrix = final_state_target.to_matrix()

    fidelities = []

    for job in jobs:
        result = job.result()
        tomography_fitter = tomo.StateTomographyFitter(
            result,
            tomography_circuits,
        )
        rho_fit = tomography_fitter.fit(method='lstsq')
        fidelity = qk.quantum_info.state_fidelity(
            rho_fit,
            target_state_matrix
        )
        fidelities.append(fidelity)

    fidelity_mean = np.mean(fidelities)
    fidelity_std = np.std(fidelities)

    if print_result:
        print(f'state tomography fidelity = {fidelity_mean:.4f}',
               '\u00B1', f'{fidelity_std:.4f}')

    return fidelity_mean, fidelity_std



def run_experiments(
    time,
    backend,
    trotter_step_list,
    min_trotter_steps,
    max_trotter_steps,
    target_time=np.pi,
    shots=8192,
    repetitions=8,
    draw_circuit=True,
    n_qubits=7,
    active_qubits=[1,3,5],
):
    """
    Container function to collect and output results for everything interesting
    to calculate. Want to make it run over trotter steps, decompositions,
    repetitions.
    """

    n_decompositions = len(trotter_step_list)

    fid_means = np.zeros(
        shape=(
            n_decompositions,
            max_trotter_steps - min_trotter_steps + 1
        )
    )
    fid_stdevs = np.zeros_like(fid_means)

    for i, decomp in enumerate(trotter_step_list):
        for j, steps in enumerate(range(min_trotter_steps,max_trotter_steps+1)):
            circuit, register = build_circuit(
                time,
                decomp,
                trotter_steps=steps,
                target_time=target_time,
                draw_circuit=draw_circuit,
                n_qubits=n_qubits,
                active_qubits=active_qubits,
            )
            tomography_circuits = generate_tomography_circuits(
                circuit,
                register,
                draw_all_tomography_circuits=False,
            )
            jobs = []
            for k in range(repetitions):
                job = execute_circuit(
                    tomography_circuits,
                    backend,
                    shots,
                )
                print(f'Job ID: {job.job_id()}')
                jobs.append(job)

                qk.tools.monitor.job_monitor(job)
                try:
                    if job.error_message() is not None:
                        print(job.error_message())
                except:
                    pass


            fid_means[i, j], fid_stdevs[i, j] = calculate_fidelity(
                jobs,
                tomography_circuits,
                print_result=True,
            )

    return fid_means, fid_stdevs



if __name__=='__main__':
    provider = qk.IBMQ.load_account()

    provider = qk.IBMQ.get_provider(hub='ibm-q-community', group='ibmquantumawards', project='open-science-22')
    jakarta_backend = provider.get_backend('ibmq_jakarta')
    config = jakarta_backend.configuration()

    #provider = qk.IBMQ.get_provider(hub='ibm-q', group='open', project='main')
    #belem_backend = provider.get_backend('ibmq_belem')                          # has the same topology as Jakarta with qubits 1,3,4 corresponding to 1,3,5
    #config = belem_backend.configuration()

    # set up qiskit simulators
    #sim_noisy_belem = qk.providers.aer.QasmSimulator.from_backend(belem_backend)
    #sim_noiseless_belem = qk.providers.aer.QasmSimulator()
    sim_jakarta_noiseless = qk.providers.aer.QasmSimulator()
    sim_noisy_jakarta = qk.providers.aer.QasmSimulator.from_backend(jakarta_backend)

    ket_zero = opflow.Zero
    ket_one = opflow.One
    initial_state = ket_one^ket_one^ket_zero
    time = qk.circuit.Parameter('t')
    shots = 8192
    trotter_steps = 1                                                           # Variable if just running one simulation
    end_time = np.pi                                                            # Specified in competition
    min_trotter_steps = 4                                                       # 4 minimum for competition
    max_trotter_steps = 16
    num_jobs = 8

    # should group the two lists here to one dictionary probably
    decompositions = [
        trotter_step_zyxzyx,
        trotter_step_zzyyxx,
        first_cancellations_zzyyxx,
    ]
    names = [
        'Trot zyxzyx',
        'Trot zzyyxx',
        'Cancel zzyyxx',
    ]
    linestyles = [
        'solid',
        'dotted',
        'dashed',
        'dashdot'
    ]

    """
    circuit, register = build_circuit(
        time,
        first_cancellations_zzyyxx,
        trotter_steps=trotter_steps,
        target_time=end_time,
        draw_circuit=True
    )
    """


    #"""
    active_qubits = [1, 3, 5]
    fid_means, fid_stdevs = run_experiments(
        time,
        sim_noisy_jakarta,
        decompositions,
        min_trotter_steps,
        max_trotter_steps,
        target_time=end_time,
        shots=shots,
        repetitions=num_jobs,
        draw_circuit=False,
        n_qubits=7,
        active_qubits=active_qubits,
    )
    np.save(f'../data/fidelities_mean_{min_trotter_steps}_{max_trotter_steps}_shots{shots}', fid_means)
    np.save(f'../data/fidelities_std_{min_trotter_steps}_{max_trotter_steps}_shots{shots}', fid_stdevs)
    #"""

    #fid_means = np.load(f'../data/fidelities_mean_{min_trotter_steps}_{max_trotter_steps}_shots{shots}.npy')
    #fid_stdevs = np.load(f'../data/fidelities_std_{min_trotter_steps}_{max_trotter_steps}_shots{shots}.npy')

    x_axis = range(min_trotter_steps, max_trotter_steps+1)

    for i, name in enumerate(names):
        eb1 = plt.errorbar(
            x_axis, fid_means[i, :],
            yerr=fid_stdevs[i, :],
            errorevery=1,
            label=name,
            ls=linestyles[i],
            capsize=5
        )

    plt.xlabel('Trotter Steps')
    plt.ylabel('Fidelity')
    plt.title(f'Trotter Simulation with {shots} Shots, {num_jobs} Jobs, Backend: {config.backend_name}')
    plt.legend()
    plt.savefig(f'../figures/trotter_sim_{min_trotter_steps}_{max_trotter_steps}_shots{shots}_numjobs{num_jobs}')
    plt.show()
    #"""
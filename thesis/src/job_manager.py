"""
Starting with implementation provided by IBM at:
https://github.com/qiskit-community/open-science-prize-2021/blob/main/ibmq-qsim-challenge.ipynb
@linueks
"""
import copy
import numpy as np
import time as time
import qiskit as qk
import qiskit.opflow as opflow
import matplotlib.pyplot as plt
from qiskit.quantum_info import state_fidelity
import qiskit.ignis.verification.tomography as tomo
import qiskit.ignis.mitigation.measurement as meas
from qiskit_aer.noise import NoiseModel
from decompositions import *
from simulation import TrotterSimulation
from qiskit.providers.ibmq.managed import IBMQJobManager
from qiskit.providers.ibmq import BackendJobLimit


import warnings
warnings.filterwarnings('ignore')
plt.style.use('seaborn-whitegrid')


if __name__=='__main__':
    provider = qk.IBMQ.load_account()
    #provider = qk.IBMQ.get_provider(
    #    hub='ibm-q-community',
    #    group='ibmquantumawards',
    #    project='open-science-22'
    #)
    provider = qk.IBMQ.get_provider(
        hub='ibm-q',
        group='open',
        project='main',
    )
    #nairobi_backend = provider.get_backend('ibm_nairobi')
    #config = nairobi_backend.configuration()
    #print(config.basis_gates)
    #print(config.max_experiments)
    oslo_backend = provider.get_backend('ibm_oslo')


    ket_zero = opflow.Zero
    ket_one = opflow.One
    initial_state = ket_one^ket_one^ket_zero
    qk_time = qk.circuit.Parameter('t')
    shots = 8192
    n_qubits = 7
    end_time = np.pi                                                            # Specified in competition
    min_trotter_steps = 4                                                       # 4 minimum for competition
    max_trotter_steps = 20
    trotter_steps = range(min_trotter_steps, max_trotter_steps+1)               # Variable if just running one simulation
    basis_gates=['id', 'rz', 'sx', 'x', 'cx', 'reset']
    active_qubits = [1, 3, 5]

    verbose = True

    decompositions = {
        'zyxzyx': trotter_step_zyxzyx,
        'zyxyzzyxyz': trotter_second_zyxyzzyxyz,
        'zzyyxxyyzz': trotter_second_zzyyxxyyzz,
        'zzyyxx': trotter_step_zzyyxx,
        'x+yzzx+y': trotter_step_xplusy_zz_xplusy,
        'x+yzx+yz': trotter_step_xplusy_z_xplusy_z,
        'x+y+z': trotter_step_xplusyplusz_xplusyplusz,
        'n_xyz': trotter_step_n_xyz,
    }

    def create_simulation(backend):
        simulator = TrotterSimulation(
            simulation_parameter=qk_time,
            simulation_backend=backend,
            backend_default_gates=basis_gates,
            simulation_end_time=end_time,
            number_of_qubits=n_qubits,
            shots=shots,
            active_qubits=active_qubits,
            verbose=verbose,
        )
        return simulator

    # first generate all the base circuits
    # I want to run with and without symmetry protection and with 0 and 3 transpilation level
    simulator = create_simulation(oslo_backend)
    base_circuit_list = []
    simulator.set_readout_error_mitigation(False)
    #"""
    # this means that for every 17th entry in the list there is a new decomposition
    # and in this first pass there will be 17 * 8 = 136 circuits, 1.36jobs
    simulator.set_symmetry_protection(False, False)
    simulator.set_transpilation_level(0)
    for i, (name, decomposition) in enumerate(decompositions.items()):
        for j, steps in enumerate(trotter_steps):
            circuit = simulator.make_base_circuit(
                steps,
                decomposition,
                name,
            )
            base_circuit_list.append(circuit)

    simulator.set_symmetry_protection(True, symmetry_protection_su_2)
    simulator.set_transpilation_level(0)
    # this means that from the 137th element we will have symmetry protection
    # and we have now 2 * 17 * 8 = 272 circuits, 2.72 jobs
    for i, (name, decomposition) in enumerate(decompositions.items()):
        for j, steps in enumerate(trotter_steps):
            circuit = simulator.make_base_circuit(
                steps,
                decomposition,
                name,
            )
            base_circuit_list.append(circuit)
    simulator.set_symmetry_protection(False, False)
    simulator.set_transpilation_level(3)
    # now after the 273rd circuit we have changed to transpiling
    # and we have 3 * 17 * 8 = 408 circuits, 4.08 jobs
    for i, (name, decomposition) in enumerate(decompositions.items()):
        for j, steps in enumerate(trotter_steps):
            circuit = simulator.make_base_circuit(
                steps,
                decomposition,
                name,
            )
            base_circuit_list.append(circuit)

    simulator.set_symmetry_protection(True, symmetry_protection_su_2)
    simulator.set_transpilation_level(3)
    # this means that from the 409th element we will have symmetry protection and transpilation
    # and we have now 4 * 17 * 8 = 544 circuits, 5.44 jobs
    for i, (name, decomposition) in enumerate(decompositions.items()):
        for j, steps in enumerate(trotter_steps):
            circuit = simulator.make_base_circuit(
                steps,
                decomposition,
                name,
            )
            base_circuit_list.append(circuit)

    #print(len(base_circuit_list)) # out: 544
    # for each of these circuits we need 27 to do the state tomography
    # we generate the tomography circuits and end up with 544 * 27 = 14688
    # circuits or 146.88 jobs. Next we use the qiskit job manager to execute these

    tomography_circuit_list = []




    #"""
    for circuit in base_circuit_list:
        tomography_circuits = simulator.make_tomography_circuits(circuit)
        tomography_circuit_list.append(tomography_circuits)

    """
    #job_manager = IBMQJobManager()
    #job_set = job_manager.run(tomography_circuit_list, backend=oslo_backend, name='hope this works')
    """
    job_count = 0
    while job_count < 544:
        print(job_count)
        if len(oslo_backend.active_jobs()) < 5:
            job = simulator.execute_circuit(tomography_circuit_list[job_count])
            with open('job_ids.txt', 'a') as outfile:
                outfile.write(f'{job.job_id()}\n')

            job_count += 1
        else:
            continue

        if job_count < 5:
            time.sleep(10)
        else:
            time.sleep(45) #sleep 45 seconds




    """
    fidelities = np.zeros(shape=(len(decompositions), len(trotter_steps)))
    for i, (name, decomposition) in enumerate(decompositions.items()):
        for j, steps in enumerate(trotter_steps):
            fidelity = simulator.run(
                name,
                decomposition,
                steps,
                draw_circuits='none',
            )
            fidelities[i, j] = fidelity

    """

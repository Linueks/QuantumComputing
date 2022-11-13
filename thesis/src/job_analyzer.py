import numpy as np
import time as time
import qiskit as qk
import qiskit.opflow as opflow
import matplotlib.pyplot as plt
import qiskit.ignis.verification.tomography as tomo
from qiskit.quantum_info import state_fidelity
from decompositions import *
from simulation import TrotterSimulation
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

    simulator = TrotterSimulation(
        simulation_parameter=qk_time,
        simulation_backend=oslo_backend,
        backend_default_gates=basis_gates,
        simulation_end_time=end_time,
        number_of_qubits=n_qubits,
        shots=shots,
        active_qubits=active_qubits,
        verbose=verbose
    )


    # need this ordering because of how the job_manager script is written
    # for each transpilation level, symmetry protection setting, decomposition
    # and trotter step I need to generate the corresponding tomography circuits
    # and calculate the fidelity to the target state. This should be saved in
    # the same formatting as the classically simulated results
    job_count = 0
    with open('../data/job_ids.txt', 'r') as infile:
        lines = infile.read().splitlines()
        for transpilation_level in [0, 3]:
            simulator.set_transpilation_level(transpilation_level)
            for sp in [False, True]:
                simulator.set_symmetry_protection(
                    sp,
                    symmetry_protection_su_2,
                )
                for i, (name, decomposition) in enumerate(decompositions.items()):
                    fidelities = np.zeros(shape=(len(trotter_steps), 2))
                    for j, steps in enumerate(trotter_steps):
                        circuit = simulator.make_base_circuit(
                            steps,
                            decomposition,
                            name,
                        )
                        tomography_circuits = simulator.make_tomography_circuits(
                            circuit
                        )
                        # Need to do the readout_mitigation before the fidelity calc
                        # but cannot increase the job count, because I need
                        # to do the False True for each job
                        # 544 jobs should be 32 fidelity arrays and then
                        # double it because of the readout mitigation True False
                        # so half the amount of data as the classically simulated
                        # because I cut out transpilation levels 1, 2. Makes sense.
                        for k, readout_mitigation in enumerate([False, True]):
                            simulator.set_readout_error_mitigation(readout_mitigation)
                            job_id = lines[job_count]
                            print(job_id)
                            q_job = oslo_backend.retrieve_job(
                                job_id
                            )
                            fidelity = simulator.calculate_fidelity(
                                q_job,
                                tomography_circuits,
                            )
                            fidelities[j, k] = fidelity

                        job_count += 1
                        print(job_count)

                    for b, readout_mitigation in enumerate([False, True]):
                        np.save(f'../data/final_runs/fidelities_{oslo_backend}_{name}_SP_{sp}_mitigated_{readout_mitigation}_transpilation_level_{transpilation_level}', fidelities[:, b])

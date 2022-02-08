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
import qiskit.ignis.verification.tomography as tomo
from qiskit.quantum_info import state_fidelity
from decompositions import *
import warnings
warnings.filterwarnings('ignore')
plt.style.use('seaborn-whitegrid')



class TrotterSimulation:
    def __init__(
        self,
        decompositions,
        simulation_parameter,
        simulation_backend,
        backend_default_gates,
        simulation_end_time,
        number_of_qubits,
        shots,
        number_of_jobs,
        active_qubits,
        verbose,
    ):
        self.decompositions = decompositions
        self.n_decompositions = len(decompositions)
        self.param = simulation_parameter
        self.backend = simulation_backend
        self.default_gates = backend_default_gates
        self.end_time = simulation_end_time
        self.n_qubits = number_of_qubits
        self.shots = shots
        self.active_qubits = active_qubits
        self.jobs = []
        self.repetitions = number_of_jobs
        self.verbose = verbose


    def set_transpilation_level(
        self,
        level
    ):
        self.transpilation_level = level


    def set_symmetry_protection(
        self,
        symmetry_protection_function,
    ):
        self.symmetry_protection = symmetry_protection_function


    def delete_circuits(
        self,
    ):
        del(self.quantum_register)
        del(self.quantum_circuit)
        del(self.tomography_circuits)

    def make_base_circuit(
        self,
        trotter_steps,
        trotter_step_function,
        name,
    ):
        #quantum_register = 0
        self.quantum_register = qk.QuantumRegister(
            self.n_qubits,
            name=name,
        )
        self.quantum_circuit = qk.QuantumCircuit(
            self.quantum_register,
            name=name,
        )
        # To make initial state as specified in the competition |110>
        self.quantum_circuit.x([self.active_qubits[1], self.active_qubits[2]])
        self.quantum_circuit.barrier()

        for step in range(1, trotter_steps+1):
            if symmetry_protection and step%2 == 1:
                self.quantum_circuit += symmetry_protection_function(
                    self.quantum_register,
                    self.active_qubits,
                )
                self.quantum_circuit += trotter_step_function(
                    self.param,
                    self.quantum_register,
                    self.active_qubits,
                )
                self.quantum_circuit += symmetry_protection_function(
                    self.quantum_register,
                    self.active_qubits,
                )

            else:
                self.quantum_circuit += trotter_step_function(
                    self.param,
                    self.quantum_register,
                    self.active_qubits,
                )
            self.quantum_circuit.barrier()

        self.quantum_circuit = qk.compiler.transpile(
            self.quantum_circuit,
            basis_gates=self.default_gates,
            optimization_level=self.transpilation_level,
        )
        self.quantum_circuit = self.quantum_circuit.bind_parameters(
            {time: self.end_time/trotter_steps}
        )


    def make_tomography_circuits(
        self,
    ):
        measured_qubits = self.active_qubits
        self.tomography_circuits = tomo.state_tomography_circuits(
            self.quantum_circuit,
            measured_qubits,
        )


    def draw_circuit(
        self,
        base_circuit,
        tomography_circuits,
    ):
        if base_circuit:
            self.quantum_circuit.draw(output='mpl')
            plt.tight_layout()
            plt.show()

        if tomography_circuits:
            for i in range(len(self.tomography_circuits)):
                self.tomography_circuits[i].draw(output='mpl')
                plt.tight_layout()
            plt.show()


    def draw_all_trotter_steps(
        self,
        decompositions,
    ):



    def execute_circuit(
        self,
    ):
        return qk.execute(
            self.tomography_circuits,
            self.backend,
            shots=self.shots,
        )


    def calculate_fidelity(
        self,
        jobs,
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
                self.tomography_circuits,
            )
            rho_fit = tomography_fitter.fit(
                method='lstsq',
            )
            fidelity = qk.quantum_info.state_fidelity(
                rho_fit,
                target_state_matrix,
            )
            fidelities.append(fidelity)

        #del(self.jobs)
        self.fidelity_mean = np.mean(fidelities)
        self.fidelity_stdev = np.std(fidelities)

        if self.verbose:
            print(
                f'state tomography fidelity = {self.fidelity_mean:.4f}',
                '\u00B1',
                f'{self.fidelity_stdev:.4f}')


    def run(
        self,
    ):
        fidelity_means = np.zeros(
            shape=(
                self.n_decompositions,
                max_trotter_steps - min_trotter_steps + 1
            )
        )
        fidelity_stdev = np.zeros_like(fidelity_means)
        trotter_steps = range(min_trotter_steps, max_trotter_steps+1)

        for i, (name, decomposition) in enumerate(self.decompositions.items()):
            if self.verbose:
                print(f'Decomposition: {name}')
            for j, steps in enumerate(trotter_steps):
                if self.verbose:
                    print(f'Trotter Step: {steps}')

                self.make_base_circuit(
                    steps,
                    decomposition,
                    name,
                )

                self.make_tomography_circuits()

                self.draw_circuit(
                    base_circuit=False,
                    tomography_circuits=False,
                )


                jobs = []
                for k in range(self.repetitions):
                    job = self.execute_circuit()
                    jobs.append(job)

                    if self.verbose:
                        print(f'Job ID: {job.job_id()}')
                        qk.tools.monitor.job_monitor(job)

                    try:
                        if job.error_message() is not None:
                            print(job.error_message())
                    except:
                        pass

                self.calculate_fidelity(jobs)
                fidelity_means[i, j] = self.fidelity_mean
                fidelity_stdev[i, j] = self.fidelity_stdev
                self.delete_circuits()


            if self.verbose:
                print('-------------------------------------------------------')

        return fidelity_means, fidelity_stdev



if __name__=='__main__':
    provider = qk.IBMQ.load_account()

    provider = qk.IBMQ.get_provider(
        hub='ibm-q-community',
        group='ibmquantumawards',
        project='open-science-22'
    )
    jakarta_backend = provider.get_backend('ibmq_jakarta')
    config = jakarta_backend.configuration()

    # set up qiskit simulators
    sim_jakarta_noiseless = qk.providers.aer.QasmSimulator()
    sim_noisy_jakarta = qk.providers.aer.QasmSimulator.from_backend(jakarta_backend)

    ket_zero = opflow.Zero
    ket_one = opflow.One
    initial_state = ket_one^ket_one^ket_zero
    time = qk.circuit.Parameter('t')
    shots = 8192
    n_qubits = 7
    end_time = np.pi                                                            # Specified in competition
    min_trotter_steps = 4                                                       # 4 minimum for competition
    max_trotter_steps = 4
    trotter_steps = 1                                                           # Variable if just running one simulation
    num_jobs = 2
    symmetry_protection = False
    basis_gates=['id', 'rz', 'sx', 'x', 'cx', 'reset']
    active_qubits = [1, 3, 5]
    verbose = True
    transpilation_level = 0

    decompositions = {
        #'zyxzyx': trotter_step_zyxzyx,
        #'zzyyxx': trotter_step_zzyyxx,
        'x+yz': trotter_step_partial_dot_product,
        'x+y+z': full_sum_optimal_construction,
    }

    simulator = TrotterSimulation(
        decompositions,
        time,
        sim_noisy_jakarta,
        basis_gates,
        end_time,
        n_qubits,
        shots,
        num_jobs,
        active_qubits,
        verbose,
    )
    simulator.set_transpilation_level(transpilation_level)
    simulator.set_symmetry_protection(symmetry_protection_su_2)
    fid_means, fid_stdevs = simulator.run()

    #np.save(f'../data/fidelities_mean_{min_trotter_steps}_{max_trotter_steps}_shots{shots}_SP{repr(symmetry_protection)}_{len(decompositions)}', fid_means)
    #np.save(f'../data/fidelities_std_{min_trotter_steps}_{max_trotter_steps}_shots{shots}_SP{repr(symmetry_protection)}_{len(decompositions)}', fid_stdevs)
    #"""

    #fid_means = np.load(f'../data/fidelities_mean_{min_trotter_steps}_{max_trotter_steps}_shots{shots}_SP{repr(symmetry_protection)}_{len(decompositions)}.npy')
    #fid_stdevs = np.load(f'../data/fidelities_std_{min_trotter_steps}_{max_trotter_steps}_shots{shots}_SP{repr(symmetry_protection)}_{len(decompositions)}.npy')

    linestyles = [
        'solid',
        'dotted',
        'dashed',
        'dashdot'
    ]

    x_axis = range(min_trotter_steps, max_trotter_steps+1)

    for i, (name, decomposition) in enumerate(decompositions.items()):
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
    #plt.savefig(f'../figures/trotter_sim_{min_trotter_steps}_{max_trotter_steps}_shots{shots}_numjobs{num_jobs}_SP{repr(symmetry_protection)}_{len(decompositions)}')
    plt.show()
    #"""

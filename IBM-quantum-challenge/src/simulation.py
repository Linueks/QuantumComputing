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
        repetitions_per_circuit,
        active_qubits,
        verbose,
        draw_circuit,
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
        self.repetitions = repetitions_per_circuit
        self.verbose = verbose
        self.draw_circuit = draw_circuit


    def set_transpilation_level(
        self,
        level
    ):
        self.transpilation_level = level


    def set_symmetry_protection(
        self,
        symmetry_protection,
        symmetry_protection_function,
    ):
        self.symmetry_protection = symmetry_protection
        self.symmetry_protection_function = symmetry_protection_function


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
        quantum_register = qk.QuantumRegister(
            self.n_qubits,
            name=name,
        )
        quantum_circuit = qk.QuantumCircuit(
            quantum_register,
            name=name,
        )
        # To make initial state as specified in the competition |110>
        quantum_circuit.x([self.active_qubits[1], self.active_qubits[2]])
        quantum_circuit.barrier()

        for step in range(1, trotter_steps+1):
            if self.symmetry_protection and step%2 == 1:
                quantum_circuit += self.symmetry_protection_function(
                    quantum_register,
                    self.active_qubits,
                )
                quantum_circuit += trotter_step_function(
                    self.param,
                    quantum_register,
                    self.active_qubits,
                )
                quantum_circuit += self.symmetry_protection_function(
                    quantum_register,
                    self.active_qubits,
                )

            else:
                quantum_circuit += trotter_step_function(
                    self.param,
                    quantum_register,
                    self.active_qubits,
                )

        if type(self.transpilation_level) == int:#"""
            quantum_circuit = qk.compiler.transpile(
                quantum_circuit,
                basis_gates=self.default_gates,
                optimization_level=self.transpilation_level,
            )
        quantum_circuit = quantum_circuit.bind_parameters(
            {time: self.end_time/trotter_steps}
        )

        return quantum_circuit


    def make_tomography_circuits(
        self,
        quantum_circuit,
    ):
        measured_qubits = self.active_qubits
        tomography_circuits = tomo.state_tomography_circuits(
            quantum_circuit,
            measured_qubits,
        )

        return tomography_circuits



    def circuit_drawer(
        self,
        circuits,
    ):
        if type(circuits) == qk.circuit.quantumcircuit.QuantumCircuit:
            circuits.draw(output='mpl')
            plt.tight_layout()
            plt.show()

        else:
            for circuit in tomography_circuits:
                circuit.draw(output='mpl')
                plt.tight_layout()
            plt.show()



    def execute_circuit(
        self,
        circuits,
    ):
        return qk.execute(
            circuits,
            self.backend,
            shots=self.shots,
        )



    def calculate_fidelity(
        self,
        job,
        tomography_circuits,
    ):
        ket_zero = opflow.Zero
        ket_one = opflow.One
        final_state_target = ket_one^ket_one^ket_zero
        target_state_matrix = final_state_target.to_matrix()
        fidelities = []

        result = job.result()

        tomography_fitter = tomo.StateTomographyFitter(
            result,
            tomography_circuits,
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
        fidelity_mean = np.mean(fidelities)
        fidelity_stdev = np.std(fidelities)

        print(fidelity_mean)
        print(fidelity_stdev)

        if self.verbose:
            print(
                f'state tomography fidelity = {fidelity_mean:.4f}',
                '\u00B1',
                f'{fidelity_stdev:.4f}')

        return fidelity_mean, fidelity_stdev



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

        tomography_circuits = []

        for i, (name, decomposition) in enumerate(self.decompositions.items()):
            if self.verbose:
                print(f'Decomposition: {name}')
            for j, steps in enumerate(trotter_steps):
                print(steps)
                if self.verbose:
                    print(f'Trotter Step: {steps}')

                base_circuit = self.make_base_circuit(
                    steps,
                    decomposition,
                    name,
                )

                tomography_circuits = self.make_tomography_circuits(
                    base_circuit,
                )

                if self.draw_circuit:
                    self.circuit_drawer(
                        base_circuit=True,
                        tomography_circuits=False,
                    )

        job = self.execute_circuit(
            tomography_circuits,
        )
        job_id = job.job_id()
        if self.verbose:
            print(f'Job ID: {job.job_id()}')
            qk.tools.monitor.job_monitor(job)

        fidelity_mean, fidelity_stdev = self.calculate_fidelity(
            job,
            tomography_circuits,
        )

        if self.verbose:
            print('-------------------------------------------------------')

        return fidelity_mean, fidelity_stdev



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
    sim_jakarta_noisy = qk.providers.aer.QasmSimulator.from_backend(
        jakarta_backend
    )

    ket_zero = opflow.Zero
    ket_one = opflow.One
    initial_state = ket_one^ket_one^ket_zero
    time = qk.circuit.Parameter('t')
    shots = 8192
    n_qubits = 7
    end_time = np.pi                                                            # Specified in competition
    min_trotter_steps = 8                                                       # 4 minimum for competition
    max_trotter_steps = 8
    trotter_steps = 1                                                           # Variable if just running one simulation
    repetitions = 1
    basis_gates=['id', 'rz', 'sx', 'x', 'cx', 'reset']
    active_qubits = [1, 3, 5]


    transpilation_level = 3
    noisy_simulation = True
    symmetry_protection = False
    verbose = True
    draw_circuit = False

    decompositions = {
        #'zyxzyx': trotter_step_zyxzyx,
        #'zzyyxx': trotter_step_zzyyxx,
        #'x+yzzx+y': trotter_step_xplusy_zz_xplusy,
        #'x+yzx+yz': trotter_step_xplusy_z_xplusy_z,
        'x+y+z': trotter_step_xplusyplusz_xplusyplusz,
    }

    test_decompositions = {
        'canceltest1': cancellation_test_1,
    }


    if noisy_simulation:
        simulator = TrotterSimulation(
            decompositions=decompositions,
            simulation_parameter=time,
            simulation_backend=jakarta_backend,
            backend_default_gates=basis_gates,
            simulation_end_time=end_time,
            number_of_qubits=n_qubits,
            shots=shots,
            repetitions_per_circuit=repetitions,
            active_qubits=active_qubits,
            verbose=verbose,
            draw_circuit=draw_circuit,
        )
    else:
        simulator = TrotterSimulation(
            decompositions=decompositions,
            simulation_parameter=time,
            simulation_backend=sim_jakarta_noiseless,
            backend_default_gates=basis_gates,
            simulation_end_time=end_time,
            number_of_qubits=n_qubits,
            shots=shots,
            repetitions_per_circuit=repetitions,
            active_qubits=active_qubits,
            verbose=verbose,
            draw_circuit=draw_circuit,
        )

    simulator.set_transpilation_level(transpilation_level)

    simulator.set_symmetry_protection(
        symmetry_protection,
        symmetry_protection_su_2,
    )

    fid_means, fid_stdevs = simulator.run()

    #np.save(f'../data/fidelities_mean_{min_trotter_steps}_{max_trotter_steps}_shots{shots}_SP{repr(symmetry_protection)}_{len(decompositions)}', fid_means)
    #np.save(f'../data/fidelities_std_{min_trotter_steps}_{max_trotter_steps}_shots{shots}_SP{repr(symmetry_protection)}_{len(decompositions)}', fid_stdevs)
    #"""

    #fid_means = np.load(f'../data/fidelities_mean_{min_trotter_steps}_{max_trotter_steps}_shots{shots}_SP{repr(symmetry_protection)}_{len(decompositions)}.npy')
    #fid_stdevs = np.load(f'../data/fidelities_std_{min_trotter_steps}_{max_trotter_steps}_shots{shots}_SP{repr(symmetry_protection)}_{len(decompositions)}.npy')

    linestyle_tuple = [
        ('dashed',                (0, (5, 5))),
        ('dashdotted',            (0, (3, 5, 1, 5))),
        ('dotted',                (0, (1, 1))),
        ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),

        ('densely dashed',        (0, (5, 1))),
        ('densely dashdotted',    (0, (3, 1, 1, 1))),
        ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1))),
        ('densely dotted',        (0, (1, 1))),


        ('loosely dashed',        (0, (5, 10))),
        ('loosely dashdotted',    (0, (3, 10, 1, 10))),
        ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
        ('loosely dotted',        (0, (1, 10))),
    ]

    x_axis = range(min_trotter_steps, max_trotter_steps+1)

    for i, (name, decomposition) in enumerate(decompositions.items()):
        eb1 = plt.errorbar(
            x_axis, fid_means[i, :],
            yerr=fid_stdevs[i, :],
            errorevery=1,
            label=name,
            ls=linestyle_tuple[i][1],
            capsize=5
        )

    plt.xlabel('Trotter Steps')
    plt.ylabel('Fidelity')
    plt.legend()
    if noisy_simulation:
        plt.title(f'Noisy Trotter Simulation with {shots} Shots, {num_jobs} Jobs, \n Backend: {config.backend_name}, SP: {repr(symmetry_protection)}')
        plt.savefig(f'../figures/Noisy_trotter_sim_{min_trotter_steps}_{max_trotter_steps}_shots{shots}_numjobs{num_jobs}_SP{repr(symmetry_protection)}_{len(decompositions)}')
    else:
        plt.title(f'Noiseless Trotter Simulation with {shots} Shots, {num_jobs} Jobs, \n Backend: {config.backend_name}, SP: {repr(symmetry_protection)}')
        plt.savefig(f'../figures/Noiseless_trotter_sim_{min_trotter_steps}_{max_trotter_steps}_shots{shots}_numjobs{num_jobs}_SP{repr(symmetry_protection)}_{len(decompositions)}')


    plt.show()
    #"""

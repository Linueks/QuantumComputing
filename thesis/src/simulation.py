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

import warnings
warnings.filterwarnings('ignore')
plt.style.use('seaborn-whitegrid')



class TrotterSimulation:
    def __init__(
        self,
        simulation_parameter,
        simulation_backend,
        backend_default_gates,
        simulation_end_time,
        number_of_qubits,
        shots,
        active_qubits,
        verbose,
    ):
        self.param = simulation_parameter
        self.backend = simulation_backend
        self.default_gates = backend_default_gates
        self.end_time = simulation_end_time
        self.n_qubits = number_of_qubits
        self.shots = shots
        self.active_qubits = active_qubits
        self.jobs = []
        self.verbose = verbose
        if self.verbose:
            print(f'simulation backend: {self.backend}')



    def set_transpilation_level(
        self,
        level
    ):
        """
        Utility function to set transpilation level

        Input:
            level: int (between 0 and 3)
        """
        self.transpilation_level = level

        if self.verbose:
            print(f'Transpilation level: {self.transpilation_level}')


    def set_symmetry_protection(
        self,
        symmetry_protection,
        symmetry_protection_function,
    ):
        """
        Utility function to enable symmetry protection with functionality of
        being able to implement other SP functions later down the line.
        """
        self.symmetry_protection = symmetry_protection
        self.symmetry_protection_function = symmetry_protection_function

        if self.verbose:
            print(f'Symmetry Protection {self.symmetry_protection}')



    def set_readout_error_mitigation(
        self,
        readout_mitigation,
    ):
        self.readout_error_mitigation = readout_mitigation

        if self.verbose:
            print(f'Readout error mitigation {self.readout_error_mitigation}')



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
        """
        Function to set up base circuit for a given decomposition which is used
        to generate tomography circuits. It takes a trotter step function which
        returns a qiskit.circuit.QuantumCircuit and adds this to a circuit
        trotter_steps number of times.

        Inputs:
            trotter_steps: int
            trotter_step_function: function
            name: string

        Returns:
            quantum_circuit: qiskit.circuit.QuantumCircuit
        """

        if self.verbose:
            print(f'Building base circuit with {repr(trotter_step_function)} and {trotter_steps} Trotter steps')

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
            {self.param: self.end_time/trotter_steps}
        )

        return quantum_circuit


    def make_tomography_circuits(
        self,
        quantum_circuit,
    ):
        """
        Generates 27 different quantum circuits to perform complete quantum
        state tomography.

        Inputs:
            quantum_circuit: qiskit.circuit.QuantumCircuit

        Returns:
            tomography_circuits: list of qiskit.circuit.QuantumCircuit
        """
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
        """
        Helper function to either draw the base circuit of the decomposition or
        all 27 tomography circuits. Mostly used for debugging.

        Inputs:
            circuits: qiskit.circuit.QuantumCircuit or list of circuits
        """
        if type(circuits) == qk.circuit.quantumcircuit.QuantumCircuit:
            circuits.draw(output='mpl')
            plt.tight_layout()
            plt.show()

        else:
            for circuit in circuits:
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



    def scale_circuit(
        self,
        circuit,
        scale_factor=1,
    ):
        """
        Method to scale the quantum circuit by the unitary folding method
        presented in LaRose et al. ZNE paper.

        Inputs:
            circuit: qiskit.circuit.QuantumCircuit
                (should maybe be the tomography circuits
                thinking to try both versions.. first just the main circuit)
            noise_scale_factor: int

        Returns:
            circuit: qiskit.circuit.QuantumCircuit
        """

        inverse_circuit = circuit.inverse()
        scaled_circuit = circuit
        for k in range(scale_factor):
            scaled_circuit += inverse_circuit + circuit



    def calculate_fidelity(
        self,
        job,
        tomography_circuits,
    ):
        """
        Calculates the state tomography fidelity given a qiskit job and the
        corresponding tomography circuits for the job.

        Inputs:
            job: qiskit job object
            tomography_circuits: list of qiskit circuits

        Returns:
            fidelity: float
        """
        ket_zero = opflow.Zero
        ket_one = opflow.One
        final_state_target = ket_one^ket_one^ket_zero
        target_state_matrix = final_state_target.to_matrix()
        fidelities = []

        result = job.result()

        if self.readout_error_mitigation:
            """
            if self.verbose:
                print('Readout error mitigation needs noisy backend.')
            """
            # hacky fix to make readout mitigation with real device data work
            provider = qk.IBMQ.get_provider(
                hub='ibm-q',
                group='open',
                project='main',
            )
            oslo_backend = provider.get_backend('ibm_nairobi')
            # this takes the place of the self.backend, which will cause the code
            # to wait for the readout mitigation circuits to run on the real
            # device. This code needs to be updated to do this automatically.
            sim_oslo_noisy = qk.providers.aer.QasmSimulator.from_backend(
                oslo_backend
            )

            quantum_register = qk.QuantumRegister(3)
            measurement_calibrations, state_labels = meas.complete_meas_cal(
                qr=quantum_register,
                circlabel='mcal',
            )
            noise_model = NoiseModel.from_backend(
                self.backend,
            )
            t_qc = qk.transpile(measurement_calibrations, sim_oslo_noisy)#self.backend)
            qobj = qk.assemble(t_qc)
            if self.verbose:
                print('Readout error calibration starting.')
            #calibration_results = self.backend.run(
            #    qobj,
            #    shots=self.shots
            #).result()
            calibration_results = sim_oslo_noisy.run(
                qobj,
                shots=self.shots,
            ).result()
            measurement_fitter = meas.CompleteMeasFitter(
                calibration_results,
                state_labels,
                circlabel='mcal',
            )
            measurement_filter = measurement_fitter.filter
            result = measurement_filter.apply(result)

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

        if self.verbose:
            print(f'state tomography fidelity = {fidelity:.4f}')

        return fidelity



    def run(
        self,
        name,
        decomposition,
        trotter_steps,
        draw_circuits,
    ):
        """
        Builds base circuits and tomography circuits given a decomposition and
        number of trotter steps. Then uses the circuits to calculate the state
        tomography fidelity and returns this.

        Inputs:
            name: string
            decomposition: python function
            trotter_steps: int

        Returns:
            fidelity: float
        """

        base_circuit = self.make_base_circuit(
            trotter_steps,
            decomposition,
            name,
        )

        tomography_circuits = self.make_tomography_circuits(
            base_circuit,
        )

        if draw_circuits == 'base':
            self.circuit_drawer(base_circuit)
        elif draw_circuits == 'tomo':
            self.circuit_drawer(base_circuit)

        job = self.execute_circuit(
            tomography_circuits,
        )
        if self.verbose:
            job_id = job.job_id()
            print(f'Job ID: {job.job_id()}')
            qk.tools.monitor.job_monitor(job)

        fidelity = self.calculate_fidelity(
            job,
            tomography_circuits,
        )

        if self.verbose:
            print('-----------------------------------------------------------')

        return fidelity



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
    nairobi_backend = provider.get_backend('ibm_nairobi')
    oslo_backend = provider.get_backend('ibm_oslo')
    config = nairobi_backend.configuration()
    #print(config.basis_gates)
    #print(config.max_experiments)

    # set up qiskit simulators
    sim_nairobi_noiseless = qk.providers.aer.QasmSimulator()
    sim_nairobi_noisy = qk.providers.aer.QasmSimulator.from_backend(
        nairobi_backend
    )
    sim_oslo_noisy = qk.providers.aer.QasmSimulator.from_backend(
        oslo_backend
    )
    print(sim_nairobi_noiseless)

    # statevector simulator
    statevector_simulator = qk.Aer.get_backend(
        'statevector_simulator',
        device='GPU',
    )
    #print(statevector_simulator)

    ket_zero = opflow.Zero
    ket_one = opflow.One
    initial_state = ket_one^ket_one^ket_zero
    time = qk.circuit.Parameter('t')
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
    backends = [
        statevector_simulator,
        #sim_nairobi_noiseless,
        #sim_nairobi_noisy,
        #sim_oslo_noisy
    ]
    symmetry_protection = [
        False,
        True,
    ]
    readout_error_mitigation = [
        False,
        True,
    ]

    def create_simulation(backend):
        simulator = TrotterSimulation(
            simulation_parameter=time,
            simulation_backend=backend,
            backend_default_gates=basis_gates,
            simulation_end_time=end_time,
            number_of_qubits=n_qubits,
            shots=shots,
            active_qubits=active_qubits,
            verbose=verbose,
        )
        return simulator



    # visualizing the decompositions
    simulator_vis = create_simulation(statevector_simulator)
    one_trotter = 1
    simulator_vis.set_symmetry_protection(False, False)
    simulator_vis.set_transpilation_level(0)
    simulator_vis.set_readout_error_mitigation(False)
    for i, (name, decomposition) in enumerate(decompositions.items()):
        circuit = simulator_vis.make_base_circuit(
            1,
            decomposition,
            name,
        )
        simulator_vis.circuit_drawer(circuit)



    """
    symmetry_protection = False
    error_mitigation = False
    transpilation_level = 0
    backend = statevector_simulator
    simulator = create_simulation(backend)
    simulator.set_transpilation_level(transpilation_level)
    simulator.set_symmetry_protection(
        symmetry_protection,
        symmetry_protection_su_2,
    )
    simulator.set_readout_error_mitigation(error_mitigation)
    """

    #"""
    for backend in backends:
        simulator = create_simulation(backend)
        for transpilation_level in reversed(range(4)):
            if backend==statevector_simulator:
                simulator.set_transpilation_level(3)
            else:
                simulator.set_transpilation_level(transpilation_level)
            for readout_mitigation in readout_error_mitigation:
                simulator.set_readout_error_mitigation(readout_mitigation)
                """
                if backend==statevector_simulator or sim_nairobi_noiseless:
                    simulator.set_readout_error_mitigation(False)
                    print('skalle')
                else:
                    print('hei')
                    simulator.set_readout_error_mitigation(readout_mitigation)
                """
                for sp in symmetry_protection:
                    if backend==statevector_simulator:
                        simulator.set_symmetry_protection(
                            False,
                            symmetry_protection_su_2,
                        )
                    else:
                        simulator.set_symmetry_protection(
                            sp,
                            symmetry_protection_su_2,
                        )
                    for i, (name, decomposition) in enumerate(decompositions.items()):
                        fidelities = np.zeros(shape=(len(trotter_steps)))
                        for j, steps in enumerate(trotter_steps):
                            fidelity = simulator.run(
                                name,
                                decomposition,
                                steps,
                                draw_circuits='none',
                            )
                            fidelities[j] = fidelity

                        np.save(f'../data/final_runs/fidelities_{backend}_{name}_SP_{sp}_mitigated_{readout_mitigation}_transpilation_level_{transpilation_level}', fidelities)

    #"""
    """
    for backend in backends:
        print(backend)
        for i, (name, decomposition) in enumerate(decompositions.items()):
            for j, t in enumerate(trotter_steps):
                for sp in symmetry_protection:
                    for transpilation_level in range(4):
                        for readout_mitigation in readout_error_mitigation:
                            simulator = create_simulation(backend)
                            simulator.set_transpilation_level(
                                transpilation_level
                            )
                            simulator.set_symmetry_protection(
                                sp,
                                symmetry_protection_su_2,
                            )
                            simulator.set_readout_error_mitigation(
                                readout_mitigation
                            )
            print(name)
    """





    #np.save('../data/final_runs/fidelities_1st_2nd_order_comparison', fidelities)




    """
    noisy_simulation = True
    symmetry_protection = False
    backend = sim_nairobi_noisy
    simulator = create_simulation(backend)
    transpilation_level = 3
    simulator.set_transpilation_level(transpilation_level)
    simulator.set_symmetry_protection(
        symmetry_protection,
        symmetry_protection_su_2,
    )
    simulator.set_readout_error_mitigation(True)

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

    #np.save('../data/fidelities_1st_2nd_order_comparison', fidelities)

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
        """
        eb1 = plt.errorbar(
            x_axis, fid_means[i, :],
            yerr=fid_stdevs[i, :],
            errorevery=1,
            label=name,
            ls=linestyle_tuple[i][1],
            capsize=5
        )
        #"""

        plot = plt.plot(
            x_axis, fidelities[i, :],
            label=name,
            ls=linestyle_tuple[i][1]
        )

    plt.xlabel('Trotter Steps')
    plt.ylabel('Fidelity')
    plt.legend()

    """
    if noisy_simulation:
        plt.title(f'Noisy Trotter Simulation with {shots} Shots, \n Backend: {config.backend_name}, SP: {repr(symmetry_protection)}')
        plt.savefig(f'../figures/{backend}_trotter_sim_{min_trotter_steps}_{max_trotter_steps}_shots{shots}_SP{repr(symmetry_protection)}_{len(decompositions)}')
    else:
        plt.title(f'Noiseless Trotter Simulation with {shots} Shots, \n Backend: {config.backend_name}, SP: {repr(symmetry_protection)}')
        plt.savefig(f'../figures/double_time_{backend}_trotter_sim_{min_trotter_steps}_{max_trotter_steps}_shots{shots}_SP{repr(symmetry_protection)}_{len(decompositions)}')
    """

    plt.show()
    #"""

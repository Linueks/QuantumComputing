import copy
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


decompositions = {
    #'zyxzyx': trotter_step_zyxzyx,
    'zzyyxx': trotter_step_zzyyxx,
    #'x+yzzx+y': trotter_step_xplusy_zz_xplusy,
    #'x+yzx+yz': trotter_step_xplusy_z_xplusy_z,
    #'x+y+z': trotter_step_xplusyplusz_xplusyplusz,
}

job_ids = [
    '624d6e13cfe45c3bf1e599f7',
    '624d74b7a5d4ee1e2677c35c',
    '624d75bed72033698a67ce88',
    '624d7642aacb9b785e5f4423',
    '624d76bdcaa2651ecaf19029',
]




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
repetitions = 1
basis_gates=['id', 'rz', 'sx', 'x', 'cx', 'reset']
active_qubits = [1, 3, 5]


transpilation_level = 3
noisy_simulation = True
symmetry_protection = False
verbose = True
draw_circuit = False
use_real_device = True


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

simulator.set_transpilation_level(transpilation_level)

simulator.set_symmetry_protection(
    symmetry_protection=symmetry_protection,
    symmetry_protection_function=symmetry_protection_su_2,
)


base_circuit = simulator.make_base_circuit(
    trotter_steps=8,
    trotter_step_function=trotter_step_xplusyplusz_xplusyplusz,
    name='x+y+z',
)

tomography_circuits = simulator.make_tomography_circuits(
    base_circuit,
)

qjob = jakarta_backend.retrieve_job(job_ids[4])


fidelity, _ = simulator.calculate_fidelity(
    qjob,
    tomography_circuits,
)


print(fidelity)

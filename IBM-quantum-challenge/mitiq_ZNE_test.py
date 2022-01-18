"""
Trying to learn about Zero Noise Extrapolation using Mitiq from their intro
guide: https://mitiq.readthedocs.io/en/stable/guide/guide-getting-started.html#guide-getting-started
@linueks
"""
import numpy as np
import mitiq as mt
import qiskit as qk
import matplotlib.pyplot as plt
from basic_approach import construct_trotter_gate_zyxzyx, generate_circuit, run_simulation


time = qk.circuit.Parameter('t')
provider = qk.IBMQ.load_account()
provider = qk.IBMQ.get_provider(hub='ibm-q', group='open', project='main')
#print(provider.backends())
belem_backend = provider.get_backend('ibmq_belem')                              # has the same topology as Jakarta with qubits 1,3,4 corresponding to 1,3,5
properties = belem_backend.properties()
#print(properties)
config = belem_backend.configuration()
#print(config.backend_name)
sim_noisy_belem = QasmSimulator.from_backend(belem_backend)

shots = 2*8192
trotter_steps = 7                                                               # Variable if just running one simulation
end_time = np.pi                                                                # Specified in competition
num_jobs = 8



circuit = generate_circuit()

unmitigated_fidelity, std = run_simulation()
mitigated_fidelity, mitiq_std = mt.zne.execute_with_zne()

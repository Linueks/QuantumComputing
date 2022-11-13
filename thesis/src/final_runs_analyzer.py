import numpy as np
import qiskit as qk
import matplotlib.pyplot as plt
from decompositions import *
plt.style.use('ggplot')


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
config = nairobi_backend.configuration()
#print(config.basis_gates)

# set up qiskit simulators
sim_nairobi_noiseless = qk.providers.aer.QasmSimulator()
sim_nairobi_noisy = qk.providers.aer.QasmSimulator.from_backend(
    nairobi_backend
)
print(sim_nairobi_noiseless)

# statevector simulator
statevector_simulator = qk.Aer.get_backend(
    'statevector_simulator',
    device='GPU',
)





save_location = f'../data/final_runs/fidelities_{backend}_{name}_SP_{sp}_mitigated_{readout_mitigation}_transpilation_level_{transpilation_level}'
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
    sim_nairobi_noiseless,
    sim_nairobi_noisy,
]
symmetry_protection = [
    False,
    True,
]
readout_error_mitigation = [
    False,
    True,
]
end_time = np.pi                                                            # Specified in competition
min_trotter_steps = 4                                                       # 4 minimum for competition
max_trotter_steps = 20
trotter_steps = range(min_trotter_steps, max_trotter_steps+1)               # Variable if just running one simulation

"""
    for backend in backends:
        for transpilation_level in range(4):
            if backend==statevector_simulator:

            else:

            for readout_mitigation in readout_error_mitigation:
                if backend==statevector_simulator or sim_nairobi_noiseless:

                else:

                for sp in symmetry_protection:
                    if backend==statevector_simulator:

                    else:

                    for i, (name, decomposition) in enumerate(decompositions.items()):
                        for j, steps in enumerate(trotter_steps):
                            fidelities[j] = fidelity
"""


for backend in backends:
    figure = plt.figure(16, 8)
    transpilation_level = 3
    readout_mitigation = False
    sp = False
    for i, (name, decomposition) in enumerate(decompositions.items()):
        fidelities = np.load(f'../data/final_runs/fidelities_{backend}_{name}_SP_{sp}_mitigated_{readout_mitigation}_transpilation_level_{transpilation_level}')
        plt.plot(trotter_steps, fidelities, label=name)
        plt.xlabel('Trotter Steps [r]')
        plt.ylabel('State Tomography Fidelity')

    plt.show()

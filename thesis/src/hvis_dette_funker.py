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
import pandas as pd

#import matplotlib
#rc('font',**{'family':'serif','serif':['Helvetica']})
#rc('font',**{'family':'serif','serif':['Times']})
#rc('text', usetex=True)
#matplotlib.rcParams['font.family'] = ['Family1', 'serif', 'Family2']


import warnings
warnings.filterwarnings('ignore')
plt.style.use(['ieee', 'no-latex'])
#plt.style.use('high-vis')
#plt.rcParams.update({'figure.dpi': '100'})

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
        sim_nairobi_noiseless,
        sim_nairobi_noisy,
        oslo_backend,
        sim_oslo_noisy,
    ]
    symmetry_protection = [
        False,
        True,
    ]
    readout_error_mitigation = [
        False,
        True,
    ]
    from matplotlib.font_manager import FontProperties
    font = FontProperties()
    font.set_family('serif')
    font.set_name('Times New Roman')
    font.set_size(11)
    #font.set_style('italic')

    linestyle_tuple = [
        ('solid',                 (0, ())),
        ('dotted',                (0, (1, 1))),
        ('dashed',                (0, (5, 5))),
        ('dashdotted',            (0, (3, 5, 1, 5))),
        ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
        ('densely dashed',        (0, (5, 1))),
        ('densely dashdotted',    (0, (3, 1, 1, 1))),
        ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1))),
        ('densely dotted',        (0, (1, 1))),
    ]
    colors = iter([plt.cm.tab20(i) for i in range(20)])


    """
    ############################################################################ BASIC RESULT
    fig, axes = plt.subplots(2,3, figsize=(15,10))
    axes[1][2].set_visible(False)
    axes[1][0].set_position([0.24,0.100,0.228,0.343])
    axes[1][1].set_position([0.55,0.100,0.228,0.343])
    axes = axes.ravel()
    labels = ['a)', 'b)', 'c)', 'd)', 'e)']
    for j, backend in enumerate(backends):
        #for transpilation_level in range(4):
        #    if backend == oslo_backend:
        #        continue
        transpilation_level = 0
        readout_mitigation = False
        sp = False
        for i, (name, decomposition) in enumerate(decompositions.items()):
            fidelities = np.load(f'../data/final_runs/fidelities_{backend}_{name}_SP_{sp}_mitigated_{readout_mitigation}_transpilation_level_{transpilation_level}.npy')
            axes[j].plot(
                trotter_steps,
                fidelities,
                label=name,
                #c=[next(colors)],
                ls=linestyle_tuple[i][1],
            )
            #plt.xlabel('Trotter Steps', fontproperties=font)
            #plt.ylabel('State Tomography Fidelity', fontproperties=font)

            axes[j].set_title(
                f'{labels[j]} {backend}',
                fontproperties=font,
                loc='left',
                fontsize=18,
            )
            axes[j].set_xlabel(
                'Trotter Steps',
                fontproperties=font,
                fontsize=18,
            )
            axes[j].tick_params(axis='both', which='major', labelsize=12)
            axes[j].set_ylabel(
                'Fidelity',
                fontproperties=font,
                fontsize=18,
            )
            #axes[j].legend(fontsize='medium')
            #plt.savefig(f'../data/final_figs/trotter_steps_vs_fidelity_basic_{backend}')
    #plt.subplots_adjust(bottom=0.2)
    #lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    #lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    # Finally, we invoke the legend (that you probably would like to customize...)
    #axes[0].get_shared_y_axes().join(axes[0], *axes[0:2])
    #axes[3].get_shared_y_axes().join(axes[3], *axes[3:])

    fig.legend(decompositions.keys(), loc=(0.83, 0.2), fontsize='x-large')
    #plt.tight_layout()# looks like shit
    plt.savefig(
        f'../data/final_figs/trotter_steps_vs_fidelity_basic',
        bbox_inches='tight'
    )
    #plt.show()
    ############################################################################
    #"""



    """
    ############################################################################ TRANSPILATION ON QASM OSLO NOISEMODEL
    # I just switched the model manually to generate the three plots...
    fig, axes = plt.subplots(3, 3, figsize=(15,10))
    axes[2][2].set_visible(False)
    #axes[2][0].set_position([0.24,0.100,0.228,0.343])
    #axes[2][1].set_position([0.55,0.100,0.228,0.343])
    axes = axes.ravel()
    labels = ['  a)', '  b)', '  c)', '  d)', '  e)', '  f)', '  g)', '  h)']
    backend = oslo_backend
    transpilations = [0, 3]#range(4)
    for i, (name, decomposition) in enumerate(decompositions.items()):
        readout_mitigation = False
        sp = False
        for transpilation_level in transpilations:
            fidelities = np.load(f'../data/final_runs/fidelities_{backend}_{name}_SP_{sp}_mitigated_{readout_mitigation}_transpilation_level_{transpilation_level}.npy')
            axes[i].plot(
                trotter_steps,
                fidelities
            )#, label=f'Level: {transpilation_level}')
            axes[i].set_xlabel(
                'Trotter Steps',
                fontproperties=font,
                fontsize=18,
            )
            axes[i].set_ylabel(
                'Fidelity',
                fontproperties=font,
                fontsize=18,
            )
            axes[i].tick_params(axis='both', which='major', labelsize=12)
            axes[i].set_title(
                f'{labels[i]}',
                fontproperties=font,
                loc='left',
                fontsize=18,
            )
    legend_stuff = [f'level: {level}' for level in transpilations]
    fig.legend(legend_stuff,  loc=(0.70, 0.2), fontsize='x-large')
    plt.savefig(
        f'../data/final_figs/trotter_steps_vs_fidelity_transpilation_ibm_oslo_device',
        bbox_inches='tight'
    )
    ############################################################################

    #"""

    #plt.show()

    """
    ############################################################################ SYMMETRY PROTECTION
    fig, axes = plt.subplots(2,3, figsize=(15,10))
    axes[1][2].set_visible(False)
    axes[1][0].set_position([0.24,0.100,0.228,0.343])
    axes[1][1].set_position([0.55,0.100,0.228,0.343])
    axes = axes.ravel()
    labels = ['  a)', '  b)', '  c)', '  d)', '  e)']
    for j, backend in enumerate(backends):
        #for transpilation_level in range(4):
        #    if backend == oslo_backend:
        #        continue
        transpilation_level = 0
        readout_mitigation = False
        sp = True
        for i, (name, decomposition) in enumerate(decompositions.items()):
            fidelities = np.load(f'../data/final_runs/fidelities_{backend}_{name}_SP_{sp}_mitigated_{readout_mitigation}_transpilation_level_{transpilation_level}.npy')
            axes[j].plot(
                trotter_steps,
                fidelities,
                label=name,
                #c=[next(colors)],
                ls=linestyle_tuple[i][1],
            )
            #plt.xlabel('Trotter Steps', fontproperties=font)
            #plt.ylabel('State Tomography Fidelity', fontproperties=font)

            axes[j].set_title(
                f'{labels[j]} {backend}',
                fontproperties=font,
                loc='left',
                fontsize=18,
            )
            axes[j].set_xlabel(
                'Trotter Steps',
                fontproperties=font,
                fontsize=18,

            )
            axes[j].set_ylabel(
                'Fidelity',
                fontproperties=font,
                fontsize=18,
            )
            axes[j].tick_params(axis='both', which='major', labelsize=12)

            #axes[j].legend(fontsize='medium')
            #plt.savefig(f'../data/final_figs/trotter_steps_vs_fidelity_basic_{backend}')
    #plt.subplots_adjust(bottom=0.2)
    #lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    #lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    # Finally, we invoke the legend (that you probably would like to customize...)
    fig.legend(decompositions.keys(), loc=(0.83, 0.2), fontsize='x-large')
    #plt.tight_layout()# looks like shit
    plt.savefig(
        f'../data/final_figs/trotter_steps_vs_fidelity_symmetry_protection',
        bbox_inches='tight'
    )

    #plt.show()
    ############################################################################

    """


    """
    ############################################################################ SYMMETRY PROTECTION
    fig, axes = plt.subplots(3,3, figsize=(15,10))
    axes[2][2].set_visible(False)
    axes = axes.ravel()
    labels = ['  a)', '  b)', '  c)', '  d)', '  e)', '  f)', '  g)', '  h)']
    backend = sim_nairobi_noisy
    for transpilation_level in range(4):
        #    if backend == oslo_backend:
        #        continue
        #transpilation_level = 0
        readout_mitigation = False
        sp = True
        for i, (name, decomposition) in enumerate(decompositions.items()):
            fidelities = np.load(f'../data/final_runs/fidelities_{backend}_{name}_SP_{sp}_mitigated_{readout_mitigation}_transpilation_level_{transpilation_level}.npy')
            axes[i].plot(
                trotter_steps,
                fidelities,
                #label=name,
                #c=[next(colors)],
                ls=linestyle_tuple[transpilation_level][1],
            )
            #plt.xlabel('Trotter Steps', fontproperties=font)
            #plt.ylabel('State Tomography Fidelity', fontproperties=font)

            axes[i].set_title(
                f'{labels[i]}',
                fontproperties=font,
                loc='left',
                fontsize=18,
            )
            axes[i].set_xlabel(
                'Trotter Steps',
                fontproperties=font,
                fontsize=18,
            )
            axes[i].set_ylabel(
                'Fidelity',
                fontproperties=font,
                fontsize=18,
            )
            axes[i].tick_params(axis='both', which='major', labelsize=12)

            #axes[j].legend(fontsize='medium')
            #plt.savefig(f'../data/final_figs/trotter_steps_vs_fidelity_basic_{backend}')
    #plt.subplots_adjust(bottom=0.2)
    #lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    #lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    # Finally, we invoke the legend (that you probably would like to customize...)
    fig.legend(['level: 0', 'level: 1', 'level: 2', 'level: 3'], loc=(0.72, 0.2), fontsize='x-large')
    #plt.tight_layout()# looks like shit
    plt.savefig(
        f'../data/final_figs/trotter_steps_vs_fidelity_symmetry_protection_with_transpilations',
        bbox_inches='tight'
    )

    #plt.show()
    ############################################################################

    #"""




    """
    ############################################################################ SYMMETRY PROTECTION FOR EACH BACKEND
    fig, axes = plt.subplots(2,3, figsize=(15,10))
    axes[1][2].set_visible(False)
    axes[1][0].set_position([0.24,0.100,0.228,0.343])
    axes[1][1].set_position([0.55,0.100,0.228,0.343])
    axes = axes.ravel()
    labels = ['  a)', '  b)', '  c)', '  d)', '  e)']
    for j, backend in enumerate(backends):
        #for transpilation_level in range(4):
        #    if backend == oslo_backend:
        #        continue
        transpilation_level = 0
        readout_mitigation = False
        sp = True
        for i, (name, decomposition) in enumerate(decompositions.items()):
            fidelities = np.load(f'../data/final_runs/fidelities_{backend}_{name}_SP_{sp}_mitigated_{readout_mitigation}_transpilation_level_{transpilation_level}.npy')
            axes[j].plot(
                trotter_steps,
                fidelities,
                label=name,
                #c=[next(colors)],
                ls=linestyle_tuple[i][1],
            )
            #plt.xlabel('Trotter Steps', fontproperties=font)
            #plt.ylabel('State Tomography Fidelity', fontproperties=font)

            axes[j].set_title(
                f'{labels[j]}',
                fontproperties=font,
                loc='left',
                fontsize=14,
            )
            axes[j].set_xlabel('Trotter Steps', fontproperties=font)
            axes[j].set_ylabel('Fidelity', fontproperties=font)
            #axes[j].legend(fontsize='medium')
            #plt.savefig(f'../data/final_figs/trotter_steps_vs_fidelity_basic_{backend}')
    #plt.subplots_adjust(bottom=0.2)
    #lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    #lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    # Finally, we invoke the legend (that you probably would like to customize...)
    fig.legend(decompositions.keys(), loc=(0.83, 0.2), fontsize='x-large')
    #plt.tight_layout()# looks like shit
    plt.savefig(
        f'../data/final_figs/trotter_steps_vs_fidelity_symmetry_protection',
        bbox_inches='tight'
    )
    #"""

    """
    ############################################################################ READOUT MITIGATION FOR NOISY BACKENDS
    labels = ['  a)', '  b)', '  c)', '  d)', '  e)', '  f)', '  g)', '  h)']
    backends = [sim_nairobi_noisy, sim_oslo_noisy, oslo_backend]
    transpilation_level = 3
    sp = False

    for backend in backends:
        fig, axes = plt.subplots(3,3, figsize=(15,10))
        axes[2][2].set_visible(False)
        axes = axes.ravel()
        for i, (name, decomposition) in enumerate(decompositions.items()):
            fidelities_mitigated = np.load(f'../data/final_runs/fidelities_{backend}_{name}_SP_{sp}_mitigated_{True}_transpilation_level_{transpilation_level}.npy')
            fidelities_unmitigated = np.load(f'../data/final_runs/fidelities_{backend}_{name}_SP_{sp}_mitigated_{False}_transpilation_level_{transpilation_level}.npy')
            axes[i].plot(
                trotter_steps,
                fidelities_mitigated,
                #c=[next(colors)],
                ls='dashed',
            )
            axes[i].plot(
                trotter_steps,
                fidelities_unmitigated,
                #c=[next(colors)],
                ls='solid',
            )
            #plt.xlabel('Trotter Steps', fontproperties=font)
            #plt.ylabel('State Tomography Fidelity', fontproperties=font)

            axes[i].set_title(
                f'{labels[i]}',
                fontproperties=font,
                loc='left',
                fontsize=18,
            )
            axes[i].set_xlabel(
                'Trotter Steps',
                fontproperties=font,
                fontsize=18,
            )
            axes[i].set_ylabel(
                'Fidelity',
                fontproperties=font,
                fontsize=18,
            )
            axes[i].tick_params(axis='both', which='major', labelsize=12)
            #axes[j].legend(fontsize='medium')
            #plt.savefig(f'../data/final_figs/trotter_steps_vs_fidelity_basic_{backend}')
        #plt.subplots_adjust(bottom=0.2)
        #lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
        #lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        # Finally, we invoke the legend (that you probably would like to customize...)
        fig.legend(['Readout Mitigated', 'Unmitigated'], loc=(0.76, 0.2), fontsize='x-large')
        #plt.tight_layout()# looks like shit
        plt.savefig(
            f'../data/final_figs/trotter_steps_vs_fidelity_readout_mitigation_backend_{backend}',
            bbox_inches='tight'
        )
        plt.close()

    #for i, (name, decomposition) in enumerate(decompositions.items()):
    #    sp = False
    #    for transpilation_level in range(4):
    #        fidelities_mitigated = np.load(f'../data/final_runs/fidelities_{backend}_{name}_SP_{sp}_mitigated_{True}_transpilation_level_{transpilation_level}.npy')
    #        fidelities_unmitigated = np.load(f'../data/final_runs/fidelities_{backend}_{name}_SP_{sp}_mitigated_{False}_transpilation_level_{transpilation_level}.npy')
    #        plt.plot(trotter_steps, fidelities_mitigated, linestyle='dashed', label=f'Level: {transpilation_level}')
    #        plt.plot(trotter_steps, fidelities_unmitigated, label=f'Level: {transpilation_level}')
    #        plt.xlabel('Trotter Steps', fontproperties=font)
    #        plt.ylabel('State Tomography Fidelity', fontproperties=font)
    #        plt.title(f'{name} Varying Transpilation Level', fontproperties=font)
    #        plt.legend()
    #plt.show()
    ############################################################################
    #"""


    """
    # gonna make plots for the best single combinations for oslo device first NO POINT
    backend = oslo_backend
    name, decomposition = 'n_xyz', trotter_step_n_xyz
    readout_mitigation = True
    symmetry_protection = False
    transpilation_level = 3
    fig, ax = plt.subplots()
    fidelities = np.load(f'../data/final_runs/fidelities_{backend}_{name}_SP_{symmetry_protection}_mitigated_{readout_mitigation}_transpilation_level_{transpilation_level}.npy')
    ax.plot(trotter_steps, fidelities)
    plt.show()
    """



    """
    # gonna make a table over the best performing setup for each decomposition
    print(len(backends),
    len(decompositions.items()),
    len(symmetry_protection),
    len(range(4)), # In this one some values for the real device need to set to zero
    len(readout_error_mitigation))
    best_results = np.zeros(
        shape=(
            5,
            8,
            2,
            4, # In this one some values for the real device need to set to zero
            2,
        )
    )
    #counter = 0
    for i, backend in enumerate(backends):
        for j, sp in enumerate(symmetry_protection):
            for transpilation_level in range(4):
                for k, readout in enumerate(readout_error_mitigation):
                    for h, (name, decomposition) in enumerate(decompositions.items()):
                        if backend == oslo_backend and (transpilation_level == 1 or transpilation_level == 2):
                            max_value = 0
                            best_results[
                                i,
                                h,
                                j,
                                transpilation_level,
                                k,
                            ] = max_value
                            #counter += 1
                            #print(max_value)
                        else:
                            fidelities = np.load(f'../data/final_runs/fidelities_{backend}_{name}_SP_{sp}_mitigated_{readout}_transpilation_level_{transpilation_level}.npy')
                            max_value = np.amax(fidelities)
                            max_index = np.where(fidelities == max_value)
                            #print(max_index)
                            corresponding_trotter_step = max_index[0] + 4
                            best_results[
                                i,
                                h,
                                j,
                                transpilation_level,
                                k,
                            ] = max_value
                            #counter += 1
                            #print(max_value)
    # try to find for each decomposition on each backend what combination gave the best results from trotter steps, readout mitigation, transpilation level
    #"""

    ############################################################################ FINDING BEST VALUES FOR BASIC RESULTS
    """
    print('basic results best result for each decomposition\n')
    print(best_results[:, :, 0, 0, 0])
    print(np.amax(best_results[:, :, 0, 0, 0], axis=1))
    for result in np.amax(best_results[:, :, 0, 0, 0], axis=1):
        print(np.where(best_results[:, :, 0, 0, 0] == result))

    #"""
    """
    print('readout mitigation best result for each decomposition\n')
    print(best_results[:, :, 0, 0, 1])
    print(np.amax(best_results[:, :, 0, 0, 1], axis=1))
    for result in np.amax(best_results[:, :, 0, 0, 1], axis=1):
        print(np.where(best_results[:, :, 0, 0, 1] == result))
    #"""

    """
    print('transpilation best result for each decomposition\n')
    print(best_results[:, :, 0, 0:, 0])
    print(np.amax(best_results[:, :, 0, 0:, 0], axis=1))
    for result in np.amax(best_results[:, :, 0, 0:, 0], axis=1):
        print(np.where(best_results[:, :, 0, 0:, 0] == result))
    #"""

    """
    print('symmetry protection results\n')
    print(best_results[:, :, 1, 0, 0])
    print(np.amax(best_results[:, :, 1, 0, 0], axis=1))
    for result in np.amax(best_results[:, :, 1, 0, 0], axis=1):
        print(np.where(best_results[:, :, 1, 0, 0] == result))
    #"""


    ############################################################################################


    """
    backend = sim_nairobi_noisy
    for i, (name, decomposition) in enumerate(decompositions.items()):
        sp = True
        for transpilation_level in range(4):
            fidelities_mitigated = np.load(f'../data/final_runs/fidelities_{backend}_{name}_SP_{sp}_mitigated_{True}_transpilation_level_{transpilation_level}.npy')
            fidelities_unmitigated = np.load(f'../data/final_runs/fidelities_{backend}_{name}_SP_{sp}_mitigated_{False}_transpilation_level_{transpilation_level}.npy')
            plt.plot(trotter_steps, fidelities_mitigated, linestyle='dashed', label=f'Level: {transpilation_level}')
            plt.plot(trotter_steps, fidelities_unmitigated, label=f'Level: {transpilation_level}')
            plt.xlabel(
                'Trotter Steps',
                fontproperties=font,
                fontsize=18,
            )
            plt.ylabel(
                'State Tomography Fidelity',
                fontproperties=font
                fontsize=18,
            )
            plt.title(
                f'{name}',
                fontproperties=font,
                fontsize=18,
                loc='left',

            )
            plt.legend()

        plt.show()
    #"""

    """
    ############################################################################
    # Getting the circuit depth for each circuit and their transpilations

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

    simulator = create_simulation(sim_nairobi_noisy)
    # I want for each trotter step, decomposition, transpilation level

    circuit_depths = np.zeros(
        shape = (len(decompositions), len(trotter_steps), 4),
    )
    for transpilation_level in range(4):
        print(transpilation_level)
        simulator.set_transpilation_level(transpilation_level)
        simulator.set_symmetry_protection(False, False)
        simulator.set_readout_error_mitigation(False)
        for i, steps in enumerate(trotter_steps):
            for j, (name, decomposition) in enumerate(decompositions.items()):
                circuit = simulator.make_base_circuit(
                    steps,
                    decomposition,
                    name,
                )
                circuit_depths[j, i, transpilation_level] = circuit.depth()


    print(circuit_depths)
    #np.save(f'../data/final_runs/circuit_depths_per_trotterstep_and_transpilation_level.npy', circuit_depths)
    ############################################################################
    #"""





    circuit_depths = np.load(f'../data/final_runs/circuit_depths_per_trotterstep_and_transpilation_level.npy')
    # https://github.com/Qiskit/qiskit-terra/issues/6122 The depths being shorter for t = 1, 2 is a known "issue" and should just be commented on and highlighted as a result / consideration
    """
    ############################################################################ CIRCUIT DEPTH VS TROTTER STEPS FOR EACH DECOMPOSITION
    fig, axes = plt.subplots(3,3, figsize=(15,10))
    axes[2][2].set_visible(False)
    axes = axes.ravel()
    backend = sim_nairobi_noisy
    labels = ['  a)', '  b)', '  c)', '  d)', '  e)', '  f)', '  g)', '  h)']
    for i, (name, decomposition) in enumerate(decompositions.items()):
        for transpilation_level in range(4):
        #plt.plot(trotter_steps, circuit_depths[i, :, 3] / circuit_depths[i, :, 0], label=f'{name}')
            axes[i].plot(
                trotter_steps,
                circuit_depths[i, :, transpilation_level],
                #c=[next(colors)],
                ls=linestyle_tuple[transpilation_level][1],
            )
        axes[i].set_xlabel(
            'Trotter Steps',
            fontproperties=font,
            fontsize=18,
        )
        axes[i].set_ylabel(
            'Circuit Depth',
            fontproperties=font,
            fontsize=18,
        )
        axes[i].set_title(
            f'{labels[i]}',
            fontproperties=font,
            loc='left',
            fontsize=18,
        )
        axes[i].tick_params(axis='both', which='major', labelsize=10)
    fig.legend(
        ['level: 0', 'level: 1', 'level: 2', 'level: 3'],
        loc=(0.72, 0.2),
        fontsize='x-large'
    )
    plt.savefig(
        f'../data/final_figs/trotter_steps_vs_circuit_depths_{backend}',
        bbox_inches='tight'
    )
    plt.close()
    ############################################################################
    #"""

    """
    print(circuit_depths[0, :, 3] / circuit_depths[0, :, 0])
    backend = sim_nairobi_noisy
    for i, (name, decomposition) in enumerate(decompositions.items()):
        fidelities_unmitigated_0 = np.load(f'../data/final_runs/fidelities_{backend}_{name}_SP_{False}_mitigated_{False}_transpilation_level_{0}.npy')
        fidelities_unmitigated_3 = np.load(f'../data/final_runs/fidelities_{backend}_{name}_SP_{False}_mitigated_{False}_transpilation_level_{3}.npy')
        ratio = fidelities_unmitigated_3 / fidelities_unmitigated_0
        plt.plot(trotter_steps, ratio, label=f'{name}', fontproperties=font)
    plt.legend()
    plt.show()

    for i, (name, decomposition) in enumerate(decompositions.items()):
        fidelities_unmitigated_0 = np.load(f'../data/final_runs/fidelities_{backend}_{name}_SP_{False}_mitigated_{True}_transpilation_level_{0}.npy')
        fidelities_unmitigated_3 = np.load(f'../data/final_runs/fidelities_{backend}_{name}_SP_{False}_mitigated_{True}_transpilation_level_{3}.npy')
        ratio = fidelities_unmitigated_3 / fidelities_unmitigated_0
        plt.plot(trotter_steps, ratio, label=f'{name}', fontproperties=font)
    plt.legend()
    plt.show()
    #"""




    backend = oslo_backend
    """
    for i, (name, decomposition) in enumerate(decompositions.items()):
        fidelities_unmitigated_0 = np.load(f'../data/final_runs/fidelities_{backend}_{name}_SP_{False}_mitigated_{False}_transpilation_level_{0}.npy')
        fidelities_unmitigated_3 = np.load(f'../data/final_runs/fidelities_{backend}_{name}_SP_{False}_mitigated_{False}_transpilation_level_{3}.npy')
        ratio = fidelities_unmitigated_3 / fidelities_unmitigated_0
        plt.plot(trotter_steps, ratio, label=f'{name}')
    plt.legend()
    plt.show()
    #"""

    """
    for i, (name, decomposition) in enumerate(decompositions.items()):
        sp = False
        for transpilation_level in [0, 3]:
            fidelities_mitigated = np.load(f'../data/final_runs/fidelities_{backend}_{name}_SP_{sp}_mitigated_{True}_transpilation_level_{transpilation_level}.npy')
            fidelities_unmitigated = np.load(f'../data/final_runs/fidelities_{backend}_{name}_SP_{sp}_mitigated_{False}_transpilation_level_{transpilation_level}.npy')
            plt.plot(trotter_steps, fidelities_mitigated, linestyle='dashed', label=f'Level: {transpilation_level}')
            plt.plot(trotter_steps, fidelities_unmitigated, label=f'Level: {transpilation_level}')
            plt.xlabel('Trotter Steps [r]', fontproperties=font)
            plt.ylabel('State Tomography Fidelity', fontproperties=font)
            plt.title(f'Trotter Decomposition {name} Varying Transpilation Level on {backend}', fontproperties=font)
            plt.legend()

        plt.show()

    """
    """
    for i, (name, decomposition) in enumerate(decompositions.items()):
        sp = True
        for transpilation_level in [0, 3]:
            fidelities_mitigated = np.load(f'../data/final_runs/fidelities_{backend}_{name}_SP_{sp}_mitigated_{True}_transpilation_level_{transpilation_level}.npy')
            fidelities_unmitigated = np.load(f'../data/final_runs/fidelities_{backend}_{name}_SP_{sp}_mitigated_{False}_transpilation_level_{transpilation_level}.npy')
            plt.plot(trotter_steps, fidelities_mitigated, linestyle='dashed', label=f'Level: {transpilation_level}')
            plt.plot(trotter_steps, fidelities_unmitigated, label=f'Level: {transpilation_level}')
            plt.xlabel('Trotter Steps [r]', fontproperties=font)
            plt.ylabel('State Tomography Fidelity', fontproperties=font)
            plt.title(f'Symmetry Protected {name}, Varying Transpilation Level on {backend}', fontproperties=font)
            plt.legend()

        plt.show()
    #"""


    """
    divide the results of second and third run and plot that, maybe think of surface plots
    """


    """
    ############################################################################ RANDOMIZED CIRCUITS
    fig, ax = plt.subplots(figsize=(15, 10))
    random_circuits_fidelity_ibm_nairobi = np.load(f'../data/final_runs/randomized_circuits_fidelities_max_depth200_n_circs10_qasm_simulator(ibm_nairobi).npy')
    random_circuits_fidelity_ibm_oslo = np.load(f'../data/final_runs/randomized_circuits_fidelities_max_depth200_n_circs10_qasm_simulator(ibm_oslo).npy')
    depths = range(1, 201)
    means = np.mean(random_circuits_fidelity_ibm_oslo, axis=0)
    yerr_above = means + np.std(random_circuits_fidelity_ibm_oslo, axis=0)
    yerr_below = means - np.std(random_circuits_fidelity_ibm_oslo, axis=0)
    ax.plot(depths, means)
    ax.fill_between(depths, yerr_below, yerr_above, alpha=0.2, color='red')
    ax.set_xlabel(
        'Circuit Depth',
        fontproperties=font,
        fontsize=18
    )
    ax.set_ylabel(
        'Fidelity',
        fontproperties=font,
        fontsize=18,
    )
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_ylim(-0.001)
    #fig.legend()
    plt.savefig(
        f'../data/final_figs/randomized_circuits_plot_oslo',
        bbox_inches='tight'
    )

    fig, ax = plt.subplots(figsize=(15, 10))
    depths = range(1, 201)
    means = np.mean(random_circuits_fidelity_ibm_nairobi, axis=0)
    yerr_above = means + np.std(random_circuits_fidelity_ibm_nairobi, axis=0)
    yerr_below = means - np.std(random_circuits_fidelity_ibm_nairobi, axis=0)
    ax.plot(depths, means)
    ax.fill_between(depths, yerr_below, yerr_above, alpha=0.2, color='red')
    ax.set_xlabel(
        'Circuit Depth',
        fontproperties=font,
        fontsize=18
    )
    ax.set_ylabel(
        'Fidelity',
        fontproperties=font,
        fontsize=18,
    )
    ax.tick_params(axis='both', which='major', labelsize=10)

    ax.set_ylim(-0.001)
    #fig.legend(['Oslo noisemodel', 'Nairobi noisemodel'])
    plt.savefig(
        f'../data/final_figs/randomized_circuits_plot_nairobi',
        bbox_inches='tight'
    )


    #print(random_circuits_fidelity_ibm_oslo)
    #print(random_circuits_fidelity_ibm_oslo.shape)
    #"""

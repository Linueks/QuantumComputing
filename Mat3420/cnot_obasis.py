import numpy as np
import qiskit as qk
import matplotlib.pyplot as plt
from qiskit.circuit.library.standard_gates\
    import SGate, HGate, SdgGate, CXGate, CCXGate
from qiskit.quantum_info import Operator
from qiskit.visualization import circuit_drawer
import arr_to_ltx as a2l


def cnot(circuit, register, control, target,
        print_operator=True,
        visualize=True,
        check_approx_equality=True,
        plot_circuit=False):
    """
    circuit (Qiskit circuit object)
    register (Qiskit quantum register object)
    control (int) index in the register for the control qubit
    target (int) index in the register for the target qubit
    """
    h_gate = HGate()
    control_s_gate = SGate().control(1)

    circuit.append(h_gate, [register[target]])
    circuit.append(control_s_gate, [register[control], register[target]])
    circuit.append(control_s_gate, [register[control], register[target]])
    circuit.append(h_gate, [register[target]])

    if print_operator:
        # using reverse bits here to recover conventional notation
        if print_operator=='latex':
            latex_code = a2l.to_ltx(Operator(circuit.reverse_bits()).data)
            print(repr(latex_code))
        else:

            print(Operator(circuit.reverse_bits()).data)

    if visualize:
        if visualize=='mpl':
            circuit.draw('mpl')
            plt.show()

        else:
            print(circuit)

    if check_approx_equality:
        print(Operator(circuit) == Operator(CXGate()))



def toffoli(circuit, register, control1, control2, target,
        print_operator=True,
        visualize=True,
        check_approx_equality=True,
        plot_circuit=False):
    """
    circuit (Qiskit circuit object)
    register (Qiskit quantum register object)
    control1 (int) index in the register for the first control qubit
    control2 (int) index in the register for the second control qubit
    target (int) index in the register for the target qubit
    """
    h_gate = HGate()
    control_s_gate = SGate().control(1)
    control_sdg_gate = SdgGate().control(1)


    circuit.append(h_gate, [register[target]])
    circuit.append(control_s_gate, [register[control2], register[target]])
    cnot(circuit, register, control1, control2,
            print_operator=False,
            visualize=False,
            check_approx_equality=False)
    circuit.append(control_sdg_gate, [register[control2], register[target]])
    cnot(circuit, register, control1, control2,
            print_operator=False,
            visualize=False,
            check_approx_equality=False)
    circuit.append(control_s_gate, [register[control1], register[target]])
    circuit.append(h_gate, [register[target]])

    if print_operator:
        if print_operator=='latex':
            latex_code = a2l.to_ltx(Operator(circuit.reverse_bits()).data)
            print(repr(latex_code))

        else:
            print(Operator(circuit.reverse_bits()).data)

    if visualize:
        if visualize=='mpl':
            circuit.draw('mpl')
            plt.show()

        else:
            print(circuit)

    if check_approx_equality:
        print(Operator(circuit) == Operator(CCXGate()))



if __name__=='__main__':
    #qr1 = qk.QuantumRegister(2)
    #qc1 = qk.QuantumCircuit(qr1)
    #cnot(qc1, qr1, 0, 1, visualize='mpl', print_operator='latex')


    qr2 = qk.QuantumRegister(3)
    qc2 = qk.QuantumCircuit(qr2)
    toffoli(qc2, qr2, 0, 1, 2, visualize='mpl', print_operator='latex')

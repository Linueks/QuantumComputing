import qiskit as qk
import numpy as np


def xx_subcircuit(
    time,
    quantum_register,
    qubit1,
    qubit2,
):
    """
    Subcircuit used in Trotter decomposition

    Inputs:
        time: qiskit.circuit.Parameter
        quantum_register: qiskit.circuit.QuantumRegister
        qubit1: int
        qubit2: int

    returns:
        quantum_circuit: qiskit.circuit.QuantumCircuit
    """

    # Like Alessandro says this might even be too flexible
    quantum_circuit = qk.QuantumCircuit(quantum_register)
    quantum_circuit.ry(np.pi/2, [qubit1, qubit2])
    quantum_circuit.cnot(qubit1, qubit2)
    quantum_circuit.rz(2*time, qubit2)
    quantum_circuit.cnot(qubit1, qubit2)
    quantum_circuit.ry(-np.pi/2, [qubit1, qubit2])
    quantum_circuit.barrier()

    return quantum_circuit



def yy_subcircuit(
    time,
    quantum_register,
    qubit1,
    qubit2,
):
    """
    Subcircuit used in Trotter decomposition

    Inputs:
        time: qiskit.circuit.Parameter
        quantum_register: qiskit.circuit.QuantumRegister
        qubit1: int
        qubit2: int

    returns:
        quantum_circuit: qiskit.circuit.QuantumCircuit
    """

    quantum_circuit = qk.QuantumCircuit(quantum_register)
    quantum_circuit.rx(np.pi/2, [qubit1, qubit2])
    quantum_circuit.cnot(qubit1, qubit2)
    quantum_circuit.rz(2*time, qubit2)
    quantum_circuit.cnot(qubit1, qubit2)
    quantum_circuit.rx(-np.pi/2, [qubit1, qubit2])
    quantum_circuit.barrier()

    return quantum_circuit



def zz_subcircuit(
    time,
    quantum_register,
    qubit1,
    qubit2,
):
    """
    Subcircuit used in Trotter decomposition

    Inputs:
        time: qiskit.circuit.Parameter
        quantum_register: qiskit.circuit.QuantumRegister
        qubit1: int
        qubit2: int

    returns:
        quantum_circuit: qiskit.circuit.QuantumCircuit
    """

    quantum_circuit = qk.QuantumCircuit(quantum_register)
    quantum_circuit.cnot(qubit1, qubit2)
    quantum_circuit.rz(2*time, qubit2)
    quantum_circuit.cnot(qubit1, qubit2)
    quantum_circuit.barrier()

    return quantum_circuit



def n_xy_subcircuit(
    time,
    quantum_register,
    qubit1,
    qubit2,
):
    """
    Subcircuit for my specific Hamiltonian based on the description in the paper
    https://arxiv.org/pdf/quant-ph/0308006.pdf specifically figure 6.

    Inputs:
        time: qiskit.circuit.Parameter
        quantum_register: qiskit.circuit.QuantumRegister
        qubit1: int
        qubit2: int

    returns:
        quantum_circuit: qiskit.circuit.QuantumCircuit
    """
    quantum_circuit = qk.QuantumCircuit(quantum_register)
    quantum_circuit.rz((-np.pi/2), qubit2)
    quantum_circuit.cnot(qubit2, qubit1)
    quantum_circuit.rz((-np.pi/2), qubit1)
    quantum_circuit.ry((np.pi/2 + 2*time), qubit2)
    quantum_circuit.cnot(qubit1, qubit2)
    quantum_circuit.ry((-2*time - np.pi/2), qubit2)
    quantum_circuit.cnot(qubit2, qubit1)
    quantum_circuit.rz((np.pi/2), qubit1)

    return quantum_circuit



def symmetry_protection_su_2(
    quantum_register,
    active_qubits
):
    """
    Function to add a SU2 symmetry protection step into the register. This is
    the symmetry group that commutes with the Heisenberg model Hamiltonian.
    """
    quantum_circuit = qk.QuantumCircuit(quantum_register)
    quantum_circuit.h(active_qubits)

    return quantum_circuit



def trotter_step_partial_dot_product(
    time,
    quantum_register,
    active_qubits,
):
    """
    I think this is actually what he meant...

    Inputs:
        time: qiskit.circuit.Parameter
        quantum_register: qiskit.circuit.QuantumRegister
        active_qubits: list of qubit positions

    returns:
        quantum_circuit: qiskit.circuit.QuantumCircuit
    """
    qubit1, qubit2, qubit3 = active_qubits
    quantum_circuit = qk.QuantumCircuit(quantum_register)
    quantum_circuit += n_xy_subcircuit(time, quantum_register, qubit1, qubit2)
    quantum_circuit += zz_subcircuit(time, quantum_register, qubit1, qubit2)
    quantum_circuit += zz_subcircuit(time, quantum_register, qubit2, qubit3)
    quantum_circuit += n_xy_subcircuit(time, quantum_register, qubit2, qubit3)

    return quantum_circuit



def full_sum_optimal_construction(
    time,
    quantum_register,
    active_qubits,
):
    """
    Testing with the whole sum in the N circuit.

    Inputs:
        time: qiskit.circuit.Parameter
        quantum_register: qiskit.circuit.QuantumRegister
        active_qubits: list of qubit positions

    returns:
        quantum_circuit: qiskit.circuit.QuantumCircuit
    """
    qubit1, qubit2, qubit3 = active_qubits
    quantum_circuit = qk.QuantumCircuit(quantum_register)
    quantum_circuit.rz(-np.pi/2, qubit2)
    quantum_circuit.cnot(qubit2, qubit1)
    quantum_circuit.rz(-2*time - np.pi/2, qubit1)
    quantum_circuit.ry(np.pi/2 + 2*time, qubit2)
    quantum_circuit.cnot(qubit1, qubit2)
    quantum_circuit.ry(-2*time - np.pi/2, qubit2)
    quantum_circuit.cnot(qubit2, qubit1)
    quantum_circuit.rz(np.pi/2, qubit1)

    return quantum_circuit


def trotter_step_zyxzyx(
    time,
    quantum_register,
    active_qubits,
):
    """
    One Trotter step build out of xx, yy, zz subcircuits. This approach might
    prove worse than just hardcoding everything when it comes time to optimize.
    This is a three qubit operation, thus active qubits needs three positions

    Inputs:
        time: qiskit.circuit.Parameter
        quantum_register: qiskit.circuit.QuantumRegister
        active_qubits: list of qubit positions

    returns:
        quantum_circuit: qiskit.circuit.QuantumCircuit
    """
    qubit1, qubit2, qubit3 = active_qubits
    quantum_circuit = qk.QuantumCircuit(quantum_register)
    quantum_circuit += zz_subcircuit(time, quantum_register, qubit1, qubit2)
    quantum_circuit += yy_subcircuit(time, quantum_register, qubit1, qubit2)
    quantum_circuit += xx_subcircuit(time, quantum_register, qubit1, qubit2)
    quantum_circuit += zz_subcircuit(time, quantum_register, qubit2, qubit3)
    quantum_circuit += yy_subcircuit(time, quantum_register, qubit2, qubit3)
    quantum_circuit += xx_subcircuit(time, quantum_register, qubit2, qubit3)

    return quantum_circuit



def trotter_step_zzyyxx(
    time,
    quantum_register,
    active_qubits,
):
    """
    One Trotter step build out of xx, yy, zz subcircuits. This approach might
    prove worse than just hardcoding everything when it comes time to optimize.
    This is a three qubit operation, thus active qubits needs three positions

    Inputs:
        time: qiskit.circuit.Parameter
        quantum_register: qiskit.circuit.QuantumRegister
        active_qubits: list of qubit positions

    returns:
        quantum_circuit: qiskit.circuit.QuantumCircuit
    """
    qubit1, qubit2, qubit3 = active_qubits
    quantum_circuit = qk.QuantumCircuit(quantum_register)
    quantum_circuit += zz_subcircuit(time, quantum_register, qubit1, qubit2)
    quantum_circuit += zz_subcircuit(time, quantum_register, qubit2, qubit3)
    quantum_circuit += yy_subcircuit(time, quantum_register, qubit1, qubit2)
    quantum_circuit += yy_subcircuit(time, quantum_register, qubit2, qubit3)
    quantum_circuit += xx_subcircuit(time, quantum_register, qubit1, qubit2)
    quantum_circuit += xx_subcircuit(time, quantum_register, qubit2, qubit3)

    return quantum_circuit



def symmetry_protection_u_1(
    quantum_register,
    active_qubits
):
    """
    Function to add a U1 symmetry protection step into the register. This is the
    symmetry group which is preserved by the initial state |110>.
    """
    quantum_circuit = qk.QuantumCircuit(quantum_register)

    """
    Putting this function on hold until I've implemented the other Trotter
    decomposition.
    """
    return

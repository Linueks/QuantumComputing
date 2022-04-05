import numpy as np
import qiskit as qk
#from keras.datasets import mnist
from qibo import hamiltonians, gates, models, K
"""
#What I need to do

def encoder_hamiltonian_simple(nqubits, ncompress):
    m0 = K.to_numpy(hamiltonians.Z(ncompress).matrix)
    m1 = np.eye(2 ** (nqubits - ncompress), dtype=m0.dtype)
    ham = hamiltonians.Hamiltonian(nqubits, np.kron(m1, m0))
    return 0.5 * (ham + ncompress)


if example == 0:

    ising_groundstates = []
    lambdas = np.linspace(0.5, 1.0, 20)
    for lamb in lambdas:
        ising_ham = -1 * hamiltonians.TFIM(nqubits, h=lamb)
        ising_groundstates.append(ising_ham.eigenvectors()[0])

    if autoencoder == 1:
        circuit = models.Circuit(nqubits)
        for l in range(layers):
            for q in range(nqubits):
                circuit.add(gates.RY(q, theta=0))
            circuit.add(gates.CZ(5, 4))
            circuit.add(gates.CZ(5, 3))
            circuit.add(gates.CZ(5, 1))
            circuit.add(gates.CZ(4, 2))
            circuit.add(gates.CZ(4, 0))
            for q in range(nqubits):
                circuit.add(gates.RY(q, theta=0))
            circuit.add(gates.CZ(5, 4))
            circuit.add(gates.CZ(5, 2))
            circuit.add(gates.CZ(4, 3))
            circuit.add(gates.CZ(5, 0))
            circuit.add(gates.CZ(4, 1))
        for q in range(nqubits-compress, nqubits, 1):
            circuit.add(gates.RY(q, theta=0))

        def cost_function_QAE_Ising(params, count):
            cost = 0
            circuit.set_parameters(params) # this will change all thetas to the appropriate values
            for i in range(len(ising_groundstates)):
                final_state = circuit.execute(np.copy(ising_groundstates[i]))
                cost += K.to_numpy(encoder.expectation(final_state)).real

            cost_function_steps.append(cost/len(ising_groundstates)) # save cost function value after each step

            if count[0] % 50 == 0:
                print(count[0], cost/len(ising_groundstates))
            count[0] += 1

            return cost/len(ising_groundstates)

        nparams = 2 * nqubits * layers + compress
        initial_params = np.random.uniform(0, 2*np.pi, nparams)

        result = minimize(cost_function_QAE_Ising, initial_params,
                          args=(count), method='BFGS', options={'maxiter': maxiter})
"""



class QAE:
    def __init__(
        self,
        n_quantum,
        n_trash_qubits,
        n_classic,
        n_layers,
        feature_matrix,
        backend,
        seed,
        model_name,
        loss_function,
        loss_derivative,
    ):
        self.n_quantum = n_quantum
        self.n_trash_qubits = n_trash_qubits
        self.n_classic = n_classic
        self.n_layers = n_layers
        self.backend = backend
        self.seed = seed
        #self.n_model_parameters = n_model_parameters

        self.models = {
            'basic_model': self.basic_model,
        }

        self.model_name = model_name
        self.model = self.models[self.model_name]

        self.n_model_parameters = 2 * self.n_qubits * self.n_layers \
                                + self.n_trash_qubits
        self.theta = 2 * np.pi * np.random.uniform(
            low=0.0,
            high=1.0,
            size=(self.n_model_parameters)
        )

        """
        #this stuff is all needed later
        #self.loss = {"BCE":self.BCE}
        #self.delLoss = {"BCEderivative": self.BCEderivative}
        self.lossFunction = self.loss[lossfunc]
        self.lossDerivative = self.delLoss[delLoss]

        self.theta = 2*np.pi*np.random.uniform(0,1, size=(self.n_model_parameters))
        print(self.theta.shape)
        """

    def basic_model(
        self,
    ):
        """
        First try at a quantum autoencoder.
        """

        for layer in range(self.n_layers):
            for qubit in range(self.n_qubits):





        return



    def cost_function(
        self,
        parameters,
        circuit,
    ):
        """
        This function implements the Hamming distance cost function as described in
        the EF-QAE paper. Basically if we manage to obtain |0> on every trash qubit
        then we have achieved a complete compression into the latent qubits.
        """
        quantum_register = self.quantum_register
        classical_register = self.classical_register


        results = qk.execute(
            circuit,
            backend=Aer.get_backend('aer_simulator'),
            shots=self.shots,
        ).result().get_counts(circuit)

        for state, counts in results:
            z = np.zeros(self.n_trash_qubits)
            for idx, qubit in enumerate(state):
                if qubit == '0':
                    z[idx] += counts / self.shots
                else:
                    z[idx] -= counts / self.shots

        return (n_trash_qubits - np.sum(z))



    def make_model_circuit(
        self,
        print_circuit=False,
    ):
        self.quantum_register = qk.QuantumRegister(
            self.n_quantum,
            name='q_reg',
        )
        self.classical_register = qk.ClassicalRegister(
            self.n_classic,
            name='c_reg',
        )
        self.circuit = qk.QuantumCircuit(
            self.quantum_register,
            self.classical_register,
            name=self.model_name,
        )

        self.model()














if __name__=='__main__':

    dataset = generate_ising_dataset()

    #print(dataset)
    #print(type(dataset))

    n_qubits = 4
    compress = 3
    encoder = encoder_hamiltonian_simple(
        n_qubits,
        compress,
    )

    print(encoder.draw())
    def encoder_hamiltonian_simple(
        nqubits,
        ncompress
    ):
        m0 = K.to_numpy(hamiltonians.Z(ncompress).matrix)
        print(m0)
        m1 = np.eye(2 ** (nqubits - ncompress), dtype=m0.dtype)
        print(m1)
        ham = hamiltonians.Hamiltonian(nqubits, np.kron(m1, m0))
        return 0.5 * (ham + ncompress)



    def generate_ising_dataset(
        n_qubits=6
    ):
        ising_groundstates = []
        lambdas = np.linspace(0.5, 1.0, 20)
        for lamb in lambdas:
            ising_hamiltonian = -1 * hamiltonians.TFIM(n_qubits, h=lamb)
            ising_groundstates.append(ising_hamiltonian.eigenvectors()[0])

        return ising_groundstates


    def rotate(theta, x):
    new_theta = []
    index = 0
    for l in range(layers):
        for q in range(nqubits):
           new_theta.append(theta[index]*x + theta[index+1])
           index += 2
        for q in range(nqubits):
           new_theta.append(theta[index]*x + theta[index+1])
           index += 2
    for q in range(nqubits-compress, nqubits, 1):
        new_theta.append(theta[index]*x + theta[index+1])
        index += 2
    return new_theta

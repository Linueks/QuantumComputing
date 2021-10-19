import numpy as np
import qiskit as qk



def model(feature_vector, theta, shots=1000):
    n_quantum = feature_vector.shape[0]
    n_measure = 1

    q_reg = qk.QuantumRegister(n_quantum)
    c_reg = qk.ClassicalRegister(n_measure)
    circuit = qk.QuantumCircuit(q_reg, c_reg)

    for i, f in enumerate(feature_vector):
        # map the features
        circuit.ry(f, q_reg[i])
        # add model parameters to be tuned
        circuit.rx(theta[i], q_reg[i])

    for qubit in range(n_quantum - 1):
        # introduce entanglement to
        circuit.cx(q_reg[qubit], q_reg[qubit - 1])

    circuit.ry(theta[-1], q_reg[-1])
    circuit.measure(q_reg[-1], c_reg)

    #print(circuit.depth())
    print(circuit)

    job = qk.execute(circuit,
                    backend=qk.Aer.get_backend('qasm_simulator'),
                    shots=shots,
                    seed_simulator=2021
                    )
    results = job.result().get_counts(circuit)
    # this can be the model output, we want to tune the thetas
    # such that this output matches with our dataset
    output = results['0'] / shots

    return output


if __name__=='__main__':
    feature_vector = np.array([1.0, 1.5, 2.0, 0.3])
    target = np.array([0.7])

    epochs = 100
    learning_rate = 0.1

    # number of model parameters implemented in the circuit
    number_of_parameters = 5
    theta = np.random.randn(number_of_parameters)

    for epoch in range(epochs):
        y_pred = model(feature_vector, theta)
        mse = (y_pred - target[0])**2
        deriv_mse = 2 * (y_pred - target[0])

        grad_theta = np.zeros_like(theta)

        for param_idx in range(theta.shape[0]):
            theta[param_idx] += np.pi / 2
            out_1 = model(feature_vector, theta)

            theta[param_idx] -= np.pi
            out_2 = model(feature_vector, theta)
            theta[param_idx] += np.pi / 2
            grad_theta[param_idx] = (out_1 - out_2) / 2

        theta = theta - learning_rate * deriv_mse * grad_theta

        print(mse)

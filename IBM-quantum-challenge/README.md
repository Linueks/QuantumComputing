# Work on IBM Quantum Awards: Open Science Prize 2021 (using Qiskit defaults)



## Immediate TO-DO
- [x] Implement the most naive implementation of Trotter simulation on noisy backend.
- [ ] Think about preserved quantities and symmetries of the initial and final state.
- [ ] Look into usecases for the four "trash" qubits on the Jakarta system.
- [x] Look into symmetries of the Hamiltonian and think about what unitary gates preserve this symmetry.
- [x] The way of doing the Trotterization is not unique. Test different ways of splitting the product.
- [x] Learn about Mitiq Python package for error mitigation / post processing. Zero Noise Extrapolation. (Turned out fruitless)
- [ ] Think about using qubits 0, 2, 4, 6 to implement bit- or phase-flip error correction. 
- [ ] Learn about Qiskit circuit transpiler


## Preliminary Results

### Trotter Decompositions
Following are the two first different Trotter decompositions investigated. Left one is dubbed zyxzyx or cyclical while the right one is dubbed zzyyxx or ladder.

<p align="left">
  <img width="500" height="300" src="https://github.com/Linueks/QuantumMachineLearning/blob/main/IBM-quantum-challenge/figures/zyxzyx_trotter_step.png">
  <img width="500" height="300" src="https://github.com/Linueks/QuantumMachineLearning/blob/main/IBM-quantum-challenge/figures/zzyyxx_trotter_step.png">
</p>

Some things one can notice is that in the ladder formulation there are some gates that cancel against each other. Implementing the decomposition with these explicitly cancelled out we get.

### Varying Trotter Steps for the Two Most Naive Decompositions

Below is the absolute first result of testing the most basic implementation of our Trotterization. The results are not very good yet, however, they do meet the criteria for entry into the competition which is nice! I tested changing the ordering of the ZZ, XX, YY gates in the single trotter step circuit. We see it is better to first do all the operations on one set of qubits and then the other. We observe for the cyclical trotter decomposition that around 7-8 trotter steps performs optimally. This is the result on a noisy backend with absolutely no cancellations or care performed on either decomposition.

<p align="center">
  <img width="400" height="300" src="https://github.com/Linueks/QuantumMachineLearning/blob/main/IBM-quantum-challenge/figures/trotter_sim_4_16_shots8192_numjobs8.png">
</p>

Below is the error calculation when compared to the exact propagation as a function of Trotter steps for both of the above decompositions. This calculation assumes the gates are implemented with zero noise. 

<p align="center">
  <img width="400" height="300" src="https://github.com/Linueks/QuantumMachineLearning/blob/main/IBM-quantum-challenge/figures/error_comparison.png">
</p>

### Symmetry Protection
Implementing the simplest of the symmetry protections from [Tran et al.](https://github.com/Linueks/QuantumComputing/blob/main/IBM-quantum-challenge/papers/symmetry_protection_2006.16248.pdf) I was able to achieve a fairly significant leap in the fidelity for the zzyyxx decomposition using four Trotter steps on the noisy backend. Following are the results of the calculations

<p align="center">
  <img width="400" height="300" src="https://github.com/Linueks/QuantumComputing/blob/main/IBM-quantum-challenge/figures/trotter_sim_4_16_shots8192_numjobs2_SPTrue.png">
  <img width="400" height="300" src="https://github.com/Linueks/QuantumComputing/blob/main/IBM-quantum-challenge/figures/trotter_sim_4_8_shots8192_numjobs2_SPTrue.png">
</p>

The concept of symmetry protection is covered in depth in the paper. Basically you interweave unitaries that ensure that the symmetry of the specified Hamiltonian are being preserved. What I implemented follows directly from their example on the Heisenberg model:

<p align="center">
  <img width="500" height="300" src="https://github.com/Linueks/QuantumComputing/blob/main/IBM-quantum-challenge/figures/symmetry_protected_zzyyxx.png">
</p>

Here the odd steps are Hadamard-sandwiched while the even terms will have their Hadamards cancel to the identity. This setup with the zzyyxx decomposition and four Trotter steps yields a fidelity of ~54%. The circuit is below.

<p align="center">
  <img width="800" height="600" src="https://github.com/Linueks/QuantumComputing/blob/main/IBM-quantum-challenge/figures/SP_best.png">
</p>



## Ideas and Notes
Meant to be a section to quickly jot down notes and work out ideas.

### Zero Noise Extrapolation Using Mitiq
Unitary folding is a method for noise scaling that operates directly at the gate level. This makes it easy to use with current quantum computing libraries. It is especially appropriate when the underlying noise scales with the depth and/or the number of gates of a quantum program. 

In Mitiq, folding functions input a circuit and a scale factor, i.e., a floating point value which corresponds to (approximately) how much the length of the circuit is scaled. The minimum scale factor is one (which corresponds to folding no gates). A scale factor of three corresponds to folding all gates locally. Scale factors beyond three begin to fold gates more than once. For local folding, there is a degree of freedom for which gates to fold first. The order in which gates are folded can have an important effect on how the noise is scaled.

For IBM challenge Mitiq seems like it is not the way to go. Spent most of Jan. 18 messing with the code to get it working and it seems to not give better results at all. In fact the results are much worse. A different final state to the one it should be producing is consequently the output with higher probability than the unmitigated circuit. Not sure why this is. Also not sure if it is possible to use with Qiskit Tomography Circuits. Definitely something to keep in mind in other projects though! Their execute_with_zne function works when you have a backend executor taking a circuit as input with an expectation value as output. 

[Mitiq Getting Started](https://mitiq.readthedocs.io/en/stable/guide/guide-getting-started.html#guide-getting-started)


### Qiskit Error Mitigation Introduction
[Useful Qiskit Introduction](https://github.com/a-auer/qiskit/blob/master/EntanglementPurification.ipynb)



### Properties of the Initial and Final State
The initial and final states both have negative parity. 









## Sources
Some sources included by the Qiskit team has been downloaded from arxiv and put into the papers folder

- [Denoising Quantum Autoencoders](papers/denoising_quantum_autoencoder_1910.09169.pdf)
- [Digital Quantum Simulation of Spin Systems](papers/digital_qc_sim.pdf)
- [Quantum Computers as Universal Simulators](papers/qc_as_uqs_907.03505.pdf)
- [Mitigating Measurement Errors](papers/mitigating_measure_error_2006.14044.pdf)
- [Reducing Unitary Errors](papers/reducing_unitary_error_cross_resonance_2007.02925.pdf)



## IBM Quantum Awards: Open Science Prize 2021 (description)

IBM Quantum is excited to announce the fourth annual quantum awards (and the second annual Open Science Prize)—an award for those who can present an open source solution to some of the most pressing problems in the field of quantum computing.

This year, the challenge will feature one problem from the field of quantum simulation, solvable through one of two approaches. The best open source solution to each approach will receive a $40,000 prize, and the winner overall will receive another $20,000.

Simulating physical systems on quantum computers is a promising application of near-term quantum processors. This year's problem asks participants to simulate a Heisenberg model Hamiltonian for three interacting atoms on IBM Quantum's 7-qubit Jakarta system. The goal is to simulate the evolution of a known quantum state with the best fidelity as possible using Trotterization.

**Read more at our [blog](https://www.research.ibm.com/blog/quantum-open-science-prize) and register [here](https://ibmquantumawards.bemyapp.com)**.

The competition will conclude and judging will commence on  April 16, 2022.

Participants must choose one solution method and may submit their answer using 1) Qiskit Pulse or 2) solving the problem using Qiskit defaults, as outlined below:

- Each team or participant may only contribute to one submission
- Solution may only be executed on the designated device (ibmq_jakarta)
- Each submission must use Trotterization to evolve the specified state, under the specified Hamiltonian, for the specified duration (as outlined in the included Jupyter Notebook) with at least 4 Trotter steps.
- Only use Open Pulse and or pulsed gates functionality as outlined in the included Jupyter notebooks.
- Only use libraries that can be installed using either pip install or conda install and no purchased libraries.
- Document code with concise, clear language about the chosen methodology.
- State tomography fidelity (for 4 or more trotter steps) must meet a minimum value of 30%.

The submissions will be judged on the following criteria:
- Performance as measured by the state tomography fidelity in comparison to other submissions (Max 15 points)
- Clarity of provided documentation and solution code (Max 5 points)
- Creativity in developing a unique, innovative, and original solution (Max 5 points)

# QuantumComputingJourney

This repository documents my exploration of Quantum Computing. I use it to track experiments, note observations, and complete exercises from courses.

---

## Course 1: IBM — [Use a Quantum Computer Today](https://eu-de.quantum.cloud.ibm.com/learning/en/courses/use-a-qc-today)

### Part 1: Build and run your first quantum program

The exercise at the end of this first part was to create the circuits for other Bell states ($\Phi$<sup>-</sup>, $\Psi$<sup>+</sup> and $\Psi$<sup>-</sup>). I have created a Jupyter Notebook describing how I understood them with simple representations, and the results I got (hopefully the answers are correct). The notebook is [course1_part1.ipynb](course1_part1.ipynb).

### Part 2: Quantum mechanics basics

In this second part, the exercise was to work through the matrix algebra for the Bell states created in part 1. I have created another Jupyter notebook that shows the matrix algebra for the four circuits I created in part 1: [course1_part2.ipynb](course1_part2.ipynb). Also, I noted two ways of buiding the states $\Phi$<sup>-</sup> and $\Psi$<sup>-</sup>.

---

## Course 2: IBM — [Fundamentals of quantum algorithms](https://quantum.cloud.ibm.com/learning/en/courses/fundamentals-of-quantum-algorithms)

### Simon's algorithm

I've recreated Simon's algorithm with the example function given in the course. The notebook is [course2_simon.ipynb](course2_simon.ipynb).

---

## Quantum Fourier Transform with cards shuffling

An illustration of how the QFT can find a period inside a cyclic pattern with a deck of cards' shuffling (by modular multiplications). The notebook is [QFT_with_cards_shuffling.ipynb](QFT_with_cards_shuffling.ipynb).

---

## Hybrid Quantum-Classical Reinforcement Learning series

This series of experiments is based on the 2025 paper by Nagy et al.: "[Hybrid Quantum-Classical Reinforcement Learning in Latent Observation Spaces](https://arxiv.org/abs/2410.18284)". The aim is to compare the resources cost of classical, qubit-based and photonic-based PPO agents on different RL environments. 

The series is divided in three parts:
- [Part I: Classical vs Qubit agents on the Cart Pole environment](QRL_experiment_1.ipynb)
- [Part II: Classical vs Qubit agents on the Lunar Lander and Maze environments](QRL_experiment_2.ipynb)
- [Part III: Photonic agents on the three environments](QRL_experiment_3.ipynb)
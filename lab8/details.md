## Overview & Objectives

This project focuses on three primary tasks using Restricted Boltzmann Machines (RBM):
1. **Classification:** Predict the class label of a given input image.
2. **Reconstruction:** Recreate the original image from learned feature representations.
3. **Generation:** Sample and generate new image instances from the learned data distribution.

---

## What is a Boltzmann Machine?

A **Boltzmann Machine** is a stochastic generative neural network that learns a joint probability distribution over its neurons. It operates on an energy-based model framework, assigning an energy value to every configuration of its visible ($v$) and hidden ($h$) neurons:

$$E(v, h)$$

The joint probability of a configuration $(v, h)$ is derived using the Boltzmann distribution:

$$P(v, h) = \frac{e^{-E(v, h)}}{Z}$$

*where **$Z$** is the partition function (normalizing factor).*

> **Key Takeaway:** Lower energy configurations correspond to higher probabilities ($\text{Lower } E \implies \text{Higher } P$).

---

## Restricted Boltzmann Machine (RBM) Architecture

For this implementation, we use a **Restricted Boltzmann Machine (RBM)**—an efficient variant where no connections exist within the same layer (no visible-to-visible or hidden-to-hidden connections).

* **Visible Layer ($v$):** Represents the input image.
  * **Dimensions:** $28 \times 28$
  * **Input Vector:** 784 visible neurons
* **Hidden Layer ($h$):** Extracts latent feature representations.
  * **Baseline:** 64 hidden neurons *(scalable to 128 or 256 for experimentation)*
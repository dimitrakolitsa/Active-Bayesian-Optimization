# Optimization of Expensive Black-Box Functions with ML-Based Surrogate Modeling and Active Bayesian Learning

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![BoTorch](https://img.shields.io/badge/BoTorch-Enabled-purple)
![PyTorch](https://img.shields.io/badge/PyTorch-Enabled-red)
![GPyTorch](https://img.shields.io/badge/GPyTorch-Enabled-yellow)

This repository contains the source code and experimental framework for the Diploma Thesis titled **"Optimization of Expensive Black-Box Functions with ML-Based Surrogate Modeling and Active Bayesian Learning"**, conducted at the **Aristotle University of Thessaloniki (AUTh)**.

## 📖 Overview

The goal of this project is to optimize complex, expensive **Black-Box functions**, specifically in the context of Computer-Aided Engineering (CAE). The target application is the design optimization of an automotive subframe using **ANSA** simulation software.

Traditional evolutionary algorithms (like NSGA-II or Differential Evolution) require a significant amount of function evaluations, which is computationally prohibitive when a single simulation takes hours. This project implements advanced **Bayesian Optimization (BO)** strategies to significantly improve **sample efficiency**, finding optimal solutions with a fraction of the computational cost.

## 🚀 Key Features & Implementation

This repository implements two distinct optimization pipelines designed to handle different levels of problem complexity:

### 1. Basic BO Pipeline with Fallback Mechanism

Applied across **all problem categories** (Single, Constrained, Multi, and Constrained Multi-objective), this pipeline enhances standard Bayesian Optimization.

- **Stagnation Detection:** Automatically detects when the optimization process is trapped in a local optimum.
- **Fallback Trigger:** Switches the acquisition function (e.g., to Lower Confidence Bound - LCB) to force exploration and escape local optima.

### 2. Hybrid Adaptive BO Framework

Applied specifically to the most complex **Multi-Constrained** problems (Problems 7 & 8).

- **Dynamic Control:** A novel approach combining a Finite State Machine (FSM) controller with BO.
- **Adaptive Strategy:** Dynamically adjusts the trade-off between exploration and exploitation based on real-time metrics like Hypervolume improvement and model uncertainty.
- **CS-LCB:** Utilizes a custom _Constrained Scalarized Lower Confidence Bound_ acquisition function.

## 📂 Repository Structure

The repository is organized by problem category. Each directory contains the optimization scripts relevant to those specific problem definitions.

- **`/single`**

  - Contains optimization scripts for **Single Objective** problems (Problems 1 & 2).
  - _Implementation:_ Basic BO with Fallback (LEI).

- **`/single_constrained`**

  - Contains optimization scripts for **Constrained Single Objective** problems (Problems 3 & 4).
  - _Implementation:_ Basic BO with Fallback (LogCEI).

- **`/multi`**

  - Contains optimization scripts for **Multi-Objective** problems (Problems 5 & 6).
  - _Implementation:_ Basic BO with Fallback (EHVI/qLogEHVI).

- **`/multi_constrained`**

  - Contains optimization scripts for **Constrained Multi-Objective** problems (Problems 7 & 8).
  - _Implementation:_
    1. Basic BO with Fallback (qLogEHVI).
    2. **Hybrid Adaptive BO** (Custom CS-LCB + FSM).

- **`/setup_ansa_env`**
  - Contains the interface scripts required to connect the external Python optimization algorithms with the ANSA pre-processor/simulation environment.
  - `remote_ansa_evaluator.py`: The client-side script called by the optimizer to request a simulation.
  - `remote_ansa_worker.py`: The server-side script that runs **inside ANSA**, executes the simulation geometry changes, and returns results.

## 🛠️ Methodology

The core methodology revolves around an iterative cycle utilizing **BoTorch** and **GPyTorch**:

1.  **Surrogate Modeling:** Training Gaussian Processes (GPs) to approximate the expensive black-box function.
2.  **Acquisition Function:** Maximizing utility functions to select the next best design point.
3.  **Evaluation:** Using the `setup_ansa_env` interface to trigger simulations via the ANSA worker and retrieve results.
4.  **Model Update:** Retraining the GP with new data to refine predictions.

## ⚙️ Installation & Setup

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/dimitrakolitsa/Active-Bayesian-Optimization.git
    cd Active-Bayesian-Optimization
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    This project uses a specific set of requirements located in `env_requirements.txt`.
    ```bash
    pip install -r env_requirements.txt
    ```

## 📊 Results Highlights

The proposed methods demonstrated superior performance compared to standard evolutionary algorithms (DE, NSGA-II):

- **Sample Efficiency:** The BO methods achieved optimal results using **~80-90% fewer evaluations** than NSGA-II.
- **Convergence:** In Multi-objective problems (Problem 7 & 8), the **Hybrid Adaptive BO** framework achieved **95-99.9%** of the optimal Hypervolume of the standard BO approach in significantly less time.
- **Quality:** Discovered denser and broader Pareto fronts in complex constrained environments.

## 🎓 Thesis Details

- **Author:** Dimitra Georgia Kolitsa
- **Supervisor:** Prof. Panagiotis Petrantonakis
- **Institution:** Aristotle University of Thessaloniki, Dept. of Electrical & Computer Engineering.

## 🤝 Acknowledgments

Special thanks to **BETA CAE Systems** for providing the software tools (ANSA) and data utilized in the case studies of this thesis.

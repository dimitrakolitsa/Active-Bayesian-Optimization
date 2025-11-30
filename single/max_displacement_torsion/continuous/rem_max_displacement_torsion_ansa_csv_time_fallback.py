# PURPOSE: Main script to run a closed-loop Bayesian Optimization for 'max_displacement_torsion'
#          using the ANSA client-server (IAP) approach with a UCB fallback for stagnation.

import pandas as pd
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import LogExpectedImprovement
# NEW: Import the UpperConfidenceBound acquisition function for the fallback
from botorch.acquisition.analytic import UpperConfidenceBound
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel
import gpytorch
import matplotlib.pyplot as plt
import warnings
from botorch.exceptions import InputDataWarning
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import time
import sys
import json
import os
from matplotlib.lines import Line2D # For custom legend

from remote_ansa_evaluator import AnsaRemoteEvaluator, VARIABLE_NAMES

# --- Configuration Section ---
warnings.filterwarnings('ignore', category=InputDataWarning)
torch.manual_seed(42); np.random.seed(42); torch.set_default_dtype(torch.float64)

PROJECT_DIR = r"C:\Users\Dimitra\Desktop\25DVs_DM"
ANSA_WORKER_SCRIPT = os.path.join(PROJECT_DIR, "remote_ansa_worker.py")
RESULTS_CSV_PATH = "optimization_results_live_with_fallback.csv"

TARGET_TO_OPTIMIZE_NAME = "max_displacement_torsion"

BASE_PATH = '../../'
INITIAL_DATA_SIZE = 50
BASE_ITERS = 500
MAX_TOTAL_ITERS = 500
PATIENCE_WINDOW = 10
REL_IMPROVEMENT_TOL = 1e-5

# NEW: Parameters for stagnation detection and exploration fallback
STAGNATION_WINDOW = 15
EXPLORATION_DURATION = 5
BETA_EXPLORE = 2.5       # UCB beta for the exploration phase

# --- Variable Definitions and Initial Data Loading ---
bounds_dict = {
    "FrontRear_height": [0.0, 3.0], "side_height": [0.0, 5.0], "side_width": [0.0, 4.0], "holes": [-3.0, 4.0], "edge_fit": [0.0, 1.5], "rear_offset": [-3.0, 3.0],
    "PSHELL_1_T": [2.0, 3.25], "PSHELL_2_T": [2.0, 3.25], "PSHELL_42733768_T": [1.6, 2.6], "PSHELL_42733769_T": [1.6, 2.6], "PSHELL_42733770_T": [1.6, 2.6], "PSHELL_42733772_T": [1.6, 2.6],
    "PSHELL_42733773_T": [1.6, 2.6], "PSHELL_42733774_T": [1.6, 2.6], "PSHELL_42733779_T": [2.0, 3.25], "PSHELL_42733780_T": [1.6, 2.6], "PSHELL_42733781_T": [2.399952, 3.899922], "PSHELL_42733782_T": [1.599936, 2.599896],
    "PSHELL_42733871_T": [1.199888, 1.949818], "PSHELL_42733879_T": [2.4, 3.9], "MAT1_1_E": [110000.0, 250000.0], "MAT1_42733768_E": [110000.0, 250000.0],
    "scale_x": [0.0, 1.5], "scale_y": [0.0, 1.5], "scale_z": [0.0, 1.5],
}
stepped_vars = { "side_width": 0.1, "holes": 0.1, "edge_fit": 0.1, "rear_offset": 0.5 }
categorical_vars = {"scale_x": [0.0, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5], "scale_y": [0.0, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5], "scale_z": [0.0, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]}
df_init_raw = pd.read_csv(f'{BASE_PATH}init/inputs.txt', header=0, sep=',')
df_init_raw = df_init_raw[VARIABLE_NAMES]
target_files = [f"{BASE_PATH}init/{name}.txt" for name in ["mass", "max_displacement_bending", "max_displacement_torsion", "max_stress_bending", "max_stress_torsion"]]
target_column_names = ["mass", "max_displacement_bending", "max_displacement_torsion", "max_stress_bending", "max_stress_torsion"]
df_targets = pd.concat([pd.read_csv(file, header=None) for file in target_files], axis=1)
df_targets.columns = target_column_names
df_init_raw = pd.concat([df_init_raw, df_targets], axis=1).head(INITIAL_DATA_SIZE)

# --- Prepare and Save Initial Data to CSV ---
print(f"Saving initial {INITIAL_DATA_SIZE} data points to {RESULTS_CSV_PATH}")
df_for_csv = df_init_raw.copy()
df_for_csv['evaluation_type'] = 'initial_data'
df_for_csv['iteration_number'] = range(INITIAL_DATA_SIZE)
df_for_csv['duration_seconds'] = 0.0
df_for_csv['bo_selection_time'] = 0.0
#Add acquisition_function column to header
df_for_csv['acquisition_function'] = 'N/A'
CSV_HEADER = VARIABLE_NAMES + target_column_names + ['acquisition_function', 'evaluation_type', 'iteration_number', 'duration_seconds', 'bo_selection_time']
df_for_csv = df_for_csv[CSV_HEADER]
df_for_csv.to_csv(RESULTS_CSV_PATH, index=False)
print("Initial data saved.")

# --- Data Transformation ---
bounds_unscaled_list = [bounds_dict[name] for name in VARIABLE_NAMES]
bounds_unscaled = torch.tensor(bounds_unscaled_list, dtype=torch.float64).transpose(0, 1)

X_init = torch.tensor(df_init_raw[VARIABLE_NAMES].values, dtype=torch.float64)
Y_init = torch.tensor(df_init_raw[target_column_names].values, dtype=torch.float64)

def transform_candidate_to_real_world(candidate_unscaled_tensor):
    real_candidate_1d = candidate_unscaled_tensor.clone().squeeze(0)
    for i, name in enumerate(VARIABLE_NAMES):
        if name in categorical_vars:
            choices = torch.tensor(categorical_vars[name], dtype=torch.float64)
            closest_idx = torch.argmin(torch.abs(choices - real_candidate_1d[i]))
            real_candidate_1d[i] = choices[closest_idx]
        elif name in stepped_vars:
            step = stepped_vars[name]
            rounded_val = torch.round(real_candidate_1d[i] / step) * step
            real_candidate_1d[i] = rounded_val
    lower_bounds_unscaled_val = bounds_unscaled[0]
    upper_bounds_unscaled_val = bounds_unscaled[1]
    real_candidate_1d = torch.max(lower_bounds_unscaled_val, torch.min(upper_bounds_unscaled_val, real_candidate_1d))
    return real_candidate_1d.unsqueeze(0)


target_index_to_optimize = target_column_names.index(TARGET_TO_OPTIMIZE_NAME)
train_y_raw_neg = -Y_init[:, target_index_to_optimize].unsqueeze(-1)
x_scaler = MinMaxScaler(); x_scaler.fit(bounds_unscaled.numpy())
train_x_scaled = torch.tensor(x_scaler.transform(X_init.numpy()), dtype=torch.float64)
scaled_bounds = torch.tensor([[0.0] * X_init.shape[-1], [1.0] * X_init.shape[-1]], dtype=torch.float64)
y_scaler = StandardScaler()
train_y = torch.tensor(y_scaler.fit_transform(train_y_raw_neg.numpy()), dtype=torch.float64)

def train_gp_model(train_x, train_y):
    model = SingleTaskGP(train_x, train_y, covar_module=ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=train_x.shape[-1])))
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    try: 
        with gpytorch.settings.cholesky_jitter(1e-5):
            fit_gpytorch_mll(mll)
    except Exception as e: 
        print(f"  WARNING: GP model fitting failed: {e}"); return None
    return model

initial_best_raw = -train_y_raw_neg.max().item()
best_values = [initial_best_raw]
print(f"Initial best (minimum) target value: {initial_best_raw:.4f}")
print(f"\nStarting Live Bayesian Optimization with ANSA...")

evaluator = None
actual_iterations_run = 0
bo_start_time = time.perf_counter()
optimization_completed_successfully = False
last_successful_model = None
#state trackers for stagnation
stagnation_counter = 0
exploration_iters_left = 0

try:
    evaluator = AnsaRemoteEvaluator(project_dir=PROJECT_DIR, ansa_worker_script=ANSA_WORKER_SCRIPT)
    for i in range(MAX_TOTAL_ITERS):
        actual_iterations_run = i + 1
        print(f"\nIteration {i+1}/{MAX_TOTAL_ITERS}")
        iter_start_time = time.perf_counter()
        bo_selection_start_time = time.perf_counter()
        
        model = train_gp_model(train_x_scaled, train_y)
        if model is None:
            if last_successful_model is not None:
                print("  RECOVERY: GP fitting failed. Reusing model from previous iteration.")
                model = last_successful_model
            else:
                print("  FATAL: Initial GP model fitting failed. Stopping.")
                break
        else:
            last_successful_model = model

        # --- Acquisition Function Selection Logic ---
        acq_function_to_use = None
        acqf_name_this_iter = ""

        if exploration_iters_left > 0:
            acqf_name_this_iter = "custom_ucb"
            print(f"  Using EXPLORATION acquisition function ({acqf_name_this_iter}). {exploration_iters_left} iterations left.")
            acq_function_to_use = UpperConfidenceBound(model=model, beta=BETA_EXPLORE)
            exploration_iters_left -= 1
        else:
            acqf_name_this_iter = "logei"
            print(f"  Using standard acquisition function ({acqf_name_this_iter}).")
            acq_function_to_use = LogExpectedImprovement(model=model, best_f=train_y.max().item())

        try:
            next_x_scaled, _ = optimize_acqf(acq_function_to_use, bounds=scaled_bounds, q=1, num_restarts=15, raw_samples=5000)
        except Exception as e:
            print(f"  FATAL: Acquisition function optimization failed: {e}"); break
        bo_selection_duration = time.perf_counter() - bo_selection_start_time
        print(f"  BO point selection took: {bo_selection_duration:.2f} seconds.")
            
        next_x_unscaled_internal = torch.tensor(x_scaler.inverse_transform(next_x_scaled.numpy()))
        next_x_real_world_tensor = transform_candidate_to_real_world(next_x_unscaled_internal)
        sample_to_evaluate = dict(zip(VARIABLE_NAMES, [v.item() for v in next_x_real_world_tensor.squeeze(0)]))
        
        print("Sample to be evaluated by ANSA:"); print(json.dumps(sample_to_evaluate, indent=4))
        simulation_results = evaluator.evaluate(sample_to_evaluate)
        iteration_duration = time.perf_counter() - iter_start_time

        # Logic to save data now includes the acquisition function name
        if simulation_results is None or "error" in simulation_results or TARGET_TO_OPTIMIZE_NAME not in simulation_results:
            error_msg = simulation_results.get('error', f"'{TARGET_TO_OPTIMIZE_NAME}' not in results") if simulation_results else 'Evaluator returned None'
            print(f"  ANSA evaluation failed: {error_msg}. Saving failed point and skipping.")
            failed_row_data = sample_to_evaluate.copy()
            for col in target_column_names: failed_row_data[col] = np.nan
            failed_row_data.update({'acquisition_function': acqf_name_this_iter, 'evaluation_type': 'failed_evaluation', 'iteration_number': INITIAL_DATA_SIZE + i, 'duration_seconds': iteration_duration, 'bo_selection_time': bo_selection_duration})
            pd.DataFrame([failed_row_data])[CSV_HEADER].to_csv(RESULTS_CSV_PATH, mode='a', header=False, index=False)
            best_values.append(best_values[-1])
            continue
        else:
            new_row_data = sample_to_evaluate.copy()
            new_row_data.update(simulation_results)
            new_row_data.update({'acquisition_function': acqf_name_this_iter, 'evaluation_type': 'new_evaluation', 'iteration_number': INITIAL_DATA_SIZE + i, 'duration_seconds': iteration_duration, 'bo_selection_time': bo_selection_duration})
            pd.DataFrame([new_row_data])[CSV_HEADER].to_csv(RESULTS_CSV_PATH, mode='a', header=False, index=False)
            print(f"  Saved new evaluation to {RESULTS_CSV_PATH}")

            new_y_raw = simulation_results[TARGET_TO_OPTIMIZE_NAME]
            print(f"  Received result: {TARGET_TO_OPTIMIZE_NAME} = {new_y_raw:.4f}")
            new_y_neg_raw = -torch.tensor([[new_y_raw]], dtype=torch.float64)
            new_y_standardized = torch.tensor(y_scaler.transform(new_y_neg_raw.numpy()), dtype=torch.float64)
            
            next_x_scaled_for_retrain = torch.tensor(x_scaler.transform(next_x_real_world_tensor.numpy()), dtype=torch.float64)
            train_x_scaled = torch.cat([train_x_scaled, next_x_scaled_for_retrain], dim=0)
            train_y = torch.cat([train_y, new_y_standardized], dim=0)
            
            current_best_actual = min(best_values[-1], new_y_raw)
            best_values.append(current_best_actual)
            print(f"  Best value so far: {current_best_actual:.4f}")

        # Stagnation Check Logic
        if len(best_values) > STAGNATION_WINDOW:
            # look back over the last `STAGNATION_WINDOW` new evaluations
            past_best = best_values[-(STAGNATION_WINDOW + 1)]
            current_best = best_values[-1]

            if current_best >= past_best: # Stall if not improving (minimizing)
                stagnation_counter += 1
                print(f"   Stagnation counter incremented to: {stagnation_counter}")
            else:
                if stagnation_counter > 0: print("   Improvement detected, resetting stagnation counter.")
                stagnation_counter = 0

            # If stalled and not already exploring, trigger exploration phase
            if stagnation_counter >= STAGNATION_WINDOW and exploration_iters_left == 0:
                print(f"   STALLED! No improvement for {STAGNATION_WINDOW} iterations. Forcing exploration for {EXPLORATION_DURATION} iterations.")
                exploration_iters_left = EXPLORATION_DURATION
                stagnation_counter = 0 # Reset counter to avoid immediate re-triggering

        # Convergence Check
        if i >= BASE_ITERS - 1 and len(best_values) > PATIENCE_WINDOW:
            past_best = best_values[-(PATIENCE_WINDOW + 1)]
            current_best = best_values[-1]
            if abs(past_best) > 1e-9:
                relative_improvement = (past_best - current_best) / abs(past_best)
                if relative_improvement < REL_IMPROVEMENT_TOL:
                    print(f"\nSTOPPING: Relative improvement ({relative_improvement:.2e}) is below tolerance.")
                    break
    else:
        if actual_iterations_run == MAX_TOTAL_ITERS:
            print(f"\nSTOPPING: Reached maximum total iterations ({MAX_TOTAL_ITERS}).")
    optimization_completed_successfully = True

except Exception as e:
    print("\n" + "="*50); print(f"An unexpected error occurred: {e}"); 
    import traceback
    traceback.print_exc()
    print("Optimization stopped prematurely.")

finally:
    if evaluator:
        print("\nClosing persistent ANSA process...")
        evaluator.close()

bo_duration = time.perf_counter() - bo_start_time
print("\n" + "="*50); print("Optimization finished.")
print(f"Ran for {actual_iterations_run} new evaluations in {bo_duration:.2f} seconds ({bo_duration/60:.2f} minutes).")

if not optimization_completed_successfully or len(best_values) <= 1:
    print("\nNo new valid results were obtained during the optimization."); sys.exit()

best_actual_value = best_values[-1]

train_y_actual_scale = -y_scaler.inverse_transform(train_y.numpy())
best_idx_in_train = np.argmin(np.abs(train_y_actual_scale.flatten() - best_actual_value))
best_x_unscaled = torch.tensor(x_scaler.inverse_transform(train_x_scaled[best_idx_in_train].reshape(1, -1)))
best_x_real_tensor = transform_candidate_to_real_world(best_x_unscaled)
best_features_dict = {name: val.item() for name, val in zip(VARIABLE_NAMES, best_x_real_tensor.squeeze(0))}

print(f"\nBest observed value ({TARGET_TO_OPTIMIZE_NAME}): {best_actual_value:.6f}")
print(f"Achieved with parameters:"); print(pd.Series(best_features_dict))

# --- Plotting ---
fig, ax = plt.subplots(figsize=(12, 7))
iterations = list(range(len(best_values)))
plot_values = best_values
ax.plot(iterations, plot_values, marker='o', linestyle='-', label='Best Value After Each Evaluation')
legend_elements = [Line2D([0], [0], color='w', marker='', linestyle='', label=f'Minimum Value Found: {best_actual_value:.4f}')]
handles, labels = ax.get_legend_handles_labels()
handles.extend(legend_elements)
ax.legend(handles=handles, loc='upper right')
ax.set_xlabel("ANSA Evaluation Number (0 = Initial State)")
ax.set_ylabel(f"Best Observed Minimum {TARGET_TO_OPTIMIZE_NAME}")
ax.set_title(f"Live Bayesian Optimization with ANSA - Minimizing {TARGET_TO_OPTIMIZE_NAME}")
ax.grid(True, which='both', linestyle='--')
ax.set_xlim(left=0)
plt.tight_layout()
plt.savefig(f'live_optimization_progress_with_fallback.png')
print(f"\nSaved progress plot to 'live_optimization_progress_with_fallback.png'")
plt.show()
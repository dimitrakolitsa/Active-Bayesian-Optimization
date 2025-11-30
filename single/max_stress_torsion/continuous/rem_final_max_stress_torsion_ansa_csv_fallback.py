import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import warnings
import time
import sys
import json
import os
import traceback

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import LogExpectedImprovement
from botorch.acquisition.analytic import UpperConfidenceBound
from botorch.optim import optimize_acqf

import gpytorch
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from botorch.exceptions import InputDataWarning, BadInitialCandidatesWarning, UnsupportedError
from matplotlib.lines import Line2D

# Import the live evaluator
from remote_ansa_evaluator import AnsaRemoteEvaluator, VARIABLE_NAMES


# --- Configuration Section ---
warnings.filterwarnings('ignore', category=InputDataWarning)
warnings.filterwarnings('ignore', category=BadInitialCandidatesWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
torch.manual_seed(42)
np.random.seed(42)
torch.set_default_dtype(torch.float64)

PROJECT_DIR = r"C:\Users\Dimitra\Desktop\25DVs_DM"
ANSA_WORKER_SCRIPT = os.path.join(PROJECT_DIR, "remote_ansa_worker.py")

OUTPUT_ALL_POINTS_CSV_FILENAME = 'bo_single_obj_with_fallback_v2.csv' # v2 to avoid overwriting
TARGET_TO_OPTIMIZE_NAME = "max_stress_torsion"

BASE_PATH = '../../'
INITIAL_DATA_SIZE = 50
MAX_TOTAL_ITERS = 500

STAGNATION_WINDOW = 15
EXPLORATION_DURATION = 5
BETA_EXPLORE = 2.5

# --- Helper Function for Data Loading (unchanged) ---
def load_initial_data(base_path, variable_names, target_files_map, initial_size):
    print("--- Starting Initial Data Loading ---")
    try:
        init_dir = os.path.join(base_path, 'init')
        input_vars_path = os.path.join(init_dir, 'inputs.txt')
        df_inputs = pd.read_csv(input_vars_path, header=0, sep=',')
        missing_cols = set(variable_names) - set(df_inputs.columns)
        if missing_cols:
            raise ValueError(f"Your 'inputs.txt' is missing required columns: {sorted(list(missing_cols))}")
        target_dfs = []
        for col_name, file_name in target_files_map.items():
            target_file_path = os.path.join(init_dir, file_name)
            df_target_col = pd.read_csv(target_file_path, header=None, names=[col_name])
            target_dfs.append(df_target_col)
        df_targets = pd.concat(target_dfs, axis=1)
        if len(df_inputs) != len(df_targets):
            raise ValueError(f"Row count mismatch! 'inputs.txt' has {len(df_inputs)} rows, but target files have {len(df_targets)} rows.")
        df_inputs_reordered = df_inputs[variable_names]
        df_combined = pd.concat([df_inputs_reordered, df_targets], axis=1)
        if len(df_combined) < initial_size:
            warnings.warn(f"Requested INITIAL_DATA_SIZE of {initial_size} is larger than the available data ({len(df_combined)}). Using all available data.")
            initial_size = len(df_combined)
        df_final = df_combined.head(initial_size)
        print(f"Successfully loaded and validated {len(df_final)} initial data points.")
        print("-----------------------------------")
        return df_final
    except FileNotFoundError as e:
        print(f"FATAL ERROR: File not found during initial data loading: {e.filename}"); sys.exit(1)
    except ValueError as e:
        print(f"FATAL ERROR: Data validation failed: {e}"); sys.exit(1)


# --- Variable Definitions (unchanged) ---
bounds_dict = {
    "FrontRear_height": [0.0, 3.0], "side_height": [0.0, 5.0], "side_width": [0.0, 4.0], "holes": [-3.0, 4.0], "edge_fit": [0.0, 1.5], "rear_offset": [-3.0, 3.0],
    "PSHELL_1_T": [2.0, 3.25], "PSHELL_2_T": [2.0, 3.25], "PSHELL_42733768_T": [1.6, 2.6], "PSHELL_42733769_T": [1.6, 2.6], "PSHELL_42733770_T": [1.6, 2.6], "PSHELL_42733772_T": [1.6, 2.6],
    "PSHELL_42733773_T": [1.6, 2.6], "PSHELL_42733774_T": [1.6, 2.6], "PSHELL_42733779_T": [2.0, 3.25], "PSHELL_42733780_T": [1.6, 2.6], "PSHELL_42733781_T": [2.399952, 3.899922], "PSHELL_42733782_T": [1.599936, 2.599896],
    "PSHELL_42733871_T": [1.199888, 1.949818], "PSHELL_42733879_T": [2.4, 3.9], "MAT1_1_E": [110000.0, 250000.0], "MAT1_42733768_E": [110000.0, 250000.0],
    "scale_x": [0.0, 1.5], "scale_y": [0.0, 1.5], "scale_z": [0.0, 1.5],
}
stepped_vars = { "side_width": 0.1, "holes": 0.1, "edge_fit": 0.1, "rear_offset": 0.5 }
categorical_vars = {"scale_x": [0.0, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5], "scale_y": [0.0, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5], "scale_z": [0.0, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]}

# --- Data Loading & Preprocessing ---
target_column_names = ["mass", "max_displacement_bending", "max_displacement_torsion", "max_stress_bending", "max_stress_torsion"]
target_files_to_load = {name: f"{name}.txt" for name in target_column_names}
df_init_raw = load_initial_data(
    base_path=BASE_PATH, variable_names=VARIABLE_NAMES,
    target_files_map=target_files_to_load, initial_size=INITIAL_DATA_SIZE
)

bounds_unscaled_list = []
for name in VARIABLE_NAMES:
    bounds_unscaled_list.append(bounds_dict[name])
bounds_unscaled = torch.tensor(bounds_unscaled_list, dtype=torch.float64).transpose(0, 1)
df_init_processed_x = df_init_raw[VARIABLE_NAMES].copy()
X_init = torch.tensor(df_init_processed_x.values, dtype=torch.float64)
Y_init_full_responses = torch.tensor(df_init_raw[target_column_names].values, dtype=torch.float64)

# --- Helper Functions ---
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

def save_evaluated_points_to_csv(iteration_number, design_variables, objective_value, eval_type, filename, header_written, acqf_name, duration_seconds=np.nan, bo_selection_duration_seconds=np.nan):
    if isinstance(design_variables, list):
        design_variables = dict(zip(VARIABLE_NAMES, design_variables))
    row_data = {**design_variables, TARGET_TO_OPTIMIZE_NAME: objective_value}
    row_data['evaluation_type'] = eval_type; row_data['iteration_number'] = iteration_number
    row_data['duration_seconds'] = duration_seconds; row_data['bo_selection_duration_seconds'] = bo_selection_duration_seconds
    row_data['acquisition_function'] = acqf_name
    all_column_names = VARIABLE_NAMES + [TARGET_TO_OPTIMIZE_NAME, 'acquisition_function', 'evaluation_type', 'iteration_number', 'duration_seconds', 'bo_selection_duration_seconds']
    df_row = pd.DataFrame([row_data])
    df_row_ordered = df_row[all_column_names]
    mode = 'a' if header_written else 'w'
    header = not header_written
    df_row_ordered.to_csv(filename, index=False, mode=mode, header=header)

# --- Data Scaling & Model Training ---
target_index_to_optimize = target_column_names.index(TARGET_TO_OPTIMIZE_NAME)
train_y_raw_neg = -Y_init_full_responses[:, target_index_to_optimize].unsqueeze(-1)
x_scaler = MinMaxScaler(); x_scaler.fit(bounds_unscaled.numpy())
train_x_scaled = torch.tensor(x_scaler.transform(X_init.numpy()), dtype=torch.float64)
scaled_bounds = torch.tensor([[0.0] * X_init.shape[-1], [1.0] * X_init.shape[-1]], dtype=torch.float64)
y_scaler = StandardScaler()
train_y = torch.tensor(y_scaler.fit_transform(train_y_raw_neg.numpy()), dtype=torch.float64)

# MODIFIED: Function to train the GP model with improved stability
def train_gp_model(train_x, train_y):
    if torch.std(train_y) < 1e-6:
        print("  WARNING: train_y has near-zero standard deviation. Adding jitter.")
        train_y = train_y + torch.randn_like(train_y) * 1e-5
    model = SingleTaskGP(train_x, train_y, covar_module=ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=train_x.shape[-1])))
    model.likelihood.noise_constraint = gpytorch.constraints.GreaterThan(1e-6)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    try:
        # NEW: Added cholesky_jitter for numerical stability
        with gpytorch.settings.cholesky_jitter(1e-5):
            fit_gpytorch_mll(mll, max_retries=5)
    except Exception as e:
        print(f"  WARNING: GP model fitting failed: {e}") # Changed from FATAL to WARNING
        return None
    return model

# --- BO Loop Setup ---
initial_best_raw = -train_y_raw_neg.max().item()
best_values = [initial_best_raw]
print(f"Initial best (minimum) target value: {initial_best_raw:.4f}")

stagnation_counter = 0
exploration_iters_left = 0
csv_header_written = False
evaluated_x_hashes = set()

print(f"Saving initial {INITIAL_DATA_SIZE} points to '{OUTPUT_ALL_POINTS_CSV_FILENAME}'...")
for i in range(INITIAL_DATA_SIZE):
    real_world_dv_series = df_init_raw.loc[i, VARIABLE_NAMES]
    design_vars_dict = real_world_dv_series.to_dict()
    real_world_for_hashing_tensor = torch.tensor(real_world_dv_series.values, dtype=torch.float64).unsqueeze(0)
    evaluated_x_hashes.add(real_world_for_hashing_tensor.numpy().flatten().tobytes())
    obj_val = Y_init_full_responses[i, target_index_to_optimize].item()
    save_evaluated_points_to_csv(
        iteration_number=i, design_variables=design_vars_dict, objective_value=obj_val,
        eval_type='initial_data', filename=OUTPUT_ALL_POINTS_CSV_FILENAME,
        header_written=csv_header_written, acqf_name='N/A',
        duration_seconds=0.0, bo_selection_duration_seconds=0.0
    )
    if not csv_header_written: csv_header_written = True
print(f"Initial data saved. Initialized with {len(evaluated_x_hashes)} unique real-world points.")

print(f"\nStarting Live Bayesian Optimization with ANSA - Minimizing {TARGET_TO_OPTIMIZE_NAME}...")
evaluator = None; actual_iterations_run = 0; bo_start_time = time.perf_counter(); optimization_completed_successfully = False
# NEW: Variable to hold the last good model
last_successful_model = None

try:
    evaluator = AnsaRemoteEvaluator(project_dir=PROJECT_DIR, ansa_worker_script=ANSA_WORKER_SCRIPT)
    for i in range(MAX_TOTAL_ITERS):
        actual_iterations_run = i + 1
        print(f"\nIteration {i+1}/{MAX_TOTAL_ITERS}")
        iter_start_time = time.perf_counter(); bo_selection_start_time = time.perf_counter()
        
        # --- MODIFIED: Model training with recovery mechanism ---
        model = train_gp_model(train_x_scaled, train_y)
        if model is None:
            if last_successful_model is not None:
                print("  RECOVERY: GP fitting failed. Reusing model from the previous successful iteration.")
                model = last_successful_model
            else:
                # This would only happen if the very first model fit fails
                print("  FATAL: Initial GP model fitting failed and no previous model to fall back on. Stopping.")
                break
        else:
            # If training was successful, update our backup model
            last_successful_model = model

        # --- Acquisition Function Selection (unchanged) ---
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
            next_x_scaled_optimizer_proposal, _ = optimize_acqf(acq_function_to_use, bounds=scaled_bounds, q=1, num_restarts=15, raw_samples=5000)
        except Exception as e:
            print(f"  FATAL: Acquisition function optimization failed: {e}"); traceback.print_exc(); break
        bo_selection_duration = time.perf_counter() - bo_selection_start_time
        print(f"  BO point selection took: {bo_selection_duration:.2f} seconds.")
        
        # --- The rest of the loop is mostly the same, with robust failure checks ---
        next_x_unscaled_from_optimizer = torch.tensor(x_scaler.inverse_transform(next_x_scaled_optimizer_proposal.numpy()))
        next_x_real_world_for_ansa = transform_candidate_to_real_world(next_x_unscaled_from_optimizer)
        candidate_hash = next_x_real_world_for_ansa.numpy().flatten().tobytes()
        sample_to_evaluate = dict(zip(VARIABLE_NAMES, [v.item() for v in next_x_real_world_for_ansa.squeeze(0)]))

        def log_and_skip(eval_type, full_response, obj_val=np.nan):
            iteration_duration = time.perf_counter() - iter_start_time
            error_msg = "Unknown failure"
            if full_response is None: error_msg = "Evaluator returned None"
            elif "error" in full_response: error_msg = full_response.get('error', 'error key present but empty')
            elif TARGET_TO_OPTIMIZE_NAME not in full_response: error_msg = f"Key '{TARGET_TO_OPTIMIZE_NAME}' not in results."
            print(f"  ANSA evaluation failed: {error_msg}")
            save_evaluated_points_to_csv(
                i + INITIAL_DATA_SIZE, sample_to_evaluate, obj_val, eval_type,
                OUTPUT_ALL_POINTS_CSV_FILENAME, csv_header_written, acqf_name_this_iter,
                duration_seconds=iteration_duration, bo_selection_duration_seconds=bo_selection_duration
            )
            best_values.append(best_values[-1])
        
        if candidate_hash in evaluated_x_hashes:
            print("  WARNING: Duplicate REAL-WORLD point proposed. Skipping evaluation.")
            log_and_skip('skipped_duplicate', {'error': 'Duplicate point proposed'}); continue

        simulation_results = evaluator.evaluate(sample_to_evaluate)
        if (simulation_results is None or "error" in simulation_results or TARGET_TO_OPTIMIZE_NAME not in simulation_results):
            log_and_skip('failed_evaluation', simulation_results); continue
        
        new_y_raw = simulation_results[TARGET_TO_OPTIMIZE_NAME]
        print(f"  Received result: {TARGET_TO_OPTIMIZE_NAME} = {new_y_raw:.4f}")
        evaluated_x_hashes.add(candidate_hash)
        iteration_duration = time.perf_counter() - iter_start_time
        save_evaluated_points_to_csv(
            i + INITIAL_DATA_SIZE, sample_to_evaluate, new_y_raw, 'new_evaluation',
            OUTPUT_ALL_POINTS_CSV_FILENAME, csv_header_written, acqf_name_this_iter,
            duration_seconds=iteration_duration, bo_selection_duration_seconds=bo_selection_duration
        )
        print(f"  Point for iteration {i+1} saved.")

        new_y_neg_raw = -torch.tensor([[new_y_raw]], dtype=torch.float64)
        new_y_standardized = torch.tensor(y_scaler.transform(new_y_neg_raw.numpy()), dtype=torch.float64)
        next_x_scaled_for_retraining = torch.tensor(x_scaler.transform(next_x_real_world_for_ansa.numpy()), dtype=torch.float64)
        train_x_scaled = torch.cat([train_x_scaled, next_x_scaled_for_retraining], dim=0)
        train_y = torch.cat([train_y, new_y_standardized], dim=0)

        current_best_actual = min(best_values[-1], new_y_raw)
        best_values.append(current_best_actual)
        print(f"  Best value so far: {current_best_actual:.4f}")

        if len(best_values) > STAGNATION_WINDOW:
            past_best = best_values[-(STAGNATION_WINDOW + 1)]
            current_best = best_values[-1]
            if current_best >= past_best:
                stagnation_counter += 1
                print(f"   Stagnation counter incremented to: {stagnation_counter}")
            else:
                if stagnation_counter > 0: print("   Improvement detected, resetting stagnation counter.")
                stagnation_counter = 0
            if stagnation_counter >= STAGNATION_WINDOW and exploration_iters_left == 0:
                print(f"   STALLED! No improvement for {STAGNATION_WINDOW} iterations. Forcing exploration for {EXPLORATION_DURATION} iterations.")
                exploration_iters_left = EXPLORATION_DURATION
                stagnation_counter = 0

    else:
        if actual_iterations_run == MAX_TOTAL_ITERS:
            print(f"\nSTOPPING: Reached maximum total iterations ({MAX_TOTAL_ITERS}).")
    optimization_completed_successfully = True

except Exception as e:
    print("\n" + "="*50); print(f"An unexpected error occurred: {e}"); traceback.print_exc(); print("Optimization stopped prematurely.")
finally:
    if evaluator: print("\nClosing persistent ANSA process..."); evaluator.close()

# --- Final Reporting & Plotting ---
bo_duration = time.perf_counter() - bo_start_time
print("\n" + "="*50); print("Optimization finished.")
print(f"Ran for {actual_iterations_run} new evaluations in {bo_duration:.2f} seconds ({bo_duration/60:.2f} minutes).")

if not optimization_completed_successfully and len(best_values) <= 1:
    print("\nNo new valid results were obtained during the optimization."); sys.exit()

best_actual_value = best_values[-1]
train_y_actual_scale = -y_scaler.inverse_transform(train_y.numpy())
best_idx_in_train = np.argmin(np.abs(train_y_actual_scale.flatten() - best_actual_value))
best_x_scaled_from_train = train_x_scaled[best_idx_in_train]
best_x_unscaled_internal = torch.tensor(x_scaler.inverse_transform(best_x_scaled_from_train.reshape(1, -1)))
best_x_real = transform_candidate_to_real_world(best_x_unscaled_internal)
best_features_dict = {name: val.item() for name, val in zip(VARIABLE_NAMES, best_x_real.squeeze(0))}
print(f"\nBest observed value ({TARGET_TO_OPTIMIZE_NAME}): {best_actual_value:.6f}")
print(f"Achieved with parameters:")
print(pd.Series(best_features_dict))

# --- Plotting Convergence ---
fig, ax = plt.subplots(figsize=(12, 7))
num_new_evals = actual_iterations_run
iterations_for_plot = list(range(num_new_evals + 1))
plot_values = [initial_best_raw] + best_values[INITIAL_DATA_SIZE : INITIAL_DATA_SIZE + num_new_evals]

ax.plot(iterations_for_plot, plot_values, marker='o', linestyle='-', label='Best Value After Each Evaluation')

legend_elements = [
    Line2D([0], [0], color='w', marker='', linestyle='',
           label=f'Minimum Value Found: {best_actual_value:.4f}')
]
handles, labels = ax.get_legend_handles_labels()
handles.extend(legend_elements)

ax.legend(handles=handles, loc='upper right')
ax.set_xlabel("ANSA Evaluation Number (0 = Initial State)")
ax.set_ylabel(f"Best Observed Minimum {TARGET_TO_OPTIMIZE_NAME}")
ax.set_title(f"Live Bayesian Optimization with ANSA - Minimizing {TARGET_TO_OPTIMIZE_NAME}")
ax.grid(True, which='both', linestyle='--')
ax.set_xlim(left=0)

plt.tight_layout()
plt.savefig(f'live_optimization_progress_IAP.png')
print("\nSaved progress plot to 'live_optimization_progress_IAP_fallback.png'")
plt.show()
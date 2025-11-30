import pandas as pd
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.analytic import LogConstrainedExpectedImprovement
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

# Import Normal distribution for the custom acquisition function
from torch.distributions import Normal

from remote_ansa_evaluator import AnsaRemoteEvaluator, VARIABLE_NAMES

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# --- Configuration Section ---
warnings.filterwarnings('ignore', category=InputDataWarning)
torch.manual_seed(42)
np.random.seed(42)
torch.set_default_dtype(torch.float64)

# --- GPU/CPU Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

# --- Project and ANSA script paths ---
PROJECT_DIR = r"C:\Users\Dimitra\Desktop\25DVs_DM"
ANSA_WORKER_SCRIPT = os.path.join(PROJECT_DIR, "remote_ansa_worker.py")
RESULTS_CSV_PATH = "constrained_bo_results_with_fallback.csv"


# --- Optimization settings ---
new_columns = VARIABLE_NAMES
objective_target_name = "max_displacement_torsion"
constraint_target_name = "mass"
constraint_threshold = 0.018

# File paths for initial data
BASE_PATH = '../../'
INITIAL_DATA_SIZE = 50

# BO loop settings
MAX_TOTAL_ITERS = 100 #500

# Parameters for stagnation detection and exploration fallback
STAGNATION_WINDOW = 15
EXPLORATION_DURATION = 5
BETA_EXPLORE = 2.5

# --- Initial Data Loading ---
df_init_raw = pd.read_csv(f'{BASE_PATH}init/inputs.txt', header=0, sep=',')
df_init_raw = df_init_raw[new_columns]

target_column_names = ["mass", "max_displacement_bending", "max_displacement_torsion", "max_stress_bending", "max_stress_torsion"]
target_files = [f"{BASE_PATH}init/{name}.txt" for name in target_column_names]

df_targets = pd.concat([pd.read_csv(file, header=None) for file in target_files], axis=1)
df_targets.columns = target_column_names
df_init_raw = pd.concat([df_init_raw, df_targets], axis=1).head(INITIAL_DATA_SIZE)

# --- Prepare and Save Initial Data to CSV ---
print(f"\nSaving initial {INITIAL_DATA_SIZE} data points to {RESULTS_CSV_PATH}")
df_for_csv = df_init_raw.copy()
df_for_csv['evaluation_type'] = 'initial_data'
df_for_csv['iteration_number'] = range(INITIAL_DATA_SIZE)
df_for_csv['bo_duration_sec'] = 0
df_for_csv['iteration_duration_sec'] = 0
df_for_csv['acquisition_function'] = 'N/A'

CSV_HEADER = new_columns + target_column_names + ['acquisition_function', 'bo_duration_sec', 'iteration_duration_sec', 'evaluation_type', 'iteration_number']

df_for_csv = df_for_csv[CSV_HEADER]
df_for_csv.to_csv(RESULTS_CSV_PATH, index=False)
print("Initial data saved.")

# --- Data Transformation ---
X_init = torch.tensor(df_init_raw[new_columns].values, dtype=torch.float64)
Y_init = torch.tensor(df_init_raw[target_column_names].values, dtype=torch.float64)

objective_index = target_column_names.index(objective_target_name)
constraint_index = target_column_names.index(constraint_target_name)

train_y_raw = torch.cat([
    -Y_init[:, objective_index].unsqueeze(-1),
    Y_init[:, constraint_index].unsqueeze(-1)
], dim=1)

print(f"\nOptimizing objective: Minimize {objective_target_name}")
print(f"Subject to constraint: {constraint_target_name} < {constraint_threshold}")

# --- Scaling ---
bounds_dict = {
    "FrontRear_height": [0.0, 3.0], "side_height": [0.0, 5.0], "side_width": [0.0, 4.0], "holes": [-3.0, 4.0], "edge_fit": [0.0, 1.5], "rear_offset": [-3.0, 3.0],
    "PSHELL_1_T": [2.0, 3.25], "PSHELL_2_T": [2.0, 3.25], "PSHELL_42733768_T": [1.6, 2.6], "PSHELL_42733769_T": [1.6, 2.6], "PSHELL_42733770_T": [1.6, 2.6], "PSHELL_42733772_T": [1.6, 2.6],
    "PSHELL_42733773_T": [1.6, 2.6], "PSHELL_42733774_T": [1.6, 2.6], "PSHELL_42733779_T": [2.0, 3.25], "PSHELL_42733780_T": [1.6, 2.6], "PSHELL_42733781_T": [2.399952, 3.899922], "PSHELL_42733782_T": [1.599936, 2.599896],
    "PSHELL_42733871_T": [1.199888, 1.949818], "PSHELL_42733879_T": [2.4, 3.9], "MAT1_1_E": [110000.0, 250000.0], "MAT1_42733768_E": [110000.0, 250000.0],
    "scale_x": [0.0, 1.5], "scale_y": [0.0, 1.5], "scale_z": [0.0, 1.5],
}
stepped_vars = { "side_width": 0.1, "holes": 0.1, "edge_fit": 0.1, "rear_offset": 0.5 }
categorical_vars = {"scale_x": [0.0, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5], "scale_y": [0.0, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5], "scale_z": [0.0, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]}
bounds_unscaled = torch.tensor([bounds_dict[name] for name in new_columns], dtype=torch.float64).transpose(0, 1)

x_scaler = MinMaxScaler()
x_scaler.fit(bounds_unscaled.numpy())
train_x_scaled = torch.tensor(x_scaler.transform(X_init.numpy()), dtype=torch.float64).to(device)
scaled_bounds = torch.tensor([[0.0] * X_init.shape[-1], [1.0] * X_init.shape[-1]], dtype=torch.float64).to(device)

y_scaler = StandardScaler()
train_y_standardized = torch.tensor(y_scaler.fit_transform(train_y_raw.numpy()), dtype=torch.float64).to(device)
constraint_threshold_standardized = y_scaler.transform(torch.tensor([[0.0, constraint_threshold]], dtype=torch.float64))[:, 1].item()

def train_multi_output_gp_model(train_x, train_y):
    model = SingleTaskGP(train_x, train_y, covar_module=ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=train_x.shape[-1])))
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    try: fit_gpytorch_mll(mll)
    except Exception as e: print(f"  WARNING: GP fitting failed: {e}");
    return model

def transform_candidate_to_real_world(candidate_tensor):
    real_candidate = candidate_tensor.clone().squeeze()
    for i, name in enumerate(new_columns):
        if name in categorical_vars:
            choices = categorical_vars[name]
            val_unscaled = real_candidate[i]
            closest_idx = torch.argmin(torch.abs(torch.tensor(choices) - val_unscaled))
            real_candidate[i] = choices[closest_idx]
        elif name in stepped_vars:
            step = stepped_vars[name]
            real_candidate[i] = torch.round(real_candidate[i] / step) * step
    return real_candidate

# --- BO loop Setup ---
best_feasible_values = []
actual_iterations_run = 0
initial_mass_values = Y_init[:, constraint_index]
feasible_initial_mask = initial_mass_values < constraint_threshold
if feasible_initial_mask.any():
    initial_best_objective_raw = Y_init[:, objective_index][feasible_initial_mask].min().item()
    best_feasible_values.append(initial_best_objective_raw)
    print(f"\nInitial best feasible objective value: {initial_best_objective_raw:.6f}")
else:
    initial_best_objective_raw = float('inf')
    best_feasible_values.append(initial_best_objective_raw)
    print("\nNo initial feasible points found.")

stagnation_counter = 0
exploration_iters_left = 0
norm = Normal(0.0, 1.0)


print(f"\nStarting Live Constrained Bayesian Optimization...")
bo_total_start_time = time.perf_counter()

evaluator = None
try:
    evaluator = AnsaRemoteEvaluator(project_dir=PROJECT_DIR, ansa_worker_script=ANSA_WORKER_SCRIPT)

    # --- Main BO Loop ---
    for i in range(MAX_TOTAL_ITERS):
        actual_iterations_run = i + 1
        iter_start_time = time.perf_counter()
        
        print(f"\nIteration {i+1}/{MAX_TOTAL_ITERS}")
        
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        bo_start_time_iter = time.perf_counter()

        model = train_multi_output_gp_model(train_x_scaled, train_y_standardized)

        # --- Acquisition Function Selection Logic ---
        acq_function_to_use = None
        acqf_name_this_iter = ""

        if exploration_iters_left > 0:
            acqf_name_this_iter = "custom"
            print(f"  Using EXPLORATION acquisition function ({acqf_name_this_iter}). {exploration_iters_left} iterations left.")
            
            def exploration_acqf(X):
                posterior = model.posterior(X)
                means, stds = posterior.mean, posterior.variance.clamp_min(1e-9).sqrt()
                pof = norm.cdf((constraint_threshold_standardized - means[..., 1]) / stds[..., 1])
                lcb = means[..., 0] - BETA_EXPLORE * stds[..., 0]
                return (pof * lcb).squeeze(-1)
            
            acq_function_to_use = exploration_acqf
            exploration_iters_left -= 1
        else:
            acqf_name_this_iter = "logcei"
            print(f"  Using standard acquisition function ({acqf_name_this_iter}).")
            
            current_y_standardized_cpu = train_y_standardized.cpu().numpy()
            current_objective_values_raw = -y_scaler.inverse_transform(current_y_standardized_cpu)[:, 0]
            current_constraint_values_raw = y_scaler.inverse_transform(current_y_standardized_cpu)[:, 1]
            feasible_mask = current_constraint_values_raw < constraint_threshold

            if feasible_mask.any():
                best_f_raw = current_objective_values_raw[feasible_mask].min().item()
                best_f_standardized = y_scaler.transform(torch.tensor([[-best_f_raw, 0.0]], dtype=torch.float64))[:, 0].item()
            else:
                best_f_standardized = train_y_standardized.min().item() - 3.0

            constraints = {1: (None, constraint_threshold_standardized)}
            acq_function_to_use = LogConstrainedExpectedImprovement(model, best_f=best_f_standardized, objective_index=0, constraints=constraints)

        try:
            next_x_scaled, _ = optimize_acqf(acq_function_to_use, bounds=scaled_bounds, q=1, num_restarts=10, raw_samples=4000)
        except Exception as e:
            print(f"  FATAL: Acquisition function optimization failed: {e}")
            break
        
        bo_duration_iter = time.perf_counter() - bo_start_time_iter
        print(f"  Next point selection took {bo_duration_iter:.2f} seconds.")

        next_x_scaled_cpu = next_x_scaled.cpu()
        next_x_unscaled_internal = torch.tensor(x_scaler.inverse_transform(next_x_scaled_cpu.numpy()))
        next_x_real_world = transform_candidate_to_real_world(next_x_unscaled_internal)
        sample_to_evaluate = dict(zip(new_columns, [v.item() for v in next_x_real_world]))

        print("Sample to be evaluated by ANSA:")
        print(json.dumps(sample_to_evaluate, indent=4))

        simulation_results = evaluator.evaluate(sample_to_evaluate)
        iteration_duration_sec = time.perf_counter() - iter_start_time
        
        # Handle evaluation failure by logging it and continuing
        if simulation_results is None or "error" in simulation_results:
            error_msg = simulation_results.get('error', 'Evaluator returned None') if simulation_results else 'Evaluator returned None'
            print(f"  ANSA evaluation failed: {error_msg}. Saving failed point and skipping iteration.")
            
            failed_row_data = sample_to_evaluate.copy()
            for col in target_column_names: failed_row_data[col] = np.nan
            failed_row_data.update({
                'acquisition_function': acqf_name_this_iter,
                'bo_duration_sec': bo_duration_iter,
                'iteration_duration_sec': iteration_duration_sec,
                'evaluation_type': 'failed_evaluation',
                'iteration_number': INITIAL_DATA_SIZE + i
            })
            pd.DataFrame([failed_row_data])[CSV_HEADER].to_csv(RESULTS_CSV_PATH, mode='a', header=False, index=False)
            
            best_feasible_values.append(best_feasible_values[-1])
            continue
        
        # This block now only runs on successful evaluation
        new_objective_raw = simulation_results[objective_target_name]
        new_constraint_raw = simulation_results[constraint_target_name]
        print(f"  Received result: {objective_target_name} = {new_objective_raw:.4f}, {constraint_target_name} = {new_constraint_raw:.6f}")
        print(f"  Full iteration (BO + ANSA) took {iteration_duration_sec:.2f} seconds.")

        new_row_data = sample_to_evaluate.copy()
        new_row_data.update(simulation_results)
        new_row_data.update({
            'acquisition_function': acqf_name_this_iter,
            'bo_duration_sec': bo_duration_iter,
            'iteration_duration_sec': iteration_duration_sec,
            'evaluation_type': 'new_evaluation',
            'iteration_number': INITIAL_DATA_SIZE + i
        })
        pd.DataFrame([new_row_data])[CSV_HEADER].to_csv(RESULTS_CSV_PATH, mode='a', header=False, index=False)
        print(f"  Saved new evaluation to {RESULTS_CSV_PATH}")

        next_y_objective_raw = -torch.tensor([[new_objective_raw]])
        next_y_constraint_raw = torch.tensor([[new_constraint_raw]])
        next_y_combined_raw = torch.cat([next_y_objective_raw, next_y_constraint_raw], dim=1)
        next_y_standardized = torch.tensor(y_scaler.transform(next_y_combined_raw.numpy()), dtype=torch.float64).to(device)

        train_x_scaled = torch.cat([train_x_scaled, next_x_scaled], dim=0)
        train_y_standardized = torch.cat([train_y_standardized, next_y_standardized], dim=0)

        current_best_objective_raw = best_feasible_values[-1]
        if new_constraint_raw < constraint_threshold:
            if new_objective_raw < current_best_objective_raw:
                current_best_objective_raw = new_objective_raw
        best_feasible_values.append(current_best_objective_raw)
        print(f"   Best feasible value so far: {current_best_objective_raw:.6f}")

        # --- Stagnation Check Logic ---
        if len(best_feasible_values) > STAGNATION_WINDOW:
            past_best = best_feasible_values[-(STAGNATION_WINDOW + 1)]
            current_best = best_feasible_values[-1]

            if past_best != float('inf') and current_best >= past_best:
                stagnation_counter += 1
                print(f"   Stagnation counter incremented to: {stagnation_counter}")
            else:
                if stagnation_counter > 0:
                    print("   Improvement detected, resetting stagnation counter.")
                stagnation_counter = 0

            if stagnation_counter >= STAGNATION_WINDOW and exploration_iters_left == 0:
                print(f"   STALLED! No improvement for {STAGNATION_WINDOW} iterations. Forcing exploration for {EXPLORATION_DURATION} iterations.")
                exploration_iters_left = EXPLORATION_DURATION
                stagnation_counter = 0 # Reset counter to avoid immediate re-triggering

    else:
        if actual_iterations_run == MAX_TOTAL_ITERS:
            print(f"\nSTOPPING: Reached maximum total iterations ({MAX_TOTAL_ITERS}).")

except Exception as e:
    print("\n" + "="*50)
    print(f"An unexpected error occurred during the optimization loop: {e}")
    import traceback
    traceback.print_exc()
    print("Optimization stopped prematurely.")

finally:
    if evaluator:
        print("\nClosing persistent ANSA process...")
        evaluator.close()

# Results & Plotting
bo_total_duration = time.perf_counter() - bo_total_start_time
print("\n" + "="*50)
print("Optimization finished.")
print(f"Ran for {actual_iterations_run} new evaluations in {bo_total_duration:.2f} seconds ({bo_total_duration/60:.2f} minutes).")

valid_results = [v for v in best_feasible_values if v != float('inf')]
if not valid_results:
    print("\nNo feasible results were obtained during the optimization.")
    sys.exit()

best_actual_value = valid_results[-1]

final_y_raw = y_scaler.inverse_transform(train_y_standardized.cpu().numpy())
final_objective_values_raw = -final_y_raw[:, 0]
final_constraint_values_raw = final_y_raw[:, 1]
feasible_mask = final_constraint_values_raw < constraint_threshold

if np.any(feasible_mask):
    feasible_indices = np.where(feasible_mask)[0]
    best_in_feasible_idx = np.argmin(final_objective_values_raw[feasible_mask])
    best_idx_in_train = feasible_indices[best_in_feasible_idx]
    
    best_x_scaled_cpu = train_x_scaled[best_idx_in_train].reshape(1, -1).cpu()
    best_x_internal = torch.tensor(x_scaler.inverse_transform(best_x_scaled_cpu.numpy()))
    best_x_real = transform_candidate_to_real_world(best_x_internal)
    best_features_dict = {name: val.item() for name, val in zip(new_columns, best_x_real)}

    print(f"\nBest observed feasible value ({objective_target_name}): {best_actual_value:.6f}")
    print(f"Achieved with parameters:")
    print(pd.Series(best_features_dict))
else:
    print("\nCould not identify parameters for the best feasible value (none found).")

# --- Plotting ---
fig, ax = plt.subplots(figsize=(12, 7))
iterations = list(range(len(best_feasible_values)))
plot_values = [v if v != float('inf') else np.nan for v in best_feasible_values]
ax.plot(iterations, plot_values, marker='o', linestyle='-', label='Best Feasible Value After Each Evaluation')
legend_elements = [
    Line2D([0], [0], color='w', marker='', linestyle='',
            label=f'Minimum Feasible Value Found: {best_actual_value:.4f}')
]
handles, labels = ax.get_legend_handles_labels()
handles.extend(legend_elements)
ax.legend(handles=handles, loc='upper right')
ax.set_xlabel("ANSA Evaluation Number (0 = Initial State)")
ax.set_ylabel(f"Best Observed Feasible Minimum {objective_target_name}")
ax.set_title(f"Live Constrained BO - Minimize {objective_target_name} s.t. {constraint_target_name} < {constraint_threshold}")
ax.grid(True, which='both', linestyle='--')
ax.set_xlim(left=0)
plt.tight_layout()
plt.savefig(f'live_constrained_bo_progress_with_fallback.png')
print(f"\nSaved progress plot to 'live_constrained_bo_progress_with_fallback.png'")
plt.show()
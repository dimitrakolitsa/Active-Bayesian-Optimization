import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import warnings
import time
import sys
import json
import os
from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.multi_objective import ExpectedHypervolumeImprovement
from botorch.optim import optimize_acqf
from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning
from botorch.utils.multi_objective.hypervolume import Hypervolume
import gpytorch
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from botorch.exceptions import InputDataWarning, BadInitialCandidatesWarning
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D

# Import the live evaluator
from remote_ansa_evaluator import AnsaRemoteEvaluator, VARIABLE_NAMES

warnings.filterwarnings('ignore', category=InputDataWarning)
warnings.filterwarnings('ignore', category=BadInitialCandidatesWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
torch.manual_seed(42); np.random.seed(42); torch.set_default_dtype(torch.float64)

# --- Configuration ---
PROJECT_DIR = r"C:\Users\Dimitra\Desktop\25DVs_DM"
ANSA_WORKER_SCRIPT = os.path.join(PROJECT_DIR, "remote_ansa_worker.py")
# MODIFIED: Path for the results CSV file
RESULTS_CSV_PATH = "moo_3obj_results_with_fallback.csv"

new_columns = VARIABLE_NAMES
objective_target_name_1 = "max_displacement_torsion"
objective_target_name_2 = "max_stress_bending"
objective_target_name_3 = "mass"
objective_names = [objective_target_name_1, objective_target_name_2, objective_target_name_3]

BASE_PATH = '../../'
INITIAL_DATA_SIZE = 50
MAX_TOTAL_ITERS = 500

# NEW: Parameters for stagnation detection and LCB fallback
STAGNATION_WINDOW = 15
EXPLORATION_DURATION = 5
BETA_EXPLORE = 2.5       # LCB beta for the exploration phase
STAGNATION_HV_TOLERANCE = 1e-6 # Small tolerance for HV improvement

def simple_is_non_dominated(points):
    is_nd = torch.ones(points.shape[0], dtype=torch.bool, device=points.device)
    for i, p1 in enumerate(points):
        for j, p2 in enumerate(points):
            if i == j: continue
            if torch.all(p2 <= p1) and torch.any(p2 < p1):
                is_nd[i] = False; break
    return is_nd

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
# Add acquisition function column for initial data
df_for_csv['acquisition_function'] = 'N/A'

# Define the full header order with the new column
CSV_HEADER = new_columns + target_column_names + ['acquisition_function', 'bo_duration_sec', 'iteration_duration_sec', 'evaluation_type', 'iteration_number']

df_for_csv = df_for_csv[CSV_HEADER]
df_for_csv.to_csv(RESULTS_CSV_PATH, index=False)
print("Initial data saved.")

# --- Data Transformation ---
X_init = torch.tensor(df_init_raw[new_columns].values, dtype=torch.float64)
Y_init = torch.tensor(df_init_raw[target_column_names].values, dtype=torch.float64)

objective_index_1 = target_column_names.index(objective_target_name_1)
objective_index_2 = target_column_names.index(objective_target_name_2)
objective_index_3 = target_column_names.index(objective_target_name_3)
train_y_raw = torch.cat([-Y_init[:, objective_index_1].unsqueeze(-1), -Y_init[:, objective_index_2].unsqueeze(-1), -Y_init[:, objective_index_3].unsqueeze(-1)], dim=1)
obj_indices = [0, 1, 2]; n_objectives = len(obj_indices)
print(f"\nOptimizing objectives: Minimize {', '.join(objective_names)} (Unconstrained)")

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
x_scaler = MinMaxScaler(); x_scaler.fit(bounds_unscaled.numpy())
train_x_scaled = torch.tensor(x_scaler.transform(X_init.numpy()), dtype=torch.float64)
scaled_bounds = torch.tensor([[0.0] * X_init.shape[-1], [1.0] * X_init.shape[-1]], dtype=torch.float64)
y_scaler = StandardScaler(); y_scaler.fit(train_y_raw.numpy())
train_y_standardized = torch.tensor(y_scaler.transform(train_y_raw.numpy()), dtype=torch.float64)

# --- Helper Functions ---
def train_independent_gps(train_x, train_y):
    models = []
    for i in range(train_y.shape[-1]):
        model_i = SingleTaskGP(train_x, train_y[:, i].unsqueeze(-1), covar_module=ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=train_x.shape[-1])))
        mll_i = ExactMarginalLogLikelihood(model_i.likelihood, model_i);
        try: fit_gpytorch_mll(mll_i, max_retries=3)
        except Exception as e: print(f"Warning: GP fitting failed for objective {i}: {e}.")
        models.append(model_i)
    return ModelListGP(*models)

def transform_candidate_to_real_world(candidate_tensor):
    real_candidate = candidate_tensor.clone().squeeze()
    for i, name in enumerate(new_columns):
        if name in categorical_vars:
            choices = categorical_vars[name]
            val_unscaled = real_candidate[i]
            closest_idx = torch.argmin(torch.abs(torch.tensor(choices) - val_unscaled))
            real_candidate[i] = choices[closest_idx]
        elif name in stepped_vars:
            step = stepped_vars[name]; real_candidate[i] = torch.round(real_candidate[i] / step) * step
    return real_candidate

# --- BO loop Setup ---
actual_iterations_run = 0
hypervolume_history = []
hv_ref_point_raw = torch.tensor([1500.0, 1500.0, 1.0], dtype=torch.float64) 
hv_ref_point_std_neg = torch.tensor(y_scaler.transform(-hv_ref_point_raw.numpy().reshape(1, -1)), dtype=torch.float64).squeeze(0)

initial_objectives_raw = Y_init[:, [objective_index_1, objective_index_2, objective_index_3]]
non_dominated_mask_init = simple_is_non_dominated(initial_objectives_raw)
initial_pareto_raw = initial_objectives_raw[non_dominated_mask_init]
hv_calculator = Hypervolume(ref_point=-hv_ref_point_raw)
initial_hv = hv_calculator.compute(-initial_pareto_raw) if initial_pareto_raw.shape[0] > 0 else 0.0
hypervolume_history.append(initial_hv); print(f"Initial Hypervolume: {initial_hv:.6f}")

# NEW: Initialize state trackers for stagnation and exploration
stagnation_counter = 0
exploration_iters_left = 0

print(f"\nStarting Live 3-Objective Bayesian Optimization...")
bo_total_start_time = time.perf_counter()

# --- Main Optimization Block ---
evaluator = None
optimization_completed_successfully = False
try:
    evaluator = AnsaRemoteEvaluator(project_dir=PROJECT_DIR, ansa_worker_script=ANSA_WORKER_SCRIPT)

    for i in range(MAX_TOTAL_ITERS):
        actual_iterations_run = i + 1
        iter_start_time = time.perf_counter()
        print(f"\nIteration {i+1}/{MAX_TOTAL_ITERS}")
        bo_start_time_iter = time.perf_counter()
        model = train_independent_gps(train_x_scaled, train_y_standardized)

        # --- NEW: Acquisition Function Selection Logic ---
        acq_function_to_use = None
        acqf_name_this_iter = ""

        if exploration_iters_left > 0:
            acqf_name_this_iter = "custom_lcb"
            print(f"  Using EXPLORATION acquisition function ({acqf_name_this_iter}). {exploration_iters_left} iterations left.")
            
            # Define the 3-objective scalarized LCB acquisition function
            def scalarized_lcb(X):
                posterior = model.posterior(X)
                means, stds = posterior.mean, posterior.variance.clamp_min(1e-9).sqrt()
                lcb_obj1 = means[..., 0] - BETA_EXPLORE * stds[..., 0]
                lcb_obj2 = means[..., 1] - BETA_EXPLORE * stds[..., 1]
                lcb_obj3 = means[..., 2] - BETA_EXPLORE * stds[..., 2]
                # Return the weighted sum (equal weights for balanced exploration)
                return (lcb_obj1 + lcb_obj2 + lcb_obj3).squeeze(-1) # Weights sum to > 1, it's fine for argmax
            
            acq_function_to_use = scalarized_lcb
            exploration_iters_left -= 1
        else:
            acqf_name_this_iter = "ehvi"
            print(f"  Using standard acquisition function ({acqf_name_this_iter}).")
            
            partitioning = FastNondominatedPartitioning(ref_point=hv_ref_point_std_neg, Y=train_y_standardized)
            acq_function_to_use = ExpectedHypervolumeImprovement(model=model, ref_point=hv_ref_point_std_neg.tolist(), partitioning=partitioning)

        try:
            next_x_scaled, _ = optimize_acqf(acq_function_to_use, bounds=scaled_bounds, q=1, num_restarts=20, raw_samples=3072)
        except Exception as e:
            print(f"  FATAL: Acquisition function optimization failed: {e}. Stopping.")
            break

        bo_duration_iter = time.perf_counter() - bo_start_time_iter
        print(f"  Next point selection took {bo_duration_iter:.2f} seconds.")

        next_x_unscaled_internal = torch.tensor(x_scaler.inverse_transform(next_x_scaled.numpy()))
        next_x_real_world = transform_candidate_to_real_world(next_x_unscaled_internal)
        sample_to_evaluate = dict(zip(new_columns, [v.item() for v in next_x_real_world]))
        print("Sample to be evaluated by ANSA:"); print(json.dumps(sample_to_evaluate, indent=4))

        simulation_results = evaluator.evaluate(sample_to_evaluate)
        iteration_duration_sec = time.perf_counter() - iter_start_time

        # MODIFIED: Handle evaluation failure
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

            hypervolume_history.append(hypervolume_history[-1])
            continue
        
        # This block runs on success
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

        new_obj1_raw = simulation_results[objective_target_name_1]
        new_obj2_raw = simulation_results[objective_target_name_2]
        new_obj3_raw = simulation_results[objective_target_name_3]
        print(f"  Received results: {objective_names[0]}={new_obj1_raw:.4f}, {objective_names[1]}={new_obj2_raw:.4f}, {objective_names[2]}={new_obj3_raw:.6f}")

        next_y_combined_raw = torch.tensor([[-new_obj1_raw, -new_obj2_raw, -new_obj3_raw]])
        next_y_standardized = torch.tensor(y_scaler.transform(next_y_combined_raw.numpy()), dtype=torch.float64)
        
        train_x_scaled = torch.cat([train_x_scaled, next_x_scaled], dim=0)
        train_y_standardized = torch.cat([train_y_standardized, next_y_standardized], dim=0)
        
        all_y_raw_negated = torch.tensor(y_scaler.inverse_transform(train_y_standardized.numpy()))
        all_objectives_raw = -all_y_raw_negated[:, obj_indices]
        non_dominated_mask = simple_is_non_dominated(all_objectives_raw)
        pareto_front_raw = all_objectives_raw[non_dominated_mask]
        
        current_hv = hv_calculator.compute(-pareto_front_raw) if pareto_front_raw.shape[0] > 0 else 0.0
        hypervolume_history.append(current_hv)
        print(f"   Current Hypervolume: {current_hv:.6f}, Pareto Front Size: {pareto_front_raw.shape[0]}")

        # --- NEW: Stagnation Check Logic ---
        if len(hypervolume_history) > STAGNATION_WINDOW:
            past_hv = hypervolume_history[-(STAGNATION_WINDOW + 1)]
            
            if current_hv > past_hv + STAGNATION_HV_TOLERANCE:
                if stagnation_counter > 0:
                    print("   Improvement detected, resetting stagnation counter.")
                stagnation_counter = 0
            else:
                stagnation_counter += 1
                print(f"   Stagnation counter incremented to: {stagnation_counter}")
            
            if stagnation_counter >= STAGNATION_WINDOW and exploration_iters_left == 0:
                print(f"   STALLED! No significant HV improvement for {STAGNATION_WINDOW} iterations. Forcing exploration for {EXPLORATION_DURATION} iterations.")
                exploration_iters_left = EXPLORATION_DURATION
                stagnation_counter = 0

    else:
        if actual_iterations_run == MAX_TOTAL_ITERS:
            print(f"\nSTOPPING: Reached maximum total iterations ({MAX_TOTAL_ITERS}).")
    optimization_completed_successfully = True

except Exception as e:
    print("\n" + "="*50)
    import traceback
    traceback.print_exc()
    print("Optimization stopped prematurely.")

finally:
    if evaluator:
        print("\nClosing persistent ANSA process...")
        evaluator.close()

# --- Final Reporting & Plotting ---
bo_duration = time.perf_counter() - bo_total_start_time
print("\n" + "="*50); print("Optimization finished.")
print(f"Ran for {actual_iterations_run} new evaluations in {bo_duration:.2f} seconds ({bo_duration/60:.2f} minutes).")

if not optimization_completed_successfully and len(hypervolume_history) <= 1:
    print("\nNo new valid results were obtained during the optimization."); sys.exit()

final_y_raw_negated = torch.tensor(y_scaler.inverse_transform(train_y_standardized.numpy()))
final_objectives_raw = -final_y_raw_negated[:, obj_indices]
non_dominated_mask_final = simple_is_non_dominated(final_objectives_raw)
final_pareto_points_raw = final_objectives_raw[non_dominated_mask_final]
print(f"\nFound {len(final_pareto_points_raw)} non-dominated points in the final Pareto front.")
if final_pareto_points_raw.shape[0] > 0:
    sorted_indices = torch.argsort(final_pareto_points_raw[:, 0])
    sorted_pareto_front = final_pareto_points_raw[sorted_indices]
    print("Objectives (Raw, sorted):")
    print(pd.DataFrame(sorted_pareto_front.numpy(), columns=objective_names))

# --- Plotting ---
# 1. Hypervolume Convergence Plot
fig1, ax1 = plt.subplots(figsize=(12, 6))
ax1.plot(range(len(hypervolume_history)), hypervolume_history, marker='o', linestyle='-', label='Hypervolume per Evaluation')
max_hv_achieved = max(hypervolume_history) if hypervolume_history else 0.0
legend_elements = [Line2D([0], [0], color='w', marker='', linestyle='', label=f'Max Hypervolume Achieved: {max_hv_achieved:.4f}')]
handles, labels = ax1.get_legend_handles_labels()
handles.extend(legend_elements)
ax1.legend(handles=handles, loc='lower right')
ax1.set_xlabel("ANSA Evaluation Number (0 = Initial)")
ax1.set_ylabel("Hypervolume")
ax1.set_title("Hypervolume Convergence Plot (3 Objectives)")
ax1.grid(True, which='both', linestyle='--')
ax1.set_xlim(left=0)
ax1.set_ylim(bottom=0)
plt.tight_layout()
plt.savefig('live_moo_3obj_hypervolume_with_fallback.png')
print("\nSaved plot to 'live_moo_3obj_hypervolume_with_fallback.png'")
plt.show()

# 2. 3D Objective Space Plot
print("\nGenerating 3D Objective Space Plot...")
fig2 = plt.figure(figsize=(11, 9))
ax2 = fig2.add_subplot(111, projection='3d')
if final_objectives_raw.shape[0] > 0:
    ax2.scatter(final_objectives_raw[:, 0].numpy(), final_objectives_raw[:, 1].numpy(), final_objectives_raw[:, 2].numpy(), c='blue', alpha=0.5, s=20, label='All Evaluated Points')
    if final_pareto_points_raw.shape[0] > 0:
        ax2.scatter(final_pareto_points_raw[:, 0].numpy(), final_pareto_points_raw[:, 1].numpy(), final_pareto_points_raw[:, 2].numpy(), c='lime', s=150, edgecolor='black', marker='*', label='Final Pareto Front', depthshade=False, zorder=3)
    ax2.set_xlabel(f"{objective_names[0]} (Minimize)")
    ax2.set_ylabel(f"{objective_names[1]} (Minimize)")
    ax2.set_zlabel(f"{objective_names[2]} (Minimize)")
    ax2.set_title("3D Objective Space - Live Optimization Results")
    ax2.legend(loc='best')
    plt.tight_layout()
    plt.savefig('live_moo_3obj_objective_space_with_fallback.png')
    print("Saved plot to 'live_moo_3obj_objective_space_with_fallback.png'")
    plt.show()
else:
    print("\nNo points evaluated, cannot generate 3D plot.")
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
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.multi_objective.logei import qLogExpectedHypervolumeImprovement
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.objective import apply_constraints

from botorch.acquisition.multi_objective.objective import MCMultiOutputObjective

from botorch.optim import optimize_acqf
from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning
from botorch.utils.multi_objective.hypervolume import Hypervolume
#Import Normal distribution for PoF calculation in the fallback
from torch.distributions import Normal

import gpytorch
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from botorch.exceptions import InputDataWarning, BadInitialCandidatesWarning, UnsupportedError
from matplotlib.lines import Line2D

# Import the live evaluator
from remote_ansa_evaluator import AnsaRemoteEvaluator, VARIABLE_NAMES

warnings.filterwarnings('ignore', category=InputDataWarning)
warnings.filterwarnings('ignore', category=BadInitialCandidatesWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
torch.manual_seed(42)
np.random.seed(42)
torch.set_default_dtype(torch.float64)

# --- Configuration ---
PROJECT_DIR = r"C:\Users\Dimitra\Desktop\25DVs_DM"
ANSA_WORKER_SCRIPT = os.path.join(PROJECT_DIR, "remote_ansa_worker.py")
RESULTS_CSV_PATH = "moo_2obj_2constr_with_fallback.csv"

new_columns = VARIABLE_NAMES

# --- Objectives ---
objective_target_name_1 = "max_displacement_torsion" # Objective 1
objective_target_name_2 = "max_stress_bending"       # Objective 2
objective_names = [objective_target_name_1, objective_target_name_2]

# --- Constraints ---
CONSTRAINT_TARGET_NAME_1 = "max_displacement_bending" # Constraint 1
CONSTRAINT_THRESHOLD_1 = 70.0
CONSTRAINT_TARGET_NAME_2 = "mass"                     # Constraint 2
CONSTRAINT_THRESHOLD_2 = 0.018

BASE_PATH = '../../'
INITIAL_DATA_SIZE = 50
BASE_ITERS = 500
MAX_TOTAL_ITERS = 500
PATIENCE_WINDOW = 10
REL_IMPROVEMENT_TOL = 1e-5

# Parameters for stagnation detection and LCB fallback
STAGNATION_WINDOW = 15
EXPLORATION_DURATION = 5
BETA_EXPLORE = 2.5
STAGNATION_HV_TOLERANCE = 1e-6

def simple_is_non_dominated(points):
    is_nd = torch.ones(points.shape[0], dtype=torch.bool, device=points.device)
    for i, p1 in enumerate(points):
        for j, p2 in enumerate(points):
            if i == j: continue
            if torch.all(p2 <= p1) and torch.any(p2 < p1):
                is_nd[i] = False
                break
    return is_nd

class SlicerObjective(MCMultiOutputObjective):
    def __init__(self, objective_indices):
        super().__init__()
        self.objective_indices = objective_indices
    def forward(self, samples, X=None):
        return samples[..., self.objective_indices]

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
objective_index_1 = target_column_names.index(objective_target_name_1)
objective_index_2 = target_column_names.index(objective_target_name_2)
train_y_raw = torch.cat([-Y_init[:, objective_index_1].unsqueeze(-1), -Y_init[:, objective_index_2].unsqueeze(-1)], dim=1)
obj_indices = [0, 1]
n_objectives = len(obj_indices)
constraint_index_1 = target_column_names.index(CONSTRAINT_TARGET_NAME_1)
constraint_index_2 = target_column_names.index(CONSTRAINT_TARGET_NAME_2)
train_c_raw_1 = CONSTRAINT_THRESHOLD_1 - Y_init[:, constraint_index_1]
train_c_raw_2 = CONSTRAINT_THRESHOLD_2 - Y_init[:, constraint_index_2]
train_c_raw_combined = torch.cat([train_c_raw_1.unsqueeze(-1), train_c_raw_2.unsqueeze(-1)], dim=1)
print(f"\nOptimizing objectives: Minimize {', '.join(objective_names)}")
print(f"Subject to constraints: {CONSTRAINT_TARGET_NAME_1} < {CONSTRAINT_THRESHOLD_1} AND {CONSTRAINT_TARGET_NAME_2} < {CONSTRAINT_THRESHOLD_2}")

# --- Scaling ---
bounds_dict = {
    "FrontRear_height": [0.0, 3.0], "side_height": [0.0, 5.0], "side_width": [0.0, 4.0], "holes": [-3.0, 4.0], "edge_fit": [0.0, 1.5], "rear_offset": [-3.0, 3.0],
    "PSHELL_1_T": [2.0, 3.25], "PSHELL_2_T": [2.0, 3.25], "PSHELL_42733768_T": [1.6, 2.6], "PSHELL_42733769_T": [1.6, 2.6], "PSHELL_42733770_T": [1.6, 2.6], "PSHELL_42733772_T": [1.6, 2.6],
    "PSHELL_42733773_T": [1.6, 2.6], "PSHELL_42733774_T": [1.6, 2.6], "PSHELL_42733779_T": [2.0, 3.25], "PSHELL_42733780_T": [1.6, 2.6], "PSHELL_42733781_T": [2.399952, 3.899922], "PSHELL_42733782_T": [1.599936, 2.599896],
    "PSHELL_42733871_T": [1.199888, 1.949818], "PSHELL_42733879_T": [2.4, 3.9], "MAT1_1_E": [110000.0, 250000.0], "MAT1_42733768_E": [110000.0, 250000.0],
    "scale_x": [0.0, 1.5], "scale_y": [0.0, 1.5], "scale_z": [0.0, 1.5] }
stepped_vars = { "side_width": 0.1, "holes": 0.1, "edge_fit": 0.1, "rear_offset": 0.5 }
categorical_vars = {"scale_x": [0.0, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5], "scale_y": [0.0, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5], "scale_z": [0.0, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]}
bounds_unscaled = torch.tensor([bounds_dict[name] for name in new_columns], dtype=torch.float64).transpose(0, 1)
x_scaler = MinMaxScaler(); x_scaler.fit(bounds_unscaled.numpy())
train_x_scaled = torch.tensor(x_scaler.transform(X_init.numpy()), dtype=torch.float64)
scaled_bounds = torch.tensor([[0.0] * X_init.shape[-1], [1.0] * X_init.shape[-1]], dtype=torch.float64)
y_scaler = StandardScaler(); y_scaler.fit(train_y_raw.numpy())
train_y_standardized = torch.tensor(y_scaler.transform(train_y_raw.numpy()), dtype=torch.float64)
c_scaler = StandardScaler(); c_scaler.fit(train_c_raw_combined.numpy())
train_c_standardized = torch.tensor(c_scaler.transform(train_c_raw_combined.numpy()), dtype=torch.float64)
train_all_y_standardized = torch.cat([train_y_standardized, train_c_standardized], dim=1)

# --- Helper Functions ---
def train_independent_gps(train_x, train_y):
    models = []
    for i in range(train_y.shape[-1]):
        model_i = SingleTaskGP(train_x, train_y[:, i].unsqueeze(-1), covar_module=ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=train_x.shape[-1])))
        mll_i = ExactMarginalLogLikelihood(model_i.likelihood, model_i);
        try:
            with gpytorch.settings.cholesky_jitter(1e-5):
                fit_gpytorch_mll(mll_i, max_retries=3)
        except Exception as e:
            print(f"Warning: GP fitting failed for output {i}: {e}.")
        models.append(model_i)
    return ModelListGP(*models)

def transform_candidate_to_real_world(candidate_unscaled_tensor):
    real_candidate_1d = candidate_unscaled_tensor.clone().squeeze(0)
    for i, name in enumerate(new_columns):
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

# --- BO loop Setup ---
actual_iterations_run = 0
hypervolume_history = []
hv_ref_point_raw = torch.tensor([1500.0, 2000.0], dtype=torch.float64)
hv_ref_point_std_neg = torch.tensor(y_scaler.transform(-hv_ref_point_raw.numpy().reshape(1, -1)), dtype=torch.float64).squeeze(0)
initial_objectives_raw = Y_init[:, [target_column_names.index(objective_target_name_1), target_column_names.index(objective_target_name_2)]]
initial_constraint_vals_1_raw = Y_init[:, constraint_index_1]
initial_constraint_vals_2_raw = Y_init[:, constraint_index_2]
is_feasible_init = (initial_constraint_vals_1_raw < CONSTRAINT_THRESHOLD_1) & (initial_constraint_vals_2_raw < CONSTRAINT_THRESHOLD_2)
initial_pareto_raw_candidates = initial_objectives_raw[is_feasible_init]
if initial_pareto_raw_candidates.shape[0] > 0:
    non_dominated_mask_init = simple_is_non_dominated(initial_pareto_raw_candidates)
    initial_pareto_raw = initial_pareto_raw_candidates[non_dominated_mask_init]
else:
    initial_pareto_raw = torch.empty(0, n_objectives)
hv_calculator = Hypervolume(ref_point=-hv_ref_point_raw)
initial_hv = hv_calculator.compute(-initial_pareto_raw) if initial_pareto_raw.shape[0] > 0 else 0.0
hypervolume_history.append(initial_hv)
print(f"Initial FEASIBLE Hypervolume: {initial_hv:.6f}")
print(f"Number of initial FEASIBLE Pareto points: {initial_pareto_raw.shape[0]}")
print(f"\nStarting Live 2-Objective Bayesian Optimization with 2 Constraints...")
bo_total_start_time = time.perf_counter()

# --- Main Optimization Block ---
evaluator = None; optimization_completed_successfully = False
last_successful_model = None; stagnation_counter = 0; exploration_iters_left = 0
norm = Normal(0.0, 1.0)

try:
    evaluator = AnsaRemoteEvaluator(project_dir=PROJECT_DIR, ansa_worker_script=ANSA_WORKER_SCRIPT)
    for i in range(MAX_TOTAL_ITERS):
        actual_iterations_run = i + 1
        iter_start_time = time.perf_counter()
        print(f"\nIteration {i+1}/{MAX_TOTAL_ITERS}")
        bo_start_time_iter = time.perf_counter()
        
        model = train_independent_gps(train_x_scaled, train_all_y_standardized)
        if all(m is None for m in model.models):
            if last_successful_model is not None:
                print("  RECOVERY: All GP models failed to fit. Reusing models from previous iteration.")
                model = last_successful_model
            else:
                print("  FATAL: Initial GP models failed to fit. Stopping.")
                break
        else:
            last_successful_model = model
        
        acq_function_to_use = None
        acqf_name_this_iter = ""
        constraint_idx1 = n_objectives
        constraint_idx2 = n_objectives + 1

        if exploration_iters_left > 0:
            acqf_name_this_iter = "custom_lcb"
            print(f"  Using EXPLORATION acquisition function ({acqf_name_this_iter}). {exploration_iters_left} iterations left.")
            
            std_thresholds_for_zero = c_scaler.transform(np.array([[0.0, 0.0]]))[0]

            def constrained_scalarized_lcb(X):
                posterior = model.posterior(X)
                means, stds = posterior.mean, posterior.variance.clamp_min(1e-9).sqrt()
                
                pof1 = norm.cdf((means[..., constraint_idx1] - std_thresholds_for_zero[0]) / stds[..., constraint_idx1])
                pof2 = norm.cdf((means[..., constraint_idx2] - std_thresholds_for_zero[1]) / stds[..., constraint_idx2])
                joint_pof = pof1 * pof2
                
                lcb_obj1 = means[..., 0] - BETA_EXPLORE * stds[..., 0]
                lcb_obj2 = means[..., 1] - BETA_EXPLORE * stds[..., 1]
                scalarized_lcb_score = lcb_obj1 + lcb_obj2
                
                return (joint_pof * scalarized_lcb_score).squeeze(-1)
            
            acq_function_to_use = constrained_scalarized_lcb
            exploration_iters_left -= 1
        else:
            acqf_name_this_iter = "qLogEHVI"
            print(f"  Using standard acquisition function ({acqf_name_this_iter}).")
            mc_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([512]))
            mc_objective = SlicerObjective(objective_indices=obj_indices)
            constraint_callable = [lambda Z: -Z[..., constraint_idx1], lambda Z: -Z[..., constraint_idx2]]
            train_y_standardized_for_partitioning = train_all_y_standardized[:, obj_indices]

            std_thresholds_for_zero = c_scaler.transform(np.array([[0.0, 0.0]]))[0]
            is_feasible_for_partitioning = (train_all_y_standardized[:, constraint_idx1] > std_thresholds_for_zero[0]) & \
                                           (train_all_y_standardized[:, constraint_idx2] > std_thresholds_for_zero[1])

            partitioning = FastNondominatedPartitioning(ref_point=hv_ref_point_std_neg, Y=train_y_standardized_for_partitioning[is_feasible_for_partitioning])
            acq_function_to_use = qLogExpectedHypervolumeImprovement(model=model, ref_point=hv_ref_point_std_neg.tolist(), partitioning=partitioning, objective=mc_objective, constraints=constraint_callable, sampler=mc_sampler)

        try:
            next_x_scaled, _ = optimize_acqf(acq_function_to_use, bounds=scaled_bounds, q=1, num_restarts=10, raw_samples=1024)
        except Exception as e:
            print(f"  FATAL: Acquisition function optimization failed: {e}. Stopping."); traceback.print_exc(); break

        bo_duration_iter = time.perf_counter() - bo_start_time_iter
        print(f"  Next point selection took {bo_duration_iter:.2f} seconds.")
        
        next_x_unscaled_internal = torch.tensor(x_scaler.inverse_transform(next_x_scaled.numpy()))
        next_x_real_world_tensor = transform_candidate_to_real_world(next_x_unscaled_internal)
        sample_to_evaluate = dict(zip(new_columns, [v.item() for v in next_x_real_world_tensor.squeeze(0)]))
        print("Sample to be evaluated by ANSA:"); print(json.dumps(sample_to_evaluate, indent=4))
        simulation_results = evaluator.evaluate(sample_to_evaluate)

        if simulation_results is None or "error" in simulation_results or all(k not in simulation_results for k in objective_names):
            error_msg = simulation_results.get('error', 'Incomplete results from evaluator') if simulation_results else 'Evaluator returned None'
            print(f"  ANSA evaluation failed: {error_msg}. Saving failed point and skipping iteration.")
            failed_row_data = sample_to_evaluate.copy()
            for col in target_column_names: failed_row_data[col] = np.nan
            failed_row_data.update({'acquisition_function': acqf_name_this_iter, 'evaluation_type': 'failed_evaluation', 'iteration_number': INITIAL_DATA_SIZE + i, 'bo_duration_sec': bo_duration_iter, 'iteration_duration_sec': time.perf_counter() - iter_start_time})
            pd.DataFrame([failed_row_data])[CSV_HEADER].to_csv(RESULTS_CSV_PATH, mode='a', header=False, index=False)
            hypervolume_history.append(hypervolume_history[-1])
            continue
        else:
            iteration_duration_sec = time.perf_counter() - iter_start_time
            print(f"  Full iteration (BO + ANSA) took {iteration_duration_sec:.2f} seconds.")
            new_row_data = sample_to_evaluate.copy()
            new_row_data.update(simulation_results)
            new_row_data.update({'acquisition_function': acqf_name_this_iter, 'evaluation_type': 'new_evaluation', 'iteration_number': INITIAL_DATA_SIZE + i, 'bo_duration_sec': bo_duration_iter, 'iteration_duration_sec': iteration_duration_sec})
            pd.DataFrame([new_row_data])[CSV_HEADER].to_csv(RESULTS_CSV_PATH, mode='a', header=False, index=False)
            print(f"  Saved new evaluation to {RESULTS_CSV_PATH}")

            new_obj1_raw = simulation_results[objective_target_name_1]
            new_obj2_raw = simulation_results[objective_target_name_2]
            new_constraint_val_1_raw = simulation_results[CONSTRAINT_TARGET_NAME_1]
            new_constraint_val_2_raw = simulation_results[CONSTRAINT_TARGET_NAME_2]
            print(f"  Received results: {objective_names[0]}={new_obj1_raw:.4f}, {objective_names[1]}={new_obj2_raw:.4f}")
            print(f"  Constraint 1: {CONSTRAINT_TARGET_NAME_1}={new_constraint_val_1_raw:.4f} (Feasible: {new_constraint_val_1_raw < CONSTRAINT_THRESHOLD_1})")
            print(f"  Constraint 2: {CONSTRAINT_TARGET_NAME_2}={new_constraint_val_2_raw:.6f} (Feasible: {new_constraint_val_2_raw < CONSTRAINT_THRESHOLD_2})")

            next_y_combined_raw = torch.tensor([[-new_obj1_raw, -new_obj2_raw]])
            next_y_standardized = torch.tensor(y_scaler.transform(next_y_combined_raw.numpy()), dtype=torch.float64)
            next_c_raw_1 = CONSTRAINT_THRESHOLD_1 - new_constraint_val_1_raw
            next_c_raw_2 = CONSTRAINT_THRESHOLD_2 - new_constraint_val_2_raw
            next_c_raw_combined = torch.tensor([[next_c_raw_1, next_c_raw_2]])
            next_c_standardized = torch.tensor(c_scaler.transform(next_c_raw_combined.numpy()), dtype=torch.float64)
            next_all_y_standardized = torch.cat([next_y_standardized, next_c_standardized], dim=1)
            
            next_x_scaled_for_retrain = torch.tensor(x_scaler.transform(next_x_real_world_tensor.numpy()), dtype=torch.float64)
            train_x_scaled = torch.cat([train_x_scaled, next_x_scaled_for_retrain], dim=0)
            train_all_y_standardized = torch.cat([train_all_y_standardized, next_all_y_standardized], dim=0)
            
            all_y_raw_negated_from_std = torch.tensor(y_scaler.inverse_transform(train_all_y_standardized[:, obj_indices].numpy()))
            all_objectives_raw = -all_y_raw_negated_from_std
            constraint_residuals_raw = torch.tensor(c_scaler.inverse_transform(train_all_y_standardized[:, n_objectives:].numpy()))
            
            is_feasible_mask = (constraint_residuals_raw[:, 0] > 0) & (constraint_residuals_raw[:, 1] > 0)
            feasible_objectives = all_objectives_raw[is_feasible_mask]
            if feasible_objectives.shape[0] > 0:
                non_dominated_mask = simple_is_non_dominated(feasible_objectives)
                pareto_front_raw = feasible_objectives[non_dominated_mask]
            else:
                pareto_front_raw = torch.empty(0, n_objectives)
            
            current_hv = hv_calculator.compute(-pareto_front_raw) if pareto_front_raw.shape[0] > 0 else 0.0
            hypervolume_history.append(current_hv)
            print(f"   Current FEASIBLE Hypervolume: {current_hv:.6f}, Feasible Pareto Front Size: {pareto_front_raw.shape[0]}")

        # Stagnation Check Logic
        if len(hypervolume_history) > STAGNATION_WINDOW:
            past_hv = hypervolume_history[-(STAGNATION_WINDOW + 1)]
            if current_hv > past_hv + STAGNATION_HV_TOLERANCE:
                if stagnation_counter > 0: print("   Improvement detected, resetting stagnation counter.")
                stagnation_counter = 0
            else:
                stagnation_counter += 1
                print(f"   Stagnation counter incremented to: {stagnation_counter}")
            if stagnation_counter >= STAGNATION_WINDOW and exploration_iters_left == 0:
                print(f"   STALLED! No significant HV improvement for {STAGNATION_WINDOW} iterations. Forcing exploration for {EXPLORATION_DURATION} iterations.")
                exploration_iters_left = EXPLORATION_DURATION
                stagnation_counter = 0
        
        # Convergence Check
        if i >= BASE_ITERS - 1 and len(hypervolume_history) > PATIENCE_WINDOW:
            past_hv = hypervolume_history[-(PATIENCE_WINDOW + 1)]; current_hv_check = hypervolume_history[-1]
            if abs(past_hv) > 1e-9:
                relative_improvement = (current_hv_check - past_hv) / abs(past_hv)
                if relative_improvement < REL_IMPROVEMENT_TOL:
                    print(f"\nSTOPPING: Relative HV improvement ({relative_improvement:.2e}) is below tolerance.")
                    break
    else:
        if actual_iterations_run == MAX_TOTAL_ITERS:
            print(f"\nSTOPPING: Reached maximum total iterations ({MAX_TOTAL_ITERS}).")
    optimization_completed_successfully = True

except Exception as e:
    print("\n" + "="*50); print(f"An unexpected error occurred: {e}"); traceback.print_exc(); print("Optimization stopped prematurely.")
finally:
    if evaluator: print("\nClosing persistent ANSA process..."); evaluator.close()

# --- Final Reporting & Plotting ---
bo_duration = time.perf_counter() - bo_total_start_time
print("\n" + "="*50); print("Optimization finished.")
print(f"Ran for {actual_iterations_run} new evaluations in {bo_duration:.2f} seconds ({bo_duration/60:.2f} minutes).")

if not optimization_completed_successfully and len(hypervolume_history) <= 1:
    print("\nNo new valid results were obtained during the optimization."); sys.exit()

final_y_raw_negated_from_std = torch.tensor(y_scaler.inverse_transform(train_all_y_standardized[:, obj_indices].numpy()), dtype=torch.float64)
final_objectives_raw = -final_y_raw_negated_from_std
final_constraint_residuals_raw = torch.tensor(c_scaler.inverse_transform(train_all_y_standardized[:, n_objectives:].numpy()), dtype=torch.float64)
final_is_feasible_mask = (final_constraint_residuals_raw[:, 0] > 0) & (final_constraint_residuals_raw[:, 1] > 0)
final_feasible_objectives_raw = final_objectives_raw[final_is_feasible_mask]
final_infeasible_objectives_raw = final_objectives_raw[~final_is_feasible_mask]
if final_feasible_objectives_raw.shape[0] > 0:
    non_dominated_mask_final = simple_is_non_dominated(final_feasible_objectives_raw)
    final_pareto_points_raw = final_feasible_objectives_raw[non_dominated_mask_final]
else:
    final_pareto_points_raw = torch.empty(0, n_objectives)
print(f"\nFound {len(final_pareto_points_raw)} non-dominated points in the final FEASIBLE Pareto front.")
if final_pareto_points_raw.shape[0] > 0:
    sorted_indices = torch.argsort(final_pareto_points_raw[:, 0])
    sorted_pareto_front = final_pareto_points_raw[sorted_indices]
    print("Objectives (Raw, Feasible, Pareto, sorted):")
    print(pd.DataFrame(sorted_pareto_front.numpy(), columns=objective_names))

# --- Plotting ---
fig1, ax1 = plt.subplots(figsize=(12, 6))
ax1.plot(range(len(hypervolume_history)), hypervolume_history, marker='o', linestyle='-', label='Feasible Hypervolume per Evaluation')
max_hv_achieved = max(hypervolume_history) if hypervolume_history else 0.0
legend_elements = [Line2D([0], [0], color='w', marker='', linestyle='', label=f'Max Feasible HV Achieved: {max_hv_achieved:.4f}')]
handles, labels = ax1.get_legend_handles_labels()
handles.extend(legend_elements)
ax1.legend(handles=handles, loc='lower right')
ax1.set_xlabel("ANSA Evaluation Number (0 = Initial)")
ax1.set_ylabel("Feasible Hypervolume")
ax1.set_title("Feasible Hypervolume Convergence Plot (2 Objectives, 2 Constraints)")
ax1.grid(True, which='both', linestyle='--')
ax1.set_xlim(left=0); ax1.set_ylim(bottom=0)
plt.tight_layout()
plt.savefig('live_moo_2obj_2constr_hypervolume_with_fallback.png')
print("\nSaved plot to 'live_moo_2obj_2constr_hypervolume_with_fallback.png'")
plt.show()

fig2 = plt.figure(figsize=(10, 8))
ax2 = fig2.add_subplot(111)
if final_objectives_raw.shape[0] > 0:
    ax2.scatter(final_infeasible_objectives_raw[:, 0].numpy(), final_infeasible_objectives_raw[:, 1].numpy(), c='lightgray', alpha=0.5, s=15, label='Infeasible Points')
    ax2.scatter(final_feasible_objectives_raw[:, 0].numpy(), final_feasible_objectives_raw[:, 1].numpy(), c='blue', alpha=0.6, s=25, label='Feasible Points')
    if final_pareto_points_raw.shape[0] > 0:
        ax2.scatter(final_pareto_points_raw[:, 0].numpy(), final_pareto_points_raw[:, 1].numpy(), c='lime', s=150, edgecolor='black', marker='*', label='Final Feasible Pareto Front', zorder=3)
    ax2.set_xlabel(f"{objective_names[0]} (Minimize)"); ax2.set_ylabel(f"{objective_names[1]} (Minimize)")
    ax2.set_title("2D Objective Space - Live Optimization Results (with Constraints)")
    ax2.legend(loc='best'); ax2.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('live_moo_2obj_2constr_objective_space_with_fallback.png')
    print("Saved plot to 'live_moo_2obj_2constr_objective_space_with_fallback.png'")
    plt.show()
else:
    print("\nNo points evaluated, cannot generate 2D plot.")
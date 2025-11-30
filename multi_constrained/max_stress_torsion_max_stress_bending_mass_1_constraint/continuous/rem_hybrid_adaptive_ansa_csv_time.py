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

# BoTorch
from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.fit import fit_gpytorch_mll
from botorch.optim import optimize_acqf
from botorch.utils.multi_objective.hypervolume import Hypervolume
from torch.distributions import Normal

# GPyTorch
import gpytorch
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.priors import GammaPrior, SmoothedBoxPrior

# sklearn
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Output & Plotting
from botorch.exceptions import InputDataWarning, BadInitialCandidatesWarning
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D

# Import the live evaluator
from remote_ansa_evaluator import AnsaRemoteEvaluator, VARIABLE_NAMES

# --- Warnings and Seeding ---
warnings.filterwarnings('ignore', category=InputDataWarning)
warnings.filterwarnings('ignore', category=BadInitialCandidatesWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
torch.manual_seed(42)
np.random.seed(42)
torch.set_default_dtype(torch.float64)

# --- Configuration ---
PROJECT_DIR = r"C:\Users\Dimitra\Desktop\25DVs_DM"
ANSA_WORKER_SCRIPT = os.path.join(PROJECT_DIR, "remote_ansa_worker.py")
RESULTS_CSV_PATH = "moo_live_adaptive_lcb_results.csv" # New unique name for results file

# Define variable and objective names
new_columns = VARIABLE_NAMES
objective_target_name_1 = "max_stress_torsion"
objective_target_name_2 = "max_stress_bending"
objective_target_name_3 = "mass"
constraint_target_name = "max_displacement_bending"
objective_names = [objective_target_name_1, objective_target_name_2, objective_target_name_3]
target_column_names = ["mass", "max_displacement_bending", "max_displacement_torsion", "max_stress_bending", "max_stress_torsion"]

# BO Parameters
CONSTRAINT_THRESHOLD = 70.0
INITIAL_DATA_SIZE = 50
MAX_TOTAL_ITERS = 500

# --- Helper Functions ---

# def simple_is_non_dominated(points):
#     """Simple pairwise non-dominated check for minimization."""
#     n_points = points.shape[0]
#     if n_points == 0:
#         return torch.tensor([], dtype=torch.bool)
#     is_nd = torch.ones(n_points, dtype=torch.bool, device=points.device)
#     for i_point_loop in range(n_points):
#         if not is_nd[i_point_loop]: continue
#         for j_point_loop in range(n_points):
#             if i_point_loop == j_point_loop: continue
#             if torch.all(points[j_point_loop] <= points[i_point_loop]) and torch.any(points[j_point_loop] < points[i_point_loop]):
#                 is_nd[i_point_loop] = False
#                 break
#     return is_nd

def simple_is_non_dominated(points, tol=1e-6):
    """
    Simple pairwise non-dominated check for minimization with a tolerance.
    A point p2 dominates p1 if (p2 <= p1 - tol) for all objectives,
    and (p2 < p1 - tol) for at least one objective.
    """
    n_points = points.shape[0]
    if n_points == 0:
        return torch.tensor([], dtype=torch.bool)
    
    is_nd = torch.ones(n_points, dtype=torch.bool, device=points.device)
    for i_point_loop in range(n_points):
        if not is_nd[i_point_loop]: continue
        for j_point_loop in range(n_points):
            if i_point_loop == j_point_loop: continue
            
            p1 = points[i_point_loop]
            p2 = points[j_point_loop]
            
            # Check if p2 dominates p1 with tolerance
            all_le = torch.all(p2 <= p1 - tol)
            any_l = torch.any(p2 < p1 - tol)
            
            if all_le and any_l:
                is_nd[i_point_loop] = False
                break
    return is_nd

def train_independent_gps(train_x, train_y):
    """Trains a list of independent SingleTaskGP models."""
    models = []
    num_outputs = train_y.shape[-1]
    for i in range(num_outputs):
        train_y_i = train_y[:, i].unsqueeze(-1)
        # Using more robust priors from the adaptive LCB script
        lengthscale_prior = GammaPrior(2.0, 0.15)
        outputscale_prior = GammaPrior(2.0, 0.15)
        noise_prior = SmoothedBoxPrior(1e-5, 1e-1, sigma=0.1)
        covar_module_i = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=train_x.shape[-1], lengthscale_prior=lengthscale_prior),
                                    outputscale_prior=outputscale_prior)
        model_i = SingleTaskGP(train_x, train_y_i, covar_module=covar_module_i)
        model_i.likelihood.noise_prior = noise_prior
        mll_i = ExactMarginalLogLikelihood(model_i.likelihood, model_i)
        try:
            with gpytorch.settings.cholesky_jitter(1e-4):
                fit_gpytorch_mll(mll_i, max_retries=15, options={'maxiter': 300})
        except Exception as e:
            print(f"Warning: GP fitting failed for output {i}: {e}.")
        models.append(model_i)
    return ModelListGP(*models)

def transform_candidate_to_real_world(candidate_unscaled_tensor, bounds_unscaled, stepped_vars, categorical_vars, col_names):
    """Snaps a continuous candidate to the nearest valid real-world value."""
    real_candidate = candidate_unscaled_tensor.clone().squeeze()
    for i, name in enumerate(col_names):
        if name in categorical_vars:
            choices = torch.tensor(categorical_vars[name], dtype=torch.float64)
            closest_idx = torch.argmin(torch.abs(choices - real_candidate[i]))
            real_candidate[i] = choices[closest_idx]
        elif name in stepped_vars:
            step = stepped_vars[name]
            rounded_val = torch.round(real_candidate[i] / step) * step
            real_candidate[i] = rounded_val
    # Enforce hard bounds after snapping
    lower_bounds_unscaled = bounds_unscaled[0]
    upper_bounds_unscaled = bounds_unscaled[1]
    real_candidate = torch.max(lower_bounds_unscaled, torch.min(upper_bounds_unscaled, real_candidate))
    return real_candidate

# --- Initial Data Loading ---
BASE_PATH = '../../'
df_init_raw = pd.read_csv(f'{BASE_PATH}init/inputs.txt', header=0, sep=',')
df_init_raw = df_init_raw[new_columns]
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
CSV_HEADER = new_columns + target_column_names + ['bo_duration_sec', 'iteration_duration_sec', 'evaluation_type', 'iteration_number']
df_for_csv = df_for_csv[CSV_HEADER]
df_for_csv.to_csv(RESULTS_CSV_PATH, index=False)
print("Initial data saved.")

# --- Data Transformation and Scaling ---
X_init = torch.tensor(df_init_raw[new_columns].values, dtype=torch.float64)
Y_init = torch.tensor(df_init_raw[target_column_names].values, dtype=torch.float64)

# Define objective and constraint indices
objective_index_1 = target_column_names.index(objective_target_name_1)
objective_index_2 = target_column_names.index(objective_target_name_2)
objective_index_3 = target_column_names.index(objective_target_name_3)
constraint_index = target_column_names.index(constraint_target_name)
obj_indices = [0, 1, 2] # Indices in the model output tensor
constraint_index_in_y = 3 # Index in the model output tensor
n_objectives = len(obj_indices)

# Combine objectives (negated for maximization) and constraint into one tensor for the model
train_y_raw = torch.cat([
    -Y_init[:, objective_index_1].unsqueeze(-1),
    -Y_init[:, objective_index_2].unsqueeze(-1),
    -Y_init[:, objective_index_3].unsqueeze(-1),
    Y_init[:, constraint_index].unsqueeze(-1) # Constraint is NOT negated
], dim=1)

print(f"\nOptimizing objectives: Minimize {', '.join(objective_names)}")
print(f"Subject to constraint: {constraint_target_name} < {CONSTRAINT_THRESHOLD}")

# Define bounds and scalers
bounds_dict = {
    "FrontRear_height": [0.0, 3.0], "side_height": [0.0, 5.0], "side_width": [0.0, 4.0], "holes": [-3.0, 4.0], "edge_fit": [0.0, 1.5], "rear_offset": [-3.0, 3.0],
    "PSHELL_1_T": [2.0, 3.25], "PSHELL_2_T": [2.0, 3.25], "PSHELL_42733768_T": [1.6, 2.6], "PSHELL_42733769_T": [1.6, 2.6], "PSHELL_42733770_T": [1.6, 2.6], "PSHELL_42733772_T": [1.6, 2.6],
    "PSHELL_42733773_T": [1.6, 2.6], "PSHELL_42733774_T": [1.6, 2.6], "PSHELL_42733779_T": [2.0, 3.25], "PSHELL_42733780_T": [1.6, 2.6], "PSHELL_42733781_T": [2.399952, 3.899922], "PSHELL_42733782_T": [1.599936, 2.599896],
    "PSHELL_42733871_T": [1.199888, 1.949818], "PSHELL_42733879_T": [2.4, 3.9], "MAT1_1_E": [110000.0, 250000.0], "MAT1_42733768_E": [110000.0, 250000.0],
    "scale_x": [0.0, 1.5], "scale_y": [0.0, 1.5], "scale_z": [0.0, 1.5]
}
stepped_vars = { "side_width": 0.1, "holes": 0.1, "edge_fit": 0.1, "rear_offset": 0.5 }
categorical_vars = {"scale_x": [0.0, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5], "scale_y": [0.0, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5], "scale_z": [0.0, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]}
bounds_unscaled = torch.tensor([bounds_dict[name] for name in new_columns], dtype=torch.float64).transpose(0, 1)

# X scaler (input features)
x_scaler = MinMaxScaler()
x_scaler.fit(bounds_unscaled.numpy()) # Fit on the bounds, not just initial data
train_x_scaled = torch.tensor(x_scaler.transform(X_init.numpy()), dtype=torch.float64)
scaled_bounds = torch.tensor([[0.0] * X_init.shape[-1], [1.0] * X_init.shape[-1]], dtype=torch.float64)

# Y scaler (objectives and constraint)
y_scaler = StandardScaler()
y_scaler.fit(train_y_raw.numpy())
train_y_standardized = torch.tensor(y_scaler.transform(train_y_raw.numpy()), dtype=torch.float64)

# Standardize the constraint threshold for use in the acquisition function
constraint_mean_model = y_scaler.mean_[constraint_index_in_y]
constraint_scale_model = y_scaler.scale_[constraint_index_in_y]
constraint_threshold_std = (CONSTRAINT_THRESHOLD - constraint_mean_model) / constraint_scale_model
print(f"Raw constraint threshold: {CONSTRAINT_THRESHOLD}, Standardized constraint threshold: {constraint_threshold_std:.4f}")

# --- Adaptive BO Logic Setup ---
alpha_variance = 0.2
alpha_hv_improvement = 0.3
beta_states = torch.tensor([0.5, 1.0, 1.5, 2.0, 2.5, 3.0], dtype=torch.float64)
smooth_avg_variance = torch.tensor(0.0, dtype=torch.float64)
smooth_hv_improvement = torch.tensor(0.0, dtype=torch.float64)
stall_counter = 0
N_STALL_THRESH = 8
FORCED_EXPLORATION_DURATION = 10
forced_exploration_iters_left = 0
STALL_HV_IMPROVEMENT_TOLERANCE = 1e-5

# --- Hypervolume Setup ---
hypervolume_history = []
hv_ref_point_raw = torch.tensor([1500.0, 1500.0, 1.0], dtype=torch.float64)
negated_ref_point_for_hv_calc = -hv_ref_point_raw

# Calculate Initial Hypervolume
initial_objectives_raw = Y_init[:, [objective_index_1, objective_index_2, objective_index_3]]
initial_constraints_raw = Y_init[:, constraint_index]
feasible_mask_init = initial_constraints_raw <= CONSTRAINT_THRESHOLD
initial_objectives_feasible = initial_objectives_raw[feasible_mask_init]
initial_pareto_raw = torch.empty((0, n_objectives), dtype=torch.float64)
if initial_objectives_feasible.shape[0] > 0:
    non_dominated_mask_init = simple_is_non_dominated(initial_objectives_feasible)
    initial_pareto_raw = initial_objectives_feasible[non_dominated_mask_init]

hv_calculator = Hypervolume(ref_point=negated_ref_point_for_hv_calc)
initial_hv = hv_calculator.compute(-initial_pareto_raw) if initial_pareto_raw.shape[0] > 0 else 0.0
hypervolume_history.append(initial_hv)
print(f"Initial FEASIBLE Hypervolume: {initial_hv:.6f}, Initial Pareto Size: {initial_pareto_raw.shape[0]}")
previous_pareto_front_size = initial_pareto_raw.shape[0]

# --- Main Optimization Block ---
script_version_name = "Live_AdaptiveLCB_with_ANSA"
print(f"\nStarting Live Bayesian Optimization ({script_version_name})...")
bo_total_start_time = time.perf_counter()
actual_iterations_run = 0
evaluator = None
optimization_completed_successfully = False

try:
    evaluator = AnsaRemoteEvaluator(project_dir=PROJECT_DIR, ansa_worker_script=ANSA_WORKER_SCRIPT)

    for i in range(MAX_TOTAL_ITERS):
        actual_iterations_run = i + 1
        iter_start_time = time.perf_counter()
        print(f"\nIteration {i+1}/{MAX_TOTAL_ITERS}")

        # --- BO Point Selection Step ---
        bo_start_time_iter = time.perf_counter()

        model = train_independent_gps(train_x_scaled, train_y_standardized)
        model.eval()

        # --- Adaptive Beta and Weight Logic ---
        # (This entire block is from the advanced adaptive LCB script)
        current_beta_lcb: float; w1: float; w2: float; w3: float
        rule_reason: str
        
        with torch.no_grad():
            posterior_for_variance = model.posterior(train_x_scaled) # Estimate variance from evaluated points
            objective_variances_std = posterior_for_variance.variance[:, obj_indices]
        observed_avg_variance_k = torch.mean(objective_variances_std)
        observed_hv_improvement_k = torch.tensor(0.0)
        if len(hypervolume_history) > 1:
            observed_hv_improvement_k = hypervolume_history[-1] - hypervolume_history[-2]

        if i == 0:
            smooth_avg_variance = observed_avg_variance_k.clone()
            smooth_hv_improvement = observed_hv_improvement_k.clone()
        else:
            smooth_avg_variance = (1.0 - alpha_variance) * smooth_avg_variance + alpha_variance * observed_avg_variance_k
            smooth_hv_improvement = (1.0 - alpha_hv_improvement) * smooth_hv_improvement + alpha_hv_improvement * observed_hv_improvement_k
        
        print(f"    EMA Metrics: SmoothAvgVar={smooth_avg_variance.item():.4f}, SmoothHVImprove={smooth_hv_improvement.item():.6f}")
        print(f"    Stall counter: {stall_counter}, Forced exploration iters left: {forced_exploration_iters_left}")
        
        # Determine beta
        # if forced_exploration_iters_left > 0:
        #     current_beta_lcb = beta_states[-1].item() # Max exploration
        #     rule_reason = f"Forced stall exploration ({forced_exploration_iters_left} left, Beta={current_beta_lcb})"
        # elif i < 10: # Early iterations
        #     current_beta_lcb = beta_states[-1].item()
        #     rule_reason = f"Early iter, force exploration (Beta={current_beta_lcb})"
        # elif smooth_hv_improvement < 1e-5: # Stagnation
        #     if smooth_avg_variance > 0.6:
        #         current_beta_lcb = beta_states[-1].item()
        #         rule_reason = f"Stagnant, high variance (Beta={current_beta_lcb})"
        #     else:
        #         current_beta_lcb = beta_states[1].item()
        #         rule_reason = f"Stagnant, low variance -> exploit (Beta={current_beta_lcb})"
        # else: # Normal operation
        #     current_beta_lcb = beta_states[3].item()
        #     rule_reason = f"Normal operation, balanced (Beta={current_beta_lcb})"

                # --- REVISED: Determine beta with a structured hybrid strategy ---

        norm = Normal(0.0, 1.0)

        if i < 50:  # Phase 1: Forced Exploration
            current_beta_lcb = 3.0
            rule_reason = f"Initial phase, force exploration (Beta={current_beta_lcb})"
            # Use EQUAL weights for unbiased exploration
            w1, w2, w3 = 1/3., 1/3., 1/3.

        elif forced_exploration_iters_left > 0: # If stalled: Forced Exploration
            current_beta_lcb = 3.0
            rule_reason = f"Forced stall exploration ({forced_exploration_iters_left} left, Beta={current_beta_lcb})"
            # Use DYNAMIC weights during stall recovery to focus the search
            with torch.no_grad():
                posterior = model.posterior(scaled_bounds)
                means_std, stds_std = posterior.mean, posterior.variance.clamp_min(1e-9).sqrt()
            best_obj_vals = torch.max(train_y_standardized[:, obj_indices], dim=0).values
            norm = Normal(0.0, 1.0)
            z = (means_std[:, obj_indices] - best_obj_vals) / stds_std[:, obj_indices]
            ei_vals = (means_std[:, obj_indices] - best_obj_vals) * norm.cdf(z) + stds_std[:, obj_indices] * torch.exp(norm.log_prob(z))
            max_ei = torch.max(ei_vals, dim=0).values + 1e-7
            weights = max_ei / torch.sum(max_ei)
            min_weight = 0.1
            weights = (1 - n_objectives * min_weight) * weights + min_weight
            weights = weights / torch.sum(weights)
            w1, w2, w3 = weights[0].item(), weights[1].item(), weights[2].item()

        else:  # Phase 2: Adaptive Logic
            # Determine beta adaptively
            HIGH_VARIANCE_THRESH = 0.5
            MODERATE_VARIANCE_THRESH = 0.1
            if smooth_hv_improvement < STALL_HV_IMPROVEMENT_TOLERANCE:
                if smooth_avg_variance > HIGH_VARIANCE_THRESH:
                    current_beta_lcb = 2.5
                    rule_reason = f"Adaptive: Stagnant, HIGH variance (Beta={current_beta_lcb})"
                elif smooth_avg_variance > MODERATE_VARIANCE_THRESH:
                    current_beta_lcb = 2.0
                    rule_reason = f"Adaptive: Stagnant, MODERATE variance (Beta={current_beta_lcb})"
                else:
                    current_beta_lcb = 1.5 #1.0
                    rule_reason = f"Adaptive: Stagnant, LOW variance -> exploit (Beta={current_beta_lcb})"
            else:
                current_beta_lcb = 2.5 #1.5
                rule_reason = f"Adaptive: Normal operation, balanced (Beta={current_beta_lcb})"

            # Determine weights dynamically
            with torch.no_grad():
                posterior = model.posterior(scaled_bounds)
                means_std, stds_std = posterior.mean, posterior.variance.clamp_min(1e-9).sqrt()
            best_obj_vals = torch.max(train_y_standardized[:, obj_indices], dim=0).values
            norm = Normal(0.0, 1.0)
            z = (means_std[:, obj_indices] - best_obj_vals) / stds_std[:, obj_indices]
            ei_vals = (means_std[:, obj_indices] - best_obj_vals) * norm.cdf(z) + stds_std[:, obj_indices] * torch.exp(norm.log_prob(z))
            max_ei = torch.max(ei_vals, dim=0).values + 1e-7
            weights = max_ei / torch.sum(max_ei)
            min_weight = 0.1
            weights = (1 - n_objectives * min_weight) * weights + min_weight
            weights = weights / torch.sum(weights)
            w1, w2, w3 = weights[0].item(), weights[1].item(), weights[2].item()
        
        print(f"  Using LCB params: {rule_reason}, Dynamic Weights=[{w1:.3f}, {w2:.3f}, {w3:.3f}]")

        # --- Define Acquisition Function ---
        def constrained_scalarized_lcb(X):
            """
            This function must be differentiable, so the `with torch.no_grad()`
            block has been removed.
            """
            # `optimize_acqf` provides X in the correct batch shape [b, q, d].
            posterior = model.posterior(X)
            means, stds = posterior.mean, posterior.variance.clamp_min(1e-9).sqrt()

            # Probability of Feasibility (PoF) for the constraint
            # `norm` is a global variable defined in the main script block
            mean_constr, std_constr = means[..., constraint_index_in_y], stds[..., constraint_index_in_y]
            pof = norm.cdf((constraint_threshold_std - mean_constr) / std_constr)

            # Scalarized LCB for the objectives
            lcb = (
                w1 * (means[..., obj_indices[0]] - current_beta_lcb * stds[..., obj_indices[0]]) +
                w2 * (means[..., obj_indices[1]] - current_beta_lcb * stds[..., obj_indices[1]]) +
                w3 * (means[..., obj_indices[2]] - current_beta_lcb * stds[..., obj_indices[2]])
            )
            
            # Squeeze the last dimension to return a 1D tensor [b]
            return (pof * lcb).squeeze(-1)

        # --- Optimize Acquisition Function ---
        try:
            next_x_scaled, _ = optimize_acqf(
                acq_function=constrained_scalarized_lcb,
                bounds=scaled_bounds, q=1, num_restarts=15, raw_samples=1024,
            )
        except Exception as e:
            print(f"  FATAL: Acquisition function optimization failed: {e}. Stopping.")
            traceback.print_exc()
            break
        
        bo_duration_iter = time.perf_counter() - bo_start_time_iter
        print(f"  Next point selection took {bo_duration_iter:.2f} seconds.")

        # --- Live Evaluation Block ---
        next_x_unscaled_internal = torch.tensor(x_scaler.inverse_transform(next_x_scaled.numpy()))
        next_x_real_world = transform_candidate_to_real_world(
            next_x_unscaled_internal, bounds_unscaled, stepped_vars, categorical_vars, new_columns
        )
        sample_to_evaluate = dict(zip(new_columns, [v.item() for v in next_x_real_world]))
        print("Sample to be evaluated by ANSA:"); print(json.dumps(sample_to_evaluate, indent=4))
        simulation_results = evaluator.evaluate(sample_to_evaluate)

        if simulation_results is None or "error" in simulation_results:
            error_msg = simulation_results.get('error', 'Evaluator returned None') if simulation_results else 'Evaluator returned None'
            print(f"  ANSA evaluation failed: {error_msg}. Skipping iteration.")
            hypervolume_history.append(hypervolume_history[-1])
            continue
        
        iteration_duration_sec = time.perf_counter() - iter_start_time
        print(f"  Full iteration (BO + ANSA) took {iteration_duration_sec:.2f} seconds.")

        # --- Process and Save New Data ---
        new_row_data = sample_to_evaluate.copy()
        new_row_data.update(simulation_results)
        new_row_data.update({
            'evaluation_type': 'bo_evaluation', 'iteration_number': INITIAL_DATA_SIZE + i,
            'bo_duration_sec': bo_duration_iter, 'iteration_duration_sec': iteration_duration_sec
        })
        new_row_df = pd.DataFrame([new_row_data])[CSV_HEADER]
        new_row_df.to_csv(RESULTS_CSV_PATH, mode='a', header=False, index=False)
        print(f"  Saved new evaluation to {RESULTS_CSV_PATH}")

        new_obj1_raw = simulation_results[objective_target_name_1]
        new_obj2_raw = simulation_results[objective_target_name_2]
        new_obj3_raw = simulation_results[objective_target_name_3]
        new_constraint_val_raw = simulation_results[constraint_target_name]
        print(f"  Received results: StressT={new_obj1_raw:.2f}, StressB={new_obj2_raw:.2f}, Mass={new_obj3_raw:.4f}")
        print(f"  Constraint value: DispB={new_constraint_val_raw:.4f} (Feasible: {new_constraint_val_raw < CONSTRAINT_THRESHOLD})")

        # --- Update Training Data ---
        next_y_raw = torch.tensor([[-new_obj1_raw, -new_obj2_raw, -new_obj3_raw, new_constraint_val_raw]])
        next_y_standardized = torch.tensor(y_scaler.transform(next_y_raw.numpy()), dtype=torch.float64)
        train_x_scaled = torch.cat([train_x_scaled, next_x_scaled], dim=0)
        train_y_standardized = torch.cat([train_y_standardized, next_y_standardized], dim=0)
        
        # --- Update Hypervolume and Stall Counter ---
        all_y_raw = torch.tensor(y_scaler.inverse_transform(train_y_standardized.numpy()))
        all_objectives_raw_min = -all_y_raw[:, obj_indices]
        all_constraints_raw = all_y_raw[:, constraint_index_in_y]
        
        feasible_mask = all_constraints_raw <= CONSTRAINT_THRESHOLD
        current_objectives_feasible = all_objectives_raw_min[feasible_mask]
        
        pareto_front_raw = torch.empty(0, n_objectives)
        if current_objectives_feasible.shape[0] > 0:
            non_dominated_mask = simple_is_non_dominated(current_objectives_feasible)
            pareto_front_raw = current_objectives_feasible[non_dominated_mask]
        
        current_hv = hv_calculator.compute(-pareto_front_raw) if pareto_front_raw.shape[0] > 0 else 0.0
        hypervolume_history.append(current_hv)
        current_pareto_front_size = pareto_front_raw.shape[0]

        print(f"   Current FEASIBLE Hypervolume: {current_hv:.6f}, Feasible Pareto Front Size: {current_pareto_front_size}")
        
        # Stall detection logic
        hv_improved = (current_hv > hypervolume_history[-2] + STALL_HV_IMPROVEMENT_TOLERANCE) if len(hypervolume_history) > 1 else True
        pareto_size_changed = current_pareto_front_size != previous_pareto_front_size
        
        if hv_improved or pareto_size_changed:
            stall_counter = 0
        else:
            stall_counter += 1
            print(f"    Stall counter incremented to {stall_counter}.")
        
        previous_pareto_front_size = current_pareto_front_size

        if stall_counter >= N_STALL_THRESH and forced_exploration_iters_left == 0:
            print(f"    STALLED. Forcing exploration for {FORCED_EXPLORATION_DURATION} iterations.")
            forced_exploration_iters_left = FORCED_EXPLORATION_DURATION
        
        if forced_exploration_iters_left > 0:
            forced_exploration_iters_left -= 1
            
    else: # This else belongs to the for loop
        if actual_iterations_run == MAX_TOTAL_ITERS:
            print(f"\nSTOPPING: Reached maximum total iterations ({MAX_TOTAL_ITERS}).")
            
    optimization_completed_successfully = True

except Exception as e:
    print("\n" + "="*50 + "\nAn unexpected error occurred in the main loop.")
    traceback.print_exc()
    print("Optimization stopped prematurely.")

finally:
    if evaluator:
        print("\nClosing persistent ANSA process...")
        evaluator.close()

# --- Final Reporting & Plotting ---
bo_duration = time.perf_counter() - bo_total_start_time
print("\n" + "="*50 + "\nOptimization finished.")
print(f"Ran for {actual_iterations_run} new evaluations in {bo_duration:.2f} seconds ({bo_duration/60:.2f} minutes).")

if not optimization_completed_successfully and len(hypervolume_history) <= 1:
    print("\nNo new valid results were obtained during the optimization."); sys.exit()

# Final Pareto Front Calculation
final_y_raw = torch.tensor(y_scaler.inverse_transform(train_y_standardized.numpy()))
final_objectives_raw_min = -final_y_raw[:, obj_indices]
final_constraints_raw = final_y_raw[:, constraint_index_in_y]
final_feasible_mask = final_constraints_raw <= CONSTRAINT_THRESHOLD
final_feasible_objectives = final_objectives_raw_min[final_feasible_mask]
final_infeasible_objectives = final_objectives_raw_min[~final_feasible_mask]

final_pareto_points_raw = torch.empty(0, n_objectives)
if final_feasible_objectives.shape[0] > 0:
    non_dominated_mask_final = simple_is_non_dominated(final_feasible_objectives)
    final_pareto_points_raw = final_feasible_objectives[non_dominated_mask_final]

print(f"\nFound {len(final_pareto_points_raw)} non-dominated points in the final FEASIBLE Pareto front.")
if final_pareto_points_raw.shape[0] > 0:
    sorted_indices = torch.argsort(final_pareto_points_raw[:, 0])
    sorted_pareto_front = final_pareto_points_raw[sorted_indices]
    print("Objectives (Raw, Feasible, Pareto, sorted):")
    print(pd.DataFrame(sorted_pareto_front.numpy(), columns=objective_names))

# --- Plotting ---
plt.figure(figsize=(12, 6))
plt.plot(range(len(hypervolume_history)), hypervolume_history, marker='o', linestyle='-')
plt.xlabel("ANSA Evaluation Number (0 = Initial)")
plt.ylabel("Feasible Hypervolume")
plt.title(f"Feasible Hypervolume Convergence ({script_version_name})")
plt.grid(True, which='both', linestyle='--')
plt.xlim(left=0)
plt.ylim(bottom=0)
plt.tight_layout()
plt.savefig('live_moo_adaptive_lcb_hypervolume.png')
print("\nSaved plot to 'live_moo_adaptive_lcb_hypervolume.png'")
plt.show()

fig2 = plt.figure(figsize=(11, 9))
ax2 = fig2.add_subplot(111, projection='3d')
if final_objectives_raw_min.shape[0] > 0:
    ax2.scatter(final_infeasible_objectives[:, 0].numpy(), final_infeasible_objectives[:, 1].numpy(), final_infeasible_objectives[:, 2].numpy(),
                c='lightgray', alpha=0.5, s=15, label='Infeasible Points')
    ax2.scatter(final_feasible_objectives[:, 0].numpy(), final_feasible_objectives[:, 1].numpy(), final_feasible_objectives[:, 2].numpy(),
                c='blue', alpha=0.6, s=25, label='Feasible Points')
    if final_pareto_points_raw.shape[0] > 0:
        ax2.scatter(final_pareto_points_raw[:, 0].numpy(), final_pareto_points_raw[:, 1].numpy(), final_pareto_points_raw[:, 2].numpy(),
                    c='lime', s=150, edgecolor='black', marker='*', label='Final Feasible Pareto Front', zorder=3)
ax2.set_xlabel(f"{objective_names[0]} (Minimize)")
ax2.set_ylabel(f"{objective_names[1]} (Minimize)")
ax2.set_zlabel(f"{objective_names[2]} (Minimize)")
ax2.set_title(f"3D Objective Space ({script_version_name})")
ax2.legend(loc='best')
plt.tight_layout()
plt.savefig('live_moo_adaptive_lcb_objective_space.png')
print("Saved plot to 'live_moo_adaptive_lcb_objective_space.png'")
plt.show()
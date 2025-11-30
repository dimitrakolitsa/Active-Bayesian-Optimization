import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import warnings
import time
from uuid import uuid4
from scipy.optimize import minimize # Import the scipy optimizer


# BoTorch
from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.fit import fit_gpytorch_mll
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.acquisition.multi_objective.predictive_entropy_search import qMultiObjectivePredictiveEntropySearch
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.optim import optimize_acqf
from botorch.utils.multi_objective.pareto import is_non_dominated

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

warnings.filterwarnings('ignore', category=InputDataWarning)
warnings.filterwarnings('ignore', category=BadInitialCandidatesWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

# seed pytorch and numpy
torch.manual_seed(42)
np.random.seed(42)
torch.set_default_dtype(torch.float64)

# --- Simple Pairwise Non-Dominated Check (for minimization) ---
def simple_is_non_dominated(points):
    n_points = points.shape[0]
    if n_points == 0:
        return torch.tensor([], dtype=torch.bool)
    is_nd = torch.ones(n_points, dtype=torch.bool, device=points.device)
    for i_point_loop in range(n_points):
        if not is_nd[i_point_loop]:
            continue
        for j_point_loop in range(n_points):
            if i_point_loop == j_point_loop:
                continue
            if torch.all(points[j_point_loop] <= points[i_point_loop]) and torch.any(points[j_point_loop] < points[i_point_loop]):
                is_nd[i_point_loop] = False
                break
    return is_nd


# --- Function to dynamically determine early exploration duration ---
def calculate_dynamic_early_exploration_duration(
    n_total_iterations, num_initial_samples, initial_gp_avg_variance,
    base_explore_fraction=0.30, min_explore_iters_abs=10, max_explore_fraction=0.50,
    baseline_initial_samples=50, iters_change_per_10_init_samples=-5,
    low_variance_threshold=0.2, high_variance_threshold=0.8,
    max_additional_iters_from_variance_config=None):
    if max_additional_iters_from_variance_config is None:
        max_iters_from_variance = int(n_total_iterations * 0.20)
    else:
        max_iters_from_variance = max_additional_iters_from_variance_config
    calculated_iters = float(n_total_iterations * base_explore_fraction)
    sample_diff = num_initial_samples - baseline_initial_samples
    sample_adjustment = (sample_diff/10.0) * iters_change_per_10_init_samples
    calculated_iters += sample_adjustment
    print(f"    Dynamic Early Explore: Base={n_total_iterations * base_explore_fraction:.1f}, Sample adj={sample_adjustment:.1f}")
    if initial_gp_avg_variance > low_variance_threshold:
        clamped_variance = min(high_variance_threshold, max(low_variance_threshold, initial_gp_avg_variance))
        if (high_variance_threshold - low_variance_threshold) > 1e-6:
            variance_effect_scaled = ((clamped_variance - low_variance_threshold) / (high_variance_threshold - low_variance_threshold))
        else:
            variance_effect_scaled = 1.0 if initial_gp_avg_variance > low_variance_threshold else 0.0
        variance_adjustment = variance_effect_scaled * max_iters_from_variance
        calculated_iters += variance_adjustment
        print(f"    Dynamic Early Explore: Var adj={variance_adjustment:.1f} (avg var={initial_gp_avg_variance:.3f})")
    final_duration = int(round(calculated_iters))
    final_duration = max(min_explore_iters_abs, final_duration)
    final_duration = min(final_duration, int(n_total_iterations * max_explore_fraction))
    print(f"    Dynamic Early Explore: Final calculated duration = {final_duration} iterations.")
    return final_duration

# --- Data Preprocessing ---
new_columns = [
    "FrontRear_height", "side_height", "side_width", "holes", "edge_fit", "rear_offset",
    "PSHELL_1_T", "PSHELL_2_T", "PSHELL_42733768_T", "PSHELL_42733769_T",
    "PSHELL_42733770_T", "PSHELL_42733772_T", "PSHELL_42733773_T", "PSHELL_42733774_T",
    "PSHELL_42733779_T", "PSHELL_42733780_T", "PSHELL_42733781_T", "PSHELL_42733782_T",
    "PSHELL_42733871_T", "PSHELL_42733879_T", "MAT1_1_E", "MAT1_42733768_E",
    "scale_x", "scale_y", "scale_z"
]
base_path = '../../'
try:
    df_init = pd.read_csv(f'{base_path}init/inputs.txt', header=0, sep=',')
    df_init = df_init[new_columns]
    df_all_candidates = pd.read_csv(f'{base_path}all_candidates/inputs.txt', header=0, sep=',')
    df_all_candidates = df_all_candidates[new_columns]
    init_target_files = [f"{base_path}init/mass.txt", f"{base_path}init/max_displacement_bending.txt", f"{base_path}init/max_displacement_torsion.txt", f"{base_path}init/max_stress_bending.txt", f"{base_path}init/max_stress_torsion.txt"]
    all_candidates_target_files = [f"{base_path}all_candidates/targets_mass.txt", f"{base_path}all_candidates/targets_max_displacement_bending.txt", f"{base_path}all_candidates/targets_max_displacement_torsion.txt", f"{base_path}all_candidates/targets_max_stress_bending.txt", f"{base_path}all_candidates/targets_max_stress_torsion.txt"]
    target_column_names = ["mass", "max_displacement_bending", "max_displacement_torsion", "max_stress_bending", "max_stress_torsion"]
    init_target_dfs = [pd.read_csv(file, header=None, sep=',') for file in init_target_files]
    init_df_targets = pd.concat(init_target_dfs, axis=1)
    init_df_targets.columns = target_column_names
    df_init = pd.concat([df_init, init_df_targets], axis=1)
    all_candidates_target_dfs = [pd.read_csv(file, header=None, sep=',') for file in all_candidates_target_files]
    all_candidates_df_targets = pd.concat(all_candidates_target_dfs, axis=1)
    all_candidates_df_targets.columns = target_column_names
    df_all_candidates = pd.concat([df_all_candidates, all_candidates_df_targets], axis=1)
except FileNotFoundError as e:
    print(f"Error: Data file not found: {e}. Please check base_path.")
    raise SystemExit("Data loading failed.")

X_init = torch.tensor(df_init[new_columns].values, dtype=torch.float64)
Y_init = torch.tensor(df_init[target_column_names].values, dtype=torch.float64)
X_candidates = torch.tensor(df_all_candidates[new_columns].values, dtype=torch.float64)
Y_candidates = torch.tensor(df_all_candidates[target_column_names].values, dtype=torch.float64)

num_initial_samples = X_init.shape[0]

# --- Problem Definition ---
objective_target_name_1 = "max_stress_torsion"
objective_target_name_2 = "max_stress_bending"
objective_target_name_3 = "mass"
constraint_target_name = "max_displacement_bending"
objective_names = [objective_target_name_1, objective_target_name_2, objective_target_name_3]
objective_index_1 = target_column_names.index(objective_target_name_1)
objective_index_2 = target_column_names.index(objective_target_name_2)
objective_index_3 = target_column_names.index(objective_target_name_3)
constraint_index = target_column_names.index(constraint_target_name)

train_y_raw = torch.cat([
    -Y_init[:, objective_index_1].unsqueeze(-1), -Y_init[:, objective_index_2].unsqueeze(-1),
    -Y_init[:, objective_index_3].unsqueeze(-1), Y_init[:, constraint_index].unsqueeze(-1)], dim=1)

obj_indices = [0, 1, 2]
constraint_index_in_y = 3
n_objectives = len(obj_indices)

print(f"\nOptimizing objectives: Minimize {', '.join(objective_names)}")
print(f"Subject to constraint: {constraint_target_name} < 70")

# --- Scaling ---
X_combined = torch.cat([X_init, X_candidates], dim=0)
x_scaler = MinMaxScaler()
x_scaler.fit(X_combined.numpy())
X_init_scaled = torch.tensor(x_scaler.transform(X_init.numpy()), dtype=torch.float64)
X_candidates_scaled = torch.tensor(x_scaler.transform(X_candidates.numpy()), dtype=torch.float64)
scaled_bounds = torch.tensor([[0.0] * X_init.shape[-1], [1.0] * X_init.shape[-1]], dtype=torch.float64)

y_scaler = StandardScaler()
y_scaler.fit(train_y_raw.numpy())
train_y_standardized = torch.tensor(y_scaler.transform(train_y_raw.numpy()), dtype=torch.float64)

constraint_threshold = 70.0
constraint_mean_model = y_scaler.mean_[constraint_index_in_y]
constraint_scale_model = y_scaler.scale_[constraint_index_in_y]
if constraint_scale_model == 0:
    constraint_threshold_std = 0.0 if constraint_threshold == constraint_mean_model else (float('inf') if constraint_threshold > constraint_mean_model else float('-inf'))
else:
    constraint_threshold_std = (constraint_threshold - constraint_mean_model) / constraint_scale_model
print(f"Raw constraint threshold: {constraint_threshold}, Standardized constraint threshold: {constraint_threshold_std:.4f}")

# --- GP Training Function ---
def train_independent_gps(train_x, train_y):
    models = []
    num_outputs = train_y.shape[-1]
    for i_model_fit_loop in range(num_outputs):
        train_y_i = train_y[:, i_model_fit_loop].unsqueeze(-1)
        lengthscale_prior = GammaPrior(2.0, 0.15)
        outputscale_prior = GammaPrior(2.0, 0.15)
        noise_prior = SmoothedBoxPrior(1e-5, 1e-1, sigma=0.1)
        covar_module_i = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=train_x.shape[-1], lengthscale_prior=lengthscale_prior), outputscale_prior=outputscale_prior)
        model_i = SingleTaskGP(train_x, train_y_i, covar_module=covar_module_i)
        model_i.likelihood.noise_prior = noise_prior
        mll_i = ExactMarginalLogLikelihood(model_i.likelihood, model_i)
        try:
            with gpytorch.settings.cholesky_jitter(1e-4):
                fit_gpytorch_mll(mll_i, max_retries=15, options={'maxiter': 300})
        except Exception as e:
            print(f"Warning: GP fitting failed for output {i_model_fit_loop}: {e}.")
        models.append(model_i)
    model = ModelListGP(*models)
    return model

# --- Helper Functions for c-MO-PES and Candidate Selection ---
def sample_pareto_inputs_from_posterior(full_model, obj_indices, constraint_index, constraint_threshold_std, num_samples, num_fantasy_points):
    dim = full_model.train_inputs[0][0].shape[-1]
    device = full_model.train_inputs[0][0].device
    dtype = full_model.train_inputs[0][0].dtype
    
    X_fantasy = torch.rand(num_fantasy_points, dim, device=device, dtype=dtype)
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([num_samples]))
    
    with torch.no_grad():
        posterior = full_model.posterior(X_fantasy)
        samples = sampler(posterior).squeeze(-2)
    
    sampled_pareto_inputs = []
    for i in range(num_samples):
        sample_i = samples[i]
        sample_obj = sample_i[:, obj_indices]
        sample_con = sample_i[:, constraint_index]
        
        feasible_mask = (sample_con <= constraint_threshold_std).squeeze(-1)
        if torch.any(feasible_mask):
            feasible_inputs = X_fantasy[feasible_mask]
            feasible_obj = sample_obj[feasible_mask]
            
            if feasible_obj.shape[0] > 0:
                pareto_mask = is_non_dominated(feasible_obj)
                if pareto_mask.any():
                    pareto_inputs = feasible_inputs[pareto_mask]
                    sampled_pareto_inputs.append(pareto_inputs)
                    
    return sampled_pareto_inputs

def prepare_pareto_sets_for_pes(sampled_inputs_list, full_model):
    if not sampled_inputs_list:
        print("    Helper `prepare_pareto_sets_for_pes`: Input list is empty.")
        return None

    max_p = max(s.shape[0] for s in sampled_inputs_list)
    num_samples_eff = len(sampled_inputs_list)
    dim = full_model.train_inputs[0][0].shape[-1]
    device = full_model.train_inputs[0][0].device
    dtype = full_model.train_inputs[0][0].dtype

    padded_sets = torch.full((num_samples_eff, max_p, dim), torch.nan, device=device, dtype=dtype)

    for i, s in enumerate(sampled_inputs_list):
        padded_sets[i, :s.shape[0], :] = s
    
    for i in range(num_samples_eff):
        last_valid_idx_tensor = torch.where(~torch.isnan(padded_sets[i, :, 0]))[0]
        if len(last_valid_idx_tensor) > 0:
            last_valid_idx = last_valid_idx_tensor.max()
            for j in range(last_valid_idx + 1, max_p):
                padded_sets[i, j, :] = padded_sets[i, last_valid_idx, :]
        else:
            padded_sets[i, :, :] = 0.0
            
    return padded_sets

# --- NEW: Custom Gradient-Free Optimizer Wrapper ---
def constrained_gradient_free_optimizer(
    acq_function,
    full_model,
    constraint_index_in_y,
    constraint_threshold_std,
    bounds,
    num_restarts,
    raw_samples,
):
    """
    Optimizes a BoTorch acquisition function using a gradient-free
    scipy optimizer (COBYLA) while enforcing a model-predicted constraint.
    """
    dim = bounds.shape[1]
    device = bounds.device
    dtype = bounds.dtype

    initial_conditions = torch.rand(num_restarts, 1, dim, device=device, dtype=dtype)
    initial_conditions = bounds[0] + (bounds[1] - bounds[0]) * initial_conditions

    best_x = None
    best_acq_value = -float('inf')

    # 1. Define the objective for scipy.optimize.minimize
    def obj_func(x):
        x_tensor = torch.from_numpy(x).reshape(1, 1, dim).to(device, dtype)
        # We negate because scipy minimizes and botorch maximizes
        acq_val = -acq_function(x_tensor).item()
        return acq_val
        
    # 2. Define the constraint function for scipy
    # The constraint is of the form: c(x) >= 0
    # Our original constraint is: displacement <= 70
    # In standardized space: y_std <= C_std
    # So, we want: C_std - y_std >= 0
    def constraint_func(x):
        x_tensor = torch.from_numpy(x).reshape(1, dim).to(device, dtype)
        with torch.no_grad():
            posterior = full_model.posterior(x_tensor)
            mean_constraint = posterior.mean[..., constraint_index_in_y]
        
        # We return C_std - model_mean(x)
        return constraint_threshold_std - mean_constraint.item()

    # Create constraint dictionary for scipy
    scipy_constraints = [{'type': 'ineq', 'fun': constraint_func}]
    
    # Create bounds for scipy
    scipy_bounds = [(low.item(), high.item()) for low, high in zip(bounds[0], bounds[1])]

    for i, x0 in enumerate(initial_conditions):
        res = minimize(
            fun=obj_func,
            x0=x0.squeeze().cpu().numpy(),
            bounds=scipy_bounds,
            constraints=scipy_constraints, # Pass the constraint here
            method='COBYLA',
            options={'maxiter': 500}
        )
        # Check if the solution is considered feasible by the constraint function
        # before accepting it as the best.
        if res.success and constraint_func(res.x) >= 0:
            if -res.fun > best_acq_value:
                best_acq_value = -res.fun
                best_x = torch.from_numpy(res.x).reshape(1, dim).to(device, dtype)

    return best_x, torch.tensor(best_acq_value, device=device, dtype=dtype)



def get_cmo_pes_candidate(full_model, obj_indices, constraint_index_in_y, constraint_threshold_std, bounds):
    print("  Setting up Constrained Multi-Objective PES (c-MO-PES)...")
    try:
        print("    Step 1: Sampling Pareto sets from posterior...")
        num_samples = 8 #16
        num_fantasy_points = 128 #256 #512
        sampled_inputs_list = sample_pareto_inputs_from_posterior(
            full_model, obj_indices, constraint_index_in_y, constraint_threshold_std, num_samples, num_fantasy_points
        )
        if not sampled_inputs_list:
            raise ValueError("Sampling from posterior yielded no feasible Pareto sets.")
        print(f"    ... success, found {len(sampled_inputs_list)} potential Pareto sets.")

        print("    Step 2: Preparing and padding Pareto sets...")
        padded_sets = prepare_pareto_sets_for_pes(sampled_inputs_list, full_model)
        if padded_sets is None:
            raise ValueError("Failed to prepare and pad Pareto sets.")
        print(f"    ... success, padded sets created with shape {padded_sets.shape}.")

        print("    Step 3: Creating objective-only sub-model...")
        objective_model = ModelListGP(*full_model.models[:len(obj_indices)])
        print("    ... success.")

        print("    Step 4: Creating the c-MO-PES acquisition function...")
        acqf = qMultiObjectivePredictiveEntropySearch(
            model=objective_model,
            pareto_sets=padded_sets,
            ep_jitter=1e-3, 
            test_jitter=1e-3,
        )
        print("    ... success.")

        print("    Step 5: Optimizing using CUSTOM CONSTRAINED gradient-free wrapper...")
        # Use the new constrained optimizer
        next_x_scaled, acq_value = constrained_gradient_free_optimizer(
            acq_function=acqf,
            full_model=full_model, # Pass the full model for constraint evaluation
            constraint_index_in_y=constraint_index_in_y,
            constraint_threshold_std=constraint_threshold_std,
            bounds=bounds,
            num_restarts=5,#10,
            raw_samples=256 #1024,
        )
        
        print(f"    ... success. c-MO-PES optimized with max value: {acq_value.item():.4f}")
        return next_x_scaled, acq_value

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"  Warning: c-MO-PES pipeline failed: {e}. Falling back to LCB logic.")
        return None, None
    

def find_candidate_by_index(X_cands, Y_cands, obj_idx_1, obj_idx_2, obj_idx_3, constr_idx, candidate_idx):
    closest_X_unscaled = X_cands[candidate_idx].unsqueeze(0)
    raw_obj1 = Y_cands[candidate_idx, obj_idx_1].item()
    raw_obj2 = Y_cands[candidate_idx, obj_idx_2].item()
    raw_obj3 = Y_cands[candidate_idx, obj_idx_3].item()
    raw_constr = Y_cands[candidate_idx, constr_idx].item()
    return (candidate_idx, closest_X_unscaled,
            torch.tensor([[-raw_obj1]]), torch.tensor([[-raw_obj2]]),
            torch.tensor([[-raw_obj3]]), torch.tensor([[raw_constr]]))

# --- BO Loop Setup ---
n_iterations = 150
evaluated_candidate_indices = set(range(len(X_init)))
train_x_scaled = X_init_scaled.clone()
train_y_standardized = train_y_standardized.clone()
hypervolume_history = []
selection_methods = []
stall_counter = 0

alpha_variance = 0.2
alpha_hv_improvement = 0.3
beta_states = torch.tensor([0.5, 1.0, 1.5, 2.0, 2.5, 3.0], dtype=torch.float64, device=X_init.device)
smooth_avg_variance = torch.tensor(0.0, dtype=torch.float64, device=X_init.device)
smooth_hv_improvement = torch.tensor(0.0, dtype=torch.float64, device=X_init.device)
LOW_HV_IMPROVEMENT_THRESH = 1e-5
HIGH_VARIANCE_THRESH = 0.6
MODERATE_VARIANCE_THRESH = 0.3
LOW_VARIANCE_THRESH = 0.1
DYNAMIC_EARLY_EXPLORE_ITERATIONS = None
MID_ITER_MAX_EMA = n_iterations * 0.70
N_STALL_THRESH = 8
FORCED_EXPLORATION_DURATION = 10
forced_exploration_iters_left = 0
STALL_HV_IMPROVEMENT_TOLERANCE = 1e-5
LAST_N_ITER_FORCED_BETA_2_0_AND_EI = 50
beta_alternation_block_length = 10
forced_stall_weight_cycle = [torch.tensor([0.8, 0.1, 0.1]), torch.tensor([0.1, 0.8, 0.1]), torch.tensor([0.1, 0.1, 0.8]), torch.tensor([0.4, 0.4, 0.2]), torch.tensor([0.4, 0.2, 0.4]), torch.tensor([0.2, 0.4, 0.4])]
forced_stall_weight_idx = 0

hv_ref_point_raw = torch.tensor([1500.0, 1500.0, 1.0], dtype=torch.float64)
negated_ref_point_for_hv_calc = -hv_ref_point_raw
print(f"\nUsing raw HV reference point for plots: {hv_ref_point_raw.tolist()}")

# --- Initial HV Calculation ---
initial_objectives_raw = Y_init[:, [objective_index_1, objective_index_2, objective_index_3]]
initial_constraints_raw = Y_init[:, constraint_index]
feasible_mask_init = initial_constraints_raw <= constraint_threshold
initial_objectives_feasible = initial_objectives_raw[feasible_mask_init]
initial_hv = 0.0
initial_pareto_raw_size = 0
if initial_objectives_feasible.shape[0] > 0:
    non_dominated_mask_init = simple_is_non_dominated(initial_objectives_feasible)
    initial_pareto_raw = initial_objectives_feasible[non_dominated_mask_init]
    initial_pareto_raw_size = initial_pareto_raw.shape[0]
    if initial_pareto_raw.shape[0] > 0:
        try:
            hv_calculator = Hypervolume(ref_point=negated_ref_point_for_hv_calc)
            initial_hv = hv_calculator.compute(-initial_pareto_raw)
        except Exception as e:
            print(f"Warning: Initial HV calc failed: {e}.")
print(f"Initial Hypervolume: {initial_hv:.6f}, Initial Pareto Size: {initial_pareto_raw_size}")
hypervolume_history.append(initial_hv)
previous_pareto_front_size = initial_pareto_raw_size

script_version_name = "c-MO-PES_with_LCB_Fallback_Final"
print(f"\nStarting Bayesian Optimization for {n_iterations} iterations ({script_version_name})...")
start_bo_time = time.monotonic()

# --- BO Loop ---
for i_iter_loop in range(n_iterations):
    iter_display = i_iter_loop + 1
    print(f"\nIteration {iter_display}/{n_iterations}")

    print("  Training Independent GPs...")
    full_model = train_independent_gps(train_x_scaled, train_y_standardized)
    full_model.eval()

    next_x_scaled = None
    
    train_y_raw_inv = torch.tensor(y_scaler.inverse_transform(train_y_standardized.numpy()))
    train_constraints_raw = train_y_raw_inv[:, constraint_index_in_y]
    feasible_mask_train = train_constraints_raw <= constraint_threshold
    
    if torch.any(feasible_mask_train):
        next_x_scaled, _ = get_cmo_pes_candidate(
            full_model=full_model,
            obj_indices=obj_indices,
            constraint_index_in_y=constraint_index_in_y,
            constraint_threshold_std=constraint_threshold_std,
            bounds=scaled_bounds
        )
    else:
        print("  Warning: No feasible points found yet. Cannot use c-MO-PES. Falling back to LCB logic.")
        next_x_scaled = None
    
    selected_candidate_original_idx = -1
    
    if next_x_scaled is not None:
        selection_methods.append('c-MO-PES')
        next_x_unscaled = torch.tensor(x_scaler.inverse_transform(next_x_scaled.numpy()))
        available_indices_list = [idx for idx in range(len(X_candidates)) if idx not in evaluated_candidate_indices]
        if not available_indices_list:
            break
        available_indices = torch.tensor(available_indices_list, dtype=torch.long)
        distances = torch.norm(X_candidates[available_indices] - next_x_unscaled.squeeze(0), dim=1)
        closest_idx_in_subset = torch.argmin(distances)
        selected_candidate_original_idx = available_indices[closest_idx_in_subset].item()
    
    if selected_candidate_original_idx == -1:
        if not selection_methods or (len(selection_methods) == i_iter_loop):
             selection_methods.append('LCB_Fallback')
        elif selection_methods[-1] != 'LCB_Fallback':
             selection_methods.append('LCB_Fallback')
        
        print("   Executing LCB Fallback...")
        
        available_candidate_indices_orig = [idx for idx in range(len(X_candidates)) if idx not in evaluated_candidate_indices]
        if not available_candidate_indices_orig:
            print("   LCB Fallback: No unevaluated candidates left. Terminating early.")
            break
        
        unevaluated_indices_tensor = torch.tensor(available_candidate_indices_orig, dtype=torch.long)
        X_candidates_scaled_subset = X_candidates_scaled[unevaluated_indices_tensor]
        
        with torch.no_grad():
            posterior = full_model.posterior(X_candidates_scaled_subset)
            means_standardized, variances_standardized = posterior.mean, posterior.variance
        stds_standardized = (variances_standardized.clamp(min=1e-9)).sqrt()

        mean_obj1_std = means_standardized[:, obj_indices[0]]
        std_obj1_std = stds_standardized[:, obj_indices[0]]
        mean_obj2_std = means_standardized[:, obj_indices[1]]
        std_obj2_std = stds_standardized[:, obj_indices[1]]
        mean_obj3_std = means_standardized[:, obj_indices[2]]
        std_obj3_std = stds_standardized[:, obj_indices[2]]
        mean_constr_std = means_standardized[:, constraint_index_in_y]
        std_constr_std = stds_standardized[:, constraint_index_in_y]

        # NOTE: Your full adaptive LCB logic should be placed here for best performance
        current_beta_lcb = 1.5
        w1, w2, w3 = 1/3., 1/3., 1/3.
        beta_constraint = 1.0
        constraint_lcb_values = mean_constr_std - beta_constraint * std_constr_std
        predicted_feasibility_mask = (constraint_lcb_values <= constraint_threshold_std)
        penalty_value = -1e9
        
        if not torch.any(predicted_feasibility_mask):
            print("    Warning: LCB Fallback - All candidates predicted infeasible. Selecting best LCB ignoring constraints.")
            scalarized_lcb_std = (
                w1 * (mean_obj1_std - current_beta_lcb * std_obj1_std) +
                w2 * (mean_obj2_std - current_beta_lcb * std_obj2_std) +
                w3 * (mean_obj3_std - current_beta_lcb * std_obj3_std) )
            best_lcb_idx_in_subset = torch.argmax(scalarized_lcb_std)
        else:
            effective_scalarized_lcb_std = torch.full_like(mean_obj1_std, penalty_value)
            feasible_indices = torch.where(predicted_feasibility_mask)[0]
            if feasible_indices.numel() > 0:
                scalarized_lcb_std_values_feas = (
                    w1 * (mean_obj1_std[feasible_indices] - current_beta_lcb * std_obj1_std[feasible_indices]) +
                    w2 * (mean_obj2_std[feasible_indices] - current_beta_lcb * std_obj2_std[feasible_indices]) +
                    w3 * (mean_obj3_std[feasible_indices] - current_beta_lcb * std_obj3_std[feasible_indices]) )
                effective_scalarized_lcb_std[feasible_indices] = scalarized_lcb_std_values_feas
            best_lcb_idx_in_subset = torch.argmax(effective_scalarized_lcb_std)

        selected_candidate_original_idx = unevaluated_indices_tensor[best_lcb_idx_in_subset].item()

    print(f"   Selected candidate (Index: {selected_candidate_original_idx}) via {selection_methods[-1]}")

    _, _, next_y_obj1_raw, next_y_obj2_raw, next_y_obj3_raw, next_y_constr_raw = find_candidate_by_index(
        X_candidates, Y_candidates, objective_index_1, objective_index_2, objective_index_3, constraint_index, selected_candidate_original_idx)
    
    closest_cand_X_scaled = X_candidates_scaled[selected_candidate_original_idx].unsqueeze(0)

    print(f"   Raw Objective 1 ({objective_target_name_1}): {-next_y_obj1_raw.item():.4f}")
    print(f"   Raw Objective 2 ({objective_target_name_2}): {-next_y_obj2_raw.item():.4f}")
    print(f"   Raw Objective 3 ({objective_target_name_3}): {-next_y_obj3_raw.item():.6f}")
    print(f"   Constraint ({constraint_target_name}): {next_y_constr_raw.item():.4f}")

    next_y_combined_raw = torch.cat([next_y_obj1_raw, next_y_obj2_raw, next_y_obj3_raw, next_y_constr_raw], dim=1)
    next_y_standardized = torch.tensor(y_scaler.transform(next_y_combined_raw.numpy()))

    train_x_scaled = torch.cat([train_x_scaled, closest_cand_X_scaled], dim=0)
    train_y_standardized = torch.cat([train_y_standardized, next_y_standardized], dim=0)
    evaluated_candidate_indices.add(selected_candidate_original_idx)

    # --- HV Calculation & Stall Counter Logic ---
    all_y_raw_from_standardized = torch.tensor(y_scaler.inverse_transform(train_y_standardized.numpy()))
    all_objectives_raw_minimization = -all_y_raw_from_standardized[:, obj_indices]
    all_constraints_raw = all_y_raw_from_standardized[:, constraint_index_in_y]
    feasible_mask = all_constraints_raw <= constraint_threshold
    current_objectives_feasible_raw = all_objectives_raw_minimization[feasible_mask]
    current_hv = 0.0
    current_pareto_front_size = 0
    if current_objectives_feasible_raw.shape[0] > 0:
        non_dominated_mask = simple_is_non_dominated(current_objectives_feasible_raw)
        pareto_front_raw_minimization = current_objectives_feasible_raw[non_dominated_mask]
        current_pareto_front_size = pareto_front_raw_minimization.shape[0]
        if current_pareto_front_size > 0:
            try:
                hv_calculator = Hypervolume(ref_point=negated_ref_point_for_hv_calc)
                current_hv = hv_calculator.compute(-pareto_front_raw_minimization)
            except Exception as e:
                current_hv = hypervolume_history[-1] if hypervolume_history else 0.0
    hypervolume_history.append(current_hv)
    
    hv_improved = (len(hypervolume_history) > 1 and hypervolume_history[-1] > hypervolume_history[-2] + STALL_HV_IMPROVEMENT_TOLERANCE)
    pareto_size_changed = (current_pareto_front_size != previous_pareto_front_size)
    if hv_improved or pareto_size_changed:
        if stall_counter > 0:
            print(f"    Stall counter reset from {stall_counter} to 0.")
        stall_counter = 0
    else:
        stall_counter += 1
        print(f"    Stall counter incremented to {stall_counter}.")
    previous_pareto_front_size = current_pareto_front_size

    if stall_counter >= N_STALL_THRESH and forced_exploration_iters_left == 0:
        print(f"    STALLED. Forcing exploration for {FORCED_EXPLORATION_DURATION} iters.")
    
    print(f"   Current Hypervolume: {current_hv:.6f}, Pareto Front Size: {current_pareto_front_size}")

# --- End of BO Loop ---
end_bo_time = time.monotonic()
bo_duration_seconds = end_bo_time - start_bo_time
print("\nOptimization finished.")
print(f"Total BO loop duration: {bo_duration_seconds:.2f} seconds ({bo_duration_seconds/60:.2f} minutes)")

# --- Results, True Pareto, and Plotting Code (UNCHANGED) ---
final_y_all_standardized = train_y_standardized
final_y_all_raw_model_form = torch.tensor(y_scaler.inverse_transform(final_y_all_standardized.numpy()), dtype=torch.float64)
final_objectives_raw_minimization = -final_y_all_raw_model_form[:, obj_indices]
final_constraints_raw = final_y_all_raw_model_form[:, constraint_index_in_y]
feasible_mask_final = final_constraints_raw <= constraint_threshold
final_objectives_feasible_minimization = final_objectives_raw_minimization[feasible_mask_final]
final_bo_pareto_points_raw_minimization = torch.empty((0, n_objectives), dtype=torch.float64)
if final_objectives_feasible_minimization.shape[0] > 0:
    non_dominated_mask_final = simple_is_non_dominated(final_objectives_feasible_minimization)
    final_bo_pareto_points_raw_minimization = final_objectives_feasible_minimization[non_dominated_mask_final]
    if final_bo_pareto_points_raw_minimization.shape[0] > 0:
        print(f"\nFound {len(final_bo_pareto_points_raw_minimization)} BO Pareto points:")
        print(pd.DataFrame(final_bo_pareto_points_raw_minimization.numpy(), columns=objective_names))
Y_init_objectives_raw_min = Y_init[:, [objective_index_1, objective_index_2, objective_index_3]]
Y_init_constraints_raw = Y_init[:, constraint_index]
Y_candidates_objectives_raw_min = Y_candidates[:, [objective_index_1, objective_index_2, objective_index_3]]
Y_candidates_constraints_raw = Y_candidates[:, constraint_index]
true_space_objectives_min = torch.cat([Y_init_objectives_raw_min, Y_candidates_objectives_raw_min], dim=0)
true_space_constraints = torch.cat([Y_init_constraints_raw, Y_candidates_constraints_raw], dim=0)
feasible_mask_true_space = true_space_constraints <= constraint_threshold
true_space_objectives_feasible_min = true_space_objectives_min[feasible_mask_true_space]
unique_true_space_objectives_min = None
try:
    precision_dup = 1e-5 
    rounded_objectives = np.round(true_space_objectives_feasible_min.numpy() / precision_dup) * precision_dup
    df_true_space = pd.DataFrame(rounded_objectives, columns=[f'obj{i+1}' for i in range(n_objectives)])
    unique_indices = df_true_space.drop_duplicates().index
    unique_true_space_objectives_min = true_space_objectives_feasible_min[unique_indices]
except Exception as e:
    print(f"Warning: Error during de-duplication of true space objectives: {e}")
    unique_true_space_objectives_min = true_space_objectives_feasible_min 
true_pareto_front_raw_minimization = torch.empty((0, n_objectives), dtype=torch.float64)
true_hv = 0.0
if unique_true_space_objectives_min is not None and unique_true_space_objectives_min.shape[0] > 0:
    non_dominated_mask_true_space = simple_is_non_dominated(unique_true_space_objectives_min)
    true_pareto_front_raw_minimization = unique_true_space_objectives_min[non_dominated_mask_true_space]
    
    print(f"\nTrue Pareto Front ({true_pareto_front_raw_minimization.shape[0]} points, minimization objectives):")
    print(pd.DataFrame(true_pareto_front_raw_minimization.numpy(), columns=objective_names).to_string()) 

    if true_pareto_front_raw_minimization.shape[0] > 0:
        try:
            hv_calculator_true = Hypervolume(ref_point=negated_ref_point_for_hv_calc)
            true_hv = hv_calculator_true.compute(-true_pareto_front_raw_minimization)
            print(f"True Max Hypervolume (from unique feasible points): {true_hv:.6f}")
        except Exception as e:
            print(f"Warning: True HV calculation error: {e}")
            true_hv = 0.0
        final_bo_hv = hypervolume_history[-1] if hypervolume_history else 0.0
        print(f"Final BO Hypervolume: {final_bo_hv:.6f}")
        if true_hv > 1e-12:
            print(f"HV Ratio (BO/True): {final_bo_hv / true_hv:.4f}")
else:
    print("Could not determine true Pareto front (no unique feasible points or error).")

plot_suffix = f"{script_version_name}_iter{n_iterations}"

plt.figure(figsize=(12, 6))
if len(selection_methods) > 0:
    bo_iterations_plot = list(range(1, len(selection_methods) + 1))
    bo_hypervolumes_plot = hypervolume_history[1:]
    if len(bo_iterations_plot) == len(bo_hypervolumes_plot):
        plt.plot(bo_iterations_plot, bo_hypervolumes_plot, color='darkgrey', linestyle='-', zorder=1, alpha=0.7)
        color_map = {'c-MO-PES': '#377EB8', 'LCB_Fallback': '#E41A1C', 'Error': 'grey'}
        plot_colors = [color_map.get(method, 'black') for method in selection_methods]
        plt.scatter(bo_iterations_plot, bo_hypervolumes_plot, c=plot_colors, marker='o', s=50, zorder=2, edgecolors='grey', alpha=0.9)
if true_pareto_front_raw_minimization.shape[0] > 0 and true_hv > 1e-9:
    plt.axhline(y=true_hv, color='green', linestyle='--', label=f'True Max HV ({true_hv:.4f})')
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='c-MO-PES', markerfacecolor='#377EB8', markersize=8),
    Line2D([0], [0], marker='o', color='w', label='LCB Fallback', markerfacecolor='#E41A1C', markersize=8)
]
plt.legend(handles=legend_elements)
plt.xlabel("BO Iteration")
plt.ylabel("Current HV")
plt.title(f"BO HV Progression by Selection Method ({plot_suffix})")
plt.grid(True)
plt.xlim(left=0)
plt.tight_layout()
plt.savefig(f'selection_method_plot_{plot_suffix}.png')
plt.close()

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
if final_objectives_raw_minimization.shape[0] > 0:
    obj1_all_min=final_objectives_raw_minimization[:,0].numpy()
    obj2_all_min=final_objectives_raw_minimization[:,1].numpy()
    obj3_all_min=final_objectives_raw_minimization[:,2].numpy()
    constr_all_final=final_constraints_raw.numpy()
    feasible_plot_mask = constr_all_final <= constraint_threshold
    ax.scatter(obj1_all_min[~feasible_plot_mask], obj2_all_min[~feasible_plot_mask], obj3_all_min[~feasible_plot_mask], c='grey', alpha=0.2, s=15, label='Infeasible Eval')
    ax.scatter(obj1_all_min[feasible_plot_mask], obj2_all_min[feasible_plot_mask], obj3_all_min[feasible_plot_mask], c='blue', alpha=0.4, s=15, label='Feasible Eval')
    if final_bo_pareto_points_raw_minimization.shape[0] > 0: 
        ax.scatter(final_bo_pareto_points_raw_minimization[:,0].numpy(), final_bo_pareto_points_raw_minimization[:,1].numpy(), final_bo_pareto_points_raw_minimization[:,2].numpy(), c='lime', s=150, edgecolor='black', marker='*', label=f'BO Pareto ({plot_suffix})', zorder=3)
    if true_pareto_front_raw_minimization.shape[0] > 0: 
        ax.scatter(true_pareto_front_raw_minimization[:,0].numpy(), true_pareto_front_raw_minimization[:,1].numpy(), true_pareto_front_raw_minimization[:,2].numpy(), facecolors='none', edgecolors='red', marker='o', s=60, linewidth=1.5, label='True Pareto', zorder=2)
    ax.set_xlabel(f"{objective_names[0]} (Min)")
    ax.set_ylabel(f"{objective_names[1]} (Min)")
    ax.set_zlabel(f"{objective_names[2]} (Min)")
    ax.set_title(f"3D Objective Space ({plot_suffix})")
    ax.legend(loc='best')
    plt.tight_layout()
    plt.savefig(f'objective_space_3d_{plot_suffix}.png')
    plt.close()
else: 
    print("\nNo points to plot 3D objective space.")

print("\nFinal training data size (scaled features):", train_x_scaled.shape)
print(f"Total points in hypervolume_history plot: {len(hypervolume_history)}")
print(f"Selection methods recorded: {selection_methods.count('c-MO-PES')} c-MO-PES, {selection_methods.count('LCB_Fallback')} LCB_Fallback, {selection_methods.count('Error')} Error")
print(f"Script finished. ({script_version_name})")
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import warnings
import time
from uuid import uuid4

# BoTorch
from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.fit import fit_gpytorch_mll
from botorch.utils.multi_objective.hypervolume import Hypervolume

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
            if torch.all(points[j_point_loop] <= points[i_point_loop]) and \
               torch.any(points[j_point_loop] < points[i_point_loop]):
                is_nd[i_point_loop] = False
                break
    return is_nd

# --- Function to dynamically determine early exploration duration ---
def calculate_dynamic_early_exploration_duration(
    n_total_iterations,
    num_initial_samples,
    initial_gp_avg_variance,
    base_explore_fraction=0.30,
    min_explore_iters_abs=10,
    max_explore_fraction=0.50,
    baseline_initial_samples=50,
    iters_change_per_10_init_samples=-5,
    low_variance_threshold=0.2,
    high_variance_threshold=0.8,
    max_additional_iters_from_variance_config=None):

    if max_additional_iters_from_variance_config is None:
        max_iters_from_variance = int(n_total_iterations * 0.20)
    else:
        max_iters_from_variance = max_additional_iters_from_variance_config
    calculated_iters = float(n_total_iterations * base_explore_fraction)
    sample_diff = num_initial_samples - baseline_initial_samples
    sample_adjustment = (sample_diff/10.0) * iters_change_per_10_init_samples
    calculated_iters += sample_adjustment
    print(f"    Dynamic Early Explore: Base iters ({base_explore_fraction*100:.0f}% of budget={n_total_iterations})={n_total_iterations * base_explore_fraction:.1f}, Sample adjustment based on N_init={num_initial_samples} is {sample_adjustment:.1f}")
    if initial_gp_avg_variance > low_variance_threshold:
        clamped_variance = min(high_variance_threshold, max(low_variance_threshold, initial_gp_avg_variance))
        if (high_variance_threshold - low_variance_threshold) > 1e-6:
            variance_effect_scaled = ((clamped_variance - low_variance_threshold) / (high_variance_threshold - low_variance_threshold))
        else:
            variance_effect_scaled = 1.0 if initial_gp_avg_variance > low_variance_threshold else 0.0
        variance_adjustment = variance_effect_scaled * max_iters_from_variance
        calculated_iters += variance_adjustment
        print(f"    Dynamic Early Explore: Initial GP avg var={initial_gp_avg_variance:.3f} (scaled effect: {variance_effect_scaled:.2f}), Variance adjustment={variance_adjustment:.1f}")
    else:
        print(f"    Dynamic Early Explore: Initial GP avg var={initial_gp_avg_variance:.3f} <= low_thresh={low_variance_threshold}, no positive variance adjustment.")
    final_duration = int(round(calculated_iters))
    print(f"    Dynamic Early Explore: Calculated iters before caps = {final_duration}")
    final_duration = max(min_explore_iters_abs, final_duration)
    final_duration = min(final_duration, int(n_total_iterations * max_explore_fraction))
    print(f"    Dynamic Early Explore: Final calculated duration = {final_duration} iterations.")
    return final_duration

# --- Data Preprocessing  ---
new_columns = [
    "FrontRear_height", "side_height", "side_width", "holes", "edge_fit", "rear_offset",
    "PSHELL_1_T", "PSHELL_2_T", "PSHELL_42733768_T", "PSHELL_42733769_T",
    "PSHELL_42733770_T", "PSHELL_42733772_T", "PSHELL_42733773_T", "PSHELL_42733774_T",
    "PSHELL_42733779_T", "PSHELL_42733780_T", "PSHELL_42733781_T", "PSHELL_42733782_T",
    "PSHELL_42733871_T", "PSHELL_42733879_T", "MAT1_1_E", "MAT1_42733768_E",
    "scale_x", "scale_y", "scale_z"
]
base_path = '../../' # IMPORTANT: Adjust this to your actual data path
try:
    df_init = pd.read_csv(f'{base_path}init/inputs.txt', header=0, sep=',')
    df_init = df_init[new_columns]
    df_all_candidates = pd.read_csv(f'{base_path}all_candidates/inputs.txt', header=0, sep=',')
    df_all_candidates = df_all_candidates[new_columns]
    init_target_files = [
        f"{base_path}init/mass.txt",
        f"{base_path}init/max_displacement_bending.txt",
        f"{base_path}init/max_displacement_torsion.txt",
        f"{base_path}init/max_stress_bending.txt",
        f"{base_path}init/max_stress_torsion.txt"
    ]
    all_candidates_target_files = [
        f"{base_path}all_candidates/targets_mass.txt",
        f"{base_path}all_candidates/targets_max_displacement_bending.txt",
        f"{base_path}all_candidates/targets_max_displacement_torsion.txt",
        f"{base_path}all_candidates/targets_max_stress_bending.txt",
        f"{base_path}all_candidates/targets_max_stress_torsion.txt"
    ]
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
    print(f"Error: Data file not found: {e}. Please check base_path: '{base_path}'")
    raise SystemExit("Data loading failed.")

X_init = torch.tensor(df_init[new_columns].values, dtype=torch.float64)
Y_init = torch.tensor(df_init[target_column_names].values, dtype=torch.float64)
X_candidates = torch.tensor(df_all_candidates[new_columns].values, dtype=torch.float64)
Y_candidates = torch.tensor(df_all_candidates[target_column_names].values, dtype=torch.float64)
num_initial_samples = X_init.shape[0]

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
    -Y_init[:, objective_index_1].unsqueeze(-1),
    -Y_init[:, objective_index_2].unsqueeze(-1),
    -Y_init[:, objective_index_3].unsqueeze(-1),
    Y_init[:, constraint_index].unsqueeze(-1)
], dim=1)
obj_indices = [0, 1, 2]
constraint_index_in_y = 3
n_objectives = len(obj_indices)
print(f"\nOptimizing objectives: Minimize {', '.join(objective_names)}")
print(f"Subject to constraint: {constraint_target_name} < 70")

X_combined = torch.cat([X_init, X_candidates], dim=0)
x_scaler = MinMaxScaler()
x_scaler.fit(X_combined.numpy())
X_init_scaled = torch.tensor(x_scaler.transform(X_init.numpy()), dtype=torch.float64)
X_candidates_scaled = torch.tensor(x_scaler.transform(X_candidates.numpy()), dtype=torch.float64)

y_scaler = StandardScaler()
y_scaler.fit(train_y_raw.numpy())
train_y_standardized = torch.tensor(y_scaler.transform(train_y_raw.numpy()), dtype=torch.float64)

constraint_threshold = 70.0
constraint_mean_model = y_scaler.mean_[constraint_index_in_y]
constraint_scale_model = y_scaler.scale_[constraint_index_in_y]
if constraint_scale_model == 0:
    if constraint_threshold == constraint_mean_model:
        constraint_threshold_std = 0.0
    elif constraint_threshold > constraint_mean_model:
        constraint_threshold_std = float('inf')
    else:
        constraint_threshold_std = float('-inf')
else:
    constraint_threshold_std = (constraint_threshold - constraint_mean_model) / constraint_scale_model
print(f"Raw constraint threshold: {constraint_threshold}, Standardized constraint threshold: {constraint_threshold_std:.4f}")

def train_independent_gps(train_x, train_y):
    models = []
    num_outputs = train_y.shape[-1]
    for i_model_fit_loop in range(num_outputs):
        train_y_i = train_y[:, i_model_fit_loop].unsqueeze(-1)
        lengthscale_prior = GammaPrior(2.0, 0.15)
        outputscale_prior = GammaPrior(2.0, 0.15)
        noise_prior = SmoothedBoxPrior(1e-5, 1e-1, sigma=0.1)
        covar_module_i = ScaleKernel(
            MaternKernel(nu=2.5, ard_num_dims=train_x.shape[-1], lengthscale_prior=lengthscale_prior),
            outputscale_prior=outputscale_prior
        )
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

def find_closest_candidate(X_cands, Y_cands, obj_idx_1, obj_idx_2, obj_idx_3, constr_idx, candidate_index_from_lcb):
    closest_X = X_cands[candidate_index_from_lcb].unsqueeze(0)
    raw_obj1 = Y_cands[candidate_index_from_lcb, obj_idx_1].item()
    raw_obj2 = Y_cands[candidate_index_from_lcb, obj_idx_2].item()
    raw_obj3 = Y_cands[candidate_index_from_lcb, obj_idx_3].item()
    raw_constr = Y_cands[candidate_index_from_lcb, constr_idx].item()
    return (candidate_index_from_lcb, closest_X,
            torch.tensor([[-raw_obj1]]), torch.tensor([[-raw_obj2]]),
            torch.tensor([[-raw_obj3]]), torch.tensor([[raw_constr]]))

# --- BO Loop Setup ---
# Scenario A (Fixed 300 iterations for 15k candidates):
# BASE_ITERATIONS = 300
# MAX_POSSIBLE_ITERATIONS_CAP = 300 # This makes it stop at 300

# Scenario B (Min 300, then simple stopping for 20k+ candidates):
BASE_ITERATIONS = 300
MAX_POSSIBLE_ITERATIONS_CAP = 600 # Max iterations if stopping criterion is not met

# New Simple Stopping Criterion (Only active *after* BASE_ITERATIONS)
MIN_HV_IMPROVEMENT_FOR_CONTINUATION = 1e-6 # 0.000001
PATIENCE_FOR_NO_HV_IMPROVEMENT_AFTER_BASE = 25
iters_with_no_significant_hv_improvement_after_base = 0

alpha_variance = 0.2
alpha_hv_improvement = 0.2
beta_states = torch.tensor([0.5, 1.0, 1.5, 2.0, 2.5, 3.0], dtype=torch.float64, device=X_init.device)
smooth_avg_variance = torch.tensor(0.0, dtype=torch.float64, device=X_init.device)
smooth_hv_improvement_for_beta = torch.tensor(0.0, dtype=torch.float64, device=X_init.device)

LOW_HV_IMPROVEMENT_THRESH = 1e-5
HIGH_VARIANCE_THRESH = 0.6
MODERATE_VARIANCE_THRESH = 0.3
LOW_VARIANCE_THRESH = 0.1
DYNAMIC_EARLY_EXPLORE_ITERATIONS = None
MID_ITER_MAX_EMA = BASE_ITERATIONS * 0.70 # Based on BASE_ITERATIONS
stall_counter = 0
N_STALL_THRESH = 8
FORCED_EXPLORATION_DURATION = 10
forced_exploration_iters_left = 0
STALL_HV_IMPROVEMENT_TOLERANCE = 1e-5
LAST_N_ITER_FORCED_BETA_2_0_AND_EI = 50 # This will be used with effective_total_iterations_for_late_stage
beta_alternation_block_length = 10
forced_stall_weight_cycle = [
    torch.tensor([0.8,0.1,0.1], device=X_init.device), torch.tensor([0.1,0.8,0.1], device=X_init.device),
    torch.tensor([0.1,0.1,0.8], device=X_init.device), torch.tensor([0.4,0.4,0.2], device=X_init.device),
    torch.tensor([0.4,0.2,0.4], device=X_init.device), torch.tensor([0.2,0.4,0.4], device=X_init.device)
]
forced_stall_weight_idx = 0
hypervolume_history = []
selection_methods = []
evaluated_candidate_indices = set()
train_x_scaled = X_init_scaled.clone()
train_y_standardized = train_y_standardized.clone()
hv_ref_point_raw = torch.tensor([1500.0, 1500.0, 1.0], dtype=torch.float64)
negated_ref_point_for_hv_calc = -hv_ref_point_raw
print(f"\nUsing FIXED Hypervolume reference point (raw objectives): {hv_ref_point_raw.tolist()}")
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
            print(f"Warning: Initial hypervolume calculation failed: {e}. Setting initial HV to 0.")
print(f"Initial Hypervolume: {initial_hv:.6f}, Initial Pareto Size: {initial_pareto_raw_size}")
hypervolume_history.append(initial_hv)
previous_pareto_front_size = initial_pareto_raw_size
if len(hypervolume_history) > 1: # Should be true if initial_hv was appended.
    initial_observed_hv_improvement = hypervolume_history[-1] - hypervolume_history[-2] # Will be 0 if only initial_hv
    smooth_hv_improvement_for_beta = torch.tensor(initial_observed_hv_improvement, device=X_init.device)
else: # Only initial_hv exists, so improvement from a "prior state" is 0
    smooth_hv_improvement_for_beta = torch.tensor(0.0, device=X_init.device)

script_version_name = "LCB_AdaptiveBeta_V12_SimpleStop_CorrectedLateStage"
print(f"\nStarting Bayesian Optimization: Base {BASE_ITERATIONS} iters, then simple stopping up to {MAX_POSSIBLE_ITERATIONS_CAP} ({script_version_name}).")
print(f"  After base iters, will stop if raw HV improv. <= {MIN_HV_IMPROVEMENT_FOR_CONTINUATION:.1e} for {PATIENCE_FOR_NO_HV_IMPROVEMENT_AFTER_BASE} consecutive iters.")
print(f"  Initial exploration phase duration will be determined dynamically based on N_init={num_initial_samples}, initial GP variance, and base budget of {BASE_ITERATIONS} iters.")

start_bo_time = time.monotonic()
actual_iterations_completed = 0

# --- BO Loop ---
for i_iter_loop in range(MAX_POSSIBLE_ITERATIONS_CAP):
    iter_display = i_iter_loop + 1
    no_hv_impr_streak_display = 0
    if i_iter_loop >= BASE_ITERATIONS:
        no_hv_impr_streak_display = iters_with_no_significant_hv_improvement_after_base
    
    print(f"\nIteration {iter_display}/{MAX_POSSIBLE_ITERATIONS_CAP} (Base: {BASE_ITERATIONS}, NoHVImprStreakPostBase: {no_hv_impr_streak_display}/{PATIENCE_FOR_NO_HV_IMPROVEMENT_AFTER_BASE})")

    print("  Training Independent GPs...")
    try:
        model = train_independent_gps(train_x_scaled, train_y_standardized)
        model.eval()
    except Exception as e:
        print(f"  Skipping iteration {iter_display} due to GP training error: {e}")
        hypervolume_history.append(hypervolume_history[-1] if hypervolume_history else 0.0)
        selection_methods.append('Error')
        continue

    print("   Using Constrained Scalarized LCB for candidate selection.")
    available_candidate_indices_orig = [idx for idx in range(len(X_candidates)) if idx not in evaluated_candidate_indices]
    if not available_candidate_indices_orig:
        print("   No unevaluated candidates left. Terminating early.")
        break
    unevaluated_indices_tensor = torch.tensor(available_candidate_indices_orig, dtype=torch.long, device=X_init.device)
    X_candidates_scaled_subset = X_candidates_scaled[unevaluated_indices_tensor]
    if X_candidates_scaled_subset.shape[0] == 0:
        print("   No candidates left for LCB selection. Terminating early.")
        break

    with torch.no_grad():
        posterior = model.posterior(X_candidates_scaled_subset)
        means_standardized = posterior.mean
        variances_standardized = posterior.variance
    stds_standardized = (variances_standardized.clamp(min=1e-9)).sqrt()

    objective_variances_std = variances_standardized[:, obj_indices]
    current_observed_avg_variance_k = torch.mean(torch.mean(objective_variances_std, dim=1)) if objective_variances_std.numel() > 0 else torch.tensor(0.0, device=X_init.device)

    observed_hv_improvement_for_beta_ema_calc = 0.0
    if len(hypervolume_history) > 1 :
        observed_hv_improvement_for_beta_ema_calc = hypervolume_history[-1] - hypervolume_history[-2]

    if i_iter_loop == 0:
        smooth_avg_variance = current_observed_avg_variance_k.clone()
        # smooth_hv_improvement_for_beta initialized before loop for i_iter_loop=0
        initial_gp_variance_for_schedule = current_observed_avg_variance_k.item()
        if DYNAMIC_EARLY_EXPLORE_ITERATIONS is None:
            DYNAMIC_EARLY_EXPLORE_ITERATIONS = calculate_dynamic_early_exploration_duration(
                BASE_ITERATIONS, num_initial_samples, initial_gp_variance_for_schedule
            )
            print(f"    Dynamic early exploration phase set to the first {DYNAMIC_EARLY_EXPLORE_ITERATIONS} iterations (of base {BASE_ITERATIONS}).")
    else:
        smooth_avg_variance = (1.0 - alpha_variance) * smooth_avg_variance + alpha_variance * current_observed_avg_variance_k
        smooth_hv_improvement_for_beta = (1.0 - alpha_hv_improvement) * smooth_hv_improvement_for_beta + \
                                         alpha_hv_improvement * torch.tensor(observed_hv_improvement_for_beta_ema_calc, device=X_init.device)

    print(f"    EMA Metrics (for Beta choice): SmoothAvgVar={smooth_avg_variance.item():.4f}, SmoothHVImprove(prior)={smooth_hv_improvement_for_beta.item():.6f}")
    print(f"    Stall counter: {stall_counter}, Forced exploration iters left: {forced_exploration_iters_left}")

    mean_obj1_std = means_standardized[:, obj_indices[0]]
    std_obj1_std = stds_standardized[:, obj_indices[0]]
    mean_obj2_std = means_standardized[:, obj_indices[1]]
    std_obj2_std = stds_standardized[:, obj_indices[1]]
    mean_obj3_std = means_standardized[:, obj_indices[2]]
    std_obj3_std = stds_standardized[:, obj_indices[2]]
    mean_constr_std = means_standardized[:, constraint_index_in_y]
    std_constr_std = stds_standardized[:, constraint_index_in_y]

    current_beta_lcb: float
    w1: float
    w2: float
    w3: float
    rule_reason: str

    # *** MODIFIED is_late_stage_iteration logic ***
    if i_iter_loop < BASE_ITERATIONS:
        effective_total_iterations_for_late_stage = BASE_ITERATIONS
    else:
        effective_total_iterations_for_late_stage = MAX_POSSIBLE_ITERATIONS_CAP
    is_late_stage_iteration = (i_iter_loop >= (effective_total_iterations_for_late_stage - LAST_N_ITER_FORCED_BETA_2_0_AND_EI))
    # *** END MODIFICATION ***

    use_dynamic_ei_weights_for_this_iteration = True

    if forced_exploration_iters_left > 0:
        if is_late_stage_iteration:
            current_beta_lcb = 2.0
            _rule_reason_beta_part_for_print = f"Forced stall exploration ({forced_exploration_iters_left} iters left, Beta 2.0 (late stage))"
        else:
            current_beta_lcb = beta_states[5].item() # Beta 3.0 for non-late-stage forced exploration
            current_forced_weights = forced_stall_weight_cycle[forced_stall_weight_idx % len(forced_stall_weight_cycle)]
            w1 = current_forced_weights[0].item()
            w2 = current_forced_weights[1].item()
            w3 = current_forced_weights[2].item()
            rule_reason = f"Forced stall exploration ({forced_exploration_iters_left} iters left, Beta 3.0, Forced Stall W=[{w1:.1f},{w2:.1f},{w3:.1f}])"
            forced_stall_weight_idx += 1
            use_dynamic_ei_weights_for_this_iteration = False
    else: # Not in forced exploration
        if DYNAMIC_EARLY_EXPLORE_ITERATIONS is None:
            DYNAMIC_EARLY_EXPLORE_ITERATIONS = int(BASE_ITERATIONS * 0.30)
            print(f"    Warning: DYNAMIC_EARLY_EXPLORE_ITERATIONS fallback: {DYNAMIC_EARLY_EXPLORE_ITERATIONS}.")
        if i_iter_loop < DYNAMIC_EARLY_EXPLORE_ITERATIONS:
            current_beta_lcb = beta_states[5].item() # Beta 3.0
            _rule_reason_beta_part_for_print = f"Dynamic early explore (iter {i_iter_loop+1}/{DYNAMIC_EARLY_EXPLORE_ITERATIONS}, Beta 3.0)"
        elif smooth_hv_improvement_for_beta < LOW_HV_IMPROVEMENT_THRESH: # Using smoothed HV for beta choice
            if smooth_avg_variance > HIGH_VARIANCE_THRESH:
                current_beta_lcb = beta_states[5].item() # Beta 3.0
                _rule_reason_beta_part_for_print = "EMA Stagnant (prior HV), high variance (Beta 3.0)"
            elif smooth_avg_variance > MODERATE_VARIANCE_THRESH:
                current_beta_lcb = beta_states[4].item() # Beta 2.5
                _rule_reason_beta_part_for_print = "EMA Stagnant (prior HV), moderate variance (Beta 2.5)"
            else: # Low variance
                current_beta_lcb = beta_states[1].item() # Beta 1.0
                _rule_reason_beta_part_for_print = "EMA Stagnant (prior HV), low var -> precise exploitation (Beta 1.0)"
        elif smooth_avg_variance > HIGH_VARIANCE_THRESH: # Decent improvement (implied), high variance
            current_beta_lcb = beta_states[3].item() # Beta 2.0
            _rule_reason_beta_part_for_print = "Decent improv (prior HV implied), high variance (Beta 2.0)"
        elif i_iter_loop > MID_ITER_MAX_EMA: # MID_ITER_MAX_EMA is based on BASE_ITERATIONS
            if smooth_avg_variance < LOW_VARIANCE_THRESH:
                current_beta_lcb = beta_states[0].item() # Beta 0.5
                _rule_reason_beta_part_for_print = f"Past initial mid-stage ({MID_ITER_MAX_EMA:.0f}), low var (Beta 0.5)"
            elif smooth_avg_variance < MODERATE_VARIANCE_THRESH:
                current_beta_lcb = beta_states[1].item() # Beta 1.0
                _rule_reason_beta_part_for_print = f"Past initial mid-stage ({MID_ITER_MAX_EMA:.0f}), mod-low var (Beta 1.0)"
            else: # Moderate to high variance
                current_beta_lcb = beta_states[2].item() # Beta 1.5
                _rule_reason_beta_part_for_print = f"Past initial mid-stage ({MID_ITER_MAX_EMA:.0f}), default balanced (Beta 1.5)"
        else: # Mid-stage (relative to BASE_ITERATIONS)
            block_phase_alternating = (i_iter_loop // beta_alternation_block_length) % 2
            if block_phase_alternating == 0: # Explore more
                current_beta_lcb = beta_states[3].item() # Beta 2.0
            else: # Balance/Exploit
                current_beta_lcb = beta_states[2].item() # Beta 1.5
            _rule_reason_beta_part_for_print = f"Mid-stage (initial plan), Alternating Beta ({current_beta_lcb:.1f})"

    if use_dynamic_ei_weights_for_this_iteration:
        current_train_y_raw_neg_inv = torch.tensor(y_scaler.inverse_transform(train_y_standardized.numpy()), dtype=torch.float64)
        current_train_constr_raw = current_train_y_raw_neg_inv[:, constraint_index_in_y]
        current_feas_mask_train = current_train_constr_raw <= constraint_threshold
        default_tau_std = torch.tensor(-3.0, device=train_x_scaled.device, dtype=torch.float64)
        tau_obj1_std_neg = default_tau_std.clone()
        tau_obj2_std_neg = default_tau_std.clone()
        tau_obj3_std_neg = default_tau_std.clone()
        if torch.any(current_feas_mask_train):
            feas_train_y_std = train_y_standardized[current_feas_mask_train]
            if feas_train_y_std.shape[0] > 0:
                tau_obj1_std_neg = torch.max(feas_train_y_std[:, obj_indices[0]])
                tau_obj2_std_neg = torch.max(feas_train_y_std[:, obj_indices[1]])
                tau_obj3_std_neg = torch.max(feas_train_y_std[:, obj_indices[2]])
        eps_sig = 1e-9
        norm_dist = torch.distributions.Normal(torch.tensor(0.0, device=train_x_scaled.device), torch.tensor(1.0, device=train_x_scaled.device))
        mu1, sig1 = mean_obj1_std, std_obj1_std + eps_sig
        Z1 = (mu1 - tau_obj1_std_neg) / sig1
        ei1_vals = torch.clamp((mu1 - tau_obj1_std_neg) * norm_dist.cdf(Z1) + sig1 * norm_dist.log_prob(Z1).exp(), min=0.0)
        mu2, sig2 = mean_obj2_std, std_obj2_std + eps_sig
        Z2 = (mu2 - tau_obj2_std_neg) / sig2
        ei2_vals = torch.clamp((mu2 - tau_obj2_std_neg) * norm_dist.cdf(Z2) + sig2 * norm_dist.log_prob(Z2).exp(), min=0.0)
        mu3, sig3 = mean_obj3_std, std_obj3_std + eps_sig
        Z3 = (mu3 - tau_obj3_std_neg) / sig3
        ei3_vals = torch.clamp((mu3 - tau_obj3_std_neg) * norm_dist.cdf(Z3) + sig3 * norm_dist.log_prob(Z3).exp(), min=0.0)
        S1 = torch.max(ei1_vals) if ei1_vals.numel() > 0 else torch.tensor(0.0)
        S2 = torch.max(ei2_vals) if ei2_vals.numel() > 0 else torch.tensor(0.0)
        S3 = torch.max(ei3_vals) if ei3_vals.numel() > 0 else torch.tensor(0.0)
        eps_score = 1e-7
        s1e = S1 + eps_score
        s2e = S2 + eps_score
        s3e = S3 + eps_score
        sum_Se = s1e + s2e + s3e
        if sum_Se <= (n_objectives * eps_score):
            w1 = 1/3.
            w2 = 1/3.
            w3 = 1/3.
        else:
            w1 = s1e / sum_Se
            w2 = s2e / sum_Se
            w3 = s3e / sum_Se
        if forced_exploration_iters_left > 0 and is_late_stage_iteration: # During forced stall in late stage
            rule_reason = f"{_rule_reason_beta_part_for_print}, Dynamic EI W=[{w1:.3f},{w2:.3f},{w3:.3f}])"
        else: # Normal adaptive beta or early/mid stage
            rule_reason = f"Adaptive Beta={current_beta_lcb:.1f} (Rule: {_rule_reason_beta_part_for_print}, Dynamic EI W=[{w1:.3f},{w2:.3f},{w3:.3f}])"
    print(f"  Using LCB parameters for objectives: {rule_reason}")

    beta_constraint = 1.0
    constraint_lcb_values = mean_constr_std - beta_constraint * std_constr_std
    predicted_feasibility_mask = (constraint_lcb_values <= constraint_threshold_std)
    penalty_value = -1e9
    if not torch.any(predicted_feasibility_mask):
        print("    Warning: All candidates predicted infeasible by LCB. Selecting best LCB among them (ignoring constraint).")
        eff_lcb_std = (w1 * (mean_obj1_std - current_beta_lcb * std_obj1_std) +
                       w2 * (mean_obj2_std - current_beta_lcb * std_obj2_std) +
                       w3 * (mean_obj3_std - current_beta_lcb * std_obj3_std))
        best_lcb_idx_subset = torch.argmax(eff_lcb_std)
    else:
        eff_lcb_std = torch.full_like(mean_obj1_std, penalty_value, dtype=torch.float64)
        feas_idx = torch.where(predicted_feasibility_mask)[0]
        if feas_idx.numel() > 0:
            lcb_feas = (w1 * (mean_obj1_std[feas_idx] - current_beta_lcb * std_obj1_std[feas_idx]) +
                        w2 * (mean_obj2_std[feas_idx] - current_beta_lcb * std_obj2_std[feas_idx]) +
                        w3 * (mean_obj3_std[feas_idx] - current_beta_lcb * std_obj3_std[feas_idx]))
            eff_lcb_std[feas_idx] = lcb_feas
        best_lcb_idx_subset = torch.argmax(eff_lcb_std)

    sel_cand_orig_idx = unevaluated_indices_tensor[best_lcb_idx_subset].item()
    _, _, next_y_obj1_raw, next_y_obj2_raw, next_y_obj3_raw, next_y_constr_raw = find_closest_candidate(
        X_candidates, Y_candidates, objective_index_1, objective_index_2, objective_index_3, constraint_index, sel_cand_orig_idx
    )
    closest_cand_X_scaled = X_candidates_scaled[sel_cand_orig_idx].unsqueeze(0)
    print(f"   Selected candidate (Original Index: {sel_cand_orig_idx}) via constrained LCB.")
    selection_methods.append('LCB')
    print(f"   Raw Objectives: {objective_names[0]}={-next_y_obj1_raw.item():.4f}, {objective_names[1]}={-next_y_obj2_raw.item():.4f}, {objective_names[2]}={-next_y_obj3_raw.item():.6f}")
    print(f"   Constraint ({constraint_target_name}): {next_y_constr_raw.item():.4f}")

    next_y_comb_raw_model = torch.cat([next_y_obj1_raw, next_y_obj2_raw, next_y_obj3_raw, next_y_constr_raw], dim=1)
    next_y_std = torch.tensor(y_scaler.transform(next_y_comb_raw_model.numpy()), dtype=torch.float64)
    train_x_scaled = torch.cat([train_x_scaled, closest_cand_X_scaled], dim=0)
    train_y_standardized = torch.cat([train_y_standardized, next_y_std], dim=0)
    evaluated_candidate_indices.add(sel_cand_orig_idx)

    all_y_raw_std_inv = torch.tensor(y_scaler.inverse_transform(train_y_standardized.numpy()), dtype=torch.float64)
    all_obj_raw_min = -all_y_raw_std_inv[:, obj_indices]
    all_constr_raw = all_y_raw_std_inv[:, constraint_index_in_y]
    feas_mask = all_constr_raw <= constraint_threshold
    curr_obj_feas_raw = all_obj_raw_min[feas_mask]
    current_hv = 0.0
    pareto_raw_min = torch.empty((0, n_objectives), dtype=curr_obj_feas_raw.dtype)
    curr_pareto_size = 0
    if curr_obj_feas_raw.shape[0] > 0:
        non_dom_mask = simple_is_non_dominated(curr_obj_feas_raw)
        pareto_raw_min = curr_obj_feas_raw[non_dom_mask]
        curr_pareto_size = pareto_raw_min.shape[0]
        if pareto_raw_min.shape[0] > 0:
            try:
                hv_calc = Hypervolume(ref_point=negated_ref_point_for_hv_calc)
                current_hv = hv_calc.compute(-pareto_raw_min)
            except Exception as e:
                print(f"    Warning: HV calculation error: {e}")
                current_hv = hypervolume_history[-1] if hypervolume_history else 0.0
    hypervolume_history.append(current_hv)

    observed_hv_improvement_this_iter = current_hv - (hypervolume_history[-2] if len(hypervolume_history) > 1 else initial_hv)
    print(f"   Current Hypervolume: {current_hv:.6f}, Raw HV Improv this iter: {observed_hv_improvement_this_iter:.6f}")

    # --- Simple Stopping Criterion (Active *after* BASE_ITERATIONS) ---
    if i_iter_loop >= BASE_ITERATIONS:
        if observed_hv_improvement_this_iter > MIN_HV_IMPROVEMENT_FOR_CONTINUATION:
            iters_with_no_significant_hv_improvement_after_base = 0
        else:
            iters_with_no_significant_hv_improvement_after_base += 1

        if iters_with_no_significant_hv_improvement_after_base >= PATIENCE_FOR_NO_HV_IMPROVEMENT_AFTER_BASE:
            print(f"    STOPPING: No raw HV improvement > {MIN_HV_IMPROVEMENT_FOR_CONTINUATION:.1e} for {PATIENCE_FOR_NO_HV_IMPROVEMENT_AFTER_BASE} consecutive iterations after base run.")
            actual_iterations_completed = i_iter_loop + 1
            break # Exit BO loop

    # --- Stall Counter Logic ---
    pareto_size_changed_this_iter = (curr_pareto_size != previous_pareto_front_size)
    hv_improved_sig = (current_hv > (hypervolume_history[-2] if len(hypervolume_history) > 1 else initial_hv) + STALL_HV_IMPROVEMENT_TOLERANCE)
    if hv_improved_sig or pareto_size_changed_this_iter:
        if stall_counter > 0:
            reasons_for_reset = []
            if hv_improved_sig:
                reasons_for_reset.append("HV improvement")
            if pareto_size_changed_this_iter:
                reasons_for_reset.append(f"Pareto front size change ({previous_pareto_front_size} -> {curr_pareto_size})")
            print(f"    Stall counter reset from {stall_counter} to 0. Reasons: {', '.join(reasons_for_reset)}")
        stall_counter = 0
    else:
        if i_iter_loop > 0:
            stall_counter += 1
            print(f"    Stall counter incremented to {stall_counter}.")
    previous_pareto_front_size = curr_pareto_size

    if stall_counter >= N_STALL_THRESH and forced_exploration_iters_left == 0:
        print(f"    STALLED for {stall_counter} iterations. Forcing exploration for {FORCED_EXPLORATION_DURATION} iters.")
        forced_exploration_iters_left = FORCED_EXPLORATION_DURATION
        forced_stall_weight_idx = 0
    if rule_reason.startswith("Forced stall exploration"):
         forced_exploration_iters_left -=1

    print(f"   Pareto Front Size: {curr_pareto_size}")
    print(f"   Total unique candidates evaluated: {len(evaluated_candidate_indices)}")
    print(f"   Total points in training data: {train_x_scaled.shape[0]}")
    actual_iterations_completed = i_iter_loop + 1
# --- End of BO Loop ---

end_bo_time = time.monotonic()
bo_duration_seconds = end_bo_time - start_bo_time
print("\nOptimization finished.")
print(f"Ran for {actual_iterations_completed} iterations.")
if actual_iterations_completed < BASE_ITERATIONS:
    print(f"Stopped before completing BASE_ITERATIONS (likely due to no candidates).")
elif actual_iterations_completed < MAX_POSSIBLE_ITERATIONS_CAP and iters_with_no_significant_hv_improvement_after_base >= PATIENCE_FOR_NO_HV_IMPROVEMENT_AFTER_BASE:
    print(f"Stopped after BASE_ITERATIONS due to lack of significant HV improvement for {PATIENCE_FOR_NO_HV_IMPROVEMENT_AFTER_BASE} iterations.")
elif actual_iterations_completed == MAX_POSSIBLE_ITERATIONS_CAP:
    print(f"Completed all {MAX_POSSIBLE_ITERATIONS_CAP} iterations (max cap).")
else: # Should cover cases where it stopped due to no candidates after base but before new stopping criterion hit
    print(f"Completed {actual_iterations_completed} iterations (stopped for other reasons like no candidates after base).")

print(f"Total BO loop duration: {bo_duration_seconds:.2f} seconds ({bo_duration_seconds/60:.2f} minutes)")

# --- Results, Plotting etc. ---
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
        print(pd.DataFrame(final_bo_pareto_points_raw_minimization.numpy(), columns=objective_names).to_string(max_rows=20))

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
    print(pd.DataFrame(true_pareto_front_raw_minimization.numpy(), columns=objective_names).to_string(max_rows=20))
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

print("\n--- Checking Overlap Between BO Front and True Front ---")
if final_bo_pareto_points_raw_minimization.shape[0] > 0 and true_pareto_front_raw_minimization.shape[0] > 0:
    true_points_set_for_match = set(tuple(map(lambda x: round(x, 6), point.cpu().tolist())) for point in true_pareto_front_raw_minimization)
    bo_points_for_match = [tuple(map(lambda x: round(x, 6), point.cpu().tolist())) for point in final_bo_pareto_points_raw_minimization]
    match_count = 0
    for bo_pt_tuple in bo_points_for_match:
        if bo_pt_tuple in true_points_set_for_match:
            match_count += 1
    print(f"  - BO found {final_bo_pareto_points_raw_minimization.shape[0]} Pareto points.")
    print(f"  - True Pareto Front contains {true_pareto_front_raw_minimization.shape[0]} points.")
    print(f"  - {match_count} BO Pareto points (rounded to 6dp) match points on the True Pareto Front.")
    if match_count < true_pareto_front_raw_minimization.shape[0] and true_pareto_front_raw_minimization.shape[0] > 0 :
        print("\n--- Missed True Pareto Points ---")
        bo_points_tuples_set = set(bo_points_for_match)
        missed_true_points = []
        for true_pt in true_pareto_front_raw_minimization:
            true_pt_tuple_rounded = tuple(map(lambda x: round(x.item(), 6), true_pt))
            if true_pt_tuple_rounded not in bo_points_tuples_set:
                missed_true_points.append(true_pt.numpy())
        if missed_true_points:
            print(pd.DataFrame(np.array(missed_true_points), columns=objective_names).to_string(max_rows=20))
        elif not missed_true_points and match_count == true_pareto_front_raw_minimization.shape[0]:
             print("  All true Pareto points were found by BO.")
        else:
            print("  No missed points identified or all true Pareto points were found.")
elif final_bo_pareto_points_raw_minimization.shape[0] == 0:
    print("\nCannot check overlap: Final BO Pareto Front is empty.")
else:
    print("\nCannot check overlap: True Pareto Front is empty (or BO front is empty).")

plot_suffix = f"{script_version_name}_base{BASE_ITERATIONS}_actual{actual_iterations_completed}"
plt.figure(figsize=(10, 6))
plt.plot(range(len(hypervolume_history)), hypervolume_history, marker='o', linestyle='-', label=f'BO HV ({plot_suffix})')
if true_pareto_front_raw_minimization.shape[0] > 0 and true_hv > 1e-9:
    plt.axhline(y=true_hv, color='r', linestyle='--', label=f'True Max HV ({true_hv:.4f})')
plt.xlabel("Iteration (0 = Initial)")
plt.ylabel("Hypervolume")
plt.title(f"HV Convergence ({plot_suffix})")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(f'hypervolume_convergence_{plot_suffix}.png')
plt.close()

plt.figure(figsize=(12, 6))
if len(selection_methods) > 0:
    bo_iterations_plot = list(range(1, len(selection_methods) + 1))
    bo_hypervolumes_plot = hypervolume_history[1:len(selection_methods) + 1]
    if len(bo_iterations_plot) == len(bo_hypervolumes_plot) and len(bo_hypervolumes_plot) > 0:
        plt.plot(bo_iterations_plot, bo_hypervolumes_plot, color='darkgrey', linestyle='-', zorder=1, alpha=0.7)
        color_map = {'LCB': '#377EB8', 'Error': 'grey'}
        plot_colors = [color_map.get(method, 'black') for method in selection_methods]
        plt.scatter(bo_iterations_plot, bo_hypervolumes_plot, c=plot_colors, marker='o', s=50, zorder=2, edgecolors='grey', alpha=0.9)
if true_pareto_front_raw_minimization.shape[0] > 0 and true_hv > 1e-9:
    plt.axhline(y=true_hv, color='green', linestyle='--', label=f'True Max HV ({true_hv:.4f})')
legend_elements = [Line2D([0], [0], marker='o', color='w', label='LCB', markerfacecolor='#377EB8', markersize=8)]
if 'Error' in selection_methods:
    legend_elements.append(Line2D([0], [0], marker='o', color='w', label='GP Error', markerfacecolor='grey', markersize=8))
if true_pareto_front_raw_minimization.shape[0] > 0 and true_hv > 1e-9:
    legend_elements.append(Line2D([0],[0], color='green', linestyle='--', label=f'True Max HV ({true_hv:.4f})'))
plt.legend(handles=legend_elements)
plt.xlabel("BO Iteration")
plt.ylabel("Current HV")
plt.title(f"BO HV Progression ({plot_suffix})")
plt.grid(True)
plt.xlim(left=0, right=max(1, len(selection_methods) if len(selection_methods) > 0 else 1 ))
plt.tight_layout()
plt.savefig(f'selection_method_plot_{plot_suffix}.png')
plt.close()

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
if final_objectives_raw_minimization.shape[0] > 0:
    obj1_all_min = final_objectives_raw_minimization[:,0].numpy()
    obj2_all_min = final_objectives_raw_minimization[:,1].numpy()
    obj3_all_min = final_objectives_raw_minimization[:,2].numpy()
    constr_all_final = final_constraints_raw.numpy()
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
    print("\nNo points from BO iterations to plot in 3D objective space.")

print("\nFinal training data size (scaled features):", train_x_scaled.shape)
print(f"Total points in hypervolume_history plot: {len(hypervolume_history)}")
print(f"Selection methods recorded: {selection_methods.count('LCB')} LCB, {selection_methods.count('Error')} Error")
print(f"Script finished. ({script_version_name}) Base: {BASE_ITERATIONS}, MaxCap: {MAX_POSSIBLE_ITERATIONS_CAP}. Simple post-base stopping active.")
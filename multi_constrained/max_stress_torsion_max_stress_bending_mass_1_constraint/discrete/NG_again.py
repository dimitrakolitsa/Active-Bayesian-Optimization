import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import warnings
import time
import os

# --- NEW IMPORTS FOR NGBOOST ---
from ngboost import NGBRegressor
from ngboost.distns import Normal # Common choice for regression distribution
from ngboost.scores import LogScore # Scoring rule for Normal distribution
from sklearn.tree import DecisionTreeRegressor # Default base learner for NGBoost
# --- END NEW IMPORTS ---

from typing import Optional, Dict, Any, List, Tuple

# BoTorch
from botorch.models.model import Model
from botorch.posteriors import GPyTorchPosterior
from botorch.utils.multi_objective.hypervolume import Hypervolume

# GPyTorch
from gpytorch.distributions import MultivariateNormal

# sklearn
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# from sklearn.utils import resample # Not needed if not ensembling NGBoost

# Output & Plotting
from botorch.exceptions import InputDataWarning, BadInitialCandidatesWarning
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D

warnings.filterwarnings('ignore', category=InputDataWarning)
warnings.filterwarnings('ignore', category=BadInitialCandidatesWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# seed pytorch and numpy
torch.manual_seed(42)
np.random.seed(42)
torch.set_default_dtype(torch.float64)


# --- Custom NGBoost Model Definition ---
class CustomNGBoostModel(Model):
    """
    A custom NGBoost model for BoTorch.
    Fits an NGBoost regressor for each output dimension.
    Mean and variance are obtained directly from NGBoost's predicted distribution.
    """
    def __init__(self,
                 train_X: torch.Tensor,
                 train_Y: torch.Tensor,
                 ngb_params: Optional[Dict[str, Any]] = None):
        super().__init__()

        if train_X.ndim == 1:
            train_X = train_X.unsqueeze(-1)
        if train_Y.ndim == 1:
            train_Y = train_Y.unsqueeze(-1)

        self.train_inputs = (train_X,)
        self.train_targets = train_Y
        self._num_outputs = train_Y.shape[-1]

        train_X_np = train_X.cpu().detach().numpy()
        train_Y_np = train_Y.cpu().detach().numpy()

        self.models = []  # List of fitted NGBoost models (one per output)

        # Default parameters for NGBoost
        # Users can override these by passing ngb_params
        default_params = {
            'Dist': Normal,
            'Score': LogScore, # Use LogScore for Normal distribution
            'n_estimators': 100,
            'learning_rate': 0.05,
            'Base': DecisionTreeRegressor(max_depth=3), # Default base learner
            'verbose': False # Set to True for NGBoost training verbosity
        }
        current_ngb_params = ngb_params or {}
        final_ngb_params = {**default_params, **current_ngb_params}

        print(f"  Fitting NGBoost models with parameters: {final_ngb_params}")

        for i_output in range(self._num_outputs):
            model = NGBRegressor(**final_ngb_params)
            try:
                # NGBoost expects 1D Y target
                model.fit(train_X_np, train_Y_np[:, i_output])
            except Exception as e:
                print(f"ERROR: NGBoost fitting failed for output {i_output}: {e}")
                # Option 1: Raise error to stop execution
                raise RuntimeError(f"NGBoost fitting failed for output {i_output}: {e}") from e
                # Option 2: Store None and handle in posterior (less safe for BO)
                # self.models.append(None)
                # continue
            self.models.append(model)
        
        self._is_trained = True # Internal flag

    @property
    def num_outputs(self) -> int:
        return self._num_outputs

    def train(self, mode: bool = True):
        # This model is "trained" at instantiation.
        if mode:
            self._is_trained = False
        else:
            self._is_trained = True
        super().train(mode)

    def eval(self):
        self._is_trained = True
        super().eval()

    def posterior(self, X: torch.Tensor) -> GPyTorchPosterior:
        if not self._is_trained:
             warnings.warn("CustomNGBoostModel is not marked as trained. Ensure it's fitted.", UserWarning)

        # X is `batch_shape x q x d`
        # We need to reshape X for NGBoost: (N, d) where N = product of batch_shape * q
        original_X_shape = X.shape
        d = original_X_shape[-1]
        X_reshaped_for_ngb = X.reshape(-1, d)
        X_np = X_reshaped_for_ngb.cpu().detach().numpy()
        
        num_total_points_to_predict = X_reshaped_for_ngb.shape[0]

        means_list = []
        variances_list = []

        for i_output in range(self._num_outputs):
            model = self.models[i_output]
            if model is None: # Should not happen if we raise error in __init__
                 raise RuntimeError(f"NGBoost model for output {i_output} is None.")

            try:
                pred_dist = model.pred_dist(X_np) # Returns a distribution object
                # For Normal distribution from NGBoost:
                # .params['loc'] is the mean
                # .params['scale'] is the standard deviation
                current_means_np = pred_dist.params['loc']
                current_std_devs_np = pred_dist.params['scale']
                current_variances_np = current_std_devs_np**2
            except Exception as e:
                print(f"Warning: NGBoost prediction failed for output {i_output} on X shape {X_np.shape}: {e}. Returning defaults.")
                current_means_np = np.zeros(num_total_points_to_predict)
                current_variances_np = np.ones(num_total_points_to_predict) * 1e6 # High variance

            means_list.append(torch.from_numpy(current_means_np).to(X))
            variances_list.append(torch.from_numpy(current_variances_np).to(X))

        # Stack along the output dimension
        means_stacked_flat = torch.stack(means_list, dim=-1)     # Shape: (N, num_outputs)
        variances_stacked_flat = torch.stack(variances_list, dim=-1) # Shape: (N, num_outputs)

        # Reshape back to BoTorch's expected shape: (*batch_shape, q, num_outputs)
        # original_X_shape[:-1] gives (*batch_shape, q)
        target_shape_for_output = (*original_X_shape[:-1], self._num_outputs)
        means = means_stacked_flat.reshape(target_shape_for_output)
        variances = variances_stacked_flat.reshape(target_shape_for_output)
        
        variances = variances.clamp_min(1e-6) # Crucial for numerical stability

        mvn = MultivariateNormal(mean=means, covariance_matrix=torch.diag_embed(variances))
        return GPyTorchPosterior(mvn)

# --- Simple Pairwise Non-Dominated Check (for minimization) ---
def simple_is_non_dominated(points):
    n_points = points.shape[0]
    if n_points == 0:
        return torch.tensor([], dtype=torch.bool)
    is_nd = torch.ones(n_points, dtype=torch.bool, device=points.device)
    for i_point_loop in range(n_points):
        if not is_nd[i_point_loop]: continue
        for j_point_loop in range(n_points):
            if i_point_loop == j_point_loop: continue
            if torch.all(points[j_point_loop] <= points[i_point_loop]) and torch.any(points[j_point_loop] < points[i_point_loop]):
                is_nd[i_point_loop] = False
                break
    return is_nd

# --- Function to dynamically determine early exploration duration ---
def calculate_dynamic_early_exploration_duration(
    n_total_iterations, #budget
    num_initial_samples,
    initial_model_avg_variance, # Changed from initial_gp_avg_variance
    base_explore_fraction=0.40, #0.3
    min_explore_iters_abs=10, #minimum number of iterations for early exploration
    max_explore_fraction=0.50, #maximum fraction of iterations for early exploration

    baseline_initial_samples=50,         #baseline number of initial samples
    iters_change_per_10_init_samples=-5, #for every 10 samples MORE than baseline, REDUCE iters by 5
                                         #for every 10 samples LESS than baseline, INCREASE iters by 5

    low_variance_threshold=0.2,         # average initial variance below this doesn't add much exploration time
    high_variance_threshold=0.8,        # average initial variance above this gives max variance bonus
    max_additional_iters_from_variance_config=None):

    if max_additional_iters_from_variance_config is None:
        max_iters_from_variance = int(n_total_iterations * 0.20)
    else:
        max_iters_from_variance = max_additional_iters_from_variance_config

    calculated_iters = float(n_total_iterations * base_explore_fraction)

    sample_diff = num_initial_samples - baseline_initial_samples
    sample_adjustment = (sample_diff/10.0) * iters_change_per_10_init_samples
    calculated_iters += sample_adjustment
    print(f"    Dynamic Early Explore: Base iters ({base_explore_fraction*100:.0f}% of total)={n_total_iterations * base_explore_fraction:.1f}, Sample adjustment based on N_init={num_initial_samples} is {sample_adjustment:.1f}")

    if initial_model_avg_variance > low_variance_threshold:
        clamped_variance = min(high_variance_threshold, max(low_variance_threshold, initial_model_avg_variance))

        if (high_variance_threshold - low_variance_threshold) > 1e-6:
            variance_effect_scaled = ((clamped_variance - low_variance_threshold) / (high_variance_threshold - low_variance_threshold))
        else:
            variance_effect_scaled = 1.0 if initial_model_avg_variance > low_variance_threshold else 0.0

        variance_adjustment = variance_effect_scaled * max_iters_from_variance
        calculated_iters += variance_adjustment
        print(f"    Dynamic Early Explore: Initial Model avg var={initial_model_avg_variance:.3f} (scaled effect: {variance_effect_scaled:.2f}), Variance adjustment={variance_adjustment:.1f}")
    else:
        print(f"    Dynamic Early Explore: Initial Model avg var={initial_model_avg_variance:.3f} <= low_thresh={low_variance_threshold}, no positive variance adjustment.")

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
base_path = '../../' # Make sure this path is correct for your setup
try:
    df_init = pd.read_csv(f'{base_path}init/inputs.txt', header=0, sep=',')
    df_init = df_init[new_columns]
    df_all_candidates = pd.read_csv(f'{base_path}all_candidates/inputs.txt', header=0, sep=',')
    df_all_candidates = df_all_candidates[new_columns]
    init_target_files = [
        f"{base_path}init/mass.txt", f"{base_path}init/max_displacement_bending.txt",
        f"{base_path}init/max_displacement_torsion.txt", f"{base_path}init/max_stress_bending.txt",
        f"{base_path}init/max_stress_torsion.txt"
    ]
    all_candidates_target_files = [
        f"{base_path}all_candidates/targets_mass.txt", f"{base_path}all_candidates/targets_max_displacement_bending.txt",
        f"{base_path}all_candidates/targets_max_displacement_torsion.txt", f"{base_path}all_candidates/targets_max_stress_bending.txt",
        f"{base_path}all_candidates/targets_max_stress_torsion.txt"
    ]
    target_column_names = ["mass", "max_displacement_bending", "max_displacement_torsion", "max_stress_bending", "max_stress_torsion"]
    init_target_dfs = [pd.read_csv(file, header=None, sep=',') for file in init_target_files]
    init_df_targets = pd.concat(init_target_dfs, axis=1); init_df_targets.columns = target_column_names
    df_init = pd.concat([df_init, init_df_targets], axis=1)
    all_candidates_target_dfs = [pd.read_csv(file, header=None, sep=',') for file in all_candidates_target_files]
    all_candidates_df_targets = pd.concat(all_candidates_target_dfs, axis=1); all_candidates_df_targets.columns = target_column_names
    df_all_candidates = pd.concat([df_all_candidates, all_candidates_df_targets], axis=1)
except FileNotFoundError as e:
    print(f"Error: Data file not found: {e}. Please check base_path: '{base_path}'. Current working directory: {os.getcwd()}")
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
    -Y_init[:, objective_index_1].unsqueeze(-1), -Y_init[:, objective_index_2].unsqueeze(-1),
    -Y_init[:, objective_index_3].unsqueeze(-1), Y_init[:, constraint_index].unsqueeze(-1)], dim=1)

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
    constraint_threshold_std = 0.0 if constraint_threshold == constraint_mean_model else (float('inf') if constraint_threshold > constraint_mean_model else float('-inf'))
else:
    constraint_threshold_std = (constraint_threshold - constraint_mean_model) / constraint_scale_model
print(f"Raw constraint threshold: {constraint_threshold}, Standardized constraint threshold: {constraint_threshold_std:.4f}")


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
n_iterations = 300 # Adjust as needed
CURRENT_MODEL_TYPE = "NGBoost" # CHANGED FROM "XGBoost"

# Parameters for CustomNGBoostModel
NGB_PARAMS = {
    'n_estimators': 125,       # Number of boosting rounds
    'learning_rate': 0.05,
    'Base': DecisionTreeRegressor(max_depth=3), # Control complexity of base learners
    'minibatch_frac': 0.8,     # Use mini-batches for training
    #'tol': 1e-4,               # Early stopping tolerance if validation set is used
    'verbose': False           # Set True to see NGBoost training progress
}


alpha_variance = 0.2
alpha_hv_improvement = 0.3
beta_states = torch.tensor([0.5, 1.0, 1.5, 2.0, 2.5, 3.0], dtype=torch.float64, device=X_init.device) # Can add 3.5 for more exploration
smooth_avg_variance = torch.tensor(0.0, dtype=torch.float64, device=X_init.device)
smooth_hv_improvement = torch.tensor(0.0, dtype=torch.float64, device=X_init.device)

LOW_HV_IMPROVEMENT_THRESH = 1e-5
HIGH_VARIANCE_THRESH = 0.6
MODERATE_VARIANCE_THRESH = 0.3
LOW_VARIANCE_THRESH = 0.1
DYNAMIC_EARLY_EXPLORE_ITERATIONS = None
MID_ITER_MAX_EMA = n_iterations * 0.70

stall_counter = 0
N_STALL_THRESH = 8
FORCED_EXPLORATION_DURATION = 10
forced_exploration_iters_left = 0
STALL_HV_IMPROVEMENT_TOLERANCE = 1e-5 # If HV doesn't improve by more than this
LAST_N_ITER_FORCED_BETA_2_0_AND_EI = 50 # Iterations from end to apply special late stage logic
beta_alternation_block_length = 10
forced_stall_weight_cycle = [
    torch.tensor([0.8, 0.1, 0.1], device=X_init.device),
    torch.tensor([0.1, 0.8, 0.1], device=X_init.device),
    torch.tensor([0.1, 0.1, 0.8], device=X_init.device),
    torch.tensor([0.4, 0.4, 0.2], device=X_init.device),
    torch.tensor([0.4, 0.2, 0.4], device=X_init.device),
    torch.tensor([0.2, 0.4, 0.4], device=X_init.device),
]
forced_stall_weight_idx = 0

hypervolume_history = []
selection_methods = []
evaluated_candidate_indices = set()

train_x_scaled = X_init_scaled.clone()
train_y_standardized = train_y_standardized.clone() # This is Y_init standardized

hv_ref_point_raw = torch.tensor([1500.0, 1500.0, 1.0], dtype=torch.float64) # Stress1, Stress2, Mass
negated_ref_point_for_hv_calc = -hv_ref_point_raw # BoTorch HV works with maximization
print(f"\nUsing FIXED Hypervolume reference point (raw objectives, minimization form): {hv_ref_point_raw.tolist()}")
print(f"   Negated ref point for BoTorch HV calc (maximization form): {negated_ref_point_for_hv_calc.tolist()}")

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
            initial_hv = hv_calculator.compute(-initial_pareto_raw) # Input to compute should be maximized
        except Exception as e:
            print(f"Warning: Initial hypervolume calculation failed: {e}. Setting initial HV to 0.")
print(f"Initial Hypervolume: {initial_hv:.6f}, Initial Pareto Size: {initial_pareto_raw_size}")
hypervolume_history.append(initial_hv)
previous_pareto_front_size = initial_pareto_raw_size


script_version_name = f"{CURRENT_MODEL_TYPE}_no_tuning"
print(f"\nStarting Bayesian Optimization for {n_iterations} iterations ({script_version_name})...")
print(f"  Using Surrogate Model: {CURRENT_MODEL_TYPE}")
if CURRENT_MODEL_TYPE == "NGBoost":
    print(f"  NGBoost parameters: {NGB_PARAMS}")

start_bo_time = time.monotonic()

# --- BO Loop ---
for i_iter_loop in range(n_iterations):
    iter_display = i_iter_loop + 1
    print(f"\nIteration {iter_display}/{n_iterations}")

    print(f"  Training {CURRENT_MODEL_TYPE} surrogate model...")
    model_training_start_time = time.monotonic()
    try:
        if CURRENT_MODEL_TYPE == "NGBoost":
            model = CustomNGBoostModel(
                train_X=train_x_scaled,
                train_Y=train_y_standardized,
                ngb_params=NGB_PARAMS
            )
        # elif CURRENT_MODEL_TYPE == "XGBoost": # Keep for reference if needed
        #     model = CustomXGBoostModel(
        #         train_X=train_x_scaled,
        #         train_Y=train_y_standardized,
        #         num_ensemble_models=XGB_ENSEMBLE_SIZE, # Would need to define these again
        #         xgb_params=XGB_PARAMS
        #     )
        else:
            raise ValueError(f"Unsupported CURRENT_MODEL_TYPE: {CURRENT_MODEL_TYPE}")

        model.eval() # Set to evaluation mode
        model_training_duration = time.monotonic() - model_training_start_time
        print(f"  Model training took {model_training_duration:.2f} seconds.")

    except Exception as e:
        print(f"  Skipping iteration {iter_display} due to {CURRENT_MODEL_TYPE} model training error: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        hypervolume_history.append(hypervolume_history[-1] if hypervolume_history else 0.0)
        selection_methods.append('Error')
        if not hypervolume_history: hypervolume_history.append(0.0)
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

    mean_obj1_std = means_standardized[:, obj_indices[0]]
    std_obj1_std = stds_standardized[:, obj_indices[0]]
    mean_obj2_std = means_standardized[:, obj_indices[1]]
    std_obj2_std = stds_standardized[:, obj_indices[1]]
    mean_obj3_std = means_standardized[:, obj_indices[2]]
    std_obj3_std = stds_standardized[:, obj_indices[2]]
    mean_constr_std = means_standardized[:, constraint_index_in_y]
    std_constr_std = stds_standardized[:, constraint_index_in_y]

    current_beta_lcb: float
    w1: float; w2: float; w3: float
    # rule_reason: str # Declared later based on logic
    _rule_reason_beta_part_for_print = "Initial" # Placeholder

    objective_variances_std = variances_standardized[:, obj_indices]
    current_observed_avg_variance_k = torch.mean(torch.mean(objective_variances_std, dim=1)) if objective_variances_std.numel() > 0 else torch.tensor(0.0, device=X_init.device)

    observed_hv_improvement_k = torch.tensor(0.0, dtype=torch.float64, device=X_init.device)
    if len(hypervolume_history) > 1:
        observed_hv_improvement_k = hypervolume_history[-1] - hypervolume_history[-2]


    if i_iter_loop == 0:
        smooth_avg_variance = current_observed_avg_variance_k.clone()
        if len(hypervolume_history) > 1 and (hypervolume_history[-1] - hypervolume_history[-2]) != 0 :
             smooth_hv_improvement = (hypervolume_history[-1] - hypervolume_history[-2])
        else:
             smooth_hv_improvement = torch.tensor(0.0, dtype=torch.float64, device=X_init.device)

        initial_model_variance_for_schedule = current_observed_avg_variance_k.item()
        if DYNAMIC_EARLY_EXPLORE_ITERATIONS is None: # Set only once
            DYNAMIC_EARLY_EXPLORE_ITERATIONS = calculate_dynamic_early_exploration_duration(
                n_iterations,
                num_initial_samples,
                initial_model_variance_for_schedule
            )
            print(f"    Dynamic early exploration phase set to the first {DYNAMIC_EARLY_EXPLORE_ITERATIONS} iterations.")
    else:
        smooth_avg_variance = (1.0 - alpha_variance) * smooth_avg_variance + alpha_variance * current_observed_avg_variance_k
        smooth_hv_improvement = (1.0 - alpha_hv_improvement) * smooth_hv_improvement + alpha_hv_improvement * observed_hv_improvement_k

    print(f"    EMA Metrics: SmoothAvgVar={smooth_avg_variance.item():.4f}, SmoothHVImprove={smooth_hv_improvement.item():.6f}")
    print(f"    Stall counter: {stall_counter}, Forced exploration iters left: {forced_exploration_iters_left}")

    is_late_stage_iteration = (i_iter_loop >= (n_iterations - LAST_N_ITER_FORCED_BETA_2_0_AND_EI))
    use_dynamic_ei_weights_for_this_iteration = True # Default
    rule_reason = "" # Initialize rule_reason

    if forced_exploration_iters_left > 0:
        # Always use max exploration beta and forced stall weights during ANY forced exploration
        current_beta_lcb = 2.0 # Max exploration beta from your beta_states list
        current_forced_weights = forced_stall_weight_cycle[forced_stall_weight_idx % len(forced_stall_weight_cycle)]
        w1, w2, w3 = current_forced_weights[0].item(), current_forced_weights[1].item(), current_forced_weights[2].item()
        
        _reason_beta_part_for_print = f"Max Beta {current_beta_lcb:.1f}"
        if is_late_stage_iteration: # Specific annotation for late stage, though beta is max
            _reason_beta_part_for_print = f"Max Beta {current_beta_lcb:.1f} (Late Stage Forced)"

        rule_reason = f"Forced stall exploration ({forced_exploration_iters_left} iters left, {_reason_beta_part_for_print}, Forced Stall W=[{w1:.1f},{w2:.1f},{w3:.1f}])"
        forced_stall_weight_idx += 1
        use_dynamic_ei_weights_for_this_iteration = False
    else: # Not in forced exploration
        if DYNAMIC_EARLY_EXPLORE_ITERATIONS is None:
             DYNAMIC_EARLY_EXPLORE_ITERATIONS = int(n_iterations * 0.30) # Fallback
             print("    Warning: DYNAMIC_EARLY_EXPLORE_ITERATIONS was not set, using default fallback.")

        if i_iter_loop < DYNAMIC_EARLY_EXPLORE_ITERATIONS:
            current_beta_lcb = beta_states[-1].item() # Max exploration beta
            _rule_reason_beta_part_for_print = f"Dynamic early explore (iter {i_iter_loop+1}/{DYNAMIC_EARLY_EXPLORE_ITERATIONS}, Beta {current_beta_lcb:.1f})"
        elif smooth_hv_improvement < LOW_HV_IMPROVEMENT_THRESH:
            if smooth_avg_variance > HIGH_VARIANCE_THRESH:
                current_beta_lcb = beta_states[-1].item() # Max exploration
                _rule_reason_beta_part_for_print = f"EMA Stagnant, high variance (Beta {current_beta_lcb:.1f})"
            elif smooth_avg_variance > MODERATE_VARIANCE_THRESH:
                current_beta_lcb = beta_states[-2].item() # Second highest exploration
                _rule_reason_beta_part_for_print = f"EMA Stagnant, moderate variance (Beta {current_beta_lcb:.1f})"
            else: # low variance
                current_beta_lcb = beta_states[1].item() # Precise exploitation
                _rule_reason_beta_part_for_print = f"EMA Stagnant, low var -> precise exploitation (Beta {current_beta_lcb:.1f})"
        elif smooth_avg_variance > HIGH_VARIANCE_THRESH:
            current_beta_lcb = beta_states[3].item() # Explore
            _rule_reason_beta_part_for_print = f"Decent improv (implied), high variance (Beta {current_beta_lcb:.1f})"
        elif i_iter_loop > MID_ITER_MAX_EMA:
            if smooth_avg_variance < LOW_VARIANCE_THRESH:
                current_beta_lcb = beta_states[0].item() # Max exploit
                _rule_reason_beta_part_for_print = f"Late stage, low variance -> exploit (Beta {current_beta_lcb:.1f})"
            elif smooth_avg_variance < MODERATE_VARIANCE_THRESH:
                current_beta_lcb = beta_states[1].item() # Mild exploit
                _rule_reason_beta_part_for_print = f"Late stage, mod-low variance -> mild exploit (Beta {current_beta_lcb:.1f})"
            else: 
                current_beta_lcb = beta_states[2].item() # Balanced
                _rule_reason_beta_part_for_print = f"Late stage, default balanced (Beta {current_beta_lcb:.1f})"
        else: # Mid-stage, decent HV improvement, moderate/low variance
            block_phase_alternating = (i_iter_loop // beta_alternation_block_length) % 2
            if block_phase_alternating == 0: 
                current_beta_lcb = beta_states[3].item() # Explore a bit more
            else: 
                current_beta_lcb = beta_states[2].item() # Exploit a bit more
            _rule_reason_beta_part_for_print = f"Mid-stage, Alternating Beta ({current_beta_lcb:.1f})"
        
        # Dynamic EI weights will be calculated next if use_dynamic_ei_weights_for_this_iteration is True

    if use_dynamic_ei_weights_for_this_iteration:
        # ... (rest of your EI calculation logic for w1, w2, w3)
        current_train_y_raw_negated_inv_transform = torch.tensor(y_scaler.inverse_transform(train_y_standardized.cpu().numpy()), dtype=torch.float64) # ensure cpu for scaler
        current_train_constraints_raw = current_train_y_raw_negated_inv_transform[:, constraint_index_in_y]
        current_feasible_mask_train = current_train_constraints_raw <= constraint_threshold

        default_tau_std_value = torch.tensor(-3.0, device=train_x_scaled.device, dtype=torch.float64)

        tau_obj1_std_neg = default_tau_std_value.clone()
        tau_obj2_std_neg = default_tau_std_value.clone()
        tau_obj3_std_neg = default_tau_std_value.clone()

        if torch.any(current_feasible_mask_train):
            feasible_train_y_standardized = train_y_standardized[current_feasible_mask_train]
            if feasible_train_y_standardized.shape[0] > 0:
                tau_obj1_std_neg = torch.max(feasible_train_y_standardized[:, obj_indices[0]])
                tau_obj2_std_neg = torch.max(feasible_train_y_standardized[:, obj_indices[1]])
                tau_obj3_std_neg = torch.max(feasible_train_y_standardized[:, obj_indices[2]])

        epsilon_sigma = 1e-9
        norm_dist = torch.distributions.Normal(torch.tensor(0.0, device=train_x_scaled.device), torch.tensor(1.0, device=train_x_scaled.device))

        mu1, sigma1 = mean_obj1_std, std_obj1_std + epsilon_sigma
        Z1 = (mu1 - tau_obj1_std_neg) / sigma1
        ei1_values = torch.clamp((mu1 - tau_obj1_std_neg) * norm_dist.cdf(Z1) + sigma1 * norm_dist.log_prob(Z1).exp(), min=0.0)

        mu2, sigma2 = mean_obj2_std, std_obj2_std + epsilon_sigma
        Z2 = (mu2 - tau_obj2_std_neg) / sigma2
        ei2_values = torch.clamp((mu2 - tau_obj2_std_neg) * norm_dist.cdf(Z2) + sigma2 * norm_dist.log_prob(Z2).exp(), min=0.0)

        mu3, sigma3 = mean_obj3_std, std_obj3_std + epsilon_sigma
        Z3 = (mu3 - tau_obj3_std_neg) / sigma3
        ei3_values = torch.clamp((mu3 - tau_obj3_std_neg) * norm_dist.cdf(Z3) + sigma3 * norm_dist.log_prob(Z3).exp(), min=0.0)

        S1 = torch.max(ei1_values) if ei1_values.numel() > 0 else torch.tensor(0.0, device=train_x_scaled.device)
        S2 = torch.max(ei2_values) if ei2_values.numel() > 0 else torch.tensor(0.0, device=train_x_scaled.device)
        S3 = torch.max(ei3_values) if ei3_values.numel() > 0 else torch.tensor(0.0, device=train_x_scaled.device)

        epsilon_score = 1e-7
        s1_eff = S1 + epsilon_score
        s2_eff = S2 + epsilon_score
        s3_eff = S3 + epsilon_score
        sum_S_eff = s1_eff + s2_eff + s3_eff

        if sum_S_eff <= (n_objectives * epsilon_score): # Avoid division by zero if all S_eff are ~0
            w1, w2, w3 = 1/3., 1/3., 1/3.
        else:
            w1 = (s1_eff / sum_S_eff).item()
            w2 = (s2_eff / sum_S_eff).item()
            w3 = (s3_eff / sum_S_eff).item()
        
        rule_reason = f"Adaptive Beta (Rule: {_rule_reason_beta_part_for_print}, Dynamic EI W=[{w1:.3f},{w2:.3f},{w3:.3f}])"


    print(f"  Using LCB parameters for objectives: {rule_reason}")

    # Constraint LCB beta logic (this was the main change from your previous run)
    beta_constraint = 1.0
    constraint_lcb_values = mean_constr_std - beta_constraint * std_constr_std
    
    predicted_feasibility_mask = (constraint_lcb_values <= constraint_threshold_std)
    penalty_value = -1e9

    if X_candidates_scaled_subset.shape[0] == 0:
        print("    Error: No candidates to select from. Skipping iteration.")
        hypervolume_history.append(hypervolume_history[-1] if hypervolume_history else 0.0)
        selection_methods.append('Error')
        continue

    scalarized_lcb_objectives_std = (
        w1 * (mean_obj1_std - current_beta_lcb * std_obj1_std) +
        w2 * (mean_obj2_std - current_beta_lcb * std_obj2_std) +
        w3 * (mean_obj3_std - current_beta_lcb * std_obj3_std)
    )

    if not torch.any(predicted_feasibility_mask):
        print("    Warning: All candidates predicted infeasible by LCB. Selecting best LCB among them (ignoring constraint for this one selection).")
        effective_scalarized_lcb_std = scalarized_lcb_objectives_std
        best_lcb_idx_in_subset = torch.argmax(effective_scalarized_lcb_std)
    else:
        effective_scalarized_lcb_std = torch.full_like(mean_obj1_std, penalty_value, dtype=torch.float64, device=mean_obj1_std.device)
        feasible_indices = torch.where(predicted_feasibility_mask)[0]

        if feasible_indices.numel() > 0:
            effective_scalarized_lcb_std[feasible_indices] = scalarized_lcb_objectives_std[feasible_indices]
        # Ensure there's at least one non-penalty value before argmax if all feasible are -inf LCB
        if torch.all(effective_scalarized_lcb_std[feasible_indices] == penalty_value) and feasible_indices.numel() > 0 :
             print("    Warning: All LCB-feasible candidates have penalty LCB objective value. Picking first feasible by LCB constraint.")
             best_lcb_idx_in_subset = feasible_indices[0]
        elif feasible_indices.numel() == 0: # Should be caught by "All candidates predicted infeasible"
             print("    Error state: No feasible_indices but not caught by initial check. Selecting overall best LCB.")
             effective_scalarized_lcb_std = scalarized_lcb_objectives_std
             best_lcb_idx_in_subset = torch.argmax(effective_scalarized_lcb_std)
        else:
            best_lcb_idx_in_subset = torch.argmax(effective_scalarized_lcb_std)


    selected_candidate_original_idx = unevaluated_indices_tensor[best_lcb_idx_in_subset].item()
    _original_idx, _closest_cand_X_unscaled, next_y_obj1_raw, next_y_obj2_raw, next_y_obj3_raw, next_y_constr_raw = find_closest_candidate(
        X_candidates, Y_candidates, objective_index_1, objective_index_2, objective_index_3, constraint_index, selected_candidate_original_idx)
    closest_cand_X_scaled = X_candidates_scaled[selected_candidate_original_idx].unsqueeze(0)

    print(f"   Selected candidate (Original Index in X_candidates: {selected_candidate_original_idx}) via constrained LCB.")
    selected_constr_lcb_val_display = constraint_lcb_values[best_lcb_idx_in_subset].item()
    selected_feasibility_prediction_display = predicted_feasibility_mask[best_lcb_idx_in_subset].item()
    print(f"   Constraint LCB value for selected: {selected_constr_lcb_val_display:.4f} (Threshold_std: {constraint_threshold_std:.4f})")
    print(f"   LCB predicted feasibility for selected: {selected_feasibility_prediction_display}")
    selection_methods.append('LCB')

    print(f"   Adding Candidate Index: {selected_candidate_original_idx}")
    print(f"   Raw Objective 1 ({objective_target_name_1}): {-next_y_obj1_raw.item():.4f}")
    print(f"   Raw Objective 2 ({objective_target_name_2}): {-next_y_obj2_raw.item():.4f}")
    print(f"   Raw Objective 3 ({objective_target_name_3}): {-next_y_obj3_raw.item():.6f}")
    print(f"   Constraint ({constraint_target_name}): {next_y_constr_raw.item():.4f}")

    next_y_combined_raw_for_model = torch.cat([next_y_obj1_raw, next_y_obj2_raw, next_y_obj3_raw, next_y_constr_raw], dim=1)
    next_y_standardized = torch.tensor(y_scaler.transform(next_y_combined_raw_for_model.numpy()), dtype=torch.float64)

    train_x_scaled = torch.cat([train_x_scaled, closest_cand_X_scaled], dim=0)
    train_y_standardized = torch.cat([train_y_standardized, next_y_standardized], dim=0)
    evaluated_candidate_indices.add(selected_candidate_original_idx)

    all_y_raw_from_standardized = torch.tensor(y_scaler.inverse_transform(train_y_standardized.cpu().numpy()), dtype=torch.float64)
    all_objectives_raw_minimization = -all_y_raw_from_standardized[:, obj_indices]
    all_constraints_raw = all_y_raw_from_standardized[:, constraint_index_in_y]
    feasible_mask = all_constraints_raw <= constraint_threshold
    current_objectives_feasible_raw = all_objectives_raw_minimization[feasible_mask]
    current_hv = 0.0
    pareto_front_raw_minimization = torch.empty((0, n_objectives), dtype=current_objectives_feasible_raw.dtype, device=X_init.device)
    current_pareto_front_size = 0
    if current_objectives_feasible_raw.shape[0] > 0:
        non_dominated_mask = simple_is_non_dominated(current_objectives_feasible_raw)
        pareto_front_raw_minimization = current_objectives_feasible_raw[non_dominated_mask]
        current_pareto_front_size = pareto_front_raw_minimization.shape[0]
        if pareto_front_raw_minimization.shape[0] > 0:
            try:
                hv_calculator = Hypervolume(ref_point=negated_ref_point_for_hv_calc)
                current_hv = hv_calculator.compute(-pareto_front_raw_minimization)
            except Exception as e:
                print(f"    Warning: HV calculation error: {e}")
                current_hv = hypervolume_history[-1] if hypervolume_history else 0.0
    hypervolume_history.append(current_hv)

    hv_improved_significantly = False
    if len(hypervolume_history) > 1:
        if hypervolume_history[-1] > hypervolume_history[-2] + STALL_HV_IMPROVEMENT_TOLERANCE:
            hv_improved_significantly = True

    pareto_front_size_changed = False
    if current_pareto_front_size != previous_pareto_front_size:
        pareto_front_size_changed = True

    if hv_improved_significantly or pareto_front_size_changed:
        if stall_counter > 0:
            reasons_for_reset = []
            if hv_improved_significantly: reasons_for_reset.append("HV improvement")
            if pareto_front_size_changed: reasons_for_reset.append(f"Pareto front size change ({previous_pareto_front_size} -> {current_pareto_front_size})")
            print(f"    Stall counter reset from {stall_counter} to 0 due to: {', '.join(reasons_for_reset)}.")
        stall_counter = 0
    else:
        if i_iter_loop > 0 : # Don't increment stall counter on the very first iteration after init
            hv_diff = hypervolume_history[-1] - hypervolume_history[-2] if len(hypervolume_history) > 1 else 0.0
            stall_counter += 1
            print(f"    Stall counter incremented to {stall_counter}. (HV change: {hv_diff:.6f}, Pareto size: {previous_pareto_front_size} -> {current_pareto_front_size})")

    previous_pareto_front_size = current_pareto_front_size

    if stall_counter >= N_STALL_THRESH and forced_exploration_iters_left == 0:
        print(f"    STALLED for {stall_counter} iterations. Forcing exploration for {FORCED_EXPLORATION_DURATION} iters.")
        forced_exploration_iters_left = FORCED_EXPLORATION_DURATION
        forced_stall_weight_idx = 0 # Reset weight cycle index

    # Decrement forced exploration counter if it was active and a selection rule for it was used
    if forced_exploration_iters_left > 0 and rule_reason.startswith("Forced stall exploration"):
         forced_exploration_iters_left -=1


    print(f"   Current Hypervolume: {current_hv:.6f}")
    print(f"   Pareto Front Size: {pareto_front_raw_minimization.shape[0]}")
    print(f"   Total unique candidates evaluated: {len(evaluated_candidate_indices)}")
    print(f"   Total points in training data: {train_x_scaled.shape[0]}")

# --- End of BO Loop ---
end_bo_time = time.monotonic()
bo_duration_seconds = end_bo_time - start_bo_time
print("\nOptimization finished.")
print(f"Total BO loop duration: {bo_duration_seconds:.2f} seconds ({bo_duration_seconds/60:.2f} minutes)")

# --- Results ---
final_y_all_standardized = train_y_standardized
final_y_all_raw_model_form = torch.tensor(y_scaler.inverse_transform(final_y_all_standardized.cpu().numpy()), dtype=torch.float64)
final_objectives_raw_minimization = -final_y_all_raw_model_form[:, obj_indices]
final_constraints_raw = final_y_all_raw_model_form[:, constraint_index_in_y]
feasible_mask_final = final_constraints_raw <= constraint_threshold
final_objectives_feasible_minimization = final_objectives_raw_minimization[feasible_mask_final]
final_bo_pareto_points_raw_minimization = torch.empty((0, n_objectives), dtype=torch.float64, device=X_init.device)
if final_objectives_feasible_minimization.shape[0] > 0:
    non_dominated_mask_final = simple_is_non_dominated(final_objectives_feasible_minimization)
    final_bo_pareto_points_raw_minimization = final_objectives_feasible_minimization[non_dominated_mask_final]
    if final_bo_pareto_points_raw_minimization.shape[0] > 0:
        print(f"\nFound {len(final_bo_pareto_points_raw_minimization)} BO Pareto points ({CURRENT_MODEL_TYPE}):")
        # Sort for consistent output before printing (optional)
        # final_bo_pareto_points_raw_minimization = final_bo_pareto_points_raw_minimization[torch.argsort(final_bo_pareto_points_raw_minimization[:,0])]
        print(pd.DataFrame(final_bo_pareto_points_raw_minimization.cpu().numpy(), columns=objective_names))

# Find true pareto front
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
    precision_dup = 1e-5 # Adjust precision as needed for de-duplication
    rounded_objectives = np.round(true_space_objectives_feasible_min.cpu().numpy() / precision_dup) * precision_dup
    # Create a DataFrame from the rounded objectives for de-duplication
    df_true_space = pd.DataFrame(rounded_objectives) # No column names needed just for drop_duplicates
    unique_indices = df_true_space.drop_duplicates().index
    unique_true_space_objectives_min = true_space_objectives_feasible_min[unique_indices]
except Exception as e:
    print(f"Warning: Error during de-duplication of true space objectives: {e}")
    unique_true_space_objectives_min = true_space_objectives_feasible_min # Fallback

true_pareto_front_raw_minimization = torch.empty((0, n_objectives), dtype=torch.float64, device=X_init.device); true_hv = 0.0
if unique_true_space_objectives_min is not None and unique_true_space_objectives_min.shape[0] > 0:
    non_dominated_mask_true_space = simple_is_non_dominated(unique_true_space_objectives_min)
    true_pareto_front_raw_minimization = unique_true_space_objectives_min[non_dominated_mask_true_space]

    print(f"\nTrue Pareto Front ({true_pareto_front_raw_minimization.shape[0]} points, minimization objectives):")
    # Sort for consistent output (optional)
    # true_pareto_front_raw_minimization = true_pareto_front_raw_minimization[torch.argsort(true_pareto_front_raw_minimization[:,0])]
    print(pd.DataFrame(true_pareto_front_raw_minimization.cpu().numpy(), columns=objective_names).to_string())

    if true_pareto_front_raw_minimization.shape[0] > 0:
        try:
            hv_calculator_true = Hypervolume(ref_point=negated_ref_point_for_hv_calc)
            true_hv = hv_calculator_true.compute(-true_pareto_front_raw_minimization)
            print(f"True Max Hypervolume (from unique feasible points): {true_hv:.6f}")
        except Exception as e:
            print(f"Warning: True HV calculation error: {e}")
            true_hv = 0.0 # Fallback
        final_bo_hv = hypervolume_history[-1] if hypervolume_history else 0.0
        print(f"Final BO Hypervolume: {final_bo_hv:.6f}")
        if true_hv > 1e-12: # Avoid division by zero or tiny numbers
             print(f"HV Ratio (BO/True): {final_bo_hv / true_hv:.4f}")
else:
    print("Could not determine true Pareto front (no unique feasible points or error).")


print("\n--- Checking Overlap Between BO Front and True Front ---")

# Use a consistent rounding precision for comparison
# You used 5dp in your output, let's stick with that or make it a variable
comparison_precision_dp = 5 # Number of decimal places for comparison

if final_bo_pareto_points_raw_minimization.shape[0] > 0 and true_pareto_front_raw_minimization.shape[0] > 0:
    
    # Convert BO Pareto points to a set of tuples of rounded numbers
    bo_pareto_tuples_rounded = set()
    for point_tensor in final_bo_pareto_points_raw_minimization:
        # Ensure tensor is on CPU and converted to list of Python floats before rounding
        rounded_tuple = tuple(round(x.item(), comparison_precision_dp) for x in point_tensor.cpu())
        bo_pareto_tuples_rounded.add(rounded_tuple)

    # Convert True Pareto points to a list of tuples of rounded numbers (for iteration)
    # and also a set for efficient lookup
    true_pareto_tuples_rounded_list = []
    true_pareto_tuples_rounded_set = set()
    true_pareto_original_tensors_map = {} # To store original tensor for printing missed ones

    for point_tensor in true_pareto_front_raw_minimization:
        rounded_tuple = tuple(round(x.item(), comparison_precision_dp) for x in point_tensor.cpu())
        true_pareto_tuples_rounded_list.append(rounded_tuple)
        true_pareto_tuples_rounded_set.add(rounded_tuple)
        # Store the original tensor if you want to print the unrounded values of missed points
        if rounded_tuple not in true_pareto_original_tensors_map: # Keep first occurrence if duplicates after rounding
             true_pareto_original_tensors_map[rounded_tuple] = point_tensor.cpu().numpy()


    # --- Count Matches ---
    # Points in the BO front that are also in the True Pareto front
    matched_bo_points_count = 0
    for bo_tuple in bo_pareto_tuples_rounded:
        if bo_tuple in true_pareto_tuples_rounded_set:
            matched_bo_points_count += 1
    
    # Points on the True Pareto front that were found by BO
    # This is essentially the same as above if we consider the sets.
    # More directly: intersection_size = len(bo_pareto_tuples_rounded.intersection(true_pareto_tuples_rounded_set))
    # matched_bo_points_count will be this intersection_size.

    print(f"  - BO found {final_bo_pareto_points_raw_minimization.shape[0]} unique Pareto points (minimization objectives).")
    print(f"  - True Pareto Front contains {true_pareto_front_raw_minimization.shape[0]} unique points (minimization objectives).")
    print(f"  - {matched_bo_points_count} of the BO Pareto points (rounded to {comparison_precision_dp}dp) are present in the True Pareto Front.")
    
    # --- Identify Missed True Pareto Points ---
    missed_true_pareto_points_tensors = []
    for true_rounded_tuple in true_pareto_tuples_rounded_set: # Iterate through unique true pareto points
        if true_rounded_tuple not in bo_pareto_tuples_rounded:
            # Retrieve the original (unrounded or less rounded) tensor for printing
            # This assumes true_pareto_original_tensors_map was populated correctly
            original_tensor_np = true_pareto_original_tensors_map.get(true_rounded_tuple)
            if original_tensor_np is not None:
                missed_true_pareto_points_tensors.append(original_tensor_np)
            else:
                # Fallback if map wasn't populated (should not happen with current logic)
                # Convert tuple back to a representative numpy array, though it's already rounded
                missed_true_pareto_points_tensors.append(np.array(true_rounded_tuple))


    if missed_true_pareto_points_tensors:
        print(f"\n--- {len(missed_true_pareto_points_tensors)} True Pareto Points MISSED by BO (objectives rounded to {comparison_precision_dp}dp for matching) ---")
        # Create DataFrame from the list of numpy arrays
        df_missed = pd.DataFrame(missed_true_pareto_points_tensors, columns=objective_names)
        # Optional: Sort for consistent viewing
        # df_missed = df_missed.sort_values(by=objective_names[0]).reset_index(drop=True)
        print(df_missed.to_string())
    elif matched_bo_points_count == true_pareto_front_raw_minimization.shape[0] and final_bo_pareto_points_raw_minimization.shape[0] >= true_pareto_front_raw_minimization.shape[0]:
        print("\n  All True Pareto points were found by BO (after rounding for matching).")
    else:
        # This case might occur if counts are off due to rounding or other logic issues
        print("\n  Could not definitively list missed points, or all were found. Please check counts.")
        print(f"  (Matched: {matched_bo_points_count}, True Pareto Size: {true_pareto_front_raw_minimization.shape[0]})")


    # --- Identify BO Pareto Points NOT in True Pareto Front (Optional, but good for sanity) ---
    bo_points_not_in_true_pareto_tensors = []
    temp_bo_original_tensors_map = {} # Map rounded tuples of BO points to their original tensors
    for point_tensor in final_bo_pareto_points_raw_minimization:
        rounded_tuple = tuple(round(x.item(), comparison_precision_dp) for x in point_tensor.cpu())
        if rounded_tuple not in temp_bo_original_tensors_map:
            temp_bo_original_tensors_map[rounded_tuple] = point_tensor.cpu().numpy()

    for bo_rounded_tuple in bo_pareto_tuples_rounded:
        if bo_rounded_tuple not in true_pareto_tuples_rounded_set:
            original_tensor_np = temp_bo_original_tensors_map.get(bo_rounded_tuple)
            if original_tensor_np is not None:
                 bo_points_not_in_true_pareto_tensors.append(original_tensor_np)

    if bo_points_not_in_true_pareto_tensors:
        print(f"\n--- {len(bo_points_not_in_true_pareto_tensors)} BO Pareto Points NOT in True Pareto Front (objectives rounded to {comparison_precision_dp}dp for matching) ---")
        df_bo_extra = pd.DataFrame(bo_points_not_in_true_pareto_tensors, columns=objective_names)
        # df_bo_extra = df_bo_extra.sort_values(by=objective_names[0]).reset_index(drop=True)
        print(df_bo_extra.to_string())


elif final_bo_pareto_points_raw_minimization.shape[0] == 0:
    print("\nCannot check overlap: Final BO Pareto Front is empty.")
    if true_pareto_front_raw_minimization.shape[0] > 0:
        print(f"  True Pareto Front contains {true_pareto_front_raw_minimization.shape[0]} points.")
else: # True Pareto front is empty
    print("\nCannot check overlap: True Pareto Front is empty.")
    if final_bo_pareto_points_raw_minimization.shape[0] > 0:
        print(f"  BO found {final_bo_pareto_points_raw_minimization.shape[0]} Pareto points.")


plot_suffix = f"{script_version_name}_iter{n_iterations}"

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
bo_hypervolumes_plot = [] # Initialize
if len(selection_methods) > 0:
    bo_iterations_plot = list(range(1, len(selection_methods) + 1))
    # Ensure bo_hypervolumes_plot aligns with bo_iterations_plot
    if len(hypervolume_history) > len(selection_methods):
        bo_hypervolumes_plot = hypervolume_history[1:len(selection_methods)+1]
    elif len(hypervolume_history) > 0 : # handles case where len(hv_hist) <= len(sel_methods) + 1
        bo_hypervolumes_plot = hypervolume_history[1:]


    if len(bo_iterations_plot) == len(bo_hypervolumes_plot) and len(bo_hypervolumes_plot) > 0:
        plt.plot(bo_iterations_plot, bo_hypervolumes_plot, color='darkgrey', linestyle='-', zorder=1, alpha=0.7)
        color_map = {'LCB': '#377EB8', 'Error': 'grey'} # Add more if other methods are logged
        plot_colors = [color_map.get(method, 'black') for method in selection_methods[:len(bo_iterations_plot)]]
        plt.scatter(bo_iterations_plot, bo_hypervolumes_plot, c=plot_colors, marker='o', s=50, zorder=2, edgecolors='grey', alpha=0.9)

if true_pareto_front_raw_minimization.shape[0] > 0 and true_hv > 1e-9:
     plt.axhline(y=true_hv, color='green', linestyle='--', label=f'True Max HV ({true_hv:.4f})')

legend_elements = [Line2D([0], [0], marker='o', color='w', label='LCB', markerfacecolor='#377EB8', markersize=8)]
if 'Error' in selection_methods: # Check if 'Error' actually occurred
    legend_elements.append(Line2D([0], [0], marker='o', color='w', label=f'{CURRENT_MODEL_TYPE} Error', markerfacecolor='grey', markersize=8))
if true_pareto_front_raw_minimization.shape[0] > 0 and true_hv > 1e-9:
    legend_elements.append(Line2D([0],[0], color='green', linestyle='--', label=f'True Max HV ({true_hv:.4f})'))

plt.legend(handles=legend_elements)
plt.xlabel("BO Iteration")
plt.ylabel("Current HV")
plt.title(f"BO HV Progression ({plot_suffix})")
plt.grid(True)
plt.xlim(left=0)
if len(bo_hypervolumes_plot) > 0 :
    min_hv_plot = min(bo_hypervolumes_plot)
    plt.ylim(bottom=min(0, min_hv_plot * 1.1 if min_hv_plot < 0 else min_hv_plot * 0.9 ))
plt.tight_layout()
plt.savefig(f'selection_method_plot_{plot_suffix}.png')
plt.close()

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
if final_objectives_raw_minimization.shape[0] > 0:
    obj1_all_min=final_objectives_raw_minimization[:,0].cpu().numpy()
    obj2_all_min=final_objectives_raw_minimization[:,1].cpu().numpy()
    obj3_all_min=final_objectives_raw_minimization[:,2].cpu().numpy()
    constr_all_final=final_constraints_raw.cpu().numpy()
    feasible_plot_mask = constr_all_final <= constraint_threshold
    ax.scatter(obj1_all_min[~feasible_plot_mask], obj2_all_min[~feasible_plot_mask], obj3_all_min[~feasible_plot_mask], c='grey', alpha=0.2, s=15, label='Infeasible Eval') # Lighter grey
    ax.scatter(obj1_all_min[feasible_plot_mask], obj2_all_min[feasible_plot_mask], obj3_all_min[feasible_plot_mask], c='blue', alpha=0.4, s=15, label='Feasible Eval') # Different blue

    if final_bo_pareto_points_raw_minimization.shape[0] > 0:
        ax.scatter(final_bo_pareto_points_raw_minimization[:,0].cpu().numpy(), final_bo_pareto_points_raw_minimization[:,1].cpu().numpy(), final_bo_pareto_points_raw_minimization[:,2].cpu().numpy(), c='limegreen', s=150, edgecolor='black', marker='*', label=f'BO Pareto ({plot_suffix})', zorder=3) # Lime green star
    if true_pareto_front_raw_minimization.shape[0] > 0:
        ax.scatter(true_pareto_front_raw_minimization[:,0].cpu().numpy(), true_pareto_front_raw_minimization[:,1].cpu().numpy(), true_pareto_front_raw_minimization[:,2].cpu().numpy(), facecolors='none', edgecolors='red', marker='o', s=60, linewidth=1.5, label='True Pareto', zorder=2)
    ax.set_xlabel(f"{objective_names[0]} (Min)")
    ax.set_ylabel(f"{objective_names[1]} (Min)")
    ax.set_zlabel(f"{objective_names[2]} (Min)")
    ax.set_title(f"3D Objective Space ({plot_suffix})")
    ax.legend(loc='best')
    plt.tight_layout()
    # Optional: Adjust view angle for 3D plot
    # ax.view_init(elev=20., azim=-35)
    plt.savefig(f'objective_space_3d_{plot_suffix}.png')
    plt.close()
else:
    print("\nNo points to plot 3D objective space (final_objectives_raw_minimization is empty).")


print("\nFinal training data size (scaled features):", train_x_scaled.shape)
print(f"Total points in hypervolume_history plot: {len(hypervolume_history)}")
print(f"Selection methods recorded: {selection_methods.count('LCB')} LCB, {selection_methods.count('Error')} Error")
print(f"Script finished. ({script_version_name})")
print(f"Using {CURRENT_MODEL_TYPE} as surrogate model.")
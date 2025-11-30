import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import warnings
import time

# botorch 
from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.multi_objective import ExpectedHypervolumeImprovement
from botorch.optim import optimize_acqf
from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning
from botorch.utils.multi_objective.hypervolume import Hypervolume

# gpytorch 
import gpytorch
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel

# sklearn 
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#output & plotting
from botorch.exceptions import InputDataWarning, BadInitialCandidatesWarning
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D


warnings.filterwarnings('ignore', category=InputDataWarning)
warnings.filterwarnings('ignore', category=BadInitialCandidatesWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)


#seed pytorch and numpy
torch.manual_seed(42)
np.random.seed(42)
torch.set_default_dtype(torch.float64)


# --- Simple Pairwise Non-Dominated Check (for minimization) ---
def simple_is_non_dominated(points):
    n_points = points.shape[0]
    if n_points == 0:
        return torch.tensor([], dtype=torch.bool)
    is_nd = torch.ones(n_points, dtype=torch.bool, device=points.device)
    for i in range(n_points):
        if not is_nd[i]: continue
        for j in range(n_points):
            if i == j: continue
            if torch.all(points[j] <= points[i]) and torch.any(points[j] < points[i]):
                is_nd[i] = False
                break
    return is_nd


# --- Preprocessing data ---

new_columns = [
    "FrontRear_height", "side_height", "side_width", "holes", "edge_fit", "rear_offset",
    "PSHELL_1_T", "PSHELL_2_T", "PSHELL_42733768_T", "PSHELL_42733769_T",
    "PSHELL_42733770_T", "PSHELL_42733772_T", "PSHELL_42733773_T", "PSHELL_42733774_T",
    "PSHELL_42733779_T", "PSHELL_42733780_T", "PSHELL_42733781_T", "PSHELL_42733782_T",
    "PSHELL_42733871_T", "PSHELL_42733879_T", "MAT1_1_E", "MAT1_42733768_E",
    "scale_x", "scale_y", "scale_z"
]

base_path = '../../'
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
init_df_targets = pd.concat(init_target_dfs, axis=1); init_df_targets.columns = target_column_names
df_init = pd.concat([df_init, init_df_targets], axis=1)

all_candidates_target_dfs = [pd.read_csv(file, header=None, sep=',') for file in all_candidates_target_files]
all_candidates_df_targets = pd.concat(all_candidates_target_dfs, axis=1); all_candidates_df_targets.columns = target_column_names
df_all_candidates = pd.concat([df_all_candidates, all_candidates_df_targets], axis=1)

X_init = torch.tensor(df_init[new_columns].values, dtype=torch.float64)
Y_init = torch.tensor(df_init[target_column_names].values, dtype=torch.float64)
X_candidates = torch.tensor(df_all_candidates[new_columns].values, dtype=torch.float64)
Y_candidates = torch.tensor(df_all_candidates[target_column_names].values, dtype=torch.float64)


# --- Multi-objective BO Setup ---
objective_target_name_1 = "max_displacement_torsion"
objective_target_name_2 = "max_stress_bending"
objective_target_name_3 = "mass" 
objective_names = [objective_target_name_1, objective_target_name_2, objective_target_name_3]

objective_index_1 = target_column_names.index(objective_target_name_1)
objective_index_2 = target_column_names.index(objective_target_name_2)
objective_index_3 = target_column_names.index(objective_target_name_3)

train_y_raw = torch.cat([
    -Y_init[:, objective_index_1].unsqueeze(-1),
    -Y_init[:, objective_index_2].unsqueeze(-1),
    -Y_init[:, objective_index_3].unsqueeze(-1) 
], dim=1)

obj_indices = [0, 1, 2]
n_objectives = len(obj_indices)
print(f"\nOptimizing objectives: Minimize {', '.join(objective_names)} (Unconstrained)")

# --- Feature Scaling & Standardization ---
X_combined = torch.cat([X_init, X_candidates], dim=0)
x_scaler = MinMaxScaler(); x_scaler.fit(X_combined.numpy())
X_init_scaled = torch.tensor(x_scaler.transform(X_init.numpy()), dtype=torch.float64)
X_candidates_scaled = torch.tensor(x_scaler.transform(X_candidates.numpy()), dtype=torch.float64)
scaled_bounds = torch.tensor([[0.0] * X_init.shape[-1], [1.0] * X_init.shape[-1]], dtype=torch.float64)

y_scaler = StandardScaler(); y_scaler.fit(train_y_raw.numpy()) # fit on 3 objectives
train_y_standardized = torch.tensor(y_scaler.transform(train_y_raw.numpy()), dtype=torch.float64)


# --- Independent GP Model Training Function ---
def train_independent_gps(train_x, train_y):
    models = []
    num_outputs = train_y.shape[-1]
    for i in range(num_outputs): #3 gps: 1 for each objective
        train_y_i = train_y[:, i].unsqueeze(-1)
        model_i = SingleTaskGP(train_x, train_y_i, covar_module=ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=train_x.shape[-1])))
        mll_i = ExactMarginalLogLikelihood(model_i.likelihood, model_i)
        try:
            with gpytorch.settings.cholesky_jitter(1e-4): fit_gpytorch_mll(mll_i, max_retries=5, options={'maxiter': 150})
        except Exception as e: print(f"Warning: GP fitting failed for objective {i}: {e}.")
        models.append(model_i)
    model = ModelListGP(*models) 
    return model


# --- Candidate Selection Function ---
def find_closest_candidate(x_proposed_unscaled, X_cands, Y_cands, obj_idx_1, obj_idx_2, obj_idx_3):
    distances = torch.norm(X_cands - x_proposed_unscaled.squeeze(0), dim=1) #euclidean distance
    closest_index = torch.argmin(distances)
    closest_X = X_cands[closest_index].unsqueeze(0)

    raw_obj1 = Y_cands[closest_index, obj_idx_1].item()
    raw_obj2 = Y_cands[closest_index, obj_idx_2].item()
    raw_obj3 = Y_cands[closest_index, obj_idx_3].item() 

    return closest_index.item(), closest_X, torch.tensor([[ -raw_obj1]]), torch.tensor([[ -raw_obj2]]), torch.tensor([[ -raw_obj3]]) # negated for minimization


# --- BO loop Setup ---
# --- Stopping Criterion Configuration ---
BASE_ITERS = 290
MAX_TOTAL_ITERS = 300
PATIENCE_WINDOW = 15
REL_IMPROVEMENT_TOL = 1e-5 # Stop if relative HV improvement is less than 0.001%

# --- BO State Initialization ---
actual_iterations_run = 0
hypervolume_history = []
selection_methods = []
original_indices = torch.tensor(df_init.index.tolist(), dtype=torch.long)
train_x_scaled = X_init_scaled.clone()
train_y_standardized = train_y_standardized.clone()

# --- Reference Point ---
hv_ref_point_raw = torch.tensor([1500.0, 1500.0, 1.0], dtype=torch.float64) #FIXED 3D reference point => bigger than all max objectives
print(f"\nUsing FIXED Hypervolume reference point (raw objectives): {hv_ref_point_raw.tolist()}")
hv_ref_point_std_neg = torch.tensor(y_scaler.transform(-hv_ref_point_raw.numpy().reshape(1, -1)), dtype=torch.float64).squeeze(0)
print(f"EHVI reference point (standardized negated objectives): {hv_ref_point_std_neg.tolist()}")


# --- Calculate Initial Hypervolume ---
initial_objectives_raw = Y_init[:, [objective_index_1, objective_index_2, objective_index_3]] # 3 objectives
initial_hv = 0.0
initial_pareto_raw = torch.empty((0, n_objectives), dtype=initial_objectives_raw.dtype)
try:
    if initial_objectives_raw.shape[0] > 0:
        non_dominated_mask_init = simple_is_non_dominated(initial_objectives_raw)
        initial_pareto_raw = initial_objectives_raw[non_dominated_mask_init]
        if initial_pareto_raw.shape[0] > 0:
            negated_initial_pareto = -initial_pareto_raw
            negated_ref_point = -hv_ref_point_raw
            hv_calculator = Hypervolume(ref_point=negated_ref_point)
            initial_hv = hv_calculator.compute(negated_initial_pareto)
            print(f"Initial Pareto Front (raw objectives, {len(initial_pareto_raw)} points):\n{initial_pareto_raw}")
        else: print("No non-dominated points found in the initial set.")
    else: print("No initial points available to calculate initial hypervolume.")
    print(f"Initial Hypervolume: {initial_hv:.6f}") 
    hypervolume_history.append(initial_hv)
except Exception as e: print(f"Warning: Initial hypervolume calculation failed: {e}. Setting initial HV to 0."); hypervolume_history.append(0.0)

print(f"\nStarting Bayesian Optimization...")
print(f"Total candidates: {len(X_candidates)}, Initial points: {len(X_init)}")
print(f"Will run for at least {BASE_ITERS} iterations, up to a max of {MAX_TOTAL_ITERS}.")
print(f"Will stop if relative HV improvement is less than {REL_IMPROVEMENT_TOL:.1e} over a {PATIENCE_WINDOW}-iteration window.")

#start timer
start_bo_time = time.monotonic()


# --- BO Loop ---
for i in range(MAX_TOTAL_ITERS):
    actual_iterations_run = i + 1
    print(f"\nIteration {i+1}/{MAX_TOTAL_ITERS}")
    if i < BASE_ITERS:
        print(f"  (Running guaranteed base iteration {i+1}/{BASE_ITERS})")
        
    fallback_used_this_iter = False #used for tracking selection method

    print("  Training Independent GPs...")
    try:
        model = train_independent_gps(train_x_scaled, train_y_standardized)
        model.eval()
    except Exception as e:
        print(f"  Skipping iteration due to GP training error: {e}")
        hypervolume_history.append(hypervolume_history[-1] if hypervolume_history else 0.0)
        selection_methods.append('Error')
        continue

    print("  Setting up EHVI acquisition function...")
    acqf = None
    try:
        if train_y_standardized.shape[0] > 0: #make sure train_y_standardized is not empty
             partitioning = FastNondominatedPartitioning( #takes the current best points so far and slices up the potential improvement region relative to the reference point so that the EHVI calculation is faster
                 ref_point=hv_ref_point_std_neg, # 3D std neg ref point
                 Y=train_y_standardized # 3D std neg objectives
             )
        else: # should not happen
             print("Warning: No training data for EHVI partitioning...")
             continue
        
        #define acquisition
        acqf = ExpectedHypervolumeImprovement(model=model, ref_point=hv_ref_point_std_neg.tolist(), partitioning=partitioning)

    except Exception as e:
        print(f"  Error creating EHVI: {e}. Falling back to scalarized LCB.")
        hypervolume_history.append(hypervolume_history[-1] if hypervolume_history else 0.0)
        next_x_scaled = None #trigger fallback

    if acqf is not None:
        print("  Optimizing acquisition function (EHVI)...")
        next_x_scaled = None
        acq_value = None
        try:
            next_x_scaled, acq_value = optimize_acqf(
                acq_function=acqf, bounds=scaled_bounds, q=1,
                num_restarts=20, raw_samples=3072, 
                options={"batch_limit": 5, "maxiter": 150}
            )
            next_x_unscaled = torch.tensor(x_scaler.inverse_transform(next_x_scaled.numpy()), dtype=torch.float64) #go back to initial objective space
            print(f"  Acquisition function optimized. Max value: {acq_value.item():.4f}")
        except Exception as e:
            print(f"  Warning: Acquisition function optimization failed: {e}. Falling back to scalarized LCB.")
            next_x_scaled = None

    # Find closest candidate
    closest_cand_X_scaled = None
    closest_idx = -1
    candidate_already_evaluated = False
    next_y_obj1_raw = None
    next_y_obj2_raw = None
    next_y_obj3_raw = None 

    if next_x_scaled is not None: # EHVI path
        closest_idx, _, next_y_obj1_raw, next_y_obj2_raw, next_y_obj3_raw = find_closest_candidate(
            next_x_unscaled, X_candidates, Y_candidates, objective_index_1, objective_index_2, objective_index_3
        )
        if closest_idx in original_indices.tolist(): #if candidate has already been evaluated
            print(f"   Suggested candidate (Index {closest_idx}) already evaluated by EHVI. Falling back to LCB.")
            candidate_already_evaluated = True
            next_x_scaled = None #trigger fallback
        else:
            closest_cand_X = X_candidates[closest_idx].unsqueeze(0)
            closest_cand_X_scaled = torch.tensor(x_scaler.transform(closest_cand_X.numpy()), dtype=torch.float64)
            # fallback_used_this_iter remains False
    else:
        fallback_used_this_iter = True

    # Fallback LCB
    if next_x_scaled is None:
        fallback_used_this_iter = True #mark fallback flag as true
        if acqf is None: 
            print("   Executing fallback because EHVI creation failed.")
        elif not candidate_already_evaluated: 
            print("   Executing fallback (optimize_acqf failed): Scalarized LCB.")

        unevaluated_indices_all = [idx for idx in range(len(X_candidates)) if idx not in original_indices.tolist()]

        if not unevaluated_indices_all: 
            print("   No unevaluated candidates left. Terminating early.")
            break

        unevaluated_indices_tensor = torch.tensor(unevaluated_indices_all, dtype=torch.long)
        
        with torch.no_grad():
            # USE ALL UNEVALUATED POINTS:
            unevaluated_X_scaled_fallback = X_candidates_scaled[unevaluated_indices_tensor]
            print(f"   Calculating posterior for {len(unevaluated_X_scaled_fallback)} LCB fallback candidates...")

            posterior = model.posterior(unevaluated_X_scaled_fallback) #gp posterior has 3 outputs
            means_standardized = posterior.mean; variances_standardized = posterior.variance
            stds_standardized = variances_standardized.sqrt()

        # Calculate LCB for 3 objectives
        mean_obj1_std = means_standardized[:, 0]; std_obj1_std = stds_standardized[:, 0]
        mean_obj2_std = means_standardized[:, 1]; std_obj2_std = stds_standardized[:, 1]
        mean_obj3_std = means_standardized[:, 2]; std_obj3_std = stds_standardized[:, 2] 

        beta_lcb = 1.5 #exploration/exploitation parameter (higher => exploration)
        w1, w2, w3 = 1/3., 1/3., 1/3. # equal weights for the 3 objectives

        scalarized_lcb_std = (w1 * (mean_obj1_std - beta_lcb * std_obj1_std) +
                              w2 * (mean_obj2_std - beta_lcb * std_obj2_std) +
                              w3 * (mean_obj3_std - beta_lcb * std_obj3_std)) # sum of the 3 lcbs

        best_lcb_idx_in_subset = torch.argmax(scalarized_lcb_std)
        closest_idx = unevaluated_indices_tensor[best_lcb_idx_in_subset].item()

        # Get data for the selected candidate
        closest_cand_X = X_candidates[closest_idx].unsqueeze(0)
        closest_cand_X_scaled = torch.tensor(x_scaler.transform(closest_cand_X.numpy()), dtype=torch.float64)

        raw_obj1 = Y_candidates[closest_idx, objective_index_1].item()
        raw_obj2 = Y_candidates[closest_idx, objective_index_2].item()
        raw_obj3 = Y_candidates[closest_idx, objective_index_3].item()

        next_y_obj1_raw = torch.tensor([[ -raw_obj1]])
        next_y_obj2_raw = torch.tensor([[ -raw_obj2]])
        next_y_obj3_raw = torch.tensor([[ -raw_obj3]]) 
        print(f"   Selected candidate (Index {closest_idx}) via scalarized LCB. LCB value: {scalarized_lcb_std[best_lcb_idx_in_subset].item():.4f}")

    # track Selection Method
    selection_methods.append('LCB' if fallback_used_this_iter else 'EHVI')

    # Update training data 
    print(f"   Adding Candidate Index: {closest_idx} (Selected via {selection_methods[-1]})")
    print(f"   Raw Objective 1 ({objective_target_name_1}): {-next_y_obj1_raw.item():.4f}")
    print(f"   Raw Objective 2 ({objective_target_name_2}): {-next_y_obj2_raw.item():.4f}")
    print(f"   Raw Objective 3 ({objective_target_name_3}): {-next_y_obj3_raw.item():.6f}") 
    next_y_combined_raw = torch.cat([next_y_obj1_raw, next_y_obj2_raw, next_y_obj3_raw], dim=1) # combine the 3 objectives into 1 tensor
    next_y_standardized = torch.tensor(y_scaler.transform(next_y_combined_raw.numpy()), dtype=torch.float64) # standardize it
    train_x_scaled = torch.cat([train_x_scaled, closest_cand_X_scaled], dim=0)
    train_y_standardized = torch.cat([train_y_standardized, next_y_standardized], dim=0) # add 3-column y
    original_indices = torch.cat([original_indices, torch.tensor([closest_idx], dtype=torch.long)])

    # --- Update and Record Hypervolume ---
    all_y_raw_negated = torch.tensor(y_scaler.inverse_transform(train_y_standardized.numpy()), dtype=torch.float64)
    all_objectives_raw = -all_y_raw_negated[:, obj_indices] # (n_eval, 3)
    current_hv = 0.0
    pareto_front_raw = torch.empty((0, n_objectives), dtype=all_objectives_raw.dtype)
    try:
        if all_objectives_raw.shape[0] > 0:
            non_dominated_mask = simple_is_non_dominated(all_objectives_raw) 
            pareto_front_raw = all_objectives_raw[non_dominated_mask] # (n_pareto, 3)
            if pareto_front_raw.shape[0] > 0:
                negated_pareto_front = -pareto_front_raw
                negated_ref_point = -hv_ref_point_raw
                hv_calculator = Hypervolume(ref_point=negated_ref_point) 
                current_hv = hv_calculator.compute(negated_pareto_front) 
    except Exception as e:
        print(f"  Warning: Hypervolume calculation failed: {e}. Using previous value.")
        current_hv = hypervolume_history[-1] if hypervolume_history else 0.0

    hypervolume_history.append(current_hv)
    print(f"   Current Hypervolume: {current_hv:.6f}") 
    print(f"   Pareto Front Size: {pareto_front_raw.shape[0]}")
    print(f"   Current number of unique points evaluated: {len(original_indices)}")

    # --- STOPPING CRITERION LOGIC ---
    if i >= BASE_ITERS - 1:
        if len(hypervolume_history) > PATIENCE_WINDOW:
            past_hv = hypervolume_history[-(PATIENCE_WINDOW + 1)]
            current_hv_check = hypervolume_history[-1]
            
            if abs(past_hv) > 1e-9:
                relative_improvement = (current_hv_check - past_hv) / abs(past_hv)
                print(f"   [Stopping Check] Rel. HV improvement over last {PATIENCE_WINDOW} iters: {relative_improvement:.2e}")
                if relative_improvement < REL_IMPROVEMENT_TOL:
                    print(f"\nSTOPPING: Relative hypervolume improvement ({relative_improvement:.2e}) is below tolerance ({REL_IMPROVEMENT_TOL:.1e}) over the last {PATIENCE_WINDOW} iterations.")
                    break
            elif (current_hv_check - past_hv) < 1e-7:
                print(f"\nSTOPPING: Absolute hypervolume improvement is negligible over the last {PATIENCE_WINDOW} iterations (past HV was near zero).")
                break
else:
    print(f"\nSTOPPING: Reached maximum total iterations ({MAX_TOTAL_ITERS}).")


# Stop BO timer and calculate BO duration
end_bo_time = time.monotonic()
bo_duration_seconds = end_bo_time - start_bo_time

# --- Results ---
print("\nOptimization finished.")
print(f"Ran for {actual_iterations_run} iterations.")
print(f"Total BO loop duration: {bo_duration_seconds:.2f} seconds ({bo_duration_seconds/60:.2f} minutes)")

# --- Final Pareto Front from BO ---
final_y_raw_negated = torch.tensor(y_scaler.inverse_transform(train_y_standardized.numpy()), dtype=torch.float64)
final_objectives_raw = -final_y_raw_negated[:, obj_indices] # (n_total_eval, 3)
final_pareto_points_raw = None; final_pareto_features_unscaled = None
print("\n--- Calculating Final BO Pareto Front ---")
try:
    if final_objectives_raw.shape[0] > 0:
        non_dominated_mask_final = simple_is_non_dominated(final_objectives_raw)
        final_pareto_points_raw = final_objectives_raw[non_dominated_mask_final] # (n_final_pareto, 3)
        print(f"Found {final_pareto_points_raw.shape[0]} points using simple_is_non_dominated on final BO results.")

        if final_pareto_points_raw.shape[0] > 0:
            final_indices_in_train = torch.where(non_dominated_mask_final)[0]
            final_pareto_features_scaled = train_x_scaled[final_indices_in_train] 
            final_pareto_features_unscaled = torch.tensor(x_scaler.inverse_transform(final_pareto_features_scaled.numpy()), dtype=torch.float64)
            print(f"\nFound {len(final_pareto_points_raw)} non-dominated points (Pareto front) by BO:")
            print("Objectives (Raw):")
            df_bo_pareto = pd.DataFrame(final_pareto_points_raw.numpy(), columns=objective_names) 
            print(df_bo_pareto)
        else: 
            print("\nNo non-dominated points found in BO results.")
    else: 
        print("\nNo points evaluated to determine final BO Pareto front.")
except Exception as e: print(f"\nError calculating final Pareto front from BO results: {e}")

# --- Determine True Pareto Front from Initial + Candidate Data (using Simple Check) ---
print("\n--- Determining True Pareto Front from Initial + Candidate Data (using Simple Check) ---")
Y_init_objectives_raw = Y_init[:, [objective_index_1, objective_index_2, objective_index_3]] 
Y_candidates_objectives_raw = Y_candidates[:, [objective_index_1, objective_index_2, objective_index_3]] 
true_space_objectives = torch.cat([Y_init_objectives_raw, Y_candidates_objectives_raw], dim=0)
print(f"Combined true space objectives shape (Init + Candidates): {true_space_objectives.shape}")
unique_true_space_objectives = None
try:
    precision_dup = 1e5
    df_true_space = pd.DataFrame(true_space_objectives.numpy(), columns=['o1', 'o2', 'o3']) # 3 columns
    df_true_space['r1'] = np.round(df_true_space['o1'] * precision_dup) / precision_dup
    df_true_space['r2'] = np.round(df_true_space['o2'] * precision_dup) / precision_dup
    df_true_space['r3'] = np.round(df_true_space['o3'] * precision_dup) / precision_dup 
    unique_true_space_df = df_true_space.drop_duplicates(subset=['r1', 'r2', 'r3']) # 3 keys
    unique_true_space_objectives = torch.tensor(unique_true_space_df[['o1', 'o2', 'o3']].values, dtype=torch.float64) # select 3 columns
    print(f"Combined unique true space objective points after duplicate removal: {unique_true_space_objectives.shape[0]}")
except Exception as e: print(f"Warning: Could not remove duplicates, using all. Error: {e}"); unique_true_space_objectives = true_space_objectives
true_pareto_front_raw = None; true_hv = 0.0
try:
    if unique_true_space_objectives.shape[0] > 0:
        print(f"DEBUG: Calculating non-dominated using simple pairwise check (N={unique_true_space_objectives.shape[0]})...")
        start_time = time.time()
        non_dominated_mask_true = simple_is_non_dominated(unique_true_space_objectives) 
        true_pareto_front_raw = unique_true_space_objectives[non_dominated_mask_true] # (n_true_pareto, 3)
        end_time = time.time()
        print(f"Simple check duration: {end_time - start_time:.2f} seconds")
        print(f"Found {true_pareto_front_raw.shape[0]} points in True Pareto front using simple check.")
        if true_pareto_front_raw.shape[0] > 0:
            df_true_pareto = pd.DataFrame(true_pareto_front_raw.numpy(), columns=objective_names); print("True Pareto Front (Raw - Simple Check):"); print(df_true_pareto)
            negated_true_pareto = -true_pareto_front_raw # Shape (n_true_pareto, 3)
            negated_ref_point = -hv_ref_point_raw # Shape (3,)
            hv_calculator_true = Hypervolume(ref_point=negated_ref_point) 
            true_hv = hv_calculator_true.compute(negated_true_pareto) 
            print(f"True Maximum Hypervolume (Simple Check): {true_hv:.6f}") 
            final_bo_hv = hypervolume_history[-1] if hypervolume_history else 0.0
            print(f"Final BO Hypervolume (from history): {final_bo_hv:.6f}") 
            if true_hv > 1e-12: # tolerance
                hv_ratio = final_bo_hv / true_hv
                print(f"Final BO Hypervolume / True Max Hypervolume = {hv_ratio:.4f}")
                if hv_ratio > 1.0001: print("ERROR: Calculated BO HV is still significantly larger than True HV!")
            else: print("Cannot calculate hypervolume ratio (true max HV is near zero).")
        else: print("\nTrue Pareto front determined to be empty (Simple Check)."); true_hv = 0.0
    else: print("\nNo unique combined points (Init+Candidates) found."); true_hv = 0.0
except Exception as e: print(f"\nError calculating true Pareto front using simple check: {e}"); true_hv = 0.0


# Check Overlap Between BO Front and True Front 
print("\n--- Checking Overlap Between BO Front and True Front ---")
if final_pareto_points_raw is not None and true_pareto_front_raw is not None and \
   final_pareto_points_raw.shape[0] > 0 and true_pareto_front_raw.shape[0] > 0:

    precision_comp = 1e5 
    rounded_bo_pareto = torch.round(final_pareto_points_raw * precision_comp) / precision_comp
    rounded_true_pareto = torch.round(true_pareto_front_raw * precision_comp) / precision_comp

    true_points_set = set(tuple(point.cpu().tolist()) for point in rounded_true_pareto)

    match_count = 0
    for point in rounded_bo_pareto: 
        if tuple(point.cpu().tolist()) in true_points_set:
            match_count += 1

    print(f"\nOverlap Check Results:")
    print(f"  - BO found {final_pareto_points_raw.shape[0]} Pareto points.")
    print(f"  - True Pareto Front contains {true_pareto_front_raw.shape[0]} points.")
    print(f"  - {match_count} out of {final_pareto_points_raw.shape[0]} BO Pareto points exactly match points on the True Pareto Front (based on rounded values).")

elif final_pareto_points_raw is None or final_pareto_points_raw.shape[0] == 0:
    print("\nCannot check overlap: Final BO Pareto Front is empty.")
else: # true_pareto_front_raw is None or empty
     print("\nCannot check overlap: True Pareto Front is empty.")


# --- Plotting ---

# 1. Hypervolume Convergence Plot
plt.figure(figsize=(10, 6)); plt.plot(range(len(hypervolume_history)), hypervolume_history, marker='o', linestyle='-', label='BO Hypervolume')
if true_pareto_front_raw is not None and true_pareto_front_raw.shape[0] > 0 and true_hv > 1e-9: 
    plt.axhline(y=true_hv, color='r', linestyle='--', label=f'True Max HV ({true_hv:.4f})')
ax1 = plt.gca()
handles1, labels1 = ax1.get_legend_handles_labels();
if handles1: 
    ax1.legend()
plt.xlabel("Iteration (0 = Initial)")
plt.ylabel("Hypervolume")
plt.title("Hypervolume Convergence Plot (3 Objectives)")
plt.grid(True)
plt.ylim(bottom=0)
plt.tight_layout()
plt.savefig('hypervolume_convergence_3obj.png')
plt.show()

# 2. Selection Method Plot
plt.figure(figsize=(12, 6))
if len(selection_methods) > 0:
    bo_iterations = list(range(1, len(selection_methods) + 1))
    if len(hypervolume_history) == len(selection_methods) + 1: bo_hypervolumes = hypervolume_history[1:]
    else: print("Warning: Mismatch between hypervolume history and selection methods."); bo_hypervolumes = []
    if len(bo_iterations) == len(bo_hypervolumes):
        plt.plot(bo_iterations, bo_hypervolumes, color='darkgrey', linestyle='-', zorder=1, alpha=0.7, label='_nolegend_')
        color_map = {'EHVI': '#E41A1C', 'LCB': '#377EB8', 'Error': 'grey'}
        plot_colors = [color_map.get(method, 'grey') for method in selection_methods]
        plt.scatter(bo_iterations, bo_hypervolumes, c=plot_colors, marker='o', s=50, zorder=2, edgecolors='grey', alpha=0.9)
    else: print("Warning: Mismatch between BO iterations and hypervolumes for plotting breakdown.")
if true_pareto_front_raw is not None and true_pareto_front_raw.shape[0] > 0 and true_hv > 1e-9: 
    plt.axhline(y=true_hv, color='green', linestyle='--', label=f'True Max HV ({true_hv:.4f})')
legend_elements = [ Line2D([0], [0], marker='o', color='w', label='Selected via EHVI', markerfacecolor='#E41A1C', markeredgecolor='grey', markersize=8), Line2D([0], [0], marker='o', color='w', label='Selected via LCB Fallback', markerfacecolor='#377EB8', markeredgecolor='grey', markersize=8)]
if true_pareto_front_raw is not None and true_pareto_front_raw.shape[0] > 0 and true_hv > 1e-9: 
    legend_elements.append(Line2D([0], [0], color='green', linestyle='--', label=f'True Max HV ({true_hv:.4f})'))
plt.legend(handles=legend_elements, loc='lower right', fontsize='medium')
plt.xlabel("BO Iteration Number", fontsize=12)
plt.ylabel("Current Hypervolume", fontsize=12)
plt.title("BO Hypervolume Progression by Selection Method (3 Objectives)", fontsize=14)
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.6)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.xlim(left=0)
plt.ylim(bottom=0)
plt.tight_layout()
plt.savefig('selection_method_plot_3obj.png')
plt.show()

# 3. 3D Objective Space Plot
print("\nGenerating 3D Objective Space Plot...")
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Get data for plotting (use final raw objectives)
if final_objectives_raw is not None and final_objectives_raw.shape[0] > 0:
    obj1_all = final_objectives_raw[:, 0].numpy()
    obj2_all = final_objectives_raw[:, 1].numpy()
    obj3_all = final_objectives_raw[:, 2].numpy() 

    # Plot all evaluated points
    ax.scatter(obj1_all, obj2_all, obj3_all,
               c='blue', alpha=0.4, s=15, label='Evaluated Points (Other)')

    # Plot BO Pareto front points
    if final_pareto_points_raw is not None and final_pareto_points_raw.shape[0] > 0:
        ax.scatter(final_pareto_points_raw[:, 0].numpy(),
                   final_pareto_points_raw[:, 1].numpy(),
                   final_pareto_points_raw[:, 2].numpy(),
                   c='lime', s=150, edgecolor='black', marker='*', label='BO Pareto Front', depthshade=False, zorder=3) 

    # Plot True Pareto front points
    if true_pareto_front_raw is not None and true_pareto_front_raw.shape[0] > 0:
         ax.scatter(true_pareto_front_raw[:, 0].numpy(),
                   true_pareto_front_raw[:, 1].numpy(),
                   true_pareto_front_raw[:, 2].numpy(),
                   facecolors='none', edgecolors='red', marker='o', s=60, linewidth=1.5, label='True Pareto Front (Simple Check)', zorder=2) 

    ax.set_xlabel(f"{objective_names[0]} (Minimize)")
    ax.set_ylabel(f"{objective_names[1]} (Minimize)")
    ax.set_zlabel(f"{objective_names[2]} (Minimize)")
    ax.set_title("3D Objective Space - Pareto Front Comparison")
    ax.legend(loc='best')
    plt.tight_layout()
    plt.savefig('objective_space_3d.png')
    plt.show()
else:
    print("\nNo points evaluated to plot 3D objective space.")

print("\nFinal training data size (scaled features):", train_x_scaled.shape)
print("\nFinal training data size (standardized outputs):", train_y_standardized.shape) #3 columns
print(f"Total points in hypervolume_history plot: {len(hypervolume_history)}")
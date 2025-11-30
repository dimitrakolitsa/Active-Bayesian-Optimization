#LogCEI + LCB fallback + sklearn scalers

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
from matplotlib.lines import Line2D # For custom legend


warnings.filterwarnings('ignore', category=InputDataWarning)

torch.manual_seed(42)
np.random.seed(42) # seed numpy too
torch.set_default_dtype(torch.float64)

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
init_df_targets = pd.concat(init_target_dfs, axis=1)
init_df_targets.columns = target_column_names
df_init = pd.concat([df_init, init_df_targets], axis=1)

all_candidates_target_dfs = [pd.read_csv(file, header=None, sep=',') for file in all_candidates_target_files]
all_candidates_df_targets = pd.concat(all_candidates_target_dfs, axis=1)
all_candidates_df_targets.columns = target_column_names
df_all_candidates = pd.concat([df_all_candidates, all_candidates_df_targets], axis=1)


X_init = torch.tensor(df_init[new_columns].values, dtype=torch.float64)
Y_init = torch.tensor(df_init[target_column_names].values, dtype=torch.float64)
X_candidates = torch.tensor(df_all_candidates[new_columns].values, dtype=torch.float64)
Y_candidates = torch.tensor(df_all_candidates[target_column_names].values, dtype=torch.float64)


# --- Single objective constrained BO ---
objective_target_name = "max_stress_torsion"
constraint_target_name = "mass"
constraint_threshold = 0.018

objective_index = target_column_names.index(objective_target_name)
constraint_index = target_column_names.index(constraint_target_name)

# Combine objective and constraint into a single training Y tensor
train_y_raw = torch.cat([
    -Y_init[:, objective_index].unsqueeze(-1),
    Y_init[:, constraint_index].unsqueeze(-1)
], dim=1)

print(f"\nOptimizing objective: Minimize {objective_target_name}")
print(f"Subject to constraint: {constraint_target_name} < {constraint_threshold}")

# --- Feature Scaling (MinMaxScaler) ---
X_combined = torch.cat([X_init, X_candidates], dim=0)
x_scaler = MinMaxScaler()
x_scaler.fit(X_combined.numpy())
X_init_scaled = torch.tensor(x_scaler.transform(X_init.numpy()), dtype=torch.float64)
X_candidates_scaled = torch.tensor(x_scaler.transform(X_candidates.numpy()), dtype=torch.float64)
scaled_bounds = torch.tensor([[0.0] * X_init.shape[-1], [1.0] * X_init.shape[-1]], dtype=torch.float64)

# --- Target Y Standardization (with StandardScaler) ---
y_scaler = StandardScaler()
y_scaler.fit(train_y_raw.numpy())
train_y_standardized = torch.tensor(y_scaler.transform(train_y_raw.numpy()), dtype=torch.float64)
constraint_threshold_standardized = y_scaler.transform(torch.tensor([[0.0, constraint_threshold]], dtype=torch.float64))[:, 1].item()


# --- multi-output GP Model Training  ---
def train_multi_output_gp_model(train_x, train_y):
    model = SingleTaskGP(
        train_x,
        train_y,
        covar_module=ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=train_x.shape[-1])))
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    try:
        with gpytorch.settings.cholesky_jitter(1e-4):
            fit_gpytorch_mll(mll, max_retries=5, options={'maxiter': 150})
    except Exception as e:
        print(f"Warning: GP fitting failed: {e}. Model parameters might not be optimal.")
    return model

# --- Candidate Selection Function ---
def find_closest_candidate(x_proposed_unscaled, X_cands, Y_cands, objective_idx, constraint_idx):
    distances = torch.norm(X_cands - x_proposed_unscaled.squeeze(0), dim=1)
    closest_index = torch.argmin(distances)
    closest_X = X_cands[closest_index].unsqueeze(0)
    scalar_objective_value = -Y_cands[closest_index, objective_idx]
    closest_Y_objective_raw = scalar_objective_value.view(1, 1)
    scalar_constraint_value = Y_cands[closest_index, constraint_idx]
    closest_Y_constraint_raw = scalar_constraint_value.view(1, 1)
    return closest_index.item(), closest_X, closest_Y_objective_raw, closest_Y_constraint_raw


# --- BO loop Setup ---
# --- Stopping Criterion Configuration ---
BASE_ITERS = 30
MAX_TOTAL_ITERS = 100
PATIENCE_WINDOW = 10
REL_IMPROVEMENT_TOL = 1e-5 # Stop if relative improvement is less than 0.001%

# --- BO State Initialization ---
best_feasible_values = []
selection_methods = []
original_indices = torch.tensor(df_init.index.tolist(), dtype=torch.long)
actual_iterations_run = 0

# initialize training data
train_x_scaled = X_init_scaled.clone()
train_y_standardized = train_y_standardized.clone()

# --- Initial Best Feasible Value ---
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

print(f"\nStarting Bayesian Optimization...")
print(f"Total candidates: {len(X_candidates)}, Initial points: {len(X_init)}")
print(f"Will run for at least {BASE_ITERS} iterations, up to a max of {MAX_TOTAL_ITERS}.")
print(f"Will stop if relative improvement is less than {REL_IMPROVEMENT_TOL:.1e} over a {PATIENCE_WINDOW}-iteration window (after base run).")

#start timer
bo_start_time = time.perf_counter()


# --- BO Loop ---
for i in range(MAX_TOTAL_ITERS):
    actual_iterations_run = i + 1
    print(f"\nIteration {i+1}/{MAX_TOTAL_ITERS}")

    # Check if we should even run this iteration (handles forced base runs)
    if i < BASE_ITERS:
        print(f"  (Running guaranteed base iteration {i+1}/{BASE_ITERS})")
    
    print("  Training Multi-Output GP...")
    try:
        model = train_multi_output_gp_model(train_x_scaled, train_y_standardized)
        model.eval()
    except ValueError as e:
        print(f"  Skipping iteration due to invalid data: {e}")
        continue

    # --- Calculate Best Feasible Value (Standardized) for LogCEI ---
    current_objective_values_raw = -y_scaler.inverse_transform(train_y_standardized.numpy())[:, 0]
    current_constraint_values_raw = y_scaler.inverse_transform(train_y_standardized.numpy())[:, 1]
    feasible_mask = current_constraint_values_raw < constraint_threshold

    if feasible_mask.any():
        best_feasible_raw_objective = current_objective_values_raw[feasible_mask].min().item()
        current_best_feasible_neg_objective_standardized = y_scaler.transform(torch.tensor([[-best_feasible_raw_objective, 0.0]], dtype=torch.float64))[:, 0].item()
    else:
        print("  Warning: No feasible points observed yet. Using a fallback best_f for LogCEI.")
        current_best_feasible_neg_objective_standardized = train_y_standardized[:, 0].min().item() - 10.0

    print(f"  Current best feasible value (negated, std) for LogCEI: {current_best_feasible_neg_objective_standardized:.4f}")

    # --- Define Acquisition Function ---
    constraints = {1: (None, constraint_threshold_standardized)}
    try:
        LogCEI = LogConstrainedExpectedImprovement(
            model=model,
            best_f=current_best_feasible_neg_objective_standardized,
            objective_index=0,
            constraints=constraints,
            maximize=True
        )
    except Exception as e:
        print(f"  Error creating LogConstrainedExpectedImprovement: {e}. Skipping optimization step.")
        continue

    # --- Optimize Acquisition Function ---
    print("  Optimizing acquisition function (LogCEI)...")
    try:
        next_x_scaled, acq_value = optimize_acqf(
            LogCEI,
            bounds=scaled_bounds,
            q=1,
            num_restarts=15,
            raw_samples=4000,
            options={"batch_limit": 5, "maxiter": 150}
        )
        next_x_unscaled = torch.tensor(x_scaler.inverse_transform(next_x_scaled.numpy()), dtype=torch.float64)
    except Exception as e:
        print(f"  Warning: Acquisition function optimization failed: {e}. Falling back to LCB.")
        next_x_scaled = None

    # --- Find Closest Candidate and Handle Fallback ---
    closest_cand_X = None
    closest_cand_X_scaled = None
    next_y_objective_raw = None
    next_y_constraint_raw = None
    closest_idx = -1
    selection_method = "LogCEI"

    if next_x_scaled is not None:
        closest_idx, closest_cand_X, next_y_objective_raw, next_y_constraint_raw = find_closest_candidate(
            next_x_unscaled, X_candidates, Y_candidates, objective_index, constraint_index
        )
        if closest_idx in original_indices:
            print(f"   Suggested candidate (Index {closest_idx}) already evaluated. Selecting best *feasible* unevaluated point via LCB.")
            next_x_scaled = None
        else:
            closest_cand_X_scaled = torch.tensor(x_scaler.transform(closest_cand_X.numpy()), dtype=torch.float64)

    # --- Fallback: LCB on feasible unevaluated candidates ---
    if next_x_scaled is None:
        selection_method = "LCB (fallback)"
        unevaluated_indices_all = [idx for idx in range(len(X_candidates)) if idx not in original_indices]
        if not unevaluated_indices_all:
            print("   No unevaluated candidates left. Terminating early.")
            break
        unevaluated_masses_raw = Y_candidates[unevaluated_indices_all, constraint_index]
        feasible_unevaluated_mask = unevaluated_masses_raw < constraint_threshold
        feasible_unevaluated_indices = torch.tensor(unevaluated_indices_all, dtype=torch.long)[feasible_unevaluated_mask]
        if len(feasible_unevaluated_indices) == 0:
            print("   No *feasible* unevaluated candidates left. Selecting best unevaluated based on predicted LCB (ignoring feasibility for selection).")
            feasible_unevaluated_indices = torch.tensor(unevaluated_indices_all, dtype=torch.long)
            if len(feasible_unevaluated_indices) == 0:
                print("   Error: No unevaluated candidates found in fallback.")
                break
        with torch.no_grad():
            unevaluated_X_scaled = X_candidates_scaled[feasible_unevaluated_indices]
            posterior = model.posterior(unevaluated_X_scaled)
            predicted_means_standardized = posterior.mean[:, 0]
            predicted_stds_standardized = posterior.variance[:, 0].sqrt()
        beta_lcb = 2.0
        lcb_std = predicted_means_standardized - beta_lcb * predicted_stds_standardized
        best_lcb_idx_in_filtered = torch.argmax(lcb_std)
        closest_idx = feasible_unevaluated_indices[best_lcb_idx_in_filtered].item()
        closest_cand_X = X_candidates[closest_idx].unsqueeze(0)
        closest_cand_X_scaled = torch.tensor(x_scaler.transform(closest_cand_X.numpy()), dtype=torch.float64)
        next_y_objective_raw = -Y_candidates[closest_idx, objective_index].view(1, 1)
        next_y_constraint_raw = Y_candidates[closest_idx, constraint_index].view(1, 1)
        print(f"   Selected candidate (Index {closest_idx}) via LCB: {lcb_std[best_lcb_idx_in_filtered].item():.4f}")

    # --- Update Training Data ---
    next_y_combined_raw = torch.cat([next_y_objective_raw, next_y_constraint_raw], dim=1)
    next_y_standardized = torch.tensor(y_scaler.transform(next_y_combined_raw.numpy()), dtype=torch.float64)
    train_x_scaled = torch.cat([train_x_scaled, closest_cand_X_scaled], dim=0)
    train_y_standardized = torch.cat([train_y_standardized, next_y_standardized], dim=0)
    original_indices = torch.cat([original_indices, torch.tensor([closest_idx], dtype=torch.long)])

    # --- Update and Record Best Feasible Value ---
    current_best_objective_raw = best_feasible_values[-1]
    current_objective_val_raw = -next_y_objective_raw.item()
    current_constraint_val_raw = next_y_constraint_raw.item()
    print(f"   Adding Candidate Index: {closest_idx} (Selected via {selection_method})")
    print(f"   Raw Objective Value ({objective_target_name}): {current_objective_val_raw:.6f}")
    print(f"   Raw Constraint Value ({constraint_target_name}): {current_constraint_val_raw:.6f}")
    if current_constraint_val_raw < constraint_threshold:
        if current_objective_val_raw < current_best_objective_raw:
            current_best_objective_raw = current_objective_val_raw
    best_feasible_values.append(current_best_objective_raw)
    selection_methods.append(selection_method)
    print(f"   Best feasible value so far: {current_best_objective_raw:.6f}")
    print(f"   Current number of unique points evaluated: {len(original_indices)}")

    # --- STOPPING CRITERION LOGIC ---
    if i >= BASE_ITERS - 1:
        if len(best_feasible_values) > PATIENCE_WINDOW and best_feasible_values[-(PATIENCE_WINDOW + 1)] != float('inf'):
            past_best = best_feasible_values[-(PATIENCE_WINDOW + 1)]
            current_best = best_feasible_values[-1]
            if abs(past_best) > 1e-9:
                relative_improvement = (past_best - current_best) / abs(past_best)
                if relative_improvement < REL_IMPROVEMENT_TOL:
                    print(f"\nSTOPPING: Relative improvement ({relative_improvement:.2e}) is below tolerance ({REL_IMPROVEMENT_TOL}) over the last {PATIENCE_WINDOW} iterations.")
                    break
            elif (past_best - current_best) < 1e-7:
                 print(f"\nSTOPPING: Absolute improvement is negligible over the last {PATIENCE_WINDOW} iterations (past best was near zero).")
                 break

else:
    print(f"\nSTOPPING: Reached maximum total iterations ({MAX_TOTAL_ITERS}).")

#end timer
bo_end_time = time.perf_counter()
bo_duration = bo_end_time - bo_start_time

# --- Results ---
print("\nOptimization finished.")
print(f"Ran for {actual_iterations_run} iterations.")
print(f"\nBO loop finished in {bo_duration:.2f} seconds ({bo_duration/60:.2f} minutes).")

# Extract raw values from standardized outputs
final_y_raw = y_scaler.inverse_transform(train_y_standardized.numpy())
final_objective_values_raw = final_y_raw[:, 0]
final_mass_values_raw = final_y_raw[:, 1]
feasible_final_mask = final_mass_values_raw < constraint_threshold

if feasible_final_mask.any():
    final_objective_values_negated = -final_objective_values_raw
    best_bo_objective_raw = final_objective_values_negated[feasible_final_mask].min().item()
    best_idx_in_feasible_train = torch.argmin(torch.tensor(final_objective_values_raw[feasible_final_mask]))
    feasible_final_mask_tensor = torch.tensor(feasible_final_mask, dtype=torch.bool)
    original_best_idx_in_train_set = torch.where(feasible_final_mask_tensor)[0][best_idx_in_feasible_train]
    best_x_scaled = train_x_scaled[original_best_idx_in_train_set]
    best_x_unscaled = torch.tensor(x_scaler.inverse_transform(best_x_scaled.numpy().reshape(1, -1)), dtype=torch.float64).squeeze()
    best_mass_raw = final_mass_values_raw[original_best_idx_in_train_set].item()
    print(f"\nBest feasible value found by BO ({objective_target_name}): {best_bo_objective_raw:.6f}")
    print(f"   Constraint value ({constraint_target_name}): {best_mass_raw:.6f} (Constraint: < {constraint_threshold})")
    print(f"   Achieved at feature vector (unscaled) from BO:")
    best_features_dict = {name: val.item() for name, val in zip(new_columns, best_x_unscaled)}
    print(pd.Series(best_features_dict))
else:
    print("\nNo feasible solution found by BO.")
    best_bo_objective_raw = float('inf')

# --- Compare with Actual Best Feasible Candidate ---
global_min_objective_raw = Y_candidates[:, objective_index].min().item()
global_max_objective_raw = Y_candidates[:, objective_index].max().item()
print(f"\nOverall minimum {objective_target_name} from all candidates (unconstrained): {global_min_objective_raw:.6f}")
print(f"Overall maximum {objective_target_name} from all candidates (unconstrained): {global_max_objective_raw:.6f}")

all_candidate_masses_raw = Y_candidates[:, constraint_index]
feasible_candidate_mask = all_candidate_masses_raw < constraint_threshold
if feasible_candidate_mask.any():
    actual_min_feasible_objective_raw = Y_candidates[feasible_candidate_mask, objective_index].min().item()
    actual_max_feasible_objective_raw = Y_candidates[feasible_candidate_mask, objective_index].max().item()
    print(f"\nActual minimum feasible {objective_target_name} from all candidates: {actual_min_feasible_objective_raw:.6f}")
    print(f"Actual maximum feasible {objective_target_name} from all candidates: {actual_max_feasible_objective_raw:.6f}")
    if best_bo_objective_raw != float('inf'):
        percent_error = abs(best_bo_objective_raw - actual_min_feasible_objective_raw) / abs(actual_min_feasible_objective_raw) * 100
        print(f"Percentage error between BO result and actual minimum feasible: {percent_error:.2f}%")
    else:
        print("Cannot calculate percentage error as BO found no feasible solution.")
else:
    print("\nNo feasible candidates exist in the entire dataset.")
    actual_min_feasible_objective_raw = float('inf')

# --- Plotting ---
plt.figure(figsize=(12, 7))
iterations = list(range(len(best_feasible_values)))
plt.plot(iterations, best_feasible_values, marker='o', linestyle='-', label='BO Best Feasible Value')
if actual_min_feasible_objective_raw != float('inf'):
    plt.axhline(y=actual_min_feasible_objective_raw, color='r', linestyle='--', label=f'Actual Min Feasible ({actual_min_feasible_objective_raw:.6f})')
if best_bo_objective_raw != float('inf'):
    plt.axhline(y=best_bo_objective_raw, color='g', linestyle='--', label=f'BO Min Feasible ({best_bo_objective_raw:.6f})')
plt.xlabel("Iteration (0 = Initial Best Feasible)")
plt.ylabel(f"Best Observed Feasible Minimum {objective_target_name}")
plt.title(f"Constrained BO Progress - Minimize {objective_target_name} s.t. {constraint_target_name} < {constraint_threshold}")
plt.legend()
plt.grid(True)
y_min_limit = min(v for v in best_feasible_values if v != float('inf')) if any(v != float('inf') for v in best_feasible_values) else 0
plt.ylim(bottom=max(0, y_min_limit * 0.9))
plt.tight_layout()
plt.show()

print("\nFinal training data size (scaled features):", train_x_scaled.shape)
print("Final training data size (standardized outputs):", train_y_standardized.shape)
print(f"Total points in best_feasible_values plot: {len(best_feasible_values)}")

print(f"Number of BO selections via LogCEI: {selection_methods.count('LogCEI')}")
print(f"Number of BO selections via LCB Fallback: {selection_methods.count('LCB (fallback)')}")


#plot selection method through iterations
plt.figure(figsize=(14, 8))
iterations = list(range(len(best_feasible_values)))
num_bo_iters = len(selection_methods)
plt.plot(iterations, best_feasible_values, color='darkgrey', linestyle='-', zorder=1, alpha=0.8, label='_nolegend_')
if len(iterations) > 0:
    plt.scatter(iterations[0], best_feasible_values[0], c='black', marker='s', s=60, label='Initial Best', zorder=3, edgecolors='grey')
if num_bo_iters > 0:
    plot_bo_iterations = iterations[1 : num_bo_iters + 1]
    plot_bo_values = best_feasible_values[1 : num_bo_iters + 1]
    plot_colors = ['#E41A1C' if method == 'LogCEI' else '#377EB8' for method in selection_methods]
    plt.scatter(plot_bo_iterations, plot_bo_values, c=plot_colors, marker='o', s=60, zorder=2, edgecolors='grey')
hline_labels = {}
if actual_min_feasible_objective_raw != float('inf'):
    plt.axhline(y=actual_min_feasible_objective_raw, color='#FF7F00', linestyle=':', linewidth=2, label=f'Actual Min Feasible ({actual_min_feasible_objective_raw:.6f})', alpha=0.9)
    hline_labels['Actual Min Feasible'] = f'{actual_min_feasible_objective_raw:.6f}'
if best_bo_objective_raw != float('inf'):
    plt.axhline(y=best_bo_objective_raw, color='#4DAF4A', linestyle=':', linewidth=2, label=f'BO Min Feasible ({best_bo_objective_raw:.6f})', alpha=0.9)
    hline_labels['BO Min Feasible'] = f'{best_bo_objective_raw:.6f}'
legend_elements = [
    Line2D([0], [0], marker='s', color='w', label='Initial Best', markerfacecolor='black', markeredgecolor='grey', markersize=8),
    Line2D([0], [0], marker='o', color='w', label='LogCEI Selection', markerfacecolor='#E41A1C', markeredgecolor='grey', markersize=8),
    Line2D([0], [0], marker='o', color='w', label='LCB Fallback Selection', markerfacecolor='#377EB8', markeredgecolor='grey', markersize=8),
]
if 'Actual Min Feasible' in hline_labels:
     legend_elements.append(Line2D([0], [0], color='#FF7F00', linestyle=':', linewidth=2, label=f'Actual Min Feasible ({hline_labels["Actual Min Feasible"]})'))
if 'BO Min Feasible' in hline_labels:
     legend_elements.append(Line2D([0], [0], color='#4DAF4A', linestyle=':', linewidth=2, label=f'BO Min Feasible ({hline_labels["BO Min Feasible"]})'))
plt.legend(handles=legend_elements, loc='best', fontsize='medium')
plt.xlabel("Iteration (0 = Initial Best Feasible)", fontsize=12)
plt.ylabel(f"Best Observed Feasible Minimum {objective_target_name}", fontsize=12)
plt.title(f"Constrained BO Progress - Minimize {objective_target_name} s.t. {constraint_target_name} < {constraint_threshold}", fontsize=14)
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.6)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
y_plot_values = [v for v in best_feasible_values if v != float('inf')]
if actual_min_feasible_objective_raw != float('inf'):
    y_plot_values.append(actual_min_feasible_objective_raw)
if y_plot_values:
    y_min_limit = min(y_plot_values)
    y_max_limit = max(y_plot_values)
    y_range = max(y_max_limit - y_min_limit, 1e-6)
    plot_bottom = y_min_limit - y_range * 0.1
    plot_top = y_max_limit + y_range * 0.1
    plt.ylim(bottom=plot_bottom, top=plot_top)
else:
     plt.ylim(bottom=0)
plt.tight_layout()
plt.show()
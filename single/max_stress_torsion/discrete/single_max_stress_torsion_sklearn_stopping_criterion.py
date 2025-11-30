#lei + lcb fallback + sklearn scalers

import pandas as pd
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import LogExpectedImprovement
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
np.random.seed(42)
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


# --- Single Objective BO ---
target_to_optimize_name = "max_stress_torsion"
target_index_to_optimize = target_column_names.index(target_to_optimize_name)
# BO maximizes, so we negate our objective, which we want to minimize
train_y_raw_neg = -Y_init[:, target_index_to_optimize].unsqueeze(-1) 

print(f"\nOptimizing target: Minimize {target_to_optimize_name}")

# --- Scaling with Sklearn ---
# Scale X (features) to [0, 1]
X_combined = torch.cat([X_init, X_candidates], dim=0)
x_scaler = MinMaxScaler()
x_scaler.fit(X_combined.numpy())
X_init_scaled = torch.tensor(x_scaler.transform(X_init.numpy()), dtype=torch.float64)
X_candidates_scaled = torch.tensor(x_scaler.transform(X_candidates.numpy()), dtype=torch.float64)
scaled_bounds = torch.tensor([[0.0] * X_init.shape[-1], [1.0] * X_init.shape[-1]], dtype=torch.float64)

# Standardize Y (target) to have zero mean and unit variance
y_scaler = StandardScaler()
train_y_standardized = torch.tensor(y_scaler.fit_transform(train_y_raw_neg.numpy()), dtype=torch.float64)

# --- GP Model Training Function ---
def train_gp_model(train_x, train_y):
    model = SingleTaskGP(train_x, train_y,
                         covar_module=ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=train_x.shape[-1])))
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    try:
        with gpytorch.settings.cholesky_jitter(1e-4):
            fit_gpytorch_mll(mll, max_retries=5)
    except Exception as e:
        print(f"Warning: GP fitting failed: {e}")
    return model

def find_closest_candidate(x_proposed_unscaled, X_cands, Y_cands, target_idx):
    distances = torch.norm(X_cands - x_proposed_unscaled.squeeze(0), dim=1)
    closest_index = torch.argmin(distances).item()
    closest_X = X_cands[closest_index].unsqueeze(0)
    # Return the negated raw value for maximization
    closest_Y_raw_neg = -Y_cands[closest_index, target_idx].view(1, 1)
    return closest_index, closest_X, closest_Y_raw_neg

# --- BO loop Setup ---
# --- Stopping Criterion Configuration ---
BASE_ITERS = 100
MAX_TOTAL_ITERS = 200
PATIENCE_WINDOW = 10
REL_IMPROVEMENT_TOL = 1e-5 # Stop if relative improvement is less than 0.001%

# --- BO State Initialization ---
initial_best_raw = -train_y_raw_neg.max().item()
best_values = [initial_best_raw]

original_indices = torch.tensor(df_init.index.tolist(), dtype=torch.long)
train_x_scaled = X_init_scaled.clone()
train_y = train_y_standardized.clone() #shorter name for the loop
actual_iterations_run = 0

print(f"Initial best (minimum) target value: {initial_best_raw:.4f}")
print(f"\nStarting Bayesian Optimization...")
print(f"Total candidates: {len(X_candidates)}, Initial points: {len(X_init)}")
print(f"Will run for at least {BASE_ITERS} iterations, up to a max of {MAX_TOTAL_ITERS}.")
print(f"Will stop if relative improvement is less than {REL_IMPROVEMENT_TOL:.1e} over a {PATIENCE_WINDOW}-iteration window (after base run).")

bo_start_time = time.perf_counter()

# --- BO Loop ---
for i in range(MAX_TOTAL_ITERS):
    actual_iterations_run = i + 1
    print(f"\nIteration {i+1}/{MAX_TOTAL_ITERS}")
    if i < BASE_ITERS:
        print(f"  (Running guaranteed base iteration {i+1}/{BASE_ITERS})")

    model = train_gp_model(train_x_scaled, train_y)
    model.eval()

    current_best_value_std = train_y.max().item()
    LEI = LogExpectedImprovement(model=model, best_f=current_best_value_std)
    
    try:
        next_x_scaled, _ = optimize_acqf(
            LEI, 
            bounds=scaled_bounds, 
            q=1, 
            num_restarts=15, 
            raw_samples=5000, 
            options={"batch_limit": 5, "maxiter": 150}
        )
        next_x_unscaled = torch.tensor(x_scaler.inverse_transform(next_x_scaled.numpy()), dtype=torch.float64)
    except Exception as e:
        print(f"  Warning: Acquisition function optimization failed: {e}. Falling back to LCB.")
        next_x_scaled = None

    # Find closest candidate, handling acquisition function failure
    if next_x_scaled is not None:
        closest_idx, closest_cand_X, next_y_neg_raw = find_closest_candidate(
            next_x_unscaled, X_candidates, Y_candidates, target_index_to_optimize
        )
        if closest_idx in original_indices:
            print(f"   Suggested candidate (Index {closest_idx}) already evaluated. Falling back to UCB.")
            next_x_scaled = None # Trigger fallback
    
    # UCB Fallback logic
    if next_x_scaled is None:
        unevaluated_indices = torch.tensor([idx for idx in range(len(X_candidates)) if idx not in original_indices])
        if len(unevaluated_indices) == 0:
            print("   No unevaluated candidates left. Terminating early.")
            break
            
        with torch.no_grad():
            unevaluated_X_scaled = X_candidates_scaled[unevaluated_indices]
            posterior = model.posterior(unevaluated_X_scaled)
            # Maximize the Upper Confidence Bound (UCB = mean + beta * std) to find the most promising point
            beta = 2.0
            ucb = posterior.mean + beta * posterior.variance.sqrt()
            best_fallback_idx = torch.argmax(ucb)
            closest_idx = unevaluated_indices[best_fallback_idx].item()

        closest_cand_X = X_candidates[closest_idx].unsqueeze(0)
        next_y_neg_raw = -Y_candidates[closest_idx, target_index_to_optimize].view(1, 1)
        print(f"   Selected candidate (Index {closest_idx}) via UCB fallback.")

    print(f"   Adding Candidate Index: {closest_idx}. Raw Objective Value: {-next_y_neg_raw.item():.4f}")
    
    # Update data with the new point
    closest_cand_X_scaled = torch.tensor(x_scaler.transform(closest_cand_X.numpy()), dtype=torch.float64)
    next_y_standardized = torch.tensor(y_scaler.transform(next_y_neg_raw.numpy()), dtype=torch.float64)
    
    train_x_scaled = torch.cat([train_x_scaled, closest_cand_X_scaled], dim=0)
    train_y = torch.cat([train_y, next_y_standardized], dim=0)
    original_indices = torch.cat([original_indices, torch.tensor([closest_idx], dtype=torch.long)])
    
    # Update and record the best *actual* value found so far
    current_best_neg_raw = y_scaler.inverse_transform(train_y.numpy()).max()
    current_best_actual = -current_best_neg_raw
    best_values.append(current_best_actual)
    print(f"   Best value so far: {current_best_actual:.4f}")
    print(f"   Current number of unique points: {len(original_indices)}")

    # --- STOPPING CRITERION LOGIC ---
    if i >= BASE_ITERS - 1:
        if len(best_values) > PATIENCE_WINDOW:
            past_best = best_values[-(PATIENCE_WINDOW + 1)]
            current_best = best_values[-1]
            if abs(past_best) > 1e-9:
                relative_improvement = (past_best - current_best) / abs(past_best)
                if relative_improvement < REL_IMPROVEMENT_TOL:
                    print(f"\nSTOPPING: Relative improvement ({relative_improvement:.2e}) is below tolerance ({REL_IMPROVEMENT_TOL}) over the last {PATIENCE_WINDOW} iterations.")
                    break
            elif (past_best - current_best) < 1e-7: # Fallback for near-zero objectives
                 print(f"\nSTOPPING: Absolute improvement is negligible over the last {PATIENCE_WINDOW} iterations (past best was near zero).")
                 break
else:
    print(f"\nSTOPPING: Reached maximum total iterations ({MAX_TOTAL_ITERS}).")

bo_end_time = time.perf_counter()
bo_duration = bo_end_time - bo_start_time

# --- Results ---
print("\nOptimization finished.")
print(f"Ran for {actual_iterations_run} iterations.")
print(f"BO loop finished in {bo_duration:.2f} seconds ({bo_duration/60:.2f} minutes).")

best_actual_value = best_values[-1]
best_idx_in_train = torch.argmax(train_y) # Index in the standardized data
best_x_unscaled = torch.tensor(x_scaler.inverse_transform(train_x_scaled[best_idx_in_train].reshape(1, -1)), dtype=torch.float64).squeeze()

actual_min = df_all_candidates[target_to_optimize_name].min()
percent_error = abs(best_actual_value - actual_min) / actual_min * 100
actual_max = df_all_candidates[target_to_optimize_name].max()


print(f"\nBest observed value from BO ({target_to_optimize_name}): {best_actual_value:.6f}")
print(f"Actual maximum {target_to_optimize_name} from all candidates: {actual_max:.6f}")
print(f"Actual minimum {target_to_optimize_name} from all candidates: {actual_min:.6f}")
print(f"Percentage error between BO result and actual minimum: {percent_error:.2f}%")
print(f"Achieved at feature vector (unscaled) from BO:")
best_features_dict = {name: val.item() for name, val in zip(new_columns, best_x_unscaled)}
print(pd.Series(best_features_dict))

# --- Plotting ---
plt.figure(figsize=(10, 6))
iterations = list(range(len(best_values)))
plt.plot(iterations, best_values, marker='o', linestyle='-', label='BO Progress')
plt.axhline(y=actual_min, color='r', linestyle='--', label=f'Actual Minimum ({actual_min:.6f})')
plt.axhline(y=best_actual_value, color='g', linestyle='--', label=f'BO Minimum ({best_actual_value:.6f})')
plt.xlabel("Iteration (0 = Initial Data)")
plt.ylabel(f"Best Observed Minimum {target_to_optimize_name}")
plt.title(f"Bayesian Optimization Progress - Minimizing {target_to_optimize_name}")
plt.legend()
plt.grid(True)
plt.xlim(left=0)
plt.tight_layout()
plt.savefig(f'final_stopping_criterion.png')
plt.close()

print("\nFinal training data size (scaled features):", train_x_scaled.shape)
print("Final training data size (targets, standardized & negated):", train_y.shape)
print(f"Total points in best_values plot: {len(best_values)}")
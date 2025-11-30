#a remote control for ANSA

import os
import sys
import json
import time

# --- STEP 1: Make the "Remote Control" module findable ---
ANSA_REMOTE_CONTROL_PATH = r"C://Users//Dimitra//AppData//Local//Apps//BETA_CAE_Systems//ansa_v25.1.2//scripts//RemoteControl//ansa"
if ANSA_REMOTE_CONTROL_PATH not in sys.path:
    sys.path.append(ANSA_REMOTE_CONTROL_PATH)

# --- STEP 2: Import the "Remote Control" module ---
try:
    from AnsaProcessModule import AnsaProcess, IAPConnection, PostConnectionAction, PreExecutionDatabaseAction
except ImportError:
    print(f"FATAL ERROR: Could not import AnsaProcessModule from the standard Python environment.")
    print(f"Please ensure the path is correct and accessible: {ANSA_REMOTE_CONTROL_PATH}")
    sys.exit(1)

VARIABLE_CONFIG = [
    [2, "FrontRear_height"], [3, "side_height"], [4, "side_width"], [6, "holes"],
    [9, "edge_fit"], [11, "rear_offset"], [13, "PSHELL_1_T"], [14, "PSHELL_2_T"],
    [15, "PSHELL_42733768_T"], [16, "PSHELL_42733769_T"], [17, "PSHELL_42733770_T"],
    [18, "PSHELL_42733772_T"], [19, "PSHELL_42733773_T"], [20, "PSHELL_42733774_T"],
    [21, "PSHELL_42733779_T"], [22, "PSHELL_42733780_T"], [23, "PSHELL_42733781_T"],
    [24, "PSHELL_42733782_T"], [25, "PSHELL_42733871_T"], [26, "PSHELL_42733879_T"],
    [1, "MAT1_1_E"], [27, "MAT1_42733768_E"], [28, "scale_x"], [30, "scale_y"], [29, "scale_z"]
]
VARIABLE_NAMES = [var[1] for var in VARIABLE_CONFIG]


class AnsaRemoteEvaluator:
    """
    Manages the persistent ANSA instance
    """
    def __init__(self, project_dir, ansa_worker_script):
        print("--- Initializing AnsaRemoteEvaluator (in standard Python) ---")
        self.project_dir = project_dir
        self.ansa_worker_script = ansa_worker_script
        self.combinations_filepath = os.path.join(self.project_dir, "combinations.txt")
        self.ansa_process = None
        self.connection = None

        # --- STEP 3: Explicitly define the path to the ANSA executable ---
        ansa_executable = r"C:\Users\Dimitra\AppData\Local\Apps\BETA_CAE_Systems\ansa_v25.1.2\ansa64.bat"
        
        # Additional options for starting ANSA
        other_options = ['-b', '-dmroot', self.project_dir]

        # --- STEP 4: Start the ANSA Worker Process ---
        # pass the `ansa_command` directly to prevent AnsaProcessModule
        # from calling its internal _get_ansa_command method, which would
        # fail because it tries to 'import ansa'
        print("Starting persistent ANSA worker process...")
        self.ansa_process = AnsaProcess(
            ansa_command=ansa_executable,
            run_in_batch=False,
            other_running_options=other_options
        )

        # The rest of the logic uses the connection to talk to the now-running ANSA process
        self.connection = self.ansa_process.get_iap_connection()

        print("Connecting to ANSA worker...")
        response = self.connection.hello()
        if not response.success():
            raise RuntimeError("Failed to establish handshake with ANSA process.")
        print("Connection successful!")
        
        print("Running one-time setup in ANSA (opening database)...")
        response = self.connection.run_script_file(
            self.ansa_worker_script,
            "initialize_environment" 
        )
        if not response.success():
            self.close()
            raise RuntimeError(f"Failed to run initialization script in ANSA. Response: {response.get_response_dict()}")
        print("ANSA environment initialized.")

    def _create_combinations_file(self, sample_dict):
        """Creates the combinations.txt file for a single sample point"""
        id_line = "DESIGN VARIABLE ID: " + ", ".join([str(var[0]) for var in VARIABLE_CONFIG])
        name_line = "# DESIGN VARIABLE NAME: " + ", ".join([var[1] for var in VARIABLE_CONFIG])
        
        ordered_values = []
        for var_name in VARIABLE_NAMES:
            value = sample_dict.get(var_name)
            if value is None:
                raise KeyError(f"Variable '{var_name}' not found in the provided sample dictionary.")
            formatted_value = f"{value:.8f}" if isinstance(value, float) else str(value)
            ordered_values.append(formatted_value)
            
        data_line = "Experiment 1: " + ", ".join(ordered_values)
        file_content = f"#\n#\n{id_line}\n{name_line}\n#\n{data_line}\n#"
        
        with open(self.combinations_filepath, 'w') as f:
            f.write(file_content)

    def evaluate(self, sample_dict):
        """
        Runs a single evaluation by sending a command to the persistent ANSA process
        """
        print(f"\n--- Evaluating new sample point ---")
        try:
            self._create_combinations_file(sample_dict)
            print(f"Generated combinations file for evaluation.")

            print("Sending evaluation task to ANSA worker...")
            start_time = time.monotonic()
            
            response = self.connection.run_script_file(
                filepath=self.ansa_worker_script,
                function_name="run_single_evaluation",
                pre_execution_database_action=PreExecutionDatabaseAction.keep_database
            )
            duration = time.monotonic() - start_time
            print(f"ANSA task finished in {duration:.2f} seconds.")

            if not response.success():
                print("ERROR: ANSA script execution failed.")
                return None

            result_dict_str = response.get_response_dict()
            if not result_dict_str or "error" in result_dict_str:
                print(f"ERROR: ANSA script returned an error: {result_dict_str}")
                return None
            
            final_results = {k: float(v) for k, v in result_dict_str.items()}
            print("[EVALUATOR] Simulation Responses Received:")
            print(json.dumps(final_results, indent=2))
            return final_results

        except Exception as e:
            print(f"An error occurred in the Python master script during evaluation: {e}")
            return None

    def close(self):
        """
        Gracefully shuts down the ANSA worker process
        """
        if self.connection:
            print("\n--- Shutting down ANSA worker process ---")
            self.connection.goodbye(PostConnectionAction.shut_down)
            self.connection.close()
            self.ansa_process.ansa_process.wait() 
            print("ANSA worker has been shut down.")
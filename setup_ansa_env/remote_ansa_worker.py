import sys
import json
import logging
import os
import ansa
from ansa import base, morph, constants, dm

# --- Global Configuration (paths are relative to the DM Root) ---
PROJECT_DIR = r"C:\Users\Dimitra\Desktop\25DVs_DM"
DB_FILEPATH = os.path.join(PROJECT_DIR, "25DVs.ansa")
COMB_FILEPATH = os.path.join(PROJECT_DIR, "combinations.txt")
RUN_DIR_DM_PATH = "DM:" # Path for PerformDoeStudy
TASK_NAME = "OPTIMIZATION_TASK_1"
OPT_TASK_ID = 1

def parse_dm_reports_to_str_dict(report_objects_list):
    """
    Parses a list of DM Report objects and returns a dictionary
    with all values converted to strings, as required by IAPConnection.
    """
    results = {}
    print("--- [ANSA] Parsing DM Reports ---")
    for report_obj in report_objects_list:
        report_data = report_obj.get_all_values()
        report_name = report_data.get("Name")
        report_value = report_data.get("Value")
        if report_name and report_value is not None:
            #The remote connection can only transport strings.
            results[report_name] = str(report_value)
            print(f"  > [ANSA] Parsed: '{report_name}': {report_value}")
    return results

def initialize_environment():
    """
    This function is called ONCE when the connection is established.
    It opens the database so it's ready for all subsequent evaluations.
    """
    print("--- [ANSA] Initializing environment ---")
    try:
        if not os.path.exists(DB_FILEPATH):
             raise FileNotFoundError(f"ANSA database not found at: {DB_FILEPATH}")
        base.Open(DB_FILEPATH)
        print(f"--- [ANSA] Successfully opened database: {DB_FILEPATH} ---")
        return {"status": "success"} # Return a success message
    except Exception as e:
        print(f"--- [ANSA] FATAL ERROR during initialization: {e} ---")
        # In case of error, return error message
        return {"status": "error", "message": str(e)}

def run_single_evaluation():
    """
    Main evaluation function called in a loop by the optimizer.
    It assumes the database is already open. It runs the study and returns the results.
    """
    print("--- [ANSA] Starting single evaluation ---")
    try:
        print(f"--- [ANSA] Starting PerformDoeStudy with comb file: {COMB_FILEPATH} ---")

        ret_code = morph.PerformDoeStudy(
            opt_task_id=OPT_TASK_ID, 
            comb_file=COMB_FILEPATH,
            dir_path=RUN_DIR_DM_PATH,
            detailed_results=True,
            upload_doe_study=True
        )

        if ret_code == 0: 
            raise RuntimeError("morph.PerformDoeStudy failed.")
        
        print("--- [ANSA] PerformDoeStudy completed successfully. ---")
        
        sim_run_server_ids = ret_code.sim_runs_server_ids
        if not sim_run_server_ids:
            raise RuntimeError("PerformDoeStudy did not return any simulation run IDs.")

        # Process the first simulation run ID to get the report
        run_id = sim_run_server_ids[0]
        print(f"--- [ANSA] Processing Simulation_Run with Server ID: {run_id} ---")
        sim_run_object = dm.DMObject(server_id=run_id, type="Simulation_Run")
        
        contained_report_objects = sim_run_object.get_contained_objects("Report")
        if not contained_report_objects:
            print("  [ANSA] No 'Report' objects found in this Simulation_Run.")
            return {"error": "No reports found"}

        # Call the parser function and return the results dictionary
        results_dict = parse_dm_reports_to_str_dict(contained_report_objects)
        
        print("--- [ANSA] SCRIPT SUCCEEDED, returning results. ---")
        return results_dict

    except Exception as e:
        # Log the error 
        print(f"--- [ANSA] EXCEPTION IN WORKFLOW: {e} ---", file=sys.stderr)
        # Return an error dictionary
        return {"error": f"Exception in ANSA script: {str(e)}"}
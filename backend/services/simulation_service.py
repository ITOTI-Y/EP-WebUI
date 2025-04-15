import os
import time
import subprocess
import pandas as pd
import shutil
import logging
from eppy.modeleditor import IDF
from eppy.runner.run_functions import runIDFs, run

class EnergyPlusRunner:
    """
    Encapsulate the execution of EnergyPlus simulation
    """

    def __init__(self, eplus_executable_path: str = None):
        """
        Initialize the EnergyPlusRunner

        Args:
            eplus_executable_path (str, optional): Path to the EnergyPlus executable. Defaults to None.
        """
        self.eplus_path = eplus_executable_path

        if self.eplus_path is None:
            logging.warning("EnergyPlus executable path not provided. eppy will attempt to locate it automatically.")
        elif not os.path.exists(self.eplus_path):
            raise FileNotFoundError(f"EnergyPlus executable not found at {self.eplus_path}")
        
    def run_simulation(self, idf_path: str, weather_path:str, output_dir:str, output_prefix:str=None, output_suffix:str='C', cleanup:bool=True, config:str=None):
        """
        Run an EnergyPlus simulation

        Args:
            idf_path (str): Path to the IDF file
            weather_path (str): Path to the weather file
            output_dir (str): Directory to store the output files
            output_prefix (str, optional): Prefix for the output files. Defaults to None.
            output_suffix (str, optional): Suffix for the output files. Defaults to 'C'.
            cleanup (bool, optional): Whether to clean up the output files after the simulation is run. Defaults to True.
            config (str, optional): Path to the EnergyPlus configuration file. Defaults to None.

        Returns:
            tuple: (success: bool, message: str) Simulation Status and Message
        """

        # Check input files and directories
        if not os.path.exists(idf_path):
            return False, f"IDF file not found at {idf_path}"
        
        if not os.path.exists(weather_path):
            return False, f"Weather file not found at {weather_path}"
        
        os.makedirs(output_dir, exist_ok=True)

        # Determine the run name
        run_name = output_prefix if output_prefix else os.path.splitext(os.path.basename(idf_path))[0]

        # Load the IDF file
        try:
            # IDF.setiddname(self.eplus_path) # setting the IDD file path (if needed)
            idf_object = IDF(idf_path)
        except Exception as e:
            return False, f"Error loading IDF file: {e} with path {idf_path}"

        logging.info(f"Start running EnergyPlus simulation: IDF='{os.path.basename(idf_path)}', EPW='{os.path.basename(weather_path)}', Output='{output_dir}' ")
        start_time = time.time()

        try:
            # Executing the 'runIDFs' function
            # runIDFs expects a list of IDF objects, 
            # verbose='q' signifies "quiet" mode, minimizing Eppy's printed output.
            # ep_args is used to pass additional command line arguments to EnergyPlus
            # readvars=True allows eppy to attempt reading the .eso file after running
            run_results = run(
                idf_object,
                weather=weather_path,
                output_directory=output_dir,
                output_suffix=output_suffix,
                verbose='v',
                readvars=True, # It will generate .csv files from .eso files
            )

            if not run_results or len(run_results) == 0:
                raise RuntimeError("EnergyPlus returned no results")
            
            result = run_results # get the result
            stderr_output = result[1] if isinstance(result, tuple) and len(result) > 1 else ""
            if stderr_output is None: 
                stderr_output = ""

            if "EnergyPlus Completed Successfully" in stderr_output or \
                ("Error" not in stderr_output and "Fatal" not in stderr_output):
                success = True
                message = f"EnergyPlus simulation completed successfully, used {time.time() - start_time:.2f} seconds"
            else:
                success = False
                stdout_output = result[0] if isinstance(result, tuple) else ""
                message = f"EnergyPlus simulation failed, used {time.time() - start_time:.2f} seconds \n" \
                        f"Stderr: \n {stderr_output} \n" \
                        f"Stdout (): \n {stdout_output[:500]}"
        
        except FileNotFoundError as e:
            success = False
            message = f"{e} \n EnergyPlus executable not found at {self.eplus_path}"

        except Exception as e:
            success = False
            message = f"Encountered an unexpected error while running EnergyPlus (eppy): {e}.  Execution time: {time.time() - start_time:.2f} seconds"

        logging.info(message)

        if cleanup:
            cleanup_extensions = config.get('simulation', {}).get('cleanup_files', [])
            if cleanup_extensions:
                for filename in os.listdir(output_dir):
                    if any(filename.lower().endswith(ext) for ext in cleanup_extensions) and \
                        not filename.lower().endswith('.csv') and \
                        not filename.lower().endswith('.idf') and \
                        run_name not in filename:
                        try:
                             file_to_remove = output_dir / filename
                             if os.path.isfile(file_to_remove):
                                 os.remove(file_to_remove)
                        except OSError as e:
                            logging.error(f"Error removing file {filename}: {e}")

        return success, message
    
class SimulationResult:
    """
    Parsing EnergyPlus simulation output files
    """
    def __init__(self, output_dir: str, output_prefix: str):
        """
        Initialize the SimulationResult

        Args:
            output_dir (str): Directory containing the simulation output files
            output_prefix (str): Prefix of the simulation output files
        """
        self.output_dir = output_dir
        self.output_prefix = output_prefix

        self.table_csv_path = os.path.join(output_dir, f"{output_prefix}Table.csv")
        self.meter_csv_path = os.path.join(output_dir, f"{output_prefix}Meter.csv")
        self.sql_path = os.path.join(output_dir, f"{output_prefix}.sql")

        # Check if the output files exist
        if not os.path.exists(self.table_csv_path):
            logging.warning(f"Table CSV file not found at {self.table_csv_path}")
        if not os.path.exists(self.meter_csv_path):
            logging.warning(f"Meter CSV file not found at {self.meter_csv_path}")
    
    def get_meter_data(self,  columns_map=None, year=None):
        pass

    def get_source_eui(self, ng_conversion_factor: float):
        pass

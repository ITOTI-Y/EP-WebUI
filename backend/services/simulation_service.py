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
                output_prefix=output_prefix,
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
                        not filename.lower().endswith('.idf'):
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
    
    def get_meter_data(self,  columns_map:dict =None, year:int =None) -> pd.DataFrame:
        """
        Load and parse the Meter CSV file

        Args:
            columns_map (dict, optional): A dictionary mapping column names to new names. Defaults to None.
            year (int, optional): The year to filter the data by. Defaults to None.

        Returns:
            pd.DataFrame: A pandas DataFrame containing the meter data
        """
        if not os.path.exists(self.meter_csv_path):
            logging.warning(f"Meter CSV file not found at {self.meter_csv_path}")
            return None
        try:
            df = pd.read_csv(self.meter_csv_path)
            if 'Date/Time' in df.columns:
                df['Date/Time'] = df['Date/Time'].astype(str).str.strip()
                df['Timestamp'] = pd.to_datetime(df['Date/Time'], format=' %m/%d  %H:%M:%S', errors='coerce') # Note the space in E+ output format
                df = df.dropna(subset=['Timestamp'])
                if year: df['Timestamp'] = df['Timestamp'].apply(lambda dt: dt.replace(year=year))
                df.set_index('Timestamp', inplace=True)
                # df.drop(columns=['Date/Time'], inplace=True) # Optional
            else: logging.warning("Warning: 'Date/Time' column not found in the CSV file.")

            if columns_map: df.rename(columns=columns_map, inplace=True)

            # Unit conversion (J -> kWh or W -> kW)
            # EnergyPlus variables and Meter output to CSV/ESO are usually J or W
            for col in df.columns:
                if isinstance(df[col].iloc[0], (int, float)): # Only process numeric columns
                    # Determine if it's energy (J) or power (W) - usually look at variable name or E+ documentation
                    if 'Energy' in col and '[J]' in col: # Assume energy variable output unit is J
                        df[col] = df[col] / 3_600_000 # J to kWh
                        # (Optional) Rename column
                        # df.rename(columns={col: col.replace('[J]', '[kWh]')}, inplace=True)
                    elif 'Rate' in col and '[W]' in col: # Assume power variable output unit is W
                        df[col] = df[col] / 1000 # W to kW
                        # (Optional) Rename column
                        # df.rename(columns={col: col.replace('[W]', '[kW]')}, inplace=True)
                    elif '[J/m2]' in col: # Radiant energy density
                        df[col] = df[col] / 3_600_000 # J/m2 to kWh/m2
                    elif '[W/m2]' in col: # Radiant power density
                        df[col] = df[col] / 1000 # W/m2 to kW/m2
                    # (Add other possible unit conversions)
            return df
        except Exception as e:
            print(f"Error: Error processing CSV file '{self.meter_csv_path}': {e}")
            return None

    def get_source_eui(self, ng_conversion_factor: float) -> float:
        """
        Extract the source EUI from the "Table.csv" file.

        Args:
            ng_conversion_factor (float): The conversion factor for natural gas to energy.

        Returns:
            float: The source EUI (kWh/m2/yr) or None if the source EUI is not found.
        """
        if not os.path.exists(self.table_csv_path):
            logging.warning(f"Table CSV file not found at {self.table_csv_path}")
            return None
        target_section_start = "REPORT:,Annual Building Utility Performance Summary".lower()
        target_data_row_start = ",Total Source Energy,".lower()
        target_column_header = "Energy Per Total Building Area [kWh/m2]".lower()
        
        # Row above the data rows that serves as the header row indicator (used for locating target column indices).
        header_row_start = ",,Total Energy [kWh]".lower()

        in_target_section = False
        header_found = False
        target_col_index = -1

        try:
            with open(self.table_csv_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            for line in lines:
                line_processed = line.strip().lower()

                # Check if the current line indicates the start of the target section
                if target_section_start in line_processed:
                    in_target_section = True
                    continue
                
                if not in_target_section:
                    continue
                
                if not header_found and line_processed.startswith(header_row_start):
                    header_parts = [h.strip() for h in line_processed.split(',')]
                    try:
                        target_col_index = header_parts.index(target_column_header)
                        header_found = True
                    except ValueError:
                        logging.warning(f"Target column header '{target_column_header}' not found in the header row.")
                        return None
                    continue
                
                if header_found and line_processed.startswith(target_data_row_start):
                    data_parts = [p.strip() for p in line_processed.split(',')]
                    logging.info(f"DEBUG: Found data row: {data_parts}")
                    if len(data_parts) > target_col_index:
                        try:
                            source_eui_str = data_parts[target_col_index]
                            source_eui = float(source_eui_str)
                            logging.info(f"DEBUG: Extracted Source EUI string: '{source_eui_str}', float: {source_eui}")
                            return source_eui
                        except ValueError:
                            logging.warning(f"Failed to convert source EUI string '{source_eui_str}' to float.")
                            return None
                    else:
                        logging.error(f"Error: The target dataset has {len(data_parts)} columns, which is insufficient to retrieve the value at index {target_col_index}. Offending line: {line_processed}")
                        return None
            # If the loop completes without a match.
            if target_col_index == -1:
                logging.error(f"Warning: Unable to locate the header row, '{header_row_start}', within the '{target_section_start}' section of '{self.table_csv_path}'.")
            else:
                logging.error(f"Warning: Unable to locate the data row, '{target_data_row_start}', within the '{target_section_start}' section of '{self.table_csv_path}'.")
            return None
        except FileNotFoundError as e:
            logging.error(f"Error: File not found: {e}")
            return None
        except Exception as e:
            logging.error(f"Error: Unexpected error occurred while processing the table CSV file: {e}")
            return None

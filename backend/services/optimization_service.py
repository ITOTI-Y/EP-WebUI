import time
import collections
import numpy as np
import pandas as pd
import shutil
import logging
from .idf_service import IDFModel
from .simulation_service import EnergyPlusRunner, SimulationResult
from joblib import Parallel, delayed
from SALib.sample import saltelli
from SALib.analyze import sobol
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor
from statsmodels.formula.api import ols

class OptimizationPipeline:
    """
    Encapsulates the entire building energy optimization process: 
    simulation, sensitivity, analysis, optimization, and validation.
    """
    def __init__(self, city: str, ssp: int, btype: str, config: dict):
        """
        Initialize the optimization pipeline

        Args:
            city (str): City name
            ssp (int): SSP scenario ('TMY', 126, 245, 370, 585)
            btype (str): Building type
            config (dict): Configuration dictionary
        """

        self.city = city
        self.ssp = str(ssp)
        self.btype = btype
        self.config = config
        self.unique_id = f"{city}_{btype}_{ssp}"

        # Path Configuration
        self.prototype_idf_path = self._find_prototype_idf() # Path to the prototype IDF file
        self.weather_path = self._find_weather_file() # Path to the weather file

        # WorkDirectory Configuration
        self.work_dir = self.config['paths']['results_dir'] / 'Temp' / self.unique_id
        self.work_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Simulation Components
        self.runner = EnergyPlusRunner(self.config['paths']['eplus_executable'])
        self.ecm_ranges = config['ecm_ranges']
        self.ecm_names = list(self.ecm_ranges.keys())

        # Initialize Results
        self.baseline_eui = None # Benchmark EUI
        self.sensitivity_samples = None # Sample sensitivity analysis (discrete)
        self.sensitivity_results = None # Sobol Analysis Results (Indices)
        self.surrogate_model = None # Surrogate Model
        self.surrogate_model_summary = None # Surrogate Model Summary or Feature Importance
        self.optimization_result = None # Raw Results from scipy.optimize
        self.optimal_params = None # Optimized parameter set achieving the best results
        self.optimal_eui_predicted = None # Optimal EUI Prediction via Surrogate Model
        self.optimal_eui_simulated = None # EUI Optimized Through Real-World Simulation
        self.optimization_improvement = None # EUI Improvement Rate
        self.optimization_bias = None # Optimization bias

    def _find_prototype_idf(self):
        """
        Find the prototype IDF file for the given city and building type
        """
        proto_dir = self.config['paths']['prototypes_dir']
        if not proto_dir.exists():
            logging.error(f"Prototype directory not found: {proto_dir}")
            return None
        found = [f for f in proto_dir.glob("*.idf") if self.btype.lower() in f.stem.lower()]
        if not found:
            logging.error(f"No prototype IDF file found for {self.btype}")
            return None
        logging.info(f"Found prototype IDF file: {found[0]}")
        return found[0]
    
    def _find_weather_file(self):
        """
        Find the weather file for the given city and SSP
        """
        search_city = self.city.lower()
        search_ssp = self.ssp.lower()
        if self.ssp.upper() == 'TMY':
            epw_dir = self.config['paths']['tmy_dir']
        else:
            epw_dir = self.config['paths']['ftmy_dir']

        if not epw_dir.exists():
            logging.error(f"Weather directory not found: {epw_dir}")
            return None
        found = [f for f in epw_dir.glob("*.epw") if search_city in f.stem.lower() and search_ssp in f.stem.lower()]
        if not found:
            logging.error(f"No weather file found for {self.city} and {self.ssp}")
            return None
        logging.info(f"Found weather file: {found[0]}")
        return found[0]
    
    def _continuous_to_discrete(self, value, param_name):
        """
        Maps a continuous value within the range of [0, 1) to 
        a discrete level based on provided parameters.

        Args:
            value (_type_): _description_
            param_name (_type_): _description_
        """

        discrete_levels = self.ecm_ranges.get(param_name)
        if not discrete_levels:
            logging.error(f"No discrete levels defined for {param_name}")
            raise ValueError(f"Unknown ECM Parameter: {param_name}")
        if not isinstance(discrete_levels. list) or len(discrete_levels) == 0:
            logging.error(f"Invalid discrete levels for {param_name}: {discrete_levels}")
            raise ValueError(f"The provided discrete levels for parameter '{param_name}' are invalid: {discrete_levels}")

        # Create split points, one more than the number of discrete levels
        space = np.linspace(0, 1, len(discrete_levels) + 1)
        # Ensure the value is within the range [0,1)
        value = np.max([0, np.min([value, 0.999999])])

        for i in range(len(discrete_levels)):
            # check if value falls in the i-th interval [space[i], space[i+1]
            if value >= space[i] and value < space[i+1]:
                return discrete_levels[i]
        
        # If the value is nearing 1, return the last discrete value (handling edge cases).
        return discrete_levels[-1]

    def _params_array_to_dict(self, params_array):
        """
        Converts an array of parameters to a parameter name dictionary.
        """
        return {name: val for name, val in zip(self.ecm_names, params_array)}
    
    def _apply_ecms_to_idf(self, idf_model: IDFModel, params_dict: dict):
        """
        Applies the ECMs to the IDF model based on the provided parameters.

        Args:
            idf_model (IDF): IDF model to apply the ECMs to
            params_dict (dict): Dictionary of parameters to apply the ECMs to
        """
        for param_name, value in params_dict.items():
            if param_name in self.ecm_ranges:
                if value == 0 and param_name not in ['nv_area']:
                    continue

            if param_name == 'shgc':
                # 需要结合 win_U 一起处理，或者假设单独修改？
                # 这里简化为假设 apply_window_properties 能处理部分参数更新
                # 可能需要获取当前的 U 值来配合修改
                # current_u, _, current_vt = idf_model.get_window_properties_from_table() # 需要实现此方法或缓存
                # if current_u and current_vt:
                #     idf_model.apply_window_properties(current_u, value, current_vt)
                pass

            elif param_name == 'win_u':
                pass # 暂时跳过单独修改 U 值

            elif param_name == 'nv_area':
                if value >= 0:
                    idf_model.apply_natural_ventilation(value)
            elif param_name == 'insu':
                idf_model.apply_wall_insulation(value)
            elif param_name == 'infl':
                idf_model.apply_air_infiltration(value)
            elif param_name == 'cool_cop':
                idf_model.apply_cooling_cop(value)
            elif param_name == 'cool_air_temp':
                idf_model.apply_cooling_supply_temp(value)
            elif param_name == 'lighting':
                lighting_level = int(value)
                if lighting_level == 0:
                    continue
                    
                reduction_map = self.config.get('lighting_reduction_map', {})

                # Find reduction factors based on building type and classification.
                btype_key = self.btype.lower()

                # Retrieve the inner dictionary associated with a specific building type from the outer dictionary.
                inner_map = reduction_map.get(btype_key, {})
                if not inner_map:
                    logging.warning(f"No lighting reduction map found for {btype_key}")
                
                # Retrieve the reduction factor for the specified lighting level.
                reduction_factor = inner_map.get(lighting_level, 1.0)
                
                if reduction_factor < 1.0:
                    idf_model.apply_lighting_reduction(reduction_factor, btype_key)
                elif reduction_factor >= 1.0 and lighting_level != 0 :
                    logging.warning(f"Lighting reduction factor ({reduction_factor}) found is either invalid or not required \
                                    for building type '{self.btype}' and level '{lighting_level}'. Lighting is not modified.")
                
            # Process Window properties
            win_u = params_dict.get('win_u', 0)
            shgc = params_dict.get('shgc', 0)
            vt = params_dict.get('vt', 0)
            if win_u > 0 and shgc > 0:
                idf_model.apply_window_properties(win_u, shgc, vt)
            elif win_u > 0:
                # Only update U-value while keeping SHGC and VT the same
                pass # Not implemented yet
            elif shgc > 0:
                # Only update SHGC while keeping U-value and VT the same
                pass # Not implemented yet
            elif vt > 0:
                # Only update VT while keeping U-value and SHGC the same
                pass # Not implemented yet

    def _run_single_simulation_internal(self, params_dict: dict={}, run_id: str=None, is_baseline:bool=False):
        """
        Set up, execute, and analyze a single EnergyPlus simulation.

        Args:
            params_dict (dict, optional): A dictionary containing ECM parameter values. An empty dictionary signifies the baseline run. Defaults to {}.
            run_id (str, optional): Run identifier for naming output subdirectories and file prefixes.. Defaults to None.
            is_baseline (bool, optional): Explicitly designated as a benchmark run. Defaults to False.
        """

        run_label = run_id if run_id else 'baseline' if is_baseline else 'ecm_run'
        run_dir = self.work_dir / run_label
        run_dir.mkdir(parents=True, exist_ok=True)

        run_idf_path = run_dir / f"{run_label}.idf"
        run_output_prefix = run_label

        try:
            # Copy the prototype IDF file to the run directory
            shutil.copy(self.prototype_idf_path, run_idf_path)

            # Load and Modify the IDF file
            idf = IDFModel(str(run_idf_path))
            idf.apply_run_peroid(self.config['simulation']['start_year'], self.config['simulation']['end_year'])
            idf.apply_output_requests() # Apply standard output requests
            idf.apply_simulation_control_settings()

            if not is_baseline and params_dict:
                self._apply_ecms_to_idf(idf, params_dict)

            idf.save()

            # Run EnergyPlus Simulation
            success, message = self.runner.run_simulation(
                idf_path=run_idf_path,
                weather_path=self.weather_path,
                output_dir=run_dir,
                output_prefix=run_output_prefix,
                config=self.config
            )
            
            # Parse the simulation results
            if success:
                result_parser = SimulationResult(
                    output_dir=run_dir,
                    output_prefix=run_output_prefix
                )
                eui = result_parser.get_source_eui(self.config['constants']['ng_conversion_factor'])
                if eui is None:
                    logging.error(f"Failed to calculate source EUI for run {run_label}")
                return eui 
            else:
                logging.error(f"Simulation failed for run {run_label}: {message}")
                return None
        
        except Exception as e:
            logging.error(f"An error occurred while running simulation {run_label}: {e}")
            shutil.rmtree(run_dir, ignore_errors=True)
            return None
    
    def run_baseline_simulation(self):
        """
        Run a baseline simulation and save EUI
        """
        logging.info(f"Running baseline simulation for {self.unique_id}")
        self.baseline_eui = self._run_single_simulation_internal(is_baseline=True, run_id="baseline")
        if self.baseline_eui is not None:
            logging.info(f"Baseline simulation completed successfully. Reference source EUI: {self.baseline_eui} kWh/m2/yr")
        else:
            logging.error(f"Baseline simulation failed for {self.unique_id}")
        return self.baseline_eui

    def _run_sensitivity_analysis(self, params_array: np.ndarray, sample_index: int) -> dict|None:
        """
        Run a sensitivity analysis for a specific sample index.

        Args:
            params_array (np.ndarray): Array of parameters to analyze
            sample_index (int): Index of the sample to analyze
        
        Returns:
            dict|None: Dictionary containing the sensitivity analysis results or None if the simulation fails
        """

        # Convert the sample index to a parameter dictionary
        params_dict = self._params_array_to_dict(params_array[sample_index])

        # Run the simulation
        eui = self._run_single_simulation_internal(params_dict=params_dict, run_id=f"sensitivity_{sample_index}", is_baseline=False)
        params_dict = self._params_array_to_dict(params_array)
        run_id = f"sample_{sample_index}"
        eui = self._run_single_simulation_internal(params_dict=params_dict, run_id=run_id, is_baseline=False)
        if eui:
            result_dict = params_dict.copy()
            result_dict['eui'] = eui
            return result_dict
        else:
            logging.error(f"Sensitivity analysis failed for sample {sample_index}")
            return None
        
    def _refill_continuous_space(self, continuous_samples: np.ndarray, discrete_results_df: pd.DataFrame) -> np.ndarray:
        """
        Find or estimate the corresponding Energy Use Intensity (EUI) for a continuous sample in Sobol analysis.

        Args:
            continuous_samples (np.ndarray): Array of continuous samples
            discrete_results_df (pd.DataFrame): DataFrame containing the results of the discrete samples
        
        Returns:
            np.ndarray: Array of EUI values
        """

        # Initialize an empty array to store the EUI values
        num_continuous = continuous_samples.shape[0] # Get the number of continuous samples
        Y = np.full(num_continuous, np.nan) # Initialize the EUI array with NaNs

        # Create a lookup dictionary based on discrete parameter combinations for efficiency
        discrete_lookup = {}
        # Iterate through the discrete results DataFrame
        for index,row in discrete_results_df.iterrows():
            key = tuple(row[self.ecm_names].values)
            discrete_lookup[key] = row['eui']
        
        for i in range(num_continuous):
            discrete_params_list = []
            # Iterate through each parameter
            for j, param_name in enumerate(self.ecm_names):
                # Mapping the continuous sample to a discrete value
                discrete_val = self._continuous_to_discrete(continuous_samples[i, j], param_name)
                discrete_params_list.append(discrete_val)
            # Converting discrete parameter lists to tuples for lookups
            discrete_key = tuple(discrete_params_list)
            # Retrieving the EUI from the dictionary.
            if discrete_key in discrete_lookup:
                Y[i] = discrete_lookup[discrete_key]
            else:
                # If an exact match can't be found among the discrete results.
                logging.warning(f"Warning: Exact match: {discrete_key} was not found for sample {i} in the discrete result. the EUI will be NaN.")
        
        # Check if any NaNs are present in the EUI array
        nan_count = np.isnan(Y).sum()
        if nan_count > 0:
            logging.warning(f"Warning: {nan_count}/{num_continuous} consecutive samples failed to find the corresponding discrete EUI.")
        
        return Y
    
    def run_sensitivity_analysis(self):
        """
        Perform the Sobol sensitivity analysis process.
        """
        logging.info(f"--- {self.unique_id}: Run sensitivity analysis ---")
        num_vars = len(self.ecm_names)
        


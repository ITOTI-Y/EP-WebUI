import time
import collections
import numpy as np
import pandas as pd
import shutil
import logging
import json
from .pv_service import PVManager
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
        self.building_floor_area = None # Building floor area
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
        self.suitable_surfaces = None # Suitable surfaces for PV installation
        self.pv_idf_path = None # Path to the IDF file with PV
        self.pv_generation_results = None # PV generation results
        self.gross_eui_with_pv = None # Gross EUI with PV
        self.net_eui_with_pv = None # Net EUI with PV

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
    
    def _continuous_to_discrete(self, value: float, param_name: str) -> float:
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
        if not isinstance(discrete_levels, list) or len(discrete_levels) == 0:
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
                
            elif param_name in ['shgc', 'win_u', 'vt']:
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

        run_label = str(run_id) if run_id is not None else 'baseline' if is_baseline else 'ecm_run'
        run_dir = self.work_dir / run_label
        run_dir.mkdir(parents=True, exist_ok=True)

        run_idf_path = run_dir / f"{run_label}.idf"
        run_output_prefix = run_label
        floor_area = None

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
            floor_area = idf.get_total_floor_area() # Get the total floor area
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
                return eui, floor_area
            else:
                logging.error(f"Simulation failed for run {run_label}: {message}")
                return None, floor_area
        
        except Exception as e:
            logging.error(f"An error occurred while running simulation {run_label}: {e}")
            shutil.rmtree(run_dir, ignore_errors=True)
            return None, floor_area
    
    def run_baseline_simulation(self):
        """
        Run a baseline simulation and save EUI
        """
        logging.info(f"Running baseline simulation for {self.unique_id}")
        self.baseline_eui, self.building_floor_area = self._run_single_simulation_internal(is_baseline=True, run_id="baseline")
        if self.baseline_eui is not None:
            logging.info(f"Baseline simulation completed successfully. Reference source EUI: {self.baseline_eui} kWh/m2/yr")
        else:
            logging.error(f"Baseline simulation failed for {self.unique_id}")
        if self.building_floor_area is not None and self.building_floor_area > 0:
            logging.info(f"Building floor area: {self.building_floor_area:.2f} m2")
        else:
            logging.warning(f"Warning: Unable to obtain a valid building floor area.")
        return self.baseline_eui
    
    def _run_sensitivity_sample_point(self, params_array: np.ndarray, sample_index: int) -> dict|None:
        """
        Evaluating a single parameter set for sensitivity analysis.

        Args:
            params_array (np.ndarray): Array of parameters to analyze
            sample_index (int): Index of the sample to analyze

        Returns:
            dict|None: A dictionary containing the parameters and resulting EUI, or None.
        """
        params_dict = self._params_array_to_dict(params_array)
        run_id = f"sample_{sample_index}"
        eui, _ = self._run_single_simulation_internal(params_dict=params_dict, run_id=run_id)
        if eui is not None:
            result_dict = params_dict.copy()
            result_dict['eui'] = eui
            return result_dict
        else:
            return None

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
        N_samples_base = self.config['analysis']['sensitivity_samples_n']
        num_cpu = self.config['constants']['cpu_count_override']

        # Define the Sobol Problem
        problem = {
            "num_vars": num_vars,
            "names": self.ecm_names,
            "bounds": [[0.0, 1.0] * num_vars]
        }

        # Generate continuous samples
        param_values_continuous = saltelli.sample(problem, N_samples_base, calc_second_order=True)

        # Transform the continuous samples to discrete samples
        param_values_discrete_unique = set()
        for i in range(param_values_continuous.shape[0]):
            discrete_sample = tuple(self._continuous_to_discrete(param_values_continuous[i, j], name)
                                    for j, name in enumerate(self.ecm_names))
            param_values_discrete_unique.add(discrete_sample)
        
        param_values_discrete_list = [list(s) for s in param_values_discrete_unique]
        num_unique_samples = len(param_values_discrete_list)
        logging.info(f"Generated {param_values_continuous.shape[0]} continuous samples, which map to {num_unique_samples} unique discrete parameter combinations.")

        # Run Simulation for Discrete Samples in Parallel
        discrete_results_file = self.work_dir / 'sensitivity_discrete_results.csv'
        if not discrete_results_file.exists():
            # Set n_jobs to 1 if debug, otherwise use all cores.
            n_jobs = 1 if self.config['constants']['debug'] else num_cpu

            logging.info(f"Running {num_unique_samples} discrete sample simulations in parallel (using {n_jobs} cores)...")
            results_list = Parallel(n_jobs=n_jobs, verbose=10)(
                delayed(self._run_sensitivity_sample_point)(params, i)
                for i, params in enumerate(param_values_discrete_list)
            )

            successful_results = [r for r in results_list if r is not None]
            if not successful_results:
                logging.error("No successful simulations completed during sensitivity analysis.")
                return
            
            self.sensitivity_samples = pd.DataFrame(successful_results)
            self.sensitivity_samples.to_csv(discrete_results_file, index=False)
            logging.info(f"Discrete sensitivity analysis results saved to {discrete_results_file}")
        else:
            logging.info(f"Loading discrete sensitivity analysis results from {discrete_results_file}")
            self.sensitivity_samples = pd.read_csv(discrete_results_file)

        # Refilling the EUI for Continuous Sample Spaces.
        logging.info("Re-populating the EUI of a continuous sample space for Sobol analysis...")
        Y = self._refill_continuous_space(param_values_continuous, self.sensitivity_samples)
        if np.isnan(Y).all():
            logging.error("Error: Unable to locate EUIs for any consecutive samples, Sobol analysis cannot be performed.")
            return

        # Handle NaN in Y (e.g., remove corresponding samples or padding, choose remove here)
        valid_indices = ~np.isnan(Y)
        if not np.all(valid_indices):
            logging.warning(f"Warning: {np.isnan(Y).sum()} samples with NaN EUI values were removed from the analysis.")
            param_values_continuous_valid = param_values_continuous[valid_indices]
            Y_valid = Y[valid_indices]
            if len(Y_valid) < 2:
                logging.error("Error: Not enough valid samples for Sobol analysis.")
                return
        else:
            param_values_continuous_valid = param_values_continuous
            Y_valid = Y

        # Perform Sobol Analysis
        logging.info("Performing Sobol sensitivity analysis...")
        try:
            Si = sobol.analyze(problem, Y_valid, calc_second_order=True, print_to_console=True)
            self.sensitivity_results = Si
        except Exception as e:
            logging.error(f"Error: An issue arose while performing the Sobol analysis: {e}")
            return

        first_order = pd.Series(Si['S1'], index=problem['names'], name='S1')
        total_order = pd.Series(Si['ST'], index=problem['names'], name='ST')
        sensitivity_indices = pd.concat([first_order, total_order], axis=1)
        si_file = self.work_dir / 'sensitivity_indices.csv'
        sensitivity_indices.to_csv(si_file)
        logging.info(f"Sobol sensitivity indices saved to {si_file}")
        if 'S2' in Si and Si['S2'] is not None:
            second_order = pd.DataFrame(Si['S2'], index=problem['names'], columns=problem['names'])
            s2_file = self.work_dir / 'sensitivity_indices_S2.csv'
            second_order.to_csv(s2_file)
            logging.info(f"Sobol second-order sensitivity indices saved to {s2_file}")
    
    def build_surrogate_model(self, model_type: str=None):
        """
        Based on the sensitivity analysis, construct a surrogate model.

        Args:
            model_type (str, optional): Optimization model type. Defaults to None.
        """
        model_type = model_type if model_type else self.config['analysis']['optimization_model']
        logging.info(f"Building a {model_type} surrogate model...")

        if self.sensitivity_samples is None:
            samples_file = self.work_dir / 'sensitivity_discrete_results.csv'
            if samples_file.exists():
                logging.info(f"Loading discrete sample data from file: {samples_file}")
                self.sensitivity_samples = pd.read_csv(samples_file)
            else:
                logging.error("Error: Sensitivity analysis sample data not found; a surrogate model cannot be built.\
                            Please run `run_sensitivity_analysis()` first.")
                return
        
        # Preparing data for modeling (X: Parameters, Y: EUI).
        df = self.sensitivity_samples.dropna(subset=['eui'])
        X = df[self.ecm_names]
        Y = df['eui']

        if X.empty or Y.empty:
            logging.error("Error: Unable to build a surrogate model due to missing sample data or an invalid EUI column.")
            return
    
        model = None
        summary_info = None
        model_file = self.work_dir / f"surrogate_model_{model_type}.txt"

        try:
            if model_type == 'ols':
                formula = 'eui ~ ' + ' + '.join(self.ecm_names)
                model = ols(formula, data=df).fit()
                summary_info = model.summary().as_text()
                with open(model_file, 'w') as f: f.write(summary_info)
            elif model_type.lower() == 'rf':
                model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                model.fit(X, Y)
                importances = pd.Series(model.feature_importances_, index=self.ecm_names).sort_values(ascending=False) # Extracting feature importance.
                summary_info = importances.to_string()
                importances.to_csv(model_file.replace('.txt', '_importance.csv'))
                with open(model_file, 'w') as f: f.write("Random Forest Feature Importances:\n" + summary_info)
            else:
                logging.warning(f"Warning: The surrogate model type '{model_type}' is not supported. Reverting to Ordinary Least Squares (OLS).")
                formula = 'eui ~ ' + ' + '.join(self.ecm_names)
                model = ols(formula, data=df).fit()
                summary_info = model.summary().as_text()
                with open(model_file.replace(f'_{model_type}.txt', '_ols.txt'), 'w') as f: f.write(summary_info)

            self.surrogate_model = model
            self.surrogate_model_summary = summary_info
            logging.info(f"A surrogate model ({model_type}) has been successfully constructed, and a summary of its key information has been saved to: {model_file}.")
        
        except Exception as e:
            logging.error(f"Error: An issue arose while building the surrogate model: {e}")
            self.surrogate_model = None
            self.surrogate_model_summary = None
    
    def optimize(self, model_type: str=None):
        """
        Employ surrogate modeling to pinpoint the optimal combination of ECM parameters for minimizing Energy Use Intensity (EUI).

        Args:
            model_type (str, optional): Optimization model type. Defaults to None.
        """
        model_type = model_type if model_type else self.config['analysis']['optimization_model']
        logging.info(f"--- {self.unique_id}: Optimizing with a Surrogate Model ({model_type}) ---")

        if self.surrogate_model is None:
            logging.info("Info: The surrogate model hasn't been built yet; attempting to construct it now...")
            self.build_surrogate_model(model_type=model_type)
            if self.surrogate_model is None:
                logging.error("Error: Unable to build agent model, optimization aborted.")
            return
        
        # Define the objective function for the surrogate model
        def objective_function(params_array: np.ndarray) -> float:
            """
            Surrogate model for predicting the EUI objective function.

            Args:
                params_array (np.ndarray): Array of parameters to evaluate

            Returns:
                float: Predicted EUI value
            """
            params_df = pd.DataFrame([params_array], columns=self.ecm_names)
            try:
                # Predict the EUI using the surrogate model
                predicted_eui = self.surrogate_model.predict(params_df)
                return float(predicted_eui[0]) if hasattr(predicted_eui, '__len__') else float(predicted_eui)
            except Exception as e:
                logging.warning(f"Warning: The surrogate model's prediction failed: {e}. Parameters: {params_array}. Returning the maximum value as a fallback.")
                return float('inf')
        
        # Define the bounds for the optimization
        bounds = [] 
        initial_guess = []
        for name in self.ecm_names:
            levels = self.ecm_ranges[name]
            bounds.append((min(levels), max(levels)))
            # Set the initial guess to the midpoint of the range, or the first non-zero value encountered.
            non_zero_levels = [l for l in levels if l != 0]
            initial_guess.append(np.mean(levels) if not non_zero_levels else non_zero_levels[0])

        # Perform the optimization
        logging.info("Starting the optimization process...")
        try:
            result = minimize(
                objective_function, # objective function
                x0=initial_guess, # initial guess
                method='L-BFGS-B', # optimization algorithm
                bounds=bounds, # parameter bounds
                options={'maxiter': 100} # optimization options
            )

            self.optimization_results = result

            if result.success:
                optimal_params_continuous = result.x
                self.optimal_params = self._map_continuous_to_nearest_discrete(optimal_params_continuous) # Map the continuous parameters to the nearest discrete values.
                # Leveraging a surrogate model to predict the EUI of discrete optima
                self.optimal_eui_predicted = objective_function(list(self.optimal_params.values()))
                logging.info(f"Optimization successful. Projected optimal EUI: {self.optimal_eui_predicted:.2f}")
                logging.info("Optimal parameter set (discrete):")
                for name, val in self.optimal_params.items():
                    logging.info(f"{name}: {val:.2f}")
            else:
                logging.error(f"Optimization failed. Reason: {result.message}")
                self.optimal_params = None
                self.optimal_eui_predicted = None
        except Exception as e:
            logging.error(f"Error: An issue arose while optimizing: {e}")
            self.optimal_params = None
            self.optimal_eui_predicted = None

    def _map_continuous_to_nearest_discrete(self, continuous_params: np.ndarray) -> dict:
        """
        Maps the continuous optimization results to the nearest permissible discrete value for each parameter.

        Args:
            continuous_params (np.ndarray): Continuous parameter values from the optimization results.

        Returns:
            dict: Dictionary containing the optimal parameter values mapped to the nearest discrete values.
        """
        discrete_params = {}
        for i, name in enumerate(self.ecm_names):
            continuous_val = continuous_params[i]
            discrete_levels = self.ecm_ranges[name]
            nearest_discrete_val = min(discrete_levels, key=lambda x: abs(x - continuous_val))
            discrete_params[name] = nearest_discrete_val
        return discrete_params
    
    def validate_optimum(self):
        """
        Execute EnergyPlus simulations with optimized parameters to validate the predicted results.
        """
        if self.optimal_params is None:
            logging.error("Error: Optimization parameters not found; verification cannot proceed. Please run `optimize()` first.")
            return
        
        logging.info(f"--- {self.unique_id}: Validating the optimal parameter set ---")
        self.optimal_eui_simulated, floor_area = self._run_single_simulation_internal(
            params_dict=self.optimal_params,
            run_id=f"optimized",
        )
        if floor_area is not None and floor_area > 0:
            self.building_floor_area = floor_area # Update the building floor area

        if self.optimal_eui_simulated:
            logging.info(f"Optimal Simulated EUI: {self.optimal_eui_simulated:.2f} kWh/m².")

            if self.optimal_eui_predicted:
                self.optimization_bias = abs(self.optimal_eui_simulated - self.optimal_eui_predicted) / self.optimal_eui_simulated * 100
                logging.info(f"Optimization bias: {self.optimization_bias:.2f}%")
            else:
                logging.warning("Warning: The predicted EUI is not available. The bias cannot be calculated.")

            if not self.baseline_eui:
                logging.info("Info: The baseline EUI hasn't been calculated. Attempting to run it now...")
                self.run_baseline_simulation()
            
            if self.baseline_eui and self.baseline_eui > 0:
                self.optimization_improvement = (self.baseline_eui - self.optimal_eui_simulated) / self.baseline_eui * 100
                logging.info(f"Info: EUI Improvement (Relative to Baseline): {self.optimization_improvement:.2f}%")

            elif self.baseline_eui == 0:
                logging.warning("Warning: Improvement rate cannot be calculated (baseline EUI is zero).")
            else:
                logging.warning("Warning: The baseline EUI is not available. The improvement rate cannot be calculated.")
        else:
            logging.warning("Warning: The optimal EUI simulation failed. The validation cannot be performed.")

    def run_pv_analysis(self):
        """Find suitable surfaces, add PV, run simulation, and calculate net EUI."""
        if not self.config.get('pv_analysis', {}).get('enabled', False):
            logging.info(f"--- {self.unique_id}: PV analysis is disabled ---")
            return
        if self.optimal_eui_simulated is None or self.optimal_params is None:
            logging.error(f"Error: Optimization or validation not completed, cannot perform PV analysis.")
            return
        if self.building_floor_area is None or self.building_floor_area <= 0:
            logging.error(f"Error: Missing valid floor area, cannot calculate net EUI.")
            return
        logging.info(f"--- {self.unique_id}: Starting PV analysis ---")
        try:
            optimized_idf_obj_path = self.work_dir / "optimized" / "optimized.idf"
            if not optimized_idf_obj_path.exists():
                logging.error(f"Error: Verified optimized IDF not found: {optimized_idf_obj_path}")
                return
            optimized_idf_model = IDFModel(optimized_idf_obj_path) # Load the verified optimized IDF
            pv_manager = PVManager(optimized_idf_model=optimized_idf_model, runner=self.runner, config=self.config,
                                   weather_path=self.weather_path, base_work_dir=self.work_dir)
            self.suitable_surfaces = pv_manager.find_suitable_surfaces() # Find surfaces

            if self.suitable_surfaces:
                pv_run_id = "optimized_pv"
                self.pv_idf_path = pv_manager.add_pv_to_idf(self.suitable_surfaces, pv_run_id) # Add PV
                if self.pv_idf_path:
                    pv_output_dir = self.work_dir / pv_run_id
                    pv_output_prefix = self.config['pv_analysis'].get('pv_output_prefix', 'pv')
                    success, message = self.runner.run_simulation( # Run the PV simulation
                        idf_path=self.pv_idf_path, weather_path=self.weather_path,
                        output_dir=pv_output_dir, output_prefix=pv_output_prefix, config=self.config)
                    if success:
                        self.pv_generation_results = pv_manager.analyze_pv_generation(pv_output_prefix) # Analyze the PV generation
                        pv_result_parser = SimulationResult(pv_output_dir, pv_output_prefix) # Get the total EUI
                        self.gross_eui_with_pv = pv_result_parser.get_source_eui(self.config['constants']['ng_conversion_factor'])
                        if self.gross_eui_with_pv is not None and self.pv_generation_results is not None:
                            total_pv_kwh = self.pv_generation_results.get('total_annual_kwh', 0.0)
                            pv_kwh_per_m2 = total_pv_kwh / self.building_floor_area # Calculate the PV generation intensity
                            self.net_eui_with_pv = self.gross_eui_with_pv - pv_kwh_per_m2 # Calculate the net EUI
                            logging.info(f"Gross EUI with PV: {self.gross_eui_with_pv:.2f} kWh/m2")
                            logging.info(f"Annual PV generation: {total_pv_kwh:.2f} kWh ({pv_kwh_per_m2:.2f} kWh/m2)")
                            logging.info(f"Net source EUI (Net): {self.net_eui_with_pv:.2f} kWh/m2")
                        else: logging.warning("Warning: Unable to obtain the total EUI with PV or PV generation data, unable to calculate the net EUI.")
                    else:
                        logging.error(f"Error: Final PV simulation failed: {message}")
                else:
                    logging.error("Error: Failed to add PV to IDF.")
            else:
                logging.info("Info: No suitable surfaces found for PV installation. Skipping PV simulation.")
                self.net_eui_with_pv = self.optimal_eui_simulated
        except Exception as e:
            logging.error(f"Error: An issue arose while executing the PV analysis: {e}")
            import traceback; traceback.print_exc()
    
    def save_results(self):
        """
        Save key results from the process to a file 
        """
        results_data = {
            "city": self.city,
            "ssp": self.ssp,
            "btype": self.btype,
            'building_floor_area_m2': self.building_floor_area,
            "baseline_eui": self.baseline_eui,
            "optimal_params": self.optimal_params,
            "optimal_eui_predicted": self.optimal_eui_predicted,
            "optimal_eui_simulated": self.optimal_eui_simulated,
            "optimization_improvement_percent": self.optimization_improvement,
            "optimization_bias_percent": self.optimization_bias,
            'pv_analysis_enabled': self.config.get('pv_analysis', {}).get('enabled', False),
            'suitable_surfaces_for_pv': self.suitable_surfaces,
            'pv_generation_analysis': self.pv_generation_results,
            'gross_eui_with_pv': self.gross_eui_with_pv,
            'net_eui_with_pv': self.net_eui_with_pv,
            "sensitivity_indices": self.sensitivity_results,
        }
        result_file = self.work_dir / "pipeline_results.json"
        try:
            def numpy_converter(obj): # Process NumPy types
                if isinstance(obj, np.integer): return int(obj)
                elif isinstance(obj, np.floating): return float(round(obj, 6)) if not np.isnan(obj) else "NaN"
                elif isinstance(obj, np.ndarray): return obj.tolist()
                elif isinstance(obj, (np.bool_, bool)): return bool(obj)
                elif pd.isna(obj): return "NaN"
                # Process NumPy arrays in SALib output dictionaries
                if isinstance(obj, dict):
                    return {k: numpy_converter(v) for k, v in obj.items()}
                return '<not serializable>'
            with open(result_file, "w") as f:
                json.dump(results_data, f, indent=4, default=numpy_converter)
            logging.info(f"Optimization results saved to {result_file}")
        except Exception as e:
            logging.error(f"Error: Failed to save results to {result_file}: {e}")

    def run_full_pipeline(self, run_sens:bool=True, build_model:bool=True, run_opt=True, validate=True, run_pv=True, save=True):
        """
        Execute each stage of the entire optimization process sequentially.

        Args:
            run_sens (bool, optional): Run the sensitivity analysis. Defaults to True.
            build_model (bool, optional): Build the surrogate model. Defaults to True.
            run_opt (bool, optional): Run the optimization. Defaults to True.
            validate (bool, optional): Validate the results. Defaults to True.
            save (bool, optional): Save the results. Defaults to True.
        """
        logging.info(f"======== Start processing: {self.unique_id} ========")
        if self.run_baseline_simulation() is None and (run_sens or validate):
            logging.error("Error: The baseline EUI hasn't been calculated. The pipeline cannot proceed.")
            return
        
        if run_sens:
            self.run_sensitivity_analysis()
            if self.sensitivity_results is None:
                logging.error("Error: Sensitivity analysis failed, subsequent steps may be impacted.")
        
        if build_model:
            self.build_surrogate_model()
            if self.surrogate_model is None:
                logging.error("Error: The attempt to build a surrogate model has failed, precluding optimization and validation.")

        if run_opt:
            self.optimize()
            if self.optimization_results is None:
                logging.error("Error: Optimization failed, preventing validation from proceeding.")
                return 
        
        if validate:
            self.validate_optimum()
            if self.optimal_eui_simulated is None and run_pv:
                logging.error("Error: Optimization validation failed, aborting PV analysis.")
                return

        if run_pv and self.config.get('pv_analysis', {}).get('enabled', False): # Check if PV analysis is enabled in the configuration
            self.run_pv_analysis()
        
        if save:
            self.save_results()
        
        logging.info(f"======== End processing: {self.unique_id} ========")

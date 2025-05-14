import os
import xgboost as xgb
import optuna
import joblib
import numpy as np
import pandas as pd
import shutil
import logging
import json
from joblib import Parallel, delayed
from SALib.sample import saltelli
from SALib.analyze import sobol

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import root_mean_squared_error
from statsmodels.formula.api import ols
from pathlib import Path

# For L-BFGS-B
from scipy.optimize import minimize

# For GA
import random
from deap import base, creator, tools, algorithms


# Local imports
from .pv_service import PVManager
from .idf_service import IDFModel
from .simulation_service import EnergyPlusRunner, SimulationResult

logger = logging.getLogger(__name__)


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
        # Path to the prototype IDF file
        self.prototype_idf_path = self._find_prototype_idf()
        self.weather_path = self._find_weather_file()  # Path to the weather file

        # WorkDirectory Configuration
        self.work_dir = self.config['paths']['epsim_dir'] / self.unique_id
        self.work_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Simulation Components
        self.runner = EnergyPlusRunner(
            self.config['paths']['eplus_executable'])
        self.ecm_ranges = config['ecm_ranges']
        self.ecm_names = list(self.ecm_ranges.keys())

        # Initialize Results
        self.baseline_eui = None  # Benchmark EUI
        self.building_floor_area = None  # Building floor area
        # Sample sensitivity analysis (discrete)
        self.sensitivity_samples = None
        self.sensitivity_results = None  # Sobol Analysis Results (Indices)
        self.surrogate_model = None  # Surrogate Model
        # Surrogate Model Summary or Feature Importance
        self.surrogate_model_summary = None
        self.optimization_result = None  # Raw Results from scipy.optimize
        self.optimal_params = None  # Optimized parameter set achieving the best results
        self.optimal_eui_predicted = None  # Optimal EUI Prediction via Surrogate Model
        self.optimal_eui_simulated = None  # EUI Optimized Through Real-World Simulation
        self.optimization_improvement = None  # EUI Improvement Rate
        self.optimization_bias = None  # Optimization bias
        self.suitable_surfaces = None  # Suitable surfaces for PV installation
        self.pv_idf_path = None  # Path to the IDF file with PV
        self.pv_generation_results = None  # PV generation results
        self.gross_eui_with_pv = None  # Gross EUI with PV
        self.net_eui_with_pv = None  # Net EUI with PV

    def _find_prototype_idf(self):
        """
        Find the prototype IDF file for the given city and building type
        """
        proto_dir = self.config['paths']['prototypes_dir']
        if not proto_dir.exists():
            logger.error(f"Prototype directory not found: {proto_dir}")
            return None
        found = [f for f in proto_dir.glob(
            "*.idf") if self.btype.lower() in f.stem.lower()]
        if not found:
            logger.error(f"No prototype IDF file found for {self.btype}")
            return None
        logger.info(f"Found prototype IDF file: {found[0]}")
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
            logger.error(f"Weather directory not found: {epw_dir}")
            return None
        found = [f for f in epw_dir.glob(
            "*.epw") if search_city in f.stem.lower() and search_ssp in f.stem.lower()]
        if not found:
            logger.error(
                f"No weather file found for {self.city} and {self.ssp}")
            return None
        logger.info(f"Found weather file: {found[0]}")
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
            logger.error(f"No discrete levels defined for {param_name}")
            raise ValueError(f"Unknown ECM Parameter: {param_name}")
        if not isinstance(discrete_levels, list) or len(discrete_levels) == 0:
            logger.error(
                f"Invalid discrete levels for {param_name}: {discrete_levels}")
            raise ValueError(
                f"The provided discrete levels for parameter '{param_name}' are invalid: {discrete_levels}")

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
                pass  # 暂时跳过单独修改 U 值

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
                    logger.warning(
                        f"No lighting reduction map found for {btype_key}")

                # Retrieve the reduction factor for the specified lighting level.
                reduction_factor = inner_map.get(lighting_level, 1.0)

                if reduction_factor < 1.0:
                    idf_model.apply_lighting_reduction(
                        reduction_factor, btype_key)
                elif reduction_factor >= 1.0 and lighting_level != 0:
                    logger.warning(
                        f"Lighting reduction factor ({reduction_factor}) found is either invalid or not required for building type '{self.btype}' and level '{lighting_level}'. Lighting is not modified.")

            elif param_name in ['shgc', 'win_u', 'vt']:
                # Process Window properties
                win_u = params_dict.get('win_u', 0)
                shgc = params_dict.get('shgc', 0)
                vt = params_dict.get('vt', 0)
                if win_u > 0 and shgc > 0:
                    idf_model.apply_window_properties(win_u, shgc, vt)
                elif win_u > 0:
                    # Only update U-value while keeping SHGC and VT the same
                    pass  # Not implemented yet
                elif shgc > 0:
                    # Only update SHGC while keeping U-value and VT the same
                    pass  # Not implemented yet
                elif vt > 0:
                    # Only update VT while keeping U-value and SHGC the same
                    pass  # Not implemented yet

    def _run_single_simulation_internal(self, params_dict: dict = {}, run_id: str = None, is_baseline: bool = False, output_intermediary_files: bool = True):
        """
        Set up, execute, and analyze a single EnergyPlus simulation.

        Args:
            params_dict (dict, optional): A dictionary containing ECM parameter values. An empty dictionary signifies the baseline run. Defaults to {}.
            run_id (str, optional): Run identifier for naming output subdirectories and file prefixes.. Defaults to None.
            is_baseline (bool, optional): Explicitly designated as a benchmark run. Defaults to False.
            output_intermediary_files (bool, optional): Whether to output intermediary files. Defaults to True.
        """

        run_label = str(
            run_id) if run_id is not None else 'baseline' if is_baseline else 'ecm_run'
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
            idf.apply_run_peroid(
                self.config['simulation']['start_year'], self.config['simulation']['end_year'])
            idf.apply_output_requests()  # Apply standard output requests
            idf.apply_simulation_control_settings()

            if not is_baseline and params_dict:
                self._apply_ecms_to_idf(idf, params_dict)
            floor_area = idf.get_total_floor_area()  # Get the total floor area
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
                eui = result_parser.get_source_eui(
                    self.config['constants']['ng_conversion_factor'])
                if eui is None:
                    logger.error(
                        f"Failed to calculate source EUI for run {run_label}")
                if not output_intermediary_files:
                    shutil.rmtree(run_dir, ignore_errors=True)
                return eui, floor_area
            else:
                logger.error(
                    f"Simulation failed for run {run_label}: {message}")
                return None, floor_area

        except Exception as e:
            logger.error(
                f"An error occurred while running simulation {run_label}: {e}")
            shutil.rmtree(run_dir, ignore_errors=True)
            return None, floor_area

    def run_baseline_simulation(self):
        """
        Run a baseline simulation and save EUI
        """
        logger.info(f"Running baseline simulation for {self.unique_id}")
        self.baseline_eui, self.building_floor_area = self._run_single_simulation_internal(
            is_baseline=True, run_id="baseline")
        if self.baseline_eui is not None:
            logger.info(
                f"Baseline simulation completed successfully. Reference source EUI: {self.baseline_eui} kWh/m2/yr")
        else:
            logger.error(f"Baseline simulation failed for {self.unique_id}")
        if self.building_floor_area is not None and self.building_floor_area > 0:
            logger.info(
                f"Building floor area: {self.building_floor_area:.2f} m2")
        else:
            logger.warning(
                f"Warning: Unable to obtain a valid building floor area.")
        return self.baseline_eui

    def _run_sensitivity_sample_point(self, params_array: np.ndarray, sample_index: int) -> dict | None:
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
        eui, _ = self._run_single_simulation_internal(
            params_dict=params_dict, run_id=run_id, output_intermediary_files=False)
        logger.info(f"Completed sample {sample_index} with EUI: {eui}")
        if eui is not None:
            result_dict = params_dict.copy()
            result_dict['eui'] = eui
            return result_dict
        else:
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
        # Get the number of continuous samples
        num_continuous = continuous_samples.shape[0]
        # Initialize the EUI array with NaNs
        Y = np.full(num_continuous, np.nan)

        # Create a lookup dictionary based on discrete parameter combinations for efficiency
        discrete_lookup = {}
        # Iterate through the discrete results DataFrame
        for index, row in discrete_results_df.iterrows():
            key = tuple(row[self.ecm_names].values)
            discrete_lookup[key] = row['eui']

        for i in range(num_continuous):
            discrete_params_list = []
            # Iterate through each parameter
            for j, param_name in enumerate(self.ecm_names):
                # Mapping the continuous sample to a discrete value
                discrete_val = self._continuous_to_discrete(
                    continuous_samples[i, j], param_name)
                discrete_params_list.append(discrete_val)
            # Converting discrete parameter lists to tuples for lookups
            discrete_key = tuple(discrete_params_list)
            # Retrieving the EUI from the dictionary.
            if discrete_key in discrete_lookup:
                Y[i] = discrete_lookup[discrete_key]
            else:
                # If an exact match can't be found among the discrete results.
                logger.warning(
                    f"Warning: Exact match: {discrete_key} was not found for sample {i} in the discrete result. the EUI will be NaN.")

        # Check if any NaNs are present in the EUI array
        nan_count = np.isnan(Y).sum()
        if nan_count > 0:
            logger.warning(
                f"Warning: {nan_count}/{num_continuous} consecutive samples failed to find the corresponding discrete EUI.")

        return Y

    def run_sensitivity_analysis(self):
        """
        Perform the Sobol sensitivity analysis process.
        """
        logger.info(f"--- {self.unique_id}: Run sensitivity analysis ---")
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
        param_values_continuous = saltelli.sample(
            problem, N_samples_base, calc_second_order=True)

        # Transform the continuous samples to discrete samples
        param_values_discrete_unique = set()
        for i in range(param_values_continuous.shape[0]):
            discrete_sample = tuple(self._continuous_to_discrete(param_values_continuous[i, j], name)
                                    for j, name in enumerate(self.ecm_names))
            param_values_discrete_unique.add(discrete_sample)

        param_values_discrete_list = [list(s)
                                      for s in param_values_discrete_unique]
        num_unique_samples = len(param_values_discrete_list)
        logger.info(
            f"Generated {param_values_continuous.shape[0]} continuous samples, which map to {num_unique_samples} unique discrete parameter combinations.")

        # Run Simulation for Discrete Samples in Parallel
        discrete_results_file = self.work_dir / 'sensitivity_discrete_results.csv'
        if not discrete_results_file.exists():
            # Set n_jobs to 1 if debug, otherwise use all cores.
            n_jobs = 1 if self.config['constants']['debug'] else num_cpu

            logger.info(
                f"Running {num_unique_samples} discrete sample simulations in parallel (using {n_jobs} cores)...")
            results_list = Parallel(n_jobs=n_jobs, verbose=10)(
                delayed(self._run_sensitivity_sample_point)(params, i)
                for i, params in enumerate(param_values_discrete_list)
            )

            successful_results = [r for r in results_list if r is not None]
            if not successful_results:
                logger.error(
                    "No successful simulations completed during sensitivity analysis.")
                return

            self.sensitivity_samples = pd.DataFrame(successful_results)
            self.sensitivity_samples.to_csv(discrete_results_file, index=False)
            logger.info(
                f"Discrete sensitivity analysis results saved to {discrete_results_file}")
        else:
            logger.info(
                f"Loading discrete sensitivity analysis results from {discrete_results_file}")
            self.sensitivity_samples = pd.read_csv(discrete_results_file)

        # Refilling the EUI for Continuous Sample Spaces.
        logger.info(
            "Re-populating the EUI of a continuous sample space for Sobol analysis...")
        Y = self._refill_continuous_space(
            param_values_continuous, self.sensitivity_samples)
        if np.isnan(Y).all():
            logger.error(
                "Error: Unable to locate EUIs for any consecutive samples, Sobol analysis cannot be performed.")
            return

        # Handle NaN in Y (e.g., remove corresponding samples or padding, choose remove here)
        valid_indices = ~np.isnan(Y)
        if not np.all(valid_indices):
            logger.warning(
                f"Warning: {np.isnan(Y).sum()} samples with NaN EUI values were removed from the analysis.")
            param_values_continuous_valid = param_values_continuous[valid_indices]
            Y_valid = Y[valid_indices]
            if len(Y_valid) < 2:
                logger.error(
                    "Error: Not enough valid samples for Sobol analysis.")
                return
        else:
            param_values_continuous_valid = param_values_continuous
            Y_valid = Y

        # Perform Sobol Analysis
        logger.info("Performing Sobol sensitivity analysis...")
        try:
            Si = sobol.analyze(
                problem, Y_valid, calc_second_order=True, print_to_console=True)
            self.sensitivity_results = Si
        except Exception as e:
            logger.error(
                f"Error: An issue arose while performing the Sobol analysis: {e}")
            return

        first_order = pd.Series(Si['S1'], index=problem['names'], name='S1')
        total_order = pd.Series(Si['ST'], index=problem['names'], name='ST')
        sensitivity_indices = pd.concat([first_order, total_order], axis=1)
        si_file = self.work_dir / 'sensitivity_indices.csv'
        sensitivity_indices.to_csv(si_file)
        logger.info(f"Sobol sensitivity indices saved to {si_file}")
        if 'S2' in Si and Si['S2'] is not None:
            second_order = pd.DataFrame(
                Si['S2'], index=problem['names'], columns=problem['names'])
            s2_file = self.work_dir / 'sensitivity_indices_S2.csv'
            second_order.to_csv(s2_file)
            logger.info(
                f"Sobol second-order sensitivity indices saved to {s2_file}")

    def build_surrogate_model(self, model_type: str = None):
        """Builds a surrogate model based on sensitivity analysis results.

        This method acts as a dispatcher to specific model building methods
        based on the model_type.

        Args:
            model_type (str, optional): The type of surrogate model to build.
                If None, uses the type specified in the configuration.
                Defaults to None.
        """
        model_type = model_type if model_type else self.config['analysis']['surrogate_model']
        logger.info(
            f"Attempting to build a {model_type.upper()} surrogate model for {self.unique_id}...")

        if self.sensitivity_samples is None:
            samples_file = self.work_dir / 'sensitivity_discrete_results.csv'
            if samples_file.exists():
                logger.info(
                    f"Loading discrete sample data from: {samples_file}")
                self.sensitivity_samples = pd.read_csv(samples_file)
            else:
                logger.error(
                    "Error: Sensitivity analysis sample data not found. "
                    "Cannot build surrogate model. Please run `run_sensitivity_analysis()` first."
                )
                return

        df_clean = self.sensitivity_samples.dropna(
            subset=['eui'] + self.ecm_names)
        if df_clean.empty:
            logger.error(
                "Error: No valid (non-NaN) sample data available after cleaning. Cannot build surrogate model.")
            return

        X_all = df_clean[self.ecm_names]
        Y_all = df_clean['eui']

        if X_all.empty or Y_all.empty:
            logger.error(
                "Error: Feature (X) or target (Y) data is empty. Cannot build surrogate model.")
            return

        model_type_lower = model_type.lower()
        model = None
        summary_info_dict = {}  # Store various summary pieces

        try:
            if model_type_lower == 'ols':
                model, summary_info_dict = self._build_ols_model(X_all, Y_all)
            elif model_type_lower == 'rf':
                model, summary_info_dict = self._build_rf_model(X_all, Y_all)
            elif model_type_lower == 'xgb':
                model, summary_info_dict = self._build_xgboost_model(
                    X_all, Y_all)
            else:
                logger.warning(
                    f"Unsupported surrogate model type: {model_type}. Defaulting to OLS."
                )
                model, summary_info_dict = self._build_ols_model(X_all, Y_all)
                model_type_lower = 'ols'  # Update for correct file naming

            if model:
                self.surrogate_model = model
                # Construct a comprehensive summary string from the dictionary
                summary_string_parts = [f"{k.replace('_', ' ').title()}:\n{v}\n" for k, v in summary_info_dict.items(
                ) if (isinstance(v, pd.Series) and not v.empty) or (not isinstance(v, pd.Series) and v)]
                self.surrogate_model_summary = "\n".join(summary_string_parts)

                # --- Unified File Saving ---
                model_file_base_name = f"surrogate_model_{model_type_lower}"
                summary_file_path = self.work_dir / \
                    f"{model_file_base_name}_summary.txt"

                with open(summary_file_path, 'w') as f:
                    f.write(
                        f"{model_type_lower.upper()} Surrogate Model Details for {self.unique_id}\n")
                    f.write("=" * 50 + "\n")
                    if 'parameters' in summary_info_dict and summary_info_dict['parameters']:
                        f.write("Model Parameters:\n")
                        if isinstance(summary_info_dict['parameters'], dict):
                            json.dump(
                                summary_info_dict['parameters'], f, indent=4)
                        else:
                            f.write(str(summary_info_dict['parameters']))
                        f.write("\n\n")
                    # OLS summary
                    if 'summary_text' in summary_info_dict and summary_info_dict['summary_text']:
                        f.write("Model Summary / Statistics:\n")
                        f.write(str(summary_info_dict['summary_text']))
                        f.write("\n\n")
                    if 'feature_importances' in summary_info_dict and summary_info_dict['feature_importances'] is not None:
                        f.write("Feature Importances:\n")
                        # Assuming it's a Pandas Series
                        f.write(
                            summary_info_dict['feature_importances'].to_string())
                        f.write("\n\n")
                        # Save importances to CSV
                        importances_csv_path = self.work_dir / \
                            f"{model_file_base_name}_importances.csv"
                        try:
                            summary_info_dict['feature_importances'].to_csv(
                                importances_csv_path)
                            logger.info(
                                f"Feature importances saved to {importances_csv_path}")
                        except Exception as e_csv:
                            logger.error(
                                f"Error saving feature importances to CSV {importances_csv_path}: {e_csv}")

                logger.info(
                    f"{model_type_lower.upper()} surrogate model built. "
                    f"Details saved to {summary_file_path}"
                )

            else:
                logger.error(
                    f"Failed to build surrogate model of type: {model_type_lower}")
                self.surrogate_model = None
                self.surrogate_model_summary = None

        except Exception as e:
            logger.error(
                f"An unexpected error occurred during build_surrogate_model for type {model_type_lower}: {e}", exc_info=True)
            self.surrogate_model = None
            self.surrogate_model_summary = None

    def _build_ols_model(self, X: pd.DataFrame, Y: pd.Series) -> tuple[object | None, dict]:
        """Builds an Ordinary Least Squares (OLS) surrogate model.

        Args:
            X (pd.DataFrame): DataFrame of features.
            Y (pd.Series): Series of target EUI values.

        Returns:
            tuple[object | None, dict]: A tuple containing the
                trained OLS model object (or None if failed) and a dictionary
                containing summary information (e.g., 'summary_text').
        """
        logger.info(f"Building OLS model for {self.unique_id}...")
        summary_info_dict = {'parameters': 'OLS defaults'}
        try:
            df_for_ols = pd.concat([X, Y.rename('eui')], axis=1)
            formula = 'eui ~ ' + ' + '.join(self.ecm_names)
            model = ols(formula, data=df_for_ols).fit()
            summary_info_dict['summary_text'] = model.summary().as_text()
            logger.info(f"OLS model built successfully for {self.unique_id}.")
            return model, summary_info_dict
        except Exception as e:
            logger.error(
                f"Error building OLS model for {self.unique_id}: {e}", exc_info=True)
            return None, {'summary_text': f"Failed to build OLS model: {e}"}

    def _build_rf_model(self, X: pd.DataFrame, Y: pd.Series) -> tuple[object | None, dict]:
        """Builds a Random Forest (RF) surrogate model.

        Args:
            X (pd.DataFrame): DataFrame of features.
            Y (pd.Series): Series of target EUI values.

        Returns:
            tuple[object | None, dict]: A tuple containing the
                trained RF model object (or None if failed) and a dictionary
                containing summary information (e.g., 'feature_importances').
        """
        logger.info(f"Building Random Forest model for {self.unique_id}...")
        rf_params = {
            'n_estimators': self.config['analysis'].get('n_estimators', 100),
            'random_state': self.config['analysis'].get('random_state', 10),
            'n_jobs': self.config['constants'].get('cpu_count_override', -1)
            # Add more RF params from config if needed
        }
        summary_info_dict = {'parameters': rf_params}

        try:
            model = RandomForestRegressor(**rf_params)
            model.fit(X, Y)
            importances = pd.Series(
                model.feature_importances_, index=self.ecm_names).sort_values(ascending=False)
            summary_info_dict['feature_importances'] = importances
            logger.info(
                f"Random Forest model built successfully for {self.unique_id}.")
            return model, summary_info_dict
        except Exception as e:
            logger.error(
                f"Error building Random Forest model for {self.unique_id}: {e}", exc_info=True)
            summary_info_dict['feature_importances'] = None
            summary_info_dict['error'] = f"Failed to build RF model: {e}"
            return None, summary_info_dict

    def _build_xgboost_model(self, X_all: pd.DataFrame, Y_all: pd.Series) -> tuple[object | None, dict]:
        """Builds an XGBoost surrogate model, with optional Optuna tuning and model persistence.

        Args:
            X_all (pd.DataFrame): DataFrame of all available features.
            Y_all (pd.Series): Series of all available target EUI values.

        Returns:
            tuple[object | None, dict]: A tuple containing the
                trained XGBoost model object (or None if failed) and a dictionary
                containing summary information (e.g., 'best_params', 'feature_importances').
        """
        logger.info(f"Building XGBoost model for {self.unique_id}...")
        summary_info_dict = {}

        xgb_config = self.config['analysis'].get('xgboost_params', {})
        model_filename = xgb_config.get(
            'model_save_filename', f"xgboost_surrogate_{self.unique_id}.json")
        model_save_path = self.work_dir / model_filename
        best_params_filename = xgb_config.get(
            'best_params_filename', f"xgboost_best_params_{self.unique_id}.json")
        best_params_path = self.work_dir / best_params_filename

        # 1. Attempt to load saved model
        if xgb_config.get('load_saved_model', False) and model_save_path.exists():
            try:
                loaded_model = xgb.XGBRegressor()
                loaded_model.load_model(model_save_path)
                logger.info(
                    f"Loaded saved XGBoost model from {model_save_path} for {self.unique_id}.")

                # Load best_params if available
                best_params_loaded = {}
                if best_params_path.exists():
                    with open(best_params_path, 'r') as f:
                        best_params_loaded = json.load(f)
                    summary_info_dict['parameters'] = best_params_loaded
                    logger.info(
                        f"Loaded best parameters from {best_params_path}")
                else:
                    summary_info_dict['parameters'] = "Parameters not found, using loaded model's internal params."

                importances = pd.Series(
                    loaded_model.feature_importances_, index=X_all.columns).sort_values(ascending=False)
                summary_info_dict['feature_importances'] = importances
                return loaded_model, summary_info_dict
            except Exception as e:
                logger.warning(
                    f"Failed to load saved XGBoost model from {model_save_path} for {self.unique_id}: {e}. "
                    "Proceeding to train a new model."
                )

        # 2. Data Split for training and (XGBoost's internal) validation/early stopping
        # This validation set is for the final model training's early stopping,
        # Optuna will do its own internal CV or splits.
        val_split_ratio = xgb_config.get('validation_split_ratio', 0.2)
        if val_split_ratio <= 0 or val_split_ratio >= 1:  # Ensure some data for both
            X_train, Y_train = X_all, Y_all
            # Empty val set if no split
            X_val, Y_val = X_all.iloc[0:0], Y_all.iloc[0:0]
            logger.info(
                f"Using all data for training XGBoost for {self.unique_id}, no validation set for early stopping.")
        else:
            X_train, X_val, Y_train, Y_val = train_test_split(
                X_all, Y_all, test_size=val_split_ratio, random_state=self.config['analysis'].get(
                    'random_state', 10)
            )
            logger.info(
                f"Split data for XGBoost training: {len(X_train)} train, {len(X_val)} validation samples for {self.unique_id}."
            )

        # 3. Hyperparameter Acquisition
        best_params = {}
        if xgb_config.get('use_optuna_tuning', False):
            logger.info(
                f"Starting Optuna hyperparameter tuning for XGBoost for {self.unique_id}...")
            try:
                # Pass the training portion of the data to Optuna
                best_params = self._tune_xgboost_with_optuna(X_train, Y_train)
                # Store Optuna chosen params
                summary_info_dict['optuna_best_params'] = best_params
            except Exception as e_optuna:
                logger.error(
                    f"Optuna tuning failed for {self.unique_id}: {e_optuna}. Falling back to fixed params.", exc_info=True)
                best_params = xgb_config.get(
                    'fixed_params', {}).copy()  # Use copy
        else:
            logger.info(
                f"Using fixed hyperparameters for XGBoost for {self.unique_id}.")
            best_params = xgb_config.get('fixed_params', {}).copy()  # Use copy

        # Ensure essential params are present, if not provided by Optuna or fixed_params
        best_params.setdefault('objective', 'reg:squarederror')
        best_params.setdefault(
            'random_state', self.config['analysis'].get('random_state', 10))
        # These are the final params used for training
        summary_info_dict['parameters'] = best_params

        # 4. Train Final Model
        logger.info(
            f"Training final XGBoost model for {self.unique_id} with parameters: {best_params}")
        try:
            final_model = xgb.XGBRegressor(
                **best_params, early_stopping_rounds=xgb_config.get('early_stopping_rounds', 10))
            fit_params = {}
            if not X_val.empty:  # Only add eval_set if X_val is not empty
                fit_params['eval_set'] = [(X_val, Y_val)]

            final_model.fit(X_train, Y_train, verbose=False, **fit_params)

            # 5. Save Model
            if model_save_path:
                try:
                    model_save_path.parent.mkdir(
                        parents=True, exist_ok=True)  # Ensure directory exists
                    final_model.save_model(model_save_path)
                    logger.info(
                        f"Saved trained XGBoost model to {model_save_path} for {self.unique_id}.")
                    # Save the best_params to JSON alongside the model
                    with open(best_params_path, 'w') as f:
                        json.dump(best_params, f, indent=4, sort_keys=True)
                    logger.info(
                        f"Saved XGBoost best_params to {best_params_path} for {self.unique_id}.")
                except Exception as e_save:
                    logger.error(
                        f"Error saving XGBoost model or params for {self.unique_id}: {e_save}", exc_info=True)

            # 6. Extract Feature Importances and Return
            importances = pd.Series(
                final_model.feature_importances_, index=X_all.columns).sort_values(ascending=False)
            summary_info_dict['feature_importances'] = importances
            logger.info(
                f"XGBoost model built and trained successfully for {self.unique_id}.")
            return final_model, summary_info_dict

        except Exception as e_train:
            logger.error(
                f"Error training final XGBoost model for {self.unique_id}: {e_train}", exc_info=True)
            summary_info_dict['error'] = f"Failed to train XGBoost model: {e_train}"
            return None, summary_info_dict

    def _tune_xgboost_with_optuna(self, X_train_outer: pd.DataFrame, Y_train_outer: pd.Series) -> dict:
        """Tunes XGBoost hyperparameters using Optuna.

        This method defines an objective function for Optuna, which internally
        uses K-Fold cross-validation on the provided training data to evaluate
        each hyperparameter set.

        Args:
            X_train_outer (pd.DataFrame): Features for Optuna tuning (will be split by KFold).
            Y_train_outer (pd.Series): Target for Optuna tuning.

        Returns:
            dict: A dictionary containing the best hyperparameters found by Optuna.
        """
        xgb_config = self.config['analysis'].get('xgboost_params', {})
        optuna_param_space_config = xgb_config.get('optuna_param_space', {})
        fixed_params_for_tuning = xgb_config.get('fixed_params', {}).copy()
        fixed_params_for_tuning.setdefault('objective', 'reg:squarederror')
        fixed_params_for_tuning.setdefault(
            'random_state', self.config['analysis'].get('random_state', 10))

        def objective(trial: optuna.trial.Trial) -> float:
            params = {}
            for name, p_config in optuna_param_space_config.items():
                param_type = p_config.get('type', 'float')
                if param_type == 'int':
                    params[name] = trial.suggest_int(name, p_config['low'], p_config['high'], step=p_config.get(
                        'step', 1), log=p_config.get('log', False))
                elif param_type == 'float':
                    params[name] = trial.suggest_float(name, p_config['low'], p_config['high'], step=p_config.get(
                        'step'), log=p_config.get('log', False))
                elif param_type == 'categorical':
                    params[name] = trial.suggest_categorical(
                        name, p_config['choices'])
                # Add more types if needed (e.g., discrete_uniform)

            # Add fixed params like objective, random_state
            params.update(fixed_params_for_tuning)

            # K-Fold Cross-validation for robust evaluation
            kf = KFold(n_splits=xgb_config.get('optuna_cv_folds', 3),
                       shuffle=True,
                       random_state=self.config['analysis'].get('random_state', 10))

            scores_rmse = []
            for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_outer, Y_train_outer)):
                X_cv_train, X_cv_val = X_train_outer.iloc[train_idx], X_train_outer.iloc[val_idx]
                Y_cv_train, Y_cv_val = Y_train_outer.iloc[train_idx], Y_train_outer.iloc[val_idx]

                model_cv = xgb.XGBRegressor(
                    **params, early_stopping_rounds=xgb_config.get('early_stopping_rounds', 10))
                try:
                    model_cv.fit(X_cv_train, Y_cv_train,
                                 eval_set=[(X_cv_val, Y_cv_val)],
                                 verbose=False)
                    preds = model_cv.predict(X_cv_val)
                    rmse = root_mean_squared_error(Y_cv_val, preds)
                    scores_rmse.append(rmse)
                except Exception as e_cv:
                    logger.warning(
                        f"Optuna CV fold {fold+1} failed for trial {trial.number} with params {params}: {e_cv}")
                    return float('inf')  # Penalize failed trials

            if not scores_rmse:  # All folds failed
                logger.warning(
                    f"Optuna trial {trial.number} failed across all CV folds with params {params}.")
                return float('inf')

            return sum(scores_rmse) / len(scores_rmse)  # Average RMSE

        study_name = f"xgboost-tuning-{self.unique_id}"
        # You might want to configure Optuna storage for persistent studies
        # storage_name = f"sqlite:///{self.work_dir / 'optuna_study.db'}"
        # study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True, direction='minimize')
        study = optuna.create_study(
            study_name=study_name, direction='minimize')

        try:
            study.optimize(objective,
                           n_trials=xgb_config.get('optuna_n_trials', 50),
                           timeout=xgb_config.get('optuna_timeout', None),
                           n_jobs=xgb_config.get('optuna_n_jobs', 1))  # Optuna can parallelize trials
        except optuna.exceptions.OptunaError as e_opt_study:
            logger.error(
                f"Optuna study optimization encountered an error for {self.unique_id}: {e_opt_study}", exc_info=True)
            # Fallback or re-raise, here we'll return current best or empty
            if study.best_trial:
                logger.warning(
                    "Optuna study stopped prematurely, returning best trial so far.")
                return study.best_params
            else:
                logger.warning(
                    "Optuna study stopped prematurely with no successful trials. Returning empty params.")
                return {}

        logger.info(
            f"Optuna study for {self.unique_id} best trial RMSE: {study.best_value:.4f}")
        logger.info(
            f"Optuna study for {self.unique_id} best params: {study.best_params}")

        # Combine Optuna's best params with any other fixed params that weren't part of the search space
        final_best_params = fixed_params_for_tuning.copy()
        final_best_params.update(study.best_params)

        return final_best_params

    def optimize(self, model_type: str = None): # model_type is for surrogate, not optimizer
        """Optimizes ECM parameters using the built surrogate model and a configured optimizer.

        This method dispatches to specific optimization algorithms based on configuration.

        Args:
            model_type (str, optional): The type of surrogate model to build if not already built.
                Defaults to the one specified in the configuration.
        """
        optimizer_config_name = self.config['analysis'].get('optimizer_type', 'lbfgsb').lower()
        logging.info(
            f"--- {self.unique_id}: Starting optimization with optimizer: {optimizer_config_name.upper()} ---"
        )

        if self.surrogate_model is None:
            surrogate_model_type_to_build = model_type if model_type else self.config['analysis']['optimization_model']
            logging.info(
                f"Surrogate model not built for {self.unique_id}. Attempting to build "
                f"{surrogate_model_type_to_build.upper()} model now..."
            )
            self.build_surrogate_model(model_type=surrogate_model_type_to_build)
            if self.surrogate_model is None:
                logging.error(
                    f"Failed to build surrogate model for {self.unique_id}. Optimization aborted."
                )
                self.optimal_params = None
                self.optimal_eui_predicted = None
                self.optimization_results = None
                return

        # Clear previous results before running a new optimization
        self.optimal_params = None
        self.optimal_eui_predicted = None
        self.optimization_results = None # Store raw optimizer result/log

        opt_params, opt_eui, raw_result = None, None, None

        if optimizer_config_name == 'lbfgsb':
            opt_params, opt_eui, raw_result = self._optimize_with_lbfgsb()
        elif optimizer_config_name == 'ga':
            opt_params, opt_eui, raw_result = self._optimize_with_deap_ga()
        else:
            logging.error(f"Unsupported optimizer_type: {optimizer_config_name} for {self.unique_id}. Aborting.")
            return

        self.optimal_params = opt_params
        self.optimal_eui_predicted = opt_eui
        self.optimization_results = raw_result # Store the raw result (e.g., scipy result object or DEAP logbook)

        if self.optimal_params is not None and self.optimal_eui_predicted is not None:
            # Check for inf EUI which indicates an issue
            if self.optimal_eui_predicted == float('inf'):
                logging.error(
                    f"Optimization with {optimizer_config_name.upper()} for {self.unique_id} "
                    f"resulted in an infinite EUI. This indicates a problem with the surrogate or optimization."
                )
                self.optimal_params = None # Invalidate if EUI is inf
                self.optimal_eui_predicted = None
            else:
                logging.info(
                    f"Optimization with {optimizer_config_name.upper()} for {self.unique_id} successful. "
                    f"Predicted optimal EUI: {self.optimal_eui_predicted:.4f}"
                )
                log_params_str = "\n".join([f"  {name}: {val}" for name, val in self.optimal_params.items()])
                logging.info(f"Optimal parameter set (discrete):\n{log_params_str}")
        else:
            logging.error(f"Optimization with {optimizer_config_name.upper()} failed for {self.unique_id} or returned no valid solution.")

    def _objective_for_discrete_optimizers(self, params_dict: dict) -> float:
        """Objective function for optimizers that provide discrete parameter dictionaries.

        This function is used by GA and for re-evaluating L-BFGS-B's discrete result.

        Args:
            params_dict (dict): A dictionary {param_name: discrete_value}.

        Returns:
            float: The predicted EUI from the surrogate model.
        """
        # logging.debug(f"Evaluating discrete params for {self.unique_id}: {params_dict}")
        try:
            # Ensure columns are in the same order as during surrogate model training
            # self.ecm_names should store this order
            params_df = pd.DataFrame([params_dict], columns=self.ecm_names)
            predicted_eui_array = self.surrogate_model.predict(params_df)

            # Ensure a single float is returned
            eui_value = float(predicted_eui_array[0]) if hasattr(predicted_eui_array, '__len__') and len(
                predicted_eui_array) > 0 else float(predicted_eui_array)
            # logging.debug(f"Predicted EUI for {self.unique_id}: {eui_value}")
            return eui_value
        except Exception as e:
            logging.warning(
                f"Surrogate model prediction failed in discrete objective for {self.unique_id}. "
                f"Params: {params_dict}. Error: {e}. Returning float('inf').", exc_info=True
            )
            return float('inf')

    def _map_continuous_to_nearest_discrete(self, continuous_params: np.ndarray) -> dict:
        """Maps continuous optimization results to the nearest permissible discrete value.

        Args:
            continuous_params (np.ndarray): Continuous parameter values from an optimizer like L-BFGS-B.

        Returns:
            dict: Dictionary containing optimal parameter values mapped to nearest discrete values.
        """
        discrete_params = {}
        for i, name in enumerate(self.ecm_names):
            continuous_val = continuous_params[i]
            discrete_levels = self.ecm_ranges[name]
            # Find the discrete level closest to the continuous value
            nearest_discrete_val = min(
                discrete_levels, key=lambda x: abs(x - continuous_val))
            discrete_params[name] = nearest_discrete_val
        return discrete_params

    def _optimize_with_lbfgsb(self) -> tuple[dict | None, float | None, object | None]:
        """Performs optimization using the L-BFGS-B algorithm.

        Returns:
            tuple[dict | None, float | None, object | None]:
                - Optimal discrete parameters dictionary, or None if failed.
                - Predicted EUI for the optimal discrete parameters, or None if failed.
                - Raw result object from scipy.optimize.minimize, or None.
        """
        logging.info(f"Starting L-BFGS-B optimization for {self.unique_id}...")

        def objective_function_continuous(params_array: np.ndarray) -> float:
            """Objective function for L-BFGS-B working in continuous space."""
            # L-BFGS-B provides a numpy array. Convert to DataFrame for surrogate model.
            params_df = pd.DataFrame([params_array], columns=self.ecm_names)
            try:
                predicted_eui_array = self.surrogate_model.predict(params_df)
                eui_value = float(predicted_eui_array[0]) if hasattr(
                    predicted_eui_array, '__len__') else float(predicted_eui_array)
                return eui_value
            except Exception as e:
                logging.warning(
                    f"Surrogate model prediction failed in L-BFGS-B objective for {self.unique_id}. "
                    f"Params: {params_array}. Error: {e}. Returning float('inf').", exc_info=True
                )
                return float('inf')

        bounds = []
        initial_guess = []
        for name in self.ecm_names:
            levels = self.ecm_ranges[name]
            bounds.append((min(levels), max(levels)))
            # Sensible initial guess: midpoint or first non-zero if applicable
            non_zero_levels = [l for l in levels if l != 0]
            initial_guess.append(np.mean(levels) if not non_zero_levels or len(
                non_zero_levels) == len(levels) else non_zero_levels[0])

        lbfgsb_options = self.config['analysis'].get(
            'lbfgsb_params', {'maxiter': 100, 'disp': False})  # Get from config

        try:
            result = minimize(
                objective_function_continuous,
                x0=np.array(initial_guess),
                method='L-BFGS-B',
                bounds=bounds,
                options=lbfgsb_options
            )

            if result.success:
                optimal_params_continuous = result.x
                # Map continuous result to nearest discrete values
                optimal_params_discrete = self._map_continuous_to_nearest_discrete(
                    optimal_params_continuous)
                # Re-evaluate the EUI with the discrete parameters using the surrogate model
                optimal_eui_reassessed = self._objective_for_discrete_optimizers(
                    optimal_params_discrete)

                logging.info(
                    f"L-BFGS-B optimization successful for {self.unique_id}. "
                    f"Continuous EUI: {result.fun:.4f}, Reassessed Discrete EUI: {optimal_eui_reassessed:.4f}"
                )
                return optimal_params_discrete, optimal_eui_reassessed, result
            else:
                logging.error(
                    f"L-BFGS-B optimization failed for {self.unique_id}. Message: {result.message}"
                )
                return None, None, result
        except Exception as e:
            logging.error(
                f"Error during L-BFGS-B optimization for {self.unique_id}: {e}", exc_info=True)
            return None, None, None

    def _optimize_with_deap_ga(self) -> tuple[dict | None, float | None, object | None]:
        """Performs optimization using a Genetic Algorithm (GA) with DEAP.

        Returns:
            tuple[dict | None, float | None, object | None]:
                - Optimal discrete parameters dictionary, or None if failed.
                - Predicted EUI for the optimal discrete parameters, or None if failed.
                - DEAP logbook object, or None.
        """
        logging.info(
            f"Starting Genetic Algorithm (DEAP) optimization for {self.unique_id}...")
        ga_config = self.config['analysis'].get('ga_params', {})
        if not ga_config:
            logging.error(
                f"GA parameters not found in config for {self.unique_id}. Aborting GA.")
            return None, None, None

        # --- DEAP Setup ---
        # Fitness: Minimize EUI (weight is -1.0)
        # Avoid re-creating if run multiple times in same session
        if not hasattr(creator, "FitnessMin"):
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMin)

        toolbox = base.Toolbox()

        # Store choices for each parameter and their lengths for attribute generation
        # These are instance variables because eval_individual and mut_custom_random_reset might need them
        # if not defined as closures or passed explicitly.
        self._ga_param_names = list(self.ecm_names)  # Ensure consistent order
        self._ga_param_choices = [list(self.ecm_ranges[name])
                                  for name in self._ga_param_names]
        self._ga_param_indices_len = [len(choices)
                                      for choices in self._ga_param_choices]

        # Attribute generator: creates an index for a parameter's choice list
        def generate_attribute_index(param_idx_in_list: int) -> int:
            return random.randint(0, self._ga_param_indices_len[param_idx_in_list] - 1)

        # Register attribute generators for each parameter
        # An individual will be a list of these indices
        attributes_for_individual = []
        for i in range(len(self._ga_param_names)):
            toolbox.register(f"attr_idx_{i}", generate_attribute_index, i)
            attributes_for_individual.append(getattr(toolbox, f"attr_idx_{i}"))

        # Individual generator: a cycle of these attribute generators
        toolbox.register("individual", tools.initCycle, creator.Individual, tuple(
            attributes_for_individual), n=1)
        # Population generator
        toolbox.register("population", tools.initRepeat,
                         list, toolbox.individual)

        # Evaluation function wrapper
        def eval_individual(individual_indices: list) -> tuple[float]:
            params_dict = {}
            for i, param_name in enumerate(self._ga_param_names):
                # This is the index from the individual
                choice_index = individual_indices[i]
                params_dict[param_name] = self._ga_param_choices[i][choice_index]

            eui = self._objective_for_discrete_optimizers(params_dict)
            return (eui,)  # DEAP fitness must be a tuple

        toolbox.register("evaluate", eval_individual)

        # Genetic operators
        toolbox.register("mate", tools.cxTwoPoint)  # Two-point crossover

        # Custom mutation: for each gene (index), with probability indpb,
        # randomly pick a new valid index for that parameter.
        def mut_custom_random_reset(individual: creator.Individual, indpb: float) -> tuple[creator.Individual,]:
            for i in range(len(individual)):
                if random.random() < indpb:
                    # individual[i] is the current index for parameter i
                    # Re-generate a new valid index for this parameter
                    individual[i] = generate_attribute_index(i)
            return individual,

        toolbox.register("mutate", mut_custom_random_reset,
                         indpb=ga_config.get('indpb', 0.05))
        toolbox.register("select", tools.selTournament,
                         tournsize=ga_config.get('tournament_size', 3))

        # --- Run GA ---
        population_size = ga_config.get('population_size', 50)
        num_generations = ga_config.get('num_generations', 40)
        cxpb = ga_config.get('cxpb', 0.7)
        mutpb = ga_config.get('mutpb', 0.2)

        pop = toolbox.population(n=population_size)
        # Store the best individual
        hof = tools.HallOfFame(ga_config.get('hall_of_fame_size', 1))

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)

        logging.info(
            f"Running GA for {self.unique_id} with pop_size={population_size}, ngen={num_generations}, "
            f"cxpb={cxpb}, mutpb={mutpb}"
        )

        try:
            # algorithms.eaSimple returns the final population and the logbook
            final_pop, logbook = algorithms.eaSimple(pop, toolbox,
                                                     cxpb=cxpb, mutpb=mutpb,
                                                     ngen=num_generations, stats=stats,
                                                     halloffame=hof, verbose=True)  # verbose=True prints stats
        except Exception as e_ga_run:
            logging.error(
                f"Error during GA execution (algorithms.eaSimple) for {self.unique_id}: {e_ga_run}", exc_info=True)
            return None, None, None

        if not hof or not hof[0].fitness.valid:
            logging.error(
                f"GA for {self.unique_id} did not find a valid solution in Hall of Fame.")
            # Fallback: try to get best from final population if HoF is empty or invalid
            if final_pop:
                best_from_pop = tools.selBest(final_pop, k=1)
                if best_from_pop and best_from_pop[0].fitness.valid:
                    logging.info(
                        "Using best individual from final population as HoF was problematic.")
                    best_individual_indices = best_from_pop[0]
                    best_eui = best_from_pop[0].fitness.values[0]
                else:
                    return None, None, logbook
            else:
                return None, None, logbook
        else:
            best_individual_indices = hof[0]  # This is a list of indices
            best_eui = hof[0].fitness.values[0]  # Fitness is a tuple

        # Convert best individual (indices) back to parameter dictionary
        optimal_params_dict = {}
        for i, param_name in enumerate(self._ga_param_names):
            choice_index = best_individual_indices[i]
            optimal_params_dict[param_name] = self._ga_param_choices[i][choice_index]

        logging.info(
            f"GA optimization completed for {self.unique_id}. Best EUI: {best_eui:.4f}")
        return optimal_params_dict, best_eui, logbook

    def validate_optimum(self):
        """
        Execute EnergyPlus simulations with optimized parameters to validate the predicted results.
        """
        if self.optimal_params is None:
            logger.error(
                "Error: Optimization parameters not found; verification cannot proceed. Please run `optimize()` first.")
            return

        logger.info(
            f"--- {self.unique_id}: Validating the optimal parameter set ---")
        self.optimal_eui_simulated, floor_area = self._run_single_simulation_internal(
            params_dict=self.optimal_params,
            run_id=f"optimized",
        )
        if floor_area is not None and floor_area > 0:
            self.building_floor_area = floor_area  # Update the building floor area

        if self.optimal_eui_simulated:
            logger.info(
                f"Optimal Simulated EUI: {self.optimal_eui_simulated:.2f} kWh/m².")

            if self.optimal_eui_predicted:
                self.optimization_bias = abs(
                    self.optimal_eui_simulated - self.optimal_eui_predicted) / self.optimal_eui_simulated * 100
                logger.info(
                    f"Optimization bias: {self.optimization_bias:.2f}%")
            else:
                logger.warning(
                    "Warning: The predicted EUI is not available. The bias cannot be calculated.")

            if not self.baseline_eui:
                logger.info(
                    "Info: The baseline EUI hasn't been calculated. Attempting to run it now...")
                self.run_baseline_simulation()

            if self.baseline_eui and self.baseline_eui > 0:
                self.optimization_improvement = (
                    self.baseline_eui - self.optimal_eui_simulated) / self.baseline_eui * 100
                logger.info(
                    f"Info: EUI Improvement (Relative to Baseline): {self.optimization_improvement:.2f}%")

            elif self.baseline_eui == 0:
                logger.warning(
                    "Warning: Improvement rate cannot be calculated (baseline EUI is zero).")
            else:
                logger.warning(
                    "Warning: The baseline EUI is not available. The improvement rate cannot be calculated.")
        else:
            logger.warning(
                "Warning: The optimal EUI simulation failed. The validation cannot be performed.")

    def run_pv_analysis(self):
        """Find suitable surfaces, add PV, run simulation, and calculate net EUI."""
        if not self.config.get('pv_analysis', {}).get('enabled', False):
            logger.info(f"--- {self.unique_id}: PV analysis is disabled ---")
            return
        if self.optimal_eui_simulated is None and self.optimal_params is None:
            logger.error(
                f"Error: Optimization or validation not completed, cannot perform PV analysis.")
            return
        if self.building_floor_area is None or self.building_floor_area <= 0:
            logger.error(
                f"Error: Missing valid floor area, cannot calculate net EUI.")
            return
        logger.info(f"--- {self.unique_id}: Starting PV analysis ---")
        try:
            optimized_idf_obj_path = self.work_dir / "optimized" / "optimized.idf"
            if not optimized_idf_obj_path.exists():
                logger.error(
                    f"Error: Verified optimized IDF not found: {optimized_idf_obj_path}")
                return
            # Load the verified optimized IDF
            optimized_idf_model = IDFModel(optimized_idf_obj_path)
            pv_manager = PVManager(optimized_idf_model=optimized_idf_model, runner=self.runner, config=self.config,
                                   weather_path=self.weather_path, base_work_dir=self.work_dir)
            self.suitable_surfaces = pv_manager.find_suitable_surfaces()  # Find surfaces

            if self.suitable_surfaces:
                pv_run_id = "optimized_pv"
                self.pv_idf_path = pv_manager.add_pv_to_idf(
                    self.suitable_surfaces, pv_run_id)  # Add PV
                if self.pv_idf_path:
                    pv_output_dir = self.work_dir / pv_run_id
                    pv_output_prefix = self.config['pv_analysis'].get(
                        'pv_output_prefix', 'pv')
                    success, message = self.runner.run_simulation(  # Run the PV simulation
                        idf_path=self.pv_idf_path, weather_path=self.weather_path,
                        output_dir=pv_output_dir, output_prefix=pv_output_prefix, config=self.config)
                    if success:
                        self.pv_generation_results = pv_manager.analyze_pv_generation(
                            pv_output_prefix)  # Analyze the PV generation
                        pv_result_parser = SimulationResult(
                            pv_output_dir, pv_output_prefix)  # Get the total EUI
                        self.gross_eui_with_pv = pv_result_parser.get_source_eui(
                            self.config['constants']['ng_conversion_factor'])
                        if self.gross_eui_with_pv is not None and self.pv_generation_results is not None:
                            total_pv_kwh = self.pv_generation_results.get(
                                'total_annual_kwh', 0.0)
                            # Calculate the PV generation intensity
                            pv_kwh_per_m2 = total_pv_kwh / self.building_floor_area
                            self.net_eui_with_pv = self.gross_eui_with_pv - \
                                pv_kwh_per_m2  # Calculate the net EUI
                            self.optimization_improvement_with_pv = (
                                self.baseline_eui - self.net_eui_with_pv) / self.baseline_eui * 100
                            logger.info(
                                f"Gross EUI with PV: {self.gross_eui_with_pv:.2f} kWh/m2")
                            logger.info(
                                f"Annual PV generation: {total_pv_kwh:.2f} kWh ({pv_kwh_per_m2:.2f} kWh/m2)")
                            logger.info(
                                f"Net source EUI (Net): {self.net_eui_with_pv:.2f} kWh/m2")
                        else:
                            logger.warning(
                                "Warning: Unable to obtain the total EUI with PV or PV generation data, unable to calculate the net EUI.")
                    else:
                        logger.error(
                            f"Error: Final PV simulation failed: {message}")
                else:
                    logger.error("Error: Failed to add PV to IDF.")
            else:
                logger.info(
                    "Info: No suitable surfaces found for PV installation. Skipping PV simulation.")
                self.net_eui_with_pv = self.optimal_eui_simulated
        except Exception as e:
            logger.error(
                f"Error: An issue arose while executing the PV analysis: {e}")
            import traceback
            traceback.print_exc()

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
            'optimization_improvement_with_pv': self.optimization_improvement_with_pv,
            "sensitivity_indices": self.sensitivity_results,
        }
        result_file = self.work_dir / "pipeline_results.json"
        try:
            def numpy_converter(obj):  # Process NumPy types
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(round(obj, 6)) if not np.isnan(obj) else "NaN"
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.bool_, bool)):
                    return bool(obj)
                elif pd.isna(obj):
                    return "NaN"
                # Process NumPy arrays in SALib output dictionaries
                if isinstance(obj, dict):
                    return {k: numpy_converter(v) for k, v in obj.items()}
                return '<not serializable>'
            with open(result_file, "w") as f:
                json.dump(results_data, f, indent=4, default=numpy_converter)
            logger.info(f"Optimization results saved to {result_file}")
        except Exception as e:
            logger.error(
                f"Error: Failed to save results to {result_file}: {e}")

    def run_full_pipeline(self, run_sens: bool = True, build_model: bool = True, run_opt=True, validate=True, run_pv=True, save=True):
        """
        Execute each stage of the entire optimization process sequentially.

        Args:
            run_sens (bool, optional): Run the sensitivity analysis. Defaults to True.
            build_model (bool, optional): Build the surrogate model. Defaults to True.
            run_opt (bool, optional): Run the optimization. Defaults to True.
            validate (bool, optional): Validate the results. Defaults to True.
            save (bool, optional): Save the results. Defaults to True.
        """
        logger.info(f"======== Start processing: {self.unique_id} ========")
        if self.run_baseline_simulation() is None and (run_sens or validate):
            logger.error(
                "Error: The baseline EUI hasn't been calculated. The pipeline cannot proceed.")
            return

        if run_sens:
            self.run_sensitivity_analysis()
            if self.sensitivity_results is None:
                logger.error(
                    "Error: Sensitivity analysis failed, subsequent steps may be impacted.")

        if build_model:
            self.build_surrogate_model()
            if self.surrogate_model is None:
                logger.error(
                    "Error: The attempt to build a surrogate model has failed, precluding optimization and validation.")

        if run_opt:
            self.optimize()
            if self.optimization_results is None:
                logger.error(
                    "Error: Optimization failed, preventing validation from proceeding.")
                return

        if validate:
            self.validate_optimum()
            if self.optimal_eui_simulated is None and run_pv:
                logger.error(
                    "Error: Optimization validation failed, aborting PV analysis.")
                return

        # Check if PV analysis is enabled in the configuration
        if run_pv and self.config.get('pv_analysis', {}).get('enabled', False):
            self.run_pv_analysis()

        if save:
            self.save_results()

        logger.info(f"======== End processing: {self.unique_id} ========")

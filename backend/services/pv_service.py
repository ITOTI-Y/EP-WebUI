# backend/services/pv_service.py

import pandas as pd
import logging
from copy import copy
from pathlib import Path
from .idf_service import IDFModel
from .simulation_service import EnergyPlusRunner


class PVManager:
    """
    Manage the analysis, addition and simulation of PV systems.
    """

    def __init__(self, optimized_idf_model: IDFModel, runner: EnergyPlusRunner, config: dict, weather_path: str, base_work_dir: Path):
        """
        Initialize the PVManager with the necessary components.

        Args:
            optimized_idf_model (IDFModel): The optimized IDF model.
            runner (EnergyPlusRunner): The EnergyPlus runner.
            config (dict): The configuration dictionary.
            weather_path (str): The path to the weather file.
            base_work_dir (str): The base working directory.
        """
        self.optimized_idf_model = optimized_idf_model
        self.runner = runner
        self.config = config
        self.weather_path = weather_path
        self.work_dir = base_work_dir
        self.pv_config = config.get('pv_analysis', {})
        if not self.pv_config:
            print("Warning: 'pv_analysis' section missing in config.")

    def _add_shadow_outputs_to_idf(self, idf_model: IDFModel):
        """Add output variables required for shadow analysis to the IDF object (W/m2)."""
        output_variables = [
            "Surface Outside Face Incident Solar Radiation Rate per Area",  # Unit: W/m2
            "Surface Outside Face Sunlit Fraction",  # Unitless
        ]
        idf = idf_model.idf
        for var_name in output_variables:
            exists = False
            for ov in idf.idfobjects.get("OUTPUT:VARIABLE", []):
                if ov.Key_Value == "*" and ov.Variable_Name.lower() == var_name.lower():
                    exists = True
                    break
            if not exists:
                idf.newidfobject("OUTPUT:VARIABLE", Key_Value="*",
                                 Variable_Name=var_name, Reporting_Frequency="Hourly")

    def _calculate_radiation_score(self, annual_radiation_kwh_per_m2: float) -> float:
        """
        Calculate radiation score based on annual radiation (kWh/m2).

        Args:
            annual_radiation_kwh_per_m2 (float): Annual radiation (kWh/m2).

        Returns:
            float: Radiation score (0-100).
        """
        # Get thresholds from config (already in kWh/m2)
        high_threshold = self.pv_config.get("radiation_threshold_high", 1000.0)
        low_threshold = self.pv_config.get("radiation_threshold_low", 600.0)
        max_score = self.pv_config.get("max_score", 100.0)
        min_score = self.pv_config.get("min_score", 0.0)

        if annual_radiation_kwh_per_m2 > high_threshold:
            return max_score
        elif annual_radiation_kwh_per_m2 >= low_threshold:
            # Ensure denominator is not zero
            if high_threshold > low_threshold:
                return min_score + (max_score - min_score) * \
                    (annual_radiation_kwh_per_m2 - low_threshold) / \
                    (high_threshold - low_threshold)
            else:  # If thresholds are the same, score is max if above threshold
                return max_score if annual_radiation_kwh_per_m2 >= low_threshold else min_score
        else:
            return min_score

    def find_suitable_surfaces(self) -> list[dict] | None:
        """Execute shadow analysis simulation and find suitable surfaces for PV installation."""
        print("--- Start shadow analysis to find suitable PV surfaces ---")
        shadow_run_id = "optimized_shadow"
        shadow_idf_path = self.work_dir / \
            shadow_run_id / f"{shadow_run_id}.idf"
        shadow_output_dir = self.work_dir / shadow_run_id
        shadow_output_prefix = self.pv_config.get(
            'shadow_output_prefix', 'shadow')
        shadow_output_dir.mkdir(parents=True, exist_ok=True)

        try:
            temp_idf = copy(self.optimized_idf_model.idf)
            shadow_idf = IDFModel(shadow_idf_path, eppy_idf_object=temp_idf)
            self._add_shadow_outputs_to_idf(shadow_idf)
            shadow_idf.save()

            success, message = self.runner.run_simulation(
                idf_path=shadow_idf_path, weather_path=self.weather_path,
                output_dir=shadow_output_dir, output_prefix=shadow_output_prefix,
                config=self.config)
            if not success:
                logging.error(
                    f"Error: Shadow analysis simulation failed: {message}")
                return None

            # --- Parse results ---
            csv_path = shadow_output_dir / f"{shadow_output_prefix}.csv"
            if not csv_path.exists():
                logging.error(
                    f"Error: Shadow analysis output file not found: {csv_path}")
                return None

            df = pd.read_csv(csv_path)
            target_surface_types = [s.upper() for s in self.pv_config.get(
                'shadow_calculation_surface_types', ['ROOF'])]
            suitable_surfaces = []
            all_surfaces = self.optimized_idf_model.idf.idfobjects.get(
                "BUILDINGSURFACE:DETAILED", [])

            for surface in all_surfaces:
                if hasattr(surface, 'Outside_Boundary_Condition') and \
                        surface.Outside_Boundary_Condition.upper() == "OUTDOORS" and \
                        hasattr(surface, 'Surface_Type') and \
                        surface.Surface_Type.upper() in target_surface_types:
                    surface_name = surface.Name
                    radiation_col = None
                    # --- Find radiation column ---
                    rad_var_name = "Surface Outside Face Incident Solar Radiation Rate per Area".upper()
                    for col in df.columns:
                        # Improve matching logic to ensure full surface name matches (avoid partial matches) and variable name is correct
                        col_upper = col.upper()
                        # Expected format: SURFACE_NAME:VAR_NAME [W/m2](Hourly)
                        parts = col_upper.split(':')
                        if len(parts) > 1 and surface_name.upper() in col_upper and rad_var_name in col_upper:
                            radiation_col = col
                            break

                    if radiation_col:
                        try:
                            # --- Calculate annual radiation (kWh/m2) ---
                            # E+ output frequency is Hourly, value is the average rate for that hour (W/m2)
                            # Annual total energy (Wh/m2) = Σ( hourly_average_rate_W_m2 * 1_hour )
                            # W/m2 * h = Wh/m2
                            annual_wh_per_m2 = df[radiation_col].sum()
                            annual_kwh_per_m2 = annual_wh_per_m2 / 1000.0  # kWh/m2
                            # --- Calculate score ---
                            radiation_score = self._calculate_radiation_score(
                                annual_kwh_per_m2)  # Use kWh/m2
                            area = self.optimized_idf_model.get_surface_area(
                                surface_name)  # Get the area of the surface

                            # 确保面积有效
                            if radiation_score >= self.pv_config.get('radiation_score_threshold', 70) and area > 0:
                                suitable_surfaces.append({
                                    "name": surface_name,
                                    "area": round(area, 2),
                                    "radiation_score": round(radiation_score, 1),
                                    # Store kWh/m2
                                    "annual_radiation_kwh": round(annual_kwh_per_m2, 1)
                                })
                        except Exception as calc_e:
                            logging.warning(
                                f"Warning: Error calculating surface '{surface_name}' radiation or score: {calc_e}")
                    else:
                        logging.warning(
                            f"Warning: No radiation data column found for surface '{surface_name}'.")

            if not suitable_surfaces:
                logging.warning(
                    "Warning: No suitable surfaces found for PV installation.")
                return []
            suitable_surfaces.sort(
                key=lambda x: x["radiation_score"], reverse=True)
            logging.info(
                f"Found {len(suitable_surfaces)} suitable surfaces for PV installation.")
            return suitable_surfaces
        except Exception as e:
            logging.error(f"Error: Error during shadow analysis: {e}")
            import traceback
            traceback.print_exc()
            return None

    # def add_pv_to_idf(self, suitable_surfaces: list[dict], pv_run_id: str) -> str|None:
    #     """
    #     Adds a PVWatts system to the optimized IDF model object, 
    #     clearing any existing PV-related objects beforehand.

    #     Args:
    #         suitable_surfaces (list[dict]): List of suitable surfaces for PV installation.
    #         pv_run_id (str): The ID of the PV run.

    #     Returns:
    #         str|None: The path to the modified IDF file.
    #     """
    #     if not suitable_surfaces:
    #         logging.info("Info: No suitable surfaces provided, not adding PV systems.")
    #         return self.optimized_idf_model.idf_path
        
    #     if not self.config.get("use_pvwatts", False):
    #         logging.error("Error: PVWatts configuration is disabled ('use_pvwatts' is not True), and therefore a PVWatts system cannot be added.")
    #         return None
        
    #     logging.info(f"--- Purging legacy PV objects and adding {len(suitable_surfaces)} PVWatts systems to the IDF ---")
    #     pv_output_dir = self.work_dir / pv_run_id
    #     pv_output_dir.mkdir(parents=True, exist_ok=True)
    #     pv_idf_path = pv_output_dir / f"{pv_run_id}.idf"

    #     pv_efficiency = self.pv_config.get('pv_efficiency', 0.18)
    #     pv_coverage = self.pv_config.get('pv_coverage', 0.8)
    #     module_type = self.pv_config.get('pvwatts_module_type', 'Standard')
    #     array_type = self.pv_config.get('pvwatts_array_type', 'FixedOpenRack')
    #     system_losses = self.pv_config.get('pvwatts_system_losses', 0.14)
    #     dc_ac_ratio = self.pv_config.get('pvwatts_dc_ac_ratio', 1.1)
    #     inverter_efficiency = self.pv_config.get('pvwatts_inverter_efficiency', 0.96)
    #     gcr = self.pv_config.get('pvwatts_ground_coverage_ratio', 0.4)

    #     # Define new, dedicated PV component names
    #     pv_elcd_name = "PV_System_Load_Center"
    #     pv_elcg_name = "PV_System_Generator_List"
    #     pv_inverter_name = "PV_System_PVWatts_Inverter" 

    #     try:
    #         temp_idf_object = copy(self.optimized_idf_model.idf)
    #         idf_model = IDFModel(pv_idf_path, eppy_idf_object=temp_idf_object)
    #         idf = idf_model.idf

    #         # --- 1. Clear all existing PV-related objects ---
    #         # Includes Photovoltaic, PVWatts, and related performance, distribution centers, lists, and inverter objects.
    #         logging.info("--- Purging legacy PV related objects ---")
    #         specific_names_to_remove = [pv_elcd_name,
    #                                     pv_elcg_name,
    #                                     pv_inverter_name,
    #                                     "PV_System_Simple_Inverter"]
    #         for name in specific_names_to_remove:
    #             for obj_type in ["ElectricLoadCenter:Distribution",
    #                              "ElectricLoadCenter:Generators",
    #                              "ElectricLoadCenter:Inverter:Simple",
    #                              "ElectricLoadCenter:Inverter:PVWatts"]:
    #                 try:
    #                     obj = idf.getobject(obj_type.upper(), name)
    #                     if obj:
    #                         idf.removeidfobject(obj)
    #                         logging.info(f"The specified {obj_type} '{name}' has been successfully removed.")
    #                 except Exception as e:
    #                     logging.info(f"Encountered an issue while attempting to remove the specified {obj_type} '{name}' (it may not exist): {e}")

    #         # Remove PV Generators and Performance Objects by Type
    #         pv_types_to_remove = [
    #             "Generator:Photovoltaic",
    #             "PhotovoltaicPerformance:Simple",
    #             "PhotovoltaicPerformance:EquivalentOne-Diode",
    #             "PhotovoltaicPerformance:Sandia",
    #             "Generator:PVWatts"
    #         ]
    #         idf_model._remove_objects_by_type(pv_types_to_remove)
    #         logging.info("Completes the clearing of objects related to existing PVs.")

    #         # --- 2. Create PVWatts Inverter (ElectricLoadCenter:Inverter:PVWatts) ---
    #         logging.info(f"Creating PVWatts Inverter: {pv_inverter_name}")
    #         idf.newidfobject(
    #             "ElectricLoadCenter:Inverter:PVWatts",
    #             Name=pv_inverter_name,
    #             DC_to_AC_Size_Ratio=dc_ac_ratio,
    #             Inverter_Efficiency=inverter_efficiency
    #         )

    #         # --- 3. Creating a list of generators (ElectricLoadCenter:Generators) ---
    #         logging.info(f"Creating generator list: {pv_elcg_name}")
    #         pv_elcg = idf.newidfobject(
    #             "ElectricLoadCenter:Generators",
    #             Name=pv_elcg_name
    #             # Fill the generator later
    #         )

    #         # --- 4. Creation of distribution centers (ElectricLoadCenter:Distribution) ---
    #         logging.info(f"Create distribution center: {pv_elcd_name}")
    #         idf.newidfobject(
    #             "ElectricLoadCenter:Distribution",
    #             Name=pv_elcd_name,
    #             Generator_List_Name=pv_elcg_name,
    #             Generator_Operation_Scheme_Type="Baseload",
    #         )


    def add_pv_to_idf(self, suitable_surfaces: list[dict], pv_run_id: str) -> str | None:
        """
        Adds a PV system to the base IDF model objects.
        All existing PV-related objects are cleared before adding.

        Args:
            suitable_surfaces (list[dict]): List of suitable surfaces for PV installation.
            pv_run_id (str): The ID of the PV run.

        Returns:
            str | None: The path to the modified IDF file.
        """
        if not suitable_surfaces:
            logging.info("Info: No suitable surfaces provided, not adding PV systems.")
            return self.optimized_idf_model.idf_path
        logging.info(f"--- Adding {len(suitable_surfaces)} PV systems to IDF ---")

        # PV Output Path Setup
        pv_output_dir = self.work_dir / pv_run_id
        pv_output_dir.mkdir(parents=True, exist_ok=True)
        pv_idf_path = pv_output_dir / f"{pv_run_id}.idf"

        # Copy base IDF model
        try:
            idf_object_to_modify = copy(self.optimized_idf_model.idf)
        except Exception as e:
            logging.error(f"Error: Error copying base IDF model: {e}")
            return None

        idf_model = IDFModel(str(pv_idf_path), eppy_idf_object=idf_object_to_modify)
        idf = idf_model.idf

        logging.info(f"Clearing existing PV-related objects...")
        objects_to_remove = [
            "Generator:Photovoltaic",
            "PhotovoltaicPerformance:Simple",
            "PhotovoltaicPerformance:EquivalentOne-Diode",
            "PhotovoltaicPerformance:Sandia",
            "Generator:PVWatts",
            "ElectricLoadCenter:Inverter:Simple",
            "ElectricLoadCenter:Inverter:PVWatts",
        ]
        idf_model.remove_objects_by_type(objects_to_remove)
        logging.info("Successfully cleared existing PV-related objects.")

        pv_config = self.pv_config
        pv_efficiency = pv_config.get('pv_efficiency', 0.18)
        pv_coverage = pv_config.get('pv_coverage', 0.8)
        pv_inverter_efficiency = pv_config.get('pv_inverter_efficiency', 0.96)


        try:
            # --- Defining new, dedicated PV component names ---
            # Use clearer and less conflict-prone names
            pv_elcd_name = "PV_System_Load_Center"      # New distribution center name
            pv_elcg_name = "PV_System_Generator_List"  # New generator list name
            pv_inverter_name = "PV_System_Simple_Inverter"  # New inverter name

            # --- 1. Create *new* ElectricLoadCenter:Distribution ---
            logging.info(f"Creating new ElectricLoadCenter:Distribution center: {pv_elcd_name}")
            pv_elcd = idf.newidfobject(
                "ElectricLoadCenter:Distribution",
                Name=pv_elcd_name,
                Generator_Operation_Scheme_Type="Baseload", 
                Generator_List_Name=pv_elcg_name,  # Associate with new generator list
                Electrical_Buss_Type="DirectCurrentWithInverter", # Specify inverter required
                Inverter_Name=pv_inverter_name  # Associate with new inverter
            )

            # --- 2. create *new* ElectricLoadCenter:Generators ---
            logging.info(f"Creating new ElectricLoadCenter:Generators: {pv_elcg_name}")
            pv_elcg = idf.newidfobject(
                "ElectricLoadCenter:Generators",
                Name=pv_elcg_name 
                # Initialize empty, add generators later
            )

            # --- 3. Create compatible inverter ---
            logging.info(f"Creating new ElectricLoadCenter:Inverter:Simple: {pv_inverter_name}")
            pv_inverter = idf.newidfobject(
                "ElectricLoadCenter:Inverter:Simple",
                Name=pv_inverter_name,
                Availability_Schedule_Name="", # Assume always available (if need Schedule, need to define or reference in advance)
                Inverter_Efficiency=pv_inverter_efficiency, # Typical inverter efficiency
                # Other parameters can use default values
            )

            # --- 4. Add PV systems and output variables for each suitable surface ---
            added_pv_details = []
            logging.info(f"Adding Generator:Photovoltaic objects for {len(suitable_surfaces)} surfaces...")
            for i, surface_info in enumerate(suitable_surfaces):
                surface_name = surface_info["name"]
                pv_name = f"PV_{surface_name.replace(' ', '_').replace(':','_')}"
                perf_name = f"{pv_name}_Performance"

                if idf.getobject("Generator:Photovoltaic".upper(), pv_name):
                    logging.warning(f"Warning: PV generator '{pv_name}' already exists, skipping.")
                    continue

                idf.newidfobject("Generator:Photovoltaic", Name=pv_name, Surface_Name=surface_name,
                                    Photovoltaic_Performance_Object_Type="PhotovoltaicPerformance:Simple",
                                    Module_Performance_Name=perf_name, Heat_Transfer_Integration_Mode="Decoupled")
                if not idf.getobject("PhotovoltaicPerformance:Simple".upper(), perf_name):
                    idf.newidfobject("PhotovoltaicPerformance:Simple", Name=perf_name,
                                        Fraction_of_Surface_Area_with_Active_Solar_Cells=pv_coverage,
                                        Conversion_Efficiency_Input_Mode="Fixed",
                                        Value_for_Cell_Efficiency_if_Fixed=pv_efficiency)

                # --- Add PV to *new* generator list ---
                generator_index = i + 1  # Start from 1 in the cleared list
                gen_name_field = f"Generator_{generator_index}_Name"
                gen_type_field = f"Generator_{generator_index}_Object_Type"
                # Check if the field exists, if eppy does not automatically expand, you may need to manually add field definitions
                # Usually eppy will handle it, but as a robustness check
                if not hasattr(pv_elcg, gen_name_field):
                    pv_elcg.fieldnames.append(gen_name_field)
                    pv_elcg.fieldvalues.append("")  # Add empty value placeholder
                if not hasattr(pv_elcg, gen_type_field):
                    pv_elcg.fieldnames.append(gen_type_field)
                    pv_elcg.fieldvalues.append("")

                setattr(pv_elcg, gen_name_field, pv_name)
                setattr(pv_elcg, gen_type_field, "Generator:Photovoltaic")

                # --- Add output variables (logic unchanged) ---
                pv_rate_var = "Generator Produced DC Electricity Rate"  # W
                pv_energy_var = "Generator Produced DC Electric Energy"  # J
                if not any(ov.Key_Value.upper() == pv_name.upper() and ov.Variable_Name.lower() == pv_rate_var.lower()
                            for ov in idf.idfobjects.get("OUTPUT:VARIABLE", [])):
                    idf.newidfobject("Output:Variable", Key_Value=pv_name,
                                        Variable_Name=pv_rate_var, Reporting_Frequency="Hourly")
                if not any(ov.Key_Value.upper() == pv_name.upper() and ov.Variable_Name.lower() == pv_energy_var.lower()
                            for ov in idf.idfobjects.get("OUTPUT:VARIABLE", [])):
                    idf.newidfobject("Output:Variable", Key_Value=pv_name,
                                        Variable_Name=pv_energy_var, Reporting_Frequency="Hourly")

                added_pv_details.append({  # Record information
                    "pv_name": pv_name, "surface_name": surface_name, "surface_area": surface_info["area"],
                    "pv_coverage": pv_coverage, "pv_efficiency": pv_efficiency,
                    "radiation_score": surface_info.get("radiation_score")
                })

            # --- 5. (Optional) Remove or modify old "GENERATOR" load center ---
            old_generator_elcd_list = [obj for obj in idf.idfobjects.get("ElectricLoadCenter:Distribution".upper()) 
                                        if obj.Name != pv_elcd_name] # Exclude the one we just created
            if old_generator_elcd_list:
                logging.warning(f"Warning: Detected other ElectricLoadCenter:Distribution objects: {[obj.Name for obj in old_generator_elcd_list]}. These objects were not automatically removed, please check if manual processing is needed.")

            # --- 6. Save modified IDF ---
            self.base_idf_model.save(pv_idf_path)
            logging.info(f"PV systems, inverters and dedicated load center added/updated, new IDF file saved to: {pv_idf_path}")
            return pv_idf_path

        except Exception as e:
            logging.error(f"Error: Error adding PV systems to IDF: {e}")
            import traceback
            traceback.print_exc()
            return None

    def analyze_pv_generation(self, pv_output_prefix: str) -> dict | None:
        """Analyze the simulation results with PV, calculate PV generation (kWh, kW)."""
        logging.info("--- Analyze PV generation results ---")
        pv_sim_dir = self.work_dir / "optimized_pv"
        csv_path = pv_sim_dir / f"{pv_output_prefix}.csv"
        if not csv_path.exists():
            logging.error(
                f"Error: PV simulation result file not found: {csv_path}")
            return None

        try:
            df = pd.read_csv(csv_path)
            pv_results = {"systems": {}, "total_annual_kwh": 0.0}
            total_energy_kwh = 0.0
            # Find all PV generation rate columns (unit: W)
            pv_rate_var_name = "Generator Produced DC Electricity Rate".upper()
            pv_rate_cols = [
                col for col in df.columns if pv_rate_var_name in col.upper()]

            if not pv_rate_cols:
                logging.warning(
                    "Warning: No PV generation rate columns found.")
                return pv_results

            for col_name in pv_rate_cols:
                try:
                    pv_name = col_name.split(':')[0].strip()
                    # --- Calculate annual generation kWh ---
                    # Annual total Wh = Σ( hourly_rate_W )
                    annual_wh = df[col_name].sum()
                    annual_kwh = annual_wh / 1000.0
                    # --- Calculate peak power kW ---
                    max_w = df[col_name].max()
                    max_kw = max_w / 1000.0

                    pv_results["systems"][pv_name] = {
                        "annual_energy_kwh": round(annual_kwh, 2),
                        "peak_power_kw": round(max_kw, 2)
                    }
                    total_energy_kwh += annual_kwh
                except Exception as inner_e:
                    logging.warning(
                        f"Warning: Error processing PV column '{col_name}': {inner_e}")

            pv_results["total_annual_kwh"] = round(total_energy_kwh, 2)
            logging.info(
                f"PV generation analysis completed. Total annual generation: {pv_results['total_annual_kwh']:.2f} kWh.")

            # --- Calculate monthly generation kWh ---
            if 'Date/Time' in df.columns:
                try:
                    df['Timestamp'] = pd.to_datetime(
                        df['Date/Time'], format=' %m/%d  %H:%M:%S', errors='coerce')
                    df.dropna(subset=['Timestamp'], inplace=True)
                    df['Month'] = df['Timestamp'].dt.month

                    for pv_name, results_dict in pv_results["systems"].items():
                        results_dict["monthly_energy_kwh"] = {}
                        col = next(
                            (c for c in pv_rate_cols if pv_name.upper() in c.upper()), None)  # 查找对应列
                        if col:
                            monthly_wh = df.groupby(
                                'Month')[col].sum()  # 月总 Wh
                            for month in range(1, 13):
                                results_dict["monthly_energy_kwh"][month] = round(
                                    monthly_wh.get(month, 0) / 1000.0, 2)  # Wh -> kWh
                except Exception as month_e:
                    logging.warning(
                        f"Warning: Error calculating monthly PV generation: {month_e}")

            return pv_results
        except Exception as e:
            logging.error(f"Error: Error analyzing PV results: {e}")
            import traceback
            traceback.print_exc()
            return None

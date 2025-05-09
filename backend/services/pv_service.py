# backend/services/pv_service.py

import pandas as pd
import logging
import math
from copy import copy
from pathlib import Path
from eppy.modeleditor import IDF
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

    def _clear_existing_pv_objects(self, idf_model: IDFModel):
        """Clear existing PV-related objects from the IDF model."""
        logging.info("Clearing existing PV-related objects...")
        objects_to_remove = [
            "Generator:Photovoltaic",
            "PhotovoltaicPerformance:Simple",
            "PhotovoltaicPerformance:EquivalentOne-Diode",
            "PhotovoltaicPerformance:Sandia",
            "Generator:PVWatts",
            "ElectricLoadCenter:Inverter:Simple",
            "ElectricLoadCenter:Inverter:PVWatts",
            "ElectricLoadCenter:Generators",
            "ElectricLoadCenter:Distribution",
        ]
        idf_model.remove_objects_by_type(objects_to_remove)
        logging.info("Successfully cleared existing PV-related objects.")

    def _create_common_electrical_components(self, idf: IDF, inverter_name: str, inverter_type: str, elcd_name: str, elcg_name: str):
        """Create common electrical components (inverter, distribution center, generator list)."""
        logging.info(
            f"Creating common electrical components: inverter: {inverter_name}, inverter type: {inverter_type}")
        if inverter_name == "PV_System_Generic_Inverter":
            idf.newidfobject(
                inverter_type,
                Name=inverter_name,
                Availability_Schedule_Name="",
                Inverter_Efficiency=self.pv_config.get(
                    'pv_inverter_efficiency', 0.96)
            )
        elif inverter_name == "PV_System_PVWatts_Inverter":
            idf.newidfobject(
                inverter_type,
                Name=inverter_name,
                DC_to_AC_Size_Ratio=self.pv_config.get(
                    'pvwatts_dc_ac_ratio', 1.1),
                Inverter_Efficiency=self.pv_config.get(
                    'pvwatts_inverter_efficiency', 0.96)
            )
        else:
            logging.error(f"Error: Unsupported inverter type: {inverter_type}")
            return None
        logging.info(f"Creating generator list: {elcg_name}")
        pv_elcg = idf.newidfobject(
            "ElectricLoadCenter:Generators", Name=elcg_name)
        logging.info(f"Creating distribution center: {elcd_name}")
        idf.newidfobject(
            "ElectricLoadCenter:Distribution",
            Name=elcd_name,
            Generator_List_Name=elcg_name,
            Generator_Operation_Scheme_Type="Baseload",
            Electrical_Buss_Type="DirectCurrentWithInverter",
            Inverter_Name=inverter_name
        )
        return pv_elcg

    def _request_pv_output_variables(self, idf: IDF, pv_generator_name: str):
        """
        Request output variables for PV generation.

        Args:
            idf (IDF): The IDF object.
            pv_generator_name (str): The name of the PV generator.

        Returns:
            bool: True if output variables are requested successfully, False otherwise.
        """
        pv_rate_var = "Generator Produced DC Electricity Rate"
        pv_energy_var = "Generator Produced DC Electric Energy"
        if not any(ov.Key_Value.upper() == pv_generator_name.upper() and ov.Variable_Name.upper() == pv_rate_var.upper()
                   for ov in idf.idfobjects.get("OUTPUT:VARIABLE", [])):
            idf.newidfobject("Output:Variable", Key_Value=pv_generator_name,
                             Variable_Name=pv_rate_var, Reporting_Frequency="Hourly")
        if not any(ov.Key_Value.upper() == pv_generator_name.upper() and ov.Variable_Name.lower() == pv_energy_var.lower()
                   for ov in idf.idfobjects.get("OUTPUT:VARIABLE", [])):
            idf.newidfobject("Output:Variable", Key_Value=pv_generator_name,
                             Variable_Name=pv_energy_var, Reporting_Frequency="Hourly")

    def _apply_pv_simple(self, idf: IDF, suitable_surfaces: list[dict], elcg_obj) -> list[dict]:
        """
        Apply a simple PV system to the IDF model.

        Args:
            idf (IDF): The IDF object.
            suitable_surfaces (list[dict]): List of suitable surfaces for PV installation.
        """
        added_pv_details = []
        logging.info(
            f"Applying PhotovoltaicPerformance:Simple to {len(suitable_surfaces)} surfaces...")
        for i, surface_info in enumerate(suitable_surfaces):
            surface_name = surface_info["name"]
            pv_gen_name = f"PVGen_Simple_{surface_name.replace(' ', '_').replace(':','_')}"
            pv_perf_name = f"{pv_gen_name}_Perf"

            idf.newidfobject(
                "Generator:Photovoltaic",
                Name=pv_gen_name,
                Surface_Name=surface_name,
                Photovoltaic_Performance_Object_Type="PhotovoltaicPerformance:Simple",
                Module_Performance_Name=pv_perf_name,
                Heat_Transfer_Integration_Mode=self.pv_config.get(
                    'heat_transfer_integration_mode', "Decoupled")
            )

            idf.newidfobject(
                "PhotovoltaicPerformance:Simple",
                Name=pv_perf_name,
                Fraction_of_Surface_Area_with_Active_Solar_Cells=self.pv_config.get(
                    'simple_pv_coverage', 0.8),
                Conversion_Efficiency_Input_Mode="Fixed",
                Value_for_Cell_Efficiency_if_Fixed=self.pv_config.get(
                    'simple_pv_efficiency', 0.18)
            )
            setattr(elcg_obj, f"Generator_{i+1}_Name", pv_gen_name)
            setattr(
                elcg_obj, f"Generator_{i+1}_Object_Type", "Generator:Photovoltaic")

            self._request_pv_output_variables(idf, pv_gen_name)
            added_pv_details.append(
                {"pv_name": pv_gen_name, "surface": surface_name, "model": "simple"})
        return added_pv_details

    def _apply_pv_sandia(self, idf: IDF, suitable_surfaces: list[dict], elcg_obj) -> list[dict]:
        """
        Apply a Sandia PV system to the IDF model.

        Args:
            idf (IDF): The IDF object.
            suitable_surfaces (list[dict]): List of suitable surfaces for PV installation.
        """
        added_pv_details = []
        logging.info(
            f"Applying PhotovoltaicPerformance:Sandia to {len(suitable_surfaces)} surfaces...")
        sandia_params_config = self.pv_config.get('sandia_module_params', {})
        required_sandia_keys = [
            'active_area', 'num_cells_series', 'num_cells_parallel',
            'short_circuit_current', 'open_circuit_voltage',
            'current_at_mpp', 'voltage_at_mpp', 'aIsc', 'aImp', 'c0', 'c1',
            'BVoc0', 'mBVoc', 'BVmp0', 'mBVmp', 'diode_factor', 'c2', 'c3',
            'a0', 'a1', 'a2', 'a3', 'a4', 'b0', 'b1', 'b2', 'b3', 'b4', 'b5',
            'delta_tc', 'fd', 'a', 'b', 'c4', 'c5', 'Ix0', 'Ixx0', 'c6', 'c7'
        ]
        missing_keys = [
            key for key in required_sandia_keys if key not in sandia_params_config]
        if missing_keys:
            logging.error(
                f"Error: Missing required Sandia parameters: {missing_keys}")
            return []
        
        pv_perf_name = "PV_Sandia_Performance"

        for i, surface_info in enumerate(suitable_surfaces):
            surface_name = surface_info["name"]
            surface_area = surface_info["area"]
            pv_gen_name = f"PVGen_Sandia_{surface_name.replace(' ', '_').replace(':','_')}"

            module_active_area = sandia_params_config.get('active_area', 1.0)
            sandia_coverage = self.pv_config.get('sandia_pv_coverage', 0.9)
            num_modules_in_series = 1
            num_series_strings = 1

            if module_active_area > 0:
                total_modules_on_surface = math.floor(
                    (surface_area * sandia_coverage) / module_active_area)
                if total_modules_on_surface > 0:
                    num_modules_in_series = total_modules_on_surface
                else:
                    logging.warning(
                        f"Warning: Not enough surface area for PV-Sandia Component installation on {surface_name} Skipping...")
            else:
                logging.warning(
                    f"Sandia component effective area is zero or not configured, skipping surface '{surface_name}'")
                continue

            idf.newidfobject(
                "Generator:Photovoltaic",
                Name=pv_gen_name,
                Surface_Name=surface_name,
                Photovoltaic_Performance_Object_Type="PhotovoltaicPerformance:Sandia",
                Module_Performance_Name=pv_perf_name,
                Heat_Transfer_Integration_Mode=self.pv_config.get(
                    'heat_transfer_integration_mode', "Decoupled"),
                Number_of_Series_Strings_in_Parallel=num_series_strings,
                Number_of_Modules_in_Series=num_modules_in_series
            )
            setattr(elcg_obj, f"Generator_{i+1}_Name", pv_gen_name)
            setattr(elcg_obj, f"Generator_{i+1}_Object_Type", "Generator:Photovoltaic")
            added_pv_details.append({"pv_name": pv_gen_name, "surface": surface_name, "model": "sandia"})
            self._request_pv_output_variables(idf, pv_gen_name)
        self._sandia_pv_params_to_idf(idf, sandia_params_config, pv_perf_name)
        return added_pv_details
    
    def _sandia_pv_params_to_idf(self, idf, sandia_params_config, pv_perf_name):
        sandia_perf = idf.newidfobject("PhotovoltaicPerformance:Sandia", Name=pv_perf_name)
        sandia_perf.Active_Area = sandia_params_config['active_area']
        sandia_perf.Number_of_Cells_in_Series = sandia_params_config['num_cells_series']
        sandia_perf.Number_of_Cells_in_Parallel = sandia_params_config['num_cells_parallel']
        sandia_perf.Short_Circuit_Current = sandia_params_config['short_circuit_current']
        sandia_perf.Open_Circuit_Voltage = sandia_params_config['open_circuit_voltage']
        sandia_perf.Current_at_Maximum_Power_Point = sandia_params_config['current_at_mpp']
        sandia_perf.Voltage_at_Maximum_Power_Point = sandia_params_config['voltage_at_mpp']
        sandia_perf.Sandia_Database_Parameter_aIsc = sandia_params_config['aIsc']
        sandia_perf.Sandia_Database_Parameter_aImp = sandia_params_config['aImp']
        sandia_perf.Sandia_Database_Parameter_c0 = sandia_params_config['c0']
        sandia_perf.Sandia_Database_Parameter_c1 = sandia_params_config['c1']
        sandia_perf.Sandia_Database_Parameter_BVoc0 = sandia_params_config['BVoc0']
        sandia_perf.Sandia_Database_Parameter_mBVoc = sandia_params_config['mBVoc']
        sandia_perf.Sandia_Database_Parameter_BVmp0 = sandia_params_config['BVmp0']
        sandia_perf.Sandia_Database_Parameter_mBVmp = sandia_params_config['mBVmp']
        sandia_perf.Diode_Factor = sandia_params_config['diode_factor']
        sandia_perf.Sandia_Database_Parameter_c2 = sandia_params_config['c2']
        sandia_perf.Sandia_Database_Parameter_c3 = sandia_params_config['c3']
        sandia_perf.Sandia_Database_Parameter_a0 = sandia_params_config['a0']
        sandia_perf.Sandia_Database_Parameter_a1 = sandia_params_config['a1']
        sandia_perf.Sandia_Database_Parameter_a2 = sandia_params_config['a2']
        sandia_perf.Sandia_Database_Parameter_a3 = sandia_params_config['a3']
        sandia_perf.Sandia_Database_Parameter_a4 = sandia_params_config['a4']
        sandia_perf.Sandia_Database_Parameter_b0 = sandia_params_config['b0']
        sandia_perf.Sandia_Database_Parameter_b1 = sandia_params_config['b1']
        sandia_perf.Sandia_Database_Parameter_b2 = sandia_params_config['b2']
        sandia_perf.Sandia_Database_Parameter_b3 = sandia_params_config['b3']
        sandia_perf.Sandia_Database_Parameter_b4 = sandia_params_config['b4']
        sandia_perf.Sandia_Database_Parameter_b5 = sandia_params_config['b5']
        sandia_perf.Sandia_Database_Parameter_DeltaTc = sandia_params_config['delta_tc'] # IDD 中是 Sandia_Database_Parameter_Delta(Tc)
        sandia_perf.Sandia_Database_Parameter_fd = sandia_params_config['fd']
        sandia_perf.Sandia_Database_Parameter_a = sandia_params_config['a']
        sandia_perf.Sandia_Database_Parameter_b = sandia_params_config['b']
        sandia_perf.Sandia_Database_Parameter_c4 = sandia_params_config['c4']
        sandia_perf.Sandia_Database_Parameter_c5 = sandia_params_config['c5']
        sandia_perf.Sandia_Database_Parameter_Ix0 = sandia_params_config['Ix0']
        sandia_perf.Sandia_Database_Parameter_Ixx0 = sandia_params_config['Ixx0']
        sandia_perf.Sandia_Database_Parameter_c6 = sandia_params_config['c6']
        sandia_perf.Sandia_Database_Parameter_c7 = sandia_params_config['c7']
    
    def _apply_pv_pvwatts(self, idf, suitable_surfaces: list[dict], elcg_obj) -> list[dict]:
        """
        Apply a PVWatts system to the IDF model.

        Args:
            idf (IDF): The IDF object.
            suitable_surfaces (list[dict]): List of suitable surfaces for PV installation.
        """
        added_pv_details = []
        logging.info(f"应用 Generator:PVWatts 模型...")
        total_suitable_area = sum(s['area'] for s in suitable_surfaces)
        if total_suitable_area <= 0:
            logging.warning("没有合适的表面积用于 PVWatts 系统。"); return []

        dc_capacity_per_sqm = self.pv_config.get('pvwatts_dc_system_capacity_per_sqm', 0)
        if dc_capacity_per_sqm <= 0:
            simple_eff = self.pv_config.get('simple_pv_efficiency', 0.18) # 回退到用 simple 参数估算
            simple_cov = self.pv_config.get('simple_pv_coverage', 0.8)
            dc_capacity_per_sqm = 1000 * simple_eff * simple_cov
            logging.info(f"PVWatts 单位面积容量未配置，使用估算值: {dc_capacity_per_sqm:.2f} W/m2")


        total_dc_capacity = total_suitable_area * dc_capacity_per_sqm
        if total_dc_capacity <=0:
            logging.warning("PVWatts 总直流容量计算为零或负，不添加 PVWatts 系统。"); return []

        pv_gen_name = f"PVWatts_System_Overall_{self.optimized_idf_model.idf.name.split('.')[0]}" # 使用IDF文件名的一部分确保唯一性

        representative_surface_name = suitable_surfaces[0]['name'] if suitable_surfaces else None
        array_geometry_type = "Surface" if representative_surface_name else "TiltAzimuth"

        idf.newidfobject(
            "Generator:PVWatts",
            Name=pv_gen_name,
            PVWatts_Version=5,
            DC_System_Capacity=total_dc_capacity,
            Module_Type=self.pv_config.get('pvwatts_module_type', 'Standard'),
            Array_Type=self.pv_config.get('pvwatts_array_type', 'FixedRoofMounted'),
            System_Losses=self.pv_config.get('pvwatts_system_losses', 0.14),
            Array_Geometry_Type=array_geometry_type,
            Surface_Name=representative_surface_name if representative_surface_name else "",
            Ground_Coverage_Ratio=self.pv_config.get('pvwatts_ground_coverage_ratio', 0.4)
        )
        setattr(elcg_obj, "Generator_1_Name", pv_gen_name) # PVWatts通常只有一个系统
        setattr(elcg_obj, "Generator_1_Object_Type", "Generator:PVWatts")
        self._request_pv_output_variables(idf, pv_gen_name)
        added_pv_details.append({"pv_name": pv_gen_name, "total_area_used": total_suitable_area, "model": "pvwatts", "dc_capacity_w": total_dc_capacity})
        return added_pv_details


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
        if not self.pv_config.get('enabled', False):
            logging.info("PV 分析在配置中被禁用。")
            return self.optimized_idf_model.idf_path

        if not suitable_surfaces:
            logging.info("信息: 未提供适合的表面，不添加 PV 系统。")
            return self.optimized_idf_model.idf_path

        model_type = self.pv_config.get('pv_model_type', 'Simple').lower()
        logging.info(f"--- 向 IDF 添加光伏系统 (模型类型: {model_type}) ---")

        pv_output_dir = self.work_dir / pv_run_id
        pv_output_dir.mkdir(parents=True, exist_ok=True)
        pv_idf_path = pv_output_dir / f"{pv_run_id}.idf"

        try:
            idf_object_to_modify = copy(self.optimized_idf_model.idf)
            idf_model = IDFModel(str(pv_idf_path), eppy_idf_object=idf_object_to_modify)
            idf = idf_model.idf

            self._clear_existing_pv_objects(idf_model)

            pv_elcd_name = "PV_System_ELCD"
            pv_elcg_name = "PV_System_GeneratorList"
            inverter_name_to_use = ""
            inverter_type_to_use = ""
            added_pv_details = []

            if model_type == 'simple' or model_type == 'sandia':
                inverter_name_to_use = "PV_System_Generic_Inverter"
                inverter_type_to_use = "ElectricLoadCenter:Inverter:Simple"
            elif model_type == 'pvwatts':
                inverter_name_to_use = "PV_System_PVWatts_Inverter"
                inverter_type_to_use = "ElectricLoadCenter:Inverter:PVWatts"
            else:
                logging.error(f"未知的 PV 模型类型: {model_type}")
                return self.optimized_idf_model.idf_path

            elcg_obj = self._create_common_electrical_components(idf, inverter_name_to_use, inverter_type_to_use, pv_elcd_name, pv_elcg_name)
            if elcg_obj is None: # 创建电气组件失败
                logging.error("创建通用电气组件失败，无法继续添加 PV。")
                return self.optimized_idf_model.idf_path


            if model_type == 'simple':
                added_pv_details = self._apply_pv_simple(idf, suitable_surfaces, elcg_obj)
            elif model_type == 'sandia':
                added_pv_details = self._apply_pv_sandia(idf, suitable_surfaces, elcg_obj)
            elif model_type == 'pvwatts':
                added_pv_details = self._apply_pv_pvwatts(idf, suitable_surfaces, elcg_obj)

            if not added_pv_details:
                logging.warning("没有成功添加任何 PV 系统。将返回原始优化后的 IDF 路径。")
                return self.optimized_idf_model.idf_path # 如果没有添加任何 PV，可能返回原始IDF更好

            idf_model.save(pv_idf_path)
            logging.info(f"PV 系统已添加/更新 ({len(added_pv_details)} 个系统)，新的 IDF 文件保存到: {pv_idf_path}")
            logging.debug(f"添加的 PV 详情: {added_pv_details}")
            return pv_idf_path

        except Exception as e:
            logging.error(f"错误: 向 IDF 添加 PV 系统时出错: {e}")
            import traceback; traceback.print_exc()
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

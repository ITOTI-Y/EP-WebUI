# backend/services/pv_service.py

import os
import numpy as np
import pandas as pd
import logging
import eppy.runner.run_functions as run_functions
from copy import copy
from pathlib import Path
from eppy.modeleditor import IDF
from fastapi import HTTPException
from dotenv import load_dotenv
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
        self.base_idf_model = optimized_idf_model
        self.runner = runner
        self.config = config
        self.weather_path = weather_path
        self.work_dir = base_work_dir
        self.pv_config = config.get('pv_analysis', {})
        if not self.pv_config: print("Warning: 'pv_analysis' section missing in config.")

    def _add_shadow_outputs_to_idf(self, idf_model: IDFModel):
        """Add output variables required for shadow analysis to the IDF object (W/m2)."""
        output_variables = [
            "Surface Outside Face Incident Solar Radiation Rate per Area", # Unit: W/m2
            "Surface Outside Face Sunlit Fraction", # Unitless
        ]
        idf = idf_model.idf
        for var_name in output_variables:
            exists = False
            for ov in idf.idfobjects.get("OUTPUT:VARIABLE", []):
                if ov.Key_Value == "*" and ov.Variable_Name.lower() == var_name.lower():
                    exists = True; break
            if not exists:
                idf.newidfobject("OUTPUT:VARIABLE", Key_Value="*", Variable_Name=var_name, Reporting_Frequency="Hourly")

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
                        (annual_radiation_kwh_per_m2 - low_threshold) / (high_threshold - low_threshold)
            else: # If thresholds are the same, score is max if above threshold
                return max_score if annual_radiation_kwh_per_m2 >= low_threshold else min_score
        else:
            return min_score
        
    def find_suitable_surfaces(self) -> list[dict] | None:
        """Execute shadow analysis simulation and find suitable surfaces for PV installation."""
        print("--- Start shadow analysis to find suitable PV surfaces ---")
        shadow_run_id = "optimized_shadow"
        shadow_idf_path = self.work_dir / shadow_run_id / f"{shadow_run_id}.idf"
        shadow_output_dir = self.work_dir / shadow_run_id
        shadow_output_prefix = self.pv_config.get('shadow_output_prefix', 'shadow')
        shadow_output_dir.mkdir(parents=True, exist_ok=True)

        try:
            temp_idf = copy(self.base_idf_model.idf)
            shadow_idf = IDFModel(shadow_idf_path, eppy_idf_object=temp_idf)
            self._add_shadow_outputs_to_idf(shadow_idf)
            shadow_idf.save()

            success, message = self.runner.run_simulation(
                idf_path=shadow_idf_path, weather_path=self.weather_path,
                output_dir=shadow_output_dir, output_prefix=shadow_output_prefix,
                config=self.config)
            if not success:
                logging.error(f"Error: Shadow analysis simulation failed: {message}")
                return None

            # --- Parse results ---
            csv_path = shadow_output_dir / f"{shadow_output_prefix}.csv"
            if not csv_path.exists():
                logging.error(f"Error: Shadow analysis output file not found: {csv_path}")
                return None

            df = pd.read_csv(csv_path)
            target_surface_types = [s.upper() for s in self.pv_config.get('shadow_calculation_surface_types', ['ROOF'])]
            suitable_surfaces = []
            all_surfaces = self.base_idf_model.idf.idfobjects.get("BUILDINGSURFACE:DETAILED", [])

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
                        if len(parts) > 1 and (parts[0]+":"+parts[1]) == surface_name.upper() and rad_var_name in col_upper:
                                radiation_col = col
                                break

                    if radiation_col:
                        try:
                            # --- Calculate annual radiation (kWh/m2) ---
                            # E+ output frequency is Hourly, value is the average rate for that hour (W/m2)
                            # Annual total energy (Wh/m2) = Σ( hourly_average_rate_W_m2 * 1_hour )
                            annual_wh_per_m2 = df[radiation_col].sum() # W/m2 * h = Wh/m2
                            annual_kwh_per_m2 = annual_wh_per_m2 / 1000.0 # kWh/m2
                            # --- Calculate score ---
                            radiation_score = self._calculate_radiation_score(annual_kwh_per_m2) # Use kWh/m2
                            area = self.base_idf_model.get_surface_area(surface_name) # Get the area of the surface

                            if radiation_score >= self.pv_config.get('radiation_score_threshold', 70) and area > 0: # 确保面积有效
                                suitable_surfaces.append({
                                    "name": surface_name,
                                    "area": round(area, 2),
                                    "radiation_score": round(radiation_score, 1),
                                    "annual_radiation_kwh": round(annual_kwh_per_m2, 1) # Store kWh/m2
                                })
                        except Exception as calc_e:
                            logging.warning(f"Warning: Error calculating surface '{surface_name}' radiation or score: {calc_e}")
                    else:
                        logging.warning(f"Warning: No radiation data column found for surface '{surface_name}'.")

            if not suitable_surfaces:
                logging.warning("Warning: No suitable surfaces found for PV installation.")
                return []
            suitable_surfaces.sort(key=lambda x: x["radiation_score"], reverse=True)
            logging.info(f"Found {len(suitable_surfaces)} suitable surfaces for PV installation.")
            return suitable_surfaces
        except Exception as e:
            logging.error(f"Error: Error during shadow analysis: {e}")
            import traceback; traceback.print_exc()
            return None

    def add_pv_to_idf(self, suitable_surfaces: list[dict], pv_run_id: str) -> str | None:
        """Add PV systems to the base IDF model object."""
        if not suitable_surfaces:
            logging.info("Info: No suitable surfaces provided, not adding PV systems.")
            return self.base_idf_model.idf_path
        logging.info(f"--- Adding {len(suitable_surfaces)} PV systems to IDF ---")
        idf = self.base_idf_model.idf
        pv_config = self.pv_config
        pv_efficiency = pv_config.get('pv_efficiency', 0.18)
        pv_coverage = pv_config.get('pv_coverage', 0.8)
        pv_idf_path = self.work_dir / pv_run_id / f"{pv_run_id}.idf"

        try:
            # Keeping the logic for finding or creating ELCd and ELCg
            elcd = None
            elcd_name = "PV_Load_Center"
            elcg = None
            elcg_name = f"{elcd_name}_Generator_List"
            elcd_objects = idf.idfobjects.get("ElectricLoadCenter:Distribution".upper(), [])
            if not elcd_objects:
                elcd = idf.newidfobject("ElectricLoadCenter:Distribution", Name=elcd_name,
                                        Generator_Operation_Scheme_Type="Baseload", Generator_List_Name=elcg_name)
            else:
                elcd = elcd_objects[0]
                elcd_name = elcd.Name
                elcg_name = elcd.Generator_List_Name
            elcg_objects = idf.idfobjects.get("ElectricLoadCenter:Generators".upper(), [])
            for obj in elcg_objects:
                if hasattr(obj, 'Name') and obj.Name.upper() == elcg_name.upper():
                    elcg = obj
                    break
            if elcg is None: 
                elcg = idf.newidfobject("ElectricLoadCenter:Generators", Name=elcg_name)

            num_existing_generators = 0
            for i in range(1, 100):
                gen_name_field = f"Generator_{i}_Name"
                if hasattr(elcg, gen_name_field) and getattr(elcg, gen_name_field):
                    num_existing_generators = i
                else:
                    break

            # --- Add PV for each suitable surface ---
            for i, surface_info in enumerate(suitable_surfaces):
                surface_name = surface_info["name"]
                pv_name = f"PV_{surface_name.replace(' ', '_').replace(':','_')}" # Replace special characters
                perf_name = f"{pv_name}_Performance"

                # Check if a PV generator with the same name already exists, to avoid duplication
                if idf.getobject("Generator:Photovoltaic".upper(), pv_name):
                    logging.warning(f"Warning: PV generator '{pv_name}' already exists, skipping addition.")
                    continue

                idf.newidfobject("Generator:Photovoltaic", Name=pv_name, Surface_Name=surface_name,
                                    Photovoltaic_Performance_Object_Type="PhotovoltaicPerformance:Simple",
                                    Module_Performance_Name=perf_name, Heat_Transfer_Integration_Mode="Decoupled")
                if not idf.getobject("PhotovoltaicPerformance:Simple".upper(), perf_name):
                     idf.newidfobject("PhotovoltaicPerformance:Simple", Name=perf_name,
                                        Fraction_of_Surface_Area_with_Active_Solar_Cells=pv_coverage,
                                        Conversion_Efficiency_Input_Mode="Fixed",
                                        Value_for_Cell_Efficiency_if_Fixed=pv_efficiency)

                # Add to Generator List
                generator_index = num_existing_generators + i + 1 # Note that the index may change due to skipping, so we need a more robust logic, such as finding the next empty position
                # Find the next available field
                actual_index = num_existing_generators + 1
                while True:
                     gen_name_field_check = f"Generator_{actual_index}_Name"
                     if not hasattr(elcg, gen_name_field_check) or not getattr(elcg, gen_name_field_check):
                          break # Found an empty position
                     actual_index += 1
                     if actual_index > 100: # Prevent infinite loop
                          raise ValueError("Generator list is full (more than 100)")

                gen_name_field = f"Generator_{actual_index}_Name"
                gen_type_field = f"Generator_{actual_index}_Object_Type"
                setattr(elcg, gen_name_field, pv_name)
                setattr(elcg, gen_type_field, "Generator:Photovoltaic")

                # --- Add output variables (ensure no duplication) ---
                pv_rate_var = "Generator Produced DC Electricity Rate" # Unit: W
                pv_energy_var = "Generator Produced DC Electric Energy" # Unit: J (but will convert to kWh later)
                # Check and add rate variable
                if not any(ov.Key_Value.upper() == pv_name.upper() and ov.Variable_Name.lower() == pv_rate_var.lower()
                           for ov in idf.idfobjects.get("OUTPUT:VARIABLE", [])):
                    idf.newidfobject("Output:Variable", Key_Value=pv_name, Variable_Name=pv_rate_var, Reporting_Frequency="Hourly")
                # Check and add energy variable
                if not any(ov.Key_Value.upper() == pv_name.upper() and ov.Variable_Name.lower() == pv_energy_var.lower()
                           for ov in idf.idfobjects.get("OUTPUT:VARIABLE", [])):
                    idf.newidfobject("Output:Variable", Key_Value=pv_name, Variable_Name=pv_energy_var, Reporting_Frequency="Hourly") # Report J, convert to kWh later

            # adding monthly reports
            monthly_table_name = "PV Production Summary"
            if not idf.getobject("Output:Table:Monthly", monthly_table_name):
                    monthly_table = idf.newidfobject("Output:Table:Monthly", Name=monthly_table_name, Digits_After_Decimal=2)
                    # Add total electricity consumption as reference
                    setattr(monthly_table, "Variable_or_Meter_1_Name", "Electricity:Facility")
                    # (Adding total PV generation to monthly report requires Meter:Custom)

            self.base_idf_model.save(pv_idf_path) # Save IDF with PV
            logging.info(f"PV systems added, new IDF file saved to: {pv_idf_path}")
            return pv_idf_path
        except Exception as e:
            logging.error(f"Error: Error adding PV systems to IDF: {e}")
            import traceback; traceback.print_exc()
            return None

    def analyze_pv_generation(self, pv_output_prefix: str) -> dict | None:
        """Analyze the simulation results with PV, calculate PV generation (kWh, kW)."""
        logging.info("--- Analyze PV generation results ---")
        pv_sim_dir = self.work_dir / "optimized_pv"
        csv_path = pv_sim_dir / f"{pv_output_prefix}.csv"
        if not csv_path.exists():
            logging.error(f"Error: PV simulation result file not found: {csv_path}")
            return None

        try:
            df = pd.read_csv(csv_path)
            pv_results = {"systems": {}, "total_annual_kwh": 0.0}
            total_energy_kwh = 0.0
            # Find all PV generation rate columns (unit: W)
            pv_rate_var_name = "Generator Produced DC Electricity Rate".upper()
            pv_rate_cols = [col for col in df.columns if pv_rate_var_name in col.upper()]

            if not pv_rate_cols:
                logging.warning("Warning: No PV generation rate columns found.")
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
                    logging.warning(f"Warning: Error processing PV column '{col_name}': {inner_e}")

            pv_results["total_annual_kwh"] = round(total_energy_kwh, 2)
            logging.info(f"PV generation analysis completed. Total annual generation: {pv_results['total_annual_kwh']:.2f} kWh.")

            # --- Calculate monthly generation kWh ---
            if 'Date/Time' in df.columns:
                try:
                    df['Timestamp'] = pd.to_datetime(df['Date/Time'], format=' %m/%d  %H:%M:%S', errors='coerce')
                    df.dropna(subset=['Timestamp'], inplace=True)
                    df['Month'] = df['Timestamp'].dt.month

                    for pv_name, results_dict in pv_results["systems"].items():
                        results_dict["monthly_energy_kwh"] = {}
                        col = next((c for c in pv_rate_cols if pv_name.upper() in c.upper()), None) # 查找对应列
                        if col:
                            monthly_wh = df.groupby('Month')[col].sum() # 月总 Wh
                            for month in range(1, 13):
                                results_dict["monthly_energy_kwh"][month] = round(monthly_wh.get(month, 0) / 1000.0, 2) # Wh -> kWh
                except Exception as month_e:
                    logging.warning(f"Warning: Error calculating monthly PV generation: {month_e}")

            return pv_results
        except Exception as e:
            logging.error(f"Error: Error analyzing PV results: {e}")
            import traceback; traceback.print_exc()
            return None
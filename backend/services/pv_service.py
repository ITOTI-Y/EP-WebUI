# backend/services/pv_service.py

import os
import numpy as np
import pandas as pd
from pathlib import Path
from eppy.modeleditor import IDF
from fastapi import HTTPException
from dotenv import load_dotenv
import eppy.runner.run_functions as run_functions

from .idf_service import get_idf_object, get_idf_path

load_dotenv()

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR"))

class PVAnalyzer:
    """
    Analyzes and simulates photovoltaic (PV) systems integrated into EnergyPlus IDF models.

    This class handles adding PV systems to an IDF file based on suitable surfaces,
    running the EnergyPlus simulation, and analyzing the PV energy production results.
    """
    def __init__(self, idf_id: str, weather_file: str):
        """
        Initializes the PVAnalyzer.

        Args:
            idf_id: The identifier for the IDF model.
            weather_file: The path to the EPW weather file for the simulation.
        """
        self.idf_id = idf_id
        self.weather_file = weather_file
        self.idf_path = None
        self.idf_obj = None
        self.output_dir = None

    async def initialize(self):
        """
        Initializes necessary paths, IDF object, and output directory.

        Retrieves the IDF file path and object, and creates the output directory
        for simulation results.
        """
        self.idf_path = await get_idf_path(self.idf_id)
        self.idf_obj = await get_idf_object(self.idf_id)
        self.output_dir = OUTPUT_DIR / f"output_{self.idf_id}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def add_pv_systems(self, suitable_surfaces: list[dict]):
        """
        Adds PV system objects to the IDF model based on suitable surfaces.

        Creates Generator:Photovoltaic, PhotovoltaicPerformance:Simple,
        and related ElectricLoadCenter objects in the IDF model for each
        suitable surface provided. Also adds relevant output variables.

        Args:
            suitable_surfaces: A list of dictionaries, where each dictionary
                               represents a suitable surface and must contain
                               'name' (str) and 'area' (float) keys. May also
                               contain 'radiation_score'.

        Returns:
            A dictionary containing the idf_id, a list of added pv_systems details,
            and a success message.

        Raises:
            HTTPException: If an error occurs during the IDF modification process.
        """
        pv_efficiency = float(os.getenv("PV_EFFICIENCY"))
        pv_coverage = float(os.getenv("PV_COVERAGE"))

        try:
            added_pv_systems = []

            for i, surface in enumerate(suitable_surfaces):
                surface_name = surface["name"]
                area = surface["area"]
                pv_area = area * pv_coverage

                # Create PV system
                pv_name = f"PV_{surface_name}"

                # Create PV system object
                pv_generator = self.idf_obj.newidfobject(
                    "Generator:Photovoltaic",
                    Name=pv_name,
                    Surface_Name=surface_name,
                    Photovoltaic_Performance_Object_Type="PhotovoltaicPerformance:Simple",
                    Module_Performance_Name=f"{pv_name}_performance",
                    Heat_Transfer_Integration_Mode="Decoupled",
                    Number_of_Series_Strings_in_Parallel=1,
                    Number_of_Modules_in_Series=1
                )

                # Add PhotovoltaicPerformance:Simple
                pv_performance = self.idf_obj.newidfobject(
                    "PhotovoltaicPerformance:Simple",
                    Name=f"{pv_name}_performance",
                    Fraction_of_Surface_Area_with_Active_Solar_Cells=pv_coverage,
                    Conversion_Efficiency_Input_Mode="Fixed",
                    Value_for_Cell_Efficiency_if_Fixed=pv_efficiency
                )

                # Add or get ElectricLoadCenter:Distribution Object
                elcd_objects = self.idf_obj.idfobjects["ElectricLoadCenter:Distribution"]
                elcd_name = "PV_Load_Center"

                if not elcd_objects:
                    elcd = self.idf_obj.newidfobject(
                        "ElectricLoadCenter:Distribution",
                        Name=elcd_name,
                        Generator_Operation_Scheme_Type="Baseload",
                        Generator_List_Name=f"{elcd_name}_Generator_List",
                        Generator_Demand_Limit_Scheme_Purchased_Electric_Demand_Limit=0
                    )
                else:
                    elcd = elcd_objects[0]
                    elcd_name = elcd.Name

                # Add or get ElectricLoadCenter:Generators object
                elcg_objects = self.idf_obj.idfobjects["ElectricLoadCenter:Generators"]
                elcg_name = f"{elcd_name}_Generator_List"
                elcg = None

                if not elcg_objects:
                    elcg = self.idf_obj.newidfobject(
                        "ElectricLoadCenter:Generators",
                        Name=elcg_name
                    )
                else:
                    for obj in elcg_objects:
                        if obj.Name == elcg_name:
                            elcg = obj
                            break

                    if elcg is None:
                        elcg = self.idf_obj.newidfobject(
                            "ElectricLoadCenter:Generators",
                            Name=elcg_name
                        )

                # Add Generator:Photovoltaic to ElectricLoadCenter:Generators
                setattr(elcg, f"Generator_{i+1}_Name", pv_name)
                setattr(elcg, f"Generator_{i+1}_Object_Type", "Generator:Photovoltaic")
                setattr(elcg, f"Generator_{i+1}_Rated_Electric_Power_Output", 0)
                setattr(elcg, f"Generator_{i+1}_Availability_Schedule_Name", "")
                setattr(elcg, f"Generator_{i+1}_Rated_Thermal_to_Electrical_Power_Ratio", 0)

                # Add output variables
                for var_name in ["Generator Produced DC Electricity Rate", "Generator Produced DC Electricity Rate"]:
                    self.idf_obj.newidfobject(
                        "Output:Variable",
                        Key_Value= pv_name,
                        Variable_Name=var_name,
                        Reporting_Frequency="Hourly"
                    )
                
                added_pv_systems.append({
                    "pv_name": pv_name,
                    "surface_name": surface_name,
                    "area": area,
                    "pv_area": pv_area,
                    "efficiency": pv_efficiency,
                    "radiation_score": surface.get("radiation_score", 0)
                })

            # Add Monthly table
            monthly_table_exists = False
            for table in self.idf_obj.idfobjects["Output:Table:Monthly"]:
                if table.Name == "PV Production Summary":
                    monthly_table_exists = True
                    break
            
            if not monthly_table_exists:
                monthly_table = self.idf_obj.newidfobject(
                    "Output:Table:Monthly",
                    Name="PV Production Summary",
                    Digits_After_Decimal=2
                )
                setattr(monthly_table, "Variable_or_Meter_1_Name", "Generator Produced DC Electric Energy")
                setattr(monthly_table, "Aggregation_Type_for_Variable_or_Meter_1", "SumOrAverage")

            self.idf_obj.save()

            return {
                "idf_id": self.idf_id,
                "pv_systems": added_pv_systems,
                "message": f"Success Add {len(added_pv_systems)} Pv Systems"
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error when add pv systems: {str(e)}")

    async def run_pv_simulation(self):
        """
        Runs the EnergyPlus simulation with the modified IDF file.

        Executes the simulation using the specified weather file and outputs
        results to the designated output directory.

        Returns:
            A dictionary containing the idf_id, output directory path, and a
            success message.

        Raises:
            HTTPException: If an error occurs during the simulation run.
        """
        try:
            idf = IDF(self.idf_path, self.weather_file)
            run_functions.run(
                idf,
                weather=self.weather_file,
                output_directory=str(self.output_dir),
                annual=True,
                readvars=True,
                output_prefix = "pv"
            )

            return {
                "idf_id": self.idf_id,
                "output_dir": str(self.output_dir),
                "message": "Success Run PV Simulation"
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error when run pv simulation: {str(e)}")

    async def analyze_pv_results(self):
        """
        Analyzes the PV simulation results from the output CSV file.

        Reads the 'pvout.csv' file, calculates total and monthly energy
        production (kWh) and maximum power (kW) for each PV system.

        Returns:
            A dictionary containing detailed results per pv_system, the total
            energy generated across all systems, and a success message.

        Raises:
            HTTPException: If the results CSV file ('pvout.csv') is not found
                           or if an error occurs during analysis.
        """
        csv_path = self.output_dir / "pvout.csv"
        if not csv_path.exists():
            raise HTTPException(status_code=404, detail="PV simulation results not found")

        df = pd.read_csv(csv_path)

        pv_systems = {}
        total_energy_kwh = 0
        for col in df.columns:
            if "Generator Produced DC Electricity Rate" in col:
                parts = col.split(':')
                pv_name = parts[0] + ':' + parts[1]
                total_kwh = df[col].sum() / 1000
                pv_systems[pv_name] = {
                    "total_energy_kwh": total_kwh,
                    "max_power_kw": df[col].max() / 1000
                }
                total_energy_kwh += total_kwh

        if 'Date/Time' in df.columns:
            df['Month'] = df['Date/Time'].str.split('/').str[0].astype(int)
            monthly_energy = {}

            for pv_name in pv_systems:
                col_name = None
                for col in df.columns:
                    if pv_name in col and "Generator Produced DC Electricity Rate" in col:
                        col_name = col
                        break
                
                if col_name:
                    pv_systems[pv_name]["monthly_energy_kwh"] = {}
                    for month in range(1, 13):
                        month_df = df[df['Month'] == month]
                        month_energy = (month_df[col_name] / 1000).sum()  # kWh
                        pv_systems[pv_name]["monthly_energy_kwh"][month] = month_energy

        return {
            "pv_systems": pv_systems,
            "total_energy_kwh": total_energy_kwh,
            "message": "Success Analyze PV Results"
        }
    
    async def run(self, suitable_surfaces: list[dict]):
        """
        Executes the full PV analysis workflow.

        Initializes the analyzer, adds PV systems to the IDF, runs the
        simulation, and analyzes the results.

        Args:
            suitable_surfaces: A list of dictionaries defining suitable surfaces
                               for PV installation (passed to add_pv_systems).

        Returns:
            The analysis results dictionary from analyze_pv_results.

        Raises:
            HTTPException: If any step in the workflow fails.
        """
        try:
            await self.initialize()
            await self.add_pv_systems(suitable_surfaces)
            await self.run_pv_simulation()
            return await self.analyze_pv_results()
        except Exception as e:
            # Reraise exception to be handled by the caller or FastAPI
            raise HTTPException(status_code=500, detail=f"Error during PV analysis workflow: {str(e)}")

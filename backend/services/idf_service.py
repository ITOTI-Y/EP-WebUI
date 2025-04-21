# backend/services/idf_service.py
"""Service module for handling EnergyPlus IDF file uploads, storage, and retrieval."""

import os
import uuid
import numpy as np
import logging
from pathlib import Path
from dotenv import load_dotenv
from fastapi import UploadFile, HTTPException
from eppy.modeleditor import IDF
from io import BytesIO

# Define the path to the EnergyPlus Data Dictionary file.
IDD_FILE = Path(__file__).parent.parent.parent / "data" / "Energy+.idd"
# Set the IDD file path for the eppy library globally.
IDF.setiddname(str(IDD_FILE))  # Convert Path object to string

# IDF Model operation Class


class IDFModel:
    """
    Encapsulates the reading, modification, and saving of EnergyPlus IDF files, leveraging the eppy library.
    """

    def __init__(self, idf_path: str, eppy_idf_object: IDF = None):
        """
        Initialize the IDFModel object with an optional IDF path or eppy IDF object.

        Args:
            idf_path (str): The path to the IDF file.
            eppy_idf_object (IDF, optional): An existing eppy IDF object. Defaults to None.
        """
        self.idf_path = idf_path

        if eppy_idf_object is None:
            try:
                self.idf = IDF(self.idf_path)
            except Exception as e:
                raise ValueError(f"Failed to create IDF object from path: {e}")
        else:
            self.idf = eppy_idf_object
        self._zone_names = None
        self._floor_area = None

    def save(self, output_path: str = None):
        """
        Save the IDF object to a file.

        Args:
            output_path (str, optional): Destination path for saving. If None, overwrite the original file.
        """
        save_path = output_path if output_path else self.idf_path
        try:
            self.idf.saveas(save_path)
        except Exception as e:
            raise ValueError(f"Failed to save IDF object to {save_path}: {e}")

    def apply_run_peroid(self, start_year: int = None, end_year: int = None):
        """
        Add or remove a RunPeriod.

        Args:
            start_year (int, optional): Start year. Defaults to None.
            end_year (int, optional): End year. Defaults to None.
        """
        run_periods = self.idf.idfobjects.get("RunPeriod", [])

        # Setting the Start Date And End Date
        if start_year and end_year:
            logging.info(f"Configure the RunPeriod to explicitly specify the years: {start_year}-{end_year}.")
        else:
            start_year = 2025
            end_year = 2025

        if len(run_periods) > 0:
            for rp in run_periods[1:]:
                self.idf.removeidfobject(rp)

        if not run_periods:
            rp = self.idf.newidfobject(
                "RunPeriod",
                Name=f"RunPeriod_{start_year}_{end_year}",
            )
        else:
            rp = run_periods[0]

        # Set the RunPeriod filed
        rp.Begin_Month = 1
        rp.Begin_Day_of_Month = 1
        rp.Begin_Year = start_year
        rp.End_Month = 12
        rp.End_Day_of_Month = 31
        rp.End_Year = end_year
        # Leverage Weather File Holidays
        rp.Use_Weather_File_Holidays_and_Special_Days = "Yes"
        # Leverage Weather File Daylight Saving Period
        rp.Use_Weather_File_Daylight_Saving_Period = "Yes"
        rp.Apply_Weekend_Holiday_Rule = "No"  # Don't apply weekend holiday rule
        # Leverage Weather File Rain Indicators
        rp.Use_Weather_File_Rain_Indicators = "Yes"
        # Leverage Weather File Snow Indicators
        rp.Use_Weather_File_Snow_Indicators = "Yes"
        # Or "Monday", "Tuesday", etc.
        rp.Day_of_Week_for_Start_Day = "Monday"
        if hasattr(rp, 'Use_Weather_File_for_Run_Period_Calculation'):
            rp.Use_Weather_File_for_Run_Period_Calculation = "No"
        logging.info(f"RunPeriod {rp.Name} created successfully.")

    def apply_output_requests(self):
        """
        Configure output variables and reporting for the IDF object.
        """
        # --- 1. Clean up existing output related objects ---
        objects_to_remove = [
            "OutputControl:Table:Style",
            "Output:Table:SummaryReports",
            "Output:Table:Monthly",          # Clear custom monthly table.
            "Output:Table:Annual",           # Clear custom annual table.
            "Output:Table:TimeBins",         # Clear custom time bins table.
            "Output:Meter",                  # Clear meters to .eso and .mtr.
            "Output:Meter:MeterFileOnly",    # Clear meters to .mtr only.
            "Output:Meter:Cumulative",       # Clear cumulative meters.
            # Clear cumulative meters to .mtr only.
            "Output:Meter:Cumulative:MeterFileOnly",
            # Clear all variable requests. (Note: PV variables need to be added separately)
            "Output:Variable",
            "Output:SQLite",               # QLite output
            "Output:JSON",                 # SON output
            "Output:VariableDictionary",   # RDD/MDD files
        ]
        self.remove_objects_by_type(objects_to_remove)
        logging.info("Finished removing existing output related objects.")

        # --- 2. Configure table output styles (for *Table.csv) ---
        # Create a new output control table style (comma separated values, J to kWh)
        self.idf.newidfobject(
            "OutputControl:Table:Style",
            Column_Separator="Comma",
            Unit_Conversion="JtokWh"
        )
        logging.info("Set OutputControl:Table:Style to Comma and JtokWh.")

        # --- 3. Request predefined summaries and monthly reports (written to *Table.csv) ---
        # Create a new output table summary report
        summary_reports_object = self.idf.newidfobject(
            "Output:Table:SummaryReports",
        )
        reports_to_request = [
            # Annual building performance summary (includes total energy consumption, EUI, etc.)
            "AnnualBuildingUtilityPerformanceSummary",
            # Input verification and results summary (check if the model setup is reasonable)
            "InputVerificationandResultsSummary",
            # Source energy by end-use (understand the energy source composition)
            "SourceEnergyEndUseComponentsSummary",
            # Site energy by end-use (understand the building internal energy consumption distribution)
            "DemandEndUseComponentsSummary",
            # "ComponentSizingSummary",                  # Component sizing summary (HVAC equipment sizing)
            "SurfaceShadowingSummary",                 # Surface shadowing summary
            # "HVACSizingSummary",                       # HVAC system sizing summary
            # Energy meters summary (provide an overview of the main meters)
            "EnergyMeters",

            # --- 以下为月度报告精选 ---
            # Monthly electricity and natural gas consumption total
            "EnergyConsumptionElectricityNaturalGasMonthly",
            # Monthly electricity consumption by end-use (lighting, equipment, fans, cooling, etc.)
            "EndUseEnergyConsumptionElectricityMonthly",
            # Monthly natural gas consumption by end-use (heating, hot water, etc.)
            "EndUseEnergyConsumptionNaturalGasMonthly",
            # "EnergyConsumptionDistrictHeatingCoolingMonthly", # If using district heating/cooling, uncomment this
            # "ZoneSensibleHeatGainSummaryMonthly",        # Monthly zone sensible heat gain summary
            # "ZoneSensibleHeatLossSummaryMonthly",        # Monthly zone sensible heat loss summary
            "ZoneCoolingSummaryMonthly",                 # Monthly zone cooling load summary
            "ZoneHeatingSummaryMonthly",                 # Monthly zone heating load summary
            # "ComfortReportSimple55Monthly",              # Simple comfort report based on ASHRAE 55
            # "OutdoorAirSummary",                         # Outdoor air summary
        ]
        for i, report_name in enumerate(reports_to_request):
            field_name = f"Report_{i+1}_Name"
            if field_name not in summary_reports_object.fieldnames:
                pass
            try:
                setattr(summary_reports_object, field_name, report_name)
                logging.info(f"Requested summary report: {report_name}")
            except Exception as e:
                logging.error(
                    f"Failed to set field {field_name} for Output:Table:SummaryReports. Error: {e}. This might indicate an issue with eppy's extensible field handling for this specific object or exceeding a predefined limit.")

        logging.info(
            f"Requested {len(reports_to_request)} summary/monthly reports for tabular output.")

        # --- 4. Request-by-request meter data ---
        meters_to_add = [
            # --- Power consumption End Use ---
            # Electricity for Indoor Lighting
            ("InteriorLights:Electricity", "Hourly"),
            # Electricity for Indoor Equipment
            ("InteriorEquipment:Electricity", "Hourly"),
            # Electricity for Fans
            ("Fans:Electricity", "Hourly"),
            # Electricity for Cooling Systems (includes DX coils, chillers, etc.)
            ("Cooling:Electricity", "Hourly"),
            # Electricity for Pumps
            ("Pumps:Electricity", "Hourly"),
            # Electricity for Heating Systems (if using electric heating)
            ("Heating:Electricity", "Hourly"),
            # ("ExteriorLights:Electricity", "Hourly"),     # Electricity for Outdoor Lighting (if model has outdoor lighting)
            # ("Refrigeration:Electricity", "Hourly"),      # Electricity for Refrigeration Equipment (if separately modeled)
            # ("HeatRejection:Electricity", "Hourly"),      # Electricity for Heat Rejection Equipment (如冷却塔风扇)
            # ("Humidifier:Electricity", "Hourly"),         # Electricity for Humidifiers
            # ("HeatRecovery:Electricity", "Hourly"),       # Electricity for Heat Recovery Equipment

            # --- Natural Gas Consumption End Use ---
            # Natural Gas for Heating
            ("Heating:NaturalGas", "Hourly"),
            # Natural Gas for Water Heating
            ("Water Heater:WaterSystems:NaturalGas", "Hourly"),
            # Natural Gas for Indoor Equipment
            ("InteriorEquipment:NaturalGas", "Hourly"),

            # --- Other Fuels ---
            # ("Heating:FuelOilNo1", "Hourly"),             # Heating Fuel Oil No. 1
            # ("Heating:Propane", "Hourly"),                # Heating Propane
            # ("Heating:Diesel", "Hourly"),                 # Heating Diesel

            # --- District Heating/Cooling ---
            # ("DistrictCooling:Facility", "Hourly"),       # District Cooling Total Consumption
            # ("DistrictHeating:Facility", "Hourly"),       # District Heating Total Consumption

            # --- Facility Level Meters ---
            # Total Building Electricity Consumption (Grid Input)
            ("Electricity:Facility", "Hourly"),
            # Total Building Natural Gas Consumption
            ("NaturalGas:Facility", "Hourly"),
            # Total Building Water Consumption
            ("Water:Facility", "Hourly"),
            # Total Building Electricity Production (e.g., PV generation)
            ("ElectricityProduced:Facility", "Hourly"),

            # --- Energy Transfer Related ---
            # Heating Energy Transfer
            ("Heating:EnergyTransfer", "Hourly"),
            # Cooling Energy Transfer
            ("Cooling:EnergyTransfer", "Hourly"),
            # Building Total Energy Transfer
            ("EnergyTransfer:Building", "Hourly"),
            # HVAC System Energy Transfer
            ("EnergyTransfer:HVAC", "Hourly"),
        ]

        added_meters_count = 0
        for meter_name, frequency in meters_to_add:
            existing = self.idf.getobject(
                "Output:Meter:MeterFileOnly".upper(), meter_name)
            if existing:
                logging.info(f"Meter {meter_name} already exists.")
            else:
                self.idf.newidfobject(
                    "Output:Meter:MeterFileOnly",
                    Key_Name=meter_name,
                    Reporting_Frequency=frequency
                )
                added_meters_count += 1
        output_control_list = [
            "Output_SQLite",
            # "Output:JSON",
        ]
        self.idf.newidfobject("OutputControl:Files", **{f"{output_control}":'Yes' for output_control in output_control_list})
        logging.info(f"Requested {added_meters_count} hourly meters (output to *.mtr and affects readvars *.csv).")

    def apply_simulation_control_settings(self, run_for_sizing: bool = False, run_for_weather=True):
        """
        Configure simulation control settings for the IDF object.

        Args:
            run_for_sizing (bool, optional): Whether to run for sizing. Defaults to False.
            run_for_weather (bool, optional): Whether to run for weather. Defaults to True.
        """
        sim_control_list = self.idf.idfobjects.get("SimulationControl", [])
        if sim_control_list:
            sim_control = sim_control_list[0]
            sim_control.Run_Simulation_for_Sizing_Periods = 'Yes' if run_for_sizing else 'No'
            sim_control.Run_Simulation_for_Weather_File_Run_Periods = 'Yes' if run_for_weather else 'No'

            # Log the changes
            logging.info("Applied simulation control settings to IDF object.")
        else:
            logging.warning("SimulationControl object not found in IDF.")

    def remove_objects_by_type(self, object_type_list: list[str]):
        """
        Remove specific object types from the IDF object.

        Args:
            object_type_list (list[str]): Contains the object types to be removed.
        """
        for obj_type in object_type_list:
            objects = self.idf.idfobjects.get(obj_type.upper(), [])
            for obj in objects:
                self.idf.removeidfobject(obj)
            logging.info(f"Removed {len(objects)} objects of type: {obj_type}")

    def get_zone_names(self):
        """
        Get the names of all zones in the IDF object.

        Returns:
            list[str]: A list of zone names.
        """
        if self._zone_names is None:
            zones = self.idf.idfobjects.get("ZONE", [])
            self._zone_names = [zone.Name for zone in zones]
        return self._zone_names

    def apply_wall_insulation(self, r_value_si: float):
        """
        Apply wall insulation to exterior walls and roofs in the IDF object.

        Args:
            r_value_si (float): The R-value in SI units (m2K/W).
        """
        if r_value_si <= 0:
            raise ValueError("Wall insulationR-value must be greater than 0.")

        # Define the wall insulation material
        insu_mat_name = f"ExteriorInsulation_R{r_value_si}"
        self.idf.newidfobject(
            "Material:NoMass",
            Name=insu_mat_name,
            Roughness="Smooth",  # Default roughness for insulation
            Thermal_Resistance=r_value_si,  # R-value in SI units
            Thermal_Absorptance=0.9,  # Default absorptance for insulation
            Solar_Absorptance=0.6,  # Default Solar Absorptance
            Visible_Absorptance=0.7,  # Default Visible Absorptance
        )

        # Define the schedule for the insulation
        sched_name = "WallInsuSched_AlwaysOn"  # Name of the schedule
        # Check if the schedule already exists
        if not self.idf.getobject("Schedule:Constant", sched_name):
            self.idf.newidfobject(
                "Schedule:Compact",
                Name=sched_name,
                Schedule_Type_Limits_Name="Fraction",
                Field_1="Through: 12/31",
                Field_2="For: AllDays",
                Field_3="Until: 24:00",
                Field_4="1.0",
            )

        # Search all exterior walls and roofs
        surfaces = self.idf.idfobjects.get("BuildingSurface:Detailed", [])
        exterior_surfaces = []
        for surf in surfaces:
            if surf.Outside_Boundary_Condition.upper() == "OUTDOORS" and \
                    surf.Surface_Type.upper() in ["WALL", "ROOF"]:
                exterior_surfaces.append(surf.Name)

        # Apply Moveable Insulation to all exterior surfaces
        # Remove existing movable insulation objects
        self.remove_objects_by_type(['SurfaceControl:MovableInsulation'])
        for surf_name in exterior_surfaces:
            self.idf.newidfobject(
                "SurfaceControl:MovableInsulation",
                Insulation_Type="Outside",
                Surface_Name=surf_name,
                Material_Name=insu_mat_name,
                Schedule_Name=sched_name
            )

        # Log the changes
        logging.info(
            f"Applied insulation to {len(exterior_surfaces)} surfaces.")

    def apply_air_infiltration(self, ach_rate: float):
        """
        Apply air infiltration(ACH) to all zones in the IDF object.

        Args:
            ach_rate (float): Air change per Hour (Times/Hour)
        """

        # Remove existing Air Infiltration objects
        self.remove_objects_by_type([
            "ZoneInfiltration:DesignFlowRate"
        ])

        # Define the schedule for the infiltration
        sched_name = "InfilSched_AlwaysOn"
        if not self.idf.getobject("Schedule:Constant", sched_name):
            self.idf.newidfobject(
                "Schedule:Compact",
                Name=sched_name,
                Schedule_Type_Limits_Name="Fraction",
                Field_1="Through: 12/31",
                Field_2="For: AllDays",
                Field_3="Until: 24:00",
                Field_4="1.0",
            )

        # Apply Air Infiltration to all zones
        zoom_names = self.get_zone_names()
        for i, zone_name in enumerate(zoom_names):
            infiltration_object_name = f"ZoneInfil_{zone_name.replace(' ', '_')}"
            self.idf.newidfobject(
                "ZoneInfiltration:DesignFlowRate",
                Name=infiltration_object_name,
                Zone_or_ZoneList_or_Space_or_SpaceList_Name=zone_name,
                Schedule_Name=sched_name,
                Design_Flow_Rate_Calculation_Method="AirChanges/Hour",  # Air Changes per Hour
                Air_Changes_per_Hour=ach_rate,
            )

        # Log the changes
        logging.info(f"Applied air infiltration to {len(zoom_names)} zones.")

    def apply_window_properties(self, u_value: float, shgc: float, vt: float):
        """
        Modify window properties for all windows in the IDF object.
        Create new window material and assign to all windows in FenestrationSurface:Detailed.

        Args:
            u_value (float): Window U-value (W/m2K)
            shgc (float): Window Solar Heat Gain Coefficient
            vt (float): Window Visible Transmittance
        """

        if u_value <= 0 or shgc <= 0:
            return

        # Create a new window material
        # Define the new material name
        new_glazing_material_name = f"SimpleGlazing_U{u_value}_SHGC{shgc}"
        # Check if the material already exists
        existing_materials = self.idf.getobject(
            'WindowMaterial:SimpleGlazingSystem', new_glazing_material_name)
        if not existing_materials:
            self.idf.newidfobject(
                'WindowMaterial:SimpleGlazingSystem',
                Name=new_glazing_material_name,
                UFactor=u_value,
                Solar_Heat_Gain_Coefficient=shgc,
                Visible_Transmittance=vt
            )

        # Create a new window construction
        new_construction_name = f"Construction_Window_{new_glazing_material_name}"
        existing_construction = self.idf.getobject(
            'Construction', new_construction_name)
        if not existing_construction:
            self.idf.newidfobject(
                "Construction",
                Name=new_construction_name,
                Outside_Layer=new_glazing_material_name
            )

        # Get all FenestrationSurface:Detailed objects of type Window
        fenestration_surfaces = self.idf.idfobjects.get(
            "FenestrationSurface:Detailed", [])
        window_count = 0
        for fen_surf in fenestration_surfaces:
            if fen_surf.Surface_Type.upper() == "WINDOW":
                fen_surf.Construction_Name = new_construction_name
                window_count += 1

        # Log the changes
        logging.info(f"Applied window properties to {window_count} windows.")

    def apply_cooling_cop(self, cop: float):
        """
        Change the cooling COP for all cooling systems in the IDF object.

        Args:
            cop (float): Cooling COP
        """
        if cop <= 0:
            return  # Return if COP is not positive

        modified_count = 0
        # Example modified Single-Speed DX Cooling coil
        dx_coils = self.idf.idfobjects.get('Coil:Cooling:DX:SingleSpeed', [])
        for coil in dx_coils:
            # Assume COP field name is 'Rated_COP' (need to check IDD for confirmation)
            # Check if the coil has a Gross_Rated_Cooling_COP attribute
            if hasattr(coil, 'Gross_Rated_Cooling_COP'):
                coil.Gross_Rated_Cooling_COP = cop
                modified_count += 1

        # Example modified Chiller:Electric:EIR
        chillers = self.idf.idfobjects.get('Chiller:Electric:EIR', [])
        for chiller in chillers:
            # Assume COP field name is 'Reference_COP' (need to check IDD for confirmation)
            if hasattr(chiller, 'Reference_COP'):
                chiller.Reference_COP = cop
                modified_count += 1

        # Log the changes
        logging.info(
            f"Applied cooling COP to {modified_count} cooling systems.")

    def apply_cooling_supply_temp(self, temp_celsius: float):
        """
        Modify the cooling supply temperature for all Sizing:Zone objects in the IDF object.

        Args:
            temp_celsius (float): Cooling supply temperature in Celsius
        """
        if temp_celsius <= 0:
            return  # Return if temperature is not positive

        sizing_zone_objects = self.idf.idfobjects.get('Sizing:Zone', [])
        modified_count = 0
        for sz in sizing_zone_objects:
            if hasattr(sz, 'Zone_Cooling_Design_Supply_Air_Temperature'):
                sz.Zone_Cooling_Design_Supply_Air_Temperature = temp_celsius
                modified_count += 1
        if modified_count == 0:
            logging.warning("No Sizing:Zone objects found in IDF.")
        else:
            logging.info(
                f"Applied cooling supply temperature to {modified_count} Sizing:Zone objects.")

    def apply_lighting_reduction(self, reduction_factor: float, building_type: str):
        """
        Modify the lighting reduction factor for all Lights objects in the IDF object.

        Args:
            reduction_factor (float): Lighting reduction factor
            building_type (str): Building type (dosen't matter for now)
        """
        if reduction_factor <= 0 or reduction_factor >= 1:
            return  # Return if reduction factor is not between 0 and 1

        # Get all Lights objects
        lights_objects = self.idf.idfobjects.get('Lights', [])
        modified_count = 0
        for light in lights_objects:
            calc_method = light.Design_Level_Calculation_Method.upper()

            if calc_method == "LIGHTINGLEVEL":  # Based on absolute power
                if hasattr(light, 'Lighting_Level') and light.Lighting_Level > 0:
                    original_level = light.Lighting_Level
                    light.Lighting_Level = original_level * reduction_factor
                    modified_count += 1
            elif calc_method == "WATTS/AREA":  # Based on watts per square meter
                if hasattr(light, 'Watts_per_Floor_Area') and light.Watts_per_Floor_Area > 0:
                    original_wpa = light.Watts_per_Floor_Area
                    light.Watts_per_Floor_Area = original_wpa * reduction_factor
                    modified_count += 1
            elif calc_method == "WATTS/PERSON":  # Based on watts per person
                if hasattr(light, 'Watts_per_Person') and light.Watts_per_Person > 0:
                    original_wpp = light.Watts_per_Person
                    light.Watts_per_Person = original_wpp * reduction_factor
                    modified_count += 1
            else:
                logging.warning(
                    f"Unsupported lighting calculation method: {calc_method} for light: {light.Name}")

        if modified_count == 0:
            logging.warning("No Lights objects found in IDF.")
        else:
            logging.info(
                f"Applied lighting reduction to {modified_count} Lights objects.")

    def apply_natural_ventilation(self, opening_area_m2: float):
        """
        Apply natural ventilation to all zones in the IDF object.

        Args:
            opening_area_m2 (float): Effective Opening area for each zone (m2).
        """

        if opening_area_m2 <= 0:
            return  # Return if opening area is not positive

        # Remove existing AirflowNetwork:Distribution objects
        self.remove_objects_by_type(['ZoneVentilation:WindandStackOpenArea'])

        # Define the schedule for the natural ventilation
        sched_name = "NatVentSched_AlwaysOn"
        if not self.idf.getobject("Schedule:Constant", sched_name):
            self.idf.newidfobject(
                'Schedule:Compact',
                Name=sched_name,
                Schedule_Type_Limits_Name="Fraction",
                Field_1="Through: 12/31",
                Field_2="For: AllDays",
                Field_3="Until: 24:00",
                Field_4=1.0
            )

        # Apply natural ventilation objects for each zone
        zone_names = self.get_zone_names()
        for i, zone_name in enumerate(zone_names):
            nv_object_name = f"NatVent_{zone_name.replace(' ', '_')}"
            self.idf.newidfobject(
                "ZoneVentilation:WindandStackOpenArea",
                Name=nv_object_name,
                Zone_or_Space_Name=zone_name,
                Opening_Area=opening_area_m2,
                Opening_Area_Fraction_Schedule_Name=sched_name,
                # Autocalculate the opening effectiveness
                Opening_Effectiveness="Autocalculate",
                # Autocalculate the discharge coefficient for the opening
                Discharge_Coefficient_for_Opening="Autocalculate",
                # Accroding to archive/ECM.py file
                Minimum_Indoor_Temperature=22,  # Indoor minimum temperature
                # Indoor maximum temperature (set high value means no limit)
                Maximum_Indoor_Temperature=100,
                Delta_Temperature=1,  # Indoor-outdoor minimum temperature difference
                Minimum_Outdoor_Temperature=18,  # Outdoor minimum temperature
                Maximum_Outdoor_Temperature=28,  # Outdoor maximum temperature
                Maximum_Wind_Speed=15  # Maximum wind speed
            )

        # Log the changes
        logging.info(
            f"Applied natural ventilation to {len(zone_names)} zones. {opening_area_m2} m2 of opening area for each zone.")

    def get_surface_area(self, surface_name: str):
        """
        Return the area of the specified surface.

        Args:
            surface_name (str): The name of the surface to calculate the area for.

        Returns:
            float: The surface area in square meters.
        """
        surface = self.idf.getobject(
            'BuildingSurface:Detailed'.upper(), surface_name)
        if not surface:
            return 0.0
        return surface.area

    def get_total_floor_area(self):
        """
        Get the total floor area of the building by summing zone areas.
        If a zone's area is 'autocalculate', it calculates the area from the
        geometry of its floor surfaces ('BuildingSurface:Detailed' with Surface_Type 'Floor').

        Returns:
            float: The total floor area in square meters. Returns 1.0 if calculation fails or yields zero.
        """
        # Check cache first
        if self._floor_area is not None:
            return self._floor_area

        total_area = 0.0
        zones = self.idf.idfobjects.get('ZONE', [])
        # Get all surfaces once to avoid repeated lookups inside the loop
        all_surfaces = self.idf.idfobjects.get('BUILDINGSURFACE:DETAILED', [])

        if not zones:
            logging.warning("No ZONE objects found in the IDF file.")
            self._floor_area = 1.0  # Set cache to default
            return 1.0

        for zone in zones:
            zone_area = 0.0
            zone_name = zone.Name
            # Default to autocalculate if field missing
            floor_area_field = getattr(zone, 'Floor_Area', 'autocalculate')

            # Try converting the field value to float first
            try:
                zone_area = float(floor_area_field)
                if zone_area > 0:
                    total_area += zone_area
                    # Log the source of the area
                    # logging.debug(f"Zone '{zone_name}': Using explicit Floor_Area = {zone_area} m2.")
                    continue  # Go to the next zone
                else:
                    # Handle cases like 0.0 explicitly entered, treat as autocalculate needed
                    logging.debug(
                        f"Zone '{zone_name}': Explicit Floor_Area is {zone_area}, treating as autocalculate.")
                    floor_area_field = 'autocalculate'  # Force recalculation
            except (ValueError, TypeError):
                # If conversion fails, it's likely a string like 'autocalculate'
                if isinstance(floor_area_field, str) and (floor_area_field.lower() == 'autocalculate' or floor_area_field.lower() == ''):
                    # Calculate area from floor surfaces geometry within this zone
                    calculated_zone_area = 0.0
                    zone_floor_surfaces = [
                        s for s in all_surfaces
                        # Case-insensitive compare
                        if hasattr(s, 'Zone_Name') and s.Zone_Name.lower() == zone_name.lower()
                        and hasattr(s, 'Surface_Type') and s.Surface_Type.lower() == 'floor'
                    ]

                    if not zone_floor_surfaces:
                        logging.warning(
                            f"Zone '{zone_name}' Area is 'autocalculate' but no 'Floor' type surfaces were found for it.")
                    else:
                        # logging.debug(f"Zone '{zone_name}': Calculating area from {len(zone_floor_surfaces)} floor surfaces.")
                        for surface in zone_floor_surfaces:
                            surface_name = surface.Name
                            surf_area = surface.area
                            # logging.debug(f"  - Surface '{surface_name}': Calculated Area = {surf_area} m2.")
                            calculated_zone_area += surf_area

                        if calculated_zone_area > 0:
                            zone_area = calculated_zone_area
                            total_area += zone_area
                            # logging.debug(f"Zone '{zone_name}': Calculated Floor_Area = {zone_area} m2 from geometry.")
                        else:
                            # If geometric calculation failed or yielded zero, log a warning.
                            # We won't fall back to Sizing:Zone here as the primary method failed.
                            logging.warning(
                                f"Zone '{zone_name}': Area is 'autocalculate' but geometric calculation yielded {calculated_zone_area:.4f} m2. Area not added.")
                else:
                    # Handle other unexpected string values
                    logging.warning(
                        f"Zone '{zone_name}' has an unsupported Floor_Area value: '{floor_area_field}'. Area not counted.")

        # Cache the result, ensuring it's at least 1.0 if valid area is zero or calculation failed
        self._floor_area = total_area if total_area > 0 else 1.0
        if total_area <= 0:
            logging.warning(
                f"Calculated total floor area is {total_area:.4f}. Returning default value of 1.0 m2.")

        # logging.info(f"Total calculated building floor area: {self._floor_area} m2.")
        return self._floor_area

# backend/services/idf_service.py
"""Service module for handling EnergyPlus IDF file uploads, storage, and retrieval."""

import os
import uuid
import logging
from pathlib import Path
from dotenv import load_dotenv
from fastapi import UploadFile, HTTPException
from eppy.modeleditor import IDF
from io import BytesIO

load_dotenv()
temp_dir = Path(os.getenv("TEMP_DIR", "./temp")) # Use ./temp as default if not set
temp_dir.mkdir(parents=True, exist_ok=True) # Ensure temp directory exists

# Define the path to the EnergyPlus Data Dictionary file.
IDD_FILE = Path(__file__).parent.parent.parent / "data" / "Energy+.idd"
# Set the IDD file path for the eppy library globally.
IDF.setiddname(str(IDD_FILE)) # Convert Path object to string

# In-memory storage for IDF file data (path and parsed object instance).
# Keys are UUIDs generated when files are saved.
idf_storage = {}

async def save_idf_file(file: UploadFile) -> str:
    """Saves an uploaded IDF file, validates it, stores it temporarily,
    parses it using eppy, and keeps track of it in memory.

    Args:
        file: The uploaded IDF file object from FastAPI.

    Returns:
        A unique identifier (UUID string) for the saved IDF file.

    Raises:
        HTTPException: If the file is not a .idf file (400).
        HTTPException: If there's an error reading or parsing the file (500).
    """
    if not file.filename or not file.filename.endswith(".idf"):
        raise HTTPException(
            status_code=400,
            detail="Invalid file format. Please upload a .idf file."
        )

    try:
        idf_id = str(uuid.uuid4())
        file_content = await file.read()

        # Ensure the temp directory exists
        temp_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = temp_dir / f"{idf_id}.idf"

        # Save the uploaded file content to a temporary file
        with open(tmp_path, "wb") as f:
            f.write(file_content)

        # Initialize the IDF object using the temporary file path
        # Convert tmp_path to string as IDF constructor expects str
        idf_instance = IDF(str(tmp_path))

        # Store the path and the parsed IDF instance in memory
        idf_storage[idf_id] = {
            "path": str(tmp_path), # Store path as string
            "instance": idf_instance,
        }

        return idf_id

    except Exception as e:
        # Log the detailed error for debugging purposes
        print(f"Error processing IDF file '{file.filename}': {e}")
        # Raise a generic server error to the client
        raise HTTPException(status_code=500, detail="Failed to process IDF file.")

async def get_idf_object(idf_id: str) -> IDF:
    """Retrieves the parsed eppy IDF object instance using its unique ID.

    Args:
        idf_id: The unique identifier (UUID string) of the IDF file.

    Returns:
        The eppy IDF object instance.

    Raises:
        HTTPException: If no IDF object is found for the given ID (404).
    """
    storage_entry = idf_storage.get(idf_id)
    if not storage_entry:
        raise HTTPException(status_code=404, detail="IDF object not found.")
    return storage_entry["instance"]

async def get_idf_path(idf_id: str) -> str:
    """Retrieves the temporary file path of the stored IDF file using its ID.

    Args:
        idf_id: The unique identifier (UUID string) of the IDF file.

    Returns:
        The absolute path (string) to the temporary IDF file.

    Raises:
        HTTPException: If no IDF object path is found for the given ID (404).
    """
    storage_entry = idf_storage.get(idf_id)
    if not storage_entry:
        raise HTTPException(status_code=404, detail="IDF object path not found.")
    return storage_entry["path"]

class MockUploadFile:
    """A mock object simulating FastAPI's UploadFile for testing purposes."""
    def __init__(self, filename: str, content: bytes):
        """Initializes the mock file with a filename and content.

        Args:
            filename: The simulated name of the uploaded file.
            content: The byte content of the simulated file.
        """
        self.filename = filename
        self._content = BytesIO(content) # Use a private attribute

    async def read(self) -> bytes:
        """Reads the entire content of the mock file."""
        # Reset position in case it was read before, though typically read once.
        self._content.seek(0)
        return self._content.read()

    # Add close method if needed by consumers, though not strictly required by UploadFile interface
    def close(self):
        """Closes the BytesIO stream."""
        self._content.close()

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
        
    def apply_run_peroid(self, start_year:int=None, end_year:int=None):
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
            if len(run_periods) > 0:
                for rp in run_periods[1:]:
                    self.idf.removeidfobject(rp)
        
            if not run_periods:
                rp = self.idf.newidfobject(
                    "RunPeriod",
                    Name = f"RunPeriod_{start_year}_{end_year}",
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
            rp.Use_Weather_File_Holidays_and_Special_Days = "Yes" # Leverage Weather File Holidays
            rp.Use_Weather_File_Daylight_Saving_Period = "Yes" # Leverage Weather File Daylight Saving Period
            rp.Apply_Weekend_Holiday_Rule = "No" # Don't apply weekend holiday rule
            rp.Use_Weather_File_Rain_Indicators = "Yes" # Leverage Weather File Rain Indicators
            rp.Use_Weather_File_Snow_Indicators = "Yes" # Leverage Weather File Snow Indicators
            rp.Day_of_Week_for_Start_Day = "Monday" # Or "Monday", "Tuesday", etc.
            if hasattr(rp, 'Use_Weather_File_for_Run_Period_Calculation'):
                rp.Use_Weather_File_for_Run_Period_Calculation = "No"
            logging.info(f"RunPeriod {rp.Name} created successfully.")

        else:
            logging.warning("Warning: No run period mode specified (year not provided). The RunPeriod object remains unchanged.")
        
    def apply_output_requests(self):
        """
        Configure output variables and reporting for the IDF object.
        """
        # Remove existing output related objects
        self._remove_objects_by_type([
            "OutputControl:Table:Style",
            "Output:Table:SummaryReports",
            "Output:Meter",
            "Output:Meter:MeterFileOnly",
            # "Output:Variable" # Optional, remove the variable output
        ])

        # Create a new output control table style (comma separated values, J to kWh)
        self.idf.newidfobject(
            "OutputControl:Table:Style",
            Column_Separator="Comma",
            Unit_Conversion="JtokWh"
        )

        # Create a new output table summary report
        self.idf.newidfobject(
            "Output:Table:SummaryReports",
            Report_1_Name = "AllSummaryAndMonthly"
        )

        # Create a new meter for reporting
        meters_to_add = [
            ("InteriorLights:Electricity", "Hourly"),
            ("InteriorEquipment:Electricity", "Hourly"),
            ("Refrigeration:Electricity", "Hourly"),
            ("Fans:Electricity", "Hourly"),
            ("Cooling:Electricity", "Hourly"),
            ("Pumps:Electricity", "Hourly"),
            ("Heating:Electricity", "Hourly"),
            ("Heating:NaturalGas", "Hourly"),
            ("Water Heater:WaterSystems:NaturalGas", "Hourly"),
            ("Water:Facility", "Hourly"),
            ("Electricity:Facility", "Hourly"),
            ("NaturalGas:Facility", "Hourly"),
            ("Heating:EnergyTransfer", "Hourly"),
            ("Cooling:EnergyTransfer", "Hourly"),
            ("EnergyTransfer:Building", "Hourly"),
            ("EnergyTransfer:HVAC", "Hourly"),
        ]

        for meter_name, frequency in meters_to_add:
            self.idf.newidfobject(
                "Output:Meter:MeterFileOnly",
                Key_Name = meter_name,
                Reporting_Frequency = frequency
            )
        
        # Log the changes
        logging.info("Applied output requests to IDF object.")

    def apply_simulation_control_settings(self, run_for_sizing: bool = False, run_for_weather=True):
        """
        Configure simulation control settings for the IDF object.

        Args:
            run_for_sizing (bool, optional): Whether to run for sizing. Defaults to False.
            run_for_weather (bool, optional): Whether to run for weather. Defaults to True.
        """
        sim_control = self.idf.getobject(
            "SimulationControl",
            "SimulationControl"
            )
        if sim_control:
            sim_control.Run_Simulation_for_Sizing_Periods = 'Yes' if run_for_sizing else 'No'
            sim_control.Run_Simulation_for_Weather_Run_Periods = 'Yes' if run_for_weather else 'No'

            # Log the changes
            logging.info("Applied simulation control settings to IDF object.")
        else:
            logging.warning("SimulationControl object not found in IDF.")

    def _remove_objects_by_type(self, object_type_list: list[str]):
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
        insu_mat_name = f"ExteriorInsulation{r_value_si}"
        self.idf.newidfobject(
            "Material:NoMass",
            Name = insu_mat_name,
            Roughness = "Smooth", # Default roughness for insulation
            Thermal_Resistance = r_value_si, # R-value in SI units
            Thermal_Absorptance = 0.9, # Default absorptance for insulation
            Solar_Absorptance = 0.6, # Default Solar Absorptance
            Visible_Absorptance = 0.7, # Default Visible Absorptance
        )

        # Define the schedule for the insulation
        sched_name = "WallInsuSched_AlwaysOn" # Name of the schedule
        # Check if the schedule already exists
        if not self.idf.getobject("Schedule:Constant", sched_name):
            self.idf.newidfobject(
                "Schedule:Compact",
                Name = sched_name,
                Schedule_Type_Limits_Name = "Fractional",
                Field_1 = "Through: 12/31",
                Field_2 = "For: AllDays",
                Field_3 = "Until: 24:00",
                Field_4 = "1.0",
            )
        
        # Search all exterior walls and roofs
        surfaces = self.idf.idfobjects.get("BuildingSurface:Detailed", [])
        exterior_surfaces = []
        for surf in surfaces:
            if surf.Outside_Boundary_Condition.upper() == "OUTDOORS" and \
                surf.Surface_Type.upper() in ["WALL", "ROOF"]:
                exterior_surfaces.append(surf.Name)

        # Apply Moveable Insulation to all exterior surfaces
        for surf_name in exterior_surfaces:
            self.idf.newidfobject(
                "SurfaceControl:MovableInsulation",
                Insulation_Type = "Outside",
                Surface_Name=surf_name,
                Material_Name=insu_mat_name,
                Schedule_Name=sched_name
            )

        # Log the changes
        logging.info(f"Applied insulation to {len(exterior_surfaces)} surfaces.")

    def apply_air_infiltration(self, ach_rate: float):
        """
        Apply air infiltration(ACH) to all zones in the IDF object.

        Args:
            ach_rate (float): Air change per Hour (Times/Hour)
        """

        # Remove existing Air Infiltration objects
        self._remove_objects_by_type([
            "ZoneInfiltration:DesignFlowRate"
        ])

        # Define the schedule for the infiltration
        sched_name = "InfilSched_AlwaysOn"
        if not self.idf.getobject("Schedule:Constant", sched_name):
            self.idf.newidfobject(
                "Schedule:Compact",
                Name = sched_name,
                Schedule_Type_Limits_Name = "Fraction", #
                Field_1 = "Through: 12/31",
                Field_2 = "For: AllDays",
                Field_3 = "Until: 24:00",
                Field_4 = "1.0",
            )

        # Apply Air Infiltration to all zones
        zoom_names = self.get_zone_names()
        for i, zone_name in enumerate(zoom_names):
            infiltration_object_name = f"ZoneInfil_{zone_name.replace(' ', '_')}"
            self.idf.newidfobject(
                "ZoneInfiltration:DesignFlowRate",
                Name = infiltration_object_name,
                Zone_or_ZoneList_or_Space_or_SpaceList_Name = zone_name,
                Schedule_Name = sched_name,
                Design_Flow_Rate_Calculation_Method = "AirChanges/Hour", # Air Changes per Hour
                Air_Changes_per_Hour = ach_rate,
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

        if u_value <= 0 or shgc <= 0: return

        # Create a new window material
        new_glazing_material_name = f"SimpleGlazing_U{u_value}_SHGC{shgc}" # Define the new material name
        # Check if the material already exists
        existing_materials = self.idf.getobject('WindowMaterial:SimpleGlazingSystem', new_glazing_material_name)
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
        existing_construction = self.idf.getobject('Construction', new_construction_name)
        if not existing_construction:
            self.idf.newidfobject(
                "Construction",
                Name=new_construction_name,
                Outside_Layer=new_glazing_material_name
            )

        # Get all FenestrationSurface:Detailed objects of type Window
        fenestration_surfaces = self.idf.idfobjects.get("FenestrationSurface:Detailed", [])
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
        if cop <= 0: return # Return if COP is not positive

        modified_count = 0
        # Example modified Single-Speed DX Cooling coil
        dx_coils = self.idf.idfobjects.get('Coil:Cooling:DX:SingleSpeed', [])
        for coil in dx_coils:
            # Assume COP field name is 'Rated_COP' (need to check IDD for confirmation)
            if hasattr(coil, 'Gross_Rated_Cooling_COP'): # Check if the coil has a Gross_Rated_Cooling_COP attribute
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
        logging.info(f"Applied cooling COP to {modified_count} cooling systems.")

    def apply_cooling_supply_temp(self, temp_celsius: float):
        """
        Modify the cooling supply temperature for all Sizing:Zone objects in the IDF object.

        Args:
            temp_celsius (float): Cooling supply temperature in Celsius
        """
        if temp_celsius <= 0: return # Return if temperature is not positive

        sizing_zone_objects = self.idf.idfobjects.get('Sizing:Zone', [])
        modified_count = 0
        for sz in sizing_zone_objects:
            if hasattr(sz, 'Zone_Cooling_Design_Supply_Air_Temperature'):
                sz.Zone_Cooling_Design_Supply_Air_Temperature = temp_celsius
                modified_count += 1
        if modified_count == 0:
            logging.warning("No Sizing:Zone objects found in IDF.")
        else:
            logging.info(f"Applied cooling supply temperature to {modified_count} Sizing:Zone objects.")

    def apply_lighting_reduction(self, reduction_factor: float, building_type: str):
        """
        Modify the lighting reduction factor for all Lights objects in the IDF object.

        Args:
            reduction_factor (float): Lighting reduction factor
            building_type (str): Building type (dosen't matter for now)
        """
        if reduction_factor <= 0 or reduction_factor >= 1:
            return # Return if reduction factor is not between 0 and 1
        
        # Get all Lights objects
        lights_objects = self.idf.idfobjects.get('Lights', [])
        modified_count = 0
        for light in lights_objects:
            calc_method = light.Design_Level_Calculation_Method.upper()

            if calc_method == "LIGHTINGLEVEL": # Based on absolute power
                if hasattr(light, 'Lighting_Level') and light.Lighting_Level > 0:
                    original_level = light.Lighting_Level
                    light.Lighting_Level = original_level * reduction_factor
                    modified_count += 1
            elif calc_method == "WATTS/AREA": # Based on watts per square meter
                if hasattr(light, 'Watts_per_Zone_Floor_Area') and light.Watts_per_Zone_Floor_Area > 0:
                    original_wpa = light.Watts_per_Zone_Floor_Area
                    light.Watts_per_Zone_Floor_Area = original_wpa * reduction_factor
                    modified_count += 1
            elif calc_method == "WATTS/PERSON": # Based on watts per person
                if hasattr(light, 'Watts_per_Person') and light.Watts_per_Person > 0:
                    original_wpp = light.Watts_per_Person
                    light.Watts_per_Person = original_wpp * reduction_factor
                    modified_count += 1
            else:
                logging.warning(f"Unsupported lighting calculation method: {calc_method} for light: {light.Name}")

        if modified_count == 0:
            logging.warning("No Lights objects found in IDF.")
        else:
            logging.info(f"Applied lighting reduction to {modified_count} Lights objects.")

    def apply_natural_ventilation(self, opening_area_m2: float):
        """
        Apply natural ventilation to all zones in the IDF object.

        Args:
            opening_area_m2 (float): Effective Opening area for each zone (m2).
        """

        if opening_area_m2 <= 0: return # Return if opening area is not positive

        # Remove existing AirflowNetwork:Distribution objects
        self._remove_objects_by_type(['ZoneVentilation:WindandStackOpenArea'])

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
                Opening_Effectiveness="Autocalculate", # Autocalculate the opening effectiveness
                Discharge_Coefficient_for_Opening="Autocalculate", # Autocalculate the discharge coefficient for the opening
                # Accroding to archive/ECM.py file
                Minimum_Indoor_Temperature=22, # Indoor minimum temperature
                Maximum_Indoor_Temperature=100, # Indoor maximum temperature (set high value means no limit)
                Delta_Temperature=1, # Indoor-outdoor minimum temperature difference
                Minimum_Outdoor_Temperature=18, # Outdoor minimum temperature
                Maximum_Outdoor_Temperature=28, # Outdoor maximum temperature
                Maximum_Wind_Speed=15 # Maximum wind speed
            )

        # Log the changes
        logging.info(f"Applied natural ventilation to {len(zone_names)} zones. {opening_area_m2} m2 of opening area for each zone.")


if __name__ == "__main__":
    async def main_test():
        """Runs a simple asynchronous test of the IDF service functions."""
        # Define the path to the test IDF file relative to this script's location
        test_idf_path = Path(__file__).parent.parent.parent / "data" / "test.idf"

        if not test_idf_path.exists():
            print(f"Error: Test IDF file not found at {test_idf_path}")
            return # Exit test if file not found

        print(f"Attempting to load test file: {test_idf_path}")
        try:
            with open(test_idf_path, "rb") as f:
                test_content = f.read()

            # Create a mock file object
            mock_file = MockUploadFile(filename="test.idf", content=test_content)

            # Test saving the file
            print("Testing save_idf_file...")
            idf_id = await save_idf_file(mock_file)
            print(f"  Successfully saved. IDF ID: {idf_id}")

            # Test retrieving the IDF object
            print("Testing get_idf_object...")
            idf_object = await get_idf_object(idf_id)
            print(f"  Successfully retrieved IDF object. Type: {type(idf_object)}")
            # Optionally print some info from the object if needed
            # print(f"  IDF Version: {idf_object.idfobjects.get('VERSION', [{}])[0].get('Version_Identifier', 'N/A')}")


            # Test retrieving the IDF path
            print("Testing get_idf_path...")
            idf_path = await get_idf_path(idf_id)
            print(f"  Successfully retrieved IDF path: {idf_path}")
            print(f"  Does path exist? {Path(idf_path).exists()}")

            print("All tests passed successfully!")

        except HTTPException as http_exc:
            print(f"HTTP Error during test: Status={http_exc.status_code}, Detail='{http_exc.detail}'")
        except FileNotFoundError as fnf_err:
            print(f"File Not Found Error during test: {fnf_err}")
        except Exception as e:
            # Catch any other unexpected errors during the test
            print(f"An unexpected error occurred during testing: {type(e).__name__}: {e}")
        finally:
            # Clean up: Close the mock file's stream if necessary
            if 'mock_file' in locals() and hasattr(mock_file, 'close'):
                 mock_file.close()
            # Optional: Clean up created temporary files if desired,
            # requires tracking created files or clearing the temp folder based on pattern.
            # For simplicity in this example, cleanup is omitted.
            pass

    import asyncio
    print("Running idf_service main test...")
    asyncio.run(main_test())
    print("idf_service main test finished.")

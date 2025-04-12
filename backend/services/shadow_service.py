# backend/services/shadow_service.py

import os
import numpy as np
import pandas as pd
import eppy.runner.run_functions as run_functions
from pathlib import Path
from eppy.modeleditor import IDF
from fastapi import HTTPException
from dotenv import load_dotenv

load_dotenv()

from .idf_service import get_idf_object, get_idf_path
from .geometry_service import extract_vertices

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR"))
os.environ["ENERGYPLUS_DIR"] = os.getenv("ENERGY_PLUS_DIR")

class ShadowAnalyzer:
    """Analyzes EnergyPlus shadow simulation results to find suitable surfaces for PV installation.

    Attributes:
        idf_id: The identifier for the IDF model.
        weather_file: Path to the EPW weather file.
        idf_path: Path to the IDF file.
        idf_obj: The eppy IDF object representation.
        output_dir: Directory to store simulation results.
    """
    def __init__(self, idf_id: str, weather_file: str):
        self.idf_id = idf_id
        self.weather_file = weather_file
        self.idf_path = None
        self.idf_obj = None
        self.output_dir = None

    async def initialize(self):
        """Initializes the analyzer by setting paths and creating the output directory."""
        self.idf_path = await get_idf_path(self.idf_id)
        self.idf_obj = await get_idf_object(self.idf_id)
        self.output_dir = OUTPUT_DIR / f"output_{self.idf_id}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def add_shadow_outputs(self):
        """Adds required Output:Variable objects to the IDF for shadow analysis.

        Ensures that variables related to incident solar radiation and sunlit
        fraction/area are requested for hourly reporting if not already present.
        """
        output_variables = [
            "Surface Outside Face Incident Solar Radiation Rate per Area",
            "Surface Outside Face Sunlit Fraction",
            "Surface Outside Face Sunlit Area"
        ]

        for var in output_variables:
            has_var = False
            for output_var in self.idf_obj.idfobjects["OUTPUT:VARIABLE"]:
                if output_var.Variable_Name.lower() == var.lower():
                    has_var = True
                    break

            if not has_var:
                self.idf_obj.newidfobject(
                    "OUTPUT:VARIABLE",
                    Key_Value="*",  # Apply to all surfaces
                    Variable_Name=var,
                    Reporting_Frequency="Hourly"
                )

        self.idf_obj.save()

    async def run_shadow_calculation(self):
        """Runs the EnergyPlus simulation using eppy.

        Executes the simulation with the specified IDF and weather file,
        storing results in the designated output directory.

        Returns:
            A dictionary containing the idf_id, output directory path, and status.

        Raises:
            HTTPException: If the EnergyPlus simulation fails.
        """
        try:
            idf = IDF(self.idf_path, self.weather_file)
            run_functions.run(
                idf,
                weather=self.weather_file,
                output_directory=str(self.output_dir),
                annual=True,
                readvars=True
            )

            return {
                "idf_id": self.idf_id,
                "output_dir": str(self.output_dir),
                "status": "shadow_calculation_success"
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error running shadow calculation: {str(e)}")
        
    async def analyze_results(self):
        """Analyzes shadow calculation results to find suitable PV surfaces.

        Reads the simulation output CSV, calculates radiation scores for
        exterior surfaces defined by CALCULATION_TYPE, and identifies surfaces
        exceeding the RADIATION_SCORE_THRESHOLD.

        Returns:
            A list of dictionaries, each representing a suitable surface, sorted
            by radiation score in descending order. Each dictionary contains:
            'name', 'radiation_score', 'area', 'annual_radiation', 'vertices'.

        Raises:
            HTTPException: If the results CSV is not found, no suitable surfaces
                           are found, or another analysis error occurs.
            FileNotFoundError: If the eplusout.csv file does not exist.
        """
        calculation_type = os.getenv("CALCULATION_TYPE")
        try:
            # Extract Exterior Wall Surfaces
            exterior_walls = []
            for surface in self.idf_obj.idfobjects["BUILDINGSURFACE:DETAILED"]:
                if (surface.Surface_Type.upper()) in calculation_type and (surface.Outside_Boundary_Condition.upper() == "OUTDOORS"):
                    vertices = extract_vertices(surface)
                    exterior_walls.append({
                        "name": surface.Name,
                        "vertices": vertices
                    })
            
            # Read CSV result file to get surface radiation data
            csv_path = self.output_dir / "eplusout.csv" # Assuming the CSV file name is eplusout.csv
            if not csv_path.exists():
                raise FileNotFoundError(f"CSV file not found: {csv_path}")
            df = pd.read_csv(csv_path)

            # Calculate the radiation and shadow score for each surface
            suitable_surfaces = []
            for wall in exterior_walls:
                wall_name = wall["name"]
                radiation_column_name = None
                annual_radiation = 0

                for col in df.columns:
                    if wall_name.upper() in col.upper() and "Surface Outside Face Incident Solar Radiation Rate per Area" in col:
                        radiation_column_name = col
                if radiation_column_name:
                    annual_radiation = df[radiation_column_name].sum()
                    radiation_score = self._calculate_radiation_score(annual_radiation)
                    area = self._calculate_area(wall["vertices"])
                    if radiation_score >= float(os.getenv("RADIATION_SCORE_THRESHOLD")):
                        suitable_surfaces.append({
                            "name": wall_name,
                            "radiation_score": radiation_score,
                            "area": area,
                            "annual_radiation": annual_radiation,
                            "vertices": wall["vertices"]
                        })
                else:
                    print(f"No radiation data found for {wall_name}")
                    radiation_score = 0
            
            if not suitable_surfaces:
                raise HTTPException(status_code=400, detail="No suitable surfaces found")
            
            # Sort suitable surfaces by radiation score in descending order
            suitable_surfaces.sort(key=lambda x: x["radiation_score"], reverse=True)
            
            return suitable_surfaces
            
        except Exception as e:
            # Re-raise FileNotFoundError specifically if needed, otherwise wrap
            if isinstance(e, FileNotFoundError):
                 raise HTTPException(status_code=500, detail=f"Shadow Analysis Error: {str(e)}")
            elif isinstance(e, HTTPException):
                 raise # Re-raise HTTP exceptions from inner logic
            else:
                 raise HTTPException(status_code=500, detail=f"Shadow Analysis Error: {str(e)}")

    def _calculate_radiation_score(self, annual_radiation: float) -> float:
        """Calculates a score based on annual incident solar radiation.

        Linearly interpolates the score between MIN_SCORE and MAX_SCORE based
        on whether the annual_radiation falls between RADIATION_THRESHOLD_LOW
        and RADIATION_THRESHOLD_HIGH.

        Args:
            annual_radiation: Total annual incident solar radiation on the surface (in J or Wh depending on E+ output).

        Returns:
            The calculated radiation score (between MIN_SCORE and MAX_SCORE).
        """
        radiation_threshold_high = float(os.getenv("RADIATION_THRESHOLD_HIGH"))
        radiation_threshold_low = float(os.getenv("RADIATION_THRESHOLD_LOW"))
        max_score = float(os.getenv("MAX_SCORE"))
        min_score = float(os.getenv("MIN_SCORE"))

        if annual_radiation > radiation_threshold_high:
            return max_score
        elif annual_radiation >= radiation_threshold_low:
            return min_score + (max_score - min_score) * (annual_radiation - radiation_threshold_low) / (radiation_threshold_high - radiation_threshold_low)
        else:
            return min_score
        
    def _calculate_area(self, points: list[list[float]]) -> float:
        """Calculates the area of a 3D polygon using the shoelace formula projection.

        Args:
            points: A list of 3D vertex coordinates [[x1, y1, z1], [x2, y2, z2], ...].

        Returns:
            The calculated area of the polygon in square meters (assuming coordinates are in meters).
        """
        points = np.array(points)
        if len(points) < 3:
            return 0.0
        
        # Calculate the area of the polygon
        sum_cross = np.sum(np.cross(points, np.roll(points, -1, axis=0)), axis=0)
        area = 0.5 * np.linalg.norm(sum_cross)
        
        return area
    
    async def run(self):
        """Executes the complete shadow analysis workflow.

        Initializes, adds necessary outputs, runs the simulation, and analyzes
        the results to return suitable surfaces.

        Returns:
            A list of suitable surfaces as determined by analyze_results.
        """
        await self.initialize()
        await self.add_shadow_outputs()
        await self.run_shadow_calculation()
        return await self.analyze_results()
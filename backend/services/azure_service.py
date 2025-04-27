"""Service module for handling Azure Docker container communication for EUI prediction."""

import os
import json
import logging
import requests
from pathlib import Path
from typing import Dict, Any, Optional
from .simulation_service import EnergyPlusRunner, SimulationResult
from .idf_service import IDFModel


class AzureDockerService:
    """
    Manages communication with Azure Docker containers for EnergyPlus simulations.
    """
    
    def __init__(self, config: dict):
        """
        Initialize the Azure Docker service.
        
        Args:
            config (dict): Configuration dictionary containing Azure settings
        """
        self.config = config
        self.azure_config = config.get('azure_docker', {})
        self.api_endpoint = self.azure_config.get('api_endpoint', '')
        self.api_key = self.azure_config.get('api_key', '')
        self.timeout = self.azure_config.get('timeout', 300)  # 5 minutes default
        
        if not self.api_endpoint:
            logging.warning("Azure Docker API endpoint not configured")
        
    def submit_simulation(self, idf_path: str, weather_path: str, btype: str) -> Dict[str, Any]:
        """
        Submit a simulation to the Azure Docker container.
        
        Args:
            idf_path (str): Path to the IDF file
            weather_path (str): Path to the weather file
            btype (str): Building type
            
        Returns:
            Dict[str, Any]: Simulation results
        """
        if not self.api_endpoint:
            raise ValueError("Azure Docker API endpoint not configured")
            
        with open(idf_path, 'r') as f:
            idf_content = f.read()
            
        with open(weather_path, 'r') as f:
            weather_content = f.read()
            
        payload = {
            'idf_content': idf_content,
            'weather_content': weather_content,
            'building_type': btype,
            'simulation_params': {
                'output_suffix': 'C',
                'cleanup': True
            }
        }
        
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }
        
        try:
            logging.info(f"Submitting simulation for {btype} to Azure Docker container")
            response = requests.post(
                f"{self.api_endpoint}/simulate",
                json=payload,
                headers=headers,
                timeout=self.timeout
            )
            
            response.raise_for_status()
            result = response.json()
            
            logging.info(f"Simulation completed successfully for {btype}")
            return result
            
        except requests.exceptions.RequestException as e:
            logging.error(f"Error submitting simulation to Azure: {e}")
            raise
            
    def parse_simulation_results(self, results: Dict[str, Any]) -> float:
        """
        Parse the simulation results to extract EUI.
        
        Args:
            results (Dict[str, Any]): Raw simulation results from Azure
            
        Returns:
            float: Energy Use Intensity (EUI) value
        """
        try:
            if 'source_eui' in results:
                return float(results['source_eui'])
            elif 'table_csv_content' in results:
                return self._extract_eui_from_csv(results['table_csv_content'])
            else:
                raise ValueError("Unable to find EUI in simulation results")
                
        except Exception as e:
            logging.error(f"Error parsing simulation results: {e}")
            raise
            
    def _extract_eui_from_csv(self, csv_content: str) -> float:
        """
        Extract EUI from CSV content.
        
        Args:
            csv_content (str): CSV content from simulation results
            
        Returns:
            float: EUI value
        """
        lines = csv_content.split('\n')
        for i, line in enumerate(lines):
            if "total source energy" in line.lower():
                for j in range(i + 1, len(lines)):
                    if lines[j].strip():
                        parts = lines[j].split(',')
                        if len(parts) > 2:
                            try:
                                return float(parts[2])
                            except ValueError:
                                continue
        raise ValueError("Could not extract EUI from CSV content")
        
    def simulate_building_type(self, city: str, ssp: int, btype: str, ecm_dict: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Simulate a specific building type using Azure Docker container.
        
        Args:
            city (str): City name
            ssp (int): SSP scenario
            btype (str): Building type
            ecm_dict (Dict[str, Any], optional): ECM parameters
            
        Returns:
            Dict[str, Any]: Simulation results including EUI
        """
        prototype_dir = self.config['paths']['prototypes_dir']
        weather_dir = self.config['paths']['tmy_dir'] if str(ssp).upper() == 'TMY' else self.config['paths']['ftmy_dir']
        
        idf_files = [f for f in prototype_dir.glob("*.idf") if btype.lower() in f.stem.lower()]
        if not idf_files:
            raise FileNotFoundError(f"No prototype IDF file found for {btype}")
        idf_path = idf_files[0]
        
        weather_files = [f for f in weather_dir.glob("*.epw") if city.lower() in f.stem.lower()]
        if not weather_files:
            raise FileNotFoundError(f"No weather file found for {city}")
        weather_path = weather_files[0]
        
        if ecm_dict:
            temp_idf_path = Path(f"/tmp/{btype}_{city}_{ssp}_ecm.idf")
            idf_model = IDFModel(str(idf_path))
            
            for param_name, value in ecm_dict.items():
                if param_name == 'insu' and value > 0:
                    idf_model.apply_wall_insulation(value)
                elif param_name == 'infl' and value > 0:
                    idf_model.apply_air_infiltration(value)
                elif param_name == 'cool_cop' and value > 0:
                    idf_model.apply_cooling_cop(value)
                
            idf_model.save(str(temp_idf_path))
            idf_path = temp_idf_path
            
        results = self.submit_simulation(str(idf_path), str(weather_path), btype)
        
        eui = self.parse_simulation_results(results)
        
        return {
            'city': city,
            'ssp': ssp,
            'building_type': btype,
            'eui': eui,
            'ecm_parameters': ecm_dict,
            'raw_results': results
        }

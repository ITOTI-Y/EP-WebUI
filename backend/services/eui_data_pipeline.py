"""Data pipeline for collecting and organizing EUI simulation results."""

import json
import logging
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
from .azure_service import AzureDockerService


class EUIDataPipeline:
    """
    Manages the collection and organization of EUI simulation data.
    """
    
    def __init__(self, config: dict):
        """
        Initialize the EUI data pipeline.
        
        Args:
            config (dict): Configuration dictionary
        """
        self.config = config
        self.azure_service = AzureDockerService(config)
        self.results_dir = config['paths']['results_dir'] / 'EUI_Data'
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def collect_data_for_building_types(self, city: str, ssp: int, building_types: List[str]) -> pd.DataFrame:
        """
        Collect simulation data for multiple building types.
        
        Args:
            city (str): City name
            ssp (int): SSP scenario
            building_types (List[str]): List of building types to simulate
            
        Returns:
            pd.DataFrame: DataFrame containing simulation results
        """
        results = []
        
        for btype in building_types:
            try:
                logging.info(f"Collecting data for {btype} in {city} under SSP {ssp}")
                result = self.azure_service.simulate_building_type(city, ssp, btype)
                results.append(result)
                
                self._save_result(result)
                
            except Exception as e:
                logging.error(f"Error collecting data for {btype}: {e}")
                continue
                
        df = pd.DataFrame(results)
        
        output_file = self.results_dir / f"{city}_{ssp}_all_buildings.csv"
        df.to_csv(output_file, index=False)
        logging.info(f"Saved aggregated results to {output_file}")
        
        return df
        
    def _save_result(self, result: Dict[str, Any]):
        """
        Save individual simulation result.
        
        Args:
            result (Dict[str, Any]): Simulation result
        """
        city = result['city']
        ssp = result['ssp']
        btype = result['building_type']
        
        output_file = self.results_dir / f"{city}_{ssp}_{btype}.json"
        
        result_to_save = result.copy()
        if 'raw_results' in result_to_save:
            result_to_save['raw_results'] = str(result_to_save['raw_results'])
            
        with open(output_file, 'w') as f:
            json.dump(result_to_save, f, indent=2)
            
        logging.info(f"Saved result for {btype} to {output_file}")
        
    def prepare_training_data(self, cities: List[str], ssps: List[int], building_types: List[str]) -> pd.DataFrame:
        """
        Prepare training data for the GNN model.
        
        Args:
            cities (List[str]): List of cities
            ssps (List[int]): List of SSP scenarios
            building_types (List[str]): List of building types
            
        Returns:
            pd.DataFrame: Training data
        """
        all_data = []
        
        for city in cities:
            for ssp in ssps:
                df = self.collect_data_for_building_types(city, ssp, building_types)
                all_data.append(df)
                
        combined_df = pd.concat(all_data, ignore_index=True)
        
        output_file = self.results_dir / "training_data.csv"
        combined_df.to_csv(output_file, index=False)
        logging.info(f"Saved training data to {output_file}")
        
        return combined_df
        
    def load_training_data(self) -> pd.DataFrame:
        """
        Load existing training data.
        
        Returns:
            pd.DataFrame: Training data
        """
        training_file = self.results_dir / "training_data.csv"
        
        if not training_file.exists():
            raise FileNotFoundError(f"Training data not found at {training_file}")
            
        return pd.read_csv(training_file)

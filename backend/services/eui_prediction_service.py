"""Service for EUI prediction using Graph Neural Networks."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import HeteroData
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
from .gnn_model import AdaptiveGNN, BuildingGraphConstructor
from .eui_data_pipeline import EUIDataPipeline


class EUIPredictionService:
    """
    Service for training and using GNN models for EUI prediction.
    """
    
    def __init__(self, config: dict):
        """
        Initialize the EUI prediction service.
        
        Args:
            config (dict): Configuration dictionary
        """
        self.config = config
        self.model_dir = config['paths']['results_dir'] / 'EUI_Models'
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.building_type_mapping = {
            'OfficeLarge': 0,
            'OfficeMedium': 1,
            'ApartmentHighRise': 2,
            'SingleFamilyResidential': 3,
            'MultiFamilyResidential': 4,
            'RetailStandalone': 5,
            'RetailStripmall': 6,
            'SchoolPrimary': 7,
            'SchoolSecondary': 8,
            'Hospital': 9
        }
        
        self.node_features_dict = {
            'zone': 5,  # floor_area, volume, occupancy, lighting_power, equipment_power
            'surface': 4,  # area, u_value, solar_absorptance, visible_absorptance
            'equipment': 3  # power, efficiency, schedule_fraction
        }
        
        self.graph_constructor = BuildingGraphConstructor(self.building_type_mapping)
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def train_model(self, training_data: pd.DataFrame, epochs: int = 100, batch_size: int = 32):
        """
        Train the GNN model on the provided data.
        
        Args:
            training_data (pd.DataFrame): Training data
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
        """
        logging.info("Starting GNN model training")
        
        self.model = AdaptiveGNN(self.node_features_dict).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        graphs, building_types, targets = self._prepare_training_data(training_data)
        
        for epoch in range(epochs):
            total_loss = 0
            self.model.train()
            
            for i in range(0, len(graphs), batch_size):
                batch_graphs = graphs[i:i+batch_size]
                batch_types = building_types[i:i+batch_size]
                batch_targets = targets[i:i+batch_size]
                
                optimizer.zero_grad()
                
                predictions = []
                for graph, btype in zip(batch_graphs, batch_types):
                    graph = graph.to(self.device)
                    pred = self.model(graph, btype)
                    predictions.append(pred)
                    
                predictions = torch.cat(predictions)
                batch_targets = torch.tensor(batch_targets, dtype=torch.float32).to(self.device)
                
                loss = criterion(predictions.squeeze(), batch_targets)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
            avg_loss = total_loss / (len(graphs) / batch_size)
            logging.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
            
        model_path = self.model_dir / "eui_gnn_model.pth"
        torch.save(self.model.state_dict(), model_path)
        logging.info(f"Model saved to {model_path}")
        
    def _prepare_training_data(self, training_data: pd.DataFrame) -> Tuple[List[HeteroData], List[int], List[float]]:
        """
        Prepare training data for the GNN model.
        
        Args:
            training_data (pd.DataFrame): Raw training data
            
        Returns:
            Tuple[List[HeteroData], List[int], List[float]]: Graphs, building types, and target EUI values
        """
        graphs = []
        building_types = []
        targets = []
        
        for _, row in training_data.iterrows():
            building_data = {
                'building_type': row['building_type'],
                'zones': self._extract_zones(row),
                'surfaces': self._extract_surfaces(row),
                'equipment': self._extract_equipment(row),
                'zone_connections': self._extract_zone_connections(row),
                'surface_zone_map': self._extract_surface_zone_map(row),
                'equipment_zone_map': self._extract_equipment_zone_map(row)
            }
            
            graph, btype_idx = self.graph_constructor.construct_graph(building_data)
            
            graphs.append(graph)
            building_types.append(btype_idx)
            targets.append(row['eui'])
            
        return graphs, building_types, targets
        
    def _extract_zones(self, row: pd.Series) -> List[Dict]:
        """Extract zone information from row data."""
        return []
        
    def _extract_surfaces(self, row: pd.Series) -> List[Dict]:
        """Extract surface information from row data."""
        return []
        
    def _extract_equipment(self, row: pd.Series) -> List[Dict]:
        """Extract equipment information from row data."""
        return []
        
    def _extract_zone_connections(self, row: pd.Series) -> List[List[int]]:
        """Extract zone connection information from row data."""
        return []
        
    def _extract_surface_zone_map(self, row: pd.Series) -> List[List[int]]:
        """Extract surface-zone mapping from row data."""
        return []
        
    def _extract_equipment_zone_map(self, row: pd.Series) -> List[List[int]]:
        """Extract equipment-zone mapping from row data."""
        return []
        
    def predict_eui(self, building_data: Dict) -> float:
        """
        Predict EUI for a given building.
        
        Args:
            building_data (Dict): Building data
            
        Returns:
            float: Predicted EUI value
        """
        if self.model is None:
            self.load_model()
            
        graph, btype_idx = self.graph_constructor.construct_graph(building_data)
        graph = graph.to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(graph, btype_idx)
            
        return prediction.item()
        
    def load_model(self):
        """Load a trained model from disk."""
        model_path = self.model_dir / "eui_gnn_model.pth"
        
        if not model_path.exists():
            raise FileNotFoundError(f"No trained model found at {model_path}")
            
        self.model = AdaptiveGNN(self.node_features_dict).to(self.device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        logging.info(f"Model loaded from {model_path}")
        
    def evaluate_model(self, test_data: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            test_data (pd.DataFrame): Test data
            
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        if self.model is None:
            self.load_model()
            
        graphs, building_types, targets = self._prepare_training_data(test_data)
        
        predictions = []
        self.model.eval()
        
        with torch.no_grad():
            for graph, btype in zip(graphs, building_types):
                graph = graph.to(self.device)
                pred = self.model(graph, btype)
                predictions.append(pred.item())
                
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        mse = np.mean((predictions - targets) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - targets))
        mape = np.mean(np.abs((targets - predictions) / targets)) * 100
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape
        }
        
        logging.info(f"Model evaluation metrics: {metrics}")
        return metrics

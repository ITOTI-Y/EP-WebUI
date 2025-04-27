"""Graph Neural Network model for EUI prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GCNConv, global_mean_pool
from torch_geometric.data import HeteroData
import numpy as np
from typing import Dict, List, Tuple


class AdaptiveGNN(nn.Module):
    """
    Adaptive Graph Neural Network for EUI prediction across different building types.
    """
    
    def __init__(self, node_features_dict: Dict[str, int], hidden_channels: int = 64):
        """
        Initialize the adaptive GNN.
        
        Args:
            node_features_dict (Dict[str, int]): Dictionary mapping node types to feature dimensions
            hidden_channels (int): Number of hidden channels
        """
        super(AdaptiveGNN, self).__init__()
        
        self.node_features_dict = node_features_dict
        self.hidden_channels = hidden_channels
        
        self.node_embeddings = nn.ModuleDict()
        for node_type, feature_dim in node_features_dict.items():
            self.node_embeddings[node_type] = nn.Linear(feature_dim, hidden_channels)
            
        self.conv1 = HeteroConv({
            ('zone', 'connects', 'zone'): GCNConv(-1, hidden_channels),
            ('zone', 'contains', 'equipment'): GCNConv(-1, hidden_channels),
            ('equipment', 'belongs_to', 'zone'): GCNConv(-1, hidden_channels),
            ('surface', 'bounds', 'zone'): GCNConv(-1, hidden_channels),
            ('zone', 'bounded_by', 'surface'): GCNConv(-1, hidden_channels),
        }, aggr='mean')
        
        self.conv2 = HeteroConv({
            ('zone', 'connects', 'zone'): GCNConv(hidden_channels, hidden_channels),
            ('zone', 'contains', 'equipment'): GCNConv(hidden_channels, hidden_channels),
            ('equipment', 'belongs_to', 'zone'): GCNConv(hidden_channels, hidden_channels),
            ('surface', 'bounds', 'zone'): GCNConv(hidden_channels, hidden_channels),
            ('zone', 'bounded_by', 'surface'): GCNConv(hidden_channels, hidden_channels),
        }, aggr='mean')
        
        self.building_type_embeddings = nn.Embedding(10, hidden_channels)  # Assuming max 10 building types
        
        self.fc1 = nn.Linear(hidden_channels * 2, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, 1)
        
    def forward(self, data: HeteroData, building_type_idx: int) -> torch.Tensor:
        """
        Forward pass of the GNN.
        
        Args:
            data (HeteroData): Heterogeneous graph data
            building_type_idx (int): Index of the building type
            
        Returns:
            torch.Tensor: Predicted EUI value
        """
        x_dict = {}
        
        for node_type in data.node_types:
            if node_type in self.node_embeddings:
                x_dict[node_type] = self.node_embeddings[node_type](data[node_type].x)
            else:
                x_dict[node_type] = torch.zeros((data[node_type].num_nodes, self.hidden_channels))
                
        x_dict = self.conv1(x_dict, data.edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        x_dict = self.conv2(x_dict, data.edge_index_dict)
        
        pooled_features = []
        for node_type in data.node_types:
            if node_type in x_dict:
                pooled = global_mean_pool(x_dict[node_type], 
                                        torch.zeros(data[node_type].num_nodes, dtype=torch.long))
                pooled_features.append(pooled)
                
        graph_embedding = torch.cat(pooled_features, dim=1)
        
        building_type_emb = self.building_type_embeddings(torch.tensor([building_type_idx]))
        combined_embedding = torch.cat([graph_embedding, building_type_emb], dim=1)
        
        x = F.relu(self.fc1(combined_embedding))
        eui = self.fc2(x)
        
        return eui


class BuildingGraphConstructor:
    """
    Constructs heterogeneous graphs from building data.
    """
    
    def __init__(self, building_type_mapping: Dict[str, int]):
        """
        Initialize the graph constructor.
        
        Args:
            building_type_mapping (Dict[str, int]): Mapping from building type names to indices
        """
        self.building_type_mapping = building_type_mapping
        
    def construct_graph(self, building_data: Dict) -> Tuple[HeteroData, int]:
        """
        Construct a heterogeneous graph from building data.
        
        Args:
            building_data (Dict): Building data including zones, surfaces, equipment, etc.
            
        Returns:
            Tuple[HeteroData, int]: Graph data and building type index
        """
        data = HeteroData()
        
        building_type = building_data.get('building_type', 'unknown')
        building_type_idx = self.building_type_mapping.get(building_type, 0)
        
        zones = building_data.get('zones', [])
        zone_features = []
        for zone in zones:
            features = [
                zone.get('floor_area', 0),
                zone.get('volume', 0),
                zone.get('occupancy', 0),
                zone.get('lighting_power', 0),
                zone.get('equipment_power', 0)
            ]
            zone_features.append(features)
            
        if zone_features:
            data['zone'].x = torch.tensor(zone_features, dtype=torch.float)
            
        surfaces = building_data.get('surfaces', [])
        surface_features = []
        for surface in surfaces:
            features = [
                surface.get('area', 0),
                surface.get('u_value', 0),
                surface.get('solar_absorptance', 0),
                surface.get('visible_absorptance', 0)
            ]
            surface_features.append(features)
            
        if surface_features:
            data['surface'].x = torch.tensor(surface_features, dtype=torch.float)
            
        equipment = building_data.get('equipment', [])
        equipment_features = []
        for equip in equipment:
            features = [
                equip.get('power', 0),
                equip.get('efficiency', 0),
                equip.get('schedule_fraction', 0)
            ]
            equipment_features.append(features)
            
        if equipment_features:
            data['equipment'].x = torch.tensor(equipment_features, dtype=torch.float)
            
        zone_connections = building_data.get('zone_connections', [])
        if zone_connections:
            edge_index = torch.tensor(zone_connections, dtype=torch.long).t().contiguous()
            data['zone', 'connects', 'zone'].edge_index = edge_index
            
        surface_zone_map = building_data.get('surface_zone_map', [])
        if surface_zone_map:
            edge_index = torch.tensor(surface_zone_map, dtype=torch.long).t().contiguous()
            data['surface', 'bounds', 'zone'].edge_index = edge_index
            data['zone', 'bounded_by', 'surface'].edge_index = edge_index[[1, 0]]
            
        equipment_zone_map = building_data.get('equipment_zone_map', [])
        if equipment_zone_map:
            edge_index = torch.tensor(equipment_zone_map, dtype=torch.long).t().contiguous()
            data['equipment', 'belongs_to', 'zone'].edge_index = edge_index
            data['zone', 'contains', 'equipment'].edge_index = edge_index[[1, 0]]
            
        return data, building_type_idx

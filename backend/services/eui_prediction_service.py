"""Service for EUI prediction using Graph Neural Networks."""

import torch
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
from backend.services.eui_dataset_service import EUIDataset
from backend.services.model_service import ModelService

class EUIPredictionService:
    """
    Service for training and using Neural Network models for EUI prediction.
    """
    
    def __init__(self, config: dict):
        """
        Initialize the EUI prediction service.
        
        Args:
            config (dict): Configuration dictionary
        """
        self.config = config
        self.model_dir = config['paths']['eui_models_dir']
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.device = config['eui_prediction']['device']

    def train_model(self, data: pd.DataFrame, model_service: ModelService):
        dataset = EUIDataset(data, self.config)
        train_loader = dataset.get_train_data()
        val_loader = dataset.get_val_data()
        test_loader = dataset.get_test_data()

        model = model_service.get_model(len(self.config['eui_prediction']['feature_columns']))
        model = model.to(self.device)
        loss_fn = model_service.get_loss_fn()
        optimizer = model_service.get_optimizer(model)

        for epoch in range(self.config['eui_prediction']['num_epochs']):
            model.train()
            for batch_features, batch_labels in train_loader:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(batch_features)
                loss = loss_fn(outputs, batch_labels)
                loss.backward()
                optimizer.step()
            logging.info(f"Epoch {epoch+1} loss: {loss.item()}")

        pass

    def _load_artifacts(self):
        """Loads the trained model, scaler, and scaled columns list."""
        scaler_path = self.model_dir / 'scaler.joblib'
        scaled_columns_path = self.model_dir / 'scaled_columns.json'
        mapping_dict_path = self.model_dir / 'label_encoding_maps.json'

    def _build_dataset(self):
        pass
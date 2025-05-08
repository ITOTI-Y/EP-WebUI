import torch
import logging
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import json
import pandas as pd
import numpy as np
import joblib


class EUIDataset(Dataset):
    """
    PyTorch Dataset for EUI prediction data.

    Handles label encoding, stratified splitting (using indices),
    and provides DataLoaders for train, validation, and test sets.
    """

    def __init__(self, training_data: pd.DataFrame, config: dict):
        """
        Initializes the dataset.

        Args:
            training_data (pd.DataFrame): The raw training data.
            config (dict): Configuration dictionary containing parameters like
                           feature columns, target column, split ratios, etc.
        """
        self.config = config
        self.df_data = training_data.copy()
        self.feature_columns = config['eui_prediction']['feature_columns']
        self.target_column = config['eui_prediction']['target_column']
        self.stratify_columns = config['eui_prediction']['group_by_columns']
        self.mapping_dict = {}
        self._label_encode()
        self._stratified_split()
        self._prepare_data()
        logging.info(
            f"Initializing EUIDataset with {len(self.all_features)} samples and {len(self.all_labels)} labels.")

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.all_features)

    def __getitem__(self, idx):
        """
        Retrieves the feature and label tensors for a given index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple: (feature_tensor, label_tensor)
        """
        return self.all_features[idx], self.all_labels[idx]

    def _label_encode(self):
        label_encoder = LabelEncoder()
        string_columns = self.df_data.select_dtypes(
            include=['object', 'category']).columns
        for column in string_columns:
            self.df_data[column] = label_encoder.fit_transform(
                self.df_data[column])
            self.mapping_dict[column] = dict(
                zip([i if isinstance(i, (str)) else i.item() for i in label_encoder.classes_], 
                    [i if isinstance(i, (str)) else i.item() for i in label_encoder.transform(label_encoder.classes_)]))
        logging.info(f"Label encoded columns: {list(string_columns)}")
        model_dir = Path(self.config['paths']['results_dir']) / 'EUI_Models' # Keep consistent with EUIPredictionService
        model_dir.mkdir(parents=True, exist_ok=True) # Ensure directory exists
        mapping_path = model_dir / 'label_encoding_maps.json'
        try:
            with open(mapping_path, 'w') as f:
                json.dump(self.mapping_dict, f, indent=4) # indent Parameter (Optional), make JSON file more readable
            logging.info(f"Label encoding mappings saved to {mapping_path}")
        except Exception as e:
            logging.error(f"Error saving label encoding mappings: {e}")

    def _stratified_split(self):
        split_ratios = self.config['eui_prediction']['train_val_test_split']
        random_state = self.config['eui_prediction']['random_state']
        assert abs(sum(split_ratios) - 1.0 ) < 1e-6, "The sum of train_val_test_split must be 1."
        str_df = self.df_data[self.stratify_columns]
        str_df['stratify_key'] = ""
        for i, column in enumerate(self.stratify_columns):
            str_df['stratify_key'] += self.df_data[column].astype(str)
            if i < len(self.stratify_columns) - 1:
                str_df['stratify_key'] += "_"

        indices = np.arange(len(self.df_data))

        train_indices, temp_indices = train_test_split(indices, test_size=split_ratios[1] + split_ratios[2],
                                             stratify=str_df['stratify_key'], random_state=random_state)
        
        temp_stratify = str_df.loc[temp_indices]['stratify_key']

        test_indices, val_indices = train_test_split(temp_indices, test_size=split_ratios[2]/(split_ratios[1] + split_ratios[2]),
                                           stratify=temp_stratify,
                                           random_state=random_state)

        self.train_indices = train_indices
        self.val_indices = val_indices
        self.test_indices = test_indices

    def _prepare_data(self):
        target_column = self.config['eui_prediction']['target_column']
        feature_columns = self.config['eui_prediction']['feature_columns']

        self.scaler = None
        self.scaler_columns = []

        if self.config['eui_prediction'].get('scale_features', False):
            from sklearn.preprocessing import StandardScaler
            numeric_cols_to_scale = self.df_data[feature_columns].select_dtypes(include=[np.number]).columns
            self.scaled_columns = list(numeric_cols_to_scale)

            if len(self.scaled_columns) > 0:
                scaler = StandardScaler()
                # Only fit on training data
                scaler.fit(self.df_data.loc[self.train_indices][self.scaled_columns])

                self.df_data[self.scaled_columns] = scaler.transform(self.df_data[self.scaled_columns])
                logging.info(f"Applied StandardScaler to columns: {self.scaled_columns}")
                self.scaler = scaler

                model_dir = self.config['paths']['eui_models_dir']
                model_dir.mkdir(parents=True, exist_ok=True)
                scaler_path = model_dir / 'scaler.joblib'

                try:
                    joblib.dump(self.scaler, scaler_path)
                    logging.info(f"Scaler saved to {scaler_path}")
                    scaled_columns_path = model_dir / 'scaled_columns.json'
                    with open(scaled_columns_path, 'w') as f:
                        json.dump(self.scaled_columns, f)
                    logging.info(f"Scaled column names saved to {scaled_columns_path}")
                except Exception as e:
                    logging.error(f"Error saving scaler or column names: {e}")

        features_np = self.df_data[feature_columns].values.astype(np.float32)
        labels_np = self.df_data[target_column].values.astype(np.float32)
        self.all_features = torch.from_numpy(features_np)
        self.all_labels = torch.from_numpy(labels_np).unsqueeze(1)

    def get_dataloader(self, indices: np.ndarray, batch_size: int = None, shuffle: bool = True, num_workers: int = 0) -> DataLoader:
        """Helper function to create a DataLoader for a given set of indices."""
        if batch_size is None:
            batch_size = self.config['eui_prediction']['batch_size']
        subset = Subset(self, indices)
        dataloader = DataLoader(subset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        return dataloader
    

    def get_train_data(self, batch_size: int = None, shuffle: bool = True, num_workers: int = 0):
        """Returns a DataLoader for the training set."""
        return self.get_dataloader(self.train_indices, batch_size, shuffle=shuffle, num_workers=num_workers)

    def get_val_data(self, batch_size: int = None, shuffle: bool = False, num_workers: int = 0):
        """Returns a DataLoader for the validation set."""
        return self.get_dataloader(self.val_indices, batch_size, shuffle=shuffle, num_workers=num_workers)

    def get_test_data(self, batch_size: int = None, shuffle: bool = False, num_workers: int = 0):
        """Returns a DataLoader for the test set."""
        return self.get_dataloader(self.test_indices, batch_size, shuffle=shuffle, num_workers=num_workers)

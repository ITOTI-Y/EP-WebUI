import torch
from torch import nn

class ModelService:
    def __init__(self, config: dict):
        self.config = config
        self.model_dir = config['paths']['eui_models_dir']

    def get_optimizer(self, model: torch.nn.Module):
        return torch.optim.Adam(model.parameters(), lr=self.config['eui_prediction']['learning_rate'])

    def get_loss_fn(self):
        return nn.MSELoss()

    def get_model(self, input_dim: int):
        return RegressionNet(input_dim=input_dim)

class RegressionNet(torch.nn.Module):
    def __init__(self, input_dim: int):
        super(RegressionNet, self).__init__()
        self.layer_1 = nn.Linear(input_dim, 128)
        self.relu_1 = nn.ReLU()
        self.dropout_1 = nn.Dropout(0.2)
        self.layer_2 = nn.Linear(128, 64)
        self.relu_2 = nn.ReLU()
        self.dropout_2 = nn.Dropout(0.2)
        self.layer_3 = nn.Linear(64, 32)
        self.relu_3 = nn.ReLU()
        self.output_layer = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_1(x)
        x = self.relu_1(x)
        x = self.dropout_1(x)
        x = self.layer_2(x)
        x = self.relu_2(x)
        x = self.dropout_2(x)
        x = self.layer_3(x)
        x = self.relu_3(x)
        x = self.output_layer(x)
        return x


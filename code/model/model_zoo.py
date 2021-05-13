import timm
import torch.nn as nn

class BirdClefModel(nn.Module):
    def __init__(self, model_name, n_class, drop_rate=0.0, pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_name, drop_rate=drop_rate, pretrained=pretrained)
        if hasattr(self.model, "fc"):
            n_features = self.model.fc.in_features
            self.model.fc = nn.Linear(n_features, n_class)
        elif hasattr(self.model, "classifier"):
            n_features = self.model.classifier.in_features
            self.model.classifier = nn.Linear(n_features, n_class)
        elif hasattr(self.model, "head"):
            n_features = self.model.head.in_features
            self.model.head = nn.Linear(n_features, n_class)

    def forward(self, x):
        x = self.model(x)
        return x

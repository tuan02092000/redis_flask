from lib import *
import config

class Efficientnet_v2(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = self.get_model()    
    def forward(self, x):
        return self.model(x)
    def get_model(self):
        model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(in_features=1280, out_features=config.N_GENDER, bias=True)
        return model

class Efficientnet_v2m(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = self.get_model()    
    def forward(self, x):
        return self.model(x)
    def get_model(self):
        model = models.efficientnet_v2_m(weights=models.EfficientNet_V2_M_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(in_features=1280, out_features=config.N_GENDER, bias=True)
        return model

class ViTB16(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = self.get_model()
    def forward(self, x):
        return x
    def get_model(self):
        # model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1)
        model = models.vit_b_16()
        model.heads.head = nn.Linear(in_features=768, out_features=config.N_GENDER, bias=True)
        return model

if __name__ == '__main__':
    model = Efficientnet_v2m()
    print(model)
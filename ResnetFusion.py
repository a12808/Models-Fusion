
import torch
from torchvision.models import resnet50, ResNet50_Weights


class ResnetFusionSaver:
    def __init__(self, model1_path, model2_path, output_path, device='cpu'):
        self.device = device

        #
        self.model1 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.model2 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        torch.save(self.model1, model1_path)
        torch.save(self.model2, model2_path)

        self.model1 = torch.load(model1_path)
        self.model2 = torch.load(model2_path)

        fusion_dict = {
            'model1': self.model1,
            'model2': self.model2
        }
        torch.save(fusion_dict, output_path)
        print(f"Resnet Fusion model saved at: {output_path}")


class ResnetFusionLoader:
    def __init__(self, fusion_model_path, device='cpu'):
        self.device = device
        fusion_dict = torch.load(fusion_model_path, map_location=device)
        self.model1 = fusion_dict['model1']
        self.model2 = fusion_dict['model2']

        self.model1.eval()
        self.model2.eval()

        self.class_labels = ResNet50_Weights.IMAGENET1K_V1.meta["categories"]

    def get_class_labels(self):
        return self.class_labels

    @torch.no_grad()
    def infer(self, img1, img2):
        """
        img: torch.Tensor (B, C, H, W) - mesma imagem para ambos os modelos
        """
        output1 = self.model1(img1)
        output2 = self.model2(img2)
        return output1, output2

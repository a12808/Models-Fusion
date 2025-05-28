
import torch


class YOLOFusionSaver:
    def __init__(self, model1_path, model2_path, output_path, device='cpu'):
        self.device = device
        # self.model1 = torch.load(model1_path, map_location=device)
        # self.model2 = torch.load(model2_path, map_location=device)
        self.model1 = torch.hub.load('ultralytics/yolov5', 'custom', path=model1_path)
        self.model2 = torch.hub.load('ultralytics/yolov5', 'custom', path=model2_path)

        fusion_dict = {
            'model1': self.model1,
            'model2': self.model2
        }
        torch.save(fusion_dict, output_path)
        print(f"YOLO Fusion model saved at: {output_path}")


class YOLOFusionLoader:
    def __init__(self, fusion_model_path, device='cpu'):
        self.device = device
        fusion_dict = torch.load(fusion_model_path, map_location=device)
        self.model1 = fusion_dict['model1']
        self.model2 = fusion_dict['model2']

        self.model1.eval()
        self.model2.eval()

    @torch.no_grad()
    def infer(self, img1, img2):
        """
        img: torch.Tensor (B, C, H, W) - mesma imagem para ambos os modelos
        """
        output1 = self.model1(img1)
        output2 = self.model2(img2)
        return output1, output2

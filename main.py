
from YOLOFusion import YOLOFusionSaver, YOLOFusionLoader
from ResnetFusion import ResnetFusionSaver, ResnetFusionLoader

import cv2
import torchvision.transforms as T


yolo_model1_path = "models/yolov5n_1.pt"
yolo_model2_path = "models/yolov5n_2.pt"
yolo_fused_model_path = "models/yolo_fused_model.pt"

resnet_model1_path = "models/resnet50_1.pt"
resnet_model2_path = "models/resnet50_2.pt"
resnet_fused_model_path = "models/resnet_fused_model.pt"

img1_path = "img1.png"
img2_path = "img2.png"
bike_path = "bike.png"
dog_path = "dog.png"


# --- Preparar imagem
img1 = cv2.imread(bike_path)
img2 = cv2.imread(dog_path)

if img1 is None:
    raise FileNotFoundError(f"Imagem não encontrada em: {img1_path}")

if img2 is None:
    raise FileNotFoundError(f"Imagem não encontrada em: {img2_path}")


# --- Fundir os modelos
# YOLOFusionSaver(yolo_model1_path, yolo_model2_path, yolo_fused_model_path)
ResnetFusionSaver(resnet_model1_path, resnet_model2_path, resnet_fused_model_path)


# Converter BGR (OpenCV) para RGB
img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

# Transformação padrão YOLOv5 (640x640, normalização 0-1)
transform = T.Compose([
    T.ToPILImage(),               #
    T.Resize((640, 640)),         # Resize para YOLOv5
    T.ToTensor(),                 # Converte para (C, H, W) e normaliza para [0,1]
])

img1_tensor = transform(img1_rgb).unsqueeze(0)  # (1, 3, 640, 640)
img2_tensor = transform(img2_rgb).unsqueeze(0)  # (1, 3, 640, 640)


# --- Carregar modelo e fazer inferência
# yolo_fused_model = YOLOFusionLoader(yolo_fused_model_path, device='cpu')
# yolo_output1, yolo_output2 = yolo_fused_model.infer(img1_tensor, img2_tensor)

resnet_fused_model = ResnetFusionLoader(resnet_fused_model_path, device='cpu')
resnet_labels = resnet_fused_model.get_class_labels()
resnet_output1, resnet_output2 = resnet_fused_model.infer(img1_tensor, img2_tensor)

pred1 = resnet_output1.argmax(1).item()
print(f"model 2 prev: {resnet_labels[pred1]}")
pred2 = resnet_output2.argmax(1).item()
print(f"model 2 prev: {resnet_labels[pred2]}")


# YOLO
''' 
# --- Função para desenhar deteções com cores
def draw_detections(img_bgr, detections, label_prefix="Model", color=(0, 255, 0)):
    if detections is None:
        return
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        label = f"{label_prefix} {int(cls)} {conf:.2f}"
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img_bgr, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 1, cv2.LINE_AA)


# --- Definir cores para cada modelo (RGB)
color_model1 = (0, 255, 0)  # Verde
color_model2 = (0, 0, 255)  # Vermelho


img1_display = cv2.resize(img1, (640, 640))  # OpenCV usa BGR
img2_display = cv2.resize(img2, (640, 640))  # OpenCV usa BGR

from yolov5.models.common import non_max_suppression

# outputs brutos: (1, N, 85) -> aplicar NMS
output1_nms = non_max_suppression(resnet_output1, conf_thres=0.25, iou_thres=0.45)[0]
output2_nms = non_max_suppression(resnet_output2, conf_thres=0.25, iou_thres=0.45)[0]

draw_detections(img1_display, output1_nms, label_prefix="M1", color=color_model1)  # Modelo 1 em verde
draw_detections(img2_display, output2_nms, label_prefix="M2", color=color_model2)  # Modelo 2 em vermelho

cv2.imshow("img1", img1_display)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow("img2", img2_display)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''


import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
import threading
import time
from ResnetFusion import ResnetFusionLoader


# ---- Configurações ----
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
FUSION_MODEL_PATH = 'models/resnet_fused_model.pt'
BATCH_SIZE = 1  # Imagem única para simular real-time
NUM_WORKERS = 2
NUM_IMAGENS = 50  #

# ---- Transforms para ResNet50 ----
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ---- Dataset (podes mudar para ImageNet se tiveres acesso local) ----
dataset = CIFAR100(root='./data', train=False, transform=transform, download=True)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# ---- Carregar modelo composto ----
fusion = ResnetFusionLoader(FUSION_MODEL_PATH, device=DEVICE)


# ---- Inferência com paralelismo por thread ----
def threaded_infer(model, img, results, index):
    with torch.no_grad():
        img = img.to(DEVICE)
        results[index] = model(img)


def test_infer(loader, fusion):
    tempos = []
    classes = fusion.get_class_labels()

    for i, (img, _) in enumerate(loader):
        if i >= NUM_IMAGENS:
            break

        img1 = img.clone().to(DEVICE)
        img2 = img.clone().to(DEVICE)

        results = [None, None]

        t0 = time.perf_counter()

        # Criar threads para ambos os modelos
        t1 = threading.Thread(target=threaded_infer, args=(fusion.model1, img1, results, 0))
        t2 = threading.Thread(target=threaded_infer, args=(fusion.model2, img2, results, 1))

        t1.start()
        t2.start()

        t1.join()
        t2.join()

        t1_output = results[0]
        t2_output = results[1]

        t1_label = classes[t1_output.argmax()]
        t2_label = classes[t2_output.argmax()]

        t1_prob = torch.softmax(t1_output, dim=1)[0][t1_output.argmax()].item()
        t2_prob = torch.softmax(t2_output, dim=1)[0][t2_output.argmax()].item()

        t1_elapsed = time.perf_counter() - t0
        tempos.append(t1_elapsed)

        print(f"[{i+1}] {t1_label} | T1: {t1_prob:.2f} | T2: {t2_prob:.2f} | t: {t1_elapsed*1000:.2f} ms")

    tempo_medio = sum(tempos) / len(tempos)
    fps = 1.0 / tempo_medio
    print(f"\n🧠 Tempo médio por imagem: {tempo_medio:.4f} s")
    print(f"🚀 FPS: {fps:.2f}")


# ---- Run ----
if __name__ == '__main__':
    print(f"Device: {DEVICE}")
    test_infer(loader, fusion)

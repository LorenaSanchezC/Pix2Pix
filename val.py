import torch
from dataset import MapDataset
from generator_model import Generator
from discriminator_model import Discriminator
from torch.utils.data import DataLoader
import config

def normalize_image(image):
    # Normaliza una imagen de [-1, 1] a [0, 1] 
    # Los datos estan [-1,1] porque eran los resultados de una tanh
    return (image + 1) / 2

def calculate_iou(pred, target):
    #Calcula el Intersection over Union (IoU) para dos tensores de imágenes binarias
    intersection = (pred * target).sum()
    union = (pred + target).sum() - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.item()

def calculate_iou_bg(pred, target, threshold=0.9):
    # Calcula el Intersection over Union (IoU) para el fondo usando un umbral específico
    # Normalizar y binarizar predicciones y verdades del terreno para el fondo
    pred = normalize_image(pred)
    target = normalize_image(target)
    pred_bg = (pred >= threshold).float()
    target_bg = (target >= threshold).float()
    return calculate_iou(pred_bg, target_bg)

def calculate_iou_lines(pred, target, threshold=0.1):
    # Calcula el Intersection over Union (IoU) para las líneas usando un umbral específico
    # Normalizar y binarizar predicciones y verdades del terreno para las líneas
    pred = normalize_image(pred)
    target = normalize_image(target)
    pred_lines = (pred <= threshold).float()
    target_lines = (target <= threshold).float()
    return calculate_iou(pred_lines, target_lines)


def validate_model(gen, disc, val_loader):
    gen.eval()
    disc.eval()
    total_gen_loss = 0
    total_disc_loss = 0
    total_iou_bg = 0
    total_iou_lines = 0
    num_batches = len(val_loader)

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(config.DEVICE)
            y = y.to(config.DEVICE)

            # Generar imágenes falsas
            y_fake = gen(x)

            # Calcular pérdida del generador
            fake_pred = disc(x, y_fake)
            gen_loss = torch.mean(torch.abs(y - y_fake))
            total_gen_loss += gen_loss.item()

            # Calcular pérdida del discriminador
            real_pred = disc(x, y)
            disc_real_loss = torch.mean(torch.abs(real_pred - torch.ones_like(real_pred)))
            disc_fake_loss = torch.mean(torch.abs(fake_pred))
            disc_loss = (disc_real_loss + disc_fake_loss) / 2
            total_disc_loss += disc_loss.item()

           # Calcular IoU para el fondo
            iou_bg = calculate_iou_bg(y_fake, y)
            total_iou_bg += iou_bg

            # Calcular IoU para las líneas
            iou_lines = calculate_iou_lines(y_fake, y)
            total_iou_lines += iou_lines

    avg_gen_loss = total_gen_loss / num_batches
    avg_disc_loss = total_disc_loss / num_batches
    avg_iou_bg = total_iou_bg / num_batches
    avg_iou_lines = total_iou_lines / num_batches

    print(f"Average Generator Loss: {avg_gen_loss:.4f}")
    print(f"Average Discriminator Loss: {avg_disc_loss:.4f}")
   
    print(f"Average IoU for background: {avg_iou_bg:.4f}")
    print(f"Average IoU for lines: {avg_iou_lines:.4f}")

def main():
    # Cargar modelos entrenados
    gen = Generator(in_channels=3, features=64).to(config.DEVICE)
    disc = Discriminator(in_channels=3).to(config.DEVICE)
    gen.load_state_dict(torch.load(config.CHECKPOINT_GEN, map_location=config.DEVICE)["state_dict"])
    disc.load_state_dict(torch.load(config.CHECKPOINT_DISC, map_location=config.DEVICE)["state_dict"])

    # Cargar conjunto de datos de validación
    val_dataset = MapDataset(root_dir=config.VAL_DIR)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    # Validar modelos
    validate_model(gen, disc, val_loader)

if __name__ == "__main__":
    main()

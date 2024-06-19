import torch
from utils import save_checkpoint, load_checkpoint, save_some_examples
import torch.nn as nn
import torch.optim as optim
import config
from dataset import MapDataset
from generator_model import Generator
from discriminator_model import Discriminator
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import os

torch.backends.cudnn.benchmark = True

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

def train_fn(
    disc, gen, loader, opt_disc, opt_gen, l1_loss, bce, g_scaler, d_scaler, epoch
):
    loop = tqdm(loader, leave=True)
    loop.set_description(f"Epoch [{epoch+1}/{config.NUM_EPOCHS}]")
    losses = []
    total_iou_bg = 0
    total_iou_lines = 0
    total_samples = 0

    for idx, (x, y) in enumerate(loop):
        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE)

        # Train Discriminator
        with torch.cuda.amp.autocast():
            y_fake = gen(x)
            D_real = disc(x, y)
            D_real_loss = bce(D_real, torch.full_like(D_real, config.real_label_smooth))
            D_fake = disc(x, y_fake.detach())
            D_fake_loss = bce(D_fake, torch.full_like(D_fake, config.fake_label_smooth))
            D_loss = (D_real_loss + D_fake_loss) / 2

        disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train generator
        with torch.cuda.amp.autocast():
            D_fake = disc(x, y_fake)
            G_fake_loss = bce(D_fake, torch.ones_like(D_fake))
            L1 = l1_loss(y_fake, y) * config.L1_LAMBDA
            G_loss = G_fake_loss + L1

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        # Print and save losses
        #if idx % 10 == 0:
         #   loop.set_postfix(
          #      D_real=torch.sigmoid(D_real).mean().item(),
          #      D_fake=torch.sigmoid(D_fake).mean().item(),
           #     G_loss=G_loss.item(),
            #    D_loss=D_loss.item()
            #)

        #losses.append((G_loss.item(), D_loss.item()))

        # Calcular IoU para el fondo
        iou_bg = calculate_iou_bg(y_fake, y)
        total_iou_bg += iou_bg * x.size(0)

        # Calcular IoU para las líneas
        iou_lines = calculate_iou_lines(y_fake, y)
        total_iou_lines += iou_lines * x.size(0)

        # Incrementar el número total de muestras
        total_samples += x.size(0)

        
        
    # Calcular el promedio del IoU para el fondo y las líneas
    avg_iou_bg = total_iou_bg / total_samples
    avg_iou_lines = total_iou_lines / total_samples

    # Imprimir los resultados
    print(f"Average IoU bg: {avg_iou_bg:.4f}, Average IoU lines: {avg_iou_lines:.4f}")

    # Calcular y guardar gráfica de pérdida
    #plt.figure(figsize=(10, 5))
    #plt.plot(range(len(losses)), [loss[0] for loss in losses], label='Generator Loss')
    #plt.plot(range(len(losses)), [loss[1] for loss in losses], label='Discriminator Loss')
    #plt.xlabel('Iterations')
    #plt.ylabel('Loss')
    #plt.title('Training Losses')
    #plt.legend()
    # Crear la carpeta "loss" si aún no existe
    #loss_folder = 'loss'
    #os.makedirs(loss_folder, exist_ok=True)
    #plt.savefig(os.path.join(loss_folder, f'epoch_{epoch+1}_loss.png'))
    #plt.close()
    return avg_iou_bg, avg_iou_lines
    



def main():
    disc = Discriminator(in_channels=config.CHANNELS_IMG).to(config.DEVICE)
    gen = Generator(in_channels=config.CHANNELS_IMG, features=64).to(config.DEVICE)
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE_DISC, betas=(0.5, 0.999),) #Añadimos el cuarto cambio (*0.9 no funciona)cambio quinto 
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE_GEN, betas=(0.5, 0.999)) # Añadimos quinto cambio 
    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE_GEN,
        )
        load_checkpoint(
            config.CHECKPOINT_DISC, disc, opt_disc, config.LEARNING_RATE_DISC,
        )

    train_dataset = MapDataset(root_dir=config.TRAIN_DIR)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
    )
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()
    val_dataset = MapDataset(root_dir=config.VAL_DIR)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    #Guardamos los valores de IoU de todo el entrenamiento
    iou_bg_all_epochs = []
    iou_lines_all_epochs = []
    
    for epoch in range(config.NUM_EPOCHS):
        avg_iou_bg, avg_iou_lines = train_fn(
            disc, gen, train_loader, opt_disc, opt_gen, L1_LOSS, BCE, g_scaler, d_scaler, epoch
        )

        if config.SAVE_MODEL and (epoch +1) % 5 == 0:
            save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
            save_checkpoint(disc, opt_disc, filename=config.CHECKPOINT_DISC)

        save_some_examples(gen, val_loader, epoch, folder="evaluation")
        
        iou_bg_all_epochs.append(avg_iou_bg)
        iou_lines_all_epochs.append(avg_iou_lines)
        
        # Representamos los IoU en una figura 
        plt.figure(figsize=(10, 5))
        plt.plot(iou_bg_all_epochs, label='Average IoU Background')
        plt.plot(iou_lines_all_epochs, label='Average IoU Lines')
        plt.xlabel('Epoch')
        plt.ylabel('IoU')
        plt.title('IoU Metrics Over Epochs')
        plt.legend()
        plt.savefig('iou_total.png')
        plt.close()
        


if __name__ == "__main__":
    main()
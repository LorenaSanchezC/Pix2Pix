import torch
from torchvision.utils import save_image
from generator_model import Generator
import config
from dataset import MapDataset
from torch.utils.data import DataLoader
import os



def test_model(gen, test_loader, output_folder):
    gen.eval()  # Modo de evaluación

    # Crear el directorio de salida si no existe
    #os.makedirs(output_folder, exist_ok=True)

    with torch.no_grad():  # No se necesitan gradientes para la inferencia
        for i, x in enumerate(test_loader):
            print(f"Procesando lote {i+1}")
            x = x.to(config.DEVICE)
            y_fake = gen(x)
            save_path = f"{output_folder}/generated_image_{i+1}.png"
            print(f"Guardando imagen generada en: {save_path}") 
            save_image(y_fake, save_path)
            #save_image(y_fake, f"{output_folder}/generated_image_{i+1}.png")

def main():
    # Cargar el modelo entrenado
    gen = Generator(in_channels=config.CHANNELS_IMG, features=64).to(config.DEVICE)
    checkpoint = torch.load(config.CHECKPOINT_GEN, map_location=config.DEVICE)
    gen.load_state_dict(checkpoint['state_dict'])

    # Preparar los datos de entrada
    test_dataset = MapDataset(root_dir=config.TEST_DIR, is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Directorio para guardar las imágenes generadas
    output_folder = "generated_images"
    test_model(gen, test_loader, output_folder)

if __name__ == "__main__":
    main()



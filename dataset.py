import torch
import numpy as np
import config
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image



class MapDataset(Dataset):
    def __init__(self, root_dir, is_test=False):
        self.root_dir = root_dir
        self.list_files = os.listdir(self.root_dir)
        self.is_test = is_test

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, index):
        img_file = self.list_files[index]
        img_path = os.path.join(self.root_dir, img_file)
        image = np.array(Image.open(img_path))
        
        if len(image.shape) == 2:  # La imagen es en escala de grises
            image = np.stack([image]*3, axis=-1)  # Replicar el canal de grises 3 veces a lo largo del último eje
    
        # En modo test, solo carga y transforma la imagen de entrada
        if self.is_test:
            input_image = image
            if config.transform_only_input_test:  # Suponiendo que tienes transformaciones definidas específicamente para test
                input_image = config.transform_only_input_test(image=input_image)["image"]
            return input_image

        # Para entrenamiento y validación, procesa tanto la imagen de entrada como la objetivo
        else:
            input_image = image[:, :config.IMAGE_SIZE, :]
            target_image = image[:, config.IMAGE_SIZE:, :]

            if config.both_transform:  # Aplica transformaciones a ambas imágenes
                augmentations = config.both_transform(image=input_image, image0=target_image)
                input_image = augmentations["image"]
                target_image = augmentations["image0"]

            if config.transform_only_input:  # Transformaciones específicas para la imagen de entrada
                input_image = config.transform_only_input(image=input_image)["image"]

            if config.transform_only_mask:  # Suponiendo que quieras transformar la imagen objetivo de alguna manera
                target_image = config.transform_only_mask(image=target_image)["image"]

            return input_image, target_image


if __name__ == "__main__":
    dataset = MapDataset("data/train/")
    loader = DataLoader(dataset, batch_size=5)
    for x, y in loader:
        print(x.shape)
        save_image(x, "x.png")
        save_image(y, "y.png")
        import sys

        sys.exit()
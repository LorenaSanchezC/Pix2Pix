import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations import RandomRotate90, RandomBrightnessContrast
from albumentations import RandomCrop, RandomGamma, ElasticTransform
import cv2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "data/train"
VAL_DIR = "data/val"
TEST_DIR = "data/test"
#LEARNING_RATE = 2e-4
LEARNING_RATE_GEN = 2e-4  # Mantén la tasa del generador, quinto cambio
LEARNING_RATE_DISC = 1e-4  # Disminuye la tasa del discriminador, quinto cambio 

BATCH_SIZE = 10 #16
NUM_WORKERS = 2
IMAGE_SIZE = 512
CHANNELS_IMG = 3
L1_LAMBDA = 10000 #100
NUM_EPOCHS = 100
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_DISC = "modelos/disc.pth.tar"
CHECKPOINT_GEN = "modelos/gen.pth.tar"
#Label smoothing
real_label_smooth = 1
fake_label_smooth = 0



both_transform = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0),
        
        #A.Resize(542, 542, interpolation=cv2.INTER_NEAREST),  # Resize a 542x542
        #A.RandomCrop(height=512, width=512),  # Random crop a 512x512
        #A.HorizontalFlip(p=0.5),  # Random horizontal flipping
        #A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize to [-1, 1]
        
    ], additional_targets={"image0": "image"},
)

transform_only_input = A.Compose(
    [
        ToTensorV2(),
    ]
)

transform_only_input_test = A.Compose(
    [
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0),  # Normalize to [-1, 1]
        ToTensorV2(),
    ]
)

transform_only_mask = A.Compose(
    [
        ToTensorV2(),
    ]
)
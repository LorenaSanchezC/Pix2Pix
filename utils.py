import torch
import config
from torchvision.utils import save_image

def save_some_examples(gen, val_loader, epoch, folder):
    try:
        x, y = next(iter(val_loader))
        x, y = x.to(config.DEVICE), y.to(config.DEVICE)
        gen.eval()
        with torch.no_grad():
            # Verificar si las imágenes tienen dimensiones válidas
            if x.size(2) > 0 and x.size(3) > 0:
                y_fake = gen(x)
                y_fake = y_fake * 0.5 + 0.5  # remove normalization#
                save_image(y_fake, folder + f"/y_gen_{epoch+1}.png") #Le añadimos uno porque en pantalla nos muestra epoch 1 pero internamente es 1 menos por lo que lo hacemos para igualar ambos y no liarnos visualmente 
                save_image(x * 0.5 + 0.5, folder + f"/input_{epoch+1}.png") #Empezaría guardando epoch 0 en vez de 1 
                if epoch == 0: #Primera epoch
                    save_image(y * 0.5 + 0.5, folder + f"/label_{epoch+1}.png")
                print("Examples saved successfully.")
            else:
                print("Error: Invalid image dimensions.")
    except Exception as e:
        print("An error occurred while saving examples:", e)
    gen.train()

def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


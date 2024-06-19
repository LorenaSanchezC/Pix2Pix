import torch
import torch.onnx
from generator_model import Generator
from discriminator_model import Discriminator
import config
#generator.eval()
# Crea un tensor de entrada de muestra
sample_input = torch.randn(1, 3, 512, 512).to(config.DEVICE)
sample_input_target = torch.randn(1, 3, 512, 512).to(config.DEVICE)
# Define un nombre para tu archivo ONNX
onnx_model_path_gen = "modelos/generator_model.onnx"
onnx_model_path_disc = "modelos/discriminator_model.onnx"
gen= Generator(in_channels=config.CHANNELS_IMG, features=32).to(config.DEVICE)
disc= Discriminator(in_channels=config.CHANNELS_IMG).to(config.DEVICE)
# Exporta el modelo

torch.onnx.export(gen,               # model being run
                  sample_input,                         # model input (or a tuple for multiple inputs)
                  onnx_model_path_gen,                      # where to save the model (can be a file or file-like object)
                  export_params=True,                   # store the trained parameter weights inside the model file
                  opset_version=10,                     # the ONNX version to export the model to
                  do_constant_folding=True,             # whether to execute constant folding for optimization
                  input_names=['input'],                # the model's input names
                  output_names=['output'],              # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},  # variable length axes
                                'output' : {0 : 'batch_size'}})

torch.onnx.export(disc,                                 # model being run
                  (sample_input, sample_input_target),  # model input (or a tuple for multiple inputs)
                  onnx_model_path_disc,                 # where to save the model (can be a file or file-like object)
                  export_params=True,                   # store the trained parameter weights inside the model file
                  opset_version=10,                     # the ONNX version to export the model to
                  do_constant_folding=True,             # whether to execute constant folding for optimization
                  input_names=['input', 'target'],      # the model's input names
                  output_names=['output'],              # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},  # variable length axes
                                'output' : {0 : 'batch_size'}})

#print(gen)
print(f"Modelos guardados correctamente.")
#https://netron.app/
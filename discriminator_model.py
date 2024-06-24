import torch 
import torch.nn as nn
import config

#Módulo CNN (Capa convolucional)
class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2): #Stride va a ser de 2 en todos 
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride, bias = False, padding_mode= 'reflect'), #Padding mode en reflect se supone que reduce "artefactos", tamaño del kernel 4, bias es false porque usamos batch2d
            nn.BatchNorm2d(out_channels), 
            nn.LeakyReLU(0.2), #Función de activación, en el discriminador solo se usa LeakyReLU 
        )
    def forward(self, x):
        return self.conv(x)
    
#Creamos el discriminador    
#Recibe x,y -> x siendo la img real, y siendo la img output <- las recibe juntas
class Discriminator(nn.Module):
    def __init__(self, in_channels=config.CHANNELS_IMG, features=[64,128,256,512]): #256->26x26 Canales 3 -> RGB  Usamos el bloque CNN 4 veces (features)
        super().__init__()
        self.initial = nn.Sequential( # En el bloque inicial no va a haber un batchnorm
            nn.Conv2d(in_channels*2, features[0], kernel_size=4, stride=2, padding=1, padding_mode= 'reflect'),#Multiplicamos por 2 porque vienen x, y
            nn.LeakyReLU(0.2),
        )

        layers = []
        in_channels = features[0]
        for feature in features[1:]: #Nos saltamos el primero, que el el bloque de arriba el self.initial
            layers.append(
                CNNBlock(in_channels, feature, stride = 1 if feature == features[-1]else 2), #Usamos stride 2 en todas menos en el de 512, que es el último [-1] 
            )
            in_channels = feature

        layers.append( #Metemos otra capa convolucional para que el resultado este entre 0 y 1
            nn.Conv2d( in_channels, 1, kernel_size=4, stride=1, padding=1, padding_mode='reflect')
        )
        self.model = nn.Sequential(*layers) #Pasas las layers por nn.Sequential

    def forward(self, x, y):
        x = torch.cat([x,y], dim= 1)
        x = self.initial(x)
        return self.model(x)





#Para realizar pruebas y ver si funciona 
def test():
    x = torch.randn((1,3,512,512))  #1 ejemplo, 3 canales, 256 y 256 input
    y = torch.randn((1,3,512,512))
    model = Discriminator()         #Iniciamos el modelo 
    preds = model(x,y)              # Predicciones del modelo 
    print(preds.shape)              #Mostramos el tamaño de la matriz de predicciones

if __name__ == "__main__":
    test()


#El resultado de la prueba es 
    # torch.Size([1, 1, 26, 26]) -> con (1,3,256,256)
    # torch.Size([1, 1, 58, 58]) -> con (1,3,512,512)
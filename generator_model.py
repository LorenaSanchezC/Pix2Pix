import torch
import torch.nn as nn
import config

#Creamos un bloque
class Block(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, act="relu", use_dropout=False): #down = True si estamos en el encoder (U-Net), y en false sería en la parte de decoder, ponemos que el dropout es false porque hay en una parte del decoder que se usa 
        super(Block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False, padding_mode="reflect") #Si es parte encoder es capa convolucional, bias false porque vamos a usar batchnorm
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False), # Si es parte decoder es capa transpose convolucional, aqui no se puede poner el padding como reflect da error
            nn.BatchNorm2d(out_channels),
            nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2), # En la parte de encoder se usa LeakyReLU y en la parte del decoder se usa ReLU
        )

        self.use_dropout = use_dropout 
        self.dropout = nn.Dropout(0.5)
        self.down = down

    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x


class Generator(nn.Module):
    def __init__(self, in_channels=config.CHANNELS_IMG, features=64):                                                 #Input = 256x256x3
        super().__init__()

        #           -------     ENCODER     -------
        self.initial_down = nn.Sequential( #Creamos una capa inicial donde no vamos a usar batchnorm
            nn.Conv2d(in_channels, features, kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )                                                                                           #128 bloque inicial 64 features 
         
        self.down1 = Block(features, features * 2, down=True, act="leaky", use_dropout=False)       #64     
        self.down2 = Block(features * 2, features * 4, down=True, act="leaky", use_dropout=False)   #32
        self.down3 = Block(features * 4, features * 8, down=True, act="leaky", use_dropout=False)   #16
        self.down4 = Block(features * 8, features * 8, down=True, act="leaky", use_dropout=False)   #8
        self.down5 = Block(features * 8, features * 8, down=True, act="leaky", use_dropout=False)   #4
        self.down6 = Block(features * 8, features * 8, down=True, act="leaky", use_dropout=False)   #2


        self.bottleneck = nn.Sequential(                                                            #centro 1x1
            nn.Conv2d(features * 8, features * 8, kernel_size=4, stride=2, padding=1), 
            nn.ReLU()
        )                                                                                           
        #           -------     DECODER     -------
        self.up1 = Block(features * 8, features * 8, down=False, act="relu", use_dropout=True)      #Usamos dropout 2
        self.up2 = Block(features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True)  #Usamos dropout 4
        self.up3 = Block(features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True)  #Usamos dropout 8
        self.up4 = Block(features * 8 * 2, features * 8, down=False, act="relu", use_dropout=False) #16
        self.up5 = Block(features * 8 * 2, features * 4, down=False, act="relu", use_dropout=False) #32
        self.up6 = Block(features * 4 * 2, features * 2, down=False, act="relu", use_dropout=False) #64
        self.up7 = Block(features * 2 * 2, features, down=False, act="relu", use_dropout=False)     #128
        
        self.final_up = nn.Sequential(                                                              #Ultima 256 -> tamaño output que coincide con el inicial que es 256
            nn.ConvTranspose2d(features * 2, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),  #Usamos tanh porque queremos que los valores de los pixeles esten entre -1 y 1, si quisieramos que esten entre 0 y 1 usariamos sigmoid
        )


    def forward(self, x):
        d1 = self.initial_down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        d7 = self.down6(d6)
        bottleneck = self.bottleneck(d7)
        up1 = self.up1(bottleneck)
        up2 = self.up2(torch.cat([up1, d7], 1))  #Concatenamos -> skip connection -> concatenamos d7 con up1 y de 1 dimensión (vector)
        up3 = self.up3(torch.cat([up2, d6], 1))
        up4 = self.up4(torch.cat([up3, d5], 1))
        up5 = self.up5(torch.cat([up4, d4], 1))
        up6 = self.up6(torch.cat([up5, d3], 1))
        up7 = self.up7(torch.cat([up6, d2], 1))
        return self.final_up(torch.cat([up7, d1], 1))




#Prueba, para comprobar que todo va bien y no hay fallos en la programación 
def test():
    x = torch.randn((1, 3, 512, 512))
    model = Generator(in_channels=3, features=64)
    preds = model(x)
    print(preds.shape)


if __name__ == "__main__":
    test()
    #torch.Size([1, 3, 256, 256]) -> mismo tamaño de entrada que de salida 
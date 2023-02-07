'''
#########################################################################################################################
INFORMACIÓN DEL FICHERO
#########################################################################################################################

Autor: Samuel Lozano Juárez
Fecha: 05/12/2022
Institución: UBU | Grado en Ingeniería de la Salud

Este archivo forma parte del Trabajo de Fin de Grado "Detección del grado de retinopatía mediante redes convolucionales".
El alumno a cargo de este proyecto es el declarado como autor en las líneas anteriores.
Los tutores del proyecto fueron el Dr. Darío Fernández Zoppino y el Dr. Daniel Urda Muñoz.

En el código que se encuentra a continuación se va a desarrollar una primera red neuronal básica, en la que se tratará de 
entrenar un modelo con el conjunto de datos de MNIST (Modified National Institute of Standards and Technology), que consiste
en un conjunto de imágenes de dígitos numéricos escritos a mano.
'''

#primero importamos todos las bibliotecas necesarias
#importamos pytorch
import torch
#importamos el paquete de redes neuronales
import torch.nn as nn
import torch.nn.functional as F
#importamos el paquete que permite cargar los datos
from torch.utils.data import Dataset
import torchvision
from torchvision import datasets
#importamos el paquete que permite el autoentrenamiento del modelo
from torchvision.transforms import ToTensor
#importamos los dataloaders
from torch.utils.data import DataLoader
#importamos la función de optimización
from torch import optim
#importamos el paquete para el entrenamiento de la red
from torch.autograd import Variable

#establecemos el dispositivo a usar (cpu o gpu)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device

#cargamos el conjunto de datos de MNIST
#primero el conjunto de datos de entrenamiento
train_data = datasets.MNIST(
    root = 'data',
    train = True,                         
    transform = ToTensor(), 
    download = True,            
)

#y posteriormente el conjunto de datos de test
test_data = datasets.MNIST(
    root = 'data', 
    train = False, 
    transform = ToTensor()
)

#vamos a mostrar las propiedades de cada conjunto de datos(entrenamiento y test)
print(train_data)
print('')
print(test_data)

#definimos las funciones de carga de las imágenes. especificando el tamaño de batch
#las almacenamos en un diccionario (loaders)
#el tamaño del batch en ambos casos es de 100, por lo que irá cargando de 100 en 100 imágenes
loaders = {
    'train' : torch.utils.data.DataLoader(train_data, 
                                          batch_size=100, 
                                          shuffle=True, 
                                          num_workers=1),
    
    'test'  : torch.utils.data.DataLoader(test_data, 
                                          batch_size=100, 
                                          shuffle=True, 
                                          num_workers=1),
}
loaders

#creamos el modelo de la red neuronal convolucional
#usaremos Torch.nn.Module como clase base para todos los modelos
#en este ejemplo construiremos una red con 2 capas convolucionales, con sus respectivas capas de relu y de MaxPooling
#por último una capa de 'fully conected'

class CNN(nn.Module):
    def __init__(self):
        #hereda de la clase CNN 
        super(CNN, self).__init__()
        #primera capa o layer
        self.conv1 = nn.Sequential(
            #capa convolucional
            nn.Conv2d(
                #in_channels = 1 porque la imagen es en grayscale
                in_channels=1,              
                out_channels=16,
                #tamaño del kernel
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),
            #capa de relu
            nn.ReLU(),
            #capa de MaxPooling
            nn.MaxPool2d(kernel_size=2),    
        )
        #segunda capa o layer
        self.conv2 = nn.Sequential(
            #capa convolucional
            nn.Conv2d(16, 32, 5, 1, 2),
            #capa de relu
            nn.ReLU(),
            #capa de maxpooling
            nn.MaxPool2d(2),                
        )
        # fully connected layer, output 10 classes
        self.out = nn.Linear(32 * 7 * 7, 10)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)       
        output = self.out(x)
        return output, x    # return x for visualization

#vamos a crear un objeto, una instancia de esta clase y mostrar sus propiedades
cnn = CNN()
print(cnn)

#definimos una función de loss (empleada en la retropropagación y ajuste de la red)
loss_func = nn.CrossEntropyLoss()
loss_func

#definimos una función de optimización basada en el algoritmo Adam
#indicamos un valor de learning rate de 0.01 (cuánto modificar el modelo en base al error estimado)
optimizer = optim.Adam(cnn.parameters(), lr=0.01)
optimizer

#entrenamos el modelo, creando una función 'train' para ello, que recibe como parámetros el número de épocas, el modelo a entrenar y los datos
#establecemos el número de épocas a 10
num_epochs = 10

#definimos la función que permite el entrenamiento
def train(num_epochs, cnn, loaders):
    
    #entrenamos el modelo
    cnn.train()
        
    #obtenemos el número total de pasos a partir de la longitud del DataLoader de entrenamiento
    total_step = len(loaders['train'])
    
    #para cada una de las épocas:
    for epoch in range(num_epochs):
        #recorremos el par imagen-label del DataLoader de entrenamiento
        for i, (images, labels) in enumerate(loaders['train']):
            
            # gives batch data, normalize x when iterate train_loader
            b_x = Variable(images)   # batch x
            b_y = Variable(labels)   # batch y
            output = cnn(b_x)[0]               
            loss = loss_func(output, b_y)
            
            # clear gradients for this training step   
            optimizer.zero_grad()           
            
            # backpropagation, compute gradients 
            loss.backward()    
            # apply gradients             
            optimizer.step()                
            
            if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
            pass
        pass
    
    pass

train(num_epochs, cnn, loaders)

#por último evaluamos el modelo creado con el conjunto de datos de test
#nuevamente definimos una función para ejecutar esta evaluación
def test():
    # Test the model
    cnn.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in loaders['test']:
            test_output, last_layer = cnn(images)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            accuracy = (pred_y == labels).sum().item() / float(labels.size(0))
            pass
    print('Test Accuracy of the model on the 10000 test images: %.2f' % accuracy)
    
    pass
test()


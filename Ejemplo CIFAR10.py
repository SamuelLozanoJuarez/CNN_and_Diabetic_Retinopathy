'''
#########################################################################################################################
INFORMACIÓN DEL FICHERO
#########################################################################################################################

Autor: Samuel Lozano Juárez
Fecha: 10/12/2022
Institución: UBU | Grado en Ingeniería de la Salud

Este archivo forma parte del Trabajo de Fin de Grado "Detección del grado de retinopatía mediante redes convolucionales".
El alumno a cargo de este proyecto es el declarado como autor en las líneas anteriores.
Los tutores del proyecto fueron el Dr. Darío Fernández Zoppino y el Dr. Daniel Urda Muñoz.

En el código que se encuentra a continuación se va a desarrollar una red neuronal convolucional básica, en la que se tratará de 
entrenar un modelo con el conjunto de datos de CIFAR10, que consiste en un conjunto de imágenes en color de 32 x 32 píxeles.
En el conjunto de imágenes se encuentran las siguientes clases: ‘airplane’, ‘automobile’, ‘bird’, ‘cat’, ‘deer’, ‘dog’, ‘frog’, 
‘horse’, ‘ship’, ‘truck’.

Los pasos para la construcción del modelo fueron tomados directamente de la página oficial de PyTorch.
'''

#importamos las bibliotecas necesarias
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#cargamos el conjunto de imágenes, transformándola de formato PILImage a Tensor
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#establecemos el tamaño del batch
batch_size = 4

#cargamos el conjunto de imágenes de entrenamiento
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=transform)
#definimos el loader que posteriormente va a permitirnos recorrer las imágenes
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=True, num_workers=2)

#repetimos el proceso pero en este caso para las imágenes de test
testset = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,shuffle=False, num_workers=2)

#definimos las clases
classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#una vez cargadas las imágenes, con sus respectivos labels, así como los DataLoaders, vamos a crear el modelo de la CNN
#nuevamente crearemos una clase que hereda de nn.Module

class Net(nn.Module):
    #en la función __init__ se especifican las capas de las que estará compuesto nuestro modelo
    def __init__(self):
        super().__init__()
        #una primera capa convolucional con in_channels = 3 ya que se trata de imágenes en color, out_channels = 6 ya que es el número de canales fruto de la convolucion, y kernel_size = 5
        self.conv1 = nn.Conv2d(3, 6, 5)
        #una capa de pooling con kernel_size = 2 y stride = 2 (número de píxeles que se desplaza la ventana)
        self.pool = nn.MaxPool2d(2, 2)
        #una nueva capa convolucional que recibe los 6 canales de la anterior y devuelve 16, con un kernel_size de 5
        self.conv2 = nn.Conv2d(6, 16, 5)
        #una capa que linealiza los datos que recibe instancias de dimensiones 16 * 5 * 5 y los lienaliza a dimensión 120
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        #una segunda capa lineal que recibe las instancias de 120 y las convierte a 84
        self.fc2 = nn.Linear(120, 84)
        #por último una capa lineal que recibe las 84 anteriores y lo convierte a las 10 clases de salida que deseamos
        self.fc3 = nn.Linear(84, 10)

    #definimos la función forward que contiene el modo en que la red predecirá el output usando las capas anteriormente descritas    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

#creamos una instancia de nuestra clase
net = Net()

#mostramos la red creada
net

#definimos la función de loss y el optimizador
#usaremos CrossEntropy como función loss
criterion = nn.CrossEntropyLoss()
#y la función SGD (Descenso de Gradiente Estocástico) con un learning rate 0.001 como optimizador
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

#entrenamos la red
#para ello iteramos sobre nuestro DataLoader y vamos alimentando la red con imágenes y sus labels, para que se ajusten los pesos sinápticos

#únicamente iteraremos 2 épocas
for epoch in range(2):

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')

#por último vamos a probar nuestra red ya entrenada con el conjunto de imágenes de test
correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

#una vez hemos obtenido el porcentaje de acierto del modelo, vamos a ver en qué clases tiene un mayor porcentaje de acierto y en cuáles falla más
# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1

# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')


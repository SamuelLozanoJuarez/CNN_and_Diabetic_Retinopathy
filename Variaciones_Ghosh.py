'''
#########################################################################################################################
INFORMACIÓN DEL FICHERO
#########################################################################################################################

Autor: Samuel Lozano Juárez
Fecha: 14/03/2023
Institución: UBU | Grado en Ingeniería de la Salud

Este archivo forma parte del Trabajo de Fin de Grado "Detección del grado de retinopatía mediante redes convolucionales".
El alumno a cargo de este proyecto es el declarado como autor en las líneas anteriores.
Los tutores del proyecto fueron el Dr. Darío Fernández Zoppino y el Dr. Daniel Urda Muñoz.

A continuación voy a desarrollar 6 variantes de la estructura descrita en el artículo Automatic Detection and Classification of Diabetic Retinopathy stages using CNN (Ghosh R.,Ghosh K.).
Las modificaciones a realizar consistirán en la disminución del número de capas convolucionales (eliminando las últimas 7 capas) y la reducción del número de neuronas de las capas
fully-connected. El objetivo que se pretende estudiar es domprobar si disminuyendo la complejidad de la red se obtienen mejores resultados.
'''

#primero importamos todos los paquetes necesarios
import torch #contiene todas las funciones de PyTorch
import torch.nn as nn #contiene la clase padre de todos los modelos (nn.Module)
import torch.nn.functional as F #esencial para la función de activación 
import torchvision #fundamental para la importación de imágenes
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt #para poder representar las gráficas
import numpy as np #para las métricas de la red

#importamos también las funcioness definidas para el entrenamiento y puesta a prueba de los modelos
from modules.CNN_utilities import entrena, representa_test, representa_train, tester

#establecemos el tamaño del batch, la escala de las imágenes y el número de épocas de entrenamiento
batch = 4
#la arquitectura propuesta por Ghosh requiere una escala de 512, 512, 3
escala = 512
epocas = 50

#a continuación definimos la operación que permitirá transformar las imágenes del repositorio en Tensores que puedan ser empleados por PyTorch
transform = transforms.Compose(
    [transforms.ToTensor(), #transforma la imagen de formato PIL a formato tensor
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), #normaliza el tensor para que la media de sus valores sea 0 y su desviación estándar 0.5
     transforms.Resize((escala, escala))]) #redimensionamos las imágenes

#a continuación cargamos el conjunto de imágenes de train (OCT) y los dos de test (iPhone y Samsung)
OCT = ImageFolder(root = 'Datos/Classified Data/Images/OCT', transform = transform)
print(f'Tamaño del conjunto de datos de train: {len(OCT)}')

Samsung = ImageFolder(root = 'Datos/Classified Data/Images/Samsung', transform = transform)
print(f'Tamaño del conjunto de datos de test de Samsung: {len(Samsung)}')

iPhone = ImageFolder(root = 'Datos/Classified Data/Images/iPhone', transform = transform)
print(f'Tamaño del conjunto de datos de test de iPhone: {len(iPhone)}')

#establecemos una lista con el nombre de las etiquetas
classes = OCT.classes

#y definimos también las funciones que van a ir cargando las imágenes en el modelo
train_loader = DataLoader(
    dataset = OCT,
    batch_size = 4, #establecemos un tamaño de lote (batch_size) de 4, ya que son pocas imágenes y podemos permitírnoslo
    shuffle = True, #indicamos que mezcle las imágenes
    num_workers = 2 #genera subprocesos para cargar los datos y así liberamos el proceso main
)

test_S_loader = DataLoader(
    dataset = Samsung,
    batch_size = 4, #establecemos un tamaño de lote (batch_size) de 10, ya que son pocas imágenes y podemos permitírnoslo
    shuffle = True, #indicamos que mezcle las imágenes
    num_workers = 2 #genera subprocesos para cargar los datos y así liberamos el proceso main
)

test_i_loader = DataLoader(
    dataset = iPhone,
    batch_size = 4, #establecemos un tamaño de lote (batch_size) de 10, ya que son pocas imágenes y podemos permitírnoslo
    shuffle = True, #indicamos que mezcle las imágenes
    num_workers = 2 #genera subprocesos para cargar los datos y así liberamos el proceso main
)

#A lo largo de este script voy a probar a variar algunos parámetros del modelo (intentando no perder la esencia de la estructura original)
#Los parámetros modificados serán los siguientes:
# - reducción del número de capas convolucionales (incluir únicamente las 6 primeras capas para disminuir la complejidad)
# - empleando una serie decreciente de neuronas en las capas fully-connected probar las siguientes combinaciones:
#       * 1024, 512, 256
#       * 512, 256, 128
#       * 256, 128, 64
# nuevamente con el objetivo de reducir la complejidad de la red y tratar de mejorar los resultados.
#Existen por tanto 6 posibles variantes (combinando ambos parámetros)

######################################################
#Primera variante: 13 capas convolucionales y neuronas FC (fully-connected) 1024, 512, 256
class Primera_var(nn.Module):
    #esta estructura está formada por capas convolucionales, de maxpooling, de activación, de Dropout, fully-connected y de clasificación
    
    def __init__(self):
        #sobreescribimos el constructor del padre
        super(Primera_var,self).__init__()
        #primero definimos una capa convolucional
        self.conv1 = nn.Conv2d(
            in_channels = 3, #3 canales de entrada porque las imágenes son a color
            out_channels = 32, #se trata del número de salidas de la capa. Es el número de kernels de la capa
            kernel_size = 7, #suele tratarse de un número impar
            stride = 2, #cantidad píxeles que se desplaza el filtro sobre la imagen
            padding = 2, #cantidad de relleno que se va a aplicar sobre los bordes de la imagen
        )
        
        #la segunda (y tercera) capa convolucional, se pueden definir como una única porque el número de entradas y salidas coincide
        self.conv2_3 = nn.Conv2d(
            in_channels = 32, #32 canales de entrada para que coincida con las salidas de la capa anterior
            out_channels = 32, #se trata del número de salidas de la capa. Es el número de kernels de la capa
            kernel_size = 3, #suele tratarse de un número impar
            stride = 2, #cantidad píxeles que se desplaza el filtro sobre la imagen
            padding = 2, #cantidad de relleno que se va a aplicar sobre los bordes de la imagen
        )
        
        #la cuarta capa convolucional
        self.conv4 = nn.Conv2d(
            in_channels = 32, #32 canales de entrada para que coincida con las salidas de la capa anterior
            out_channels = 64, #se trata del número de salidas de la capa. Es el número de kernels de la capa
            kernel_size = 3, #suele tratarse de un número impar
            stride = 2, #cantidad píxeles que se desplaza el filtro sobre la imagen
            padding = 2, #cantidad de relleno que se va a aplicar sobre los bordes de la imagen
        )
        
        #la quinta capa convolucional
        self.conv5 = nn.Conv2d(
            in_channels = 64, #64 canales de entrada para que coincida con las salidas de la capa anterior
            out_channels = 64, #se trata del número de salidas de la capa. Es el número de kernels de la capa
            kernel_size = 3, #suele tratarse de un número impar
            stride = 2, #cantidad píxeles que se desplaza el filtro sobre la imagen
            padding = 2, #cantidad de relleno que se va a aplicar sobre los bordes de la imagen
        )
        
        #la sexta capa convolucional
        self.conv6 = nn.Conv2d(
            in_channels = 64, #64 canales de entrada para que coincida con las salidas de la capa anterior
            out_channels = 128, #se trata del número de salidas de la capa. Es el número de kernels de la capa
            kernel_size = 3, #suele tratarse de un número impar
            stride = 2, #cantidad píxeles que se desplaza el filtro sobre la imagen
            padding = 2, #cantidad de relleno que se va a aplicar sobre los bordes de la imagen
        )
        
        #la séptima (y octava y novena) capa convolucional
        self.conv7_8_9 = nn.Conv2d(
            in_channels = 128, #64 canales de entrada para que coincida con las salidas de la capa anterior
            out_channels = 128, #se trata del número de salidas de la capa. Es el número de kernels de la capa
            kernel_size = 3, #suele tratarse de un número impar
            stride = 2, #cantidad píxeles que se desplaza el filtro sobre la imagen
            padding = 2, #cantidad de relleno que se va a aplicar sobre los bordes de la imagen
        )
        
        #la décima capa convolucional
        self.conv10 = nn.Conv2d(
            in_channels = 128, #64 canales de entrada para que coincida con las salidas de la capa anterior
            out_channels = 256, #se trata del número de salidas de la capa. Es el número de kernels de la capa
            kernel_size = 3, #suele tratarse de un número impar
            stride = 2, #cantidad píxeles que se desplaza el filtro sobre la imagen
            padding = 2, #cantidad de relleno que se va a aplicar sobre los bordes de la imagen
        )
        
        #las últimas 3 capa convolucionales
        self.conv11_12_13 = nn.Conv2d(
            in_channels = 256, #64 canales de entrada para que coincida con las salidas de la capa anterior
            out_channels = 256, #se trata del número de salidas de la capa. Es el número de kernels de la capa
            kernel_size = 3, #suele tratarse de un número impar
            stride = 2, #cantidad píxeles que se desplaza el filtro sobre la imagen
            padding = 2, #cantidad de relleno que se va a aplicar sobre los bordes de la imagen
        )
        
        #la función de activación (en este caso PReLU)
        self.activation = nn.PReLU()
        
        #la capa de MaxPool
        self.pool = nn.MaxPool2d(
            kernel_size = 2, #establecemos el tamaño del kernel a 2*2
            stride = 2 #cantidad píxeles que se desplaza el filtro sobre la imagen
        )
        
        #la primera capa de neuronas a la que aplicaremos Dropout como técnica de regularización
        self.fc1 = nn.Linear(
            in_features = 256, #número de características de entrada
            out_features = 1024 #número de neuronas de salida
        )
        
        #la segunda capa fully-connected
        self.fc2 = nn.Linear(1024,512)
        
        #la tercera capa fully-connected
        self.fc3 = nn.Linear(512,256)
        
        #la capa de neuronas fully-connected final
        self.dense = nn.Linear(
            in_features = 256, #número de parámetros de entrada de la red (los valores se obtienen experimentalmente)
            out_features = 5 #número de neuronas de salida
        )
        
    def forward(self,x):
        #en esta función es donde tiene lugar la computación (y la función invocada por defecto al ejecutar la red)
        #siguiendo la estructura descrita en Ghosh et al.:
        
        #primero una capa convolucional de tipo 1, con su consecuente activación PReLU y la capa de MaxPool
        x = self.pool(self.activation(self.conv1(x)))
        #una capa convolucional de tipo 2 con su correspondiente activación
        x = self.activation(self.conv2_3(x))
        #capa convolucional de tipo 2 con activación y MaxPool
        x = self.pool(self.activation(self.conv2_3(x)))
        #cuarta convolucional con activación
        x = self.activation(self.conv4(x))
        #quinta convolucional con activación y MaxPool
        x = self.pool(self.activation(self.conv5(x)))
        #3 capas convolucionales consecutivas de tipo 2 con su correspondiente activación
        x = self.activation(self.conv6(x))
        x = self.activation(self.conv7_8_9(x))
        x = self.activation(self.conv7_8_9(x))
        #novena capa convolucional con activación y MaxPool
        x = self.pool(self.activation(self.conv7_8_9(x)))
        #se repite la misma estructura de 3 capas convolucionales con activación y una última con activación y MaxPool
        x = self.activation(self.conv10(x))
        x = self.activation(self.conv11_12_13(x))
        x = self.activation(self.conv11_12_13(x))
        x = self.pool(self.activation(self.conv11_12_13(x)))
        #aplanamos la salida, hasta convertirla de forma matricial a forma vectorial (sería la capa flatten)
        x = x.view(-1,self.num_flat_features(x))#usamos una función propia de la clase para obtener el número de características
        #aplicamos una primera red neuronal fully-connected, con la activación consecuente y la estrategia dropout para evitar el sobreentrenamiento
        x = F.dropout(self.activation(self.fc1(x)))
        #lo mismo sucede con la segunda capa fully-connected
        x = F.dropout(self.activation(self.fc2(x)))
        #y con la tercera
        x = F.dropout(self.activation(self.fc3(x)))
        #por último tiene lugar la capa de predicciones, que convierte las 512 neuronas de la tercera capa fully-connected en una salida de 5 neuronas (una por clase)
        x = self.dense(x)
        
        return x
    
    def num_flat_features(self,x):
        #por último definimos la función que permite obtener el número de características de los tensores
        size = x.size()[1:] #seleccionamos todas las dimensiones expcepto la primera (que son los batches)
        num_features = 1
        #va iterando y calcula el número de características de los datos (x)
        for s in size:
            num_features*=s
        return num_features

#creamos una instancia de esta red
primera = Primera_var()
#mostramos su estructura
print(primera)

#definimos como loss la función de tipo cross entropy 
criterion = nn.CrossEntropyLoss() 

#en este caso el optimizador será la función Adam (ampliamente utilizada)
optimizer = torch.optim.Adam(params = primera.parameters()) #dejamos el valor de learning-rate por defecto (0.001)

#entrenamos la red haciendo uso de la función 'entrena()' importada
#recogemos los resultados en 2 variables que posteriormente nos permitirán representar la evolución de accuracy y loss
acc,loss = entrena(primera,epocas,train_loader,optimizer,criterion)

#representamos la evolución temporal de accuracy
representa_train(acc,'Accuracy','Ghosh - Primera Variante')

#y representamos el loss
representa_train(acc,'Loss','Ghosh - Primera Variante')

#finalmente ponemos a prueba la red usando la función tester y recogemos los resultados para obtener las métricas
y_true_iphone, y_pred_iphone, predictions_iphone = tester(primera,test_i_loader)
#y posteriormente obtenemos y mostramos las métricas
representa_test(y_true_iphone,y_pred_iphone,predictions_iphone,'iPhone','Primera Variante')

#repetimos el mismo proceso pero empleando el conjunto de imágenes de Samsung
y_true_samsung, y_pred_samsung, predictions_samsung = tester(primera,test_S_loader)
#y posteriormente obtenemos y mostramos las métricas
representa_test(y_true_samsung,y_pred_samsung,predictions_samsung,'Samsung','Primera Variante')

######################################################
#Segunda variante: 13 capas convolucionales y neuronas FC (fully-connected) 512, 256, 128
class Segunda_var(nn.Module):
    #esta estructura está formada por capas convolucionales, de maxpooling, de activación, de Dropout, fully-connected y de clasificación
    
    def __init__(self):
        #sobreescribimos el constructor del padre
        super(Segunda_var,self).__init__()
        #primero definimos una capa convolucional
        self.conv1 = nn.Conv2d(
            in_channels = 3, #3 canales de entrada porque las imágenes son a color
            out_channels = 32, #se trata del número de salidas de la capa. Es el número de kernels de la capa
            kernel_size = 7, #suele tratarse de un número impar
            stride = 2, #cantidad píxeles que se desplaza el filtro sobre la imagen
            padding = 2, #cantidad de relleno que se va a aplicar sobre los bordes de la imagen
        )
        
        #la segunda (y tercera) capa convolucional, se pueden definir como una única porque el número de entradas y salidas coincide
        self.conv2_3 = nn.Conv2d(
            in_channels = 32, #32 canales de entrada para que coincida con las salidas de la capa anterior
            out_channels = 32, #se trata del número de salidas de la capa. Es el número de kernels de la capa
            kernel_size = 3, #suele tratarse de un número impar
            stride = 2, #cantidad píxeles que se desplaza el filtro sobre la imagen
            padding = 2, #cantidad de relleno que se va a aplicar sobre los bordes de la imagen
        )
        
        #la cuarta capa convolucional
        self.conv4 = nn.Conv2d(
            in_channels = 32, #32 canales de entrada para que coincida con las salidas de la capa anterior
            out_channels = 64, #se trata del número de salidas de la capa. Es el número de kernels de la capa
            kernel_size = 3, #suele tratarse de un número impar
            stride = 2, #cantidad píxeles que se desplaza el filtro sobre la imagen
            padding = 2, #cantidad de relleno que se va a aplicar sobre los bordes de la imagen
        )
        
        #la quinta capa convolucional
        self.conv5 = nn.Conv2d(
            in_channels = 64, #64 canales de entrada para que coincida con las salidas de la capa anterior
            out_channels = 64, #se trata del número de salidas de la capa. Es el número de kernels de la capa
            kernel_size = 3, #suele tratarse de un número impar
            stride = 2, #cantidad píxeles que se desplaza el filtro sobre la imagen
            padding = 2, #cantidad de relleno que se va a aplicar sobre los bordes de la imagen
        )
        
        #la sexta capa convolucional
        self.conv6 = nn.Conv2d(
            in_channels = 64, #64 canales de entrada para que coincida con las salidas de la capa anterior
            out_channels = 128, #se trata del número de salidas de la capa. Es el número de kernels de la capa
            kernel_size = 3, #suele tratarse de un número impar
            stride = 2, #cantidad píxeles que se desplaza el filtro sobre la imagen
            padding = 2, #cantidad de relleno que se va a aplicar sobre los bordes de la imagen
        )
        
        #la séptima (y octava y novena) capa convolucional
        self.conv7_8_9 = nn.Conv2d(
            in_channels = 128, #64 canales de entrada para que coincida con las salidas de la capa anterior
            out_channels = 128, #se trata del número de salidas de la capa. Es el número de kernels de la capa
            kernel_size = 3, #suele tratarse de un número impar
            stride = 2, #cantidad píxeles que se desplaza el filtro sobre la imagen
            padding = 2, #cantidad de relleno que se va a aplicar sobre los bordes de la imagen
        )
        
        #la décima capa convolucional
        self.conv10 = nn.Conv2d(
            in_channels = 128, #64 canales de entrada para que coincida con las salidas de la capa anterior
            out_channels = 256, #se trata del número de salidas de la capa. Es el número de kernels de la capa
            kernel_size = 3, #suele tratarse de un número impar
            stride = 2, #cantidad píxeles que se desplaza el filtro sobre la imagen
            padding = 2, #cantidad de relleno que se va a aplicar sobre los bordes de la imagen
        )
        
        #las últimas 3 capa convolucionales
        self.conv11_12_13 = nn.Conv2d(
            in_channels = 256, #64 canales de entrada para que coincida con las salidas de la capa anterior
            out_channels = 256, #se trata del número de salidas de la capa. Es el número de kernels de la capa
            kernel_size = 3, #suele tratarse de un número impar
            stride = 2, #cantidad píxeles que se desplaza el filtro sobre la imagen
            padding = 2, #cantidad de relleno que se va a aplicar sobre los bordes de la imagen
        )
        
        #la función de activación (en este caso PReLU)
        self.activation = nn.PReLU()
        
        #la capa de MaxPool
        self.pool = nn.MaxPool2d(
            kernel_size = 2, #establecemos el tamaño del kernel a 2*2
            stride = 2 #cantidad píxeles que se desplaza el filtro sobre la imagen
        )
        
        #la primera capa de neuronas a la que aplicaremos Dropout como técnica de regularización
        self.fc1 = nn.Linear(
            in_features = 256, #número de características de entrada
            out_features = 512 #número de neuronas de salida
        )
        
        #la segunda capa fully-connected
        self.fc2 = nn.Linear(512,256)
        
        #la tercera capa fully-connected
        self.fc3 = nn.Linear(256,128)
        
        #la capa de neuronas fully-connected final
        self.dense = nn.Linear(
            in_features = 128, #número de parámetros de entrada de la red (los valores se obtienen experimentalmente)
            out_features = 5 #número de neuronas de salida
        )
        
    def forward(self,x):
        #en esta función es donde tiene lugar la computación (y la función invocada por defecto al ejecutar la red)
        #siguiendo la estructura descrita en Ghosh et al.:
        
        #primero una capa convolucional de tipo 1, con su consecuente activación PReLU y la capa de MaxPool
        x = self.pool(self.activation(self.conv1(x)))
        #una capa convolucional de tipo 2 con su correspondiente activación
        x = self.activation(self.conv2_3(x))
        #capa convolucional de tipo 2 con activación y MaxPool
        x = self.pool(self.activation(self.conv2_3(x)))
        #cuarta convolucional con activación
        x = self.activation(self.conv4(x))
        #quinta convolucional con activación y MaxPool
        x = self.pool(self.activation(self.conv5(x)))
        #3 capas convolucionales consecutivas de tipo 2 con su correspondiente activación
        x = self.activation(self.conv6(x))
        x = self.activation(self.conv7_8_9(x))
        x = self.activation(self.conv7_8_9(x))
        #novena capa convolucional con activación y MaxPool
        x = self.pool(self.activation(self.conv7_8_9(x)))
        #se repite la misma estructura de 3 capas convolucionales con activación y una última con activación y MaxPool
        x = self.activation(self.conv10(x))
        x = self.activation(self.conv11_12_13(x))
        x = self.activation(self.conv11_12_13(x))
        x = self.pool(self.activation(self.conv11_12_13(x)))
        #aplanamos la salida, hasta convertirla de forma matricial a forma vectorial (sería la capa flatten)
        x = x.view(-1,self.num_flat_features(x))#usamos una función propia de la clase para obtener el número de características
        #aplicamos una primera red neuronal fully-connected, con la activación consecuente y la estrategia dropout para evitar el sobreentrenamiento
        x = F.dropout(self.activation(self.fc1(x)))
        #lo mismo sucede con la segunda capa fully-connected
        x = F.dropout(self.activation(self.fc2(x)))
        #y con la tercera
        x = F.dropout(self.activation(self.fc3(x)))
        #por último tiene lugar la capa de predicciones, que convierte las 512 neuronas de la tercera capa fully-connected en una salida de 5 neuronas (una por clase)
        x = self.dense(x)
        
        return x
    
    def num_flat_features(self,x):
        #por último definimos la función que permite obtener el número de características de los tensores
        size = x.size()[1:] #seleccionamos todas las dimensiones expcepto la primera (que son los batches)
        num_features = 1
        #va iterando y calcula el número de características de los datos (x)
        for s in size:
            num_features*=s
        return num_features

#creamos una instancia de esta red
segunda = Segunda_var()
#mostramos su estructura
print(segunda)

#definimos como loss la función de tipo cross entropy 
criterion = nn.CrossEntropyLoss() 

#en este caso el optimizador será la función Adam (ampliamente utilizada)
optimizer = torch.optim.Adam(params = segunda.parameters()) #dejamos el valor de learning-rate por defecto (0.001)

#entrenamos la red haciendo uso de la función 'entrena()' importada
#recogemos los resultados en 2 variables que posteriormente nos permitirán representar la evolución de accuracy y loss
acc,loss = entrena(segunda,epocas,train_loader,optimizer,criterion)

#representamos la evolución temporal de accuracy
representa_train(acc,'Accuracy','Ghosh - Segunda Variante')

#y representamos el loss
representa_train(acc,'Loss','Ghosh - Segunda Variante')

#finalmente ponemos a prueba la red usando la función tester y recogemos los resultados para obtener las métricas
y_true_iphone, y_pred_iphone, predictions_iphone = tester(segunda,test_i_loader)
#y posteriormente obtenemos y mostramos las métricas
representa_test(y_true_iphone,y_pred_iphone,predictions_iphone,'iPhone','Segunda Variante')

#repetimos el mismo proceso pero empleando el conjunto de imágenes de Samsung
y_true_samsung, y_pred_samsung, predictions_samsung = tester(segunda,test_S_loader)
#y posteriormente obtenemos y mostramos las métricas
representa_test(y_true_samsung,y_pred_samsung,predictions_samsung,'Samsung','Segunda Variante')

######################################################
#Tercera variante: 13 capas convolucionales y neuronas FC (fully-connected) 256, 128, 64
class Tercera_var(nn.Module):
    #esta estructura está formada por capas convolucionales, de maxpooling, de activación, de Dropout, fully-connected y de clasificación
    
    def __init__(self):
        #sobreescribimos el constructor del padre
        super(Tercera_var,self).__init__()
        #primero definimos una capa convolucional
        self.conv1 = nn.Conv2d(
            in_channels = 3, #3 canales de entrada porque las imágenes son a color
            out_channels = 32, #se trata del número de salidas de la capa. Es el número de kernels de la capa
            kernel_size = 7, #suele tratarse de un número impar
            stride = 2, #cantidad píxeles que se desplaza el filtro sobre la imagen
            padding = 2, #cantidad de relleno que se va a aplicar sobre los bordes de la imagen
        )
        
        #la segunda (y tercera) capa convolucional, se pueden definir como una única porque el número de entradas y salidas coincide
        self.conv2_3 = nn.Conv2d(
            in_channels = 32, #32 canales de entrada para que coincida con las salidas de la capa anterior
            out_channels = 32, #se trata del número de salidas de la capa. Es el número de kernels de la capa
            kernel_size = 3, #suele tratarse de un número impar
            stride = 2, #cantidad píxeles que se desplaza el filtro sobre la imagen
            padding = 2, #cantidad de relleno que se va a aplicar sobre los bordes de la imagen
        )
        
        #la cuarta capa convolucional
        self.conv4 = nn.Conv2d(
            in_channels = 32, #32 canales de entrada para que coincida con las salidas de la capa anterior
            out_channels = 64, #se trata del número de salidas de la capa. Es el número de kernels de la capa
            kernel_size = 3, #suele tratarse de un número impar
            stride = 2, #cantidad píxeles que se desplaza el filtro sobre la imagen
            padding = 2, #cantidad de relleno que se va a aplicar sobre los bordes de la imagen
        )
        
        #la quinta capa convolucional
        self.conv5 = nn.Conv2d(
            in_channels = 64, #64 canales de entrada para que coincida con las salidas de la capa anterior
            out_channels = 64, #se trata del número de salidas de la capa. Es el número de kernels de la capa
            kernel_size = 3, #suele tratarse de un número impar
            stride = 2, #cantidad píxeles que se desplaza el filtro sobre la imagen
            padding = 2, #cantidad de relleno que se va a aplicar sobre los bordes de la imagen
        )
        
        #la sexta capa convolucional
        self.conv6 = nn.Conv2d(
            in_channels = 64, #64 canales de entrada para que coincida con las salidas de la capa anterior
            out_channels = 128, #se trata del número de salidas de la capa. Es el número de kernels de la capa
            kernel_size = 3, #suele tratarse de un número impar
            stride = 2, #cantidad píxeles que se desplaza el filtro sobre la imagen
            padding = 2, #cantidad de relleno que se va a aplicar sobre los bordes de la imagen
        )
        
        #la séptima (y octava y novena) capa convolucional
        self.conv7_8_9 = nn.Conv2d(
            in_channels = 128, #64 canales de entrada para que coincida con las salidas de la capa anterior
            out_channels = 128, #se trata del número de salidas de la capa. Es el número de kernels de la capa
            kernel_size = 3, #suele tratarse de un número impar
            stride = 2, #cantidad píxeles que se desplaza el filtro sobre la imagen
            padding = 2, #cantidad de relleno que se va a aplicar sobre los bordes de la imagen
        )
        
        #la décima capa convolucional
        self.conv10 = nn.Conv2d(
            in_channels = 128, #64 canales de entrada para que coincida con las salidas de la capa anterior
            out_channels = 256, #se trata del número de salidas de la capa. Es el número de kernels de la capa
            kernel_size = 3, #suele tratarse de un número impar
            stride = 2, #cantidad píxeles que se desplaza el filtro sobre la imagen
            padding = 2, #cantidad de relleno que se va a aplicar sobre los bordes de la imagen
        )
        
        #las últimas 3 capa convolucionales
        self.conv11_12_13 = nn.Conv2d(
            in_channels = 256, #64 canales de entrada para que coincida con las salidas de la capa anterior
            out_channels = 256, #se trata del número de salidas de la capa. Es el número de kernels de la capa
            kernel_size = 3, #suele tratarse de un número impar
            stride = 2, #cantidad píxeles que se desplaza el filtro sobre la imagen
            padding = 2, #cantidad de relleno que se va a aplicar sobre los bordes de la imagen
        )
        
        #la función de activación (en este caso PReLU)
        self.activation = nn.PReLU()
        
        #la capa de MaxPool
        self.pool = nn.MaxPool2d(
            kernel_size = 2, #establecemos el tamaño del kernel a 2*2
            stride = 2 #cantidad píxeles que se desplaza el filtro sobre la imagen
        )
        
        #la primera capa de neuronas a la que aplicaremos Dropout como técnica de regularización
        self.fc1 = nn.Linear(
            in_features = 256, #número de características de entrada
            out_features = 256 #número de neuronas de salida
        )
        
        #la segunda capa fully-connected
        self.fc2 = nn.Linear(256,128)
        
        #la tercera capa fully-connected
        self.fc3 = nn.Linear(128,64)
        
        #la capa de neuronas fully-connected final
        self.dense = nn.Linear(
            in_features = 64, #número de parámetros de entrada de la red (los valores se obtienen experimentalmente)
            out_features = 5 #número de neuronas de salida
        )
        
    def forward(self,x):
        #en esta función es donde tiene lugar la computación (y la función invocada por defecto al ejecutar la red)
        #siguiendo la estructura descrita en Ghosh et al.:
        
        #primero una capa convolucional de tipo 1, con su consecuente activación PReLU y la capa de MaxPool
        x = self.pool(self.activation(self.conv1(x)))
        #una capa convolucional de tipo 2 con su correspondiente activación
        x = self.activation(self.conv2_3(x))
        #capa convolucional de tipo 2 con activación y MaxPool
        x = self.pool(self.activation(self.conv2_3(x)))
        #cuarta convolucional con activación
        x = self.activation(self.conv4(x))
        #quinta convolucional con activación y MaxPool
        x = self.pool(self.activation(self.conv5(x)))
        #3 capas convolucionales consecutivas de tipo 2 con su correspondiente activación
        x = self.activation(self.conv6(x))
        x = self.activation(self.conv7_8_9(x))
        x = self.activation(self.conv7_8_9(x))
        #novena capa convolucional con activación y MaxPool
        x = self.pool(self.activation(self.conv7_8_9(x)))
        #se repite la misma estructura de 3 capas convolucionales con activación y una última con activación y MaxPool
        x = self.activation(self.conv10(x))
        x = self.activation(self.conv11_12_13(x))
        x = self.activation(self.conv11_12_13(x))
        x = self.pool(self.activation(self.conv11_12_13(x)))
        #aplanamos la salida, hasta convertirla de forma matricial a forma vectorial (sería la capa flatten)
        x = x.view(-1,self.num_flat_features(x))#usamos una función propia de la clase para obtener el número de características
        #aplicamos una primera red neuronal fully-connected, con la activación consecuente y la estrategia dropout para evitar el sobreentrenamiento
        x = F.dropout(self.activation(self.fc1(x)))
        #lo mismo sucede con la segunda capa fully-connected
        x = F.dropout(self.activation(self.fc2(x)))
        #y con la tercera
        x = F.dropout(self.activation(self.fc3(x)))
        #por último tiene lugar la capa de predicciones, que convierte las 512 neuronas de la tercera capa fully-connected en una salida de 5 neuronas (una por clase)
        x = self.dense(x)
        
        return x
    
    def num_flat_features(self,x):
        #por último definimos la función que permite obtener el número de características de los tensores
        size = x.size()[1:] #seleccionamos todas las dimensiones expcepto la primera (que son los batches)
        num_features = 1
        #va iterando y calcula el número de características de los datos (x)
        for s in size:
            num_features*=s
        return num_features

#creamos una instancia de esta red
tercera = Tercera_var()
#mostramos su estructura
print(tercera)

#definimos como loss la función de tipo cross entropy 
criterion = nn.CrossEntropyLoss() 

#en este caso el optimizador será la función Adam (ampliamente utilizada)
optimizer = torch.optim.Adam(params = tercera.parameters()) #dejamos el valor de learning-rate por defecto (0.001)

#entrenamos la red haciendo uso de la función 'entrena()' importada
#recogemos los resultados en 2 variables que posteriormente nos permitirán representar la evolución de accuracy y loss
acc,loss = entrena(tercera,epocas,train_loader,optimizer,criterion)

#representamos la evolución temporal de accuracy
representa_train(acc,'Accuracy','Ghosh - Tercera Variante')

#y representamos el loss
representa_train(acc,'Loss','Ghosh - Tercera Variante')

#finalmente ponemos a prueba la red usando la función tester y recogemos los resultados para obtener las métricas
y_true_iphone, y_pred_iphone, predictions_iphone = tester(tercera,test_i_loader)
#y posteriormente obtenemos y mostramos las métricas
representa_test(y_true_iphone,y_pred_iphone,predictions_iphone,'iPhone','Tercera Variante')

#repetimos el mismo proceso pero empleando el conjunto de imágenes de Samsung
y_true_samsung, y_pred_samsung, predictions_samsung = tester(tercera,test_S_loader)
#y posteriormente obtenemos y mostramos las métricas
representa_test(y_true_samsung,y_pred_samsung,predictions_samsung,'Samsung','Tercera Variante')

######################################################
#Cuarta variante: 6 capas convolucionales y neuronas FC (fully-connected) 1024, 512, 256
class Cuarta_var(nn.Module):
    #esta estructura está formada por capas convolucionales, de maxpooling, de activación, de Dropout, fully-connected y de clasificación
    
    def __init__(self):
        #sobreescribimos el constructor del padre
        super(Cuarta_var,self).__init__()
        #primero definimos una capa convolucional
        self.conv1 = nn.Conv2d(
            in_channels = 3, #3 canales de entrada porque las imágenes son a color
            out_channels = 32, #se trata del número de salidas de la capa. Es el número de kernels de la capa
            kernel_size = 7, #suele tratarse de un número impar
            stride = 2, #cantidad píxeles que se desplaza el filtro sobre la imagen
            padding = 2, #cantidad de relleno que se va a aplicar sobre los bordes de la imagen
        )
        
        #la segunda (y tercera) capa convolucional, se pueden definir como una única porque el número de entradas y salidas coincide
        self.conv2_3 = nn.Conv2d(
            in_channels = 32, #32 canales de entrada para que coincida con las salidas de la capa anterior
            out_channels = 32, #se trata del número de salidas de la capa. Es el número de kernels de la capa
            kernel_size = 3, #suele tratarse de un número impar
            stride = 2, #cantidad píxeles que se desplaza el filtro sobre la imagen
            padding = 2, #cantidad de relleno que se va a aplicar sobre los bordes de la imagen
        )
        
        #la cuarta capa convolucional
        self.conv4 = nn.Conv2d(
            in_channels = 32, #32 canales de entrada para que coincida con las salidas de la capa anterior
            out_channels = 64, #se trata del número de salidas de la capa. Es el número de kernels de la capa
            kernel_size = 3, #suele tratarse de un número impar
            stride = 2, #cantidad píxeles que se desplaza el filtro sobre la imagen
            padding = 2, #cantidad de relleno que se va a aplicar sobre los bordes de la imagen
        )
        
        #la quinta capa convolucional
        self.conv5 = nn.Conv2d(
            in_channels = 64, #64 canales de entrada para que coincida con las salidas de la capa anterior
            out_channels = 64, #se trata del número de salidas de la capa. Es el número de kernels de la capa
            kernel_size = 3, #suele tratarse de un número impar
            stride = 2, #cantidad píxeles que se desplaza el filtro sobre la imagen
            padding = 2, #cantidad de relleno que se va a aplicar sobre los bordes de la imagen
        )
        
        #la sexta capa convolucional
        self.conv6 = nn.Conv2d(
            in_channels = 64, #64 canales de entrada para que coincida con las salidas de la capa anterior
            out_channels = 128, #se trata del número de salidas de la capa. Es el número de kernels de la capa
            kernel_size = 3, #suele tratarse de un número impar
            stride = 2, #cantidad píxeles que se desplaza el filtro sobre la imagen
            padding = 2, #cantidad de relleno que se va a aplicar sobre los bordes de la imagen
        )
        
        #la función de activación (en este caso PReLU)
        self.activation = nn.PReLU()
        
        #la capa de MaxPool
        self.pool = nn.MaxPool2d(
            kernel_size = 2, #establecemos el tamaño del kernel a 2*2
            stride = 2 #cantidad píxeles que se desplaza el filtro sobre la imagen
        )
        
        #la primera capa de neuronas a la que aplicaremos Dropout como técnica de regularización
        self.fc1 = nn.Linear(
            in_features = 1152, #número de características de entrada
            out_features = 1024 #número de neuronas de salida
        )
        
        #la segunda capa fully-connected
        self.fc2 = nn.Linear(1024,512)
        
        #la tercera capa fully-connected
        self.fc3 = nn.Linear(512,256)
        
        #la capa de neuronas fully-connected final
        self.dense = nn.Linear(
            in_features = 256, #número de parámetros de entrada de la red (los valores se obtienen experimentalmente)
            out_features = 5 #número de neuronas de salida
        )
        
    def forward(self,x):
        #en esta función es donde tiene lugar la computación (y la función invocada por defecto al ejecutar la red)
        #siguiendo la estructura descrita en Ghosh et al.:
        
        #primero una capa convolucional de tipo 1, con su consecuente activación PReLU y la capa de MaxPool
        x = self.pool(self.activation(self.conv1(x)))
        #una capa convolucional de tipo 2 con su correspondiente activación
        x = self.activation(self.conv2_3(x))
        #capa convolucional de tipo 2 con activación y MaxPool
        x = self.pool(self.activation(self.conv2_3(x)))
        #cuarta convolucional con activación
        x = self.activation(self.conv4(x))
        #quinta convolucional con activación y MaxPool
        x = self.pool(self.activation(self.conv5(x)))
        #una última capa convolucional con su correspondiente activación
        x = self.activation(self.conv6(x))
        #aplanamos la salida, hasta convertirla de forma matricial a forma vectorial (sería la capa flatten)
        x = x.view(-1,self.num_flat_features(x))#usamos una función propia de la clase para obtener el número de características
        #aplicamos una primera red neuronal fully-connected, con la activación consecuente y la estrategia dropout para evitar el sobreentrenamiento
        x = F.dropout(self.activation(self.fc1(x)))
        #lo mismo sucede con la segunda capa fully-connected
        x = F.dropout(self.activation(self.fc2(x)))
        #y con la tercera
        x = F.dropout(self.activation(self.fc3(x)))
        #por último tiene lugar la capa de predicciones, que convierte las 512 neuronas de la tercera capa fully-connected en una salida de 5 neuronas (una por clase)
        x = self.dense(x)
        
        return x
    
    def num_flat_features(self,x):
        #por último definimos la función que permite obtener el número de características de los tensores
        size = x.size()[1:] #seleccionamos todas las dimensiones expcepto la primera (que son los batches)
        num_features = 1
        #va iterando y calcula el número de características de los datos (x)
        for s in size:
            num_features*=s
        return num_features

#creamos una instancia de esta red
cuarta = Cuarta_var()
#mostramos su estructura
print(cuarta)

#definimos como loss la función de tipo cross entropy 
criterion = nn.CrossEntropyLoss() 

#en este caso el optimizador será la función Adam (ampliamente utilizada)
optimizer = torch.optim.Adam(params = cuarta.parameters()) #dejamos el valor de learning-rate por defecto (0.001)

#entrenamos la red haciendo uso de la función 'entrena()' importada
#recogemos los resultados en 2 variables que posteriormente nos permitirán representar la evolución de accuracy y loss
acc,loss = entrena(cuarta,epocas,train_loader,optimizer,criterion)

#representamos la evolución temporal de accuracy
representa_train(acc,'Accuracy','Ghosh - Cuarta Variante')

#y representamos el loss
representa_train(acc,'Loss','Ghosh - Cuarta Variante')

#finalmente ponemos a prueba la red usando la función tester y recogemos los resultados para obtener las métricas
y_true_iphone, y_pred_iphone, predictions_iphone = tester(cuarta,test_i_loader)
#y posteriormente obtenemos y mostramos las métricas
representa_test(y_true_iphone,y_pred_iphone,predictions_iphone,'iPhone','Cuarta Variante')

#repetimos el mismo proceso pero empleando el conjunto de imágenes de Samsung
y_true_samsung, y_pred_samsung, predictions_samsung = tester(cuarta,test_S_loader)
#y posteriormente obtenemos y mostramos las métricas
representa_test(y_true_samsung,y_pred_samsung,predictions_samsung,'Samsung','Cuarta Variante')

######################################################
#Quinta variante: 6 capas convolucionales y neuronas FC (fully-connected) 512, 256, 128
class Quinta_var(nn.Module):
    #esta estructura está formada por capas convolucionales, de maxpooling, de activación, de Dropout, fully-connected y de clasificación
    
    def __init__(self):
        #sobreescribimos el constructor del padre
        super(Quinta_var,self).__init__()
        #primero definimos una capa convolucional
        self.conv1 = nn.Conv2d(
            in_channels = 3, #3 canales de entrada porque las imágenes son a color
            out_channels = 32, #se trata del número de salidas de la capa. Es el número de kernels de la capa
            kernel_size = 7, #suele tratarse de un número impar
            stride = 2, #cantidad píxeles que se desplaza el filtro sobre la imagen
            padding = 2, #cantidad de relleno que se va a aplicar sobre los bordes de la imagen
        )
        
        #la segunda (y tercera) capa convolucional, se pueden definir como una única porque el número de entradas y salidas coincide
        self.conv2_3 = nn.Conv2d(
            in_channels = 32, #32 canales de entrada para que coincida con las salidas de la capa anterior
            out_channels = 32, #se trata del número de salidas de la capa. Es el número de kernels de la capa
            kernel_size = 3, #suele tratarse de un número impar
            stride = 2, #cantidad píxeles que se desplaza el filtro sobre la imagen
            padding = 2, #cantidad de relleno que se va a aplicar sobre los bordes de la imagen
        )
        
        #la cuarta capa convolucional
        self.conv4 = nn.Conv2d(
            in_channels = 32, #32 canales de entrada para que coincida con las salidas de la capa anterior
            out_channels = 64, #se trata del número de salidas de la capa. Es el número de kernels de la capa
            kernel_size = 3, #suele tratarse de un número impar
            stride = 2, #cantidad píxeles que se desplaza el filtro sobre la imagen
            padding = 2, #cantidad de relleno que se va a aplicar sobre los bordes de la imagen
        )
        
        #la quinta capa convolucional
        self.conv5 = nn.Conv2d(
            in_channels = 64, #64 canales de entrada para que coincida con las salidas de la capa anterior
            out_channels = 64, #se trata del número de salidas de la capa. Es el número de kernels de la capa
            kernel_size = 3, #suele tratarse de un número impar
            stride = 2, #cantidad píxeles que se desplaza el filtro sobre la imagen
            padding = 2, #cantidad de relleno que se va a aplicar sobre los bordes de la imagen
        )
        
        #la sexta capa convolucional
        self.conv6 = nn.Conv2d(
            in_channels = 64, #64 canales de entrada para que coincida con las salidas de la capa anterior
            out_channels = 128, #se trata del número de salidas de la capa. Es el número de kernels de la capa
            kernel_size = 3, #suele tratarse de un número impar
            stride = 2, #cantidad píxeles que se desplaza el filtro sobre la imagen
            padding = 2, #cantidad de relleno que se va a aplicar sobre los bordes de la imagen
        )
        
        #la función de activación (en este caso PReLU)
        self.activation = nn.PReLU()
        
        #la capa de MaxPool
        self.pool = nn.MaxPool2d(
            kernel_size = 2, #establecemos el tamaño del kernel a 2*2
            stride = 2 #cantidad píxeles que se desplaza el filtro sobre la imagen
        )
        
        #la primera capa de neuronas a la que aplicaremos Dropout como técnica de regularización
        self.fc1 = nn.Linear(
            in_features = 1152, #número de características de entrada
            out_features = 512 #número de neuronas de salida
        )
        
        #la segunda capa fully-connected
        self.fc2 = nn.Linear(512,256)
        
        #la tercera capa fully-connected
        self.fc3 = nn.Linear(256,128)
        
        #la capa de neuronas fully-connected final
        self.dense = nn.Linear(
            in_features = 128, #número de parámetros de entrada de la red (los valores se obtienen experimentalmente)
            out_features = 5 #número de neuronas de salida
        )
        
    def forward(self,x):
        #en esta función es donde tiene lugar la computación (y la función invocada por defecto al ejecutar la red)
        #siguiendo la estructura descrita en Ghosh et al.:
        
        #primero una capa convolucional de tipo 1, con su consecuente activación PReLU y la capa de MaxPool
        x = self.pool(self.activation(self.conv1(x)))
        #una capa convolucional de tipo 2 con su correspondiente activación
        x = self.activation(self.conv2_3(x))
        #capa convolucional de tipo 2 con activación y MaxPool
        x = self.pool(self.activation(self.conv2_3(x)))
        #cuarta convolucional con activación
        x = self.activation(self.conv4(x))
        #quinta convolucional con activación y MaxPool
        x = self.pool(self.activation(self.conv5(x)))
        #una última capa convolucional con su correspondiente activación
        x = self.activation(self.conv6(x))
        #aplanamos la salida, hasta convertirla de forma matricial a forma vectorial (sería la capa flatten)
        x = x.view(-1,self.num_flat_features(x))#usamos una función propia de la clase para obtener el número de características
        #aplicamos una primera red neuronal fully-connected, con la activación consecuente y la estrategia dropout para evitar el sobreentrenamiento
        x = F.dropout(self.activation(self.fc1(x)))
        #lo mismo sucede con la segunda capa fully-connected
        x = F.dropout(self.activation(self.fc2(x)))
        #y con la tercera
        x = F.dropout(self.activation(self.fc3(x)))
        #por último tiene lugar la capa de predicciones, que convierte las 512 neuronas de la tercera capa fully-connected en una salida de 5 neuronas (una por clase)
        x = self.dense(x)
        
        return x
    
    def num_flat_features(self,x):
        #por último definimos la función que permite obtener el número de características de los tensores
        size = x.size()[1:] #seleccionamos todas las dimensiones expcepto la primera (que son los batches)
        num_features = 1
        #va iterando y calcula el número de características de los datos (x)
        for s in size:
            num_features*=s
        return num_features

#creamos una instancia de esta red
quinta = Quinta_var()
#mostramos su estructura
print(quinta)

#definimos como loss la función de tipo cross entropy 
criterion = nn.CrossEntropyLoss() 

#en este caso el optimizador será la función Adam (ampliamente utilizada)
optimizer = torch.optim.Adam(params = quinta.parameters()) #dejamos el valor de learning-rate por defecto (0.001)

#entrenamos la red haciendo uso de la función 'entrena()' importada
#recogemos los resultados en 2 variables que posteriormente nos permitirán representar la evolución de accuracy y loss
acc,loss = entrena(quinta,epocas,train_loader,optimizer,criterion)

#representamos la evolución temporal de accuracy
representa_train(acc,'Accuracy','Ghosh - Quienta Variante')

#y representamos el loss
representa_train(acc,'Loss','Ghosh - Quinta Variante')

#finalmente ponemos a prueba la red usando la función tester y recogemos los resultados para obtener las métricas
y_true_iphone, y_pred_iphone, predictions_iphone = tester(quinta,test_i_loader)
#y posteriormente obtenemos y mostramos las métricas
representa_test(y_true_iphone,y_pred_iphone,predictions_iphone,'iPhone','Quinta Variante')

#repetimos el mismo proceso pero empleando el conjunto de imágenes de Samsung
y_true_samsung, y_pred_samsung, predictions_samsung = tester(quinta,test_S_loader)
#y posteriormente obtenemos y mostramos las métricas
representa_test(y_true_samsung,y_pred_samsung,predictions_samsung,'Samsung','Quinta Variante')

######################################################
#Sexta variante: 6 capas convolucionales y neuronas FC (fully-connected) 256, 128, 64
class Sexta_var(nn.Module):
    #esta estructura está formada por capas convolucionales, de maxpooling, de activación, de Dropout, fully-connected y de clasificación
    
    def __init__(self):
        #sobreescribimos el constructor del padre
        super(Sexta_var,self).__init__()
        #primero definimos una capa convolucional
        self.conv1 = nn.Conv2d(
            in_channels = 3, #3 canales de entrada porque las imágenes son a color
            out_channels = 32, #se trata del número de salidas de la capa. Es el número de kernels de la capa
            kernel_size = 7, #suele tratarse de un número impar
            stride = 2, #cantidad píxeles que se desplaza el filtro sobre la imagen
            padding = 2, #cantidad de relleno que se va a aplicar sobre los bordes de la imagen
        )
        
        #la segunda (y tercera) capa convolucional, se pueden definir como una única porque el número de entradas y salidas coincide
        self.conv2_3 = nn.Conv2d(
            in_channels = 32, #32 canales de entrada para que coincida con las salidas de la capa anterior
            out_channels = 32, #se trata del número de salidas de la capa. Es el número de kernels de la capa
            kernel_size = 3, #suele tratarse de un número impar
            stride = 2, #cantidad píxeles que se desplaza el filtro sobre la imagen
            padding = 2, #cantidad de relleno que se va a aplicar sobre los bordes de la imagen
        )
        
        #la cuarta capa convolucional
        self.conv4 = nn.Conv2d(
            in_channels = 32, #32 canales de entrada para que coincida con las salidas de la capa anterior
            out_channels = 64, #se trata del número de salidas de la capa. Es el número de kernels de la capa
            kernel_size = 3, #suele tratarse de un número impar
            stride = 2, #cantidad píxeles que se desplaza el filtro sobre la imagen
            padding = 2, #cantidad de relleno que se va a aplicar sobre los bordes de la imagen
        )
        
        #la quinta capa convolucional
        self.conv5 = nn.Conv2d(
            in_channels = 64, #64 canales de entrada para que coincida con las salidas de la capa anterior
            out_channels = 64, #se trata del número de salidas de la capa. Es el número de kernels de la capa
            kernel_size = 3, #suele tratarse de un número impar
            stride = 2, #cantidad píxeles que se desplaza el filtro sobre la imagen
            padding = 2, #cantidad de relleno que se va a aplicar sobre los bordes de la imagen
        )
        
        #la sexta capa convolucional
        self.conv6 = nn.Conv2d(
            in_channels = 64, #64 canales de entrada para que coincida con las salidas de la capa anterior
            out_channels = 128, #se trata del número de salidas de la capa. Es el número de kernels de la capa
            kernel_size = 3, #suele tratarse de un número impar
            stride = 2, #cantidad píxeles que se desplaza el filtro sobre la imagen
            padding = 2, #cantidad de relleno que se va a aplicar sobre los bordes de la imagen
        )
        
        #la función de activación (en este caso PReLU)
        self.activation = nn.PReLU()
        
        #la capa de MaxPool
        self.pool = nn.MaxPool2d(
            kernel_size = 2, #establecemos el tamaño del kernel a 2*2
            stride = 2 #cantidad píxeles que se desplaza el filtro sobre la imagen
        )
        
        #la primera capa de neuronas a la que aplicaremos Dropout como técnica de regularización
        self.fc1 = nn.Linear(
            in_features = 1152, #número de características de entrada
            out_features = 256 #número de neuronas de salida
        )
        
        #la segunda capa fully-connected
        self.fc2 = nn.Linear(256,128)
        
        #la tercera capa fully-connected
        self.fc3 = nn.Linear(128,64)
        
        #la capa de neuronas fully-connected final
        self.dense = nn.Linear(
            in_features = 64, #número de parámetros de entrada de la red (los valores se obtienen experimentalmente)
            out_features = 5 #número de neuronas de salida
        )
        
    def forward(self,x):
        #en esta función es donde tiene lugar la computación (y la función invocada por defecto al ejecutar la red)
        #siguiendo la estructura descrita en Ghosh et al.:
        
        #primero una capa convolucional de tipo 1, con su consecuente activación PReLU y la capa de MaxPool
        x = self.pool(self.activation(self.conv1(x)))
        #una capa convolucional de tipo 2 con su correspondiente activación
        x = self.activation(self.conv2_3(x))
        #capa convolucional de tipo 2 con activación y MaxPool
        x = self.pool(self.activation(self.conv2_3(x)))
        #cuarta convolucional con activación
        x = self.activation(self.conv4(x))
        #quinta convolucional con activación y MaxPool
        x = self.pool(self.activation(self.conv5(x)))
        #una última capa convolucional con su correspondiente activación
        x = self.activation(self.conv6(x))
        #aplanamos la salida, hasta convertirla de forma matricial a forma vectorial (sería la capa flatten)
        x = x.view(-1,self.num_flat_features(x))#usamos una función propia de la clase para obtener el número de características
        #aplicamos una primera red neuronal fully-connected, con la activación consecuente y la estrategia dropout para evitar el sobreentrenamiento
        x = F.dropout(self.activation(self.fc1(x)))
        #lo mismo sucede con la segunda capa fully-connected
        x = F.dropout(self.activation(self.fc2(x)))
        #y con la tercera
        x = F.dropout(self.activation(self.fc3(x)))
        #por último tiene lugar la capa de predicciones, que convierte las 512 neuronas de la tercera capa fully-connected en una salida de 5 neuronas (una por clase)
        x = self.dense(x)
        
        return x
    
    def num_flat_features(self,x):
        #por último definimos la función que permite obtener el número de características de los tensores
        size = x.size()[1:] #seleccionamos todas las dimensiones expcepto la primera (que son los batches)
        num_features = 1
        #va iterando y calcula el número de características de los datos (x)
        for s in size:
            num_features*=s
        return num_features

#creamos una instancia de esta red
sexta = Sexta_var()
#mostramos su estructura
print(sexta)

#definimos como loss la función de tipo cross entropy 
criterion = nn.CrossEntropyLoss() 

#en este caso el optimizador será la función Adam (ampliamente utilizada)
optimizer = torch.optim.Adam(params = sexta.parameters()) #dejamos el valor de learning-rate por defecto (0.001)

#entrenamos la red haciendo uso de la función 'entrena()' importada
#recogemos los resultados en 2 variables que posteriormente nos permitirán representar la evolución de accuracy y loss
acc,loss = entrena(sexta,epocas,train_loader,optimizer,criterion)

#representamos la evolución temporal de accuracy
representa_train(acc,'Accuracy','Ghosh - Sexta Variante')

#y representamos el loss
representa_train(acc,'Loss','Ghosh - Sexta Variante')

#finalmente ponemos a prueba la red usando la función tester y recogemos los resultados para obtener las métricas
y_true_iphone, y_pred_iphone, predictions_iphone = tester(sexta,test_i_loader)
#y posteriormente obtenemos y mostramos las métricas
representa_test(y_true_iphone,y_pred_iphone,predictions_iphone,'iPhone','Sexta Variante')

#repetimos el mismo proceso pero empleando el conjunto de imágenes de Samsung
y_true_samsung, y_pred_samsung, predictions_samsung = tester(sexta,test_S_loader)
#y posteriormente obtenemos y mostramos las métricas
representa_test(y_true_samsung,y_pred_samsung,predictions_samsung,'Samsung','Sexta Variante')


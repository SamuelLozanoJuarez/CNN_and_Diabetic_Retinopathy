#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


#establecemos el tamaño del batch, la escala de las imágenes y el número de épocas de entrenamiento
batch = 4
#en la arquitectura propuesta por Mobeen no se especifica ninguna escala, por lo que se empleará una escala cualquiera (512 por ejemplo)
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


# In[3]:


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


# In[4]:


#A lo largo de este script voy a probar a variar algunos parámetros del modelo (intentando no perder la esencia de la estructura original)
#Los parámetros modificados serán los siguientes:
# - inclusión o no de una capa convolucional adicional con un kernel = 3 y número de filtros = 16 o 32
# - modificar el número de neuronas de las capas fully-connected:
#        * probar con las combinaciones 256/128, 128/64, 64/32 y original (100/50)
#
#por tanto existen 12 posibles variaciones


# In[5]:


##########################################################
#Primera Variante: sin capa convolucional adicional y combinación fully-connected 256/128
class Primera_var(nn.Module):
    
    def __init__(self):
        #sobreescribimos el constructor del padre
        super(Primera_var,self).__init__()
        #creamos la primera capa convolucional
        self.conv1 = nn.Conv2d(
            in_channels = 3, #3 canales de entrada porque las imágenes son a color
            out_channels = 4, #se trata del número de salidas de la capa. Es el número de kernels de la capa convolucional
            kernel_size = 3, #suele tratarse de un número impar
            stride = 1, #cantidad píxeles que se desplaza el filtro sobre la imagen
            padding = 2, #cantidad de relleno que se va a aplicar sobre los bordes de la imagen
        )
        
        #creamos la segunda capa convolucional
        self.conv2 = nn.Conv2d(
            in_channels = 4, #3 canales de entrada porque las imágenes son a color
            out_channels = 16, #se trata del número de salidas de la capa. Es el número de kernels de la capa convolucional
            kernel_size = 3, #suele tratarse de un número impar
            stride = 1, #cantidad píxeles que se desplaza el filtro sobre la imagen
            padding = 2, #cantidad de relleno que se va a aplicar sobre los bordes de la imagen
        )
        
        #definimos la capa de maxpooling
        self.pool = nn.MaxPool2d(
            kernel_size = 2, #establecemos el tamaño del kernel a 2*2
            stride = 2 #cantidad píxeles que se desplaza el filtro sobre la imagen (por defecto se desplazará el tamaño del kernel)
        )
        
        #y continuamos con las capas de neuronas fully-connected
        self.fc1 = nn.Linear(
            in_features = 16*129*129, #número de parámetros de entrada de la red
            out_features = 256
        )
        
        #segunda capa fully-connected
        self.fc2 = nn.Linear(256,128)
        
        #y la última capa de fully connected que va a ser la que proporcione la última predicción
        self.fc3 = nn.Linear(128,5)
        
    def forward(self,x):
        #en esta función es donde tiene lugar la computación (y la función invocada por defecto al ejecutar la red)
        #en esta ocasión la función de activación que emplearemos es ReLU
        #primero tiene lugar la primera capa convolucional, con su respectiva activación y pooling
        x = self.pool(F.relu(self.conv1(x)))
        #posteriormente la segunda capa convolucional, con ReLU y pooling
        x = self.pool(F.relu(self.conv2(x)))
        #posteriormente la capa de flatten que convierte los datos de forma matricial a vectorial, para poder trabajar en las capas neuronales
        x = x.view(-1,self.num_flat_features(x))#usamos una función propia de la clase para obtener el número de características
        #por último las 3 capas neuronales fully-connected, con su correspondiente activación
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x
    
    def num_flat_features(self,x):
        #por último definimos la función que permite obtener el número de características de los tensores
        size = x.size()[1:] #seleccionamos todas las dimensiones expcepto la primera (que son los batches)
        num_features = 1
        #va iterando y calcula el número de características de los datos (x)
        for s in size:
            num_features*=s
        return num_features


# In[7]:


#una vez creada la estructura generamos una instancia de la misma y mostramos sus capas
primera = Primera_var()
print(primera)


# In[9]:


#definimos como loss la función de tipo cross entropy 
criterion = nn.CrossEntropyLoss() 

#en este caso el optimizador será la función Adam (ampliamente utilizada)
optimizer = torch.optim.Adam(params = primera.parameters()) #dejamos el valor de learning rate por defecto


# In[ ]:


#entrenamos la red haciendo uso de la función 'entrena()' importada
#recogemos los resultados en 2 variables que posteriormente nos permitirán representar la evolución de accuracy y loss
acc,loss = entrena(primera,epocas,train_loader,optimizer,criterion)


# In[ ]:


#representamos la evolución temporal de accuracy
representa_train(acc,'Accuracy','Mobeen - Primera Variante')


# In[ ]:


#y representamos el loss
representa_train(acc,'Loss','Mobeen - Primera Variante')


# In[ ]:


#finalmente ponemos a prueba la red usando la función tester y recogemos los resultados para obtener las métricas
y_true_iphone, y_pred_iphone, predictions_iphone = tester(primera,test_i_loader)
#y posteriormente obtenemos y mostramos las métricas
representa_test(y_true_iphone,y_pred_iphone,predictions_iphone,'iPhone','Primera Variante')


# In[ ]:


#repetimos el mismo proceso pero empleando el conjunto de imágenes de Samsung
y_true_samsung, y_pred_samsung, predictions_samsung = tester(primera,test_S_loader)
#y posteriormente obtenemos y mostramos las métricas
representa_test(y_true_samsung,y_pred_samsung,predictions_samsung,'Samsung','Primera Variante')


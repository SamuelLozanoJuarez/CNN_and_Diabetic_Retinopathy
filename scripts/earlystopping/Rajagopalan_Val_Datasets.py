'''
#########################################################################################################################
INFORMACIÓN DEL FICHERO
#########################################################################################################################

Autor: Samuel Lozano Juárez
Fecha: 20/05/2023
Institución: UBU | Grado en Ingeniería de la Salud

Este archivo forma parte del Trabajo de Fin de Grado "Detección del grado de retinopatía mediante redes convolucionales".
El alumno a cargo de este proyecto es el declarado como autor en las líneas anteriores.
Los tutores del proyecto fueron el Dr. Darío Fernández Zoppino y el Dr. Daniel Urda Muñoz.

A continuación se incluye el código que permite crear varios modelos según la arquitectura propuesta en el artículo de Rajagopalan, pero realizando las modificacionesdeseadas en los parámetros (número de capas convolucionales de la arquitectura, número de filtros por capa  y número de neuronas de las capas fully-connected).
Para el entrenamiento se usarán las imágenes de los repositorios y un conjunto de datos de validación y se empleará la estrategia de Early Stopping, para evitar el sobreentrenamiento.
Todas estas arquitecturas serán entrenadas y testeadas, y sus resultados se almacenarán automáticamente en un archivo .csv llamado Resultados.
Además también se guardarán el estado de los modelos (sus pesos) por si quisieran reutilizarse.
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
from modules.CNN_utilities import entrena_val, representa_test, obtiene_metricas, tester, guarda_graficas

#importamos el paquete que permite calcular el tiempo de entrenamiento
import time

#incluimos las siguientes líneas para evitar problemas al trabajar con imágenes truncadas
import PIL
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

#establecemos el tamaño del batch, la escala de las imágenes y el número de épocas de entrenamiento
batch = 64
#la arquitectura propuesta por Rajagopalan requiere una escala de 224, 224, 3
escala = 224
epocas = 150 #ya que tenemos activado el Early Stopping

#vamos a definir el valor de los parámetros capas, filtros y neuronas, ya que no podemos ejecutar todas las posibles combinaciones por el tiempo que implicaría
#las combinaciones que se van a probar son las siguientes:
#  * 7,2.0,512/256
#  * 7,0.5,128/256
#  * 7,0.5,512/256
#  * 5,1.0,512/256
capas = 7
n_filtros = 2.0
n_neuronas = '512/256'

#a continuación definimos la operación que permitirá transformar las imágenes del repositorio en Tensores que puedan ser empleados por PyTorch
transform = transforms.Compose(
    [transforms.ToTensor(), #transforma la imagen de formato PIL a formato tensor
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), #normaliza el tensor para que la media de sus valores sea 0 y su desviación estándar 0.5
     transforms.Resize((escala, escala))]) #redimensionamos las imágenes

#a continuación cargamos el conjunto de imágenes de train (Datasets) y los dos de test (iPhone y Samsung)
Datasets = ImageFolder(root = 'Datos/Classified Data/Images/Datasets', transform = transform)
print(f'Tamaño del conjunto de datos de train: {len(Datasets)}')

Samsung = ImageFolder(root = 'Datos/Classified Data/Images/Samsung/No_inpaint', transform = transform)
print(f'Tamaño del conjunto de datos de test de Samsung: {len(Samsung)}')

iPhone = ImageFolder(root = 'Datos/Classified Data/Images/iPhone/No_inpaint', transform = transform)
print(f'Tamaño del conjunto de datos de test de iPhone: {len(iPhone)}')

#establecemos una lista con el nombre de las etiquetas
classes = Datasets.classes

#en esta ocasión, debido a que vamos a implementar EarlyStopping es necesario dividir el conjunto de entrenamiento en train y validation
#Dividimos el conjunto de datos en entrenamiento y validación (80% y 20% respectivamente)
train_size = int(0.8 * len(Datasets))
val_size = len(Datasets) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(Datasets, [train_size, val_size])

# Crear cargadores de datos para cada conjunto
train_loader = torch.utils.data.DataLoader(
    dataset = train_dataset,
    batch_size = batch, 
    shuffle = True,
    num_workers = 8 #genera subprocesos para cargar los datos y así liberamos el proceso main
)

val_loader = torch.utils.data.DataLoader(
    dataset = val_dataset,
    batch_size = batch,
    shuffle = True,
    num_workers = 8
)

test_S_loader = DataLoader(
    dataset = Samsung,
    batch_size = batch, #establecemos un tamaño de lote (batch_size) de 4, ya que son pocas imágenes y podemos permitírnoslo
    shuffle = True, #indicamos que mezcle las imágenes
    num_workers = 8 #genera subprocesos para cargar los datos y así liberamos el proceso main
)

test_i_loader = DataLoader(
    dataset = iPhone,
    batch_size = batch, #establecemos un tamaño de lote (batch_size) de 4, ya que son pocas imágenes y podemos permitírnoslo
    shuffle = True, #indicamos que mezcle las imágenes
    num_workers = 8 #genera subprocesos para cargar los datos y así liberamos el proceso main
)

#Para facilitar la lectura del código y sobre todo su ejecución, al igual que en ocasiones anteriores voy a definir una función que permita lanzar las ejecuciones necesarias de manera automática

def crea_Rajagopalan(capas_conv, filtros, neuronas):
    '''
    Función que crea una red siguiendo la arquitectura Ghosh pero con las características introducidas como parámetros.
    
    Parámetros
    --------------------------------------------------------------------------
    capas_conv: número entero que puede tomar 3 posibles valores (3,5 o 7) y que representa el número de capas convolucionales que tiene la red.
    filtros: float que representa el número de filtros por capa convolucional. Puede ser 1.0 si conserva el número original, 0.5 si lo divide a la mitad y 2.0 si lo duplica.
    neuronas: String que contiene el número de neuronas de las capas fully-connected separados por barras laterales (/).
    
    Return
    --------------------------------------------------------------------------
    modelo: devuelve una instancia de la clase Rajagopalan con las características arquitectónicas deseadas, es decir, un modelo de CNN con las características indicadas en los parámetros.
    '''
    
    #Debido a las distintas variaciones que se van a producir en la arquitectura, el número de características (y por tanto de neuronas) variará.
    #Es por ello que para simplificar esta labor voy a definir 2 funciones que permiten calcular el número de características resultantes tras las capas convolucionales y de MaxPooling.
    #De esta manera se podrá pasar este valor como parámetro in_features a la primera capa fully-connected.

    #Sabemos que el número de características tras una capa convolucional se corresponde con la siguiente fórmula:
    #    features = num_filtros*ancho*alto
    #Tal y como se describe en el capítulo "Convolutional Neural Networks" en el libro "Deep Learning" de Ian Goodfellow, Yoshua Bengio y Aaron Courville

    #El número de filtros lo podemos obtener de la última capa convolucional, pero el ancho y alto de la imagen variarán tras cada capa.
    #Las dimensiones modificadas tras una capa convolucional y de maxpooling se pueden calcular gracias a la siguiente ecuación:
    #    output_size = (input_size-2*padding-kernel_size)/stride + 1
    #Tal y como se describe en el capítulo 3 "Convolutional Neural Networks" del libro "Deep Learning for Computer Vision" de Rajalingappaa Shanmugamani

    #Sabiendo esto ya podemos definir las funciones que permiten calcular las dimensiones
    def funcion(input_size,kernel_size,stride,padding):
        '''
        Aplica la ecuación para calcular las dimensiones tras una capa convolucional/maxpooling descrita en el libro Deep Learning for Computer Vision.

        Parámetros
        ----------------------------------------------------
        input_size: número entero que se corresponde con la escala de la imagen inicial, previa a la capa convolucional.
        kernel_size: número entero que representa el tamaño del filtro de la capa.
        stride:  número entero que representa el desplazamiento del filtro sobre la imagen.
        padding: número entero correspondiente al número de píxeles de relleno.

        Return
        ----------------------------------------------------
        Devuelve un número flotante correspondiente al tamaño de la imagen una vez modificada por la capa convolucional/maxpooling.
        '''
        #aplicamos la fórmula
        return ((input_size + 2*padding - kernel_size)/stride + 1)

    def calcula_dim(num_capas):
        '''
        Calcula la escala de una imagen tras haber sido transformada por n capas convolucionales y de maxpooling, según las características de la arquitectura Ghosh.

        Parámetros
        ----------------------------------------------------
        num_capas:

        Return
        ----------------------------------------------------
        Devuelve un número entero correspondiente a la escala de la imagen después de todas las capas convolucionales y de maxpooling.
        '''
        #el tamaño inicial de las imágenes es de 512x512 (la escala original)
        size = 224

        #vamos actualizando el tamaño de la imagen según vaya atravesando capas convolucionales y de maxpooling
        size = funcion(size,9,4,0)#tras primera capa convolucional
        size = funcion(size,2,2,0)#tras maxpooling
        size = funcion(size,7,1,0)#tras segunda capa convolucional
        size = funcion(size,2,2,0)#tras maxpooling
        size = funcion(size,5,1,0)#tras tercera capa convolucional
        
        if num_capas > 3:
            size = funcion(size,3,1,0)#tras cuarta capa convolucional
            size = funcion(size,3,1,0)#tras quinta capa convolucional
            
            if num_capas >5:
                size = funcion(size,3,1,0)#tras sexta capa convolucional
                size = funcion(size,3,1,0)#tras séptima capa convolucional
        
        return int(size)

    #primero definimos la clase correspondiente (Rajagopalan en este caso), incluyendo los elementos necesarios para obtener las variaciones deseadas
    class Rajagopalan(nn.Module):
        #esta estructura está formada por capas convolucionales, de maxpooling, de activación, de Dropout, fully-connected y de clasificación

        def __init__(self):
            #sobreescribimos el constructor del padre
            super(Rajagopalan,self).__init__()
            #primero definimos una capa convolucional
            #el número de filtros de cada capa irá multiplicado por el parámetro filtros (para reducirlo, duplicarlo o mantenerlo)
            self.conv1 = nn.Conv2d(
                in_channels = 3, #3 canales de entrada porque las imágenes son a color
                out_channels = int(64*filtros), #se trata del número de salidas de la capa. Es el número de kernels de la capa convolucional
                kernel_size = 9, #suele tratarse de un número impar
                stride = 4, #cantidad píxeles que se desplaza el filtro sobre la imagen
                padding = 0, #cantidad de relleno que se va a aplicar sobre los bordes de la imagen
            )

            #una segunda convolucional
            self.conv2 = nn.Conv2d(
                in_channels = int(64*filtros), #64 canales de entrada porque es el número de salidas de la capa anterior
                out_channels = int(128*filtros), #se trata del número de salidas de la capa. Es el número de kernels de la capa convolucional
                kernel_size = 7, #suele tratarse de un número impar
                stride = 1, #cantidad píxeles que se desplaza el filtro sobre la imagen
                padding = 0, #cantidad de relleno que se va a aplicar sobre los bordes de la imagen
            )

            #la tercera convolucional
            self.conv3 = nn.Conv2d(
                in_channels = int(128*filtros), #128 canales de entrada porque es el número de salidas de la capa anterior
                out_channels = int(256*filtros), #se trata del número de salidas de la capa. Es el número de kernels de la capa convolucional
                kernel_size = 5, #suele tratarse de un número impar
                stride = 1, #cantidad píxeles que se desplaza el filtro sobre la imagen
                padding = 0, #cantidad de relleno que se va a aplicar sobre los bordes de la imagen
            )

            if capas_conv >3:
                #cuarta convolucional
                self.conv4 = nn.Conv2d(
                    in_channels = int(256*filtros), #256 canales de entrada porque es el número de salidas de la capa anterior
                    out_channels = int(384*filtros), #se trata del número de salidas de la capa. Es el número de kernels de la capa convolucional
                    kernel_size = 3, #suele tratarse de un número impar
                    stride = 1, #cantidad píxeles que se desplaza el filtro sobre la imagen
                    padding = 0, #cantidad de relleno que se va a aplicar sobre los bordes de la imagen
                )

                #quinta y última capa convolucional
                self.conv5 = nn.Conv2d(
                    in_channels = int(384*filtros), #256 canales de entrada porque es el número de salidas de la capa anterior
                    out_channels = int(256*filtros), #se trata del número de salidas de la capa. Es el número de kernels de la capa convolucional
                    kernel_size = 3, #suele tratarse de un número impar
                    stride = 1, #cantidad píxeles que se desplaza el filtro sobre la imagen
                    padding = 0, #cantidad de relleno que se va a aplicar sobre los bordes de la imagen
                )

                if capas_conv >5:
                    self.conv6 = nn.Conv2d(
                        in_channels = int(256*filtros), #256 canales de entrada porque es el número de salidas de la capa anterior
                        out_channels = int(128*filtros), #se trata del número de salidas de la capa. Es el número de kernels de la capa convolucional
                        kernel_size = 2, #suele tratarse de un número impar
                        stride = 1, #cantidad píxeles que se desplaza el filtro sobre la imagen
                        padding = 0, #cantidad de relleno que se va a aplicar sobre los bordes de la imagen
                    )

                    self.conv7 = nn.Conv2d(
                        in_channels = int(128*filtros), #128 canales de entrada porque es el número de salidas de la capa anterior
                        out_channels = int(128*filtros), #se trata del número de salidas de la capa. Es el número de kernels de la capa convolucional
                        kernel_size = 1, #suele tratarse de un número impar
                        stride = 1, #cantidad píxeles que se desplaza el filtro sobre la imagen
                        padding = 0, #cantidad de relleno que se va a aplicar sobre los bordes de la imagen
                    )
                    
            #usando la función descrita anteriormente y la ecuación features = num_filtros*ancho*alto podemos calcular el número de características
            #si el número de capas convolucionales es 3, el número de filtros será 256 multiplicado por el parámetro filtros 
            if capas_conv == 3:
                neuronas_entrada = int(256*filtros)*calcula_dim(capas_conv)*calcula_dim(capas_conv)
            #si el número de capas convolucionales es 5, el número de filtros será 256 multiplicado por el parámetro filtros
            elif capas_conv == 5:
                neuronas_entrada = int(256*filtros)*calcula_dim(capas_conv)*calcula_dim(capas_conv)
            #si el número de capas convolucionales es 7, el número de filtros será 128 multiplicado por el parámetro filtros
            else:
                neuronas_entrada = int(128*filtros)*calcula_dim(capas_conv)*calcula_dim(capas_conv)

            #definimos también la función de MaxPooling que se aplicará sobre algunas de las capas convolucionales
            self.pool = nn.MaxPool2d(
                kernel_size = 2, #establecemos el tamaño del kernel a 2*2
                stride = 2 #cantidad píxeles que se desplaza el filtro sobre la imagen (por defecto se desplazará el tamaño del kernel)
            )

            #y las capas de neuronas fully-connected
            self.fc1 = nn.Linear(
                in_features = neuronas_entrada, #número de parámetros de entrada de la red (los valores se obtienen experimentalmente)
                out_features = int(neuronas.split('/')[0]) #número de neuronas de salida, obtenidas del parámetro pasado
            )

            self.fc2 = nn.Linear(int(neuronas.split('/')[0]),int(neuronas.split('/')[1]))

            #y por último la capa encargada de realizar las predicciones
            self.dense = nn.Linear(int(neuronas.split('/')[1]),5)#tiene 5 neuronas de salida, una para cada clase de nuestro problema
            
        def forward(self,x):
            #en esta función es donde tiene lugar la computación (y la función invocada por defecto al ejecutar la red)
            #siguiendo la estructura descrita en Ghosh et al. (con sus respectivas variaciones):
            
            #primero una capa convolucional con activación ReLU y maxpooling
            x = self.pool(F.relu(self.conv1(x)))
            #una segunda capa convolucional con activación y maxpooling
            x = self.pool(F.relu(self.conv2(x)))
            #a continuación 3 capas convolucionales SIN maxpooling (pero sí activación ReLU)
            x = F.relu(self.conv3(x))
            if capas_conv >3:
                x = F.relu(self.conv4(x))
                x = F.relu(self.conv5(x))
                if capas_conv >5:
                    x = F.relu(self.conv6(x))
                    x = F.relu(self.conv7(x))
                
            #aplanamos la salida usando la función view para convertir las dimensiones de los datos
            x = x.view(-1,self.num_flat_features(x))#usamos una función propia de la clase para obtener el número de características
            #posteriormente tienen lugar las dos capas fully-connected
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            #y por último la capa densa que va a proporcionarnos la predicción
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

    #por último creamos una instancia de esta red
    modelo = Rajagopalan()
    #y la devolvemos
    return modelo

#para cada combinacion de los parámetros creamos el modelo
modelo = crea_Rajagopalan(capas,n_filtros,n_neuronas)
#definimos como loss la función de tipo cross entropy 
criterion = nn.CrossEntropyLoss() 
#en este caso el optimizador será la función Adam (ampliamente utilizada)
optimizer = torch.optim.Adam(params = modelo.parameters()) #dejamos el valor de learning-rate por defecto (0.001)
#previo al entrenamiento imprimimos por pantalla las características de la red, para poder identificar su entrenamiento
print('--------------------------------------')
print(f'Entrenamiento. Características:\n  -Capas:{capas}\n  -Filtros:{n_filtros}\n  -Neuronas:{n_neuronas}\n  -Early Stopping\n')
#capturamos el tiempo previo al entrenamiento
inicio = time.time()
#entrenamos la red con 7 épocas de paciencia en Early Stopping y guardamos los valores para poder representar las gráficas
acc,loss,val_acc,val_loss = entrena_val(modelo,epocas,7,train_loader,val_loader,optimizer,criterion)
#y el tiempo tras el entrenamiento
fin = time.time()

#guardamos las gráficas
guarda_graficas('Datasets','Si','No','No','RGB','Rajagopalan',capas,n_filtros,n_neuronas,acc,loss,val_acc,val_loss)

#ponemos a prueba la red con el conjunto de iPhone usando la función tester y recogemos los resultados para obtener las métricas
y_true_iphone, y_pred_iphone, predictions_iphone = tester(modelo,test_i_loader)
#obtenemos las métricas usando la función importada obtiene_metricas, que no las muestra por pantalla
metricas_iphone = obtiene_metricas(y_true_iphone, y_pred_iphone, predictions_iphone)
#las mostramos por pantalla
print('\n--------------------------------------')
print(f'Test. Características:\n  -Capas:{capas}\n  -Filtros:{n_filtros}\n  -Neuronas:{n_neuronas}\n  -Test:iphone')
print(f' - Matriz de confusión:\n{metricas_iphone[0]}\n - Accuracy:{metricas_iphone[1]}\n - Balanced accuracy:{metricas_iphone[2]}\n - F-score:{metricas_iphone[3]}\n - Kappa:{metricas_iphone[4]}\n - AUC:{metricas_iphone[5]}\n - Tiempo:{(fin-inicio)/60} mins')
#escribimos las métricas (a excepción de la matriz de confusión) en el archivo Resultados.csv previamente creado
with open('Resultados.csv','a') as fd:
    fd.write('\n')
    fd.write(f'Datasets,Sí,No,No,RGB,Rajagopalan,{capas},{n_filtros},{n_neuronas},iphone,{metricas_iphone[1]},{metricas_iphone[2]},{metricas_iphone[3]},{metricas_iphone[4]},{metricas_iphone[5]},{(fin-inicio)/60}')


#ahora ponemos a prueba la red con el conjunto de Samsung usando la función tester y recogemos los resultados para obtener las métricas
y_true_samsung, y_pred_samsung, predictions_samsung = tester(modelo,test_S_loader)
#obtenemos las métricas usando la función importada obtiene_metricas, que no las muestra por pantalla
metricas_samsung = obtiene_metricas(y_true_samsung, y_pred_samsung, predictions_samsung)
#las mostramos por pantalla
print('\n--------------------------------------')
print(f'Test. Características:\n  -Capas:{capas}\n  -Filtros:{n_filtros}\n  -Neuronas:{n_neuronas}\n  -Test:Samsung')
print(f' - Matriz de confusión:\n{metricas_samsung[0]}\n - Accuracy:{metricas_samsung[1]}\n - Balanced accuracy:{metricas_samsung[2]}\n - F-score:{metricas_samsung[3]}\n - Kappa:{metricas_samsung[4]}\n - AUC:{metricas_samsung[5]}\n - Tiempo:{(fin-inicio)/60} mins')
#escribimos las métricas (a excepción de la matriz de confusión) en el archivo Resultados.csv previamente creado
with open('Resultados.csv','a') as fd:
    fd.write('\n')
    fd.write(f'Datasets,Sí,No,No,RGB,Rajagopalan,{capas},{n_filtros},{n_neuronas},Samsung,{metricas_samsung[1]},{metricas_samsung[2]},{metricas_samsung[3]},{metricas_samsung[4]},{metricas_samsung[5]},{(fin-inicio)/60}')

#por último vamos a guardar el modelo, sus pesos y estado actual, por si se quisiera volver a emplear
#primero para ello debemos cambiar el String de filtros y neuronas para evitar los puntos y barras laterales
filtros_str = str(n_filtros).replace(".","punto")
neuronas_str = str(n_neuronas).replace("/","slash")
torch.save(modelo.state_dict(), f'modelos/Rajagopalan/Datasets_Sival_Noprep_Noinp_RGB_{capas}_{filtros_str}_{neuronas_str}.pth')


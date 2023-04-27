'''
#########################################################################################################################
INFORMACIÓN DEL FICHERO
#########################################################################################################################

Autor: Samuel Lozano Juárez
Fecha: 27/03/2023
Institución: UBU | Grado en Ingeniería de la Salud

Este archivo forma parte del Trabajo de Fin de Grado "Detección del grado de retinopatía mediante redes convolucionales".
El alumno a cargo de este proyecto es el declarado como autor en las líneas anteriores.
Los tutores del proyecto fueron el Dr. Darío Fernández Zoppino y el Dr. Daniel Urda Muñoz.

A continuación se incluye el código que permite crear varios modelos según la arquitectura propuesta en el artículo de Alqudah, pero realizando las modificaciones
deseadas en los parámetros (número de capas convolucionales de la arquitectura, número de filtros por capa y número de neuronas de las capas fully-connected).
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
from modules.CNN_utilities import entrena, representa_test, obtiene_metricas, tester, guarda_graficas

#importamos el paquete que permite calcular el tiempo de entrenamiento
import time

#establecemos el tamaño del batch, la escala de las imágenes y el número de épocas de entrenamiento
batch = 4
#en la arquitectura propuesta por Mobeen no se especifica ninguna escala, por lo que se empleará una escala cualquiera (512 por ejemplo)
escala = 256
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
# - número de capas convolucionales (2, 4 o 6)
# - número de filtros por capa (conservando los originales, reduciéndolos a la mitad o multiplicándolos por dos)
# - número de neuronas de las capas fully-connected, probando las siguientes combinaciones:
#    * ninguna capa fully-connected
#    * 512/256
#    * 128/64
#    * 64/128
#    * 128/256
# Por tanto el número total de posibles combinaciones es 3*3*5 = 45 combinaciones

#Para facilitar la lectura del código y sobre todo su ejecución, voy a definir una función que permita lanzar las ejecuciones necesarias de manera automática (de forma similar a como se hizo en las variaciones anteriores).

def crea_Alqudah(capas_conv, filtros, neuronas):
    '''
    Función que crea una red siguiendo la arquitectura Alqudah pero con las características introducidas como parámetros.
    
    Parámetros
    --------------------------------------------------------------------------
    capas_conv: número entero que puede tomar 3 posibles valores (2, 4 o 6) y que representa el número de capas convolucionales que tiene la red.
    filtros: float que representa el número de filtros por capa convolucional. Puede ser 1.0 si conserva el número original, 0.5 si lo divide a la mitad y 2.0 si lo duplica.
    neuronas: String que contiene el número de neuronas de las capas fully-connected separados por barras laterales (/). Puede que incluya 0/0, indicando que no se incluyen capas fully-connected en la instancia del modelo.
    
    Return
    --------------------------------------------------------------------------
    modelo: devuelve una instancia de la clase Alqudah con las características arquitectónicas deseadas, es decir, un modelo de CNN con las características indicadas en los parámetros.
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
        Calcula la escala de una imagen tras haber sido transformada por n capas convolucionales y de maxpooling, según las características de la arquitectura Mobeen.

        Parámetros
        ----------------------------------------------------
        num_capas:

        Return
        ----------------------------------------------------
        Devuelve un número entero correspondiente a la escala de la imagen después de todas las capas convolucionales y de maxpooling.
        '''
        #el tamaño inicial de las imágenes es de 512x512 (la escala original)
        size = 256
        
        #vamos actualizando el tamaño de la imagen según vaya atravesando capas convolucionales y de maxpooling
        size = funcion(size,3,1,0)#tras primera capa convolucional
        size = funcion(size,2,2,0)#tras pooling
        size = funcion(size,3,1,0)#tras segunda capa convolucional
        size = funcion(size,2,2,0)#tras pooling
        if num_capas > 2:
            size = funcion(size,3,1,0)#tras tercera capa convolucional
            size = funcion(size,2,2,0)#tras pooling
            size = funcion(size,3,1,0)#tras cuarta capa convolucional
            if num_capas > 4:
                size = funcion(size,3,1,0)#tras quinta capa convolucional
                size = funcion(size,3,1,0)#tras sexta capa convolucional
                
        return int(size)

    #primero definimos la clase correspondiente (Mobeen en este caso), incluyendo los elementos necesarios para obtener las variaciones deseadas
    class Alqudah(nn.Module):
        #esta estructura está formada por capas convolucionales, de maxpooling, de activación, de Dropout, fully-connected y de clasificación

        def __init__(self):
            #sobreescribimos el constructor del padre
            super(Alqudah,self).__init__()
            #creamos la primera capa convolucional
            #el número de filtros de cada capa irá multiplicado por el parámetro filtros (para reducirlo, duplicarlo o mantenerlo)
            
            #definimos la primera capa convolucional
            self.conv1 = nn.Conv2d(
                in_channels = 3, #3 canales de entrada porque las imágenes son a color
                out_channels = int(32*filtros), #se trata del número de salidas de la capa. Es el número de kernels de la capa convolucional
                kernel_size = 3, #suele tratarse de un número impar
                stride = 1, #cantidad píxeles que se desplaza el filtro sobre la imagen (lo dejamos por defecto ya que el artículo no dice nada)
                padding = 0, #cantidad de relleno que se va a aplicar sobre los bordes de la imagen
            )

            #segunda capa convolucional
            self.conv2 = nn.Conv2d(
                in_channels = int(32*filtros), #32 canales de entrada porque es el número de salidas de la capa anterior
                out_channels = int(16*filtros), #se trata del número de salidas de la capa. Es el número de kernels de la capa convolucional
                kernel_size = 3, #suele tratarse de un número impar
                stride = 1, #cantidad píxeles que se desplaza el filtro sobre la imagen
                padding = 0, #cantidad de relleno que se va a aplicar sobre los bordes de la imagen
            )
            if capas_conv > 2:
                #tercera capa convolucional
                self.conv3 = nn.Conv2d(
                    in_channels = int(16*filtros), #16 canales de entrada porque es el número de salidas de la capa anterior
                    out_channels = int(8*filtros), #se trata del número de salidas de la capa. Es el número de kernels de la capa convolucional
                    kernel_size = 3, #suele tratarse de un número impar
                    stride = 1, #cantidad píxeles que se desplaza el filtro sobre la imagen
                    padding = 0, #cantidad de relleno que se va a aplicar sobre los bordes de la imagen
                )

                #cuarta capa convolucional
                self.conv4 = nn.Conv2d(
                    in_channels = int(8*filtros), #8 canales de entrada porque es el número de salidas de la capa anterior
                    out_channels = int(16*filtros), #se trata del número de salidas de la capa. Es el número de kernels de la capa convolucional
                    kernel_size = 3, #suele tratarse de un número impar
                    stride = 1, #cantidad píxeles que se desplaza el filtro sobre la imagen
                    padding = 0, #cantidad de relleno que se va a aplicar sobre los bordes de la imagen
                )
                
                if capas_conv > 4:
                    #quinta capa convolucional
                    self.conv5 = nn.Conv2d(
                        in_channels = int(16*filtros), #16 canales de entrada porque es el número de salidas de la capa anterior
                        out_channels = int(32*filtros), #se trata del número de salidas de la capa. Es el número de kernels de la capa convolucional
                        kernel_size = 3, #suele tratarse de un número impar
                        stride = 1, #cantidad píxeles que se desplaza el filtro sobre la imagen
                        padding = 0, #cantidad de relleno que se va a aplicar sobre los bordes de la imagen
                    )

                    #sexta y última capa convolucional
                    self.conv6 = nn.Conv2d(
                        in_channels = int(32*filtros), #32 canales de entrada porque es el número de salidas de la capa anterior
                        out_channels = int(16*filtros), #se trata del número de salidas de la capa. Es el número de kernels de la capa convolucional
                        kernel_size = 3, #suele tratarse de un número impar
                        stride = 1, #cantidad píxeles que se desplaza el filtro sobre la imagen
                        padding = 0, #cantidad de relleno que se va a aplicar sobre los bordes de la imagen
                    )
                    
            #definimos también la capa de MaxPooling
            self.pool = nn.MaxPool2d(
                kernel_size = 2, #establecemos el tamaño del kernel a 2*2
                stride = 2 #cantidad píxeles que se desplaza el filtro sobre la imagen
            )

            #como novedad se introduce un nuevo tipo de capa: BatchNormalization
            self.batch_norm1 = nn.BatchNorm2d(
                num_features = int(32*filtros) #número de características a normalizar (salidas de la capa a normalizar en este caso)
            )

            self.batch_norm2_4 = nn.BatchNorm2d(
                num_features = int(16*filtros) #número de características a normalizar (salidas de la capa a normalizar en este caso)
            )

            self.batch_norm3 = nn.BatchNorm2d(
                num_features = int(8*filtros) #número de características a normalizar (salidas de la capa a normalizar en este caso)
            )
            
            #usando la función descrita anteriormente y la ecuación features = num_filtros*ancho*alto podemos calcular el número de características
            #sea cual sea el número de capas convolucionales el número de filtros de salida será 16*filtros
            neuronas_entrada = int(16*filtros)*calcula_dim(capas_conv)*calcula_dim(capas_conv)             
                        
            #definimos las capas fully-connected que correspondan según los parámetros introducidos
            if int(neuronas.split('/')[0]) != 0:
                self.fc1 = nn.Linear(
                    in_features = neuronas_entrada, #número de características de entrada
                    out_features = int(neuronas.split('/')[0]) #número de neuronas de salida, obtenidas del parámetro pasado
                )
                
                #la segunda capa fully-connected
                self.fc2 = nn.Linear(int(neuronas.split('/')[0]),int(neuronas.split('/')[1]))
                
                #actualizamos el valor de las neuronas de entrada para la siguiente capa (la densa de salida)
                neuronas_entrada = int(neuronas.split('/')[1])
                
            #por último definimos la capa de neuronas fully-connected, con 5 neuronas de salida (una por clase del problema)
            self.dense = nn.Linear(
                in_features = neuronas_entrada,
                out_features = 5
            )

        def forward(self,x):
            #en esta función es donde tiene lugar la computación (y la función invocada por defecto al ejecutar la red)
            #siguiendo la estructura descrita en Mobeen et al. (con sus respectivas variaciones):
            #primero una capa convolucional con Batch Normalization, activación ReLU y MaxPooling
            x = self.pool(F.relu(self.batch_norm1(self.conv1(x))))
            #la misma secuencia pero variando las características de la capa convolucional y de batch normalization
            x = self.pool(F.relu(self.batch_norm2_4(self.conv2(x)))) 
            if capas_conv > 2:
                #lo mismo para la tercera capa
                x = self.pool(F.relu(self.batch_norm3(self.conv3(x))))
                #y en la última capa convolucional no tiene lugar el MaxPooling (la activación y Batch Normalization sí)
                x = F.relu(self.batch_norm2_4(self.conv4(x)))
                if capas_conv >4:
                    #las últimas 2 capas no incorporan ni pooling ni batch normalization
                    x = F.relu(self.conv5(x))
                    #y en la última capa convolucional no tiene lugar el MaxPooling (la activación y Batch Normalization sí)
                    x = F.relu(self.conv6(x))
            #aplanamos la salida para que pueda ser utilizada por la capa de neuronas fully-connected
            x = x.view(-1,self.num_flat_features(x))#usamos una función propia de la clase para obtener el número de características
            #definimos las capas fully-connected  si fueran necesarias
            if int(neuronas.split('/')[0]) != 0:
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
            #por último la capa densa 
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
    modelo = Alqudah()
    #y la devolvemos
    return modelo

#una vez definida dicha función, debemos generar las iteraciones necesarias para crear todas las redes resultantes de las combinaciones
#creamos un bucle para cada una de las características, uno por cada posible valor del número de capas
for capas in [2,4,6]:
    #otro para que recorra los posibles valores del parámetro filtros
    for n_filtros in [1.0,0.5,2.0]:
        #y uno último para el número de neuronas
        for n_neuronas in ['0/0','512/256','128/64','64/128','128/256']:
            #para cada combinacion de los parámetros creamos el modelo
            modelo = crea_Alqudah(capas,n_filtros,n_neuronas)
            #definimos como loss la función de tipo cross entropy 
            criterion = nn.CrossEntropyLoss() 
            #en este caso el optimizador será la función Adam (ampliamente utilizada)
            optimizer = torch.optim.Adam(params = modelo.parameters()) #dejamos el valor de learning-rate por defecto (0.001)
            #previo al entrenamiento imprimimos por pantalla las características de la red, para poder identificar su entrenamiento
            print('--------------------------------------')
            print(f'Entrenamiento. Características:\n  -Capas:{capas}\n  -Filtros:{n_filtros}\n  -Neuronas:{n_neuronas}\n')
            #capturamos el tiempo previo al entrenamiento
            inicio = time.time()
            #entrenamos la red y guardamos los valores para poder representar las gráficas
            acc,loss = entrena(modelo,epocas,train_loader,optimizer,criterion)
            #y el tiempo posterior al entrenamiento
            fin = time.time()
            
            #guardamos las gráficas
            guarda_graficas('OCT','No','No','No','RGB','Alqudah',capas,n_filtros,n_neuronas,acc,loss)
            
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
                fd.write(f'OCT,No,No,No,RGB,Alqudah,{capas},{n_filtros},{n_neuronas},iphone,{metricas_iphone[1]},{metricas_iphone[2]},{metricas_iphone[3]},{metricas_iphone[4]},{metricas_iphone[5]},{(fin-inicio)/60}')
                
                
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
                fd.write(f'OCT,No,No,No,RGB,Alqudah,{capas},{n_filtros},{n_neuronas},Samsung,{metricas_samsung[1]},{metricas_samsung[2]},{metricas_samsung[3]},{metricas_samsung[4]},{metricas_samsung[5]},{(fin-inicio)/60}')
                
            #por último vamos a guardar el modelo, sus pesos y estado actual, por si se quisiera volver a emplear
            #primero para ello debemos cambiar el String de filtros y neuronas para evitar los puntos y barras laterales
            filtros_str = str(n_filtros).replace(".","punto")
            neuronas_str = str(n_neuronas).replace("/","slash")
            torch.save(modelo.state_dict(), f'modelos/Alqudah/OCT_Noval_Noprep_Noinp_RGB_{capas}_{filtros_str}_{neuronas_str}.pth')
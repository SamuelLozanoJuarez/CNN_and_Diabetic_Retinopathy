'''
#########################################################################################################################
INFORMACIÓN DEL FICHERO
#########################################################################################################################
Autor: Samuel Lozano Juárez
Fecha: 13/02/2023
Institución: UBU | Grado en Ingeniería de la Salud

Este archivo forma parte del Trabajo de Fin de Grado "Detección del grado de retinopatía mediante redes convolucionales".
El alumno a cargo de este proyecto es el declarado como autor en las líneas anteriores.
Los tutores del proyecto fueron el Dr. Darío Fernández Zoppino y el Dr. Daniel Urda Muñoz.

A continuación se incluye el código necesario para la organización de las imágenes obtenidas del repositorio Kaggle. Las imágenes crudas, sin clasificar,
pueden descargarse en el siguiente enlace: https://www.kaggle.com/competitions/diabetic-retinopathy-detection/data .
En esa misma dirección se encuentra un fichero .csv que contiene la etiqueta de cada imagen. Ese fichero y esa etiqueta son los empleados en este código para la 
separación de las imágenes en los distintos directorios.
'''

#importamos los paquetes necesarios
import pandas as pd
import os
import shutil

#leemos el .csv que contiene las labels de las imágenes de Kaggle
etiquetas = pd.read_csv('Raw Data/Kaggle/trainLabels.csv')

#definimos una función que permite mover todas lass imágenes de un directorio a la carpeta que corresponda según su etiqueta
def mover(carpeta):
    '''
    Mueve las imágenes de una carpeta a las distintas carpetas clasificando las imágenes según su grado.
    El grado se obtiene del fichero .csv, que contiene la relación imagen-etiqueta.
    
    Parámetros
    ----------------------------
    carpeta: string que indica la carpeta que se desea explorar y organizar. Puede tomar 5 valores: train_1, train_2, train_3, train_4 o train_5.
    
    Return
    ----------------------------
    No devuelve ningún valor.
    '''
    #obtenemos la lista de imágenes que se encuentran en la carpeta
    lista = os.listdir('Raw Data/Kaggle/' + carpeta)
    #para cada imagen de la carpeta train_1 vamos a obtener su clase a partir del nombre de la image (sin la extensión .jpg)
    for imagen in lista:
        clase = etiquetas[etiquetas['image'] == imagen.split('.')[0]]['level'].values[0] + 1 #sumamos 1 para que las labels coincidan con las nuestras (1 a 5 y no 0 a 4 como aparece en el csv)
        #convertimos la etiqueta a formato directorio (G1, G2, G3, G4, G5)
        clase = 'G' + str(clase)
        #movemos la imagen de la carpeta original a la que corresponda
        shutil.move('Raw Data/Kaggle/' + carpeta + '/' + imagen, 'Classified Data/Images/Kaggle/' + clase + '/')


#por último llamamos a la función para cada una de las 5 carpetas de Kaggle
mover('train_1')
mover('train_2')
mover('train_3')
mover('train_4')
mover('train_5')
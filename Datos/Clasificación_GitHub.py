'''
#########################################################################################################################
INFORMACIÓN DEL FICHERO
#########################################################################################################################
Autor: Samuel Lozano Juárez
Fecha: 16/02/2023
Institución: UBU | Grado en Ingeniería de la Salud

Este archivo forma parte del Trabajo de Fin de Grado "Detección del grado de retinopatía mediante redes convolucionales".
El alumno a cargo de este proyecto es el declarado como autor en las líneas anteriores.
Los tutores del proyecto fueron el Dr. Darío Fernández Zoppino y el Dr. Daniel Urda Muñoz.

A continuación se incluye el código necesario para la organización de las imágenes obtenidas de GitHub. Las imágenes crudas, sin clasificar,
pueden descargarse en el siguiente enlace: https://github.com/deepdrdoc/DeepDRiD/tree/master/regular_fundus_images .

Las imágenes se encuentran divididas en 3 carpetas: training, validation y evaluation, y dentro de cada carpeta encontramos un fichero
.csv o .xlsx que contiene la relación entre las imágenes y su etiqueta, y un conjunto de directorios (cada uno correspondiente a un 
paciente) que son los que contienen las imágenes (2 por cada ojo).

Nosotros vamos a utilizar todas las imágenes (independientemente de que sean para validation, evaluation o training) como imágenes de 
train. Las clasificaremos en los 5 niveles posibles según el valor de la etiqueta en el fichero .csv o .xlsx.
'''

#en este caso tenemos 3 ficheros con imágenes y etiquetas: validation, training y evaluation
#nosotros vamos a usar todas las imágenes como train (y ya hará el split de validación la red)

#primero realizamos las importaciones necesarias
import pandas as pd
import os
import shutil

#a continuación vamos a extraer las imágenes de sus directorios
#comenzamos con el conjunto de training
folder = 'Raw Data/GitHub/training/Images'
for i in os.listdir(folder):
    if os.path.isdir(i):
        subfolder = os.listdir(folder + '/' + i)
        for j in subfolder:
            shutil.move(folder + '/' + i + '/' + j, folder)

#repetimos el mismo proceso para el conjunto de validation
folder = 'Raw Data/GitHub/validation/Images'
for i in os.listdir(folder):
    if os.path.isdir(i):
        subfolder = os.listdir(folder + '/' + i)
        for j in subfolder:
            shutil.move(folder + '/' + i + '/' + j, folder)

#y lo mismo para el conjunto de evaluation
folder = 'Raw Data/GitHub/evaluation/Images'
for i in os.listdir(folder):
    if os.path.isdir(i):
        subfolder = os.listdir(folder + '/' + i)
        for j in subfolder:
            shutil.move(folder + '/' + i + '/' + j, folder)

#para clasificar las imágenes vamos a cargar los .csv y .xlsx que contienen la correspondencia imagen-etiqueta
labels_training = pd.read_csv('Raw Data/GitHub/training/regular-fundus-training.csv')
#nos quedamos solo con las columnas de interés (el id de la imagen y el diagnóstico de cada ojo)
labels_training = labels_training[['image_id','left_eye_DR_Level','right_eye_DR_Level']]

#definimos una función que permita combinar las dos columnas de diagnóstico en una única
def combina(row):
    '''
    Para una fila dada como parámetro, comprueba qué columna de las dos de diganóstico es la que contiene un valor distinto de NaN
    y devuelve el valor de esa columna.
    
    Parámetros
    -----------------------------------
    row: se trata de una fila de un dataframe (un objeto tipo pandas.Series)
    
    Return
    -----------------------------------
    Devuelve un valor, en este caso numérico de tipo entero, correspondiente al grado de la imagen de la fila parámetro.
    '''
    #comprueba si el diagnóstico de ojo izquierdo es NaN
    if pd.isna(row['left_eye_DR_Level']):
        #si es así devuelve el diagnóstico del ojo derecho (asumiendo que este es el que contiene el valor numérico)
        return int(row['right_eye_DR_Level'])
    #si no es así devuelve el diagnóstico del ojo izquierdo pues ya ha comprobado que no es NaN
    else:
        return int(row['left_eye_DR_Level'])

#y creamos una columna única que contenga los diagnósticos
#usamos la función apply, que sigue un funcionamiento similar un map(), aplicando una función sobre todas las filas de un dataframe
labels_training['grado'] = labels_training.apply(lambda row: combina(row), axis=1)

#a continuación vamos a recorrer todas las imágenes de training, comprobar cuál es su grado y en base a ello clasificarlas
for i in os.listdir('Raw Data/GitHub/training/Images'):
    #comprobamos si el último caracter del nombre es una 'g', para iterar así solo sobre las imágenes (.jpg)
    if i[-1] == 'g':
        #a partir del nombre de la imagen, eliminamos la extensión .jpg y buscamos en el dataframe el valor del grado correspondiente a ese identificador
        grado = labels_training[labels_training['image_id'] == i.split('.')[0]]['grado'].values[0]
        #sabemos que si el grado de la imagen es 5, esa imagen debe ser descartada porque la calidad es inadecuada
        #por ello únicamente nos quedaremos con las imágenes de grado distinto a 5
        if grado != 5:
            #para las imágenes de grado distinto de 5 vamos a convertir el grado en etiqueta, sumándole 1 y convirtiéndolo a string
            etiqueta = 'G' + str(grado + 1)
            #posteriormente movemos la imagen a la carpeta que corresponda
            shutil.move('Raw Data/GitHub/training/Images/' + i, 'Classified Data/Images/GitHub/' + etiqueta)


#debemos repetir el proceso para el conjunto de imágenes de validation
#primero cargamos el fichero csv que contiene la relación entre imagen y etiqueta
labels_validation = pd.read_csv('Raw Data/GitHub/validation/regular-fundus-validation.csv')
#nos quedamos solo con las columnas de interés (el id de la imagen y el diagnóstico de cada ojo)
labels_validation = labels_validation[['image_id','left_eye_DR_Level','right_eye_DR_Level']]

#al igual que en el caso de training debemos crear la columna que unifique los grados
labels_validation['grado'] = labels_validation.apply(lambda row: combina(row), axis=1)

#por último seguimos el esquema empleado en training para mover las imágenes a la carpeta correspondiente
for i in os.listdir('Raw Data/GitHub/validation/Images'):
    #comprobamos si el último caracter del nombre es una 'g', para iterar así solo sobre las imágenes (.jpg)
    if i[-1] == 'g':
        #a partir del nombre de la imagen, eliminamos la extensión .jpg y buscamos en el dataframe el valor del grado correspondiente a ese identificador
        grado = labels_validation[labels_validation['image_id'] == i.split('.')[0]]['grado'].values[0]
        #sabemos que si el grado de la imagen es 5, esa imagen debe ser descartada porque la calidad es inadecuada
        #por ello únicamente nos quedaremos con las imágenes de grado distinto a 5
        if grado != 5:
            #para las imágenes de grado distinto de 5 vamos a convertir el grado en etiqueta, sumándole 1 y convirtiéndolo a string
            etiqueta = 'G' + str(grado + 1)
            #posteriormente movemos la imagen a la carpeta que corresponda
            shutil.move('Raw Data/GitHub/validation/Images/' + i, 'Classified Data/Images/GitHub/' + etiqueta)

#finalmente debemos organizar el conjunto de imágenes de evaluation
#para ello primero cargamos el fichero .xlsx que contiene las labels
labels_evaluation = pd.read_excel('Raw Data/GitHub/evaluation/Challenge1_labels.xlsx')

#en este caso ya tenemos unificados los diagnósticos en una sola columna, lo que nos facilita la tarea
#por último solo debemos mover las imágenes a la carpeta que le corresponda
for i in os.listdir('Raw Data/GitHub/evaluation/Images'):
    #comprobamos si el último caracter del nombre es una 'g', para iterar así solo sobre las imágenes (.jpg)
    if i[-1] == 'g':
        #a partir del nombre de la imagen, eliminamos la extensión .jpg y buscamos en el dataframe el valor del grado correspondiente a ese identificador
        grado = labels_evaluation[labels_evaluation['image_id'] == i.split('.')[0]]['DR_Levels'].values[0]
        #sabemos que si el grado de la imagen es 5, esa imagen debe ser descartada porque la calidad es inadecuada
        #por ello únicamente nos quedaremos con las imágenes de grado distinto a 5
        if grado != 5:
            #para las imágenes de grado distinto de 5 vamos a convertir el grado en etiqueta, sumándole 1 y convirtiéndolo a string
            etiqueta = 'G' + str(grado + 1)
            #posteriormente movemos la imagen a la carpeta que corresponda
            shutil.move('Raw Data/GitHub/evaluation/Images/' + i, 'Classified Data/Images/GitHub/' + etiqueta)


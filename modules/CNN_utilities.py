'''
#########################################################################################################################
INFORMACIÓN DEL FICHERO
#########################################################################################################################

Autor: Samuel Lozano Juárez
Fecha: 08/03/2023
Institución: UBU | Grado en Ingeniería de la Salud

Este archivo forma parte del Trabajo de Fin de Grado "Detección del grado de retinopatía mediante redes convolucionales".
El alumno a cargo de este proyecto es el declarado como autor en las líneas anteriores.
Los tutores del proyecto fueron el Dr. Darío Fernández Zoppino y el Dr. Daniel Urda Muñoz.

En este fichero se van a definir algunas funciones que son recurrentes en el entrenamiento y test de las distintas redes neuronales convolucionales, por lo que se definirán una única vez en este script y así se podrán importar y emplear en los distintos archivos.
'''

def entrena(red,epocas,train_loader,optimizer,criterion):
    '''
    Realiza el entrenamiento completo de la red pasada como parámetro, empleando para ello las imágenes de entrenamiento proporcionadas por la estructura "loader" también pasada como parámetro.
    Además de llevar a cabo el entrenamiento, también es capaz de almacenar y representar las métricas del entrenamiento (loss y accuracy).
    Esta función NO tiene en cuenta el conjunto de datos de validación en el entrenamiento, por lo que NO incluye estrategias de "early stopping".
    
    Parámetros
    ------------------------------------------------------------------------
    red: objeto de tipo 
    epocas: 
    train_loader: 
    optimizer: 
    criterion: 
    
    Return
    ------------------------------------------------------------------------
    La función no devuelve ningún valor.
    '''
    
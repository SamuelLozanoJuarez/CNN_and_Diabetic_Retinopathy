'''
#########################################################################################################################
INFORMACIÓN DEL FICHERO
#########################################################################################################################

Autor: Samuel Lozano Juárez
Fecha: 18/05/2023
Institución: UBU | Grado en Ingeniería de la Salud

Este archivo forma parte del Trabajo de Fin de Grado "Detección del grado de retinopatía mediante redes convolucionales".
El alumno a cargo de este proyecto es el declarado como autor en las líneas anteriores.
Los tutores del proyecto fueron el Dr. Darío Fernández Zoppino y el Dr. Daniel Urda Muñoz.

A continuación se incluye el código que permite comparar los resultados y las métricas de las variaciones de la arquitectura Alqudah, Mobeen, Ghosh y Rajagopalan, entrenados con las imágenes de OCT sin procesar y testeados con las imágenes de Samsung e iPhone sin inpaint, incluyendo la estrategia de early-stopping. El objetivo es encontrar aquellas 3 variaciones de cada estructura que ofrecen un mejor rendimiento para poder entrenarlas empleando el conjunto de imágenes de Datasets.
'''

#No hace falta crear ni el SparkContext ni SparkSession, ya que tanto en la consola de PySpark como en Databricks son instanciados en el arranque
#importamos la función regexp_replace que nos permite sustituir valores dentro de una columna, y la función col para poder operar con columnas
from pyspark.sql.functions import regexp_replace, col
#cargamos el CSV, indicando que contiene  cabecera y que está codificado en ANSI (Latin-1)
#después usamos la función withColumn para reescribir la columna 'validacion', eliminando los caracteres "\xAD" fruto de la decodificación 
resultados = spark.read.format("csv").option("header", "true").option("encoding", "ISO-8859-1").load("Resultados.csv").withColumn("validacion", regexp_replace("validacion", "\xAD", ""))

#vamos a cambiar las columnas de las métricas a tipo float para poder operar posteriormente con ellas
#primero debemos definirlas
numericas = ['Acc','Bal_acc','F_score','Kappa','AUC','Tiempo(mins)']
#iteramos sobre ellas
for columna in numericas:
    #y usando la función withColumn cambiamos el tipo de 'String' a 'float'
    resultados = resultados.withColumn(columna, col(columna).cast("float"))
#finalmente persistimos en memoria el Dataframe
resultados.persist()

#definimos el valor de los distintos Baselines, para poder compararlos posteriormente con los resultados de cada modelo
acc_s = 0.558
acc_i = 0.5508
balacc_s = 0.5434
balacc_i = 0.4564
fscore_s = 0.5370
fscore_i = 0.5611
auc_s = 0.6746
auc_i = 0.6874
kappa_s = 0.3543
kappa_i = 0.3538

#definimos las funciones necesarias para la elección de arquitecturas
def supera_metricas(fila):
    '''
    Dada una fila (correspondiente a un modelo, con sus características y métricas) compara las métricas del modelo con el Baseline correspondiente (iPhone o Samsung) para determinar si supera el baseline o no.
    
    Parámetros
    ------------------------------
    fila: objeto de tipo Row que se corresponde con una fila del Dataframe resultados (correspondiente a su vez con el CSV Resultados).
    
    Return
    ------------------------------
    Devuelve un valor booleano (True o False). Si alguna de las métricas de la fila superan el baseline devuelve True, sino devuelve False.
    '''
    #por defecto inicializamos el valor del resultado como False
    resultado = False
    #primero debemos comprobar el conjunto de test al que corresponden las métricas (iPhone o Samsung)
    #si es iPhone
    if fila.test == 'iphone':
        #si el accuracy de la fila es superior al baseline establecemos el valor de resultado a True
        if fila.Acc >= acc_i:
            resultado = True
        #si el balance accuracy de la fila es superior al baseline establecemos el valor de resultado a True
        elif fila.Bal_acc >= balacc_i:
            resultado = True
        #si el F score de la fila es superior al baseline establecemos el valor de resultado a True
        elif fila.F_score >= fscore_i:
            resultado = True
        #si el kappa de la fila es superior al baseline establecemos el valor de resultado a True
        elif fila.Kappa >= kappa_i:
            resultado = True
        #si el AUC de la fila es superior al baseline establecemos el valor de resultado a True
        elif fila.AUC >= auc_i:
            resultado = True 
    #repetimos el mismo proceso pero en este caso considerando que el conjunto de test es el de Samsung
    else:
        if fila.Acc >= acc_s:
            resultado = True
        elif fila.Bal_acc >= balacc_s:
            resultado = True
        elif fila.F_score >= fscore_s:
            resultado = True 
        elif fila.Kappa >= kappa_s:
            resultado = True
        elif fila.AUC >= auc_s:
            resultado = True
    #devolvemos el valor del resultado
    return resultado

def calcula_distancia(fila):
    '''
    Calcula la distancia euclídea entre las métricas de la fila y las métricas correspondientes al baseline. En la consideración de las métricas no tiene en cuenta el accuracy ya que esta métrica no es la adecuada para problemas multiclase. Para calcular la distancia euclídea halla la diferencia entre el valor de la métrica de la fila y el valor del baseline correspondiente. Finalmente eleva cada diferencia al cuadrado y calcula la raíz de dicha suma.
    
    Parámetros
    ------------------------------
    fila: objeto de tipo Row que se corresponde con una fila del Dataframe resultados (correspondiente a su vez con el CSV Resultados).
    
    Return
    ------------------------------
    Devuelve un número decimal (float) correspondiente a la distancia euclídea.
    '''
    #primero debemos comprobar si el conjunto de test es iPhone o Samsung
    #si se trata de iPhone:
    if fila.test == 'iphone':
        #calculamos la diferencia entre el balanced accuracy de la fila y el baseline
        dif_balacc = balacc_i - fila.Bal_acc
        #calculamos la diferencia entre el auc de la fila y el baseline
        diff_auc = auc_i - fila.AUC
        #calculamos la diferencia entre el f score de la fila y el baseline
        diff_fscore = fscore_i - fila.F_score
        #calculamos la diferencia entre el kappa de la fila y el baseline
        diff_kappa = kappa_i - fila.Kappa
    #si se trata de Samsung realizamos el mismo proceso pero comparando con el baseline de Samsung
    else:
        dif_balacc = balacc_s - fila.Bal_acc
        diff_auc = auc_s - fila.AUC
        diff_fscore = fscore_s - fila.F_score
        diff_kappa = kappa_s - fila.Kappa
    #elevamos cada diferencia al cuadrado y las sumamos
    distancia = (dif_balacc)**2 + (diff_auc)**2 + (diff_fscore)**2 + (diff_kappa)**2
    #devolvemos el valor de la suma al cuadrado
    return distancia**(1/2)

def selecciona_mejores(arquitectura):
    '''
    Integra las funciones definidas previamente para encontrar aquellas combinaciones del número de capas convolucionales, número de filtros por capa y número de neuronas en las capas fully-connected cuya media de la distancia euclídea para Samsung e iPhone sea la menor posible (para cada modelo).
    
    Parámetros
    ------------------------------
    arquitectura: cadena de texto que indica la arquitectura para la cual se está comparando los valores. Puede ser 'Ghosh','Alqudah','Mobeen' o 'Rajagopalan'.
    
    Return
    ------------------------------
    No devuelve ningún elemento.
    '''
    #primero filtramos el Dataframe total, seleccionando únicamente aquellas filas correspondientes a la estructura a analizar y obtenidas con validación (y early stopping), ya que el entrenamiento se realizará con early stopping
    subset = resultados.filter((col("arquitectura") == arquitectura) & (col("validacion") == "Sí"))
    #persistimos ese dataframe
    subset.persist()
    #filtramos usando la función supera_metricas, para ver si alguna fila supera el baseline (experimentalmente ya sé que no)
    superanSubset = subset.rdd.filter(supera_metricas)
    #mostramos el número de filas que han superado el baseline (en todos los casos es 0)
    print(f'El número de variaciones de {arquitectura} que logran superar alguna métrica del Baseline es: {superanSubset.count()}')
    print('Se procede al cálculo de la distacia euclídea entre cada modelo y el Baseline, para la selección de aquellos con menor distancia.\n')
    #obtenemos un nuevo pairRDD al mapear el subset y crear para cada fila una tupla de 2 elementos: el identificador (valor de los parámetros de la arquitectura) y la distancia euclídea
    distancias = subset.rdd.map(lambda fila: ((fila.capas, fila.filtros, fila.neuronas), calcula_distancia(fila)))
    #finalmente agrupamos por identificadores y calculamos la media entre iPhone y Samsung y seleccionamos las 3 mejores variaciones (las que menor distancia euclídea tengan)
    top = distancias.reduceByKey(lambda x,y: x + y).mapValues(lambda x: x/2).sortBy(lambda x: x[1], ascending = True).take(3)
    print(f'Mejores modelos de {arquitectura}:')
    #para cada uno de esos modelos vamos a mostrar por pantalla sus parámetros
    for modelo in top:
        print(f'--------------------------\n*Num. capas convolucionales: {modelo[0][0]}\n*Num. filtros por capa: {modelo[0][1]}\n*Num. neuronas fully-connected: {modelo[0][2]}\n')
    #finalmente despersistimos el dataframe para liberar memoria
    subset.unpersist()

#aplicamos la función definida anteriormente para la arquitectura Ghosh
selecciona_mejores("Ghosh")

#aplicamos la función definida anteriormente para la arquitectura Alqudah
selecciona_mejores("Alqudah")

#aplicamos la función definida anteriormente para la arquitectura Mobeen
selecciona_mejores("Mobeen")

#aplicamos la función definida anteriormente para la arquitectura Rajagopalan
selecciona_mejores("Rajagopalan")

#por último despersistimos el Dataframe original de resultados
resultados.unpersist()

'''
#########################################################################################################################
INFORMACIÓN DEL FICHERO
#########################################################################################################################

Autor: Samuel Lozano Juárez
Fecha: 06/06/2023
Institución: UBU | Grado en Ingeniería de la Salud

Este archivo forma parte del Trabajo de Fin de Grado "Detección del grado de retinopatía mediante redes convolucionales".
El alumno a cargo de este proyecto es el declarado como autor en las líneas anteriores.
Los tutores del proyecto fueron el Dr. Darío Fernández Zoppino y el Dr. Daniel Urda Muñoz.

Este archivo tiene como objetivo llevar a cabo el análisis de los resultados obtenidos en las distintas ejecuciones. 
'''

#No hace falta crear ni el SparkContext ni SparkSession, ya que tanto en la consola de PySpark como en Databricks son instanciados en el arranque
#importamos la función regexp_replace que nos permite sustituir valores dentro de una columna, y la función col para poder operar con columnas
from pyspark.sql.functions import regexp_replace, col
#cargamos el CSV, indicando que contiene  cabecera y que está codificado en ANSI (Latin-1)
#después usamos la función withColumn para reescribir la columna 'validacion', eliminando los caracteres "\xAD" fruto de la decodificación 
resultados = spark.read.format("csv").option("header", "true").option("encoding", "ISO-8859-1").load("dbfs:/FileStore/shared_uploads/slj1001@alu.ubu.es/Resultados-1.csv").withColumn("validacion", regexp_replace("validacion", "\xAD", ""))

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
balacc_s = 0.5434
balacc_i = 0.4564
fscore_s = 0.5370
fscore_i = 0.5611
auc_s = 0.6746
auc_i = 0.6874
kappa_s = 0.3543
kappa_i = 0.3538

#######################################################
#MEJORES MODELOS
#######################################################

#definimos la función para buscar el modelo (si existe) capaz de superar las métricas del baseline
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
        #si el balance accuracy de la fila es superior al baseline establecemos el valor de resultado a True
        if fila.Bal_acc >= balacc_i:
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
        if fila.Bal_acc >= balacc_s:
            resultado = True
        elif fila.F_score >= fscore_s:
            resultado = True 
        elif fila.Kappa >= kappa_s:
            resultado = True
        elif fila.AUC >= auc_s:
            resultado = True
    #devolvemos el valor del resultado
    return resultado

#usando la función definida previamente tratamos de buscar el modelo que supere alguna de las métricas
mejores = resultados.rdd.filter(supera_metricas)
print(f'El número de resultados que superan alguna de las métricas es: {mejores.count()}')

#como ninguno de los modelos supera el baseline, vamos a comprobar qué modelo obtuvo un mejor rendimiento según las distintas métricas
#para ello primero separamos los resultados entre los obtenidos evaluando con iphone y con samsung
resultados_i = resultados.rdd.filter(lambda fila: fila[9] == 'iphone').persist()
resultados_s = resultados.rdd.filter(lambda fila: fila[9] == 'Samsung').persist()

#comenzamos obteniendo el modelo que mejor rendimiento ofrece en samsung y en iphone según balanced_accuracy
#para ello ordenamos cada RDD según el valor 11 (correspondiente a bal_acc) y obtenemos el primer elemento de la lista
best_balacc_i = resultados_i.sortBy(lambda x: x[11], ascending = False).first()
best_balacc_s = resultados_s.sortBy(lambda x: x[11], ascending = False).first()

#mostramos el resultado por pantalla
print(f'------------------------------\nEl modelo que mejor valor de balanced accuracy obtuvo sobre imágenes de iPhone fue:\n * Arq: {best_balacc_i[5]}\n * Capas: {best_balacc_i[6]}\n * Filtros: {best_balacc_i[7]}\n * Neuronas: {best_balacc_i[8]}\n * Conjunto entrenamiento: {best_balacc_i[0]}\n * Validacion: {best_balacc_i[1]}\n * Preprocesamiento: {best_balacc_i[2]}\n * Inpaint: {best_balacc_i[3]}\n * Tiempo de entrenamiento (min): {best_balacc_i[15]}\n * BAL_ACCURACY: {best_balacc_i[11]}')

print(f'------------------------------\nEl modelo que mejor valor de balanced accuracy obtuvo sobre imágenes de Samsung fue:\n * Arq: {best_balacc_s[5]}\n * Capas: {best_balacc_s[6]}\n * Filtros: {best_balacc_s[7]}\n * Neuronas: {best_balacc_s[8]}\n * Conjunto entrenamiento: {best_balacc_s[0]}\n * Validacion: {best_balacc_s[1]}\n * Preprocesamiento: {best_balacc_s[2]}\n * Inpaint: {best_balacc_s[3]}\n * Tiempo de entrenamiento (min): {best_balacc_s[15]}\n * BAL_ACCURACY: {best_balacc_s[11]}')

#realizamos el mismo proceso para la métrica F_score
best_fscore_i = resultados_i.sortBy(lambda x: x[12], ascending = False).first()
best_fscore_s = resultados_s.sortBy(lambda x: x[12], ascending = False).first()

#mostramos el resultado por pantalla
print(f'------------------------------\nEl modelo que mejor valor de F-score obtuvo sobre imágenes de iPhone fue:\n * Arq: {best_fscore_i[5]}\n * Capas: {best_fscore_i[6]}\n * Filtros: {best_fscore_i[7]}\n * Neuronas: {best_fscore_i[8]}\n * Conjunto entrenamiento: {best_fscore_i[0]}\n * Validacion: {best_fscore_i[1]}\n * Preprocesamiento: {best_fscore_i[2]}\n * Inpaint: {best_fscore_i[3]}\n * Tiempo de entrenamiento (min): {best_fscore_i[15]}\n * F_SCORE: {best_fscore_i[12]}')

print(f'------------------------------\nEl modelo que mejor valor de F-score obtuvo sobre imágenes de Samsung fue:\n * Arq: {best_fscore_s[5]}\n * Capas: {best_fscore_s[6]}\n * Filtros: {best_fscore_s[7]}\n * Neuronas: {best_fscore_s[8]}\n * Conjunto entrenamiento: {best_fscore_s[0]}\n * Validacion: {best_fscore_s[1]}\n * Preprocesamiento: {best_fscore_s[2]}\n * Inpaint: {best_fscore_s[3]}\n * Tiempo de entrenamiento (min): {best_fscore_s[15]}\n * F_SCORE: {best_fscore_s[12]}')

#realizamos el mismo proceso para la métrica AUC
best_auc_i = resultados_i.sortBy(lambda x: x[14], ascending = False).first()
best_auc_s = resultados_s.sortBy(lambda x: x[14], ascending = False).first()

#mostramos el resultado por pantalla
print(f'------------------------------\nEl modelo que mejor valor de AUC obtuvo sobre imágenes de iPhone fue:\n * Arq: {best_auc_i[5]}\n * Capas: {best_auc_i[6]}\n * Filtros: {best_auc_i[7]}\n * Neuronas: {best_auc_i[8]}\n * Conjunto entrenamiento: {best_auc_i[0]}\n * Validacion: {best_auc_i[1]}\n * Preprocesamiento: {best_auc_i[2]}\n * Inpaint: {best_auc_i[3]}\n * Tiempo de entrenamiento (min): {best_auc_i[15]}\n * AUC: {best_auc_i[14]}')

print(f'------------------------------\nEl modelo que mejor valor de AUC obtuvo sobre imágenes de Samsung fue:\n * Arq: {best_auc_s[5]}\n * Capas: {best_auc_s[6]}\n * Filtros: {best_auc_s[7]}\n * Neuronas: {best_auc_s[8]}\n * Conjunto entrenamiento: {best_auc_s[0]}\n * Validacion: {best_auc_s[1]}\n * Preprocesamiento: {best_auc_s[2]}\n * Inpaint: {best_auc_s[3]}\n * Tiempo de entrenamiento (min): {best_auc_s[15]}\n * AUC: {best_auc_s[14]}')

#realizamos el mismo proceso para la métrica de Cohen's Kappa
best_kappa_i = resultados_i.sortBy(lambda x: x[13], ascending = False).first()
best_kappa_s = resultados_s.sortBy(lambda x: x[13], ascending = False).first()

#mostramos el resultado por pantalla
print(f'------------------------------\nEl modelo que mejor valor de Kappa obtuvo sobre imágenes de iPhone fue:\n * Arq: {best_kappa_i[5]}\n * Capas: {best_kappa_i[6]}\n * Filtros: {best_kappa_i[7]}\n * Neuronas: {best_kappa_i[8]}\n * Conjunto entrenamiento: {best_kappa_i[0]}\n * Validacion: {best_kappa_i[1]}\n * Preprocesamiento: {best_kappa_i[2]}\n * Inpaint: {best_kappa_i[3]}\n * Tiempo de entrenamiento (min): {best_kappa_i[15]}\n * KAPPA: {best_kappa_i[13]}')

print(f'------------------------------\nEl modelo que mejor valor de Kappa obtuvo sobre imágenes de Samsung fue:\n * Arq: {best_kappa_s[5]}\n * Capas: {best_kappa_s[6]}\n * Filtros: {best_kappa_s[7]}\n * Neuronas: {best_kappa_s[8]}\n * Conjunto entrenamiento: {best_kappa_s[0]}\n * Validacion: {best_kappa_s[1]}\n * Preprocesamiento: {best_kappa_s[2]}\n * Inpaint: {best_kappa_s[3]}\n * Tiempo de entrenamiento (min): {best_kappa_s[15]}\n * KAPPA: {best_kappa_s[13]}')

#Por último, para obtener una visión más global del rendimiento de los distintos modelos, se va a usar la distancia euclídea para determinar qué modelo ha obtenido un rendimiento más próximo al deseado (considerando todas las métricas), en Samsung y en iPhone.
#definimos la función para el cálculo de la distancia
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

#aplicamos esta función sobre cada una de las filas y seleccionamos el modelo que menos distancia euclídea ofrezca para cada conjunto de test
best_euc_i = resultados_i.map(lambda fila: (fila,calcula_distancia(fila))).sortBy(lambda x: x[1]).first()
best_euc_s = resultados_s.map(lambda fila: (fila,calcula_distancia(fila))).sortBy(lambda x: x[1]).first()

#y finalmente mostramos el resultado por pantalla
print(f'------------------------------\nEl modelo que menor distancia euclídea obtuvo con imágenes de iPhone fue:\n * Arq: {best_euc_i[0].arquitectura}\n * Capas: {best_euc_i[0].capas}\n * Filtros: {best_euc_i[0].filtros}\n * Neuronas: {best_euc_i[0].neuronas}\n * Conjunto entrenamiento: {best_euc_i[0].img_entrenamiento}\n * Validacion: {best_euc_i[0].validacion}\n * Preprocesamiento: {best_euc_i[0].preprocesamiento}\n * Inpaint: {best_euc_i[0].inpaint}\n * Tiempo de entrenamiento (min): {best_euc_i[0][15]}\n * BAL_ACCURACY: {best_euc_i[0].Bal_acc}\n * F-SCORE: {best_euc_i[0].F_score}\n * AUC: {best_euc_i[0].AUC}\n * KAPPA: {best_euc_i[0].Kappa}\n')

print(f'------------------------------\nEl modelo que menor distancia euclídea obtuvo con imágenes de Samsung fue:\n * Arq: {best_euc_s[0].arquitectura}\n * Capas: {best_euc_s[0].capas}\n * Filtros: {best_euc_s[0].filtros}\n * Neuronas: {best_euc_s[0].neuronas}\n * Conjunto entrenamiento: {best_euc_s[0].img_entrenamiento}\n * Validacion: {best_euc_s[0].validacion}\n * Preprocesamiento: {best_euc_s[0].preprocesamiento}\n * Inpaint: {best_euc_s[0].inpaint}\n * Tiempo de entrenamiento (min): {best_euc_s[0][15]}\n * BAL_ACCURACY: {best_euc_s[0].Bal_acc}\n * F-SCORE: {best_euc_s[0].F_score}\n * AUC: {best_euc_s[0].AUC}\n * KAPPA: {best_euc_s[0].Kappa}\n')

#despersistimos los RDD 
resultados_i.unpersist()
resultados_s.unpersist()

#######################################################
#ANÁLISIS ARQUITECTURAS
#######################################################

#a continuación se van a obtener algunos resultados generales de las distintas arquitecturas
#para ello primero agrupamos todos los resultados por arquitectura y por conjunto de test empleado
#posteriormente obtenemos las medias deseadas
from pyspark.sql import functions as F
resultados.groupBy("arquitectura","test").agg(
    F.round(F.avg('Bal_acc'),2).alias("avgBal_acc"),
    F.round(F.max('Bal_acc'),2).alias("maxBal_acc"),
    F.round(F.avg('F_score'),2).alias("avgFscore"),
    F.round(F.max('F_score'),2).alias("maxFscore"),
    F.round(F.avg('AUC'),2).alias("avgAUC"),
    F.round(F.max('AUC'),2).alias("maxAUC"),
    F.round(F.avg('Kappa'),2).alias("avgKappa"),
    F.round(F.max('Kappa'),2).alias("maxKappa"),
    F.round(F.avg('Tiempo(mins)'),2).alias("avgTiempo"),
    F.round(F.max('Tiempo(mins)'),2).alias("maxTiempo"),
).orderBy(["arquitectura","test"]).show()

#######################################################
#ANÁLISIS CONJUNTOS DE DATOS
#######################################################

#por último vamos a realizar una serie de análisis sencillos sobre los conjuntos de datos
#para cada posible conjunto de entrenamiento vamos a obtener la media de las métricas y del tiempo empleado en el entrenamiento
#seguimos una estructura similar al caso anterior
resultados.groupBy("img_entrenamiento","test").agg(
    F.round(F.avg('Bal_acc'),2).alias("avgBal_acc"),
    F.round(F.max('Bal_acc'),2).alias("maxBal_acc"),
    F.round(F.avg('F_score'),2).alias("avgFscore"),
    F.round(F.max('F_score'),2).alias("maxFscore"),
    F.round(F.avg('AUC'),2).alias("avgAUC"),
    F.round(F.max('AUC'),2).alias("maxAUC"),
    F.round(F.avg('Kappa'),2).alias("avgKappa"),
    F.round(F.max('Kappa'),2).alias("maxKappa"),
    F.round(F.avg('Tiempo(mins)'),2).alias("avgTiempo"),
    F.round(F.max('Tiempo(mins)'),2).alias("maxTiempo"),
).orderBy(["img_entrenamiento","test"]).show()


#finalmente despersistimos los RDD persisidos
resultados.unpersist()
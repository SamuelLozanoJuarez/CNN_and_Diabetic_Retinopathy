'''
#########################################################################################################################
INFORMACIÓN DEL FICHERO
#########################################################################################################################

Autor: Samuel Lozano Juárez
Fecha: 30/01/2023
Institución: UBU | Grado en Ingeniería de la Salud

Este archivo forma parte del Trabajo de Fin de Grado "Detección del grado de retinopatía mediante redes convolucionales".
El alumno a cargo de este proyecto es el declarado como autor en las líneas anteriores.
Los tutores del proyecto fueron el Dr. Darío Fernández Zoppino y el Dr. Daniel Urda Muñoz.

En el código a continuación se va a realizar una depuración de los datos proporcionados por los clínicos (como se podrá comprobar
esta etapa es prácticamente idéntica a la depuración realizada en el procesamiento de las imágenes), para posteriormente calcular
ciertas métricas que permitan evaluar el rendimiento de los retinólogos a la hora de predecir el grado de retinopatía diabética a partir
de imágenes de fondo de ojo tomadas con dispositivos Samsung e iPhone. Para realizar la comparación se tomará como 'gold standard'
el diagnóstico realizado por los retinólogos empleando imágenes de fondo de ojo tomadas con el retinógrafo.

Las métricas que se obtengan constituirán el 'baseline', o valor base que trataremos de mejorar con los modelos de redes neuronales
convolucionales.
'''


#importamos los paquetes necesarios para la ejecución del código
import pandas as pd
import numpy as np
import os
import seaborn as sns
import sklearn.metrics
from sklearn.metrics import accuracy_score,confusion_matrix,balanced_accuracy_score,f1_score,roc_auc_score,cohen_kappa_score


# ----------------------------------
# DEPURACIÓN DE LOS DATOS
# ------------------------------------

#cargamos el excel al completo en la variable df
df = pd.read_excel('Datos/Calidad_Diagnóstico_Fotos.xlsx', sheet_name = 'Resultados', skiprows = 1)

#eliminamos las columnas de la 5 a la 8 y de la 10 a la 23 porque no aportan información relevante
df.drop(df.columns[10:24], axis = 1, inplace = True)
df.drop(df.columns[5:9], axis = 1, inplace = True)
NHC_validos = set(df['NHC'])

#obtenemos los NHC de todos los pacientes con algún valor missing en alguna columna
NA_values = set(df[df.isna().any(axis = 1)]['NHC'])
NHC_validos = NHC_validos - NA_values

#eliminamos todos los pacientes cuyo NHC esté en la lista NA_values
df.drop(df[df['NHC'].isin(NA_values) == True].index, inplace = True)

#a continuación obtenemos los NHC de todos aquellos pacientes que no posean las 12 filas necesarias
no_12filas = set(filter(lambda x: list(df['NHC']).count(x) != 12, NHC_validos))
NHC_validos = NHC_validos - no_12filas

#y los eliminamos del dataframe 
df.drop(df[df['NHC'].isin(no_12filas) == True].index, inplace = True)

#por último vamos a comprobar si esas 12 filas están bien distribuidas (4 filas por aparato, 6 para cada ojo, y 6 para cada retinólogo)
no_distribuidos = set()
for i in NHC_validos:
    if list(df[df['NHC'] == i]['1 OCT 2 IPHONE 3 SAMSUNG']).count(1) != 4 or list(df[df['NHC'] == i]['1 OCT 2 IPHONE 3 SAMSUNG']).count(2) != 4 or list(df[df['NHC'] == i]['1 OCT 2 IPHONE 3 SAMSUNG']).count(3) != 4:
        no_distribuidos.add(i)
    if list(df[df['NHC'] == i]['lateralidad 1 Dch 2 izq']).count(1) != 6 or list(df[df['NHC'] == i]['lateralidad 1 Dch 2 izq']).count(2) != 6:
        no_distribuidos.add(i)
    if list(df[df['NHC'] == i]['Retinlogo 1 y 2']).count(1) != 6 or list(df[df['NHC'] == i]['Retinlogo 1 y 2']).count(2) != 6:
        no_distribuidos.add(i)

NHC_validos = NHC_validos - no_distribuidos

#y eliminamos del dataframe los pacientes con filas mal distribuidas
df.drop(df[df['NHC'].isin(no_distribuidos) == True].index, inplace = True)

#vamos a comprobar qué filas de cada dataset tienen sus 6 imágenes correspondientes
fotos_OCT = [i.upper() for i in os.listdir('Datos/Raw Data/FOTOS OCT')]
fotos_iphone = [i.upper() for i in os.listdir('Datos/Raw Data/FOTOS iPhone')]
fotos_samsung = [i.upper() for i in os.listdir('Datos/Raw Data/FOTOS Samsung')]

no_foto = set()

for i in NHC_validos:
    oct_der = str(int(i)).upper() + "TD.JPG"
    oct_izq = str(int(i)).upper() + "TI.JPG"
    iphone_der = str(int(i)).upper() + "ED.PNG"
    iphone_izq = str(int(i)).upper() + "EI.PNG"
    samsung_der = str(int(i)).upper() + "GD.PNG"
    samsung_izq = str(int(i)).upper() + "GI.PNG"
    
    if oct_der not in fotos_OCT or oct_izq not in fotos_OCT or iphone_der not in fotos_iphone or iphone_izq not in fotos_iphone or samsung_der not in fotos_samsung or samsung_izq not in fotos_samsung:
        no_foto.add(i)

#eliminamos los pacientes que no tienen las 6 imágenes
NHC_validos = NHC_validos - no_foto

#por último eliminamos aquellos pacientes que no poseen las 6 fotos
df.drop(df[df['NHC'].isin(no_foto) == True].index, inplace = True)

#creamos 3 sub-dataframe,dependiendo del tipo de instrumento empleado para tomar las imágenes 
retin = df[df['1 OCT 2 IPHONE 3 SAMSUNG'] == 1]
iphone = df[df['1 OCT 2 IPHONE 3 SAMSUNG'] == 2]
samsung = df[df['1 OCT 2 IPHONE 3 SAMSUNG'] == 3]

#para finalizar esta primera parte vamos a definir la función factorizar que convierte los campos de Grado y Clasificación de numérico a tipo String
def factorizar(df, columnas = ["GRADO RETINOPATÍA DIABÉTICA", "Clasificación EMD. 1 NO . 2 NO CENTRAL, 3 CENTRAL"]):
    '''
    Transforma las columnas del Grado y Clasificación EMD de tipo numérico a tipo String, ya que nuestro problema ha de ser un problema de etiquetado multiclases, no un problema numérico.
    
    Parámetros
    ----------------------------------------------------------------------
    df: pandas.DataFrame sobre el cual se van a efectuar las modificaciones de las columnas.
    columnas: una lista que contiene los nombres de las columnas que se desean modificar. Por defecto es una lista que contiene el nombre de las columnas correspondientes al Grado y la Clasificación EMD.
    
    Return
    ----------------------------------------------------------------------
    df: pandas.DataFrame idéntico al introducido como parámetro pero con las columnas ya modificadas.
    '''
    i = 0
    while i<len(columnas):
        df[columnas[i]] = df[columnas[i]].map({1.0:'uno',
                                               2.0:'dos',
                                               3.0:'tres',
                                               4.0:'cuatro',
                                               5.0:'cinco',
                                               'uno':'uno',
                                               'dos':'dos',
                                               'tres':'tres',
                                               'cuatro':'cuatro',
                                               'cinco':'cinco'})
        i += 1
    
    return df


# Como los valores de grado de retinopatía del OCT son los que vamos a emplear como salidas en el proceso de entrenamiento, necesitamos unificar sus valores. Por ello en los casos en que encontramos discrepancias vamos a sustituir el diagnóstico más favorable por el menos favorable (siempre es preferente sobrediagnosticar un caso negativo que no detectar uno positivo).

#declaro una función que sustituye el mejor por el peor diagnóstico en aquellos discrepantes
def reemplaza_diagnostico(df):
    '''
    Comprueba qué imágenes poseen un diagnóstico discrepante, y en aquellas donde se cumpla esta condición reemplaza el diagnóstico más leve (más bajo numéricamente) por aquel de mayor gravedad (más alto numéricamente).
    
    Parámetros
    ----------------------------------------------------------------------
    df: pandas.DataFrame correspondiente a la tabla en la que se encuentra la información a cotejar.
    
    Return
    ----------------------------------------------------------------------
    df: pandas.DataFrame con los valores de los diagnósticos ya actualizados (se devuelve el DataFrame factorizado, convirtiendo el Grado y Clasificación nuevamente a tipo String)
    '''
    df.sort_values(['NHC','lateralidad 1 Dch 2 izq'], inplace = True)
    k = 0
    while k<(len(df)-1):
        if df.iloc[k,5] != df.iloc[k+1,5]:
            if df.iloc[k,5] > df.iloc[k+1,5]:
                df.iloc[k+1,5] = df.iloc[k,5]
            else:
                df.iloc[k,5] = df.iloc[k+1,5]
        if df.iloc[k,6] != df.iloc[k+1][6]:
            if df.iloc[k,6] > df.iloc[k+1,6]:
                df.iloc[k+1,6] = df.iloc[k,6]
            else:
                df.iloc[k,6] = df.iloc[k+1,6]
        k+=2
    return factorizar(df)

#homogeneizamos los diagnósticos de los 3 sub-dataframes creados anteriormente
retin = reemplaza_diagnostico(retin)
samsung = reemplaza_diagnostico(samsung)
iphone = reemplaza_diagnostico(iphone)


# ----------------------------
# OBTENCIÓN DE MÉTRICAS
# -----------------------------

# Una vez depurados los datos y homogeneizados los diagnósticos de los clínicos, vamos a obtener distintas métricas que nos permitirán evaluar el grado de acierto de los retinólogos sobre el conjunto de imágenes de iPhone y Samsung. Para ello se considerará como 'gold standard' el diagnóstico realizado por los retinólogos sobre las imágenes de OCT.
# 
# Para poder comparar los diagnósticos sobre las imágenes de OCT con los diagnósticos con imágenes de iPhone y Samsung lo primero que debemos hacer es ordenar de manera idéntica los 3 sub-datasets. De esta forma la correspondencia entre filas de los sub-datasets será la que nos permita hacer la comparación.

#ordenamos los datasets en función del número de historia clínica y la lateralidad del ojo (derecho o izquierdo)
retin.sort_values(['NHC','lateralidad 1 Dch 2 izq'], inplace = True)
samsung.sort_values(['NHC','lateralidad 1 Dch 2 izq'], inplace = True)
iphone.sort_values(['NHC','lateralidad 1 Dch 2 izq'], inplace = True)


# Ahora sabemos que los dataframes _retin, iphone_ y _samsung_ poseen los mismos números de historia clínica y en el mismo orden, que la lateralidad de los ojos sigue el siguiente patrón: 1,1,2,2,1,1,2,2,1,1,2,2... y que el retinólogo va alternando entre filas de la siguiente manera: 2,1,2,1,2,1,2,1,... Todas estas conclusiones las sabemos porque hemos ordenado anteriormente los 3 datasets según la columna de NHC y de lateralidad, y hemos cribado todas las filas hasta quedarnos exclusivamente con aquellas comunes a los 3 dataframes.
# 
# Es por ello que sabemos que la fila X de la columna _GRADO RETINOPATÍA DIABÉTICA_ se corresponde con el mismo número de historia clínica en las 3 tablas.

#vamos a igualar los índices de todas las tablas, para poder facilitar la comparación entre ellas más adelante
retin.reset_index(drop = True, inplace = True)
samsung.reset_index(drop = True, inplace = True)
iphone.reset_index(drop = True, inplace = True)

#almacenamos en la variable grado_golds_1 los resultados del diagnóstico del grado usando el gold standard (OCT)
grado_golds = retin.iloc[:,5:6]

#por último cargamos los diagnósticos del grado de retinopatía para fotos con iPhone y con Samsung
grado_iphone = iphone.iloc[:, 5:6]
grado_samsung = samsung.iloc[:,5:6]


# En el proceso de evaluación se van a obtener las siguientes métricas:
#  - accuracy
#  - balanced accuracy
#  - f-score
#  - AUC de la curva ROC
#  - Quadratic Weighted Kappa

# --------------------------
# ACCURACY
# ---------------------------

# Además del valor de accuracy, se va a obtener la matriz de confusión para el diagnóstico con imágenes de iPhone y con imágenes de Samsung. Se empleará el paquete seaborn para lograr una mejor representación de la matriz.

matriz_grado_iphone = confusion_matrix(y_true = grado_golds,y_pred = grado_iphone)

print(f"Acierto en el diagnóstico del grado con fotos de iPhone: {accuracy_score(grado_golds, grado_iphone)}\n")

plot = sns.heatmap(matriz_grado_iphone, annot = True, cmap = 'Reds', cbar = False)
plot.set_title('GRADO REAL vs. IPHONE\n')
plot.set_xlabel('\nGrado Iphone')
plot.set_ylabel('Grado real\n')
plot.xaxis.set_ticklabels(['Grado1','Grado2','Grado3','Grado4','Grado5'])
plot.yaxis.set_ticklabels(['Grado1','Grado2','Grado3','Grado4','Grado5'])
print(plot)

matriz_grado_samsung = confusion_matrix(grado_golds,grado_samsung)

print(f"Acierto en el diagnóstico del grado con fotos de Samsung: {accuracy_score(grado_golds, grado_samsung)} \n")

plot = sns.heatmap(matriz_grado_samsung, annot = True, cmap = 'Reds', cbar = False)
plot.set_title('GRADO REAL vs. SAMSUNG\n')
plot.set_xlabel('\nGrado Samsung')
plot.set_ylabel('Grado real\n')
plot.xaxis.set_ticklabels(['Grado1','Grado2','Grado3','Grado4','Grado5'])
plot.yaxis.set_ticklabels(['Grado1','Grado2','Grado3','Grado4','Grado5'])
print(plot)


# ---------------------------------
# BALANCED ACCURACY
# ----------------------------------

bal_acc_gr_iphone = balanced_accuracy_score(y_true = grado_golds, y_pred = grado_iphone)
print(f"Valor de balanced accuracy para el grado con iPhone: {bal_acc_gr_iphone}")

bal_acc_gr_samsung = balanced_accuracy_score(y_true = grado_golds, y_pred = grado_samsung)
print(f"Valor de balanced accuracy para el grado con Samsung: {bal_acc_gr_samsung}")


# -------------------
# F-SCORE
# --------------------

# Es importante establecer el parámetro 'average' a 'weighted' para que se tengan en cuenta los pesos de las clases, es decir, la representación de imágenes de cada clase.

f1_gr_iphone = f1_score(y_true = grado_golds, y_pred = grado_iphone, average = 'weighted')
print(f"F-score para grado con iPhone: {f1_gr_iphone}")

f1_gr_samsung = f1_score(y_true = grado_golds, y_pred = grado_samsung, average = 'weighted')
print(f"F-score para grado con Samsung: {f1_gr_samsung}")


# -------------------
# AUC-ROC
# -------------------

# Para obtener este valor, debemos proporcionar al método no solo un array de y_pred y otro de y_true, sino que debemos proporcionar una matriz de probabilidades de predicción, para que pueda dibujar la curva y obtener el correspondiente área bajo ella. En este caso no poseemos una matriz de probabilidades, por lo que la construiremos considerando un 100% de probabilidad de pertenencia a la clase predicha por el retinólogo sobre OCT y un 0% para el resto de clases.

#primero obtenemos la matriz de probabilidades para cada clase, que consideraremos como gold standard. Es importante realizar la conversión entre tipo String y tipo numérico para la posterior computación
gr_golds_matrix = []
for i in grado_golds.values:
    if i[0] == 'uno':
        gr_golds_matrix.append([1.0,0,0,0,0])
    elif i[0] == 'dos':
        gr_golds_matrix.append([0,1.0,0,0,0])
    elif i[0] == 'tres':
        gr_golds_matrix.append([0,0,1.0,0,0])
    elif i[0] == 'cuatro':
        gr_golds_matrix.append([0,0,0,1.0,0])
    elif i[0] == 'cinco':
        gr_golds_matrix.append([0,0,0,0,1.0])
gr_golds_matrix = np.array(gr_golds_matrix)

#realizamos el mismo proceso para la matriz de predicciones con imágenes de iPhone
gr_iphone_matrix = []
for i in grado_iphone.values:
    if i[0] == 'uno':
        gr_iphone_matrix.append([1.0,0,0,0,0])
    elif i[0] == 'dos':
        gr_iphone_matrix.append([0,1.0,0,0,0])
    elif i[0] == 'tres':
        gr_iphone_matrix.append([0,0,1.0,0,0])
    elif i[0] == 'cuatro':
        gr_iphone_matrix.append([0,0,0,1.0,0])
    elif i[0] == 'cinco':
        gr_iphone_matrix.append([0,0,0,0,1.0])
gr_iphone_matrix = np.array(gr_iphone_matrix)

#y para las predicciones con imágenes de Samsung
gr_samsung_matrix = []
for i in grado_samsung.values:
    if i[0] == 'uno':
        gr_samsung_matrix.append([1.0,0,0,0,0])
    elif i[0] == 'dos':
        gr_samsung_matrix.append([0,1.0,0,0,0])
    elif i[0] == 'tres':
        gr_samsung_matrix.append([0,0,1.0,0,0])
    elif i[0] == 'cuatro':
        gr_samsung_matrix.append([0,0,0,1.0,0])
    elif i[0] == 'cinco':
        gr_samsung_matrix.append([0,0,0,0,1.0])
gr_samsung_matrix = np.array(gr_samsung_matrix)


# Una vez generadas las matrices correspondientes ya podemos calcular el valor AUC para la curva ROC correspondiente a cada predicción.

auc_gr_iphone = roc_auc_score(y_true = gr_golds_matrix, y_score = gr_iphone_matrix, average = 'weighted', multi_class = 'ovr')
print(f"El valor de AUC para el diagnóstico del grado usando iPhone es: {auc_gr_iphone}")

auc_gr_samsung = roc_auc_score(y_true = gr_golds_matrix, y_score = gr_samsung_matrix, average = 'weighted', multi_class = 'ovr')
print(f"El valor de AUC para el diagnóstico del grado usando Samsung es: {auc_gr_samsung}")


# ----------------
# QUADRATIC WEIGHTED KAPPA
# ----------------

# Para el cálculo de la métrica Kappa es necesario convertir los datos de tipo String a tipo numérico. Para ello definiremos la siguiente función.

def subs(x):
    '''
    Para una cadena dada, devuelve un valor numérico correspondiente al valor de la cadena introducida. Actúa de manera similar a un diccionario.
    
    Parámetros
    ----------------------------------------------------------------------
    x: String que actuará como clave para determinar qué valor devuelve la función.
    
    Return
    ----------------------------------------------------------------------
    Devuelve un valor numérico (1,2,3,4 o 5) si el parámetro introducido cumple alguna de las condiciones, o sino devuelve el mismo parámetro introducido sin modificar.
    '''
    if x == 'uno':
        return 1
    elif x == 'dos':
        return 2
    elif x == 'tres':
        return 3
    elif x == 'cuatro':
        return 4
    elif x == 'cinco':
        return 5
    else:
        return x

#convertimos los valores de los diagnósticos de tipo String a tipo numérico para cada dispositivo de toma de imágenes
grado_golds = list(map(subs,grado_golds['GRADO RETINOPATÍA DIABÉTICA']))

grado_iphone = list(map(subs,list(grado_iphone['GRADO RETINOPATÍA DIABÉTICA'].values)))

grado_samsung = list(map(subs,list(grado_samsung['GRADO RETINOPATÍA DIABÉTICA'].values)))

#calculamos el valor kappa para imágenes de iphone y de samsung
kappa_gr_iphone = cohen_kappa_score(y1 = grado_golds, y2 = grado_iphone, labels = list(set(grado_golds)))
print(f'El valor de kappa para el grado de iphone es: {kappa_gr_iphone}')

kappa_gr_samsung = cohen_kappa_score(y1 = grado_golds, y2 = grado_samsung, labels = list(set(grado_golds)))
print(f'El valor de kappa para el grado de samsung es: {kappa_gr_samsung}')


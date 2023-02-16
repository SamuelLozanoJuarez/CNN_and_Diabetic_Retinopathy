'''
#########################################################################################################################
INFORMACIÓN DEL FICHERO
#########################################################################################################################

Autor: Samuel Lozano Juárez
Fecha: 27/12/2022
Institución: UBU | Grado en Ingeniería de la Salud

Este archivo forma parte del Trabajo de Fin de Grado "Detección del grado de retinopatía mediante redes convolucionales".
El alumno a cargo de este proyecto es el declarado como autor en las líneas anteriores.
Los tutores del proyecto fueron el Dr. Darío Fernández Zoppino y el Dr. Daniel Urda Muñoz.

En el código que se encuentra a continuación se va a proceder al procesamiento de los datos crudos proporcionados por los clínicos para el entrenamiento de la CNN.
Se dispone de un fichero de Excel que contiene información referente al diagnóstico y calidad de las imágenes de retina proporcionadas. Se usará la información
de este fichero para filtrar las imágenes, eliminando aquellas que no sean informativas ya sea por su calidad o por su incompletitud, y se clasificarán las imágenes, 
organizándolas en distintas carpetas según su grado (desde grado 1 hasta grado 5).
'''



#importamos los paquetes necesarios para la ejecución del código
import pandas as pd
import numpy as np
import os
import shutil

#cargamos el excel al completo en la variable df
df = pd.read_excel('Calidad_Diagnóstico_Fotos.xlsx', sheet_name = 'Resultados', skiprows = 1)


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
fotos_OCT = [i.upper() for i in os.listdir('FOTOS OCT')]
fotos_iphone = [i.upper() for i in os.listdir('FOTOS iPhone')]
fotos_samsung = [i.upper() for i in os.listdir('FOTOS Samsung')]

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

#eliminamos los NCH de los pacientes que no tienen las 6 imágenes
NHC_validos = NHC_validos - no_foto


#por último eliminamos del DataFrame aquellos pacientes que no poseen las 6 fotos
df.drop(df[df['NHC'].isin(no_foto) == True].index, inplace = True)


#creamos 3 sub-dataframe,dependiendo del tipo de instrumento empleado para tomar las imágenes (OCT, iPhone o Samsung)
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


#aplicamos la factorización sobre los 3 sub-datasets creados anteriormente
factorizar(retin)
factorizar(iphone)
factorizar(samsung)


#declaro una función que sustituye el mejor por el peor diagnóstico en aquellos discrepantes
#para comparar los diagnósticos voy a declarar una función que se llama desfactorizar, para convertir los campos de 'Grado retinopatía' y 'EMD' a numérico de nuevo
def desfactorizar(df, columnas = ["GRADO RETINOPATÍA DIABÉTICA", "Clasificación EMD. 1 NO . 2 NO CENTRAL, 3 CENTRAL"]):
    '''
    Función que transforma ciertas columnas de un DataFrame de tipo String a tipo numérico (float).
    
    Parámetros
    ----------------------------------------------------------------------
    df: pandas.DataFrame sobre el cual se van a efectuar las modificaciones de las columnas.
    columnas: una lista que contiene los nombres de las columnas que se desean modificar. Por defecto es una lista que contiene el nombre de las columnas correspondientes al Grado y la Clasificación EMD.
    
    Return
    ----------------------------------------------------------------------
    df:pandas.DataFrame idéntico al introducido como parámetro pero con las columnas ya modificadas.
    '''
    i = 0
    while i<len(columnas):
        df[columnas[i]] = df[columnas[i]].map({'uno':1.0,
                                               'dos':2.0,
                                               'tres':3.0,
                                               'cuatro':4.0,
                                               'cinco':5.0,
                                               1.0:1.0,
                                               2.0:2.0,
                                               3.0:3.0,
                                               4.0:4.0,
                                               5.0:5.0})
        i += 1
    
    return df
    
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
    dataf = desfactorizar(df)
    dataf.sort_values(['NHC','lateralidad 1 Dch 2 izq'], inplace = True)
    k = 0
    while k<(len(dataf)-1):
        if dataf.iloc[k,5] != dataf.iloc[k+1,5]:
            if dataf.iloc[k,5] > dataf.iloc[k+1,5]:
                dataf.iloc[k+1,5] = dataf.iloc[k,5]
            else:
                dataf.iloc[k,5] = dataf.iloc[k+1,5]
        if dataf.iloc[k,6] != dataf.iloc[k+1][6]:
            if dataf.iloc[k,6] > dataf.iloc[k+1,6]:
                dataf.iloc[k+1,6] = dataf.iloc[k,6]
            else:
                dataf.iloc[k,6] = dataf.iloc[k+1,6]
        k+=2
    return factorizar(dataf)


#homogeneizamos los diagnósticos de los 3 sub-dataframes creados anteriormente
retin = reemplaza_diagnostico(retin)
samsung = reemplaza_diagnostico(samsung)
iphone = reemplaza_diagnostico(iphone)


#a continuación definimos una función que sustituya los valores de las imágenes por su mejor valor
def sustituye_calidad(df):
    '''
    Busca las filas del DataFrame correspondientes a una misma imagen que tienen valores de calidad discrepantes, y sustituiye el peor (más bajo) por el mejor (más alto).
    
    Parámetros
    ----------------------------------------------------------------------
    df: pandas.DataFrame sobre el cual se van a realizar los cambios en la columna de calidad
    
    Return
    ----------------------------------------------------------------------
    df: pandas.DataFrame con los valores ya actualizados (sin discrepancias entre calidades de una misma imagen)
    '''
    df.sort_values(['NHC','lateralidad 1 Dch 2 izq'], inplace = True)
    k = 0
    while k<(len(df)-1):
        if df.iloc[k,4] != df.iloc[k+1,4]:
            if df.iloc[k,4] > df.iloc[k+1,4]:
                df.iloc[k+1,4] = df.iloc[k,4]
            else:
                df.iloc[k,4] = df.iloc[k+1,4]
        k+=2
    return df

#sustituimos los valores de calidad dispares en los sub-dataframes creados anteriormente
retin = sustituye_calidad(retin)
samsung = sustituye_calidad(samsung)
iphone = sustituye_calidad(iphone)

#eliminamos de cada sub-dataframe las filas correspondientes a imágenes con una calidad inferior a 4
retin.drop(list(retin.where(retin['CALIDAD GRAL IMAGEN']<4).dropna().index), inplace = True)
samsung.drop(list(samsung.where(samsung['CALIDAD GRAL IMAGEN']<4).dropna().index), inplace = True)
iphone.drop(list(iphone.where(iphone['CALIDAD GRAL IMAGEN']<4).dropna().index), inplace = True)

#primero definimos una función que dado un grado de retinopatía permite generar 3 listas de NHC: una correspondiente a las imágenes de OCT de ese grado, otra para las imágenes de Samsung y otra para las de iPhone.
def listas_NHC(grado):
    '''
    Obtiene las listas correspondientes a los NHC de las imágenes de OCT, Samsung e iPhone para el grado pasado como parámetro.
    
    Parámetros
    ----------------------------------------------------------------------
    grado: string que puede tomar valor 'uno','dos','tres','cuatro' o 'cinco' y que indica el grado de retinopatía para el cual se desea obtener la lista.
    
    Return
    ----------------------------------------------------------------------
    Devuelve 3 objetos:
        lista_retin: lista que contiene los NHC de las imágenes de OCT con el grado pasado como parámetro.
        lista_samsung: lista que contiene los NHC de las imágenes de Samsung con el grado pedido.
        lista_iphone: lista que contiene los NHC de las imágenes de iPhone con el grado pedido.
    '''
    lista_retin = set(retin[retin['GRADO RETINOPATÍA DIABÉTICA']==grado].NHC.values)
    lista_samsung = set(samsung[samsung['GRADO RETINOPATÍA DIABÉTICA']==grado].NHC.values)
    lista_iphone = set(iphone[iphone['GRADO RETINOPATÍA DIABÉTICA']==grado].NHC.values)
    return lista_retin, lista_samsung, lista_iphone

#una vez obtenidos los NHC correspondientes a cada grado y cada dispositivo (OCT, Samsung o iPhone) vamos a obtener el nombre de las imágenes, que está compuesto por el NHC junto con dos letras más que indican si se trata del ojo izquierdo o el derecho y la extensión de la imagen.
#definimos una función para llevar a cabo esta reconstrucción de los nombres de las imágenes
def nombre_imagenes(grado):
    '''
    Obtiene las listas con los nombres de las imágenes de cada dispositivo (OCT, iPhone y Samsung) para el grado introducido como parámetro.
    
    Parámetros
    ----------------------------------------------------------------------
    grado: string que puede tomar valor 'uno','dos','tres','cuatro' o 'cinco' y que indica el grado de retinopatía para el cual se desea obtener la lista.
    
    Return
    ----------------------------------------------------------------------
    Devuelve 3 listas:
        img_retin: lista que contiene los nombres de las imágenes de OCT con el grado pasado como parámetro.
        img_samsung: lista que contiene los nombres de las imágenes de Samsung con el grado pasado como parámetro.
        img_iphone: lista que contiene los nombres de las imágenes de iPhone con el grado pasado como parámetro.
    '''
    NHC_retin, NHC_samsung, NHC_iphone = listas_NHC(grado)
    
    img_retin = []
    for i in NHC_retin:
        df = retin[retin['NHC']== i]
        if 2 in df[df['GRADO RETINOPATÍA DIABÉTICA'] == grado]['lateralidad 1 Dch 2 izq'].values:
            img_retin.append(str(i) + 'TI.jpg')
        if 1 in df[df['GRADO RETINOPATÍA DIABÉTICA'] == grado]['lateralidad 1 Dch 2 izq'].values:
            img_retin.append(str(i) + 'TD.jpg')
            
    img_samsung = []
    for i in NHC_samsung:
        df = samsung[samsung['NHC']== i]
        if 2 in df[df['GRADO RETINOPATÍA DIABÉTICA'] == grado]['lateralidad 1 Dch 2 izq'].values:
            img_samsung.append(str(i) + 'GI.png')
        if 1 in df[df['GRADO RETINOPATÍA DIABÉTICA'] == grado]['lateralidad 1 Dch 2 izq'].values:
            img_samsung.append(str(i) + 'GD.png')
            
    img_iphone = []
    for i in NHC_iphone:
        df = iphone[iphone['NHC']== i]
        if 2 in df[df['GRADO RETINOPATÍA DIABÉTICA'] == grado]['lateralidad 1 Dch 2 izq'].values:
            img_iphone.append(str(i) + 'EI.PNG')
        if 1 in df[df['GRADO RETINOPATÍA DIABÉTICA'] == grado]['lateralidad 1 Dch 2 izq'].values:
            img_iphone.append(str(i) + 'ED.PNG')
            
    return img_retin, img_samsung, img_iphone

#obtenemos todas las listas con los nombres de las imágenes para cada grado y dispositivo (OCT, iPhone y Samsung)
img_retin_1, img_samsung_1, img_iphone_1 = nombre_imagenes('uno')
img_retin_2, img_samsung_2, img_iphone_2 = nombre_imagenes('dos')
img_retin_3, img_samsung_3, img_iphone_3 = nombre_imagenes('tres')
img_retin_4, img_samsung_4, img_iphone_4 = nombre_imagenes('cuatro')
img_retin_5, img_samsung_5, img_iphone_5 = nombre_imagenes('cinco')

#cargamos los archivos de los directorios de imágenes
fotos_OCT = os.listdir('Raw Data\FOTOS OCT')
fotos_iphone = os.listdir('Raw Data\FOTOS iPhone')
fotos_samsung = os.listdir('Raw Data\FOTOS Samsung')

#ahora vamos a recorrer los elementos de cada directorio y meterlos en una u otra carpeta según sea su grado
for i in fotos_OCT:
    if i[-4:] == '.jpg':
        if i in img_retin_1:
            shutil.move('Raw Data\\FOTOS OCT\\' + i, 'Classified Data\\Images\\OCT\\G1\\' + i)
        elif i in img_retin_2:
            shutil.move('Raw Data\\FOTOS OCT\\' + i, 'Classified Data\\Images\\OCT\\G2\\' + i)
        elif i in img_retin_3:
            shutil.move('Raw Data\\FOTOS OCT\\' + i, 'Classified Data\\Images\\OCT\\G3\\' + i)
        elif i in img_retin_4:
            shutil.move('Raw Data\\FOTOS OCT\\' + i, 'Classified Data\\Images\\OCT\\G4\\' + i)
        elif i in img_retin_5:
            shutil.move('Raw Data\\FOTOS OCT\\' + i, 'Classified Data\\Images\\OCT\\G5\\' + i)
        else:
            os.remove('Raw Data\\FOTOS OCT\\' + i)


for i in fotos_iphone:
    if i[-4:] == '.PNG':
        if i in img_iphone_1:
            shutil.move('Raw Data\\FOTOS iPhone\\' + i, 'Classified Data\\Images\\iPhone\\G1\\' + i)
        elif i in img_iphone_2:
            shutil.move('Raw Data\\FOTOS iPhone\\' + i, 'Classified Data\\Images\\iPhone\\G2\\' + i)
        elif i in img_iphone_3:
            shutil.move('Raw Data\\FOTOS iPhone\\' + i, 'Classified Data\\Images\\iPhone\\G3\\' + i)
        elif i in img_iphone_4:
            shutil.move('Raw Data\\FOTOS iPhone\\' + i, 'Classified Data\\Images\\iPhone\\G4\\' + i)
        elif i in img_iphone_5:
            shutil.move('Raw Data\\FOTOS iPhone\\' + i, 'Classified Data\\Images\\iPhone\\G5\\' + i)
        else:
            os.remove('Raw Data\\FOTOS iPhone\\' + i)


for i in fotos_samsung:
    if i[-4:] == '.png':
        if i in img_samsung_1:
            shutil.move('Raw Data\FOTOS Samsung\\' + i, 'Classified Data\\Images\\Samsung\\G1\\' + i)
        elif i in img_samsung_2:
            shutil.move('Raw Data\FOTOS Samsung\\' + i, 'Classified Data\\Images\\Samsung\\G2\\' + i)
        elif i in img_samsung_3:
            shutil.move('Raw Data\FOTOS Samsung\\' + i, 'Classified Data\\Images\\Samsung\\G3\\' + i)
        elif i in img_samsung_4:
            shutil.move('Raw Data\FOTOS Samsung\\' + i, 'Classified Data\\Images\\Samsung\\G4\\' + i)
        elif i in img_samsung_5:
            shutil.move('Raw Data\FOTOS Samsung\\' + i, 'Classified Data\\Images\\Samsung\\G5\\' + i)
        else:
            os.remove('Raw Data\FOTOS Samsung\\' + i)


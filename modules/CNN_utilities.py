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
    Además de llevar a cabo el entrenamiento, también es capaz de almacenar las métricas del entrenamiento (loss y accuracy).
    Esta función NO tiene en cuenta el conjunto de datos de validación en el entrenamiento, por lo que NO incluye estrategias de "early stopping".
    
    Parámetros
    ------------------------------------------------------------------------
    red: instancia del modelo de CNN que se desea entrenar. 
    epocas: valor entero que representa el número de épocas que se desea entrenar el modelo.
    train_loader: función de tipo DataLoader() que irá generando los lotes de imágenes con sus respectivas etiquetas usados en el entrenamiento de la red.
    optimizer: función optimizadora de la red (encargada de modificar el valor de los parámetros y pesos de la red en base a la diferencia entre etiquetas reales y etiquetas predichas). Suele ser una función perteneciente al paquete torch.optim
    criterion: función de tipo "loss" que nos permite conocer cómo de alejadas están las predicciones de la red con respecto a las etiquetas reales. Suele pertenecer al módulo nn.
    
    Return
    ------------------------------------------------------------------------
    Devuelve 2 elementos: una lista que contiene los valores de accuracy a lo largo de las épocas (acc_graph) y una lista que devuelve los valores del loss a lo largo de las épocas (loss_graph).
    '''
    #primero importamos las dependencias necesarias para el funcionamiento de la función
    import torch as torch
    #definimos 2 listas en las que almacenaremos los valores de accuracy y loss de cada época para devolverlos
    acc_graph = []
    loss_graph = []
    #para entrenar el modelo vamos a iterar el número de épocas determinadas, calculando el valor de loss y accuracy para cada época
    for epoch in range(epocas):
        #establecemos el número de predicciones correctas inicial a 0
        correct = 0
        #y el acumulador de loss a 0.0
        train_loss = 0.0
        #y cargamos las imágenes de entrenamiento y sus etiquetas usando la estructura Loader pasada como parámetro
        for i, data in enumerate(train_loader):
            #obtenemos las imágenes y etiquetas del lote por separado
            inputs, labels = data
            #establecemos a 0 los parámetros del modelo
            optimizer.zero_grad()
            #generamos las predicciones a partir de los inputs
            outputs = red(inputs)
            #calculamos el loss, la desviación de las predicciones con respecto a las etiquetas
            loss = criterion(outputs, labels)
            #propagamos hacia atrás el valor loss
            loss.backward()
            #y modificamos los parámetros y pesos en función del loss y la función optimizer
            optimizer.step()
            #actualizamos el número de predicciones correctas
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            #añadimos el valor de loss al acumulador train_loss
            train_loss += loss.item()

        #una vez finalizada la época (que recorre todo el conjunto de imágenes) mostramos el valor del loss y del accuracy
        print(f'Época {epoch +1}/{epocas} - Accuracy: {correct/len(train_loader.dataset)} - Loss: {train_loss/len(train_loader.dataset)}')
        #añadimos los valores a la lista correspondiente
        loss_graph.append(train_loss/len(train_loader.dataset))
        acc_graph.append(correct/len(train_loader.dataset))
        
    #devolvemos los valores de loss y accuracy almacenados
    return acc_graph, loss_graph

def entrena_val(red,epocas,paciencia,train_loader,val_loader,optimizer,criterion):
    '''
    Realiza el entrenamiento completo de la red pasada como parámetro, empleando para ello las imágenes de entrenamiento proporcionadas por la estructura "loader" también pasada como parámetro.
    Además de llevar a cabo el entrenamiento, también es capaz de almacenar las métricas del entrenamiento (loss y accuracy).
    Esta función tiene en cuenta el conjunto de datos de validación en el entrenamiento, incluyendo estrategias de "early stopping" que evitan el sobreentrenamiento.
    
    Parámetros
    ------------------------------------------------------------------------
    red: instancia del modelo de CNN que se desea entrenar. 
    epocas: valor entero que representa el número de épocas que se desea entrenar el modelo.
    paciencia: valor entero que representa el número de épocas consecutivas con valores de val_loss en aumennto que espera la función antes de detener el entrenamiento.
    train_loader: función de tipo DataLoader() que irá generando los lotes de imágenes con sus respectivas etiquetas usados en el entrenamiento de la red.
    val_loader: función de tipo DataLoader() que irá generando los lotes de imágenes con sus respectivas etiquetas usados en la validación de la red.
    optimizer: función optimizadora de la red (encargada de modificar el valor de los parámetros y pesos de la red en base a la diferencia entre etiquetas reales y etiquetas predichas). Suele ser una función perteneciente al paquete torch.optim
    criterion: función de tipo "loss" que nos permite conocer cómo de alejadas están las predicciones de la red con respecto a las etiquetas reales. Suele pertenecer al módulo nn.
    
    Return
    ------------------------------------------------------------------------
    Devuelve 4 elementos: una lista que contiene los valores de accuracy de entrenamiento a lo largo de las épocas (acc_graph), una lista que devuelve los valores del loss de entrenamiento a lo largo de las épocas (loss_graph), una lista que contiene los valores de accuracy de validación a lo largo de las épocas (val_acc_graph) y una lista que devuelve los valores del loss de validación a lo largo de las épocas (val_loss_graph).
    '''
    #primero importamos los paquetes necesarios
    import torch
    #inicializamos best_val_loss (que es el parámetro que va a marcar el Early Stopping) como infinito
    best_val_loss = float('inf')
    #creamos también una variable para almacenar los parámetros del mejor modelo (aquel con menor val_loss)
    best_model_params = None
    #iniciamos también un contador, para poder aplicar Early Stopping con la paciencia deseada
    contador = 1
    #definimos 2 listas en las que almacenaremos los valores de accuracy y loss de train cada época para devolverlas
    acc_graph = []
    loss_graph = []
    #y 2 listas en las que almacenaremos los valores de accuracy y loss de validación cada época para poder devolverlas
    val_acc_graph = []
    val_loss_graph = []
    
    #para entrenar el modelo vamos a iterar el número de épocas determinadas, calculando el valor de loss y accuracy para cada época
    for epoch in range(epocas):
        #establecemos el número de predicciones correctas inicial a 0
        correct = 0
        #y el acumulador de loss a 0.0
        train_loss = 0.0
        #y cargamos las imágenes de entrenamiento y sus etiquetas usando la estructura Loader previamente creada
        for data in train_loader:
            inputs, labels = data
            #establecemos a 0 los parámetros del modelo
            optimizer.zero_grad()
            #generamos las predicciones de los inputs
            outputs = red(inputs)
            #calculamos el loss, la desviación de las predicciones con respecto a las etiquetas
            loss = criterion(outputs, labels)
            #propagamos hacia atrás el valor loss
            loss.backward()
            #y modificamos los pesos en función del loss y la función optimizer
            optimizer.step()
            #actualizamos el número de predicciones correctas
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            #añadimos el valor de loss al acumulador train_loss
            train_loss += loss.item()
            
        #una vez finalizada la época (que recorre todo el conjunto de imágenes) mostramos el valor del loss y del accuracy
        print(f'Época {epoch +1}/{epocas} - Accuracy: {correct/len(train_loader.dataset)} - Loss: {train_loss/len(train_loader.dataset)}')
        #añadimos los valores a la lista correspondiente
        loss_graph.append(train_loss/len(train_loader.dataset))
        acc_graph.append(correct/len(train_loader.dataset))

        #realizamos ahora las iteraciones correspondientes a las imágenes de validación
        #primero establecemos el valor del loss de validación a cero
        val_loss = 0.0
        #establecemos así mismo el número de predicciones correctas nuevamente a cero
        correct = 0
        #cargamos las imágenes de validación y sus etiquetas
        for data in val_loader:
            inputs, labels = data
            #generamos las predicciones a partir de los inputs
            outputs = red(inputs)
            #calculamos el loss
            loss = criterion(outputs, labels)
            #y lo vamos acumulando
            val_loss += loss.item()
            #finalmente calculamos el número de predicciones correctas
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
        
        #una vez finalizada la época (que recorre todo el conjunto de imágenes) mostramos el valor del loss y del accuracy de validación
        print(f'Época {epoch +1}/{epocas} - Val_accuracy: {correct/len(val_loader.dataset)} - Val_loss: {val_loss/len(val_loader.dataset)}\n')
        #añadimos los valores a la lista correspondiente
        val_loss_graph.append(val_loss/len(val_loader.dataset))
        val_acc_graph.append(correct/len(val_loader.dataset))
        
        #finalmente solo falta realizar la comprobación del Early Stopping
        #si el valor de val_loss de esta época es inferior al mejor conseguido hasta el momento:
        if (val_loss/len(val_loader.dataset)) < best_val_loss:
            #entonces actualiza el valor del mejor val_loss (ya que lo que queremos es minimizar este valor)
            best_val_loss = val_loss/len(val_loader.dataset)
            #posteriormente guarda el estado del modelo actual
            best_model_params = red.state_dict()
            #y vuelve a establecer el contador de paciencia a 1
            contador = 1
        #si el valor de val_loss no disminuye (no mejora) con respecto al último mejor:
        else:
            #si se ha llegado al límite de la paciencia establecida detiene el entrenamiento para evitar el sobreentrenamiento
            if contador == paciencia:
                break
            #si aún no ha llegado al límite de la paciencia entonces incrementa el contador en uno y sigue entrenando
            else:
                contador += 1
        
        #finalmente, una vez finalizado el entrenamiento se debe cargar el mejor estado del modelo
        red.load_state_dict(best_model_params)
        
    #y devolver las métricas almacenadas de entrenamiento y validación
    return acc_graph, loss_graph, val_acc_graph, val_loss_graph

def representa_train(valores,metrica,red):
    '''
    Genera y muestra una gráfica en la que se representa la evolución de una determinada métrica de entrenamiento de una red (Accuracy o Loss) a lo largo de las épocas.
    
    Parámetros
    ------------------------------------------------------------------------
    valores: lista que contiene los valores de la métrica a representar. Estos valores deben oscilar entre 0 y 1.
    metrica: String que indica qué métrica es la que se desea representar. Puede tomar dos valores: Accuracy o Loss.
    red: String que indica el nombre de la red de cuyo entrenamiento se han obtenido los valores de la métrica. Se incluye en el título de la gráfica para poder identificarla.
    
    Return
    ------------------------------------------------------------------------
    La función no devuelve ningún valor
    '''
    #primero creamos la figura que se va a representar y asignamos dimensiones
    plt.figure(figsize = (10,7))
    #incluimos un título en el que se muestre la red y la métrica que se va a representar
    plt.title(red + ' - Evolución del ' + metrica)
    #asignamos título a los ejes:
    #en el eje X se representarán las épocas
    plt.xlabel('Nº Época')
    #en el eje Y los valores de la métrica
    plt.ylabel(metrica)
    #si la métrica es Accuracy la representamos de color azul, si es loss de color naranja
    if metrica == 'Accuracy':
        color = 'blue'
    elif metrica == 'Loss':
        color == 'orange'
    #por último representamos estos valores
    plt.plot(range(1,len(valores)+1),valores, color = color, linewidth = 3)
    #y mostramos la figura
    plt.show()
    
    
def tester(red,loader):
    '''
    Realiza el test de la red usando el conjunto de imágenes deseado (iPhone o Samsung) y devuelve las etiquetas reales y predichas para el cálculo de las métricas básicas (matriz de confusión, accuracy, balanced accuracy, F-score, Quadratic Weighted Kappa y AUC).
    
    Parámetros
    ------------------------------------------------------------------------
    red: instancia del modelo de CNN que se desea probar.
    loader: función de tipo DataLoader() que irá generando los lotes de imágenes con sus respectivas etiquetas usados en el testeo de la red.
    
    Return
    ------------------------------------------------------------------------
    y_true: array unidimensional de numpy que contiene al lista de etiquetas reales obtenidas de la función DataLoader para cada imagen.
    y_pred: array unidimensional de numpy que contiene al lista de etiquetas predichas por el modelo para cada imagen.
    predictions: array de numpy que contiene la probabilidad de pertenencia a cada clase para cada imagen proporcionada por la red. Se emplea en el posterior cálculo de AUC.
    '''
    #realizamos las importaciones necesarias 
    import torch
    import torch.nn as nn
    import numpy as np
    #creamos las 2 listas para almacenar las etiquetas reales y las etiquetas predichas
    y_true = []
    y_pred = []
    #creamos una para almacenar también la salida del modelo transformada a probabilidad, para posteriormente poder calcular AUC
    predictions = []
    #y definimos una función para convertir la salida a forma de probabilidad (usando la función Softmax)
    #el parámetro dim=1 indica que la conversión se debe hacer en el eje de las filas (la suma de las probabilidad en una fila debe sumar 1)
    m = nn.Softmax(dim=1)
    #es importante activar torch.no_grad() para que la red no entrene al pasarle el conjunto de test, no varíen los pesos
    with torch.no_grad():
        #recorremos el conjunto de imágenes de test de iPhone
        for data in loader:
            images, labels = data #cargamos las imágenes y las etiquetas del dataloader
            outputs = red(images) #obtenemos las predicciones
            predictions.append(m(outputs).numpy()) #las convertimos a probabilidad mediante Softmax 
            _, predicted = torch.max(outputs.data,1) #y obtenemos las etiquetas o labels predichas a partir de la probabilidad
            y_pred.append(predicted.numpy()) #añadimos la predicción a la lista de predicciones
            y_true.append(labels.numpy()) #y añadimos la etiqueta real a la lista de etiquetas reales
    #convertimos los datos a formato np.array de una única dimensión        
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    predictions = np.concatenate(predictions)
    
    return y_true,y_pred,predictions

def representa_test(y_true, y_pred, predictions, test, red):
    '''
    Calcula los valores de las siguientes métricas: matriz de confusión, accuracy, balanced accuracy, F-score, Quadratic Weighted Kappa y AUC, y las muestra por pantalla.
    
    Parámetros
    ------------------------------------------------------------------------
    y_true: array unidimensional de numpy que contiene al lista de etiquetas reales.
    y_pred: array unidimensional de numpy que contiene al lista de etiquetas predichas.
    predictions: array de numpy que contiene la probabilidad de pertenencia a cada clase para cada imagen. Se emplea para el cálculo de AUC.
    test: String que indica cuál ha sido el conjunto de imágenes empleado para el test. Típicamente puede tomar 2 valores: iPhone o Samsung.
    red: String que identifica el nombre de la red empleada en el testeo.
    
    Return
    ------------------------------------------------------------------------
    metricas: estructura de tipo lista que contiene las métricas calculadas en el siguiente orden: matriz de confusión, accuracy, balanced accuracy, F-score, Quadratic Weighted Kappa y AUC.
    '''
    #en este caso vamos a realizar una importación de las dependencias necesarias, ya que puede que no estén importadas en el script
    import sklearn
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, cohen_kappa_score, roc_auc_score, confusion_matrix
    import seaborn as sns
    
    #indicamos la red empleada en la prueba y obtención de métricas
    print(f'Métricas obtenidas con el modelo {red}\n')
    
    #primero obtenemos la matriz de confusión
    matrix = confusion_matrix(y_true, y_pred)
    #usamos el paquete seaborn para mostrar de manera más visual la matriz de confusión
    plot = sns.heatmap(matrix, annot = True, cmap = 'Reds', cbar = False)
    #establecemos título
    plot.set_title('Matriz de confusión - '+ test +'\n')
    #título de cada eje
    plot.set_xlabel('\nGrado ' + test)
    plot.set_ylabel('Grado real\n')
    #y el significado de cada fila y columna de la matriz
    plot.xaxis.set_ticklabels(['Grado1','Grado2','Grado3','Grado4','Grado5'])
    plot.yaxis.set_ticklabels(['Grado1','Grado2','Grado3','Grado4','Grado5'])
    print(plot)

    #calculamos el valor de accuracy
    accuracy = accuracy_score(y_true = y_true, y_pred = y_pred)
    print(f'El valor de accuracy del modelo con imágenes de {test} es: {accuracy}')
    #el balanced accuracy
    bal_acc = balanced_accuracy_score(y_true = y_true, y_pred = y_pred)
    print(f'El valor de balanced accuracy del modelo con imágenes de {test} es: {bal_acc}')
    #el F-score
    f_score = f1_score(y_true = y_true, y_pred = y_pred,average = 'weighted')
    print(f'El valor de F-score del modelo con imágenes de {test} es: {f_score}')
    #calculamos el valor de quadratic weighted kappa
    kappa = cohen_kappa_score(y1 = y_true, y2 = y_pred)
    print(f'El valor de Kappa del modelo con imágenes de {test} es: {kappa}')
    #y por último calculamos el valor de AUC bajo la curva ROC, importante indicar que se trata de un problema "ovr" (One vs Rest)
    auc = roc_auc_score(y_true = y_true, y_score = predictions, multi_class = 'ovr')
    print(f'El valor de AUC del modelo con imágenes de {test} es: {auc}')
    
    #finalmente creamos la lista de métricas y la devolvemos
    metricas = [matrix, accuracy, bal_acc, f_score, kappa, auc]
    return metricas

def obtiene_metricas(y_true, y_pred, predictions):
    '''
    Calcula los valores de las siguientes métricas: matriz de confusión, accuracy, balanced accuracy, F-score, Quadratic Weighted Kappa y AUC.
    
    Parámetros
    ------------------------------------------------------------------------
    y_true: array unidimensional de numpy que contiene al lista de etiquetas reales.
    y_pred: array unidimensional de numpy que contiene al lista de etiquetas predichas.
    predictions: array de numpy que contiene la probabilidad de pertenencia a cada clase para cada imagen. Se emplea para el cálculo de AUC.
    
    Return
    ------------------------------------------------------------------------
    metricas: estructura de tipo lista que contiene las métricas calculadas en el siguiente orden: matriz de confusión, accuracy, balanced accuracy, F-score, Quadratic Weighted Kappa y AUC.
    '''
    #en este caso vamos a realizar una importación de las dependencias necesarias, ya que puede que no estén importadas en el script
    import sklearn
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, cohen_kappa_score, roc_auc_score, confusion_matrix
    import seaborn as sns
    
    #primero obtenemos la matriz de confusión
    matrix = confusion_matrix(y_true, y_pred)
    #calculamos el valor de accuracy
    accuracy = accuracy_score(y_true = y_true, y_pred = y_pred)
    #el balanced accuracy
    bal_acc = balanced_accuracy_score(y_true = y_true, y_pred = y_pred)
    #el F-score
    f_score = f1_score(y_true = y_true, y_pred = y_pred,average = 'weighted')
    #calculamos el valor de quadratic weighted kappa
    kappa = cohen_kappa_score(y1 = y_true, y2 = y_pred)
    #y por último calculamos el valor de AUC bajo la curva ROC, importante indicar que se trata de un problema "ovr" (One vs Rest)
    auc = roc_auc_score(y_true = y_true, y_score = predictions, multi_class = 'ovr')
    
    #finalmente creamos la lista de métricas y la devolvemos
    metricas = [matrix, accuracy, bal_acc, f_score, kappa, auc]
    return metricas

def guarda_graficas(imagenes,validacion,preproc,inpaint,color,arq,capas,filtros,neuronas,acc,loss,val_acc = 0,val_loss = 0):
    '''
    Genera las gráficas que representan la evolución de las métricas accuracy, loss, val_accuracy y val_loss a lo largo de las épocas de entrenamiento de un modelo. Además de generar estas gráficas posteriormente las almacena.
    
    Parámetros
    ------------------------------------------------------------------------
    imagenes: String que indica qué conjunto de imágenes se han usado en el entrenamiento. Puede tomar el valor "OCT" si las imágenes usadas fueron únicamente las proporcionadas por el HUBU o "BigData" si son las tomadas de los repositorios.
    validacion: String que indica si en el entrenamiento se ha usado validación o no. Puede tomar el valor "Si" o "No".
    preproc: String que indica si en el entrenamiento se han usado imágenes preprocesadas o no. Puede tomar el valor "Si" o "No".
    inpaint: String que indica si en el entrenamiento se han usado imágenes inpaintadas o no. Puede tomar el valor "Si" o "No".
    color: String que indica si las imágenes usadas eran en blanco y negro (bn) o a color (RGB).
    arq: String que indica la arquitectura empleada en la construcción del modelo (Alqudah, Ghosh, Mobeen, Rajagopalan o Basica).
    capas: entero que representa el número de capas convolucionales del modelo.
    filtros: float que representa el factor por el que se han multiplicado el número de filtros de cada capa convolucional.
    neuronas: String que representa el número de neuronas de las últimas capas fully-connected del modelo.
    acc: lista que contiene los valores del accuracy a lo largo de las distintas épocas de entrenamiento.
    loss: lista que contiene los valores del loss a lo largo de las distintas épocas de entrenamiento.
    val_acc: lista que contiene los valores del val_accuracy a lo largo de las distintas épocas de entrenamiento.
    val_loss: lista que contiene los valores del val_loss a lo largo de las distintas épocas de entrenamiento.

    Return
    ------------------------------------------------------------------------
    No devuelve ningún valor.
    '''
    #primero realizamos las importaciones necesarias
    import matplotlib
    from matplotlib import pyplot as plt
    
    #creamos la figura que representará el accuracy y loss, asignándole el tamaño deseado
    plt.figure(figsize = (10,10))
    #creamos las marcas correspondientes en cada eje
    plt.xticks(range(1,len(acc)+1))
    plt.yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
    #representamos el accuracy y el loss asignándole un color y etiqueta a cada representación
    plt.plot(range(1,len(acc)+1),acc,color = 'red',label = 'Accuracy')
    plt.plot(range(1,len(loss)+1),loss,color = 'blue', label = 'Loss')
    #damos el nombre a los ejes
    plt.xlabel('\nNº época',fontsize = 12)
    plt.ylabel('Valor\n',fontsize = 12)
    #y el título a la gráfica
    plt.title(f'Entrenamiento {arq}, capas: {capas}, filtros: {filtros}, neuronas: {neuronas}\nDataset: {imagenes}, validacion: {validacion}, preprocesamiento: {preproc}, inpainting: {inpaint}, color: {color}',fontsize = 15)
    #incluimos la leyenda
    plt.legend()
    #finalmente guardamos la figura
    #primero debemos redefinir las cadenas de filtros y neuronas para eliminar los puntos y barras laterales que darán problemas en el nombre
    filtros_str = str(filtros).replace(".","punto")
    neuronas_str = str(neuronas).replace("/","slash")
    plt.savefig(f'graficas/{arq}/Entrenamiento_{imagenes}_{validacion}val_{preproc}prep_{inpaint}inp_{color}_{capas}_{filtros_str}_{neuronas_str}.png',dpi = 300)
    #y cerramos la figura
    plt.close()
    
    #ahora debemos repetir el proceso pero para las gráficas de validación
    if val_acc != 0 and val_loss != 0:
        #creamos la figura que representará el accuracy y loss, asignándole el tamaño deseado
        plt.figure(figsize = (10,10))
        #creamos las marcas correspondientes en cada eje
        plt.xticks(range(1,len(val_acc)+1))
        plt.yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
        #representamos el accuracy y el loss asignándole un color y etiqueta a cada representación
        plt.plot(range(1,len(val_acc)+1),val_acc,color = 'orange',label = 'Val Accuracy')
        plt.plot(range(1,len(val_loss)+1),val_loss,color = 'cyan', label = 'Val Loss')
        #damos el nombre a los ejes
        plt.xlabel('\nNº época',fontsize = 12)
        plt.ylabel('Valor\n',fontsize = 12)
        #y el título a la gráfica
        plt.title(f'Validación {arq}, capas: {capas}, filtros: {filtros}, neuronas: {neuronas}\nDataset: {imagenes}, validacion: {validacion}, preprocesamiento: {preproc}, inpainting: {inpaint}, color: {color}',fontsize = 15)
        #incluimos la leyenda
        plt.legend()
        #finalmente guardamos la figura
        plt.savefig(f'graficas/{arq}/Validacion_{imagenes}_{validacion}val_{preproc}prep_{inpaint}inp_{color}_{capas}_{filtros_str}_{neuronas_str}.png',dpi = 300)
        #y cerramos la figura
        plt.close()
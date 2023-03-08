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
    #definimos 2 listas en las que almacenaremos los valores de accuracy y loss de cada época para devolverlos
    acc_graph = []
    loss_graph = []
    #para entrenar el modelo vamos a iterar el número de épocas determinadas, calculando el valor de loss y accuracy para cada época
    for epoch in range(epocas):
        #establecemos el número de predicciones correctas inicial a 0
        correct = 0
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

        #una vez finalizada la época (que recorre todo el conjunto de imágenes) mostramos el valor del loss y del accuracy
        print(f'Época {epoch +1}/{epocas} - Accuracy: {correct/len(train_loader.dataset)} - Loss: {loss.data.item()}')
        #añadimos los valores a la lista correspondiente
        loss_graph.append(loss.data.item())
        acc_graph.append(correct/len(train_loader.dataset))
        
    #devolvemos los valores de loss y accuracy almacenados
    return acc_graph, loss_graph

def representa(valores,metrica,red):
    '''
    Genera y muestra una gráfica en la que se representa la evolución de una determinada métrica de entrenamiento de una red (Accuracy o Loss) a lo largo de las épocas.
    
    Parámetros
    ------------------------------------------------------------------------
    valores: lista que contiene los valores de la métrica a representar. Estos valores deben oscilar entre 0 y 1.
    metrica: String que indica qué métrica es la que se desea representar
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
        color == 'orange
    #por último representamos estos valores
    plt.plot(range(1,len(valores)+1),valores, color = color linewidth = 3)
    #y mostramos la figura
    plt.show()
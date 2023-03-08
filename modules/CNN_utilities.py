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
        #y cargamos las imágenes de entrenamiento y sus etiquetas usando la estructura Loader previamente creada
        for i, data in enumerate(train_loader):
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

        #una vez finalizada la época (que recorre todo el conjunto de imágenes) mostramos el valor del loss y del accuracy
        print(f'Época {epoch +1}/{epocas} - Accuracy: {correct/len(OCT)} - Loss: {loss.data.item()}')
        #añadimos los valores a la lista correspondiente
        loss_graph.append(loss.data.item())
        acc_graph.append(correct/len(OCT))

    return acc_graph, loss_graph
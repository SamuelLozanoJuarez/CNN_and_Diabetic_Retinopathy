'''
#########################################################################################################################
INFORMACIÓN DEL FICHERO
#########################################################################################################################
Autor: Samuel Lozano Juárez
Fecha: 09/05/2023
Institución: UBU | Grado en Ingeniería de la Salud

Este archivo forma parte del Trabajo de Fin de Grado "Detección del grado de retinopatía mediante redes convolucionales".
El alumno a cargo de este proyecto es el declarado como autor en las líneas anteriores.
Los tutores del proyecto fueron el Dr. Darío Fernández Zoppino y el Dr. Daniel Urda Muñoz.

En el código que se encuentra a continuación se define la función que permite realizar el inpainting de una imagen, así como las líneas necesarias para ejecutar esta función sobre todas las imágenes de nuestros directorios.

Para llevar a cabo el proceso de inpainting me he guiado por el ejemplo proporcionado por la propia librería Scikit-image: https://scikit-image.org/docs/stable/auto_examples/filters/plot_inpaint.html .

Para la elaboración de la máscara me he inspirado en el ejemplo proporcionado por OpenCV para la búsqueda de círculos en imágenes: https://docs.opencv.org/4.x/da/d53/tutorial_py_houghcircles.html .
'''

#primero importamos los paquetes necesarios
import os
import skimage
import numpy as np
import cv2
import matplotlib
import pyinpaint
import PIL

from skimage import io
from matplotlib import pyplot as plt
from skimage import color
from pyinpaint import Inpaint
from skimage.restoration import inpaint
from PIL import Image

#definimos la función que permite aplicar el proceso de inpainting sobre un archivo concreto
def inpainting(dispositivo,grado,archivo):
    '''
    Dada una imagen de un fondo de retina obtenida con Samsung o iPhone, detecta los destellos de flash y procesa la imagen para eliminar dichos destellos, aplicando una técnica de inpainting biharmónico. Posteriormente guarda la imagen correspondiente.
    
    Parámetros
    ---------------------------------------------------
    dispositivo: String que representa el dispositivo con el cual se tomó la imagen (iPhone o Samsung)
    grado: String correspondiente al grado de retinopatía diabética con que se ha etiquetado la imagen (G1, G2, G3, G4, G5)
    archivo: String del nombre del archivo, con su extensión incluida.
    
    Return
    ---------------------------------------------------
    La función no devuelve ningún elemento.
    '''
    #obtiene la imagen deseada desde el directorio concreto
    img = io.imread('Classified Data/Images/' + dispositivo + '/No_inpaint/' + grado + '/' + archivo)
    #Convertimos la imagen a escala de grises
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #La desenfocamos usando un filtro de tamaño 3 por 3 píxeles
    gray_blurred = cv2.blur(gray, (3, 3))

    #Usamos la función HoughCircles para obtener las coordenadas de los círculos detectados en la imagen
    detected_circles = cv2.HoughCircles(
        gray_blurred, #este es el nombre de la imagen sobre la que se desea invocar la función
        cv2.HOUGH_GRADIENT, #esta es el método de detección de círculos que se desea emplear
        2.2, #relación inversa entre la resolución de la imagen y la resolución del resultado
        20, #distancia mínima entre círculos para ser considerados como distintos
        param1 = 50, #parámetro específico del método
        param2 = 30, #parámetro específico del método
        minRadius = 1, #radio mínimo de un círculo para ser detectado
        maxRadius = int(img.shape[0]/53) #radio máximo de un círculo para ser reconocido (se seleccionó 1/53 del ancho total de la imagen, para evitar que se reconocieran la fóvea o el disco óptico como círculos)
    )
    #inicializamos la máscara como una imagen en negro (una matriz conformada únicamente por ceros)
    mask = np.zeros((img.shape[0], img.shape[1]), dtype = 'uint8')
    #inicializamos la figura en la que almacenaremos la máscara con los círculos identificados
    fig = plt.figure(figsize = (10,10))
    
    #comprobamos si se han detectado círculos
    if detected_circles is not None:
        #redondeamos las coordenadas de esos círculos detectados (a números enteros de 16 bits)
        detected_circles = np.uint16(np.around(detected_circles))
        #inicializamos una lista para ir almacenando los círculos encontrados que cumplan determinadas características
        nuevos = []
        #recorremos las coordenadas de los círculos
        for i in detected_circles[0]:
            #comprobamos que el centro del círculo esté situado en la zona central de la imagen (entre los 11/30 y 18/30 vertical, y los 13/30 y 17/30 horizontal)
            if 11*(img.shape[1]/30)<i[1]<18*(img.shape[1]/30) and 13*(img.shape[0]/30)<i[0]<17*(img.shape[0]/30):
                #si cumple estas características añadimos ese círculo a la lista de nuevos
                nuevos.append(i)
        #creamos una línea de 400 puntos, desde 0 hasta 2*pi
        s = np.linspace(0, 2*np.pi, 400)
        #mostramos en la figura previamente creada la máscara
        plt.imshow(mask, cmap = 'gray')
        #ocultamos los ejes
        plt.axis('off')

        #para cada uno de los círculos que han pasado el filtro vamos a crear una lista de coordenadas para dibujar los círculos
        for i in nuevos:
            #creamos primero las coordenadas X de aquellos puntos que van a definir la circunferencia, obteniendo el coseno de la línea de puntos definida previamente (lo que acaba generando la componente X de la circunferencia) y lo multiplico por el radio y por 3.
            #lo multiplico por 3 para aumentar el tamaño del círculo (por lo que no quedaría una máscara perfecta, sino ligeramente más grande), pero esto lo hago con el objetivo de mejorar los resultados del inpainting
            comp_x = i[0] + (3*i[2])*np.cos(s)
            #hacemos el mismo proceso pero en este caso hallando el seno, lo que se corresponde con la coordenada Y de cada punto de la circunferencia
            comp_y = i[1] + (3*i[2])*np.sin(s)
            #la circunferencia estará definida por esos puntos delimitados por su coordenada X y su coordenada Y
            circle = np.array([comp_y, comp_x]).T
            #rellenamos esa circunferencia para obtener el círculo
            plt.fill(circle[:,1], circle[:,0],'white',1)
    
    #eliminamos cualquier espacio en blanco en los bordes de la figura
    fig.tight_layout(pad=0)
    #actualizamos el estado de la figura
    fig.canvas.draw()

    #Convertimos la figura obtenida en un array de numpy, en RGB donde cada píxel está representado por un número entero de 8 bits
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    #redimensionamos la imagen
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    #volvemos a redimensionar la imagen pero en este caso usamos interpolación para ajustar la imagen al nuevo tamaño
    mascara = cv2.resize(data, dsize=(img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
    #convertimos a blanco y negro nuevamente la máscara
    mascara = color.rgb2gray(mascara).astype('uint8')
    #por último aplicamos el proceso de inpainting usando inpainting biharmónico
    img_inp = inpaint.inpaint_biharmonic(img, mascara, multichannel = True, channel_axis=-1)
    #almacenamos la imagen resultante en el directorio correspondiente
    destino = 'Classified Data/Images/' + dispositivo + '/Inpaint/' + grado + '/' + archivo
    io.imsave(destino, img_inp)

#vamos a aplicar la función definida en la parte superior para aplicar el inpainting a todas las imágenes
#recorremos los posibles dispositivos sobre los que aplicarlo
for dispositivo in ['Samsung','iPhone']:
    #lo mismo con los grados
    for grado in ['G1','G2','G3','G4','G5']:
        #y con los archivos de cada subdirectorio
        for archivo in os.listdir('Classified Data/Images/' + dispositivo + '/No_inpaint/' + grado + '/'):
            #aplicamos la función
            inpainting(dispositivo, grado, archivo)


'''
#########################################################################################################################
INFORMACIÓN DEL FICHERO
#########################################################################################################################
Autor: Samuel Lozano Juárez
Fecha: 13/05/2023
Institución: UBU | Grado en Ingeniería de la Salud

Este archivo forma parte del Trabajo de Fin de Grado "Detección del grado de retinopatía mediante redes convolucionales".
El alumno a cargo de este proyecto es el declarado como autor en las líneas anteriores.
Los tutores del proyecto fueron el Dr. Darío Fernández Zoppino y el Dr. Daniel Urda Muñoz.

A continuación se proporciona el código para llevar a cabo el preprocesamiento de las imágenes aplicando diversas estrategias encontradas en referencias bibliográficas. A las imágenes se les va a aplicar un recorte para eliminar el espacio sobrante hasta los bordes y posteriormente un filtro Gaussiano para destacar las características más significativas.
'''


#importamos los paquetes necesarios
import cv2
import PIL
import numpy as np
import os

from PIL import Image

#definimos la función encargada de recortar la imagen y ajustarla a los bordes
def crop_image_from_gray(img, tol=7):
    '''
    Recorta una imagen (de fondo de ojo) para ajustarla a los bordesy eliminar el espacio de fondo sobrante. Además transforma la imagen a escala de grises.
    
    Parámetros
    -----------------------------------
    img: se trata de la imagen sobre la cual se desea aplicar la transformación. Puede ser un array de Numpy.
    tol: número entero que representa el valor mínimo que ha de tener un píxel de la imagen para no ser considerado como fondo a la hora de realizar la máscara.
    
    Return
    -----------------------------------
    Devuelve en todos los casos la imagen pasada como parámetro pero ya recortada y procesada (blanco y negro)
    '''
    #Verifica la dimensionalidad de la imagen de entrada, si es dos (blanco y negro):
    if img.ndim == 2:
        #Crea una máscara con los píxeles de la imagen que superan la tolerancia pasada como parámetro (para eliminar el fondo)
        mask = img > tol
        #Recorta la imagen utilizando la máscara y la devuelve
        return img[np.ix_(mask.any(1), mask.any(0))]
    #Si la imagen tiene tres dimensiones (RGB)
    elif img.ndim == 3:
        #Convierte la imagen a escala de grises utilizando cv2
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        #Crea una máscara con los píxeles de la imagen en escala de grises que superan la tolerancia (para eliminar el fondo)
        mask = gray_img > tol
        
        #Verifica si la forma de la imagen recortada es válida
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        #Si la forma no es válida, devuelve la imagen sin recortar
        if (check_shape == 0):
            return img
        #Si la forma es válida
        else:
            #Recorta cada canal de la imagen utilizando la máscara
            img1 = img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2 = img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3 = img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
            # Une los canales recortados en una sola imagen
            img = np.stack([img1,img2,img3],axis=-1)
        # Devuelve la imagen recortada
        return img

#por último definimos la función que integra la función anterior y realiza el desenfoque con filtro gaussiano
def circle_crop(img, sigmaX):   
    '''
    Procesa la imagen aplicando la función crop_image_from_gray para reducir el espacio hasta los bordes y posteriormente aplica un filtro gaussiano para destacar las características de la imagen.
    
    Parámetros
    -----------------------------------
    img: imagen que se quiere procesar. Puede ser un array de numpy
    sigmaX: número entero que representa el valor del parámetro sigmaX del filtro gaussiano. Este se corresponde con la desviación estandar del filtro gaussiano en el eje X.
    
    Return
    -----------------------------------
    Devuelve un array de numpy correspondiente a la imagen ya procesada.
    '''
    #primero aplicamos la función definida anteriormente para recortar la imagen y ajustarla a los bordes
    img = crop_image_from_gray(img)
    #la imagen resultante la convertimo de codificación BGR a RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #obtenemos las dimensiones de la imagen
    height, width, depth = img.shape    
    #obtenemos la mitad de cada uno de estos valores (para posteriormente poder calcular el radio del fondo de ojo)
    x = int(width/2)
    y = int(height/2)
    #el radio se corresponderá con el valor mínimo de ambos valores calculados anteriormente (la mitad del ancho y la mitad del alto de la imagen)
    r = np.amin((x,y))
    #creamos una máscara negra de las mismas dimensiones que la imagen original
    circle_img = np.zeros((height, width), np.uint8)
    #dibujamos un círculo en la máscara, tomando como centro las coordenadas (x,y) (el centro de la imagen) y como radio el valor r previamente calculado
    #rellenamos el círculo de color blanco (1)
    cv2.circle(circle_img, (x,y), int(r), 1, thickness=-1)
    #usamos el operador 'AND' para seleccionar de la imagen únicamente aquellos píxeles que se encuentren dentro de la máscara previamente creada
    img = cv2.bitwise_and(img, img, mask=circle_img)
    #volvemos a aplicar la función crop_image_from_gray para recortar los bordes
    img = crop_image_from_gray(img)
    #finalmente sumamos 2 imágenes: la imagen obtenida hasta ahora correspondiente con esos píxeles seleccionados gracias a la máscara y la imagen a la cual se le aplica un filtro gaussiano
    #para aplicar el filtro Gaussiano seleccionamos la imagen, el tamaño del kernel (0,0) que al ser 0,0 indica que el tamaño es obtenido del valor de sigma, y finalmente indicamos sigmaX 
    img=cv2.addWeighted(img,4,cv2.GaussianBlur(img,(0,0),sigmaX),-4,128)
    #devolvemos la imagen final
    return img 

def procesa(dispositivo, grado, imagen):
    '''
    Aplica el procesamiento a una imagen, realizando las transformaciones y adaptaciones previas necesarias, y posteriormente la guarda en el directorio correspondiente.
    
    Parámetros
    -----------------------------------
    dispositivo: String que representa el dispositivo con el que se tomó la imagen. Puede tomar los siguientes valores: iPhone, Samsung, OCT, Datasets
    grado: String que representa el grado con el que fue etiquetada la imagen (G1, G2, G3, G4, G5)
    imagen: String del nombre de la imagen sobre la que se desea aplicar el procesamiento, con extensión incluida.
    
    Return
    -----------------------------------
    No devuelve ningún elemento.
    '''
    #primero comprobamos si el dispositivo es iPhone o Samsung, ya que la estructura de directorios variará
    if dispositivo == 'Samsung' or dispositivo == 'iPhone':
        #cargamos la imagen (inpaintada)
        img = cv2.imread('Classified Data/Images/' + dispositivo + '/Inpaint/' + grado + '/' + imagen)
        #cambiamos su formato de BGR a RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #la transformamos usando la función circle_crop y sigmaX = 30
        img_t = circle_crop(img, 30)
        #convertimos de numpy array a Image (de la librería PIL)
        image = Image.fromarray(img_t, 'RGB')
        #por último guardamos la imagen
        image.save('Classified Data/Images_Proc/' + dispositivo + '/Inpaint/' + grado + '/' + imagen)
        #repetimos los mismos pasos pero en esta ocasión con la imagen correspondiente no inpaintada
        img = cv2.imread('Classified Data/Images/' + dispositivo + '/No_inpaint/' + grado + '/' + imagen)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_t = circle_crop(img, 30)
        image = Image.fromarray(img_t, 'RGB')
        image.save('Classified Data/Images_Proc/' + dispositivo + '/No_inpaint/' + grado + '/' + imagen)
    #en caso de que la imagen no sea correspondiente a Samsung o iPhone    
    else:
        #realizamos los mismos pasos que los descritos anteriormente
        img = cv2.imread('Classified Data/Images/' + dispositivo + '/' + grado + '/' + imagen)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_t = circle_crop(img, 30)
        image = Image.fromarray(img_t, 'RGB')
        image.save('Classified Data/Images_Proc/' + dispositivo + '/' + grado + '/' + imagen)

#recorremos los posibles valores de los dispositivos
for dispositivo in ['Datasets','OCT','iPhone','Samsung']:
    #recorremos los posibles valores de los grados
    for grado in ['G1','G2','G3','G4','G5']:
        #si el dispositivo es Samsung o iPhone
        if dispositivo == 'Samsung' or dispositivo == 'iPhone':
            #recorreremos la lista de imágenes de cada grado, incluyendo en la estructura de directorios la subraíz Inpaint (ya que los archivos inpaintados y no inpaintados son los mismos daría igual una subraíz que otra)
            for imagen in os.listdir('Classified Data/Images/' + dispositivo + '/Inpaint/' + grado + '/'):
                #finalmente invocamos la función procesa para que realice el procesamiento de las imágenes
                procesa(dispositivo,grado,imagen)
        #en el caso de que el dispositivo sea otro distinto a iPhone o Samsung
        else:
            #recorremos la lista de imágenes de la estructura de directorios correspondiente
            for imagen in os.listdir('Classified Data/Images/' + dispositivo + '/' + grado + '/'):
                #y para cada imagen aplicamos el procesamiento
                procesa(dispositivo,grado,imagen)
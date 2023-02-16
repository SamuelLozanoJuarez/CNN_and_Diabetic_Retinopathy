'''
#########################################################################################################################
INFORMACIÓN DEL FICHERO
#########################################################################################################################
Autor: Samuel Lozano Juárez
Fecha: 16/02/2023
Institución: UBU | Grado en Ingeniería de la Salud

Este archivo forma parte del Trabajo de Fin de Grado "Detección del grado de retinopatía mediante redes convolucionales".
El alumno a cargo de este proyecto es el declarado como autor en las líneas anteriores.
Los tutores del proyecto fueron el Dr. Darío Fernández Zoppino y el Dr. Daniel Urda Muñoz.

A continuación se incluye el código necesario para la organización de las imágenes obtenidas de Zenodo.Las imágenes crudas, sin clasificar,
pueden descargarse en el siguiente enlace: https://zenodo.org/record/4891308#.Y-35bXbMK3D .
Las imágenes ya se encuentran clasificadas en 7 grupos, y divididas en directorios acorde a esa clasificación. Sin embargo, esta clasificación
consta de 7 niveles, mientras que la empleada por nosotros en el proyecto cuenta únicamente con 5. Es por ello que se acudió al
oftalmólogo de referencia en el HUBU y nos indicó que las imágenes de grado 4 y 5 Zenodo pueden unirse como grado nuestro 4,
y las imágenes grado 6 y 7 Zenodo se pueden unificar bajo la etiqueta de grado 5 propio. 
'''

#realizamos las importaciones necesarias
import os
import shutil

#sabemos, por nuestro retinólogo contacto en el HUBU, que el grado 4 y 5 de Zenodo se pueden unificar en grado 4;
#y que el grado 6 y 7 se pueden unir en el grado 5. 
#por ello vamos a mover las imágenes de cada carpeta original a la que corresponda según nuestra clasificación
#iteramos sobre la lista de imágenes de una carpeta y con la función shutil.move vamos moviéndolas a la dirección que corresponda
for i in os.listdir('Raw Data/Zenodo/1. No DR signs'):
    shutil.move('Raw Data/Zenodo/1. No DR signs/' + i,'Classified Data/Images/Zenodo/G1')
    
for i in os.listdir('Raw Data/Zenodo/2. Mild (or early) NPDR'):
    shutil.move('Raw Data/Zenodo/2. Mild (or early) NPDR/' + i,'Classified Data/Images/Zenodo/G2')
    
for i in os.listdir('Raw Data/Zenodo/3. Moderate NPDR'):
    shutil.move('Raw Data/Zenodo/3. Moderate NPDR/' + i,'Classified Data/Images/Zenodo/G3')
    
for i in os.listdir('Raw Data/Zenodo/4. Severe NPDR'):
    shutil.move('Raw Data/Zenodo/4. Severe NPDR/' + i,'Classified Data/Images/Zenodo/G4')
    
for i in os.listdir('Raw Data/Zenodo/5. Very Severe NPDR'):
    shutil.move('Raw Data/Zenodo/5. Very Severe NPDR/' + i,'Classified Data/Images/Zenodo/G4')
    
for i in os.listdir('Raw Data/Zenodo/6. PDR'):
    shutil.move('Raw Data/Zenodo/6. PDR/' + i,'Classified Data/Images/Zenodo/G5')
    
for i in os.listdir('Raw Data/Zenodo/7. Advanced PDR'):
    shutil.move('Raw Data/Zenodo/7. Advanced PDR/' + i,'Classified Data/Images/Zenodo/G5')


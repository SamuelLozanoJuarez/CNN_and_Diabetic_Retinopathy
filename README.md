# Detección del grado de retinopatía diabética mediante redes convolucionales
---
## Contexto
El proyecto que se va a desarrollar tiene como objetivo diseñar un modelo de red neuronal convolucional (CNN por sus siglas en inglés) que sea capaz de reconocer el grado de retinopatía diabética de un ojo a partir de una imagen de su retina obtenida con un dispositivo móvil (Android o iOS). 
#### ¿Qué es la retinopatía diabética?
La retinopatía diabética es una afección del ojo, producto de una complicación de la diabetes, y constituye la principal causa de pérdida visual no recuperable en individuos de entre 20 y 64 años de edad en los países industrializados [1]. 
Esta patología aparece debido a unos niveles excesivamente altos de azúcar en sangre, lo que provoca la obstrucción de los vasos sanguíneos que irrigan la retina. Como consecuencia de este deterioro, el organismo trata de generar nuevos vasos sanguíneos, pero estos suelen ser más débiles y romperse con frecuencia. Esto produce una pérdida progresiva de la visión en el paciente y puede conllevar otras complicaciones (hemorragia vítrea, desprendimiento de retina, glaucoma...) [2].

#### Diagnóstico de la retinopatía diabética
El método general de diagnóstico consiste en la obtención de una imagen del fondo de ojo empleando un retinógrafo tras una dilatación previa de la pupila [3]. Posteriormente el oftalmólogo realiza un análisis de la imagen obtenida, buscando en ella signos representativos de la patología (neovascularización, microaneurismas o exudados entre otros). La imagen mostrada a continuación, obtenida de [iStock/Anna Koroleva](https://www.istockphoto.com/es/vector/retinopat%C3%ADa-diab%C3%A9tica-anatom%C3%ADa-detallada-informaci%C3%B3n-educativa-m%C3%A9dica-diagrama-de-gm1251233509-365089595), ilustra estos signos:

![image](https://user-images.githubusercontent.com/80346399/203657066-f6caab9f-b031-4af6-952e-ae7e2003c3d7.png)

Sin embargo, los equipos necesarios son costosos y no siempre están disponibles, lo que provoca una ralentización del proceso de diagnóstico en un problema donde la detección temprana es fundamental [4]. Es por ello que surge la necesidad de desarrollar este modelo que permita realizar un primer diagnóstico, que sirva como criba y que pueda ser realizado sin más material que un dispositivo móvil.
#### Redes Neuronales Convolucionales (CNN) 
Las redes neuronales convolucionales (Convolutional Neural Networks) son una tipología de redes neuronales artificiales que toman su nombre de la operación matemática lineal entre matrices denominada convolución, y que son muy empleadas en problemas de Deep Learning aplicado a imágenes [5]. 
Se trata pues de modelos computacionales formados por un conjunto de elementos, denominados neuronas, que optimizan su rendimiento de manera automática mediante aprendizaje. A partir del conjunto de píxeles que conforman una imagen _M x M_, las CNNs generan una matriz de pesos de dimensiones _N x N_ que se va aplicando sobre las regiones de la imagen y permite ir extrayendo los elementos más característicos de esta. El valor de los pesos de esa matriz va modificándose y actualizándose según el grado de acierto de las predicciones de la red, permitiendo así ese aprendizaje mencionado anteriormente. Estos modelos neuronales están compuestos por múltiples capas, siguiendo generalmente la siguiente estructura: capa convolucional, capa de pooling y capa totalmente conectada (_fully-conected_) [6]. En la imagen a continuación ([O’Shea and Nash. 2020](https://arxiv.org/pdf/1511.08458.pdf)) se representa este esquema:

![image](https://user-images.githubusercontent.com/80346399/203657332-6cec8cf2-ffce-4b3c-b712-7695731b5d18.png)

## Metodología de trabajo
Para el desarrollo del proyecto se ha optado por la metodología CRISP-DM (Cross-Industry Standard Process for Data Mining), que consiste en un conjunto de pasos y procedimientos diseñados para guiar el trabajo en minería de datos [7]. 
Esta metodología consta de 6 fases sobre las que se trabaja de manera contínua, permitiéndonos ejercer tareas de varias de ellas en una misma iteración e incluso 'retroceder' en las fases. Estas son las siguientes [8]:
1. Comprensión del negocio (_Business understanding_)
2. Comprensión de los datos (_Data understanding_)
3. Preparación de los datos (_Data preparation_)
4. Modelado (_Modeling_)
5. Evaluación (_Evaluation_)
6. Despliegue (_Deployment_)

Relacionándose entre sí como se puede observar en la siguiente imagen ([Niaksu. 2015](https://www.bjmc.lu.lv/fileadmin/user_upload/lu_portal/projekti/bjmc/Contents/3_2_2_Niaksu.pdf)):

![image](https://user-images.githubusercontent.com/80346399/203657472-2bd27e1b-3689-452b-a8a0-46f51fa1beca.png)

## Licencia
El proyecto se encuentra protegido con una licencia Creative Commons Zero v1.0 Universal.

## Referencias
[1] Tenorio G, Ramírez-Sánchez V. Retinopatía diabética; conceptos actuales. _Med J Hosp Gen Mexico_.2010.73(3):193-201.

[2]	Retinopatía diabética. Mayoclinic.org. 2018. Disponible en: https://www.mayoclinic.org/es-es/diseases-conditions/diabetic-retinopathy/symptoms-causes/syc-20371611

[3] Diabetic retinopathy. Nih.gov. Available from: https://www.nei.nih.gov/learn-about-eye-health/eye-conditions-and-diseases/diabetic-retinopathy

[4] Safi H, Safi S, Hafezi-Moghadam A, Ahmadieh H. Early detection of diabetic retinopathy. _Surv Ophthalmol_. 2018;63(5):601–8. Available from: http://dx.doi.org/10.1016/j.survophthal.2018.04.003

[5] S. Albawi, T. A. Mohammed. S. Al-Zawi. Understanding of a convolutional neural network. 2017 _International Conference on Engineering and Technology (ICET)_, 2017, p. 1-6.

[6] Ghosh A, Sufian A, Sultana F, Chakrabarti A, De D. Fundamental concepts of convolutional neural network. _Intelligent Systems Reference Library_. Cham: Springer International Publishing; 2020. p. 519–67.

[7] IBM documentation. Ibm.com. 2021. Available from: https://www.ibm.com/docs/en/spss-modeler/saas?topic=dm-crisp-help-overview

[8] Niaksu O. CRISP Data Mining Methodology Extension for
Medical Domain. _Baltic J. Modern Computing_.2015.3(2):92-109

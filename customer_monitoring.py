# PRACTICA 10. CUSTOMER MONITORING

# El siguiente ejercicio consiste en la observación de personas en tiempo real (puedes conectar
# un vídeo en streaming para emularlo temporalmente). La idea es conocer cómo son, y como
# se comportan, las personas que acudan a una tienda cualquiera, almacenando en una BBDD
# lo siguiente:
# 1. Afluencia de personas: contador y timestamp de entrada
# 2. Tiempo de permanencia de cada persona en la tienda (timestamp de salida)
# 3. Principales puntos de afluencia del stand (derecha, izquierda, centro.... nos ayudaría a
# conocer qué productos están curioseando)
# 4. Métricas sociodemográficas de los asistentes: edad, sexo... de forma agregada
# Puedes elegir libremente cualquier SDK o tecnología.

import cv2
import numpy as np
from ultralytics import YOLO
import time

# Cargar el modelo YOLOv8
model = YOLO('yolov8n.pt')

# Especifica la ruta del archivo de video que se va a procesar.
video_file = "C:/PYTHON/PRACTICAS/10 Customer monitoring/PRACTICA 10 Angel Lozano/cctv01.mp4"
# Especifica la ruta del archivo que contiene los nombres de las clases de objetos detectados por el modelo.
classnames_file = "C:/PYTHON/PRACTICAS/10 Customer monitoring/PRACTICA 10 Angel Lozano/classnames.txt"
# Umbral de confianza para filtrar detecciones por su puntaje de confianza. 
# Aquellas detecciones cuyo puntaje sea menor que este umbral serán descartadas.
conf_threshold = 0.5
# Umbral utilizado en el algoritmo de Supresión No Máxima (NMS) para filtrar detecciones superpuestas y 
# conservar solo las más confiables.
nms_threshold = 0.4
# Especifica la clase de objeto que se va a detectar en el video, en este caso, personas.
detect_class = "person"

# Este fragmento de código establece algunas variables relacionadas con las dimensiones 
# de los fotogramas del video y el tamaño de las celdas para la matriz de calor
frame_width = 1270
frame_height = 720
cell_size = 40  # 40x40 píxeles
n_cols = frame_width // cell_size
n_rows = frame_height // cell_size

# Factor de fusión utilizado para combinar la matriz de calor con los fotogramas del video. 
# Un valor más alto da más peso a la matriz de calor.
alpha = 0.5

# Se crea una matriz de ceros utilizando NumPy con dimensiones (n_rows, n_cols), 
# donde n_rows representa el número de filas de la matriz de calor y n_cols el número de columnas. 
# Esta matriz se utilizará para registrar la intensidad del calor en cada celda correspondiente a 
# una región del fotograma del video.
heat_matrix = np.zeros((n_rows, n_cols))

# Este valor representa la escala utilizada para normalizar la matriz de calor. 
# En este caso, el valor 0.00392156862745098 es la aproximación de 1/255. 
# Se utiliza para escalar los valores de la matriz de calor a un rango entre 0 y 1, 
# lo que permite una representación más precisa de los datos de intensidad de calor en la imagen final.
scale = 0.00392156862745098 # 1/255

# En este fragmento de código se lee el archivo de nombres de clases y se carga en la variable classes.
classes = None
with open(classnames_file, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# En esta línea de código, se crea un objeto cv2.VideoCapture que permite abrir y 
# manejar un archivo de video para su reproducción o análisis en OpenCV.
cap = cv2.VideoCapture(video_file)

# Diccionario para mantener el tiempo de detección de cada persona
detection_times = {}

# Diccionario para mantener el tiempo de permanencia de cada persona
person_permanence = {}

# Inicia un bucle while que se ejecutará mientras el objeto cap esté abierto, es decir, 
# mientras el archivo de video esté disponible para lectura.
while cap.isOpened():
    # El método cap.read() sirve para leer un frame del video. 
    # La función devuelve dos valores: success indica si la operación de lectura fue exitosa 
    # y frame contiene el frame leído del video.
    success, frame = cap.read()
    
    # La función model.track() toma el fotograma como entrada y devuelve los resultados del seguimiento, 
    # que pueden incluir las cajas delimitadoras de los objetos detectados, las clases de los objetos 
    # y otros detalles relevantes para el seguimiento. Estos resultados se almacenan en la variable results. 
    # El parámetro persist=True indica que se debe seguir realizando el seguimiento de los objetos en fotogramas sucesivos.
    if success:
        # Ejecutar el seguimiento YOLOv8 en el fotograma
        results = model.track(frame, persist=True)  

        # Contar el número total de personas detectadas
        num_persons = 0

        # En este fragmento de código, se recorren los resultados del seguimiento de objetos obtenidos en el fotograma actual, 
        # que están almacenados en la variable results. Para cada conjunto de resultados (frame_results), se verifica 
        # si hay cajas delimitadoras (boxes) asociadas a las detecciones. 
        # Si hay detecciones presentes, se procede a obtener las identificaciones únicas de las personas detectadas.
        for frame_results in results:
            try:
                if frame_results.boxes is None:
                    continue  # Saltar a la siguiente iteración del bucle si no hay cajas delimitadoras

                # Extraer los identificadores únicos de las personas detectadas.
                track_ids = frame_results.boxes.id.int().cpu().tolist()
                
                # En este fragmento de código, se itera sobre todas las cajas delimitadoras detectadas en el fotograma actual, 
                # que están contenidas en frame_results.boxes.xyxy.
                for i, box in enumerate(frame_results.boxes.xyxy):  
                    # Para cada caja delimitadora, se calculan las coordenadas del centro de la caja x_center e y_center. 
                    # Estas coordenadas se calculan promediando las coordenadas de las esquinas opuestas de la caja delimitadora. 
                    # Luego, se calculan las coordenadas de la celda de la matriz de calor a la que corresponde el centro de la caja delimitadora. 
                    # Estas coordenadas se utilizan para actualizar la matriz de calor y registrar la presencia de una persona en esa posición. 
                    x_center = int((box[0] + box[2]) / 2)
                    y_center = int((box[1] + box[3]) / 2)
                    r, c = y_center // cell_size, x_center // cell_size

                    # En esta línea de código, heat_matrix[r, c] += 1, se está actualizando la matriz de calor para registrar 
                    # la presencia de una persona en una celda específica.                 
                    # Al sumar 1 a este valor, se está incrementando el recuento de la presencia de una persona 
                    # en esa celda específica de la matriz de calor. 
                    # Este proceso se repite para cada persona detectada en el fotograma actual.
                    heat_matrix[r, c] += 1

                    # Obtener el ID único de la persona
                    person_id = track_ids[i]

                    # Calcular la duración de la detección
                    detection_duration = time.time() - detection_times.get(person_id, time.time())

                    # Actualizar el tiempo de detección
                    detection_times[person_id] = time.time()

                    # Actualizar el tiempo de permanencia de la persona
                    person_permanence[person_id] = person_permanence.get(person_id, 0) + detection_duration

                    # Mostrar el tiempo como un cronómetro encima de la caja delimitadora
                    cv2.putText(frame, f'{int(person_permanence[person_id]) // 60:02d}:{int(person_permanence[person_id]) % 60:02d}',
                                (int(box[0]) + 50, int(box[1]) + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)                   

                    # Mostrar la ID de la persona sobre la caja delimitadora (centrada en la parte superior)                    
                    text_size = cv2.getTextSize(f'ID: {person_id}', cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    text_x = x_center - text_size[0] // 2
                    text_y = int(box[1]) - 5  # Justo arriba de la caja delimitadora
                    cv2.putText(frame, f'ID: {person_id}', (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                    # Dibujar la caja delimitadora
                    cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 1)

                    # Incrementar el número de personas
                    num_persons += 1

            except Exception as e:
                print(f"Error al procesar las detecciones: {e}")

        # Mostrar el número de personas detectadas en la esquina superior izquierda
        cv2.rectangle(frame, (10, 10), (150, 50), (255, 255, 255), -1)  # Dibuja un rectángulo blanco
        cv2.putText(frame, f'PERSONAS: {num_persons}', (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 2)  # Escribe el texto sobre el rectángulo blanco


        # Normalizar la matriz de calor
        temp_heat_matrix = heat_matrix.copy()
        temp_heat_matrix = cv2.resize(temp_heat_matrix, (frame_width, frame_height))
        temp_heat_matrix = temp_heat_matrix / np.max(temp_heat_matrix)
        temp_heat_matrix = np.uint8(temp_heat_matrix * 325)

        # Aplicar mapa de calor
        image_heat = cv2.applyColorMap(temp_heat_matrix, cv2.COLORMAP_JET) # cv2.COLORMAP_JET

        # Redimensionar el mapa de calor para que coincida con las dimensiones del fotograma
        image_heat_resized = cv2.resize(image_heat, (frame.shape[1], frame.shape[0]))

        # Aplicar la fusión de imágenes
        result = cv2.addWeighted(image_heat_resized, alpha, frame, 1 - alpha, 0)

        cv2.imshow("Result", result)
        if cv2.waitKey(1) == ord('q'):
            break

    else:
        break

# Guardar los tiempos de permanencia en un archivo de texto
with open("permanencia.txt", "w") as file:
    for person_id, duration in person_permanence.items():
        file.write(f"id {person_id}: {int(duration) // 60:02d}:{int(duration) % 60:02d}\n")

cap.release()
cv2.destroyAllWindows()
# Práctica de Imágenes Hiperespectrales: Detección de Grasa en Carne

Este proyecto es una práctica académica orientada al procesamiento y análisis de datos e imágenes hiperespectrales. El objetivo principal es calibrar las capturas espectrales y analizar distintas muestras de carne (novilla y cerdo) para evaluar la cantidad y presencia de tocino (grasa) utilizando diferentes métricas de similitud y contraste.

## Estructura del Proyecto

- `main.py`: Es el script principal que dirige todo el flujo de la práctica. Carga los datos, guía al usuario a través de la selección interactiva de regiones de interés (ROIs), calibra los ejes y calcula las métricas.
- `functions.py`: Contiene funciones de utilidad separadas para mantener el código principal limpio, como la selección interactiva de ROIs (`select_named_rois`), promediado de espectros (`get_mean_spectrum`), redimensionado de los cubos de datos (`resize_spectral_cube`) y el registro de la ejecución (`Logger`).
- `outputs/`: Carpeta generada automáticamente al ejecutar el código. Aquí se guardan todas las gráficas resultantes del análisis (en formato `.png`), junto con un archivo `logfile.txt` que almacena el registro de toda la salida mostrada por consola durante la ejecución.

## Requisitos y Dependencias

El proyecto utiliza Python y depende de las siguientes librerías fundamentales:
- `numpy`: Para el cálculo matricial numérico y manejo de los cubos hiperespectrales.
- `scipy`: Para la carga de los archivos de datos en formato `.mat`.
- `matplotlib`: Para la visualización de las imágenes 2D y la graficación de los espectros.
- `opencv-python` (`cv2`): Utilizado para la interfaz gráfica interactiva que permite al usuario seleccionar las ROIs.

## Uso y Funcionamiento

Para iniciar el análisis, simplemente ejecuta el script principal desde tu terminal:

```bash
python main.py
```

El script es **interactivo** y requiere intervención del usuario para seleccionar diferentes zonas en las imágenes emergentes. El flujo de ejecución es el siguiente:

1. **Carga de Datos:** Se leen los archivos `.mat` de la cubeta de carne y los materiales de calibración.
2. **Calibración Espectral y de Reflectancia:** El script mostrará una imagen 2D con los materiales de calibración. Deberás seleccionar la región para la calibración de longitud de onda (WCS-MC-020) y la de reflectancia al 99% (SRS-99-020). Con esto, el programa corrige los espectros en crudo a valores de reflectancia.
3. **Análisis de Muestras de Carne:** Se mostrará una imagen de la cubeta con muestras de carne.
   - Tendrás que seleccionar áreas con diferentes niveles de tocino para las muestras de **Novilla**.
   - A continuación, realizarás lo mismo para las muestras de **Cerdo**.
   - El programa generará gráficas comparativas de cómo evoluciona el espectro según la cantidad de tocino.
4. **Métricas de Contraste (Detección de Grasa):** Por último, se te pedirá seleccionar una región de referencia que contenga grasa pura (Cubeta A1). Utilizando este espectro como firma de referencia (Ground Truth), el script calculará y mostrará tres métricas de similitud en toda la imagen para mapear la presencia de grasa:
   - **ED** (Distancia Euclídea)
   - **SAM** (Spectral Angle Mapper)
   - **BCS** (Bray-Curtis Similarity)

> **Nota sobre la selección de ROIs:** Durante las pausas interactivas, haz clic y arrastra el ratón sobre la imagen para dibujar el rectángulo de interés, y pulsa la tecla `ENTER` (o `ESPACIO`) para confirmar la selección.

## Resultados

Al finalizar la ejecución (y tras cerrar las ventanas de las gráficas), el script concluirá de manera exitosa. Podrás revisar la carpeta `outputs/` para consultar todas las imágenes guardadas paso a paso y el log con las acciones registradas.

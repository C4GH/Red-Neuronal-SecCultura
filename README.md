Este repositorio contiene los archivos necesarios para cargar, entrenar y evaluar el modelo rede neuronal desarrollado para la Secretaría de Cultura.

#### 1. **Dataloader**
   - **Archivo**: `dataloader.py`
   - **Descripción**: Este archivo define la clase `CustomDataset`, que se utiliza para cargar y preprocesar los datos desde archivos JSON y NumPy, adecuándolos para ser utilizados en un modelo de redes neuronales. Además, incluye la función `collate_fn` que se encarga de agrupar y preparar los datos en batches para el entrenamiento y la evaluación.

#### 2. **Modelo**
   - **Archivo**: `modelo.py`
   - **Descripción**: En este archivo se define la clase `Modelo`, que implementa un modelo de redes neuronales utilizando capas de Transformer para procesamiento de lenguaje natural. Este modelo es configurable en términos de tamaño de vocabulario, dimensiones de los embeddings, entre otros parámetros.

#### 3. **Train**
   - **Archivo**: `train.py`
   - **Descripción**: Este archivo contiene la función `train`, que se utiliza para entrenar el modelo definido en `modelo.py`. Gestiona el proceso de entrenamiento, incluyendo la optimización de parámetros, el ajuste de la tasa de aprendizaje y la evaluación del rendimiento del modelo en cada época. También se encarga de registrar en un archivo de log todos los eventos y métricas importantes durante el entrenamiento.

#### 4. **Train Utils**
   - **Archivo**: `train_utils.py`
   - **Descripción**: Aquí se encuentran definidas las funciones `log_message`, `test`, y otras utilidades que asisten en el registro de mensajes durante el entrenamiento y la evaluación del modelo, así como la función `test` que evalúa el rendimiento del modelo en un conjunto de datos de prueba.

#### 4. **Salida de JSON**
   - **Archivo**: `salida_min4.json`
   - **Descripción**: Este es un archivo de ejemplo para mostrar como estaban organizados los datos de entrenamiento y prueba de la red. Además, este archivo contiene las preguntas con sus respectivos contextos y respuestas que se utlizaron en la comparación de la red contra el personal de la CNDI.


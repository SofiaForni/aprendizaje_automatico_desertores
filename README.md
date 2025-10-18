# Predicción de Deserción Estudiantil con Aprendizaje Automático

Trabajo práctico grupal para la materia "Aprendizaje Automático" de la Maestría en Data Mining (UBA, 2025). El objetivo del proyecto es desarrollar y evaluar modelos de machine learning para predecir la deserción estudiantil.

Alesandro, Guadalupe

Firpo, María Florencia

Forni, Sofía


## Estructura del Repositorio

- **data/**: Contiene los datos iniciales y los datasets procesados listos para ser utilizados por los modelos.
- **notebooks/**: Contiene los Jupyter Notebooks con el desarrollo del proyecto.

    -   **eda.ipynb**: Realiza el Análisis Exploratorio de Datos (EDA) y el preprocesamiento. Al final, genera y guarda en la carpeta - data/ los dos datasets que se usarán para el modelado (uno para clasificación binaria y otro para multiclase).
        
    - **modelos_target_binaria.ipynb**: Entrena y evalúa modelos para predecir el objetivo binario (desertor / no desertor).
  
            - Desertor: 1
            - No desertor (En Curso y Graduado): 0
      
    - **modelos_target_multiclase.ipynb**: Entrena y evalúa modelos para predecir el objetivo multiclase.
  
            - Desertor: 0
            - En Curso: 1
            - Graduado: 2

- requirements.txt: Lista de las librerías de Python necesarias para ejecutar el proyecto.

## Instalación
Sigue estos pasos para configurar el entorno de desarrollo.


1. Clona el repositorio:

        git clone <URL_DEL_REPOSITORIO>

        cd aprendizaje_automatico_desertores


2. Crea y activa un entorno virtual:
   
        python3 -m venv venv
        source venv/bin/activate  # En Windows usa: venv\Scripts\activate


4. Instala las dependencias:

       pip install -r requirements.txt


## Uso

El orden de ejecución recomendado para los notebooks es el siguiente:

1. **notebooks/eda.ipynb**: Ejecutar primero esta notebook. Realizará el análisis exploratorio y generará los archivos de datos procesados necesarios para los siguientes pasos.
   
2. **notebooks/modelos_target_binaria.ipynb**: Una vez generados los datos, ejecutar esta notebook para entrenar y evaluar los modelos de clasificación binaria.
   
3. **notebooks/modelos_target_multiclase.ipynb**: De forma análoga, ejecuta esta notebook para los modelos de clasificación multiclase.

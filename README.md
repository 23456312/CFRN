# Clasificación de la calidad de frutas mediante redes neuronales convolucionales
 
Modelo de *deep learning* que, a partir de una fotografía, identifica el tipo de fruta (manzana, plátano, guayaba, lima, naranja o granada) y si su calidad es **buena o mala**.
El modelo final usa **transfer learning y fine-tuning de MobileNetV2** y alcanza **83.3 % de exactitud en el conjunto de prueba**.  

## Dataset

**FruitNet: Indian Fruits Dataset with Quality** (Meshram y Patil, 2022), disponible en Kaggle:
https://www.kaggle.com/datasets/ryandpark/fruit-quality-classification
 
El dataset no se incluye en este repositorio por su tamaño (~3.3 GB). El script de limpieza elimina 4,752 copias aumentadas y divide las 13,700 imágenes restantes en entrenamiento, validación y prueba (70/15/15) **por sesión de captura**, para evitar que fotografías casi idénticas queden en conjuntos distintos.
 
## Contenido del repositorio
 
| Archivo | Descripción |
|---|---|
| `01_ETL_CFRN.ipynb` | Extracción, limpieza y división de los datos, paso a paso |
| `01_limpieza_y_division.py` | El mismo proceso como script; lo usan los notebooks 02–05 |
| `02_EDA.ipynb` | Análisis exploratorio: distribución, tamaños, muestras, sesiones y sesgos |
| `03_Modelo_base.ipynb` | CNN base entrenada desde cero y su diagnóstico |
| `04_mejoras.ipynb` | Experimentos E1–E5 y comparación |
| `05_Evaluacion_Final.ipynb` | Evaluación en prueba, Grad-CAM, imágenes mixtas y predicciones (consola y Gradio) |
| `splits.csv` | Las 13,700 imágenes con su clase, fecha de captura, sesión y conjunto asignado (train, val o test) | Verificar que ninguna sesión aparece en más de un conjunto, es decir, que no hay fuga de información |
| `resumen_limpieza.csv` | Imágenes por clase antes y después de la limpieza, número de sesiones y reparto en cada conjunto | Revisar el efecto de la limpieza y el balance de clases |
| `comparacion_experimentos.csv` | Métricas de los experimentos E0–E5 en validación (exactitud, pérdida, F1, brecha, exactitud en el subconjunto de 2020) | Comparar los modelos sin volver a entrenarlos |
| `class_names.json` | El orden de las 12 clases que usa el modelo | Interpretar sus salidas: la posición *i* de la predicción corresponde a la clase *i* de esta lista |
| Paper | Reporte del proyecto |
 
Los notebooks se suben con sus salidas, por lo que los resultados y gráficas pueden revisarse sin ejecutarlos.
 
## Cómo reproducirlo (Google Colab)
 
1. Descarga el dataset de Kaggle, comprime las carpetas en un archivo llamado `fruitnet.zip` y súbelo a la raíz de tu Google Drive. Copia también `01_limpieza_y_division.py` dentro de la carpeta del dataset antes de comprimirla (los notebooks lo buscan en `archive (3)/01_limpieza_y_division.py`; ajusta la ruta si tu carpeta tiene otro nombre).
2. En Colab, activa la GPU: *Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU (T4)*.
3. Ejecuta los notebooks en orden (02 → 05) con *Entorno de ejecución → Ejecutar todo*. La primera celda de cada uno monta Drive, descomprime el dataset y genera las carpetas `train/val/test`.
4. Los modelos y resultados se guardan en `MyDrive/fruit_resultados/`. El notebook 05 carga el modelo final desde ahí.
Tiempos aproximados con una GPU T4: modelo base ~5 min, experimentos de mejora ~45 min, evaluación final ~5 min.

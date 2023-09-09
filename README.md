# Curso Streamlit


## Introducción

Este repositorio contiene el código de la aplicaciones desarrolladas en el curso de Streamlit. Existen tres aplicaciones, que se pueden encontrar en las siguientes rutas:

- [Análisis de MNIST](mnist_analisis.py): Aplicación que permite analizar el dataset [MNIST](https://en.wikipedia.org/wiki/MNIST_database). Se carga el conjunto de test del dataset MNIST, y se permite explorar las imágenes y las etiquetas de las mismas. Además, se permite filtrar las imágenes por etiqueta, y se muestra la distribución valores de pixeles de los datos divididos por las etiquetas.
- [Predicción de MNIST](mnist_prediccion.py): Aplicación que permite dibujar al usuario en un canvas, y predecir el número que ha dibujado. Para ello, se utiliza un modelo de clasificación de imágenes de MNIST entrenado previamente.
- [Q&A documentos PDF](qa.py): Aplicación que permite subir un documento PDF al usuario y responder preguntas sobre el mismo. Para ello, se utiliza la API de OpenAI, que permite realizar preguntas sobre un texto. En este caso, el texto es el contenido del documento PDF. Para extraer el texto del documento PDF, se utiliza la librería [PyPDF](https://pypi.org/project/pypdf/). Para la generación de embeddings del PDF y las preguntas se utiliza la librería [Sentence Transformers](https://www.sbert.net/). Esta librería permite generar embeddings de frases, que se utilizan para calcular la similitud entre las preguntas y las respuestas. Finalmente, para almacenar los embeddings y buscar las respuestas se utiliza la librería [ChromaDB](https://www.trychroma.com/).

## Ejecución

Para ejecutar las diferentes aplicaciones, se debe ejecutar el siguiente comando:

```bash
streamlit run <ruta_aplicacion>
```

### Requisitos

Para ejecutar las aplicaciones, se deben instalar las dependencias especificadas en el fichero [requirements.txt](requirements.txt). La versión de Python utilizada es la `3.9`. Para instalar los paquetes necesarios, se recomienda utilizar un entorno virtual. Para ello, existen varias opciones, mostrando aquí la instalación con [conda](https://docs.conda.io/en/latest/).

```bash
conda create --name <name> python==3.9
conda activate <name>
pip install -r requirements.txt
```

#### Demo Q&A

Para ejecutar la aplicación de [Q&A](qa.py) es necesario especificar una variable de entorno con la clave de la API de OpenAI. Esta clave se puede obtener en la [página de OpenAI](https://openai.com/). Para ello, se debe ejecutar el siguiente comando:

```bash
export OPENAI_API_KEY=<api_key>
```

## Despliegue

Para el despliegue proponemos el uso de Docker conjuntamente con el proveedor que mas se ajuste a las necesidades del usuario. Docker es una herramienta que permite la creación de contenedores de aplicaciones, que son entornos de ejecución aislados con todo lo necesario para ejecutar una aplicación. Esto permite que la aplicación se ejecute en cualquier entorno, independientemente de las diferencias entre el entorno en el que se desarrolló y el entorno en el que se ejecuta.

La utilización de [Docker con Streamlit](https://docs.streamlit.io/knowledge-base/tutorials/deploy/docker) es muy sencilla. Para ello, se debe crear un archivo Dockerfile, que podeis encontrar en este repositorio ([aquí](Dockerfile)). En este archivo se especifica la imagen base, que en este caso es la imagen oficial de Streamlit, y se copia el archivo de la aplicación en el directorio de trabajo del contenedor. Para construir la imagen y ejecutar el contenedor, se deben ejecutar los siguientes comandos:

```bash
docker build -t st-qa .
docker run -p 8501:8501 -e OPENAI_API_KEY=$OPENAI_API_KEY st-qa
```

Notese que se debe especificar la variable de entorno OPENAI_API_KEY, que es la clave de la API de OpenAI. Esta clave se puede obtener en la [página de OpenAI](https://openai.com/).
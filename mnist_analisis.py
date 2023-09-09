"""
Descripción: Este programa muestra los datos del dataset mnist_test.csv
Ejecución: streamlit run mnist.py

Creado por: Mario Parreño
Fecha: 05-09-2023
"""
import streamlit as st
import plotly.express as px
import pandas as pd

df = pd.read_csv("data/mnist_test.csv")


st.markdown("""
# Aplicación de Análisis de MNIST con Streamlit

¡Bienvenido a nuestra aplicación de análisis de datos MNIST!
En esta aplicación, exploraremos el famoso conjunto de datos MNIST,
que contiene imágenes de dígitos escritos a mano.
Utilizaremos Streamlit para cargar y analizar
este conjunto de datos de manera interactiva.

## Características de la Aplicación

En esta aplicación, podrás realizar las siguientes acciones:

- Cargar y explorar el conjunto de datos MNIST.
- Visualizar y filtrar muestras de imágenes MNIST.
- Realizar análisis estadísticos básicos sobre los dígitos.

""")

st.subheader("Cargar el conjunto de datos MNIST")
st.markdown(f"""
Hemos cargado el conjunto de test de MNIST,
que contiene {len(df)} imágenes de dígitos escritos a mano.
""")
st.dataframe(df)

# Las columnas son 'label', '1x1', '1x2', '1x3', ..., '28x28'
# Vamos a permitir filtrar por el valor de la columna 'label'
st.sidebar.subheader("Filtrar por etiqueta")
labels = st.sidebar.multiselect(
    "Selecciona el valor de la columna 'label'",
    sorted(df['label'].unique()),  # Valores a elegir
    sorted(df['label'].unique()),   # Valores por defecto
    placeholder="Selecciona una etiqueta"
)

if len(labels) == 0:
    st.error("Por favor, selecciona al menos una etiqueta.")
    st.stop()

# Filtramos el dataframe
df = df[df['label'].isin(labels)]

# Vamos a mostrar un slider para seleccionar el número de fila
st.sidebar.subheader("Selecciona el número de fila")
case_index = st.sidebar.slider("Selecciona el número de fila", 0, len(df)-1, 0)

# Mostramos la imagen
st.subheader("Imagen")
case_image = df.iloc[case_index, 1:].values.reshape(28, 28)
st.image(case_image, caption=f"Imagen clase {df.iloc[case_index, 0]}", width=200)

# Vamos a mostrar un barplot con los valores de elementos diferentes de 0
# que contiene cada clase
st.subheader("Píxeles diferentes de 0 por clase")
st.markdown("""
En este gráfico, mostramos el número de píxeles diferentes de 0 promedio
por cada clase.
""")

# Obtenemos por fila el número de píxeles diferentes de 0
df["non_zero_pixels"] = df.iloc[:, 1:].apply(lambda x: sum(x != 0), axis=1)

# Generamos un dict con los valores de la columna 'label' como keys
# y los valores de la columna 'non_zero_pixels' como values
hist_data = {
    str(label): (
        df[df['label'] == label]['non_zero_pixels'].sum() 
        / # Normalizamos dividiendo por el número de muestras
        len(df[df['label'] == label])
    )
    for label in df['label'].unique()
}

# Crea un countplot a partir de hist_data
fig = px.bar(
    x=list(hist_data.keys()),
    y=list(hist_data.values()),
    title="Número de píxeles diferentes diferentes a 0 medio por clase"
)
fig.update_xaxes(title_text='Etiqueta', tickvals=list(hist_data.keys()))
fig.update_yaxes(title_text='Cantidad de píxeles')
st.plotly_chart(fig)

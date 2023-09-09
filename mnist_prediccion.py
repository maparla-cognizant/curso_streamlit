"""
Descripción: Este programa muestra los datos del dataset mnist_test.csv
Ejecución: streamlit run mnist_v2.py

Creado por: Mario Parreño
Fecha: 05-09-2023

Referencias:
    - https://discuss.streamlit.io/t/drawable-canvas/3671
    - https://drawable-canvas.streamlit.app/
"""
import streamlit as st
from streamlit_drawable_canvas import st_canvas

import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


## network
class MLPNet(nn.Module):
    def __init__(self):
        super(MLPNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 500)
        self.fc2 = nn.Linear(500, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Cargamos el modelo
transform = transforms.ToTensor()
device = torch.device("cpu")
with st.spinner("Cargando modelo..."):
    model = MLPNet().to(device)
    model.load_state_dict(torch.load("data/mlp_mnist.pt", map_location=device))
    model.eval()
    st.success("Modelo cargado")

# Predicción de dígitos
st.subheader("Predicción de dígitos")
st.markdown("""
En esta sección, podrás dibujar un dígito en el canvas y predecir
qué dígito es.
""")
st.markdown("""
Primero, dibuja un dígito en el canvas.
""")
canvas_result = st_canvas(
    stroke_width=24,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=224,
    width=224,
)
st.markdown("""
Luego, pulsa el botón de abajo para predecir el dígito.
""")
predict_button = st.button("Predecir")
if predict_button:

    # shape (224, 224, 4) - 4 canales RGBA, cogemos solo el primer canal
    drawed_image = canvas_result.image_data[:, :, 0]

    if drawed_image.sum() == 0:
        st.error("Dibuja un dígito en el canvas")
        st.stop()
    
    # redimensionamos la imagen para que tenga el tamaño de entrenamiento
    drawed_image = cv2.resize(
        drawed_image,
        (28, 28),
        interpolation=cv2.INTER_NEAREST
    )
    # binarizamos la imagen
    drawed_image[drawed_image != 0] = 255

   
    # Predecimos el dígito
    ## Transformamos la imagen a PIL
    drawed_image = Image.fromarray(drawed_image)
    ## Transformamos la imagen a tensor
    drawed_image = transform(drawed_image).unsqueeze(0)
    ## Realizar la predicción
    with torch.no_grad():
        prediction = model(drawed_image)
    
    # Mostramos la predicción
    prediction = F.softmax(prediction, dim=1)
    predicted_class = prediction.argmax(dim=1).item()
    st.markdown(f"Dígito predicho: **{predicted_class}**")

    st.write("Probabilidades:")
    # convertimos a dict (json) para mejor visualizacion
    prediction = {i: prediction[0][i].item() for i in range(10)}
    st.write(prediction)

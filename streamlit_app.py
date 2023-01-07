import streamlit as st
import numpy as np
import math



# Estilos ###########################################################

# Hace más ancho el contenido de la página para aprovechar mejor la pantalla
style = f"""
<style>
    .appview-container .main .block-container{{
        max-width: 90%;
    }}
</style>
"""
st.markdown(style, unsafe_allow_html=True)



# Cabecera ##########################################################

st.image("neurona.jpg", width=360)
st.header("Simulador de neurona")



# Definición de la clase Neuron #####################################

class Neuron:
  """
  Esta clase define las características y el comportamiento
  de las neuronas artificiales
  """
  
  def __init__(self, weights=[], bias=0, func="sigmoid"):
    self.weights = weights
    self.bias = bias
    self.func = func
  
  def run(self, input_data=[]):
    """Calcula la salida en función de los datos de entrada"""
    sum = np.dot(np.array(input_data), self.weights) # suma ponderada
    sum += self.bias # se aplica el sesgo

    # Aplica la función de activación
    if self.func == "sigmoid":
      return self.__sigmoid(sum)
    elif self.func == "relu":
      return Neuron.__relu(sum)
    elif self.func == "tanh":
      return self.__tanh(sum)
    else:
      print("La función de activación proporcionada no es correcta.")
      print("Las funciones de activación permitidas son: sigmoid, relu, tanh.")
  
  def changeWeights(self, weights):
    self.weights = weights

  def changeBias(self, bias):
    self.bias = bias
  
  @staticmethod
  def __sigmoid(x):
    return 1 / (1 + math.e ** -x)
  
  @staticmethod
  def __relu(x):
    return 0 if x < 0 else x
  
  @staticmethod
  def __tanh(x):
    return math.tanh(x)



# Número de pesos y entradas ########################################

number_of_inputs = st.slider(
    "Elige el número de entradas/pesos que tendrá la neurona", 1, 10)



# Pesos #############################################################

st.subheader("Pesos")

# w_option = st.selectbox(
#     '¿Cómo quieres los pesos iniciales?',
#     ('ceros', 'aleatorios'))

w = []
col_w = st.columns(number_of_inputs)

for i in range(number_of_inputs):
    w.append(i)

    with col_w[i]:
        st.markdown(f"w<sub>{i}</sub>", unsafe_allow_html=True)
        w[i] = st.number_input(
            f"w_input_{i}",
            #value=(0.0 if w_option == 'ceros' else round(random() * 10, 2)),
            label_visibility="collapsed")

st.text(f"w = {w}")



# Entradas ##########################################################

st.subheader("Entradas")

x = []
col_x = st.columns(number_of_inputs)

for i in range(number_of_inputs):
    x.append(i)

    with col_x[i]:
        st.markdown(f"x<sub>{i}</sub>", unsafe_allow_html=True)
        x[i] = st.number_input(
            f"x_input_{i}",
            label_visibility="collapsed")

st.text(f"x = {x}")



# Sesgo #############################################################

col1, col2 = st.columns(2)

with col1:
    st.subheader("Sesgo")
    b = st.number_input("Introduce el valor del sesgo")

with col2:
    st.subheader("Función de activación")
    func_option = st.selectbox(
        'Elige la función de activación',
        ('Sigmoide', 'ReLU', 'Tangente hiperbólica'))



# Salida ############################################################

FUNCTIONS = {'Sigmoide': 'sigmoid', 'ReLU': 'relu', 'Tangente hiperbólica': 'tanh'}

if st.button("Calcular la salida"):
    my_neuron = Neuron(weights=w, bias=b, func=FUNCTIONS[func_option])
    st.text(f"La salida de la neurona es {my_neuron.run(input_data=x)}")



# Aplicación en Streamlit

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_absolute_error
import matplotlib.pyplot as plt
import pickle
import plotly.express as px

st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1663970206579-c157cba7edda?fm=jpg&q=60&w=3000&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8M3x8YWJzdHJhY3QlMjBiYWNrZ3JvdW5kfGVufDB8fDB8fHww");
        background-size: cover;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.set_page_config(layout = 'wide')

# Llamamos al modelo entrenado
with open('random_forest.pkl', 'rb') as f:
    model_pickle = pickle.load(f)

st.markdown("<h1 style = 'text-align: center; '>📊 Puesta en producción - Modelo Machine Learning</h1>", unsafe_allow_html = True)

st.sidebar.header('Características propiedad')

# Definimos una función para leer los datos que el usuario ingresará
def get_user_input():
    
    # El usuario ingresa los siguientes datos
    superficie_util     = st.sidebar.number_input('Superficie', min_value = 10, max_value = 350, step = 1, value = 30)
    superficie_terraza  = st.sidebar.number_input('Superficie terraza', min_value = 0, max_value = 100, step = 1, value = 10)
    ambientes           = st.sidebar.number_input('Ambientes', min_value = 1, max_value = 6, step = 1, value = 3)
    dormitorios         = st.sidebar.number_input('Dormitorios', min_value = 1, max_value = 6, step = 1, value = 2)
    banos               = st.sidebar.number_input('Baños', min_value = 1, max_value = 6, step = 1, value = 2)
    cant_max_habitantes = st.sidebar.number_input('Cantidad máxima de habitantes', min_value = 2, max_value = 10, step = 1, value = 4)
    cantidad_pisos      = st.sidebar.number_input('Cantidad de pisos (edificio)', min_value = 1, max_value = 30, step = 1, value = 10)
    departamentos_piso  = st.sidebar.number_input('Departamentos por piso (edificio)', min_value = 1, max_value = 20, step = 1, value = 5)
    numero_piso_unidad  = st.sidebar.number_input('Piso', min_value = 1, max_value = 20, step = 1, value = 2)
    comuna              = st.sidebar.selectbox('Comuna', ['Antofagasta','Arica','Chiguayante','Chillán','Colina','Concepción','Conchalí',
                                                          'Concón','Estación central','Huechuraba','Independencia','Iquique','La cisterna',
                                                          'La florida','La reina','La serena','Las condes','Lo barnechea','Lo prado',
                                                          'Macul','Maipú','Osorno','Pedro aguirre cerda','Peñalolén','Providencia',
                                                          'Pucón','Pudahuel','Puente alto','Puerto montt','Puerto varas','Quilicura',
                                                          'Quilpué','Quinta normal','Rancagua','Recoleta','Rm (metropolitana)','San bernardo',
                                                          'San joaquín','San miguel','San pedro de la paz','San ramón','Santiago',
                                                          'Temuco','Valdivia','Valparaíso','Villa alemana','Vitacura','Viña del mar','Ñuñoa'])
    
    user_data = {
        'superficie_util'     : superficie_util,
        'superficie_terraza'  : superficie_terraza,
        'ambientes'           : ambientes,
        'dormitorios'         : dormitorios,
        'banos'               : banos,
        'cant_max_habitantes' : cant_max_habitantes,
        'cantidad_pisos'      : cantidad_pisos,
        'departamentos_piso'  : departamentos_piso,
        'numero_piso_unidad'  : numero_piso_unidad,
        f'comuna_{comuna}'    : 1
    }

    return user_data

image_banner = Image.open('Pic_2.png')
st.image(image_banner, use_container_width = True)

left_col, right_col = st.columns(2)

# Cálculo de las métricas de desempeño
testing_data = pd.read_excel('X_test.xlsx', sheet_name = 'Sheet1')
y_test       = pd.read_excel('y_test.xlsx', sheet_name = 'Sheet1')
y_pred       = model_pickle.predict(testing_data)

predicciones = pd.DataFrame({'Predicción': y_pred, 'Precio real': np.array(y_test['precio'])})

r2score = r2_score(y_test, y_pred)
mape    = mean_absolute_percentage_error(y_test, y_pred)
mae     = mean_absolute_error(y_test, y_pred)

data = {
    'Métrica': ['R2 Score', 'MAPE', 'MAE'],
    'Desempeño en testing': [np.round(r2score, 2), np.round(mape, 2), np.round(mae, 2)]
}

df = pd.DataFrame(data)

parametros = {
    'Parámetro': ['Estimadores','Profundidad','Elementos por hoja','Elementos en split'],
    'Valor'    : [100, 'Ninguna', 1, 5]
}

parametros = pd.DataFrame(parametros)

with left_col:
    st.header('Desempeño del modelo en testing')
    st.write('#### Modelo: Random Forest Regressor')

    st.markdown(df.to_html(index=False), unsafe_allow_html=True)

    st.write('#### Características del modelo')
    st.markdown(parametros.to_html(index = False), unsafe_allow_html = True)

    st.write('#### Scatterplot testing vs actual values')
    #st.scatter_chart(predicciones, x='Predicción', y='Precio real')
    fig = px.scatter(predicciones, x="Predicción", y="Precio real", title="")
    fig.update_layout(
        plot_bgcolor="black",     # inside the plot area
        paper_bgcolor="black",    # entire figure background
        font_color="white"        # make text readable
        )
    st.plotly_chart(fig, use_container_width=True)

with right_col:

    st.header('Predicción precio de arriendo')

    user_data = get_user_input()

    features = [
        'superficie_util', 'superficie_terraza','ambientes','dormitorios','banos','cant_max_habitantes','cantidad_pisos','departamentos_piso','numero_piso_unidad',
        'comuna_Antofagasta','comuna_Arica','comuna_Chiguayante','comuna_Chillán','comuna_Colina','comuna_Concepción','comuna_Conchalí',
        'comuna_Concón','comuna_Estación central','comuna_Huechuraba','comuna_Independencia','comuna_Iquique','comuna_La cisterna',
        'comuna_La florida','comuna_La reina','comuna_La serena','comuna_Las condes','comuna_Lo barnechea','comuna_Lo prado',
        'comuna_Macul','comuna_Maipú','comuna_Osorno','comuna_Pedro aguirre cerda','comuna_Peñalolén','comuna_Providencia',
        'comuna_Pucón','comuna_Pudahuel','comuna_Puente alto','comuna_Puerto montt','comuna_Puerto varas','comuna_Quilicura',
        'comuna_Quilpué','comuna_Quinta normal','comuna_Rancagua','comuna_Recoleta','comuna_Rm (metropolitana)','comuna_San bernardo',
        'comuna_San joaquín','comuna_San miguel','comuna_San pedro de la paz','comuna_San ramón','comuna_Santiago','comuna_Temuco',
        'comuna_Valdivia','comuna_Valparaíso','comuna_Villa alemana','comuna_Vitacura','comuna_Viña del mar','comuna_Ñuñoa'
    ]

    def prepare_input(data, feature_list):
        input_data = {feature: data.get(feature, 0) for feature in feature_list}
        return np.array([list(input_data.values())])
    
    input_array = prepare_input(user_data, features)
    
    if st.button('Predecir precio'):
        prediction = model_pickle.predict(input_array)
        st.subheader(f"Precio modelo: ${int(prediction[0]):,}".replace(",", "."))

# python -m streamlit run Aplicacion_ML.py
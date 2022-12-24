import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
import warnings

def app():
    st.title('Model 4 - ARIMA')
    yf.pdr_override()
    #start = '2004-08-18'
    #end = '2022-01-20'
    start = st.date_input('Start' , value=pd.to_datetime('2004-08-18'))
    end = st.date_input('End' , value=pd.to_datetime('today'))

    st.title('Predicción de tendencia de acciones')
    user_input = st.text_input('Introducir cotización bursátil' , 'NTDOY')

    df = pdr.get_data_yahoo([user_input], start,end)
    df["Date"] = df.index
    # Candlestick chart
    st.subheader('Gráfico Financiero de la data') 
    candlestick = go.Candlestick(
                            x=df.index,
                            open=df['Open'],
                            high=df['High'],
                            low=df['Low'],
                            close=df['Close']
                            )

    fig = go.Figure(data=[candlestick])

    fig.update_layout(
        width=800, height=600,
        title=user_input,
        yaxis_title='Close'
    )

    st.plotly_chart(fig)
    
    # Información de la data
    st.subheader('Datos del año 2004 hasta el 2022') 
    st.write(df)
    st.subheader('Información de la Estadística descriptiva de la data') 
    st.write(df.describe())

    #Visualizaciones 
    st.subheader('Gráfica Close vs Date')
    st.write('Visualicemos los precios cercanos de la data')
    fig = plt.figure(figsize = (15,10))
    plt.plot(df.Close)
    st.pyplot(fig)

    st.subheader('Aplicación del método de descomposición estacional')
    st.write('Permite dividir los datos de la serie temporal en tendencia, estacional y residual para una mejor comprensión de los datos')
    result = seasonal_decompose(x=df["Close"], model='additive', extrapolate_trend='freq', period=1)
    fig = plt.figure()  
    fig = result.plot()  
    fig.set_size_inches(15, 10)
    st.pyplot(fig)
    
    st.subheader('Gráfico de predicciones aplicando el modelo ARIMA')
    p, d, q = 1, 1, 2
    model = ARIMA(df["Close"], order=(1,1,2))  
    fitted = model.fit()  
    predictions = fitted.predict()

    model=sm.tsa.statespace.SARIMAX(df['Close'],
                                    order=(1, 1, 2),
                                    seasonal_order=(1, 1, 2, 12))
    model=model.fit()

    predictions = model.predict(len(df), len(df)+10)
    fig, x = plt.subplots(1,1, figsize=(15, 10))

    x.plot(df["Close"], 'y', alpha=0.7,label="Training Data")
    x.plot(predictions, 'b', alpha=0.7, label="Predictions")

    st.pyplot(fig)
    st.write('Por lo tanto, considerando un p,d, q = 1,1,2 observamos que el modelo ARIMA genera adecuadas predicciones en el precio de valores para esta data.')
    
    #st.subheader('Gráfico del diagnóstico general aplicando el modelo ARIMA')
    #figura2 = model.plot_diagnostics(figsize=(15,8))
    #st.pyplot(figura2)

    #st.write('Arriba a la izquierda:Los errores residuales parecen fluctuar alrededor de una media de cero y tienen una varianza uniforme.')
    #st.write('Arriba a la derecha: La gráfica de densidad sugiere una distribución normal con media cero.')
    #st.write('Abajo a la izquierda: Todos los puntos deben estar perfectamente alineados con la línea roja, cualquier desviación significativa implicaría que la distribución está sesgada.')
    #st.write('Abajo a la derecha: El correlograma, tambien conocido como gráfico ACF, muestra que los errores residuales no estan autocorrelacionados. Cualquier autocorrelación implicaría que existe algún patrón en los errores residuales que no se explican en el modelo. Por lo tanto, deberá busca más X (predictores en el modelo).')
    
    st.write('El modelo ARIMA es una clase de modelos que explica una serie de tiempo determinada en función de sus propios valores pasados, es decir, sus propios retasos y los errores de pronóstico retrasados, de manera que se puede utilizar la ecuación para pronosticar valores futuros.Además, se caracteriza por tener tres términos “p” (orden del término AR), “d” (orden del término MA) y “q” (número de diferenciaciones necesarias para que la serie de tiempo sea estacionaria). Cualquier serie de tiempo "no estacional" que muestre patrones y no sea un "ruido blanco" aleatorio se puede modelar con ARIMA. Si una serie de tiempo tiene patrones estacionales, entonces necesita agregar términos estacionales y se convierte en SARIMA (abreviatura de Seasonal ARIMA). De acuerdo a la gráfica generada para el caso de la compañía Nintendo vemos el decremento de los valores precios, esto se evidencia cuando el modelo tiene los parámetros ARIMA (1,1,2) teniendo un pronóstico direccionalmente correcto y que los valores reales observados se encuentran dentro del 95% de confianza.')

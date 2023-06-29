import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from numpy.random import default_rng

# Cargar los datos desde el archivo
datos_01 = np.load('archivo_datos.npy', allow_pickle=True).item()

# Obtener la corriente en la resistencia
corriente_resistencia = datos_01['corriente_resistencia']
varianza=np.var(corriente_resistencia)

# Cargar los datos desde el archivo
datos_02 = np.load('corrientes_medidas_datos.npy', allow_pickle=True).item()

corriente_sensor1=datos_02['corriente_TC']
corriente_sensor2=datos_02['corriente_Shunt']
corriente_sensor3=datos_02['corriente_Hall']

# Cargar los datos de estimacion
datos_03 = np.load('resultado_estimacion.npy', allow_pickle=True).item()

valor_estimado_kalman=np.array(datos_03['Estimacion'])

#Tiempo de muestreo
tmin=1e-3
t=np.arange(0,10+tmin,tmin)

# Resistencia con Ruido y Varianza
plt.figure(1, figsize=(10, 10))
plt.plot(t, corriente_resistencia, linewidth=1.2, color='black', label='Valor Real')
plt.plot(t, corriente_sensor1, linewidth=0.5, color='blue', label='Sensor 1')
plt.plot(t, corriente_sensor2, linewidth=0.5, color='green', label='Sensor 2')
plt.plot(t, corriente_sensor3, linewidth=0.5, color='red', label='Sensor 3')
plt.plot(t, valor_estimado_kalman, linewidth=0.7, color='yellow', label='Estimación Kalman')
plt.xlabel('Muestras')
plt.ylabel('Corriente')
plt.title('Comparacion de Corrientes')
plt.legend()
plt.grid(True)
plt.show()
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from numpy.random import default_rng

#Tiempo de muestreo
tmin=1e-3
t=np.arange(0,10+tmin,tmin)

#Constantes del modelo
R, L, C = 10, 5, 0.2

#Entrada tipo escalon unitario
Vi=(np.ones(len(t))*10)

# Función modelo
def model(x, Vi, R, L, C):
    aux = np.array([-x[0]/(C*R) + x[1]/C, 
                    -x[0]/L + Vi/L])
    return aux

#Modelo de Espacio de estados
#Matriz A
A = np.array([[-1/(C*R), 1/C],
              [-1/L,       0]])
#Matriz B
B = np.array([0, 1/L])

#Condiciones iniciales
y=np.zeros((2,len(t)))
y[:,0]=np.array([1,0.1])

# Ruido de proceso: wk ~ N(0, Q)
sigma_wk = 0.004 
wk = lambda desv: np.vstack([norm.rvs(0, desv, size=(1, 1)), 0])

# Arreglos para almacenar los datos
corriente_resistencia = np.zeros(len(t))
corriente_capacitancia = np.zeros(len(t))
corriente_inductancia = np.zeros(len(t))
corriente_fuente = np.zeros(len(t))


# Simulación del sistema con ruido
for i in range(1, len(t)):
    dt = t[i] - t[i - 1]
    x = y[:, i - 1]
    u = Vi[i]
    w = wk(sigma_wk)
    y[:, i] = y[:, i - 1] + dt * model(x, u, R, L, C) + wk(sigma_wk).flatten()

    corriente_resistencia[i] = y[0, i]-y[1, i]  # Corriente antes de la resistencia
    corriente_capacitancia[i] = y[1, i]  # Corriente en la capacitancia
    corriente_inductancia[i] = y[0, i]   # Corriente en la inductancia
    corriente_fuente[i] = u  # Corriente entregada por la fuente

# Gráfico de la respuesta del sistema con y sin ruido
plt.figure(1, figsize=(10, 10))
plt.plot(t, y[0, :], label='i con ruido')
plt.plot(t, y[1, :], label='v con ruido')


# Simulación del sistema sin ruido
y_no_noise = np.zeros((2, len(t)))
y_no_noise[:, 0] = np.array([1, 0.1])
for i in range(1, len(t)):
    dt = t[i] - t[i - 1]
    x = y_no_noise[:, i - 1]
    u = Vi[i]
    y_no_noise[:, i] = y_no_noise[:, i - 1] + dt * model(x, u, R, L, C)
    


# Guardar los datos en un archivo (por ejemplo, archivo_datos.npy)
datos = {
    'corriente_resistencia': corriente_resistencia,
    'corriente_capacitancia': corriente_capacitancia,
    'corriente_inductancia': corriente_inductancia,
    'corriente_fuente': corriente_fuente,
}
np.save('archivo_datos.npy', datos)


plt.plot(t, y_no_noise[0, :], label='i sin ruido')
plt.plot(t, y_no_noise[1, :], label='v sin ruido')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.title('Respuesta del sistema con y sin ruido')
plt.legend()
plt.grid(True)


# Gráfico de la respuesta del sistema con y sin ruido
plt.figure(2, figsize=(10, 10))
plt.plot(t, corriente_resistencia, label='Corriente en la resistencia')
plt.plot(t, corriente_capacitancia, label='Corriente en la capacitancia')
plt.plot(t, corriente_inductancia, label='Corriente en la inductancia')
plt.plot(t, corriente_fuente, label='Corriente entregada por la fuente')

plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.title('Respuesta del sistema con y sin ruido')
plt.legend()
plt.grid(True)
plt.show()


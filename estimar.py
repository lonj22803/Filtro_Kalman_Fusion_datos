import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from numpy.random import default_rng
from FiltroKalman.kalman import Kalman


#Cargamos las lecturas de tension dadas por los sensores
# Cargar los datos desde el archivo
datos_2 = np.load('datos_sensores.npy', allow_pickle=True).item()

#Tesiones vistas por la DAQ
vt_TC = datos_2['Vdaq_TC']
vt_Shunt = datos_2['Vdaq_Shunt']
vt_Hall = datos_2['Vdaq_Hall']

# Cargar los datos desde el archivo
datos = np.load('archivo_datos.npy', allow_pickle=True).item()

# Obtener la corriente en la resistencia
corriente_resistencia = datos['corriente_resistencia']

#Recuperamos la medicion de cada uno de los sensores
i_sensor_TC=((vt_TC+9.5)*(5/(1.769*3.5)))
i_sensor_Shunt=(vt_Shunt + 9.5)*(1/(1.769*0.7))
i_sensor_Hall= (vt_Hall + 9.5)*(1/(1.07*0.6*2))


# Guardar los datos en un archivo (por ejemplo, archivo_datos.npy)
datos_3 = {
    'corriente_TC': i_sensor_TC,
    'corriente_Shunt': i_sensor_Shunt,
    'corriente_Hall': i_sensor_Hall,
}
np.save('corrientes_medidas_datos.npy', datos_3)

#Tiempo de muestreo
tmin=1e-3
t=np.arange(0,10+tmin,tmin)
rng = default_rng()

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

# Simulación del sistema con ruido
for i in range(1, len(t)):
    dt = tmin
    x = y[:, i - 1]
    u = Vi[i]
    w = wk(sigma_wk)
    y[:, i] = y[:, i - 1] + dt * model(x, u, R, L, C) + wk(sigma_wk).flatten()


#Simulacion Filtro de Kalman 
  # Aplicar kalman
F = np.array([1])
H = np.array([1])
Q = np.array([4e-2])
R = np.array([0.01])
#F = np.array(self.planta.parametros['A'])
#print(F)
#H = np.array([[1, 0]])
#Q = ((1e-5)**2)*np.array([[1, 0],[0, 1]])
#R = ((1e-2)**2)
filtro = Kalman(F, H, Q, R)
filtro.primera_iter()
salida_kalman = [[], []]
for i in range(len(i_sensor_TC)):
    #for medida in datos_DAQ:
    for medida in [i_sensor_TC,i_sensor_Hall,i_sensor_Shunt]:
        x, p = filtro.kalman(medida[i], 0, 0)#, np.array([0, 0]), np.array([[1e-5, 0], [0, 1e-3]]))
        #x, p = filtro.kalman(corriente_resistencia[i], 0, 0)
    salida_kalman[0].append(x)
    salida_kalman[1].append(p)

datos_estimacion=np.array(salida_kalman[0])

datos_4 = {
    'Estimacion': datos_estimacion,
}
np.save('resultado_estimacion.npy', datos_4)


plt.figure(1, figsize=(10, 10))
plt.plot(t,salida_kalman[0], label="estimacion" )
plt.grid(True)
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.title('Estimación de corriente entregada por el Filtro de Kalman')
plt.show()
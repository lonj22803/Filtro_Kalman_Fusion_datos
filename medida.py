import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from numpy.random import default_rng

#Tiempo de muestreo
tmin=1e-3
t=np.arange(0,10+tmin,tmin)

#Modelamiento de los sensores
"""
Para llevar a cabo la medicion de corriente se tendra en cuenta:la tension en el capcitor y la corriente que circula por 
la inductacia, para estos se utilizaran 

"""
#Agregamos Ruido Gaussiano a cada uno de los sensores

ruido=np.random.normal(0,0.02,1)
ruido1=np.random.normal(0,0.01,1)
ruido2=np.random.normal(0,0.01,1)
#Agregamos un outlier que tendran solo los sensores hall y TC


#Modelamos un TC
def v_tc(i_ent=list()):
    i_out=list()
    rel_trans= 5
    for i in i_ent:
        outlier= np.random.normal(0,10,1)
        if  abs(outlier)>30:
             outlier=0.004*outlier 
        else:
            outlier=0
        tc_out= i/rel_trans - ruido + outlier 
        i_out.append(tc_out)
    #Usamos una resistencia shunt para la salida
    v_shunt=np.array(i_out)*3.5
    #Usamos un AD620 para llevar a cabo la adquisicion Rg=62kOhms
    Vdaq_tc=((v_shunt*1.796)-9.5)

    return Vdaq_tc

#Modelamos una resistencia Shunt
def v_shunt(i_ent=list()):
    rel_shunt=0.7
    v_out=list()
    for i in i_ent:
        shunt_out=i*rel_shunt+ruido1
        v_out.append(shunt_out)
    #Usamos un AD620 para lleavr a cabo la adquisicion Rg=62kOhms
    v_sh=np.array(v_out)
    Vdaq_sh=((v_sh*1.796)-9.5)

    return Vdaq_sh

#Modelamos un sensor de efecto Hall
def v_hall(i_ent=list()):
    resistencia= 2
    sensibilidad=0.6
    v_h=list()
    for i in i_ent:
        outlier= np.random.normal(0,10,1)
        if  abs(outlier)>30:
            outlier=0.008*outlier 
        else:
            outlier=0
        hall=(i*sensibilidad*resistencia)-ruido2+outlier
        v_h.append(hall)
    #Usamos un AD620 para llevar a cabo la adquisicion Rg=700kOhm
    v_ha=np.array(v_h)
    Vdaq_sh=((v_ha*1.07)-9.5)

    return Vdaq_sh


# Cargar los datos desde el archivo
datos = np.load('archivo_datos.npy', allow_pickle=True).item()

# Obtener la corriente en la resistencia
corriente_resistencia = datos['corriente_resistencia']

current=list(corriente_resistencia)

tc=v_tc(current)
shunt=v_shunt(current)
hall=v_hall(current)

datos_Adquisicion = {
    'Vdaq_TC': tc,
    'Vdaq_Shunt': shunt,
    'Vdaq_Hall': hall,
}
np.save('datos_sensores.npy', datos_Adquisicion)

# Graficar las tensiones leidas por cada sensor
plt.figure(1, figsize=(10, 10))
plt.plot(t, tc, label='Vdaq_TC')
plt.plot(t, shunt, label='Vdaq_Shunt')
plt.plot(t, hall, label='Vdaq_Hall')

# Agregar leyenda y etiquetas de los ejes

plt.xlabel('Tiempo')
plt.ylabel('Tension de Adquision')
plt.title('Lecturas de tension en la DAQ')
plt.legend()
plt.grid(True)

# Mostrar el gráfico
plt.show()
        
    

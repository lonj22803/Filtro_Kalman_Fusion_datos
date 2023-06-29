"""
En este modulo implementa el filtros de Kalman para la asignatura Sistemas Avanzados de Medicion.
Juan Jose Londoño Cardenas-UTP
Maeatria en Ingenieria Electrica
16/06/2023

"""
import matplotlib.animation as animation
import numpy as np
from numpy.linalg import svd

class Kalman:
    """
    Implementación del filtro de Kalman.
    """
    def __init__(self, F, H, Q, R):
        """
        Inicializa los parámetros del filtro de Kalman.

        Args:
            F: Matriz de transición de estado.
            H: Matriz de observación.
            Q: Covarianza del proceso.
            R: Covarianza de la medición.
        """
        self.F = F
        self.H = H
        self.Q = Q
        self.R = R
    
        # Variables para uso en aplicaciones del filtro de Kalman en tiempo real
        self.Xprior = None
        self.Pprior = None
        
        self.contador = True
        
    def simulacion(self, medidas, x0, P0):
        """
        Ejecuta una simulación del filtro de Kalman para una secuencia de mediciones.

        Args:
            medidas: Lista de medidas.
            x0: Estado inicial.
            P0: Covarianza inicial.

        Returns:
            Tupla de arreglos con el estado estimado y la covarianza en cada paso.
        """
        X_res = list()
        P_res = list()
        
        for medida in medidas:
            if type(medida) != 'numpy.array':
                ym = np.array([medida])
            else:
                ym = medida

            if medida == medidas[0]:
                Xprior = np.dot(self.F, x0).reshape(-1)
                Pprior = np.dot(self.H, P0) + self.Q
            else:
                Xprior = np.dot(self.F, Xposterior)
                Pprior = np.dot(self.H, Pposterior) + self.Q

            if len(self.H) > 1:
                u, s, v = svd(self.H @ Pprior @ self.H.T + self.R)
                K = Pprior @ self.H.T @ (v.T @ np.identity(len(s))* 1/s @ u.T)
                Xposterior = Xprior + K @ (ym - self.H @ Xprior)
                Pposterior = (np.identity(len(K)) - K @ self.H) @ Pprior
            else:
                K = Pprior * self.H.T * 1/(self.H * Pprior * self.H.T + self.R)
                Xposterior = Xprior + K * (ym - self.H * Xprior)
                Pposterior = (np.identity(len(K)) - K * self.H) * Pprior

            

            X_res.append(Xposterior[0])
            P_res.append(Pposterior[0])
        
        return np.array(X_res), np.array(P_res)
    
    def primera_iter(self):
        """
        Establece el indicador de la primera iteración del filtro de Kalman.
        """
        self.contador = True
        
    def kalman(self, medida, x0, P0):
        """
        Realiza una iteración del filtro de Kalman.

        Args:
            medida: Medida actual.
            x0: Estado inicial.
            P0: Covarianza inicial.

        Returns:
            Tupla con el estado estimado y la covarianza posterior.
        """
        
        if type(medida) != 'numpy.array':
            ym = np.array([medida])
        else:
            ym = medida

        if self.contador:
            self.Xprior = np.dot(self.F, x0)#.reshape(-1)
            self.Pprior = np.dot(self.H, P0) + self.Q
            self.contador = False
        else:
            self.Xprior = np.dot(self.F, self.Xposterior)
            self.Pprior = np.dot(self.H, self.Pposterior) + self.Q

        if len(self.F) > 1:
            u, s, v = svd(self.H @ self.Pprior @ self.H.T + self.R)
            K = self.Pprior @ self.H.T @ (v.T @ np.identity(len(s))* 1/s @ u.T)
            self.Xposterior = self.Xprior + K @ (ym - self.H @ self.Xprior)
            self.Pposterior = (np.identity(len(K)) - K @ self.H) @ self.Pprior
        else:
            K = self.Pprior * self.H.T * 1/(self.H * self.Pprior * self.H.T + self.R)
            self.Xposterior = self.Xprior + K * (ym - self.H * self.Xprior)
            self.Pposterior = (np.identity(len(K)) - K * self.H) * self.Pprior
        return self.Xposterior[0], self.Pposterior[0]
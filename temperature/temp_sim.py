import numpy as np
from numba import jit
import time
import tqdm

def timeit_dec(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print('{} - {} ms - DONE'.format(method.__name__, (te - ts) * 1000))
        return result
    return timed

@timeit_dec
def temperatureprofile(position=None, limit=45, delta_x=0.01, delta_t=0.1, time=360.0, length=0.4, **parameters):
    """
    Lösung der fourierischen Wärmeleitgleichung
    
    :param position: (int) Beobachtungspunkt [pos] = m {Standardwert None}
    :param limit: (float) Grenztemperatur [limit] = °C {Standardwert 45}
    :param delta_x: (float) Schritte für die Ortsdiskretisierung [Δx] = m {Standardwert 0.01}
    :param delta_t: (float) Schritte für die Zeitdiskretisierung [Δt] = s {Standardwert 0.01}
    :param time: (float) Dauer der Simulation [t] = s {Standardwert 0.01}
    :param length: (float) Länge des Stabes [l] = m {Standardwert 0.4}
    :param parameters: (Dictionary) die für die Berechnung relevanten Parameter
    """
    
    par_c = parameters['c']
    par_rho = parameters['rho']
    par_lambda = parameters['lambda']
    par_alpha = parameters['alpha']
    par_r = parameters['radius']
    par_a = par_lambda/par_c/par_rho
    
    temp_boundary = parameters['temp_boundary']
    temp_init = parameters['temp_init']
    
    # Stabilitätskriterium nach Krebs et al. 
    if delta_t != 0.1:
        delta_x = (2*par_a*delta_t)**(1/2)
    if delta_x != 0.01:
        delta_t = (delta_x**2/par_a)/2
    
    # Dimensionslose Parameter
    A = par_lambda * delta_t / par_c / par_rho / delta_x**2
    B = 2 * par_alpha * delta_t / par_c / par_rho / par_r
    
    # Anzahl der time und location steps
    time_steps = (int) (time/delta_t) + 1
    location_steps = (int) (length/delta_x) + 1
    
    # Anfangstemperaturprofil
    temp_profile = np.asarray([np.insert(np.full(location_steps-1, temp_boundary), 0, temp_init)])
    
    # der Listenversatz ermöglicht die Anwendung von np.ufunc's 
    for time in tqdm.trange(time_steps):
        t1 = np.delete(temp_profile[time], 0) #Listenversatz 1
        t2 = np.delete(t1, 0) #Listenversatz 2
        t0 = np.resize(temp_profile[time], t2.size) # Längenkorrektur
        t1 = np.resize(t1, t2.size) # Längenkorrektur
        
        last = t0[-1] + (t0[-2]-t0[-1]) * A - B * (t0[-1]-t0[-2]) # Endtemperatur
        tnkp1 = t1 + (- 2*t1 + t0 + t2) * A - B * (t1-temp_boundary) 
        tnkp1 = np.append(np.insert(tnkp1, 0, temp_init), last) # Einfügen von Anfangstemperatur und Endtemperatur

        temp_profile=np.vstack([temp_profile, tnkp1])
    
    #Speichern des Temperaturprofils als CSV
    np.savetxt("temp_profile.csv", temp_profile, delimiter=";")
        
    time = find_time(
        temp_profile=temp_profile,
        limit=limit,
        position=position, 
        delta_t=delta_t, 
        delta_x=delta_x, 
        length=length)
        
    #print(temp_profile)
        
    return temp_profile

def find_time(temp_profile, limit, position, delta_t, delta_x, length):
    """
    Ermittelt an jeder Stelle den Zeitpunkt der Grenztemperatur aus dem Temperaturprofil
    
    :params temp_profile: (numpy.ndarray) Temperaturprofil Spalten = Positionen | Reihen = Zeitpunkte  [temp_profile] = °C
    :params limit: (float)  Grenztemperatur [limit] = °C 
    :params position: (float) Beobachtungspunkt [position] = m   
    :params delta_t: (float) Schritte für die Zeitdiskretisierung [Δt] = s
    :params delta_x: (float) Schritte für die Ortsdiskretisierung [Δx] = m
    :params length: (float) Länge des Stabes [l] = m
    """

    temp_profile_T = np.transpose(temp_profile)
    time_cum_sum = np.insert(np.cumsum(np.full(len(temp_profile_T[0])-1, delta_t)),0,0) # cumsum deltat
    time_gt = [np.interp(limit, temp_profile_T[i], time_cum_sum) for i in range(len(temp_profile_T))] # Linear Interpolierte Zeit
    
    if position is not None:
        #location_cumsum = np.insert(np.cumsum(np.full(len(temp_profile[0])-1, delta_x)),0,0)
        location_cumsum =  np.resize(np.arange(0.005, length+delta_x, delta_x),len(time_gt))
        print(len(location_cumsum))
        time = np.interp(position, location_cumsum, time_gt)
        print("An der Position 0.075 m wird die Grenztemperatur nach {} s erreicht".format(round(time,3)))
        return time
    else:
        print("Zeitliste: {}".format(time_gt))
        return time_gt
    
if __name__=="__main__":
    # todo: testen ob direktübergabe schneller
    parameters = {
        'c' : 888,
        'rho' : 2700,
        'lambda' : 237,
        'alpha' : 8.000,
        'radius' : 0.0025,
        'temp_boundary' : 22.00, 
        'temp_init' : 300}
    
    #TESTS
    print("| Starte Kalkulation mit Parametern |")
    temp_profile = temperatureprofile(**parameters)
    print(temp_profile)
    #temperatureprofile(delta_x=0.1,**parameters)
    #temperatureprofile(delta_t=1,**parameters)

    #print("| Starte Kalkulation mit Position |")
    #temperatureprofile(position = 0.075, **parameters)


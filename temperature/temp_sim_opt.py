import numpy as np
from numba import njit, prange, vectorize, jit
import time
import tqdm
import timeit, functools

def timeit_dec(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('{} - {} ms - DONE'.format(method.__name__, (te - ts) * 1000))
        return result
    return timed


    parameters = {
        'c' : 888,
        'rho' : 2700,
        'lambda' : 237,
        'alpha' : 8.000,
        'radius' : 0.003,
        'temp_boundary' : 22.00, 
        'temp_init' : 300}

@timeit_dec
def temperatureprofile(
                        position=None, 
                        limit=45, 
                        delta_x=0.01, 
                        delta_t=0.1, 
                        time=360.0, 
                        length=0.4, 
                        c = 888,
                        rho = 2700, 
                        lambda_ = 237, 
                        alpha = 8.000, 
                        radius = 0.003, 
                        temp_boundary = 22.00, 
                        temp_init = 300
                        ):
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
    
    a = lambda_/c/rho
    
    temp_boundary = temp_boundary
    temp_init = temp_init
    
    # Stabilitätskriterium nach Krebs et al. 
    if delta_t != 0.1:
        delta_x = (2*a*delta_t)**(1/2)
    if delta_x != 0.01:
        delta_t = (delta_x**2/a)/2
    
    # Dimensionslose Parameter
    A = lambda_ * delta_t / c / rho / delta_x**2
    B = 2 * alpha * delta_t / c / rho / radius
    
    # Anzahl der time und location steps
    time_steps = (int) (time/delta_t) + 1
    location_steps = (int) (length/delta_x) + 1
    
    blueprint = np.empty([location_steps])
    blueprint[0] = temp_init
	
    # Anfangstemperaturprofil
    temp_profile = np.empty([time_steps+1, location_steps])
    temp_profile[0] = np.append([temp_init], np.full(location_steps-1, temp_boundary))
    
    # der Listenversatz ermöglicht die Anwendung von np.ufunc's 
    for time in range(time_steps):
        t1 = temp_profile[time][1:] #Listenversatz 1
        t2 = t1[1:] #Listenversatz 2
        t0 = temp_profile[time][:-2] # Längenkorrektur
        t1 = t1[:-1] # Längenkorrektur
        
        blueprint[-1] = t0[-1] + (t0[-2]-t0[-1]) * A - B * (t0[-1]-t0[-2]) # Endtemperatur
        blueprint[1:-1] = t1 + (- 2*t1 + t0 + t2) * A - B * (t1-temp_boundary) # Einfügen von Anfangstemperatur und Endtemperatur
        temp_profile[time+1] = blueprint
    
   
    #Speichern des Temperaturprofils als CSV
    #np.savetxt("temp_profile.csv", temp_profile, delimiter=";")
    time = find_time(temp_profile=temp_profile, limit=limit, position=position, delta_t=delta_t, delta_x=delta_x, length=length)
    
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
    time_cum_sum = np.append([0],np.cumsum(np.full(temp_profile_T[0].size-1, delta_t))) # cumsum deltat
    time_gt = [np.interp(limit, temp_profile_T[i], time_cum_sum) for i in range(len(temp_profile_T))] # Linear Interpolierte Zeit
    
    if position is not None:
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

    #TESTS
    print("| Starte Kalkulation mit Parametern |")
    temp_profile = temperatureprofile(c = 888,rho = 2700, lambda_ = 237, alpha = 8.000, radius = 0.003, temp_boundary = 22.00, temp_init = 300)
    # t = timeit.Timer(functools.partial(temperatureprofile)) 
    # print(t.timeit(10))
    #import timeit
    #t = timeit.Timer(temperatureprofile(c = 888,rho = 2700, lambda_ = 237, alpha = 8.000, radius = 0.003, temp_boundary = 22.00, temp_init = 300))  
    #print(t.timeit(5))
    #temperatureprofile(delta_x=0.1,**parameters)
    #temperatureprofile(delta_t=1,**parameters)

    #print("| Starte Kalkulation mit Position |")
    #temperatureprofile(position = 0.075, **parameters)


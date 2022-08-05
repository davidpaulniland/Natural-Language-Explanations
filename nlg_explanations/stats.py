from scipy.stats import f_oneway
import seaborn as sns 
import pylab
import matplotlib.pyplot as plt

import numpy as np
import scipy.stats
import pandas as pd
import statistics
def premilinary_pilot(): 
    def stats (array):
        def std_finder(array):
            array = statistics.stdev(array)
            print("standard deviation is...", array)
        std_finder(array)

        def rangefinder(array):
            
            print("range is...", np.min(array),", ", np.max(array))
        rangefinder(array)





    ml_exp = np.array([1,4,1,4,3,4,3,4,3,2,2,4])
    xai_exp = np.array([1,2,1,4,1,4,2,3,3,1,2,4])

    pfi_cancer = np.array([5,3,2,5,5,5,4,3,5,4,5,5])
    pdp_cancer = np.array([4,2,5,4,5,4,3,5,3,2,5,5])
    ale_cancer = np.array([1,5,5,2,4,2,2,3,4,5,2])
    pfi_rain = np.array([5,1,3,5,3,4,2,2,5,3,5,5])
    pdp_rain = np.array([3,1,4,4,3,1,1,3,2,4,5,5])
    ale_rain = np.array([5,2,5,4,4,1,4,4,4,4,5,5])


    #stats(ml_exp)
    #stats(xai_exp)
    print("pfi_cancer________")
    stats(pfi_cancer)
    print("pdp_cancer________")
    stats(pdp_cancer)
    print("ale_cancer_______") 
    stats(ale_cancer)
    print("pfi_rain_________") 
    stats(pfi_rain)
    print("pdp_rain_________") 
    stats(pdp_rain)
    print("ale_rain__________") 
    stats(ale_rain)

    print(round(1.4142135623730951, 2))


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import time 

from scipy.interpolate import interp1d
from tables import vlarray




# Gotta change the saved files and input file names into autoControl.py and run each of the test before processing them here
# check the format of each of the files (OIVP ramp input is in 0.1s timestep, OIVP output is 0.01s timestep, carla outputs 0.04stimestep)
# check correct file names for ramp input and OIVP outputs are in the correct order with a correct offset
''' -------------------------------------------------------------------------------------- file 1 below'''

# anything coded after show will only be ran when the plot is closed



sim_array = ["Throttle_AC18_ramp1_result1.csv", "Throttle_AC18_ramp2_result2.csv", "Throttle_AC18_ramp3_result2.csv"]
oivp_file_array = ['test1_expanded_result_1.csv', 'test2_expanded_result_2.csv', 'test3_expanded_result_2.csv']
offset_array = [6.6,1.2,1.2]               # different tests have differnt offset times because the recording is started before the ramp files are ran in the OIVP  
test_amounts = len(sim_array)

fig, axs = plt.subplots(test_amounts)
fig.suptitle('OIVP VS CARLA')
for i in range(len(sim_array)):

# The expanded test results are downloaded into the carla/PythonAPI/examples directory 
    simData = pd.read_csv(sim_array[i])
    oivpData = pd.read_csv(oivp_file_array[i])
    oivpData.replace(' ', np.nan, inplace=True) # Pandas doesnt recognize empty strings as null, for some reason the empty data is actually a string with a space bar 
    oivp_start_offset_time = offset_array[i] # since we start recording before we run the file, there is a time that we should clip out from the file  

    simulation_time = simData['Time (s)']
    simulation_velocity = simData['Velocity (m/s)']

# ramp inputs
    rampInput_time = oivpData['Time_Ramp']
    rampInput = oivpData['Velocity_Ramp']

    rampInput_time = rampInput_time.dropna().astype(float)          # remove the NaN data from the dataframe 
    rampInput_time = rampInput_time - rampInput_time[0]
    rampInput = rampInput.dropna().astype(float)

# car outputs 
    rampOutput_time = oivpData['Time_Car']
    rampOutput = oivpData['Velocity_Car']

    rampOutput_time = rampOutput_time.dropna().astype(float)
    rampOutput_time = rampOutput_time - rampOutput_time[0]
    rampOutput = rampOutput.dropna().astype(float)
    
    outputStartIndex = int(oivp_start_offset_time*100) # since the file is in miliseconds, multiply seconds by 100 indexes
    rampOutput = rampOutput.iloc[outputStartIndex:]
    rampOutput_time = rampOutput_time.iloc[:-outputStartIndex]

    axs[i].set_title('Test' + str(i+1))
    axs[i].plot(simulation_time,simulation_velocity,label='Simulation', color = "blue")
    axs[i].plot(rampInput_time,rampInput, label='Ramp Input', color = "brown")
    axs[i].plot(rampOutput_time,rampOutput, label='OIVP Output', color = "green")
    axs[i].legend()
    axs[i].grid()
    axs[i].set_xlabel('Time (s)')
    axs[i].set_ylabel('Velocity (m/s)')

fig.tight_layout(pad=1.0)
    
plt.show()





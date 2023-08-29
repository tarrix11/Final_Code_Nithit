import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


delay = 6.6             # time delay from simulation to oivp (s) THIS FEATURE DOESNT WORK CUZ THE DELAY PART 
remove_time_indexes = delay*100


# Comparing OIVP 0.01s clock vs  interpolated Carla 0.04s clock
sim_output = pd.read_csv('Throttle_AC18_ramp1_result1.csv') # carla ramp input files
oivp_output = pd.read_csv('test1_expanded_result_1.csv')

sim_time = round(sim_output['Time (s)'],2)   # check in your collected data files what are their header names
sim_velocity = sim_output['Velocity (m/s)']

oivp_time = oivp_output['Time_Car']
oivp_time = oivp_time.replace(' ', np.nan).dropna().astype(float)
oivp_time = round(oivp_time - oivp_time[0],2)

oivp_velocity = oivp_output['Velocity_Car']
oivp_velocity = oivp_velocity.replace(' ', np.nan).dropna().astype(float)
oivp_velocity = oivp_velocity.iloc[660:]
oivp_time = oivp_time.iloc[:len(oivp_velocity)-660] # THIS DOESNT WORK

# interp1d returns a function where you can pass in the x values to get the interpolated y values 
interpolate_sim = interp1d(sim_time,sim_velocity,kind = 'linear')


interpolated_sim_velocity = []
sim_time_limit = sim_time.iloc[-1]
oivp_time_limit = (oivp_time).iloc[-1]
print(f"max sim time: {sim_time_limit}")
print(f"max oivp time: {oivp_time_limit}")     

# to interpolate, the interp values can only be between 0-max_sim_time
sim_time_array = []
for i in oivp_time:
    if i < sim_time_limit:              # only if the oivp time is below the simulation time limit 
        interpolated_sim_velocity.append(interpolate_sim(i))    # this creates a array of array(nums) so we put it into a 
        sim_time_array.append(i)                                # only appends the time value that is below the simulation limit 

    else:
        pass


interpolated_sim_velocity = pd.DataFrame(interpolated_sim_velocity)
values = pd.DataFrame()

values['Time'] = sim_time_array
values['final simulation velocity'] = interpolated_sim_velocity
values['OIVP velocity'] = oivp_velocity
values['Error'] = values['OIVP velocity'] - values['final simulation velocity']
oivp_velocity = oivp_velocity.iloc[:len(interpolated_sim_velocity)]
print(oivp_velocity)
#values['Error'] = oivp_velocity.iloc[:] - interpolated_sim_velocity.iloc[:]
print(len(sim_time_array))
print(len(interpolated_sim_velocity))

plt.plot(sim_time_array,oivp_velocity, label = "OIVP")
plt.plot(sim_time_array,interpolated_sim_velocity, label = "Simulation")
plt.legend()
plt.plot(oivp_time,values['Error'], label = "Error")
print(f"Error vs Time {values['Error']}")
print(f"time:: {oivp_time} ")
print(f"len {len(oivp_velocity)}")
plt.show()


# right now the two graphs are not aligned 



    

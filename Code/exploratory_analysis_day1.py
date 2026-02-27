#%%
import pandas as pd
import matplotlib.pyplot as plt

#%%
# Load the data
data = pd.read_csv('../Data/mystery_virus_daily_active_counts_RELEASE#1.csv', parse_dates=['date'], header=0, index_col=None)

<<<<<<< Updated upstream
#%%
# Make a plot of the active cases over time
=======

# Make a plot of the active cases over time

plt.figure(figsize=(10,6)) 
plt.plot(data["day"], data["active reported daily cases"], marker="o", linewidth=2) # Labels and title 
plt.xlabel("Day") 
plt.ylabel("Active Cases") 
plt.title("Mystery Virus: Daily Active Case Counts Over Time") 
plt.tight_layout() 
plt.show()

#1 What do you notice about the initial infections?
# Initially in the earlier days the infection has spead to few pople. But the number of active cases increases exponentially 
# because when more people are infected, they can spread teh infecton much more quickly

#2 How could we measure how quickly its spreading?
# By finding the rate of change of the graph we can measure how quickly the infection spreads as a rate across days. 

#3 What information about the virus would be helpful in determining the shape of the outbreak curve?
# It would be helpful to know how the virus spreads. Viruses that spread more easily will likely have a greater rate of 
# infection and result in a greater outbreak curve. For example, an airporn virus may infect a population faster than one 
<<<<<<< Updated upstream
# that spreads fluids
>>>>>>> Stashed changes
=======
# that spreads fluids
>>>>>>> Stashed changes

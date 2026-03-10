# Written by Will Marschall
# Updated: 2/26/2026

#%%
# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

#%%
# Load the data
data = pd.read_csv('../Data/mystery_virus_daily_active_counts_RELEASE#1.csv', parse_dates=['date'], header=0, index_col=None)

# We ceck the data
t_all = data["day"].to_numpy(dtype=float)
I_all = data["active reported daily cases"].to_numpy(dtype=float)

# Choosing an early exponential window
fit_start_day = 10
mask = t_all >= fit_start_day

t = t_all[mask]
I = I_all[mask]

# I am using time-shifted t for numerical stability
t0 = t.min()
tau = t - t0

#%%
# We have day number, date, and active cases. We can use the day number and active cases to fit an exponential growth curve to estimate R0.
# Let's define the exponential growth function

# Here is my exponential growth model: I(t) = I0 * exp(r * (t - t0))
def exponential_growth(tau, I0, r):
    return I0 * np.exp(r * tau)

# Fit the exponential growth model to the data. 
# We'll use a handy function from scipy called CURVE_FIT that allows us to fit any given function to our data. 
# We will fit the exponential growth function to the active cases data. HINT: Look up the documentation for curve_fit to see how to use it.

# Initial guesses for curve_fit
p0 = [I[0], 0.1]  

params, cov = curve_fit(exponential_growth, tau, I, p0=p0, maxfev=10000)
I0_fit, r_hat = params

# Approximate R0 using this fit
D_min = 2 + 5
D_max = 2 + 9
D_mid = 0.5 * (D_min + D_max)

R0_min = np.exp(r_hat * D_min)
R0_mid = np.exp(r_hat * D_mid)
R0_max = np.exp(r_hat * D_max)

print(f"Fit window: day >= {fit_start_day}")
print(f"Estimated exponential growth rate r: {r_hat:.4f} per day")
print(f"Estimated R0 range using D in [{D_min:.1f}, {D_max:.1f}] days:")
print(f"  R0_min (D={D_min:.1f}) = {R0_min:.3f}")
print(f"  R0_mid (D={D_mid:.1f}) = {R0_mid:.3f}")
print(f"  R0_max (D={D_max:.1f}) = {R0_max:.3f}")

# Add the fit as a line on top of your scatterplot.

# Here is my plot with data + fitted exponential curve
plt.figure(figsize=(10, 6))
plt.scatter(t_all, I_all, label='Observed active cases (Release #1)')
tau_dense = np.linspace(tau.min(), tau.max(), 300)
t_dense = tau_dense + t0
I_fit_dense = exponential_growth(tau_dense, I0_fit, r_hat)
plt.plot(t_dense, I_fit_dense, linewidth=2, label="Exponential fit (early window)")

plt.xlabel('Day')
plt.ylabel("Active Infections (Active Reported Daily Cases)")
plt.title("Mystery Virus: Exponential Fit to Early Growth (for R0 estimate)")
plt.legend()
plt.tight_layout()
plt.show()


# What viruses have a similar R0? Use the viruses.html file to find a virus or 2 with a similar R0 and give a 1-2 sentence background of the diseases.

# RESPONSE: COVID-19 (Original) was found to be near my R0 (2.937). It reported to have a R0 of 3 and CFR = 1%, and it is marked under the "Vaccine Available" category. 
# COVID-19 is has an ~5-day incubation period and it can be infectious ~2 days before symptoms.

# RESPONSE: Zika was found to be near my R0 (2.937). It reported to have a R0 of 3 and CFR = 0.02%, and it is marked under the "No Vaccine" category.
# Zika virus is a mosquito-borne pathogen that gained global attention during a 2015 outbreak.

# How accurate do you think your R0 estimate is?

# RESPONSE: my R0 is reasonable but approximate, because we explicitly treat this as an early-epidemic approximation based on fitting exponential growth and then converting.
# The biggest uncertainty is D (infectious period) as we only have limited timing information (symptomatic period 5-9 days). 
# Also the fitted growth rate r can change depending on which days we treat as the "early exponential window" (we chose day >= 10, and viruses.html notes that R0 estimates vary with behavior and population density.
# exploratory_analysis_day4_2g
# Written by Will and Reagan
# Updated 3/10/2026

#For this section we want to plot the SEIR model against day 3 release
#The code for the SEIR model was already written for Day3_2e
#Below is a copy of that code for clarity and continuity

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def euler_seir(beta, sigma, gamma, S0, E0, I0, R0, timepoints, N, h=1.0):

    # This is our Euler’s method for SEIR system.
    # timepoints = np.asarray(timepoints, dtype=float)
    n = len(timepoints) 

    S = np.zeros(n, dtype=float)
    E = np.zeros(n, dtype=float)
    I = np.zeros(n, dtype=float)
    R = np.zeros(n, dtype=float)

    S[0], E[0], I[0], R[0] = S0, E0, I0, R0

    for i in range(n - 1):
        # SEIR equations that we choose from class
        dS = -beta * S[i] * I[i] / N
        dE = beta * S[i] * I[i] / N - sigma * E[i]
        dI = sigma * E[i] - gamma * I[i]
        dR = gamma * I[i]

        # Comes from the Lecture 3 Euler update: y_{i+1} = y_i + f(t_i, y_i)*h
        S[i + 1] = S[i] + dS * h
        E[i + 1] = E[i] + dE * h
        I[i + 1] = I[i] + dI * h
        R[i + 1] = R[i] + dR * h

        # Decided to check for numerical safety (not explicitly in slides, but prevents tiny negatives due to step size)
        S[i + 1] = max(S[i + 1], 0.0)
        E[i + 1] = max(E[i + 1], 0.0)
        I[i + 1] = max(I[i + 1], 0.0)
        R[i + 1] = max(R[i + 1], 0.0)

    return S, E, I, R


def sse(I_obs, I_model):
    # SSE = Σ (I_obs - I_model)^2
    # We match the Lecture 3 format for the error function that grid search minimizes.
    I_obs = np.asarray(I_obs, dtype=float)
    I_model = np.asarray(I_model, dtype=float)
    return np.sum((I_obs - I_model) ** 2)


def grid_search_fit(timepoints, I_obs, N, S0, E0, I0, R0,
                    beta_range=(0.3, 0.7),
                    sigma_range=(0.1, 0.3),
                    gamma_range=(0.05, 0.25),
                    resolution=15,
                    h=1.0):
    
    # This is our 3-parameter grid search over (beta, sigma, gamma), minimizing SSE.
    betas = np.linspace(beta_range[0], beta_range[1], resolution)
    sigmas = np.linspace(sigma_range[0], sigma_range[1], resolution)
    gammas = np.linspace(gamma_range[0], gamma_range[1], resolution)

    best = {
        "beta": None,
        "sigma": None,
        "gamma": None,
        "sse": np.inf
    }

    total = len(betas) * len(sigmas) * len(gammas)
    count = 0

    for b in betas:
        for s in sigmas:
            for g in gammas:
                count += 1
                S, E, I, R = euler_seir(b, s, g, S0, E0, I0, R0, timepoints, N, h=h)
                err = sse(I_obs, I)

                if err < best["sse"]:
                    best.update({"beta": b, "sigma": s, "gamma": g, "sse": err})

    return best

# Repeat from 2d so that it can run

#Below is where the code changes.Now we need to load data form release #3 to plot it against SEIR

# Load Data Release #3
data = pd.read_csv(
    "C:\\Users\\Reaga\\OneDrive\\Desktop\\BME_2315\\Module-2-Epidemics-SIR-Modeling\\Data\\mystery_virus_daily_active_counts_RELEASE#3.csv",
    parse_dates=["date"],
    header=0,
    index_col=None
)

t = data["day"].to_numpy(dtype=float)
I_obs = data["active reported daily cases"].to_numpy(dtype=float)

# Here are the initial conditions (Lecture 3 pseudocode requires S0,E0,I0,R0,N)

N = 10000.0  # This is what was used in Lecture 2 (not sure what else we would use)

I0 = float(I_obs[0]) 
E0 = 0.0
R0 = 0.0
S0 = N - E0 - I0 - R0

h = 1.0  # this is our daily time step

# Parameter ranges + resolution from Class materials/seir_grid_search.html
best = grid_search_fit(
    timepoints=t,
    I_obs=I_obs,
    N=N,
    S0=S0, E0=E0, I0=I0, R0=R0,
    beta_range=(0.3, 0.7),
    sigma_range=(0.1, 0.3),
    gamma_range=(0.05, 0.25),
    resolution=15,
    h=h
)

beta_best = best["beta"]
sigma_best = best["sigma"]
gamma_best = best["gamma"]
best_sse = best["sse"]

# Lecture 2 defines R0 = beta/gamma
R0_best = beta_best / gamma_best

def exploratory_analysis_day4_2g():

    # This runs the full Day 4 workflow for Data Release #3:
    
    # Load Data Release #3
    data = pd.read_csv(
        "C:\\Users\\Reaga\\OneDrive\\Desktop\\BME_2315\\Module-2-Epidemics-SIR-Modeling\\Data\\mystery_virus_daily_active_counts_RELEASE#3.csv",
        parse_dates=["date"],
        header=0,
        index_col=None
    )

    t = data["day"].to_numpy(dtype=float)
    I_obs = data["active reported daily cases"].to_numpy(dtype=float)

    # Here are the initial conditions (Lecture 3 pseudocode requires S0,E0,I0,R0,N)

    N = 10000.0  # This is what was used in Lecture 2 (not sure what else we would use)

    I0 = float(I_obs[0]) 
    E0 = 0.0
    R0 = 0.0
    S0 = N - E0 - I0 - R0

    h = 1.0  # this is our daily time step
    # ----- 2e: Plot best-fit model vs data -----
    Sb, Eb, Ib, Rb = euler_seir(beta_best, sigma_best, gamma_best, S0, E0, I0, R0, t, N, h=h)

    plt.figure(figsize=(10, 6))
    plt.scatter(t, I_obs, label="Observed active cases (Release #3)", s=25)
    plt.plot(t, Ib, linewidth=2, label="Best-fit SEIR I(t) (Euler)")
    plt.xlabel("Day")
    plt.ylabel("Active infections")
    plt.title("Observed data vs best-fit SEIR model (Euler + grid search)")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    exploratory_analysis_day4_2g()
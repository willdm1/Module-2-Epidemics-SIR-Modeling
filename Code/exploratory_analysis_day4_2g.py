# exploratory_analysis_day4_2g
# Written by Will and Reagan
# Updated 3/14/2026

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

    for b in betas:
        for s in sigmas:
            for g in gammas:
                S, E, I, R = euler_seir(b, s, g, S0, E0, I0, R0, timepoints, N, h=h)
                err = sse(I_obs, I)

                if err < best["sse"]:
                    best.update({"beta": b, "sigma": s, "gamma": g, "sse": err})

    return best

# Repeat from 2d so that it can run

#Below is where the code changes.Now we need to load data form release #3 to plot it against SEIR

def exploratory_analysis_day4_2g():

    # for 2g, we  compare Release #3 to the model fit from Release #2,

    # load Data Release #2 to get the previously fitted parameters
    data2 = pd.read_csv(
        "../Data/mystery_virus_daily_active_counts_RELEASE#2.csv",
        parse_dates=["date"],
        header=0,
        index_col=None
    )

    # We extract the timepoints and observed active cases for Release #2
    t2 = data2["day"].to_numpy(dtype=float)
    I_obs2 = data2["active reported daily cases"].to_numpy(dtype=float)

    N2 = 10000.0
    I0_2 = float(I_obs2[0])
    E0_2 = 0.0
    R0_2 = 0.0
    S0_2 = N2 - E0_2 - I0_2 - R0_2
    h = 1.0

    # We perform the grid search fit on Release #2 data to get the best parameters (beta, sigma, gamma)
    best2 = grid_search_fit(
        timepoints=t2,
        I_obs=I_obs2,
        N=N2,
        S0=S0_2, E0=E0_2, I0=I0_2, R0=R0_2,
        beta_range=(0.3, 0.7),
        sigma_range=(0.1, 0.3),
        gamma_range=(0.05, 0.25),
        resolution=15,
        h=h
    )

    # We extract the best parameters from the grid search fit on Release #2 data
    beta_best_2 = best2["beta"]
    sigma_best_2 = best2["sigma"]
    gamma_best_2 = best2["gamma"]

    # load Release #3 only for comparison/validation
    data3 = pd.read_csv(
        "../Data/mystery_virus_daily_active_counts_RELEASE#3.csv",
        parse_dates=["date"],
        header=0,
        index_col=None
    )

    t3 = data3["day"].to_numpy(dtype=float)
    I_obs3 = data3["active reported daily cases"].to_numpy(dtype=float)

    # We set the initial conditions for Release #3 (same N, but I0 from Release #3)
    N3 = 10000.0
    I0_3 = float(I_obs3[0])
    E0_3 = 0.0
    R0_3 = 0.0
    S0_3 = N3 - E0_3 - I0_3 - R0_3

    # We run the SEIR model with the best parameters from Release #2 on the timepoints of Release #3 to get the model predictions for Release #3
    Sb, Eb, Ib, Rb = euler_seir(
        beta_best_2, sigma_best_2, gamma_best_2,
        S0_3, E0_3, I0_3, R0_3,
        t3, N3, h=h
    )

    # Peak-based error metrics

    # True (data) peak
    true_peak_idx = int(np.argmax(I_obs3))
    true_peak_day = t3[true_peak_idx]
    true_peak_I = I_obs3[true_peak_idx]

    # Model peak (approx)
    model_peak_idx = int(np.argmax(Ib))
    model_peak_day = t3[model_peak_idx]
    model_peak_I = Ib[model_peak_idx]

    # True error and percent relative error (Lecture 4)
    Et_I = true_peak_I - model_peak_I
    pct_et_I = (Et_I / true_peak_I) * 100.0 if true_peak_I != 0 else np.nan

    Et_day = true_peak_day - model_peak_day
    pct_et_day = (Et_day / true_peak_day) * 100.0 if true_peak_day != 0 else np.nan

    print("\nRelease #3 vs model (fit from Release #2): Peak error metrics")
    print(f"True peak:   day {true_peak_day:.0f}, I = {true_peak_I:.1f}")
    print(f"Model peak:  day {model_peak_day:.0f}, I = {model_peak_I:.1f}")
    print(f"Et (peak I) = {Et_I:.1f}, %et = {pct_et_I:.2f}%")
    print(f"Et (peak day) = {Et_day:.1f}, %et = {pct_et_day:.2f}%")

    # plot Release #3 data vs SEIR model fit from Release #2
    plt.figure(figsize=(10, 6))
    plt.scatter(t3, I_obs3, label="Observed active cases (Release #3)", s=25)
    plt.plot(t3, Ib, linewidth=2, label="SEIR model fit from Release #2")
    plt.xlabel("Day")
    plt.ylabel("Active infections")
    plt.title("Release #3 data vs SEIR model fit from Release #2")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    exploratory_analysis_day4_2g()
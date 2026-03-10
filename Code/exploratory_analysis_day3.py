# Updated 3/9/2026

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


def exploratory_analysis_day3():

    # This runs the full Day 3 workflow for Data Release #2:
    
    # Load Data Release #2
    data = pd.read_csv(
        "../Data/mystery_virus_daily_active_counts_RELEASE#2.csv",
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

    # ----- 2c: Here is our Euler’s method solution (we choose to use a reasonable initial parameter guess) -----
    # We chose to pick a mid-range guess from the grid search bounds shown in seir_grid_search.html.
    beta_guess = 0.5
    sigma_guess = 0.2
    gamma_guess = 0.15

    Sg, Eg, Ig, Rg = euler_seir(beta_guess, sigma_guess, gamma_guess, S0, E0, I0, R0, t, N, h=h)

    plt.figure(figsize=(10, 6))
    plt.plot(t, Sg, label="S(t) Euler")
    plt.plot(t, Eg, label="E(t) Euler")
    plt.plot(t, Ig, label="I(t) Euler")
    plt.plot(t, Rg, label="R(t) Euler")
    plt.xlabel("Day")
    plt.ylabel("People (model compartments)")
    plt.title("SEIR compartments solved with Euler’s method (initial parameter guess)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ----- 2d: Here we fit SEIR parameters using grid search minimizing SSE -----
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

    print("Best-fit parameters from 3-parameter grid search (min SSE):")
    print(f"  beta  = {beta_best:.3f}")
    print(f"  sigma = {sigma_best:.3f}")
    print(f"  gamma = {gamma_best:.3f}")
    print(f"  SSE   = {best_sse:.2f}")
    print(f"  Implied R0 (Lecture 2: R0 = beta/gamma) = {R0_best:.3f}")

    # ----- 2e: Plot best-fit model vs data -----
    Sb, Eb, Ib, Rb = euler_seir(beta_best, sigma_best, gamma_best, S0, E0, I0, R0, t, N, h=h)

    plt.figure(figsize=(10, 6))
    plt.scatter(t, I_obs, label="Observed active cases (Release #2)", s=25)
    plt.plot(t, Ib, linewidth=2, label="Best-fit SEIR I(t) (Euler)")
    plt.xlabel("Day")
    plt.ylabel("Active infections")
    plt.title("Observed data vs best-fit SEIR model (Euler + grid search)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ----- 2f: Here we predict peak by running model forward (Lecture 3 “Predicting the peak”) -----
    t_future = np.arange(1, 301, 1, dtype=float)  # run out 300 days
    Sf, Ef, If, Rf = euler_seir(beta_best, sigma_best, gamma_best, S0, E0, I0, R0, t_future, N, h=h)

    peak_idx = int(np.argmax(If))
    peak_day = t_future[peak_idx]
    peak_I = If[peak_idx]

    print("\nPeak prediction from best-fit model (running forward):")
    print(f"  Peak day (model)        = {peak_day:.0f}")
    print(f"  Peak active infections  = {peak_I:.1f}")

    plt.figure(figsize=(10, 6))
    plt.plot(t_future, If, linewidth=2, label="Model I(t) forward prediction")
    plt.axvline(peak_day, linestyle="--", label=f"Peak day = {peak_day:.0f}")
    plt.xlabel("Day")
    plt.ylabel("Active infections (model I)")
    plt.title("Predicted I(t) forward in time (peak from SEIR model)")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    exploratory_analysis_day3()
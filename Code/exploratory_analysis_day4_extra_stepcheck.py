# exploratory_analysis_day4_extra_stepcheck
# Written by Will
# Updated 3/15/2026

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



def exploratory_analysis_day4_extra_stepcheck():

    # Extra numerical check:
    # Compares forward SEIR peak predictions using smaller Euler step sizes to see whether the predicted peak is sensitive to the choice of h.

    # Load Data Release #2
    data = pd.read_csv(
        "../Data/mystery_virus_daily_active_counts_RELEASE#2.csv",
        parse_dates=["date"],
        header=0,
        index_col=None
    )

    t_obs = data["day"].to_numpy(dtype=float)
    I_obs = data["active reported daily cases"].to_numpy(dtype=float)

    # Initial conditions
    N = 10000.0
    I0 = float(I_obs[0])
    E0 = 0.0
    R0 = 0.0
    S0 = N - E0 - I0 - R0

    # Fit SEIR parameters using Release #2
    best = grid_search_fit(
        timepoints=t_obs,
        I_obs=I_obs,
        N=N,
        S0=S0, E0=E0, I0=I0, R0=R0,
        beta_range=(0.3, 0.7),
        sigma_range=(0.1, 0.3),
        gamma_range=(0.05, 0.25),
        resolution=15
    )

    beta_best = best["beta"]
    sigma_best = best["sigma"]
    gamma_best = best["gamma"]

    print("Best-fit parameters from Release #2:")
    print(f"  beta  = {beta_best:.3f}")
    print(f"  sigma = {sigma_best:.3f}")
    print(f"  gamma = {gamma_best:.3f}")
    print(f"  SSE   = {best['sse']:.2f}")

    # Step-size sensitivity check
    step_sizes = [1.0, 0.5, 0.25]
    results = []

    plt.figure(figsize=(10, 6))

    for h in step_sizes:
        # Build time array out to day 300 using the given step size
        t_future = np.arange(1.0, 300.0 + h, h, dtype=float)

        # Run forward prediction
        S, E, I, R = euler_seir(
            beta_best, sigma_best, gamma_best,
            S0, E0, I0, R0,
            t_future, N, h=h
        )

        # Find the peak in I(t)
        peak_idx = int(np.argmax(I))
        peak_day = t_future[peak_idx]
        peak_I = I[peak_idx]

        results.append((h, peak_day, peak_I))

        # Plot the forward infection curve for this step size
        plt.plot(t_future, I, linewidth=2, label=f"h = {h}")

    # Print comparison table
    print("\nEuler step-size sensitivity check (forward prediction to day 300):")
    for h, peak_day, peak_I in results:
        print(f"  h = {h:<4} -> peak day = {peak_day:.2f}, peak active infections = {peak_I:.2f}")

    # Compare smaller step sizes to the h = 1.0 baseline
    base_h, base_peak_day, base_peak_I = results[0]

    print("\nDifferences relative to h = 1.0:")
    for h, peak_day, peak_I in results[1:]:
        print(
            f"  h = {h:<4} -> "
            f"day difference = {peak_day - base_peak_day:.2f}, "
            f"peak-I difference = {peak_I - base_peak_I:.2f}"
        )

    # Plot formatting
    plt.xlabel("Day")
    plt.ylabel("Active infections (model I)")
    plt.title("Euler step-size sensitivity check for SEIR forward prediction")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    exploratory_analysis_day4_extra_stepcheck()

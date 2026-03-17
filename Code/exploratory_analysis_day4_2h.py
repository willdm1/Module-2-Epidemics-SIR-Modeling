# exploratory_analysis_day4_2h
# Written by Will and Reagan
# Updated 3/14/2026
# This script applies the best-fit SEIR model to Virginia Tech and compares baseline outbreak dynamics under multiple intervention strategies.

#Strategy 1/3: Mask mandate with reduce Beta by 40% starting Day 70
#Strategy 2/3: Vaccine Campaign will move 2000*0.9 from S to R as a single event on day 70
#Strategy 3/3: Vaccine Roll out qill move 1000*0.9 from S to R each day strting on day 70

#The differences is strategies 2 and 3 explore strategies within an intervention (vaccination) going beyond simply picking
# one of each type of strategy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Solves the SEIR system numerically using Euler's method for specified parameters, initial conditions, and timepoints.
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
    # We match the Lecture 3 format for the error function that grid search minimizes
    I_obs = np.asarray(I_obs, dtype=float)
    I_model = np.asarray(I_model, dtype=float)
    return np.sum((I_obs - I_model) ** 2)

# Searches across beta, sigma, and gamma values to find the combination that minimizes SSE between model I(t) and observed data
def grid_search_fit(timepoints, I_obs, N, S0, E0, I0, R0,
                    beta_range=(0.3, 0.7),
                    sigma_range=(0.1, 0.3),
                    gamma_range=(0.05, 0.25),
                    resolution=15,
                    h=1.0):
    
    # This is our 3-parameter grid search over (beta, sigma, gamma), minimizing SSE
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


# helper for time-varying parameters (for interventions)

def euler_seir_timevarying(beta_t, sigma_t, gamma_t,
                           S0, E0, I0, R0, timepoints, N, h=1.0):

    # identical to euler_seir, but β, σ, γ can change each day
    n = len(timepoints)
    S = np.zeros(n, dtype=float)
    E = np.zeros(n, dtype=float)
    I = np.zeros(n, dtype=float)
    R = np.zeros(n, dtype=float)

    S[0], E[0], I[0], R[0] = S0, E0, I0, R0

    for i in range(n - 1):
        b = beta_t[i]   # time‑dependent β
        s = sigma_t[i]  # time‑dependent σ (unchanged in our interventions)
        g = gamma_t[i]  # time‑dependent γ (unchanged in our interventions)

        dS = -b * S[i] * I[i] / N
        dE = b * S[i] * I[i] / N - s * E[i]
        dI = s * E[i] - g * I[i]
        dR = g * I[i]

        S[i + 1] = max(S[i] + dS * h, 0.0)
        E[i + 1] = max(E[i] + dE * h, 0.0)
        I[i + 1] = max(I[i] + dI * h, 0.0)
        R[i + 1] = max(R[i] + dR * h, 0.0)

    return S, E, I, R

def make_masking_params(t_full, beta_base, sigma_base, gamma_base, day_start=70):
    # create β(t) that drops by 40% at day 70 (mask mandate)
    beta_t = np.full_like(t_full, beta_base, dtype=float)
    sigma_t = np.full_like(t_full, sigma_base, dtype=float)
    gamma_t = np.full_like(t_full, gamma_base, dtype=float)

    mask_idx = t_full >= day_start
    beta_t[mask_idx] = 0.6 * beta_base  # 40% reduction per slides

    return beta_t, sigma_t, gamma_t

def apply_vaccination_events(S, R, t_full, events):
    """
    NEW: Apply instantaneous S → R transfers on specific days.
    events: list of (day, n_vaccinated, efficacy)
    """
    S = S.copy()
    R = R.copy()
    for day, n_vax, eff in events:
        idx = np.where(t_full == day)[0]
        if len(idx) == 0:
            continue
        i = idx[0]
        effective = n_vax * eff      # number who actually gain immunity
        move = min(effective, S[i])  # cannot vaccinate more than S
        S[i] -= move
        R[i] += move
    return S, R

# --- VT baseline and interventions ---
def vt_interventions():

    # Initial conditions
    N = 38294.0
    I0 = 1.0
    R0 = 0.0
    E0 = E0_UVA
    S0 = N - E0 - I0 - R0

    t_full = np.arange(1, 121)

    # 1. BASELINE
    Sb, Eb, Ib_base, Rb = euler_seir(
        beta_best, sigma_best, gamma_best,
        S0, E0, I0, R0,
        t_full, N, h=1.0
    )

    
    # 2. MASK MANDATE (β reduced 40% at day 70)
    beta_mask_t, sigma_mask_t, gamma_mask_t = make_masking_params(
        t_full, beta_best, sigma_best, gamma_best, day_start=70
    )

    S_mask, E_mask, I_mask, R_mask = euler_seir_timevarying(
        beta_mask_t, sigma_mask_t, gamma_mask_t,
        S0, E0, I0, R0,
        t_full, N, h=1.0
    )

    
    # 3. SINGLE VACCINE CAMPAIGN (DAY 70)

    # Run SEIR up to day 70
    t1 = np.arange(1, 71)
    S1, E1, I1, R1 = euler_seir(
        beta_best, sigma_best, gamma_best,
        S0, E0, I0, R0,
        t1, N, h=1.0
    )

    # Apply vaccination at day 70
    S70 = S1[-1]
    R70 = R1[-1]
    vaccinated = 2000 * 0.9
    S70 -= vaccinated
    R70 += vaccinated

    # Continue SEIR from day 70 → 120
    t2 = np.arange(70, 121)
    S2, E2, I2, R2 = euler_seir(
        beta_best, sigma_best, gamma_best,
        S70, E1[-1], I1[-1], R70,
        t2, N, h=1.0
    )

    # Stitch
    Ib_vax_single = np.concatenate([I1, I2[1:]])

    
    # 4. VACCINE ROLLOUT (70, 80, 90)
    
    # Run SEIR up to day 70
    S1r, E1r, I1r, R1r = S1, E1, I1, R1

    # Vaccinate at day 70
    S70r = S1r[-1] - 1000*0.9
    R70r = R1r[-1] + 1000*0.9

    # Run 70 → 80
    t2r = np.arange(70, 81)
    S2r, E2r, I2r, R2r = euler_seir(
        beta_best, sigma_best, gamma_best,
        S70r, E1r[-1], I1r[-1], R70r,
        t2r, N, h=1.0
    )

    # Vaccinate at day 80
    S80 = S2r[-1] - 1000*0.9
    R80 = R2r[-1] + 1000*0.9

    # Run 80 → 90
    t3r = np.arange(80, 91)
    S3r, E3r, I3r, R3r = euler_seir(
        beta_best, sigma_best, gamma_best,
        S80, E2r[-1], I2r[-1], R80,
        t3r, N, h=1.0
    )

    # Vaccinate at day 90
    S90 = S3r[-1] - 1000*0.9
    R90 = R3r[-1] + 1000*0.9

    # Run 90 → 120
    t4r = np.arange(90, 121)
    S4r, E4r, I4r, R4r = euler_seir(
        beta_best, sigma_best, gamma_best,
        S90, E3r[-1], I3r[-1], R90,
        t4r, N, h=1.0
    )

    # Stitch rollout
    Ib_vax_roll = np.concatenate([I1r, I2r[1:], I3r[1:], I4r[1:]])

    # 5. EXTRA Combined Intervention (Masking + Vaccine Rollout)

    # Build masked parameter arrays once
    beta_combo_t, sigma_combo_t, gamma_combo_t = make_masking_params(
        t_full, beta_best, sigma_best, gamma_best, day_start=70
    )

    # Segment 1: day 1 -> 70
    t1c = np.arange(1, 71)
    beta1 = beta_combo_t[:len(t1c)]
    sigma1 = sigma_combo_t[:len(t1c)]
    gamma1 = gamma_combo_t[:len(t1c)]

    S1c, E1c, I1c, R1c = euler_seir_timevarying(
        beta1, sigma1, gamma1,
        S0, E0, I0, R0,
        t1c, N, h=1.0
    )

    # Vaccinate at day 70
    S70c = S1c[-1] - 1000 * 0.9
    R70c = R1c[-1] + 1000 * 0.9

    # Segment 2: day 70 -> 80
    t2c = np.arange(70, 81)
    start2 = np.where(t_full == 70)[0][0]
    end2 = start2 + len(t2c)

    beta2 = beta_combo_t[start2:end2]
    sigma2 = sigma_combo_t[start2:end2]
    gamma2 = gamma_combo_t[start2:end2]

    S2c, E2c, I2c, R2c = euler_seir_timevarying(
        beta2, sigma2, gamma2,
        S70c, E1c[-1], I1c[-1], R70c,
        t2c, N, h=1.0
    )

    # Vaccinate at day 80
    S80c = S2c[-1] - 1000 * 0.9
    R80c = R2c[-1] + 1000 * 0.9

    # Segment 3: day 80 -> 90
    t3c = np.arange(80, 91)
    start3 = np.where(t_full == 80)[0][0]
    end3 = start3 + len(t3c)

    beta3 = beta_combo_t[start3:end3]
    sigma3 = sigma_combo_t[start3:end3]
    gamma3 = gamma_combo_t[start3:end3]

    S3c, E3c, I3c, R3c = euler_seir_timevarying(
        beta3, sigma3, gamma3,
        S80c, E2c[-1], I2c[-1], R80c,
        t3c, N, h=1.0
    )

    # Vaccinate at day 90
    S90c = S3c[-1] - 1000 * 0.9
    R90c = R3c[-1] + 1000 * 0.9

    # Segment 4: day 90 -> 120
    t4c = np.arange(90, 121)
    start4 = np.where(t_full == 90)[0][0]
    end4 = start4 + len(t4c)

    beta4 = beta_combo_t[start4:end4]
    sigma4 = sigma_combo_t[start4:end4]
    gamma4 = gamma_combo_t[start4:end4]

    S4c, E4c, I4c, R4c = euler_seir_timevarying(
        beta4, sigma4, gamma4,
        S90c, E3c[-1], I3c[-1], R90c,
        t4c, N, h=1.0
    )

    # Stitch combo trajectory
    Ib_combo = np.concatenate([I1c, I2c[1:], I3c[1:], I4c[1:]])

    return t_full, Ib_base, I_mask, Ib_vax_single, Ib_vax_roll, Ib_combo

# PLOTTING FUNCTIONS

def plot_baseline_vs_mask(t, base, mask):
    plt.figure(figsize=(10,6))
    plt.plot(t, base, label="Baseline")
    plt.plot(t, mask, label="Mask mandate")
    plt.title("Baseline VT Data vs. Mask Mandate Protocol")
    plt.xlabel("Day")
    plt.ylabel("Active infections")
    plt.legend(); plt.tight_layout(); plt.show()


def plot_baseline_vs_single(t, base, single):
    plt.figure(figsize=(10,6))
    plt.plot(t, base, label="Baseline")
    plt.plot(t, single, label="Single vaccine")
    plt.title("Baseline VT Data vs. Single Vaccine Protocol")
    plt.xlabel("Day")
    plt.ylabel("Active infections")
    plt.legend(); plt.tight_layout(); plt.show()


def plot_baseline_vs_rollout(t, base, roll):
    plt.figure(figsize=(10,6))
    plt.plot(t, base, label="Baseline")
    plt.plot(t, roll, label="Rollout")
    plt.title("Baseline VT Data vs. Rollout Vaccine Protocol")
    plt.xlabel("Day")
    plt.ylabel("Active infections")
    plt.legend(); plt.tight_layout(); plt.show()

def plot_baseline_vs_combo(t, base, combo):
    plt.figure(figsize=(10,6))
    plt.plot(t, base, label="Baseline")
    plt.plot(t, combo, label="Combo (Mask + Rollout)")
    plt.title("Baseline VT Data vs. Combo Intervention Protocol")
    plt.xlabel("Day")
    plt.ylabel("Active infections")
    plt.legend(); plt.tight_layout(); plt.show()

def plot_all_interventions(t, base, mask, single, roll, combo):
    plt.figure(figsize=(10,6))
    plt.plot(t, base, label="Baseline")
    plt.plot(t, mask, label="Mask")
    plt.plot(t, single, label="Single vax")
    plt.plot(t, roll, label="Rollout")
    plt.plot(t, combo, label="Combo")
    plt.title("Baseline VT Data vs. Each Chosen Intervention")
    plt.xlabel("Day")
    plt.ylabel("Active infections")
    plt.legend(); plt.tight_layout(); plt.show()

# Main function to run the VT intervention comparisons and plotting
def exploratory_analysis_day4_2h():
    global beta_best, sigma_best, gamma_best, E0_UVA

    # For 2h, VT interventions should use the best-fit parameters from Release #2,
    # not the parameters re-fit to Release #3.

    # Load Release #2 and fit parameters there
    data2 = pd.read_csv(
        "../Data/mystery_virus_daily_active_counts_RELEASE#2.csv",
        parse_dates=["date"],
        header=0,
        index_col=None
    )

    t2 = data2["day"].to_numpy(dtype=float)
    I_obs2 = data2["active reported daily cases"].to_numpy(dtype=float)

    N2 = 10000.0
    I0_2 = float(I_obs2[0])
    E0_2 = 0.0
    R0_2 = 0.0
    S0_2 = N2 - E0_2 - I0_2 - R0_2
    h = 1.0

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

    # Extract best-fit parameters from Release #2
    beta_best = best2["beta"]
    sigma_best = best2["sigma"]
    gamma_best = best2["gamma"]

    # Lecture 4 says E0(VT) = E0(UVA)
    E0_UVA = E0_2

    # Now we have our best-fit parameters from Release #2, and we can apply the SEIR model with interventions to VT data for comparison.
    t, base, mask, single, roll, combo = vt_interventions()

    def peak_metrics(t, I):
        idx = int(np.argmax(I))
        return t[idx], I[idx]

    # Peak metrics
    d_base, Ipk_base = peak_metrics(t, base)
    d_mask, Ipk_mask = peak_metrics(t, mask)
    d_single, Ipk_single = peak_metrics(t, single)
    d_roll, Ipk_roll = peak_metrics(t, roll)
    d_combo, Ipk_combo = peak_metrics(t, combo)

    # Print peak day and peak I for each scenario, and the reduction in peak I compared to baseline
    print("\nVT intervention comparison: peak metrics")
    print(f"Baseline: peak day {d_base:.0f}, peak I {Ipk_base:.1f}")
    print(f"Mask:     peak day {d_mask:.0f}, peak I {Ipk_mask:.1f}")
    print(f"Vax once: peak day {d_single:.0f}, peak I {Ipk_single:.1f}")
    print(f"Vax roll: peak day {d_roll:.0f}, peak I {Ipk_roll:.1f}")
    print(f"Combo:    peak day {d_combo:.0f}, peak I {Ipk_combo:.1f}")

    # Total active infections over days 70–120 (simple area-under-I curve proxy)
    # (This is not explicitly named in slides, but it is a direct way to compare burden over 70–120.)
    mask_70_120 = (t >= 70) & (t <= 120)
    A_base = np.sum(base[mask_70_120])
    A_mask = np.sum(mask[mask_70_120])
    A_single = np.sum(single[mask_70_120])
    A_roll = np.sum(roll[mask_70_120])
    A_combo = np.sum(combo[mask_70_120])

    # Print total active infections over days 70–120 for each scenario, and the reduction compared to baseline
    print("\nSum of active infections over days 70–120 (burden proxy)")
    print(f"Baseline: {A_base:.1f}")
    print(f"Mask:     {A_mask:.1f}  (reduction {A_base - A_mask:.1f})")
    print(f"Vax once: {A_single:.1f} (reduction {A_base - A_single:.1f})")
    print(f"Vax roll: {A_roll:.1f} (reduction {A_base - A_roll:.1f})")
    print(f"Combo:    {A_combo:.1f} (reduction {A_base - A_combo:.1f})")

    # Plotting comparisons  
    plot_baseline_vs_mask(t, base, mask)
    plot_baseline_vs_single(t, base, single)
    plot_baseline_vs_rollout(t, base, roll)
    plot_baseline_vs_combo(t, base, combo)
    plot_all_interventions(t, base, mask, single, roll, combo)


if __name__ == "__main__":
    exploratory_analysis_day4_2h()
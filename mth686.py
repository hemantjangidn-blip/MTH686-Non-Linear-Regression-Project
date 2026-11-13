"""
========================================================
MTH686 PROJECT: NON-LINEAR REGRESSION ANALYSIS
Indian Institute of Technology, Kanpur
Department of Mathematics and Statistics

Submitted by:
Name       : Hemant Jangid
Roll No.   : 220450
Course     : MTH686

Dataset Information:
Dataset Name   : data-77.dat
Dataset Format : Two columns (t, y(t))
Sample Size    : 100 observations
Data Range     : t = 0.01 to 1.00
Dataset Source : Dataset provided by the course instructor
                 for the MTH686 project.

========================================================
"""
from pathlib import Path
import numpy as np
import scipy.stats as st
from scipy.optimize import least_squares
from statsmodels.stats.diagnostic import lilliefors, acorr_ljungbox
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DATA_FILE = HERE / "data-77.dat"
RESULT_FILE = HERE / "mth686.txt"

if not DATA_FILE.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")
data = np.loadtxt(DATA_FILE)
if data.ndim != 2 or data.shape[1] != 2:
    raise ValueError("Expected a two-column dataset: t and y(t).")
t, y = data[:, 0], data[:, 1]
n = len(t)
if n <= 5:
    raise ValueError("At least six observations are required for these models.")

# Model 1: p = (a0, a1, b1, a2, b2)
def m1(p, x):
    return p[0] + p[1] * np.exp(p[2] * x) + p[3] * np.exp(p[4] * x)

def j1(p, x):
    a0, a1, b1, a2, b2 = p
    return np.column_stack((
        np.ones_like(x), np.exp(b1*x), a1*x*np.exp(b1*x),
        np.exp(b2*x), a2*x*np.exp(b2*x)
    ))

# Profile grid: for fixed b1,b2, estimate a0,a1,a2 by linear least squares.
grid = np.linspace(-30.0, 30.0, 61)
best = None
for i, b1 in enumerate(grid):
    for b2 in grid[i+1:]:
        X = np.column_stack((np.ones_like(t), np.exp(b1*t), np.exp(b2*t)))
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        rss = float(np.sum((y - X @ coef)**2))
        if best is None or rss < best[0]:
            best = (rss, b1, b2, coef)
_, b1_start, b2_start, coef_start = best
p0 = np.array([coef_start[0], coef_start[1], b1_start,
               coef_start[2], b2_start], dtype=float)

fit1 = least_squares(lambda p: m1(p, t)-y, p0,
                     jac=lambda p: j1(p, t), method="lm",
                     max_nfev=5000, xtol=1e-13, ftol=1e-13, gtol=1e-13)
# Deterministic multistart refinement around the profiled starting point.
rng = np.random.default_rng(0)
for _ in range(200):
    q = p0 * (1 + 0.5*rng.standard_normal(5)) + 0.1*rng.standard_normal(5)
    try:
        candidate = least_squares(lambda p: m1(p, t)-y, q,
                                  jac=lambda p: j1(p, t), method="lm",
                                  max_nfev=3000, xtol=1e-12,
                                  ftol=1e-12, gtol=1e-12)
        if candidate.cost < fit1.cost:
            fit1 = candidate
    except (ValueError, FloatingPointError):
        pass
p1 = fit1.x

# Model 2: scale invariance allows beta0=1; p = (a0,a1,b1).
def m2(p, x):
    return (p[0] + p[1]*x) / (1.0 + p[2]*x)

def j2(p, x):
    a0, a1, b1 = p
    den = 1.0 + b1*x
    return np.column_stack((1.0/den, x/den,
                            -(a0+a1*x)*x/(den**2)))

# Linearized initial estimate: y = a0 + a1*t - b1*t*y.
X_init = np.column_stack((np.ones_like(t), t, -t*y))
p2_start, *_ = np.linalg.lstsq(X_init, y, rcond=None)
fit2 = least_squares(lambda p: m2(p, t)-y, p2_start,
                     jac=lambda p: j2(p, t), method="lm",
                     max_nfev=5000, xtol=1e-13, ftol=1e-13, gtol=1e-13)
p2 = fit2.x

# Model 3: polynomial linear least squares, p=(b0,b1,b2,b3,b4).
X3 = np.vander(t, 5, increasing=True)
p3, *_ = np.linalg.lstsq(X3, y, rcond=None)

models = {
    "Model 1": (p1, m1(p1, t), j1(p1, t), ["a0", "a1", "b1", "a2", "b2"]),
    "Model 2 (beta0=1)": (p2, m2(p2, t), j2(p2, t), ["a0", "a1", "b1"]),
    "Model 3": (p3, X3 @ p3, X3, ["b0", "b1", "b2", "b3", "b4"]),
}

with RESULT_FILE.open("w", encoding="utf-8") as out:
    def write(*items):
        line = " ".join(str(x) for x in items)
        print(line)
        out.write(line + "\n")

    write(f"n={n}; t range=[{t.min():.2f}, {t.max():.2f}]")
    write(f"Model 1 profile grid start: b1={b1_start:.3f}, b2={b2_start:.3f}, RSS={best[0]:.9f}")
    write(f"Model 2 linearized start: {p2_start}")

    summaries = {}
    for name, (theta, fitted, J, names) in models.items():
        residuals = y - fitted
        k = len(theta)
        rss = float(residuals @ residuals)
        df = n-k
        s2 = rss/df
        covariance = s2 * np.linalg.inv(J.T @ J)
        se = np.sqrt(np.diag(covariance))
        tcrit = st.t.ppf(0.975, df)
        ci = np.column_stack((theta-tcrit*se, theta+tcrit*se))
        # Full Gaussian AIC/BIC, counting k mean parameters plus sigma^2.
        log_part = n * (np.log(2*np.pi) + 1.0 + np.log(rss/n))
        aic = log_part + 2*(k+1)
        bic = log_part + (k+1)*np.log(n)
        # Reduced scores omit constants common to all three models; rankings match.
        aic_reduced = n*np.log(rss/n) + 2*k
        bic_reduced = n*np.log(rss/n) + k*np.log(n)
        singular_values = np.linalg.svd(J, compute_uv=False)
        condition_number = singular_values[0]/singular_values[-1]
        summaries[name] = dict(theta=theta, fitted=fitted, residuals=residuals,
                               rss=rss, s2=s2, aic=aic, bic=bic,
                               aic_reduced=aic_reduced, bic_reduced=bic_reduced,
                               se=se, ci=ci, J=J,
                               condition_number=condition_number)
        write(f"\n=== {name} ===")
        write(f"RSS={rss:.9f}; sigma2_hat=RSS/(n-k)={s2:.9g}; sigma_hat={np.sqrt(s2):.7f}")
        write(f"Full Gaussian AIC={aic:.5f}; full Gaussian BIC={bic:.5f}")
        write(f"Reduced AIC={aic_reduced:.5f}; reduced BIC={bic_reduced:.5f} (same ranking)")
        for nm, value, std_err, interval in zip(names, theta, se, ci):
            write(f"{nm}: estimate={value:.8f}; SE={std_err:.8f}; 95% t-Wald CI=({interval[0]:.8f}, {interval[1]:.8f})")
        write(f"Jacobian condition number={condition_number:.3f}")

        # Residual diagnostics. Lilliefors corrects the KS calibration for
        # estimated normal mean and variance; Shapiro-Wilk is also reported.
        sw = st.shapiro(residuals)
        lf_stat, lf_p = lilliefors(residuals, dist="norm")
        dw = np.sum(np.diff(residuals)**2)/rss
        lb = acorr_ljungbox(residuals, lags=[5], return_df=True)
        write(f"Shapiro-Wilk: W={sw.statistic:.5f}, p={sw.pvalue:.5f}")
        write(f"Lilliefors: D={lf_stat:.5f}, p={lf_p:.5f}")
        write(f"Durbin-Watson={dw:.5f}; Ljung-Box lag-5 p={lb['lb_pvalue'].iloc[0]:.5f}")

    # Model ranking and AIC deltas.
    ordered = sorted(summaries.items(), key=lambda kv: kv[1]["aic"])
    best_aic = ordered[0][1]["aic"]
    write("\n=== MODEL COMPARISON ===")
    for name, s in ordered:
        write(f"{name}: delta_AIC={s['aic']-best_aic:.5f}; delta_BIC={s['bic']-min(x['bic'] for x in summaries.values()):.5f}")

# Plot fitted curves.
tt = np.linspace(t.min(), t.max(), 500)
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(t, y, s=18, color="black", label="Observed data", zorder=3)
ax.plot(tt, m1(p1, tt), label="Model 1")
ax.plot(tt, m2(p2, tt), "--", label="Model 2")
ax.plot(tt, np.vander(tt, 5, increasing=True) @ p3, ":", label="Model 3")
ax.set_xlabel("t")
ax.set_ylabel("y")
ax.set_title("Observed data and fitted regression curves")
ax.legend()
fig.tight_layout()
fig.savefig(HERE / "fitted_curves.png", dpi=220)
plt.close(fig)

# Residual and normal Q-Q plots.
fig, axes = plt.subplots(2, 3, figsize=(12, 7))
for j, (name, (_, _, _, _)) in enumerate(models.items()):
    residuals = summaries[name]["residuals"]
    axes[0, j].scatter(t, residuals, s=14)
    axes[0, j].axhline(0, color="black", linewidth=0.8)
    axes[0, j].set_title(f"{name}: residuals")
    axes[0, j].set_xlabel("t")
    axes[0, j].set_ylabel("Residual")
    st.probplot(residuals, dist="norm", plot=axes[1, j])
    axes[1, j].set_title(f"{name}: normal Q-Q")
fig.tight_layout()
fig.savefig(HERE / "residual_diagnostics.png", dpi=220)
plt.close(fig)

print(f"\nWrote {RESULT_FILE}")
print(f"Wrote {HERE / 'fitted_curves.png'}")
print(f"Wrote {HERE / 'residual_diagnostics.png'}")

"""
Projet 5 - Processus gaussiens pour l'optimisation bayésienne.

Implémentation NumPy des éléments demandés dans le projet:
GP + Cholesky, log-vraisemblance marginale, gradients, L-BFGS,
EI/PI/UCB, Rosenbrock 10D et comparaison avec Random Search.
"""

import argparse
import csv
import math
import os
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union

try:
    import numpy as np
except Exception as exc:
    print("Erreur : NumPy n'est pas installé correctement pour cette version de Python.")
    print("Conseil : utilise un environnement virtuel propre avec Python 3.10, 3.11 ou 3.12,")
    print("puis installe les dépendances avec : python -m pip install numpy matplotlib")
    raise SystemExit(1) from exc

try:
    import matplotlib.pyplot as plt
except Exception as exc:
    print("Erreur : Matplotlib n'est pas installé correctement.")
    print("Installe-le avec : python -m pip install matplotlib")
    raise SystemExit(1) from exc


JITTER = 1e-8
LOG_MIN = np.log(1e-6)
LOG_MAX = np.log(1e3)


# ---------------------------------------------------------------------------
# Fonction test
# ---------------------------------------------------------------------------


def rosenbrock(x: np.ndarray) -> float:
    """Fonction de Rosenbrock en dimension quelconque."""
    x = np.asarray(x, dtype=float)
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2))


# ---------------------------------------------------------------------------
# Noyau et outils GP
# ---------------------------------------------------------------------------


def pairwise_sqdist(X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
    """Matrice des distances euclidiennes carrées entre deux ensembles de points."""
    X1 = np.atleast_2d(X1).astype(float)
    X2 = np.atleast_2d(X2).astype(float)
    return np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2.0 * X1 @ X2.T



def rbf_kernel(X1: np.ndarray, X2: np.ndarray, log_params: np.ndarray) -> np.ndarray:
    """Noyau exponentiel quadratique : sigma_f^2 exp(-||x-x'||^2 / 2l^2)."""
    length_scale = float(np.exp(log_params[0]))
    signal_var = float(np.exp(log_params[1]))
    sqdist = pairwise_sqdist(X1, X2)
    return signal_var * np.exp(-0.5 * sqdist / (length_scale**2))



def cholesky_solve(L: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Résout (L L^T) X = B sans inverser la matrice."""
    return np.linalg.solve(L.T, np.linalg.solve(L, B))


# ---------------------------------------------------------------------------
# Log-vraisemblance marginale et gradients
# ---------------------------------------------------------------------------


def nll_and_grad(log_params: np.ndarray, X: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray]:
    """NLL du GP et gradient analytique par rapport aux log-hyperparamètres."""
    n = X.shape[0]
    log_l, log_sf, log_sn = log_params
    length_scale = float(np.exp(log_l))
    signal_var = float(np.exp(log_sf))
    noise_var = float(np.exp(log_sn))

    sqdist = pairwise_sqdist(X, X)
    K = signal_var * np.exp(-0.5 * sqdist / (length_scale**2))
    Ky = K + (noise_var + JITTER) * np.eye(n)

    try:
        L = np.linalg.cholesky(Ky)
    except np.linalg.LinAlgError:
        return 1e25, np.zeros_like(log_params)

    alpha = cholesky_solve(L, y)
    nll = 0.5 * float(y.T @ alpha) + float(np.sum(np.log(np.diag(L)))) + 0.5 * n * np.log(2.0 * np.pi)

    K_inv = cholesky_solve(L, np.eye(n))
    common = np.outer(alpha, alpha) - K_inv

    # Dérivées de Ky par rapport aux paramètres logarithmiques.
    dK_log_l = K * (sqdist / (length_scale**2))
    dK_log_sf = K
    dK_log_sn = noise_var * np.eye(n)

    grad_lml = np.array([
        0.5 * np.trace(common @ dK_log_l),
        0.5 * np.trace(common @ dK_log_sf),
        0.5 * np.trace(common @ dK_log_sn),
    ])
    return nll, -grad_lml


# ---------------------------------------------------------------------------
# L-BFGS 
# ---------------------------------------------------------------------------


def lbfgs_minimize(func_grad: Callable[[np.ndarray], Tuple[float, np.ndarray]], x0: np.ndarray,
                   max_iter: int = 25, memory: int = 7, tol: float = 1e-5) -> Tuple[np.ndarray, float]:
    """Minimise une fonction avec L-BFGS et recherche linéaire d'Armijo."""
    x = x0.astype(float).copy()
    value, grad = func_grad(x)
    s_hist: List[np.ndarray] = []
    y_hist: List[np.ndarray] = []
    rho_hist: List[float] = []

    for _ in range(max_iter):
        if np.linalg.norm(grad) < tol:
            break

        q = grad.copy()
        alphas: List[float] = []

        for s, y, rho in reversed(list(zip(s_hist, y_hist, rho_hist))):
            alpha = rho * float(np.dot(s, q))
            alphas.append(alpha)
            q -= alpha * y

        if y_hist:
            s_last, y_last = s_hist[-1], y_hist[-1]
            gamma = float(np.dot(s_last, y_last) / max(np.dot(y_last, y_last), 1e-12))
        else:
            gamma = 1.0

        r = gamma * q
        for s, y, rho, alpha in zip(s_hist, y_hist, rho_hist, reversed(alphas)):
            beta = rho * float(np.dot(y, r))
            r += s * (alpha - beta)

        direction = -r
        if np.dot(direction, grad) >= 0:
            direction = -grad

        step = 1.0
        while step > 1e-8:
            x_new = np.clip(x + step * direction, LOG_MIN, LOG_MAX)
            value_new, grad_new = func_grad(x_new)
            if value_new <= value + 1e-4 * step * float(np.dot(grad, direction)):
                break
            step *= 0.5

        s = x_new - x
        y_vec = grad_new - grad
        sy = float(np.dot(s, y_vec))

        if sy > 1e-12:
            if len(s_hist) == memory:
                s_hist.pop(0)
                y_hist.pop(0)
                rho_hist.pop(0)
            s_hist.append(s)
            y_hist.append(y_vec)
            rho_hist.append(1.0 / sy)

        x, value, grad = x_new, value_new, grad_new

    return x, float(value)



def optimize_hyperparameters(X: np.ndarray, y: np.ndarray, start: Optional[np.ndarray] = None,
                             max_iter: int = 15) -> np.ndarray:
    """Optimise les hyperparamètres du noyau avec L-BFGS."""
    if start is None:
        start = np.log(np.array([1.0, 1.0, 1e-4], dtype=float))
    params, _ = lbfgs_minimize(lambda p: nll_and_grad(p, X, y), start, max_iter=max_iter)
    return params


# ---------------------------------------------------------------------------
# Modèle GP
# ---------------------------------------------------------------------------


@dataclass
class GPModel:
    X: np.ndarray
    y_scaled: np.ndarray
    y_mean: float
    y_std: float
    log_params: np.ndarray
    L: np.ndarray
    alpha: np.ndarray



def fit_gp(X: np.ndarray, y: np.ndarray, tune: bool = True,
           start_params: Optional[np.ndarray] = None, lbfgs_iter: int = 10) -> GPModel:
    """Ajuste le GP sur les données courantes."""
    y_mean = float(np.mean(y))
    y_std = float(np.std(y)) or 1.0
    if y_std < 1e-12:
        y_std = 1.0

    y_scaled = (y - y_mean) / y_std
    if start_params is None:
        start_params = np.log(np.array([1.0, 1.0, 1e-4], dtype=float))

    log_params = optimize_hyperparameters(X, y_scaled, start=start_params, max_iter=lbfgs_iter) if tune and X.shape[0] >= 8 else start_params.copy()

    noise_var = float(np.exp(log_params[2]))
    Ky = rbf_kernel(X, X, log_params) + (noise_var + JITTER) * np.eye(X.shape[0])
    L = np.linalg.cholesky(Ky)
    alpha = cholesky_solve(L, y_scaled)
    return GPModel(X, y_scaled, y_mean, y_std, log_params, L, alpha)



def gp_predict(model: GPModel, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Retourne la moyenne et la variance a posteriori."""
    X_test = np.atleast_2d(X_test).astype(float)
    K_star = rbf_kernel(model.X, X_test, model.log_params)
    mu_scaled = K_star.T @ model.alpha

    v = np.linalg.solve(model.L, K_star)
    signal_var = float(np.exp(model.log_params[1]))
    var_scaled = np.maximum(signal_var - np.sum(v * v, axis=0), 1e-12)

    mu = model.y_mean + model.y_std * mu_scaled
    var = (model.y_std**2) * var_scaled
    return mu, var


# ---------------------------------------------------------------------------
# Fonctions d'acquisition
# ---------------------------------------------------------------------------


def normal_pdf(z: np.ndarray) -> np.ndarray:
    return (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * z * z)



def normal_cdf(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(math.erf)(z / np.sqrt(2.0)))



def expected_improvement(mu: np.ndarray, var: np.ndarray, best_y: float, xi: float = 0.01) -> np.ndarray:
    sigma = np.maximum(np.sqrt(var), 1e-12)
    gain = best_y - mu - xi
    z = gain / sigma
    return gain * normal_cdf(z) + sigma * normal_pdf(z)



def probability_improvement(mu: np.ndarray, var: np.ndarray, best_y: float, xi: float = 0.01) -> np.ndarray:
    sigma = np.maximum(np.sqrt(var), 1e-12)
    return normal_cdf((best_y - mu - xi) / sigma)



def upper_confidence_bound(mu: np.ndarray, var: np.ndarray, kappa: float = 2.0) -> np.ndarray:
    return mu + kappa * np.sqrt(np.maximum(var, 1e-12))



def lower_confidence_bound(mu: np.ndarray, var: np.ndarray, kappa: float = 2.0) -> np.ndarray:
    return mu - kappa * np.sqrt(np.maximum(var, 1e-12))


# ---------------------------------------------------------------------------
# Optimisation bayésienne
# ---------------------------------------------------------------------------


def cumulative_best(values: Union[np.ndarray, List[float]]) -> List[float]:
    best = float("inf")
    out: List[float] = []
    for value in values:
        best = min(best, float(value))
        out.append(best)
    return out


@dataclass
class BOResult:
    X: np.ndarray
    y: np.ndarray
    best_values: List[float]
    final_model: GPModel



def bayesian_optimization(objective: Callable[[np.ndarray], float], bounds: List[Tuple[float, float]],
                          n_initial: int = 10, n_iter: int = 200, n_candidates: int = 50,
                          acquisition: str = "ei",
                          tune: bool = True, tune_every: int = 25, lbfgs_iter: int = 5) -> BOResult:
    dim = len(bounds)
    lower = np.array([b[0] for b in bounds], dtype=float)
    upper = np.array([b[1] for b in bounds], dtype=float)

    X = lower + (upper - lower) * np.random.rand(n_initial, dim)
    y = np.array([objective(x) for x in X], dtype=float)
    best_values = cumulative_best(y)

    current_params: Optional[np.ndarray] = None
    model: Optional[GPModel] = None

    for iteration in range(n_iter - n_initial):
        should_tune = bool(tune and iteration % tune_every == 0)
        model = fit_gp(X, y, tune=should_tune, start_params=current_params, lbfgs_iter=lbfgs_iter)
        current_params = model.log_params

        candidates = lower + (upper - lower) * np.random.rand(n_candidates, dim)
        mu, var = gp_predict(model, candidates)
        best_y = float(np.min(y))

        if acquisition == "ei":
            scores = expected_improvement(mu, var, best_y)
            x_next = candidates[int(np.argmax(scores))]
        elif acquisition == "pi":
            scores = probability_improvement(mu, var, best_y)
            x_next = candidates[int(np.argmax(scores))]
        elif acquisition == "lcb":
            scores = lower_confidence_bound(mu, var)
            x_next = candidates[int(np.argmin(scores))]
        elif acquisition == "ucb":
            # UCB est demandée dans le sujet. Pour minimiser Rosenbrock, on l'applique à -mu.
            scores = upper_confidence_bound(-mu, var)
            x_next = candidates[int(np.argmax(scores))]
        else:
            raise ValueError("acquisition doit valoir ei, pi, ucb ou lcb")

        X = np.vstack([X, x_next])
        y = np.append(y, objective(x_next))
        best_values.append(float(np.min(y)))

    final_model = fit_gp(X, y, tune=tune, start_params=current_params, lbfgs_iter=lbfgs_iter)
    return BOResult(X, y, best_values, final_model)



def random_search(objective: Callable[[np.ndarray], float], bounds: List[Tuple[float, float]],
                  n_iter: int = 200) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    dim = len(bounds)
    lower = np.array([b[0] for b in bounds], dtype=float)
    upper = np.array([b[1] for b in bounds], dtype=float)
    X = lower + (upper - lower) * np.random.rand(n_iter, dim)
    y = np.array([objective(x) for x in X], dtype=float)
    return X, y, cumulative_best(y)


# ---------------------------------------------------------------------------
# Résultats et statistiques
# ---------------------------------------------------------------------------


def bootstrap_ci(values: List[float], n_bootstrap: int = 1000, alpha: float = 0.05) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    means = [float(np.mean(np.random.choice(values, size=len(values), replace=True))) for _ in range(n_bootstrap)]
    return float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2)))



def paired_t_statistic(a: List[float], b: List[float]) -> float:
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if len(diff) < 2 or np.std(diff, ddof=1) < 1e-12:
        return 0.0
    return float(np.mean(diff) / (np.std(diff, ddof=1) / np.sqrt(len(diff))))



def diebold_mariano_approx(curve_a: np.ndarray, curve_b: np.ndarray) -> float:
    diff = np.asarray(curve_a, dtype=float) - np.asarray(curve_b, dtype=float)
    if len(diff) < 2 or np.std(diff, ddof=1) < 1e-12:
        return 0.0
    return float(np.mean(diff) / np.sqrt(np.var(diff, ddof=1) / len(diff)))



def plot_posterior_slice(model: GPModel, output_dir: str) -> str:
    idx = int(np.argmin(model.y_scaled))
    fixed_point = model.X[idx].copy()

    xs = np.linspace(-2.0, 2.0, 200)
    X_slice = np.tile(fixed_point, (len(xs), 1))
    X_slice[:, 0] = xs

    mu, var = gp_predict(model, X_slice)
    std = np.sqrt(np.maximum(var, 1e-12))
    true_values = np.array([rosenbrock(x) for x in X_slice])

    plt.figure(figsize=(8, 5))
    plt.plot(xs, true_values, label="Coupe réelle de Rosenbrock")
    plt.plot(xs, mu, label="Moyenne a posteriori du GP")
    plt.fill_between(xs, mu - 1.96 * std, mu + 1.96 * std, alpha=0.25, label="Intervalle a posteriori 95%")
    plt.xlabel("x1, autres dimensions fixées")
    plt.ylabel("f(x)")
    plt.title("Moyenne et incertitude a posteriori")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    path = os.path.join(output_dir, "posterior_mean_variance_slice.png")
    plt.savefig(path, dpi=300)
    plt.close()
    return path



def run_experiment(n_iter: int, runs: int, candidates: int, output_dir: str,
                   tune: bool = True, tune_every: int = 25, lbfgs_iter: int = 5) -> None:
    os.makedirs(output_dir, exist_ok=True)
    bounds = [(-2.0, 2.0)] * 10

    gp_finals: List[float] = []
    rs_finals: List[float] = []
    gp_curves: List[List[float]] = []
    rs_curves: List[List[float]] = []
    last_model: Optional[GPModel] = None

    for seed in range(runs):
        np.random.seed(seed)
        bo = bayesian_optimization(rosenbrock, bounds, n_iter=n_iter, n_candidates=candidates,
                                   acquisition="ei", tune=tune, tune_every=tune_every,
                                   lbfgs_iter=lbfgs_iter)
        gp_finals.append(float(np.min(bo.y)))
        gp_curves.append(bo.best_values)
        last_model = bo.final_model

        np.random.seed(seed)
        _, y_rs, rs_curve = random_search(rosenbrock, bounds, n_iter=n_iter)
        rs_finals.append(float(np.min(y_rs)))
        rs_curves.append(rs_curve)

        print(f"Seed {seed}: GP-EI={gp_finals[-1]:.4f}, Random Search={rs_finals[-1]:.4f}")

    gp_curves_arr = np.asarray(gp_curves)
    rs_curves_arr = np.asarray(rs_curves)
    mean_gp = np.mean(gp_curves_arr, axis=0)
    mean_rs = np.mean(rs_curves_arr, axis=0)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, n_iter + 1), mean_gp, label="GP-EI")
    plt.plot(range(1, n_iter + 1), mean_rs, label="Random Search")
    plt.xlabel("Nombre d'évaluations")
    plt.ylabel("Meilleure valeur trouvée")
    plt.title("Rosenbrock 10D : GP-EI vs Random Search")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "convergence_comparison.png"), dpi=300)
    plt.close()

    posterior_path = plot_posterior_slice(last_model, output_dir) if last_model else ""

    gp_ci = bootstrap_ci(gp_finals)
    rs_ci = bootstrap_ci(rs_finals)
    t_stat = paired_t_statistic(gp_finals, rs_finals)
    dm_stat = diebold_mariano_approx(mean_gp, mean_rs)

    with open(os.path.join(output_dir, "final_values.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["run", "gp_ei_final", "random_search_final"])
        for i, (gp_value, rs_value) in enumerate(zip(gp_finals, rs_finals)):
            writer.writerow([i, gp_value, rs_value])

    summary_path = os.path.join(output_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Résumé expérimental - Projet 5\n")
        f.write("================================\n")
        f.write(f"Dimension: 10\nBudget: {n_iter}\nRuns: {runs}\nCandidats: {candidates}\n")
        f.write(f"L-BFGS actif: {tune}\nTune every: {tune_every}\nLBFGS itérations: {lbfgs_iter}\n\n")
        f.write(f"GP-EI moyenne finale: {np.mean(gp_finals):.6f}\n")
        f.write(f"GP-EI IC bootstrap 95%: [{gp_ci[0]:.6f}, {gp_ci[1]:.6f}]\n\n")
        f.write(f"Random Search moyenne finale: {np.mean(rs_finals):.6f}\n")
        f.write(f"Random Search IC bootstrap 95%: [{rs_ci[0]:.6f}, {rs_ci[1]:.6f}]\n\n")
        f.write(f"Statistique t appariée: {t_stat:.6f}\n")
        f.write(f"Statistique Diebold-Mariano approximative: {dm_stat:.6f}\n")
        f.write(f"Figure moyenne/variance: {posterior_path}\n")

    print(f"Résultats sauvegardés dans : {output_dir}")



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Projet 5 - GP pour optimisation bayésienne")
    parser.add_argument("--n_iter", type=int, default=200)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--candidates", type=int, default=50)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--tune_every", type=int, default=25)
    parser.add_argument("--lbfgs_iter", type=int, default=5)
    parser.add_argument("--no_tune", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(
        n_iter=args.n_iter,
        runs=args.runs,
        candidates=args.candidates,
        output_dir=args.output_dir,
        tune=not args.no_tune,
        tune_every=args.tune_every,
        lbfgs_iter=args.lbfgs_iter,
    )

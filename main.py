import matplotlib.pyplot as plt
import numpy as np


# detail task 1
# set up the domain and core 1-D Transfinite Interpolation technique (TFI) function.
def apply_1d_tfi(x0: float, x1: float, s: np.ndarray) -> np.ndarray:
    """
    maps normalized distribution array s in [0, 1] to physical space [x0, x1].
    :param x0:
    :param x1:
    :param s:
    :return:
    """
    return (1.0 - s) * x0 + s * x1


# detail task 2
# point distribution functions
def dist_uniform(xi: np.ndarray):
    s = xi.copy()  # normalized coordinate used by TFI
    ds_dxi = np.ones_like(xi)  # exact analytical derivative
    return s, ds_dxi


def dist_exponential(xi: np.ndarray, beta: float = 3.0):
    if np.isclose(beta, 0.0):
        return dist_uniform(xi)
    denom = np.exp(beta) - 1.0
    s = (np.exp(beta * xi) - 1.0) / denom
    ds_dxi = (beta * np.exp(beta * xi)) / denom
    return s, ds_dxi


def dist_tanh_two_sided(xi: np.ndarray, beta: float = 2.5):
    if np.isclose(beta, 0.0):
        return dist_uniform(xi)
    arg = beta * (2.0 * xi - 1.0)
    tanh_b = np.tanh(beta)
    s = 0.5 * (1.0 + np.tanh(arg) / tanh_b)
    ds_dxi = (beta / tanh_b) * (1.0 / np.cosh(arg)) ** 2
    return s, ds_dxi


def dist_sinh_interior(xi: np.ndarray, beta: float = 4.0, s_c: float = 0.4):
    if np.isclose(beta, 0.0):
        return dist_uniform(xi)
    xi_0 = (1.0 / (2.0 * beta)) * np.log(
        (1.0 + (np.exp(beta) - 1.0) * s_c) / (1.0 + (np.exp(-beta) - 1.0) * s_c)
    )
    sinh_b_xi0 = np.sinh(beta * xi_0)
    s = s_c * (1.0 + np.sinh(beta * (xi - xi_0)) / sinh_b_xi0)
    ds_dxi = (beta * s_c / sinh_b_xi0) * np.cosh(beta * (xi - xi_0))
    return s, ds_dxi


# detail task 5
# verification checks subprogram
def verify_grid(x: np.ndarray, x0: float, x1: float) -> np.ndarray:
    # check boundary conformance
    assert np.isclose(x[0], x0), f"Start point mismatch: {x[0]} != {x0}"
    assert np.isclose(x[-1], x1), f"Start point mismatch: {x[-1]} != {x1}"

    # compute cell sized and check strict monotonicity (no folded/inverted cells)
    dx = np.diff(x)
    assert np.all(dx > 0.0), "Invalid grid: non-positive cell spacing detected!"

    # check total domain length conversion
    assert np.isclose(np.sum(dx), x1 - x0), (
        "Sum of cell widths does not equal domain length!"
    )

    return dx


# detail tasks 6 & 7
# compute grid metrics & analytic comparison
def compute_grid_metrics(
    x: np.ndarray, dx: np.ndarray, x0: float, x1: float, ds_dxi: np.ndarray
):
    N = len(x)
    dxi = 1.0 / (N - 1)

    # cell midpoints (length N-1) for plotting dx vs. x
    x_mid = 0.5 * (x[:-1] + x[1:])

    # adjacent cell strecthing ratio (length N-2, evaluated at interior nodes x[1:-1])
    stretching_ratio = np.maximum(dx[1:] / dx[:-1], dx[:-1] / dx[1:])

    # analytical Jacobian: dx/dxi = (x1 - x0) * ds/dxi
    dx_dxi_exact = (x1 - x0) * ds_dxi

    # numerical Jacobian using 2nd-order finite differences
    dx_dxi_num = np.zeros_like(x)
    # interior: 2nd-order central
    dx_dxi_num[1:-1] = (x[2:] - x[:-2]) / (2.0 * dxi)

    # left boundary: 2nd-order forward
    dx_dxi_num[0] = (-3.0 * x[0] + 4.0 * x[1] - x[2]) / (2.0 * dxi)

    # right boundary: 2nd-order backward
    dx_dxi_num[-1] = (3.0 * x[-1] - 4.0 * x[-2] + x[-3]) / (2.0 * dxi)

    # L2 error norm between numerical & analytical metric
    l2_error = np.sqrt(np.mean((dx_dxi_num - dx_dxi_exact) ** 2))

    return {
        "x_mid": x_mid,
        "stretching_ratio": stretching_ratio,
        "dx_dxi_exact": dx_dxi_exact,
        "dx_dxi_num": dx_dxi_num,
        "l2_error": l2_error,
    }


# detail task 3 & 4
# parametric study loop
def run_study():
    x0, x1 = 0.0, 1.0
    point_counts = [21, 51, 101]  # test 3 increasing node counts (exceeds min. of 2)

    distributions = {
        "Uniform": dist_uniform,
        "Exponential (beta=3.0)": lambda xi: dist_exponential(xi, beta=3.0),
        "Two-Sided Tanh (beta=2.5)": lambda xi: dist_tanh_two_sided(xi, beta=2.5),
        "Interior Sinh (beta=4.0, sc=0.4)": lambda xi: dist_sinh_interior(
            xi, beta=4.0, s_c=0.4
        ),
    }

    results = {}
    print(
        f"{'Distribution':<34} | {'N':>4} | {'dx_min':>9} | {'dx_max':>9} | {'Max Ratio':>9} | {'Metric L2 Err':>13}"
    )
    print("-" * 92)

    for name, dist_fn in distributions.items():
        results[name] = {}
        for N in point_counts:
            xi = np.linspace(0.0, 1.0, N)
            s, ds_dxi = dist_fn(xi)
            x = apply_1d_tfi(x0, x1, s)
            dx = verify_grid(x, x0, x1)
            metrics = compute_grid_metrics(x, dx, x0, x1, ds_dxi)

            results[name][N] = {"xi": xi, "x": x, "dx": dx, **metrics}

            print(
                f"{name:<34} | {N:>4d} | {dx.min():>9.5f} | {dx.max():>9.5f} | "
                f"{metrics['stretching_ratio'].max():>9.4f} | {metrics['l2_error']:>13.3e}"
            )

    return results, point_counts


# detail tasks 6–8
def plot_results(results: dict, point_counts: list):
    # figure 1: compare different point distributions at a fixed N (e.g., N = 51)
    N_ref = 51
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)

    for row_idx, (name, n_dict) in enumerate(results.items()):
        data = n_dict[N_ref]
        # top panel: actual node ticks along the 1-D line
        ax1.plot(data["x"], np.full_like(data["x"], row_idx), "|", markersize=14)
        # bottom panel: cell spacing dx across the domain
        ax2.plot(data["x_mid"], data["dx"], marker="o", markersize=3, label=name)

    ax1.set_yticks(range(len(results)))
    ax1.set_yticklabels(list(results.keys()))
    ax1.set_title(f"1-D Node Placement Comparison (N = {N_ref})")
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.set_xlabel("Physical Coordinate x")
    ax2.set_ylabel("Cell Width Δx")
    ax2.set_title(f"Local Grid Spacing Distribution Δx(x) (N = {N_ref})")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("fig1_distributions.png", dpi=300)

    # figure 2: effect of increasing point numbers N (using Two-Sided Tanh)
    target = "Two-Sided Tanh (beta=2.5)"
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(11, 4.5))

    for N in point_counts:
        data = results[target][N]
        ax3.plot(data["x_mid"], data["dx"], marker=".", label=f"N = {N}")
        ax4.plot(
            data["x"][1:-1], data["stretching_ratio"], marker=".", label=f"N = {N}"
        )

    ax3.set_xlabel("Physical Coordinate x")
    ax3.set_ylabel("Cell Width Δx")
    ax3.set_title(f"Spacing Refinement ({target})")
    ax3.legend()
    ax3.grid(True, linestyle="--", alpha=0.5)

    ax4.axhline(1.2, color="red", linestyle="--", label="CFD Limit (r = 1.2)")
    ax4.set_xlabel("Interior Node x")
    ax4.set_ylabel("Stretching Ratio r_i")
    ax4.set_title(f"Stretching Ratio vs. N ({target})")
    ax4.legend()
    ax4.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("fig2_refinement.png", dpi=300)

    # figure 3: analytic vs. numerical metric convergence (task 7)
    fig3, ax5 = plt.subplots(figsize=(6.5, 4.5))
    dxi_vals = [1.0 / (N - 1) for N in point_counts]

    for name, n_dict in results.items():
        if name == "Uniform":
            continue  # uniform error is ~1e-16 (machine zero), skip log-log slope
        errors = [n_dict[N]["l2_error"] for N in point_counts]
        ax5.loglog(dxi_vals, errors, marker="s", label=name)

    # reference slope-2 line to visually confirm O(dxi^2) convergence
    ref_line = [errors[0] * (d / dxi_vals[0]) ** 2 for d in dxi_vals]
    ax5.loglog(dxi_vals, ref_line, "k--", label="Reference Slope = 2 (O(Δξ²))")

    ax5.set_xlabel("Computational Grid Spacing Δξ")
    ax5.set_ylabel("Metric L2 Error ||(dx/dξ)_num - (dx/dξ)_exact||")
    ax5.set_title("Grid Metric Convergence Against Analytic Solution")
    ax5.legend()
    ax5.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("fig3_analytic_convergence.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    study_results, n_list = run_study()
    plot_results(study_results, n_list)

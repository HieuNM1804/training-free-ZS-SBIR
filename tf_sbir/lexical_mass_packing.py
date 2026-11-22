"""Maximum common submixture under several discrete observations of a public word."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.sparse import csc_matrix
from scipy.optimize import linprog
import torch


def certify(matrix, bound, primal, dual):
    allocation = np.maximum(primal, 0.)
    dual = np.maximum(dual, 0.)
    ratio = np.min(bound / np.maximum(matrix @ allocation, 1e-200))
    allocation *= min(1., ratio)
    lower_cost = np.min(matrix.T @ dual)
    if lower_cost <= 0:
        return allocation, np.inf
    upper = (bound @ dual) / min(1., lower_cost)
    return allocation, upper - allocation.sum()


def solve_words(anchor, rhs, jobs):
    """One LP per right-hand side. Return primal coefficients and a dual certificate."""
    count, size = anchor.shape
    rows = np.arange(count, dtype=np.int32)
    scale = anchor.max(axis=1).clip(min=1e-14)
    mat = np.ascontiguousarray(anchor / scale[:, None], dtype=np.float64)
    bounds = np.ascontiguousarray(rhs / scale[:, None], dtype=np.float64)
    blocks = np.array_split(np.argsort(rhs.argmax(axis=0), kind="stable"), jobs)
    result = np.zeros((size, rhs.shape[1]), dtype=np.float64)
    certificate = np.zeros(rhs.shape[1], dtype=np.float64)

    def work(indices):
        try:
            import highspy
        except ImportError:
            highspy = None
        if highspy:
            solver = highspy.Highs()
            for name, value in (("output_flag", False), ("threads", 1), ("solver", "simplex"), ("presolve", "off"),
                                ("primal_feasibility_tolerance", 1e-10), ("dual_feasibility_tolerance", 1e-10)):
                solver.setOptionValue(name, value)
            sparse = csc_matrix(mat)
            lp = highspy.HighsLp()
            lp.num_col_, lp.num_row_ = size, count
            lp.col_cost_ = np.ones(size)
            lp.col_lower_, lp.col_upper_ = np.zeros(size), np.full(size, highspy.kHighsInf)
            lp.row_lower_, lp.row_upper_ = np.full(count, -highspy.kHighsInf), np.ones(count)
            lp.sense_ = highspy.ObjSense.kMaximize
            lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
            lp.a_matrix_.start_, lp.a_matrix_.index_, lp.a_matrix_.value_ = sparse.indptr, sparse.indices, sparse.data
            solver.passModel(lp)
            lower = np.full(count, -highspy.kHighsInf)
        for i in indices:
            bound = np.ascontiguousarray(bounds[:, i])
            if highspy:
                solver.changeRowsBounds(count, rows, lower, bound)
                solver.run()
                if solver.getModelStatus() != highspy.HighsModelStatus.kOptimal:
                    raise ValueError(f"Gallery allocation failed for word {i}")
                sol = solver.getSolution()
                allocation = np.maximum(sol.col_value, 0.)
                dual = np.maximum(sol.row_dual, 0.)
            else:
                sol = linprog(-np.ones(size), A_ub=mat, b_ub=bound, method="highs-ds")
                if not sol.success:
                    raise ValueError(sol.message)
                allocation = np.maximum(sol.x, 0.)
                dual = np.maximum(-sol.ineqlin.marginals, 0.)
            allocation, gap = certify(mat, bound, allocation, dual)
            if not (-1e-7 <= gap <= 3e-5):
                # Rare terms can have a much smaller capacity in some cells than the gallery anchors.
                # Make each constraint unit capacity and solve the unchanged LP at this term's scale.
                local = mat / bound[:, None]
                precise = linprog(-np.ones(size), A_ub=local, b_ub=np.ones(count), method="highs-ds",
                                  options={"primal_feasibility_tolerance": 1e-9, "dual_feasibility_tolerance": 1e-9})
                if not precise.success:
                    raise ValueError(f"Rescaled gallery allocation failed for word {i}: {precise.message}")
                allocation, gap = certify(local, np.ones(count), precise.x, -precise.ineqlin.marginals)
            if gap < -1e-7 or gap > 3e-5 or allocation.sum() > 1 + 1e-7:
                raise ValueError(f"Uncertified allocation for word {i}: {gap}")
            result[:, i] = allocation
            certificate[i] = gap
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        for task in [pool.submit(work, ix) for ix in blocks if len(ix)]:
            task.result()
    return result, certificate


def single_alias(panels, anchors):
    target = panels.flatten(0, 1)
    known = target[:, anchors]
    mass, who = [], []
    for block in target.split(192, dim=1):
        possible = (block[:, None, :] / known[:, :, None]).amin(0)
        m, idx = possible.max(0)
        mass.append(m)
        who.append(idx)
    return torch.cat(mass).clamp(0, 1), torch.cat(who)


def pack(panels, anchor_ids, workers):
    p = panels.double()
    p = p / p.sum(1, keepdim=True)
    flat = p.flatten(0, 1)
    known = flat[:, anchor_ids]
    alpha, gaps = solve_words(known.cpu().numpy(), flat.cpu().numpy(), workers)
    alpha = torch.from_numpy(alpha).to(p.device)
    P = p.mean(0)
    represented = P[:, anchor_ids] @ alpha
    missing = (P - represented).clamp_min(0)
    one, which = single_alias(p, anchor_ids)
    single_residual = (P - P[:, anchor_ids[which]] * one[None]).clamp_min(0)
    alias_error = float(missing[:, anchor_ids].sum(0).abs().max())
    mass_error = float((missing.sum(0) + alpha.sum(0) - 1).abs().max())
    excess = float((known @ alpha - flat).amax())
    if alias_error > 3e-5 or mass_error > 2e-7 or excess > 2e-7:
        raise ValueError("Alias or probability mass conservation failed")
    diagnostics = {"highest_duality_gap": float(gaps.max()), "max_constraint_excess": excess, "mass_error": mass_error,
                   "max_anchor_missing_mass": alias_error, "covered_mean": float(alpha.sum(0).mean()),
                   "one_alias_covered_mean": float(one.mean()), "missing_mass_quantiles":
                   missing.sum(0).quantile(missing.new_tensor([.1, .25, .5, .75, .9])).tolist()}
    return {"total": P.float().cpu(), "missing": missing.float().cpu(), "single": single_residual.float().cpu(),
            "anchors": anchor_ids.cpu(), "checks": diagnostics}

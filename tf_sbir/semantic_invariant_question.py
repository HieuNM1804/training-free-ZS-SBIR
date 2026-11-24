"""A common semantic decision with a primal/dual certificate for reversing a sketch's answer."""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import linprog


@dataclass
class Optimum:
    answer: np.ndarray
    lower: float
    upper: float
    residual: float


def linear_program(contrast, basis, scale, method="highs-ds"):
    options = dict(dual_feasibility_tolerance=1e-9, primal_feasibility_tolerance=1e-9)
    fit = linprog(-scale * contrast, A_eq=scale * basis, b_eq=np.zeros(len(basis)),
                  bounds=(-1., 1.), method=method, options=options)
    if not fit.success:
        raise RuntimeError(fit.message)
    return fit.x, fit.eqlin.marginals


def bound_answer(contrast, equalities, basis, answer, multipliers):
    answer = answer - basis.T @ (basis @ answer)
    answer /= max(1., np.max(np.abs(answer)))
    lower = float(contrast @ answer)
    upper = float(np.abs(contrast + basis.T @ multipliers).sum())
    residual = float(np.max(np.abs(equalities @ answer)))
    if upper-lower < -2e-9 or residual >= 2e-9:
        raise ArithmeticError(f"Invalid numerical certificate: {lower}, {upper}, {residual}")
    return Optimum(answer, lower, max(upper, lower), residual)


def solve_decision(contrast, equalities, *, full=False):
    """Maximise c·g with Rg=0 and certify every original word, even in the grouped solver."""
    _, singular, vh = np.linalg.svd(equalities, full_matrices=False)
    cutoff = singular[0] * max(equalities.shape) * np.finfo(float).eps
    basis = vh[singular > cutoff]
    scale = len(contrast)
    if not full:
        # Restrict the primal to one answer per group. This supplies a feasible lower bound.
        code = (contrast >= 0).astype(np.int64)
        for row in basis:
            code = 2 * code + (row >= 0)
        group = np.unique(code, return_inverse=True)[1]
        for _ in range(32):
            cost = np.bincount(group, weights=contrast)
            rows = np.stack([np.bincount(group, weights=row) for row in basis])
            try:
                ans, dual = linear_program(cost, rows, scale)
            except RuntimeError:
                # A degenerate grouped problem can return HiGHS status 'unknown'.
                # The unrestricted problem has the same original equalities and directly certifies the answer.
                break
            result = bound_answer(contrast, equalities, basis, ans[group], dual)
            if result.upper-result.lower < 2e-11:
                return result
            # A restricted dual underestimates the whole-word L1 norm only when signs cancel.
            reduced_cost = contrast + basis.T @ dual
            split = np.unique(2*group + (reduced_cost >= 0), return_inverse=True)[1]
            if split.max() == group.max() or len(cost) >= len(group)//2:
                break
            group = split
    try:
        ans, dual = linear_program(contrast, basis, scale)
    except RuntimeError:
        ans, dual = linear_program(contrast, basis, scale, method="highs-ipm")
    result = bound_answer(contrast, equalities, basis, ans, dual)
    if result.upper-result.lower > 2e-7:
        raise ArithmeticError("Unresolved whole-vocabulary primal/dual gap")
    return result


class Question:
    """Rows of each candidate's experiment are its source- and target-prompt word distributions."""
    def __init__(self, left, right, invariant):
        for candidate in (left, right):
            if candidate.shape != left.shape or candidate.shape[0] != 2 or np.any(candidate < 0):
                raise ValueError("Each candidate must have two readings on the same vocabulary")
            np.testing.assert_allclose(candidate.sum(1), 1., atol=2e-10, rtol=0)
        a, b = left.mean(0), right.mean(0)
        self.contrast = (a-b)/2
        rows = [a+b]
        if invariant:
            rows.extend((left[0]-left[1], right[0]-right[1]))
        self.rows = np.stack(rows)
        self.best = solve_decision(self.contrast, self.rows)

    def read(self, sketch):
        np.testing.assert_allclose(sketch.sum(), 1., atol=2e-10, rtol=0)
        forced_tie = solve_decision(self.contrast, np.vstack((self.rows, sketch)))
        lower = max(0., (self.best.lower-forced_tie.upper)/2)
        upper = (self.best.upper-forced_tie.lower)/2
        if upper < lower-2e-9:
            raise ArithmeticError("The constrained optimum exceeds the unconstrained one")
        response = float(sketch @ self.best.answer)
        if lower <= 1e-8:
            lower = 0.
        return dict(response=response, signed_certificate=float(np.sign(response)*lower),
                    certificate_upper=max(0., upper), decision_accuracy=(1+self.best.lower)/2,
                    primal_dual_gap=max(self.best.upper-self.best.lower, forced_tie.upper-forced_tie.lower),
                    equality_residual=max(self.best.residual, forced_tie.residual))


def validate():
    # Any informative word partition here would also discriminate prompt style.
    a = np.array([[.55,.45],[.85,.15]])
    b = np.array([[.15,.85],[.45,.55]])
    assert Question(a,b,True).best.upper < 1e-8
    assert Question(a,b,False).best.lower > .3
    # The first two words denote one identity across both names for the style.
    a = np.array([[.72,.08,.18,.02],[.08,.72,.02,.18]])
    b = np.array([[.18,.02,.72,.08],[.02,.18,.08,.72]])
    question = Question(a,b,True)
    np.testing.assert_allclose(question.best.lower, .6, atol=1e-8)
    pos = question.read(a[0])
    neg = Question(b,a,True).read(a[0])
    np.testing.assert_allclose([pos["signed_certificate"],neg["signed_certificate"]], [.3,-.3], atol=1e-8)
    assert question.read((a+b).mean(0)/2)["signed_certificate"] == 0
    # Even two maximizers that give opposite verdicts on a word absent from both photos must tie.
    padded = Question(np.pad(a, ((0,0),(0,1))), np.pad(b, ((0,0),(0,1))), True)
    assert padded.read(np.array([0.,0.,0.,0.,1.]))["signed_certificate"] == 0
    rng = np.random.default_rng(42)
    for size in (11,37,1024):
        rows = rng.dirichlet(np.ones(size), size=5)
        test = Question(rows[:2], rows[2:4], True)
        for constraints in (test.rows, np.vstack((test.rows,rows[4]))):
            small = solve_decision(test.contrast, constraints)
            full = solve_decision(test.contrast, constraints, full=True)
            assert max(full.lower,small.lower) <= min(full.upper,small.upper)+2e-9

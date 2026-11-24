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


def solve_decision(contrast, equalities):
    """Maximize c·g for Rg=0, -1<=g<=1; project and rescale for the primal bound."""
    _, singular, vh = np.linalg.svd(equalities, full_matrices=False)
    cutoff = singular[0] * max(equalities.shape) * np.finfo(float).eps
    basis = vh[singular > cutoff]
    scale = len(contrast)
    options = dict(dual_feasibility_tolerance=1e-9, primal_feasibility_tolerance=1e-9)
    fit = linprog(-scale * contrast, A_eq=scale * basis, b_eq=np.zeros(len(basis)),
                  bounds=(-1., 1.), method="highs-ds", options=options)
    if not fit.success:
        raise RuntimeError(fit.message)
    answer = fit.x - basis.T @ (basis @ fit.x)
    answer /= max(1., np.max(np.abs(answer)))
    lower = float(contrast @ answer)
    upper = float(np.abs(contrast + basis.T @ fit.eqlin.marginals).sum())
    residual = float(np.max(np.abs(equalities @ answer)))
    if not (-2e-9 <= upper - lower <= 2e-7 and residual < 2e-9):
        raise ArithmeticError(f"Invalid numerical certificate: {lower}, {upper}, {residual}")
    return Optimum(answer, lower, max(upper, lower), residual)


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

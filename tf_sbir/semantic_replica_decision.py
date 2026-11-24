"""A finite minimax word test over competing photo families with a full-vocabulary dual bound."""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog


@dataclass
class RobustDecision:
    answer: np.ndarray
    lower: float
    upper: float
    weights: np.ndarray
    residual: float


def optimize(signed_photos, sketch=None, full=False):
    n = signed_photos.shape[1]
    rows = signed_photos if sketch is None else np.vstack((signed_photos,sketch))
    # A feasible primal answer is constant in each provisional group. The dual is evaluated on all words.
    group = (signed_photos.mean(0)>=0).astype(np.int64)
    for row in signed_photos:
        group = 2*group + (abs(row)>1/n)
    group = np.unique(group,return_inverse=True)[1]
    for iteration in range(35):
        if iteration==33 or full:
            group=np.arange(n)
        grouped = np.stack([np.bincount(group,weights=v) for v in rows])
        k = len(grouped[0])
        constraints=np.column_stack((-grouped[:len(signed_photos)],np.ones(len(signed_photos))))
        equality=None if sketch is None else np.r_[grouped[-1],0.][None]
        lp=linprog(np.r_[np.zeros(k),-float(n)], A_ub=n*constraints, b_ub=np.zeros(len(constraints)),
                   A_eq=None if equality is None else n*equality, b_eq=None if equality is None else [0.],
                   bounds=[(-1.,1.)]*k+[(None,None)], method="highs-ds",
                   options=dict(primal_feasibility_tolerance=1e-9,dual_feasibility_tolerance=1e-9))
        if not lp.success:
            raise RuntimeError(lp.message)
        weights=np.maximum(-lp.ineqlin.marginals,0)
        weights/=weights.sum()
        reduced=weights@signed_photos
        answer=lp.x[:-1][group]
        residual=0.
        if sketch is not None:
            reduced+=lp.eqlin.marginals[0]*sketch
            answer-=sketch*((sketch@answer)/(sketch@sketch))
            answer/=max(1.,abs(answer).max())
            residual=float(abs(sketch@answer))
        lower=float((signed_photos@answer).min())
        upper=float(abs(reduced).sum())
        if upper < lower-1e-9 or residual>2e-9:
            raise ArithmeticError("Invalid minimax certificate")
        if upper-lower<2e-9:
            return RobustDecision(answer,lower,max(upper,lower),weights,residual)
        if k==n:
            if upper-lower<2e-7:
                return RobustDecision(answer,lower,upper,weights,residual)
            raise ArithmeticError(f"Unresolved whole-word minimax gap {upper-lower}")
        split = np.unique(2*group+(reduced>=0),return_inverse=True)[1]
        if split.max()==group.max():
            full=True
        group=split
    raise ArithmeticError("No certified solution")


def decision_rows(left, right, arm):
    if arm=="original":
        return np.stack((left[0],-right[0]))
    if arm=="pooled":
        return np.stack((left.mean(0),-right.mean(0)))
    if arm=="exchanged":
        return np.stack((left[0],right[1],-right[0],-left[1]))
    if arm=="robust":
        return np.vstack((left,-right))
    raise ValueError(arm)


def read(signed, optimal, reading):
    tie=optimize(signed,reading)
    response=float(reading@optimal.answer)
    gap=max(0.,(optimal.lower-tie.upper)/2)
    if gap<1e-8:
        gap=0.
    return dict(answer=response,certificate=float(np.sign(response)*gap),
                certificate_upper=max(0.,(optimal.upper-tie.lower)/2),
                numerical_gap=max(optimal.upper-optimal.lower,tie.upper-tie.lower),
                residual=tie.residual)


def validate():
    a=np.array([[1.,0.,0.],[0.,1.,0.]])
    b=np.array([[0.,0.,1.],[0.,1.,0.]])
    z=decision_rows(a,b,"robust")
    assert optimize(z).upper<1e-8 and optimize(decision_rows(a,b,"pooled")).lower>.49
    rng=np.random.default_rng(42)
    for size in (17,127,1024):
        rows=rng.dirichlet(np.ones(size),5)
        z=decision_rows(rows[:2],rows[2:4],"robust")
        for q in (None,rows[4]):
            grouped=optimize(z,q)
            direct=optimize(z,q,full=True)
            assert max(grouped.lower,direct.lower)<min(grouped.upper,direct.upper)+2e-8
        f=read(z,optimize(z),rows[4])["certificate"]
        rev=read(-z,optimize(-z),rows[4])["certificate"]
        assert abs(f+rev)<1e-7

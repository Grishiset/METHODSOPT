"""Weighted p-median with splittable deliveries and binary site openings."""
import sys
from itertools import combinations
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix, eye, hstack, kron, vstack, csr_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import dump, records, run, verified


def generate():
    rng = np.random.default_rng(10228)
    return dict(seed=10228, p=3, districts=rng.integers(0, 101, (40, 2)).tolist(),
                sites=rng.integers(0, 101, (40, 2)).tolist(), orders=rng.integers(8, 36, 40).tolist())


def solve(km, demand, p, closed, integer=True, multiplier=1):
    n, m = km.shape
    # Variables: first the m opening indicators, then daily flows (district-major).
    nr = n*m
    one_row = csr_matrix(np.ones((1,m)))
    opens = hstack((one_row, csr_matrix((1,nr))))
    balance = hstack((csr_matrix((n,m)), kron(eye(n), np.ones((1,m)))))
    rows = np.arange(nr)
    links = coo_matrix((-np.repeat(demand,m)*multiplier, (rows, rows % m)), shape=(nr,m))
    limits = hstack((links, eye(nr)))
    matrix = vstack((opens,balance,limits),format="csc")
    rhs = np.r_[p,demand,np.zeros(nr)]
    low = np.r_[p,demand,np.full(nr,-np.inf)]
    upper = np.ones(m)
    upper[closed] = 0
    start = perf_counter()
    result = milp(np.r_[np.zeros(m),km.ravel()],integrality=np.r_[np.full(m,int(integer)),np.zeros(nr)],
                  constraints=LinearConstraint(matrix,low,rhs),
                  bounds=Bounds(0,np.r_[upper,np.repeat(demand,m)]),
                  options={"disp":True,"mip_rel_gap":0})
    elapsed = perf_counter()-start
    if not result.success:
        raise RuntimeError(result.message)
    y, x = result.x[:m], result.x[m:].reshape(n,m)
    check = verified(dict(count_error=abs(y.sum()-p), bounds_violation=max(0,-y.min(),y.max()-1,-x.min()),
                          demand_error=max(abs(x.sum(axis=1)-demand)),
                          closure_error=max([0,*y[closed]]),
                          link_violation=max(0,(x-demand[:,None]*multiplier*y).max()),
                          integer_error=max(abs(y-np.rint(y))) if integer else 0,
                          objective_error=abs((x*km).sum()-result.fun)))
    return y,x,float(result.fun),elapsed,getattr(result,"mip_gap",None),check


def forward(km,demand,p,closed):
    selected = []
    for _ in range(p):
        remaining = sorted(set(range(km.shape[1]))-set(selected)-set(closed))
        selected.append(min(remaining,key=lambda j: demand @ km[:,selected+[j]].min(axis=1)))
    return selected,float(demand @ km[:,selected].min(axis=1))


def experiment(data,out):
    p = data["p"]
    scenarios, plans, allocations, timings, checks = [], [], [], [], {}
    chosen = []
    for name,n,mult,closed_count in [("База",15,1,0),("Дефицит",15,1,1),("Стресс",15,1,2),
                                      ("Слабая связка",15,100,0),("Масштаб 10",10,1,0),
                                      ("Масштаб 20",20,1,0),("Масштаб 40",40,1,0)]:
        districts = np.array(data["districts"][:n])
        sites = np.array(data["sites"][:n])
        demand = np.array(data["orders"][:n])
        closed = chosen[:closed_count]
        km = abs(districts[:,None,:]-sites).sum(axis=2)
        print(name,flush=True)
        lp = solve(km,demand,p,closed,False,mult)
        mip = solve(km,demand,p,closed,True,mult)
        elapsed_start = perf_counter()
        greedy_sites, heuristic = forward(km,demand,p,closed)
        greedy_time = perf_counter()-elapsed_start
        active = np.flatnonzero(mip[0]>.5)
        if name == "База":
            chosen = sorted(active.tolist(),key=lambda j: (-mip[1][:,j].sum(),j))
        gap = 0.0 if abs(mip[2]-lp[2]) < 1e-7 else 100*(mip[2]-lp[2])/mip[2]
        row = dict(case=name, n=n, closed=" ".join(str(i+1) for i in closed),
                   sites=" ".join(str(i+1) for i in active), lp=lp[2], milp=mip[2],
                   gap_pct=gap, mip_solver_gap=mip[4], greedy_sites=" ".join(str(i+1) for i in greedy_sites),
                   greedy=heuristic, greedy_gap_pct=100*(heuristic-mip[2])/mip[2],
                   lp_s=lp[3], milp_s=mip[3], greedy_s=greedy_time)
        scenarios.append(row)
        checks[name] = {"LP":lp[5],"MILP":mip[5]}
        if n <= 20:
            enum = min(float(demand @ km[:,s].min(axis=1))
                       for s in combinations(sorted(set(range(n))-set(closed)),p))
            checks[name]["перебор троек"] = verified(dict(objective_error=abs(enum-mip[2])))
        for kind,(y,x,*_) in [("LP",lp),("MILP",mip)]:
            for j,val in enumerate(y):
                plans.append(dict(case=name, model=kind, site=j+1, opening=val, orders=x[:,j].sum()))
            for i,j in zip(*np.where(x>1e-8)):
                allocations.append(dict(case=name, model=kind, district=i+1, site=j+1,
                                        deliveries=x[i,j], km=km[i,j]))
        if name.startswith("Масштаб") or name == "База":
            timings.append(row)
    records(out/"solution.csv",plans)
    records(out/"allocations.csv",allocations)
    records(out/"scenarios.csv",scenarios)
    dump(out/"checks.json",checks)
    timings.sort(key=lambda r:r["n"])
    fig, axes = plt.subplots(1,2,figsize=(9,3.6))
    for field,label in [("lp_s","LP"),("milp_s","MILP")]:
        axes[0].plot([r["n"] for r in timings],[r[field] for r in timings],"o-",label=label)
    axes[0].legend()
    axes[0].set_ylabel("Время работы HiGHS, с")
    axes[1].plot([r["n"] for r in timings],[r["gap_pct"] for r in timings],"o-",color="#13756b")
    axes[1].set_ylabel("Зазор релаксации, %")
    for ax in axes:
        ax.set_xlabel("Число районов и площадок")
        ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(out/"scale.png",dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    run(__file__,generate,experiment)

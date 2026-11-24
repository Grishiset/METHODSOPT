"""Continuous bakery production in kg per shift."""
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.optimize import linprog

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import dump, records, run, verified


def generate():
    rng = np.random.default_rng(10228)
    names = ["Багет", "Батон", "Ржаной", "Чиабатта", "Бриошь", "Круассан", "Слойка", "Фокачча"]
    return dict(seed=10228, names=names, profit=rng.integers(55, 141, 8).tolist(),
                minutes=rng.integers(1, 10, (3, 8)).tolist(), capacity=[410, 470, 450],
                demand=rng.integers(14, 36, 8).tolist())


def experiment(data, out):
    profit = np.array(data["profit"])
    time = np.array(data["minutes"])
    base = np.array(data["capacity"], dtype=float)
    demand = np.array(data["demand"])
    cases, plans, prices, checks, benchmarks = [], [], [], {}, []
    reference = None
    for name, factor in [("База", [1, 1, 1]), ("Дефицит", [.8, 1, 1]),
                         ("Стресс", [.8, .75, 1]), ("Малый шаг", [1, 1 + 1 / base[1], 1])]:
        cap = base * factor
        args = dict(c=-profit, A_ub=time, b_ub=cap, bounds=list(zip(np.zeros(8), demand)))
        print(name, flush=True)
        start = perf_counter()
        res = linprog(**args, method="highs-ds", options={"disp": True})
        elapsed = perf_counter() - start
        if not res.success:
            raise RuntimeError(res.message)
        x = res.x
        value = profit @ x
        shadows = -res.ineqlin.marginals
        if reference is None:
            reference = (value, shadows)
            benchmarks.append(dict(method="highs-ds", seconds=elapsed, iterations=res.nit, profit=value))
            start = perf_counter()
            ip = linprog(**args, method="highs-ipm", options={"disp": True})
            benchmarks.append(dict(method="highs-ipm", seconds=perf_counter()-start,
                                   iterations=ip.nit, profit=-ip.fun))
            if not ip.success or abs(ip.fun-res.fun) > 1e-7:
                raise ValueError("IPM cross-check")
            dual = linprog(np.r_[cap, demand], A_ub=-np.column_stack((time.T,np.eye(8))),
                           b_ub=-profit, method="highs-ds", options={"disp":True})
            if not dual.success:
                raise RuntimeError(dual.message)
            y, v = dual.x[:3], dual.x[3:]
            checks["Двойственная задача"] = verified(dict(gap=dual.fun-value,
                negative_price=max(0,-dual.x.min()), product_deficit=max(0,max(profit-time.T@y-v)),
                resource_complementarity=max(abs(y*(cap-time@x))),
                demand_complementarity=max(abs(v*(demand-x))),
                product_complementarity=max(abs(x*(time.T@y+v-profit)))))
            dump(out/"dual.json",dict(oven_prices=y.tolist(),demand_prices=v.tolist(),objective=dual.fun))
        checks[name] = verified(dict(capacity_violation=max(0, max(time @ x - cap)),
                                     lower_violation=max(0, max(-x)), upper_violation=max(0, max(x-demand)),
                                     objective_error=abs(value+res.fun)))
        cases.append(dict(case=name, status="optimal", profit=value, change=value-reference[0],
                          shadow_prediction=reference[1] @ (cap-base)))
        for i, product in enumerate(data["names"]):
            plans.append(dict(case=name, product=product, kg=x[i], limit_kg=demand[i],
                              profit_per_kg=profit[i], demand_shadow=-res.upper.marginals[i],
                              reduced_profit=-res.lower.marginals[i]))
        for j in range(3):
            prices.append(dict(case=name, oven=j+1, used_minutes=(time @ x)[j], limit_minutes=cap[j],
                               rub_per_extra_minute=shadows[j]))
    records(out/"solution.csv", plans)
    records(out/"ovens.csv", prices)
    records(out/"scenarios.csv", cases)
    records(out/"algorithms.csv", benchmarks)
    dump(out/"checks.json", checks)


if __name__ == "__main__":
    run(__file__, generate, experiment)

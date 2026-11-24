"""Two transport modes represented by parallel arcs, each with its own tariff."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linprog, milp, Bounds, LinearConstraint

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import dump, records, run, verified

MODES = ["Авто", "Ж/д"]


def generate():
    rng = np.random.default_rng(10228)
    distance = rng.integers(40, 201, (4, 5))
    cost = np.stack((2*distance + 25, distance + 68), axis=-1)
    limits = np.stack((np.full((4, 5), 40), rng.integers(4, 15, (4, 5))), axis=-1)
    return dict(seed=10228, cost=cost.tolist(), capacity=limits.tolist(),
                supply=[28, 33, 26, 35], demand=[20, 25, 18, 22, 20])


def model(shape):
    depots, stores, modes = shape
    ship = np.repeat(np.eye(depots), stores*modes, axis=1)
    receive = np.tile(np.repeat(np.eye(stores), modes, axis=1), depots)
    return ship, receive


def greedy(cost, cap, supply, demand):
    stock, need = supply.copy(), demand.copy()
    x = np.zeros_like(cost, dtype=float)
    for pos in np.argsort(cost, axis=None, kind="stable"):
        i, j, k = np.unravel_index(pos, cost.shape)
        x[i,j,k] = min(cap[i,j,k], stock[i], need[j])
        stock[i] -= x[i,j,k]
        need[j] -= x[i,j,k]
    if np.any(need):
        raise ValueError("Не весь спрос покрыт эвристикой")
    return x


def audit(x, cost, cap, supply, demand, value):
    return verified(dict(supply_excess=max(0, max(x.sum(axis=(1,2))-supply)),
                         demand_error=max(abs(x.sum(axis=(0,2))-demand)),
                         capacity_excess=max(0, (x-cap).max()), lower_violation=max(0, -x.min()),
                         objective_error=abs((x*cost).sum()-value)))


def draw(out, x, cap):
    fig, axes = plt.subplots(2, 4, figsize=(13, 6))
    for k, mode in enumerate(MODES):
        for i in range(4):
            ax = axes[k, i]
            for j, y in enumerate(range(4, -1, -1)):
                qty = x[i,j,k]
                color = "#11685e" if qty > 0 else "#b5bfc4"
                ax.annotate("", xy=(.73, y), xytext=(.13, 2),
                            arrowprops=dict(arrowstyle="->", color=color, lw=1+1.3*(qty>0)))
                ax.text(.77,y,f"М{j+1}   {qty:g}/{cap[i,j,k]:g}", va="center", fontsize=10)
            ax.text(.06,2,f"С{i+1}",ha="right",va="center",weight="bold")
            ax.set(xlim=(-.13,1.55), ylim=(-.5,4.5), title=f"{mode} · склад {i+1}")
            ax.axis("off")
    fig.suptitle("Поставки по дугам сети · поток / пропускная способность, т за сутки")
    fig.tight_layout()
    fig.savefig(out/"network.png", dpi=150)
    plt.close(fig)


def experiment(data, out):
    cost, base, stock, demand0 = (np.array(data[k]) for k in ("cost","capacity","supply","demand"))
    ship, receive = model(cost.shape)
    plans, scenarios, balances, checks = [], [], [], {}
    for name in ["База", "Перегрузка", "Отказ", "Рост спроса", "Стресс"]:
        cap = base.copy()
        demand = demand0.astype(float)
        if name == "Перегрузка":
            cap[critical] = int(peak_flow // 2)
        if name in ("Отказ", "Стресс"):
            cap[critical] = 0
        if name in ("Рост спроса", "Стресс"):
            demand *= 1.2
        print(name, flush=True)
        res = linprog(cost.ravel(), A_ub=ship, b_ub=stock, A_eq=receive, b_eq=demand,
                      bounds=list(zip(np.zeros(cost.size), cap.ravel())), method="highs-ds",
                      options={"disp": True})
        row = dict(case=name, status=res.message, demand=demand.sum(), supply=stock.sum(),
                   cost="", road_t="", rail_t="", greedy_cost="", integer_cost="")
        if res.success:
            x = res.x.reshape(cost.shape)
            checks[name] = audit(x,cost,cap,stock,demand,res.fun)
            row.update(cost=res.fun, road_t=x[:,:,0].sum(), rail_t=x[:,:,1].sum())
            for i,j,k in np.ndindex(cost.shape):
                pos = np.ravel_multi_index((i,j,k),cost.shape)
                plans.append(dict(case=name, depot=i+1, shop=j+1, mode=MODES[k], tonnes=x[i,j,k],
                                  capacity=cap[i,j,k], tariff=cost[i,j,k], marginal=res.upper.marginals[pos]))
            for i, used in enumerate(x.sum(axis=(1,2))):
                balances.append(dict(case=name,node=f"С{i+1}",available=stock[i],flow=used,
                                     remainder=stock[i]-used, marginal=res.ineqlin.marginals[i]))
            for j, qty in enumerate(x.sum(axis=(0,2))):
                balances.append(dict(case=name,node=f"М{j+1}",available=demand[j],flow=qty,
                                     remainder=demand[j]-qty, marginal=res.eqlin.marginals[j]))
            if name == "База":
                critical = np.unravel_index(x.argmax(),x.shape)
                peak_flow = x[critical]
                draw(out,x,cap)
                g = greedy(cost,cap,stock,demand)
                row["greedy_cost"] = float((cost*g).sum())
                checks["Эвристика"] = audit(g,cost,cap,stock,demand,row["greedy_cost"])
                integer = milp(cost.ravel(),integrality=np.ones(cost.size),
                               bounds=Bounds(0,cap.ravel()), constraints=[
                                   LinearConstraint(ship,-np.inf,stock), LinearConstraint(receive,demand,demand)],
                               options={"disp":True})
                if not integer.success:
                    raise RuntimeError(integer.message)
                row["integer_cost"] = integer.fun
                checks["Целые тонны"] = audit(integer.x.reshape(cost.shape),cost,cap,stock,demand,integer.fun)
        elif res.status != 2 or demand.sum() <= stock.sum():
            raise RuntimeError(res.message)
        else:
            checks[name] = dict(demand_exceeds_total_supply_by=float(demand.sum()-stock.sum()))
        scenarios.append(row)
    records(out/"solution.csv",plans)
    records(out/"balances.csv",balances)
    records(out/"scenarios.csv",scenarios)
    dump(out/"checks.json",checks)


if __name__ == "__main__":
    run(__file__, generate, experiment)

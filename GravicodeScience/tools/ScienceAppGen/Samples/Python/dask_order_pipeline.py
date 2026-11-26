"""Monthly order exports, analysed in parallel with dask.

A single-file example for ScienceAppGen's Python to C# converter: dask.dataframe over a folder of
CSV files, dask-sql for a running total, dask.delayed for a what-if sweep on a local
dask.distributed cluster, and matplotlib for the figure. It writes its own input files, so it runs
anywhere. Verified with dask, distributed and dask-sql 2024.5.0 and pandas 2.2 - dask-sql has
not kept up with later dask and pandas releases.
"""
import os

import numpy as np
import pandas as pd
import dask
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster
from dask_sql import Context
import matplotlib.pyplot as plt

REGIONS = ["north", "south", "east", "west"]
DATA_DIR = "orders"


def write_exports(seed=7):
    """Twelve monthly CSV exports, 2,000 orders each."""
    rng = np.random.default_rng(seed)
    os.makedirs(DATA_DIR, exist_ok=True)
    for month in range(1, 13):
        n = 2000
        frame = pd.DataFrame({
            "month": np.full(n, month),
            "region": rng.choice(REGIONS, n),
            "qty": 1 + rng.poisson(2, n),
            "price": np.round(rng.uniform(5, 120, n), 2),
        })
        frame.to_csv(os.path.join(DATA_DIR, f"orders-{month:02d}.csv"), index=False)


def revenue_by_region():
    """Filter, derive a column and aggregate, lazily, over every file at once."""
    orders = dd.read_csv(os.path.join(DATA_DIR, "orders-*.csv"))
    orders = orders[orders.qty > 0]
    orders = orders.assign(revenue=orders.qty * orders.price)
    summary = orders.groupby("region").agg({"revenue": "sum", "qty": "mean"})
    return summary.compute().sort_values("revenue", ascending=False)


def running_total(region):
    """The same files through SQL, with a window function for the year-to-date total."""
    c = Context()
    c.create_table("orders", dd.read_csv(os.path.join(DATA_DIR, "orders-*.csv")))
    return c.sql(f"""
        SELECT month, SUM(qty * price) AS revenue,
               SUM(SUM(qty * price)) OVER (ORDER BY month) AS ytd
        FROM orders
        WHERE region = '{region}'
        GROUP BY month
        ORDER BY month
    """).compute()


def margin_after_discount(revenue, discount):
    """Margin if every order got `discount` off and cost stayed at 70% of list price."""
    return revenue * (1 - discount) - revenue * 0.7


def discount_sweep(total_revenue, client):
    """One delayed task per discount level, run on the cluster."""
    discounts = [d / 100 for d in range(0, 31, 2)]
    tasks = [dask.delayed(margin_after_discount)(total_revenue, d) for d in discounts]
    margins = dask.compute(*tasks, scheduler=client)
    return discounts, list(margins)


def main():
    write_exports()
    summary = revenue_by_region()
    print(summary)

    ytd = running_total("north")
    print(ytd)

    with LocalCluster(n_workers=4, threads_per_worker=2) as cluster, Client(cluster) as client:
        discounts, margins = discount_sweep(summary["revenue"].sum(), client)
    for d, m in zip(discounts, margins):
        print(f"discount {d:4.0%}: margin {m:12,.0f}")

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.5))
    left.plot(ytd["month"], ytd["ytd"] / 1e6, marker="o")
    left.set_title("North: year-to-date revenue")
    left.set_xlabel("month")
    left.set_ylabel("million")
    right.plot([d * 100 for d in discounts], [m / 1e6 for m in margins])
    right.axhline(0, color="gray", linestyle="--")
    right.set_title("Margin by blanket discount")
    right.set_xlabel("discount (%)")
    right.set_ylabel("million")
    fig.tight_layout()
    fig.savefig("dask_orders.png")
    print("saved dask_orders.png")


if __name__ == "__main__":
    main()

import pandas as pd
import networkx as nx
import json
from itertools import combinations


def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    cols = [
        "EmployeeNumber", "Department", "JobRole", "JobLevel",
        "YearsAtCompany", "YearsWithCurrManager",
        "JobSatisfaction", "Attrition", "Age", "Gender",
        "MonthlyIncome", "OverTime", "PerformanceRating"
    ]
    df = df[cols].copy()
    df["Attrition"] = (df["Attrition"] == "Yes").astype(int)
    df["OverTime"]  = (df["OverTime"]  == "Yes").astype(int)

    print(f"Loaded {len(df)} employees | {df['Department'].nunique()} departments")
    return df

WEIGHTS = {
    "department": 0.50,  
    "job_level":  0.25,
    "job_role":   0.15,
    "tenure":     0.10,
}

EDGE_THRESHOLD = 0.40   


def _score_department(r1, r2) -> float:
    return 1.0 if r1["Department"] == r2["Department"] else 0.0


def _score_job_level(r1, r2) -> float:
    diff = abs(r1["JobLevel"] - r2["JobLevel"])
    return max(0.0, 1.0 - diff * 0.30)


def _score_job_role(r1, r2) -> float:
    return 1.0 if r1["JobRole"] == r2["JobRole"] else 0.0


def _score_tenure(r1, r2) -> float:
    diff = abs(r1["YearsAtCompany"] - r2["YearsAtCompany"])
    return max(0.0, 1.0 - diff * 0.10)


def compute_edge_weight(r1: pd.Series, r2: pd.Series) -> float:
    score = (
        WEIGHTS["department"] * _score_department(r1, r2)
        + WEIGHTS["job_level"]  * _score_job_level(r1, r2)
        + WEIGHTS["job_role"]   * _score_job_role(r1, r2)
        + WEIGHTS["tenure"]     * _score_tenure(r1, r2)
    )
    return round(score, 4)

def build_graph(df: pd.DataFrame) -> nx.Graph:
    G = nx.Graph()

    for _, row in df.iterrows():
        G.add_node(
            row["EmployeeNumber"],
            department    = row["Department"],
            job_role      = row["JobRole"],
            job_level     = int(row["JobLevel"]),
            years_company = int(row["YearsAtCompany"]),
            satisfaction  = int(row["JobSatisfaction"]),
            attrition     = int(row["Attrition"]) if isinstance(row["Attrition"], (int, float)) else (1 if row["Attrition"] == "Yes" else 0),
            age           = int(row["Age"]),
            income        = int(row["MonthlyIncome"]),
            overtime      = int(row["OverTime"]) if isinstance(row["OverTime"], (int, float)) else (1 if row["OverTime"] == "Yes" else 0),
        )

    records = df.set_index("EmployeeNumber").to_dict("index")
    emp_ids = list(records.keys())
    edge_count = 0

    for id1, id2 in combinations(emp_ids, 2):
        r1 = pd.Series(records[id1])
        r2 = pd.Series(records[id2])
        weight = compute_edge_weight(r1, r2)

        if weight >= EDGE_THRESHOLD:
            G.add_edge(id1, id2, weight=weight)
            edge_count += 1

    print(f"Graph built: {G.number_of_nodes()} nodes | {edge_count} edges")
    print(f"Edge threshold: {EDGE_THRESHOLD} | Avg degree: {sum(dict(G.degree()).values()) / G.number_of_nodes():.1f}")
    return G

def graph_summary(G: nx.Graph) -> dict:
    degrees = dict(G.degree())
    top5 = sorted(degrees, key=degrees.get, reverse=True)[:5]

    summary = {
        "nodes"            : G.number_of_nodes(),
        "edges"            : G.number_of_edges(),
        "avg_degree"       : round(sum(degrees.values()) / len(degrees), 2),
        "density"          : round(nx.density(G), 4),
        "is_connected"     : nx.is_connected(G),
        "components"       : nx.number_connected_components(G),
        "top5_by_degree"   : top5,
    }

    print("\n── Graph Summary ──────────────────")
    for k, v in summary.items():
        print(f"  {k:20s}: {v}")
    return summary


def export_graph(G: nx.Graph, out_path: str = "graph_data.json"):
    data = nx.node_link_data(G)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\nExported → {out_path}")

if __name__ == "__main__":
    import sys

    dataset_path = sys.argv[1] if len(sys.argv) > 1 else "WA_Fn-UseC_-HR-Employee-Attrition.csv"

    df = load_dataset(dataset_path)
    G  = build_graph(df)
    _  = graph_summary(G)
    export_graph(G, "graph_data.json")
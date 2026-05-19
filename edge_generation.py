"""
edge_generation.py
==================
ONA Project — Edge Generation Module
สร้างความสัมพันธ์ระหว่างพนักงาน (Edge) จากข้อมูล HR
"""

import pandas as pd
import networkx as nx
import json
import numpy as np
from itertools import combinations

# ─── Default Config ────────────────────────────────────────────────────────────
DEFAULT_WEIGHTS = {
    "department": 0.50,   # น้ำหนักความสัมพันธ์ภายในแผนก
    "job_level":  0.25,   # อิทธิพลของลำดับชั้น
    "job_role":   0.15,   # ความเชื่อมโยงสายอาชีพ
    "tenure":     0.10,   # ความผูกพันตามอายุงาน
}
DEFAULT_THRESHOLD = 0.40  # ค่าขั้นต่ำของ edge weight ที่จะสร้าง edge

REQUIRED_COLS = [
    "EmployeeNumber", "Department", "JobRole", "JobLevel",
    "YearsAtCompany", "JobSatisfaction", "Attrition",
    "Age", "MonthlyIncome",
]

# ─── Data Loading ───────────────────────────────────────────────────────────────
def load_dataset(path: str) -> pd.DataFrame:
    """
    โหลดไฟล์ CSV และเตรียมข้อมูลให้พร้อมใช้
    รองรับทั้ง IBM HR Dataset และไฟล์ที่ upload เอง
    """
    df = pd.read_csv(path)

    # เช็คว่ามี column ที่จำเป็นครบไหม
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"ไฟล์ขาด column: {missing}")

    df = df[REQUIRED_COLS].copy()

    # แปลง categorical เป็น numeric
    df["Attrition_flag"] = (df["Attrition"] == "Yes").astype(int)

    # เติม missing values
    df["JobLevel"]       = df["JobLevel"].fillna(df["JobLevel"].median())
    df["YearsAtCompany"] = df["YearsAtCompany"].fillna(0)
    df["JobSatisfaction"]= df["JobSatisfaction"].fillna(2)

    print(f"✅ โหลดสำเร็จ: {len(df)} พนักงาน | {df['Department'].nunique()} แผนก")
    return df


def load_default_data(n: int = 150) -> pd.DataFrame:
    """
    สร้างข้อมูลจำลองที่สมจริง สำหรับ demo โดยไม่ต้องอัปโหลดไฟล์
    """
    rng = np.random.default_rng(42)  # seed คงที่ ผลเหมือนเดิมทุกครั้ง

    depts = ["Sales", "Research & Development", "Human Resources"]
    roles_by_dept = {
        "Sales":                    ["Sales Executive", "Sales Representative", "Manager"],
        "Research & Development":   ["Research Scientist", "Laboratory Technician", "Manager", "Developer"],
        "Human Resources":          ["HR Representative", "HR Manager", "Recruiter"],
    }

    rows = []
    for i in range(1, n + 1):
        dept = rng.choice(depts)
        rows.append({
            "EmployeeNumber":  i,
            "Department":      dept,
            "JobRole":         rng.choice(roles_by_dept[dept]),
            "JobLevel":        int(rng.integers(1, 6)),
            "YearsAtCompany":  int(rng.integers(1, 25)),
            "JobSatisfaction": int(rng.integers(1, 5)),
            "Attrition":       rng.choice(["Yes", "No"], p=[0.16, 0.84]),
            "Age":             int(rng.integers(22, 58)),
            "MonthlyIncome":   int(rng.integers(3000, 20001)),
        })

    df = pd.DataFrame(rows)
    df["Attrition_flag"] = (df["Attrition"] == "Yes").astype(int)
    print(f"✅ สร้างข้อมูลจำลอง: {len(df)} พนักงาน")
    return df


# ─── Edge Weight Scoring ────────────────────────────────────────────────────────
def _score_department(r1: dict, r2: dict) -> float:
    """แผนกเดียวกัน = 1.0 ต่างแผนก = 0.0"""
    return 1.0 if r1["Department"] == r2["Department"] else 0.0


def _score_job_level(r1: dict, r2: dict) -> float:
    """ระดับงานใกล้กัน = score สูง ห่างกัน 1 ระดับ = 0.7, 2 ระดับ = 0.4, 3+ = 0.1"""
    diff = abs(r1["JobLevel"] - r2["JobLevel"])
    return max(0.0, 1.0 - diff * 0.30)


def _score_job_role(r1: dict, r2: dict) -> float:
    """ตำแหน่งเดียวกัน = 1.0 ต่างตำแหน่ง = 0.0"""
    return 1.0 if r1["JobRole"] == r2["JobRole"] else 0.0


def _score_tenure(r1: dict, r2: dict) -> float:
    """อายุงานใกล้กัน = score สูง ต่างกัน 10 ปี = 0.0"""
    diff = abs(r1["YearsAtCompany"] - r2["YearsAtCompany"])
    return max(0.0, 1.0 - diff * 0.10)


def compute_edge_weight(r1: dict, r2: dict, weights: dict = None) -> float:
    """
    คำนวณน้ำหนักความสัมพันธ์ระหว่างพนักงาน 2 คน
    คืนค่าระหว่าง 0.0 (ไม่มีความสัมพันธ์) ถึง 1.0 (สัมพันธ์มากที่สุด)
    """
    w = weights or DEFAULT_WEIGHTS
    total = sum(w.values())
    if total == 0:
        return 0.0

    score = (
        w["department"] * _score_department(r1, r2)
        + w["job_level"]  * _score_job_level(r1, r2)
        + w["job_role"]   * _score_job_role(r1, r2)
        + w["tenure"]     * _score_tenure(r1, r2)
    ) / total

    return round(score, 4)


# ─── Graph Builder ──────────────────────────────────────────────────────────────
def build_graph(
    df: pd.DataFrame,
    weights: dict = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> nx.Graph:
    """
    สร้าง NetworkX Graph จาก DataFrame
    - weights: น้ำหนักแต่ละ feature (ถ้าไม่ส่งจะใช้ DEFAULT_WEIGHTS)
    - threshold: ค่าขั้นต่ำ edge weight ที่จะสร้าง edge
    """
    w = weights or DEFAULT_WEIGHTS
    G = nx.Graph()

    # เพิ่ม node พร้อม attributes
    for _, row in df.iterrows():
        G.add_node(
            int(row["EmployeeNumber"]),
            department    = row["Department"],
            job_role      = row["JobRole"],
            job_level     = int(row["JobLevel"]),
            years_company = int(row["YearsAtCompany"]),
            satisfaction  = int(row["JobSatisfaction"]),
            attrition     = int(row["Attrition_flag"]),
            age           = int(row["Age"]),
            income        = int(row["MonthlyIncome"]),
        )

    # สร้าง edge โดยเปรียบเทียบพนักงานทุกคู่
    records = df.set_index("EmployeeNumber").to_dict("index")
    emp_ids = list(records.keys())
    edge_count = 0

    for id1, id2 in combinations(emp_ids, 2):
        weight = compute_edge_weight(records[id1], records[id2], w)
        if weight >= threshold:
            G.add_edge(int(id1), int(id2), weight=weight)
            edge_count += 1

    avg_deg = sum(dict(G.degree()).values()) / max(G.number_of_nodes(), 1)
    print(f"✅ Graph: {G.number_of_nodes()} nodes | {edge_count} edges | avg degree: {avg_deg:.1f}")
    return G


# ─── Graph Summary ──────────────────────────────────────────────────────────────
def graph_summary(G: nx.Graph) -> dict:
    """สรุปข้อมูล graph เบื้องต้น"""
    if G.number_of_nodes() == 0:
        return {}

    degrees = dict(G.degree())
    top5    = sorted(degrees, key=degrees.get, reverse=True)[:5]

    summary = {
        "nodes":          G.number_of_nodes(),
        "edges":          G.number_of_edges(),
        "avg_degree":     round(sum(degrees.values()) / len(degrees), 2),
        "density":        round(nx.density(G), 4),
        "is_connected":   nx.is_connected(G),
        "components":     nx.number_connected_components(G),
        "top5_by_degree": top5,
    }

    print("\n── Graph Summary ──────────────────")
    for k, v in summary.items():
        print(f"  {k:20s}: {v}")
    return summary


def export_graph(G: nx.Graph, out_path: str = "graph_data.json"):
    """Export graph เป็น JSON สำหรับใช้ใน frontend"""
    data = nx.node_link_data(G)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"✅ Export → {out_path}")


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "WA_Fn-UseC_-HR-Employee-Attrition.csv"

    try:
        df = load_dataset(path)
    except FileNotFoundError:
        print(f"⚠️  ไม่พบไฟล์ '{path}' — ใช้ข้อมูลจำลองแทน")
        df = load_default_data()

    G = build_graph(df)
    graph_summary(G)
    export_graph(G, "graph_data.json")
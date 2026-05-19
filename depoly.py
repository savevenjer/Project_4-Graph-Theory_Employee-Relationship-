"""
app.py
======
ONA Dashboard — Organizational Network Analysis
รัน: streamlit run app.py
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import networkx as nx
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import community as community_louvain
import math
import io
from itertools import combinations
from io import StringIO

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ONA Dashboard",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Thai:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] { font-family: 'IBM Plex Sans Thai', sans-serif; }
    .block-container { padding: 1.5rem 2rem; }

    .metric-card {
        background: #1a1f2e;
        border: 1px solid #2d3548;
        border-radius: 12px;
        padding: 1.1rem 1.3rem;
        text-align: center;
        transition: border-color 0.2s;
    }
    .metric-card:hover { border-color: #4a5568; }
    .metric-label { color: #8892a4; font-size: 0.75rem; margin-bottom: 4px; letter-spacing: 0.06em; text-transform: uppercase; }
    .metric-value { color: #e8eaf0; font-size: 1.9rem; font-weight: 600; line-height: 1.1; font-family: 'JetBrains Mono', monospace; }
    .metric-sub   { color: #5d6a80; font-size: 0.70rem; margin-top: 4px; }

    .risk-high { color: #ef4444; }
    .risk-mid  { color: #f59e0b; }
    .risk-low  { color: #22c55e; }

    .rec-card {
        background: #1a1f2e;
        border-left: 4px solid #4a5568;
        border-radius: 0 10px 10px 0;
        padding: 1rem 1.2rem;
        margin-bottom: 10px;
    }
    .rec-card.critical { border-left-color: #ef4444; }
    .rec-card.high     { border-left-color: #f59e0b; }
    .rec-card.medium   { border-left-color: #3b82f6; }
    .rec-card.low      { border-left-color: #22c55e; }

    .onboard-step {
        background: #1a1f2e;
        border-radius: 10px;
        padding: .75rem 1rem;
        margin-bottom: 6px;
        font-size: 0.9rem;
        color: #c8d0e0;
        display: flex;
        gap: 10px;
        align-items: flex-start;
    }
    .step-num {
        background: #7F77DD;
        color: white;
        border-radius: 50%;
        width: 22px;
        height: 22px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.75rem;
        font-weight: 600;
        flex-shrink: 0;
    }

    div[data-testid="stTabs"] button { font-size: 0.85rem; }
    .stAlert { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# ─── Constants ──────────────────────────────────────────────────────────────────
DEPT_COLORS = {
    "Sales":                  "#7F77DD",
    "Research & Development": "#1D9E75",
    "Human Resources":        "#D4537E",
}

DEFAULT_WEIGHTS = {
    "department": 0.50,
    "job_level":  0.25,
    "job_role":   0.15,
    "tenure":     0.10,
}


# ─── Edge Weight ────────────────────────────────────────────────────────────────
def compute_edge_weight(r1: dict, r2: dict, weights: dict) -> float:
    total = sum(weights.values()) or 1.0

    s_dept   = 1.0 if r1["Department"] == r2["Department"] else 0.0
    s_role   = 1.0 if r1["JobRole"]    == r2["JobRole"]    else 0.0
    s_level  = max(0.0, 1.0 - abs(r1["JobLevel"]      - r2["JobLevel"])      * 0.30)
    s_tenure = max(0.0, 1.0 - abs(r1["YearsAtCompany"]- r2["YearsAtCompany"])* 0.10)

    return round((
        weights["department"] * s_dept +
        weights["job_level"]  * s_level +
        weights["job_role"]   * s_role +
        weights["tenure"]     * s_tenure
    ) / total, 4)


# ─── Data Loaders ────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="กำลังสร้างข้อมูลตัวอย่าง...")
def load_default_data() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 150
    depts = ["Sales", "Research & Development", "Human Resources"]
    roles_by_dept = {
        "Sales":                  ["Sales Executive", "Sales Representative", "Manager"],
        "Research & Development": ["Research Scientist", "Laboratory Technician", "Manager", "Developer"],
        "Human Resources":        ["HR Representative", "HR Manager", "Recruiter"],
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
    return df


# ─── Graph & Metrics ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="กำลังสร้าง Network Graph...")
def build_graph_cached(df_json: str, threshold: float, weights: dict):
    df = pd.read_json(StringIO(df_json))
    G = nx.Graph()

    for _, row in df.iterrows():
        G.add_node(int(row["EmployeeNumber"]),
                   department    = row["Department"],
                   job_role      = row["JobRole"],
                   job_level     = int(row["JobLevel"]),
                   years_company = int(row["YearsAtCompany"]),
                   satisfaction  = int(row["JobSatisfaction"]),
                   attrition     = int(row["Attrition_flag"]),
                   age           = int(row["Age"]),
                   income        = int(row["MonthlyIncome"]))

    records = df.set_index("EmployeeNumber").to_dict("index")
    for id1, id2 in combinations(list(records.keys()), 2):
        w = compute_edge_weight(records[id1], records[id2], weights)
        if w >= threshold:
            G.add_edge(int(id1), int(id2), weight=w)
    return G


@st.cache_data(show_spinner="กำลังคำนวณ Metrics...")
def compute_metrics(df_json: str, threshold: float, weights: dict):
    df  = pd.read_json(StringIO(df_json))
    G   = build_graph_cached(df_json, threshold, weights)

    deg = nx.degree_centrality(G)
    btw = nx.betweenness_centrality(G, normalized=True)
    pgr = nx.pagerank(G, alpha=0.85) if G.number_of_edges() > 0 else {n: 0 for n in G.nodes()}
    try:
        clu = nx.clustering(G)
    except Exception:
        clu = {node: 0.0 for node in G.nodes()}

    rows = []
    for node in G.nodes():
        attr    = G.nodes[node]
        sat_n   = (attr["satisfaction"] - 1) / 3
        score   = round(
            0.35 * btw.get(node, 0)
            + 0.25 * pgr.get(node, 0) * 10
            + 0.20 * deg.get(node, 0)
            + 0.20 * (1 - sat_n), 4)
        rows.append({
            "EmployeeNumber":     node,
            "Department":         attr["department"],
            "JobRole":            attr["job_role"],
            "JobLevel":           attr["job_level"],
            "YearsAtCompany":     attr["years_company"],
            "Satisfaction":       attr["satisfaction"],
            "Attrition":          attr["attrition"],
            "Income":             attr["income"],
            "Degree":             round(deg.get(node, 0), 4),
            "Betweenness":        round(btw.get(node, 0), 4),
            "PageRank":           round(pgr.get(node, 0), 6),
            "Clustering":         round(clu.get(node, 0), 4),
            "OrgResilienceScore": score,
        })
    return pd.DataFrame(rows), G


# ─── Network Visualization ───────────────────────────────────────────────────────
def draw_network(G, metric_df, color_by="Department", size_by="Betweenness", highlight_node=None):
    if G.number_of_nodes() == 0:
        return go.Figure()

    pos = nx.spring_layout(G, seed=42, k=1.5 / math.sqrt(max(G.number_of_nodes(), 1)))

    # Edges
    ex, ey = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]; x1, y1 = pos[v]
        ex += [x0, x1, None]; ey += [y0, y1, None]
    edge_trace = go.Scatter(x=ex, y=ey, mode="lines",
                            line=dict(width=0.4, color="#2d3548"), hoverinfo="none")

    # Nodes
    nx_, ny_, colors, sizes, texts, hovers = [], [], [], [], [], []
    m = metric_df.set_index("EmployeeNumber")
    max_s = m[size_by].max() or 1

    for node in G.nodes():
        if node not in m.index:
            continue
        x, y  = pos[node]
        row   = m.loc[node]
        dept  = row["Department"]

        nx_.append(x); ny_.append(y)
        texts.append(str(node))

        # color
        if color_by == "Department":
            colors.append(DEPT_COLORS.get(dept, "#8892a4"))
        elif color_by == "Attrition":
            colors.append("#ef4444" if row["Attrition"] == 1 else "#22c55e")
        else:  # OrgResilienceScore
            s = row["OrgResilienceScore"]
            colors.append(f"rgb({int(255*s)},{int(255*(1-s))},80)")

        # size
        raw = row[size_by]
        sz  = 10 + (raw / max_s) * 30
        sizes.append(sz * 1.8 if highlight_node and node == highlight_node else sz)

        hovers.append(
            f"<b>Employee #{node}</b><br>"
            f"แผนก: {dept}<br>ตำแหน่ง: {row['JobRole']}<br>"
            f"Level: {row['JobLevel']} | อายุงาน: {row['YearsAtCompany']} ปี<br>"
            f"Betweenness: {row['Betweenness']:.3f}<br>"
            f"Org Resilience Score: {row['OrgResilienceScore']:.3f}<br>"
            f"ความเสี่ยงลาออก: {'⚠️ ใช่' if row['Attrition']==1 else '✅ ไม่'}"
        )

    node_trace = go.Scatter(
        x=nx_, y=ny_, mode="markers+text",
        text=texts, textposition="top center",
        textfont=dict(size=7, color="#8892a4"),
        marker=dict(size=sizes, color=colors, line=dict(width=1, color="#0e1117")),
        hovertext=hovers, hoverinfo="text")

    return go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
            margin=dict(l=10, r=10, t=10, b=10),
            showlegend=False, hovermode="closest", height=520,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        ))


# ─── Simulation ──────────────────────────────────────────────────────────────────
def run_simulation(G, metric_df, remove_node: int) -> dict:
    G2 = G.copy()
    G2.remove_node(remove_node)

    btw_before = nx.betweenness_centrality(G,  normalized=True)
    btw_after  = nx.betweenness_centrality(G2, normalized=True)

    affected = list(G.neighbors(remove_node))
    changes  = [{"node": n,
                 "dept": G.nodes[n]["department"],
                 "role": G.nodes[n]["job_role"],
                 "btw_delta": round(btw_after.get(n,0) - btw_before.get(n,0), 4)}
                for n in affected]

    return {
        "lost_edges":     G.number_of_edges() - G2.number_of_edges(),
        "affected_count": len(affected),
        "frag_increase":  nx.number_connected_components(G2) - nx.number_connected_components(G),
        "components_after": nx.number_connected_components(G2),
        "changes_df":     pd.DataFrame(changes).sort_values("btw_delta", ascending=False),
    }


# ─── Business Recommendation ─────────────────────────────────────────────────────
def get_recommendation(row: pd.Series) -> dict:
    score = row["OrgResilienceScore"]
    attrition = row["Attrition"] == 1

    if score >= 0.60:
        return {
            "level": "วิกฤต", "css": "critical", "icon": "🔴",
            "actions": [
                "ทำ Succession Plan ทันที — เตรียมผู้สืบทอดตำแหน่ง",
                "จัด Knowledge Transfer ถ่ายทอดความรู้ให้ผู้ใต้บังคับบัญชา",
                "ติดตาม Job Satisfaction ทุกเดือน",
                "พิจารณาปรับค่าตอบแทนและ career path",
            ]
        }
    elif score >= 0.40:
        return {
            "level": "สูง", "css": "high", "icon": "🟡",
            "actions": [
                "ทำ Succession Plan ใน 3 เดือน",
                "ติดตาม Job Satisfaction ทุกไตรมาส",
                "พิจารณาปรับ workload ให้สมดุล",
            ]
        }
    elif score >= 0.25:
        return {
            "level": "ปานกลาง", "css": "medium", "icon": "🔵",
            "actions": [
                "ติดตามเป็นระยะ ทุก 6 เดือน",
                "สนับสนุน training และ upskilling",
            ]
        }
    else:
        return {
            "level": "ต่ำ", "css": "low", "icon": "🟢",
            "actions": ["ไม่จำเป็นต้องดำเนินการเร่งด่วน"]
        }


# ─── Export PDF ──────────────────────────────────────────────────────────────────
def export_pdf(metric_df: pd.DataFrame) -> bytes:
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm

        buffer = io.BytesIO()
        doc    = SimpleDocTemplate(buffer, pagesize=A4,
                                   leftMargin=2*cm, rightMargin=2*cm,
                                   topMargin=2*cm, bottomMargin=2*cm)
        styles = getSampleStyleSheet()
        story  = []

        # Title
        title_style = ParagraphStyle("title", parent=styles["Heading1"],
                                     fontSize=16, spaceAfter=6)
        story.append(Paragraph("ONA Report — Organizational Risk Assessment", title_style))
        story.append(Paragraph("Organizational Network Analysis | IBM HR Dataset", styles["Normal"]))
        story.append(Spacer(1, 0.5*cm))

        # Summary
        high_risk = len(metric_df[metric_df["OrgResilienceScore"] >= 0.40])
        story.append(Paragraph(f"พนักงานทั้งหมด: {len(metric_df)} คน | ความเสี่ยงสูง: {high_risk} คน", styles["Normal"]))
        story.append(Spacer(1, 0.4*cm))

        # Table
        story.append(Paragraph("Top 20 — Org Resilience Score สูงสุด", styles["Heading2"]))
        story.append(Spacer(1, 0.2*cm))

        top20 = metric_df.nlargest(20, "OrgResilienceScore")[
            ["EmployeeNumber","Department","JobRole","JobLevel","OrgResilienceScore","Betweenness"]
        ]

        data = [["#", "แผนก", "ตำแหน่ง", "Level", "Score", "Betweenness"]]
        for _, r in top20.iterrows():
            data.append([
                str(int(r["EmployeeNumber"])),
                r["Department"][:20],
                r["JobRole"][:20],
                str(int(r["JobLevel"])),
                f"{r['OrgResilienceScore']:.3f}",
                f"{r['Betweenness']:.3f}",
            ])

        tbl = Table(data, colWidths=[1.5*cm, 4.5*cm, 4.5*cm, 1.5*cm, 2*cm, 2.5*cm])
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1a1f2e")),
            ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
            ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE",   (0,0), (-1,-1), 8),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f8f9fa")]),
            ("GRID",       (0,0), (-1,-1), 0.5, colors.HexColor("#dee2e6")),
            ("ALIGN",      (0,0), (-1,-1), "CENTER"),
            ("VALIGN",     (0,0), (-1,-1), "MIDDLE"),
            ("ROWHEIGHT",  (0,0), (-1,-1), 18),
        ]))
        story.append(tbl)

        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()

    except ImportError:
        # ถ้าไม่มี reportlab ให้ export เป็น CSV แทน
        csv_buf = io.StringIO()
        metric_df.nlargest(20, "OrgResilienceScore").to_csv(csv_buf, index=False)
        return csv_buf.getvalue().encode("utf-8")


# ─── Main App ────────────────────────────────────────────────────────────────────
def main():
    # ── Sidebar ──
    with st.sidebar:
        st.header("⚙️ ตั้งค่า Network")

        uploaded_file = st.file_uploader("📂 อัปโหลดไฟล์ HR Data (CSV)", type="csv",
                                         help="รองรับ IBM HR Analytics Dataset หรือ CSV ที่มี column ครบ")
        st.caption("ถ้าไม่อัปโหลด จะใช้ข้อมูลตัวอย่าง 150 คน")
        st.divider()

        st.markdown("**🎚️ ปรับน้ำหนักความสัมพันธ์**")
        w_dept   = st.slider("แผนกเดียวกัน",      0.0, 1.0, 0.50, 0.05)
        w_level  = st.slider("ระดับงานใกล้กัน",   0.0, 1.0, 0.25, 0.05)
        w_role   = st.slider("ตำแหน่งเดียวกัน",   0.0, 1.0, 0.15, 0.05)
        w_tenure = st.slider("อายุงานใกล้เคียง",  0.0, 1.0, 0.10, 0.05)

        st.divider()
        threshold = st.slider("🔗 ความหนาแน่น Network", 0.10, 0.90, 0.40, 0.05,
                              help="ยิ่งสูง = edge น้อยลง / เห็นเฉพาะความสัมพันธ์แน่นๆ")

        WEIGHTS = {"department": w_dept, "job_level": w_level,
                   "job_role": w_role, "tenure": w_tenure}

    # ── Load Data ──
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if "Attrition_flag" not in df.columns and "Attrition" in df.columns:
            df["Attrition_flag"] = (df["Attrition"] == "Yes").astype(int)
    else:
        df = load_default_data()

    df_json   = df.to_json()
    metric_df, G = compute_metrics(df_json, threshold, WEIGHTS)

    # ── Header ──
    st.markdown("## 🕸️ Organizational Network Analysis")
    st.caption("IBM HR Analytics · Graph Theory · Risk Assessment")

    # ── Onboarding Guide ──
    with st.expander("📖 วิธีใช้งาน Dashboard — คลิกอ่าน", expanded=False):
        st.markdown("""
        <div class="onboard-step">
            <div class="step-num">1</div>
            <div><b>อัปโหลดข้อมูล</b> — วาง CSV ที่ sidebar ซ้าย หรือใช้ข้อมูลตัวอย่างที่โหลดให้อัตโนมัติ</div>
        </div>
        <div class="onboard-step">
            <div class="step-num">2</div>
            <div><b>ปรับน้ำหนัก</b> — เลื่อน slider เพื่อกำหนดว่าจะให้แผนก / ระดับงาน / ตำแหน่ง มีผลมากน้อยแค่ไหน</div>
        </div>
        <div class="onboard-step">
            <div class="step-num">3</div>
            <div><b>ดู Network</b> — Tab "ภาพรวม Network" แสดงความสัมพันธ์ทั้งหมด คลิก hover ที่ node เพื่อดูรายละเอียด</div>
        </div>
        <div class="onboard-step">
            <div class="step-num">4</div>
            <div><b>จำลองการลาออก</b> — Tab "What-if Simulation" เลือกพนักงานแล้วกด Run เพื่อดูว่าถ้าคนนี้ลาออกจะกระทบใครบ้าง</div>
        </div>
        <div class="onboard-step">
            <div class="step-num">5</div>
            <div><b>ดูคำแนะนำ</b> — Tab "💡 Recommendations" สรุปว่า HR ควรทำอะไรกับพนักงานแต่ละกลุ่ม</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── Summary Metrics ──
    high_risk  = metric_df[metric_df["OrgResilienceScore"] > 0.50]
    key_person = metric_df.loc[metric_df["Betweenness"].idxmax(), "EmployeeNumber"] if not metric_df.empty else "N/A"

    c1, c2, c3, c4 = st.columns(4)
    for col, label, value, sub in [
        (c1, "พนักงานทั้งหมด", f"{G.number_of_nodes():,}", f"{metric_df['Department'].nunique()} แผนก"),
        (c2, "ความสัมพันธ์ (Edges)", f"{G.number_of_edges():,}", f"threshold {threshold}"),
        (c3, "Key Person", f"#{key_person}", "Betweenness สูงสุด"),
        (c4, "ความเสี่ยงสูง", f"{len(high_risk)}", "Resilience > 0.5"),
    ]:
        with col:
            risk_class = "risk-high" if label == "ความเสี่ยงสูง" else ""
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value {risk_class}">{value}</div>
                <div class="metric-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabs ──
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🌐 ภาพรวม Network",
        "📊 Centrality Analysis",
        "🏘️ Community Detection",
        "⚡ What-if Simulation",
        "🏢 Department Health",
        "💡 Recommendations",
    ])

    # ── Tab 1: Network Overview ──
    with tab1:
        col_ctrl, col_graph = st.columns([1, 3])
        with col_ctrl:
            st.markdown("**ตั้งค่าการแสดงผล**")
            color_by  = st.selectbox("สีตาม",   ["Department", "Attrition", "OrgResilienceScore"])
            size_by   = st.selectbox("ขนาดตาม", ["Betweenness", "Degree", "PageRank", "OrgResilienceScore"])
            max_nodes = G.number_of_nodes()
            show_n    = st.slider("จำนวน node", min_value=min(10, max_nodes),
                                  max_value=max_nodes, value=min(150, max_nodes), step=10)
            st.markdown("---")
            st.markdown("**Legend**")
            for dept, color in DEPT_COLORS.items():
                st.markdown(f"<span style='color:{color}'>●</span> {dept}", unsafe_allow_html=True)
            st.markdown("<span style='color:#8892a4'>วงกลมใหญ่ = score สูง</span>", unsafe_allow_html=True)

        with col_graph:
            G_sub = G.subgraph(list(G.nodes())[:show_n])
            m_sub = metric_df[metric_df["EmployeeNumber"].isin(list(G.nodes())[:show_n])]
            st.plotly_chart(draw_network(G_sub, m_sub, color_by, size_by), use_container_width=True)

    # ── Tab 2: Centrality Analysis ──
    with tab2:
        st.markdown("### Top 20 — Betweenness Centrality")
        st.caption("คนที่เป็น 'สะพาน' เชื่อมระหว่างแผนก — ถ้าลาออกการสื่อสารขาด")

        top20 = metric_df.nlargest(20, "Betweenness")
        fig_b = px.bar(top20, x="EmployeeNumber", y="Betweenness",
                       color="Department", color_discrete_map=DEPT_COLORS,
                       hover_data=["JobRole", "JobLevel", "OrgResilienceScore"],
                       template="plotly_dark")
        fig_b.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", height=320)
        st.plotly_chart(fig_b, use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            fig_p = px.bar(metric_df.nlargest(15, "PageRank"), x="EmployeeNumber", y="PageRank",
                           color="Department", color_discrete_map=DEPT_COLORS,
                           title="Top 15 — PageRank", template="plotly_dark")
            fig_p.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", height=280)
            st.plotly_chart(fig_p, use_container_width=True)
        with col_b:
            fig_d = px.bar(metric_df.nlargest(15, "Degree"), x="EmployeeNumber", y="Degree",
                           color="Department", color_discrete_map=DEPT_COLORS,
                           title="Top 15 — Degree Centrality", template="plotly_dark")
            fig_d.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", height=280)
            st.plotly_chart(fig_d, use_container_width=True)

        st.markdown("### Org Resilience Score — Top 30")
        st.caption("สูตร: 0.35×Betweenness + 0.25×PageRank×10 + 0.20×Degree + 0.20×(1-Satisfaction)")
        fig_s = px.scatter(metric_df.nlargest(30, "OrgResilienceScore"),
                           x="Betweenness", y="OrgResilienceScore",
                           size="Degree", color="Department", color_discrete_map=DEPT_COLORS,
                           hover_data=["EmployeeNumber", "JobRole", "Attrition"],
                           template="plotly_dark", height=360)
        fig_s.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117")
        st.plotly_chart(fig_s, use_container_width=True)

    # ── Tab 3: Community Detection ──
    with tab3:
        st.markdown("### Community Detection — Louvain Algorithm")
        st.caption("กลุ่มที่ทำงานด้วยกันจริงๆ อาจต่างจาก org chart")

        if G.number_of_edges() > 0:
            partition = community_louvain.best_partition(G)
            comm_s    = pd.Series(partition).reset_index()
            comm_s.columns = ["EmployeeNumber", "Community"]
            mwc = metric_df.merge(comm_s, on="EmployeeNumber")

            st.info(f"พบ **{mwc['Community'].nunique()} communities** จาก Louvain Algorithm")

            c1, c2 = st.columns([2, 1])
            with c1:
                cnt = mwc.groupby(["Community", "Department"]).size().reset_index(name="Count")
                fig_c = px.bar(cnt, x="Community", y="Count", color="Department",
                               color_discrete_map=DEPT_COLORS, barmode="stack",
                               title="สมาชิกใน Community แต่ละกลุ่ม แยกตามแผนก",
                               template="plotly_dark")
                fig_c.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", height=360)
                st.plotly_chart(fig_c, use_container_width=True)
            with c2:
                st.markdown("**Community Summary**")
                cs = mwc.groupby("Community").agg(
                    สมาชิก=("EmployeeNumber","count"),
                    Avg_Score=("OrgResilienceScore","mean"),
                    Attrition=("Attrition","mean"),
                ).round(3).reset_index()
                cs["Attrition"] = cs["Attrition"].apply(lambda x: f"{x:.0%}")
                st.dataframe(cs, use_container_width=True, hide_index=True)
        else:
            st.warning("ไม่มี Edges เพียงพอ กรุณาปรับ Threshold ให้ต่ำลง")

    # ── Tab 4: What-if Simulation ──
    with tab4:
        st.markdown("### ⚡ What-if Simulation")
        st.markdown("จำลองว่า **ถ้าพนักงานคนนี้ลาออก** — องค์กรจะได้รับผลกระทบอย่างไร")

        col_s1, col_s2 = st.columns([1, 2])
        with col_s1:
            top_risk = metric_df.nlargest(30, "OrgResilienceScore")[
                ["EmployeeNumber","Department","JobRole","OrgResilienceScore"]
            ].reset_index(drop=True)

            selected_id = st.selectbox(
                "เลือกพนักงาน (เรียงตาม Score สูงสุด)",
                options=top_risk["EmployeeNumber"].tolist(),
                format_func=lambda x: (
                    f"#{x} — "
                    f"{top_risk[top_risk.EmployeeNumber==x]['JobRole'].values[0]} | "
                    f"{top_risk[top_risk.EmployeeNumber==x]['Department'].values[0]}"
                )
            )

            sr = metric_df[metric_df["EmployeeNumber"] == selected_id].iloc[0]
            st.markdown("---")
            st.markdown("**ข้อมูลพนักงาน**")
            st.markdown(f"- แผนก: **{sr['Department']}**")
            st.markdown(f"- ตำแหน่ง: **{sr['JobRole']}** (Level {sr['JobLevel']})")
            st.markdown(f"- อายุงาน: **{sr['YearsAtCompany']} ปี**")
            st.markdown(f"- Betweenness: **{sr['Betweenness']:.4f}**")
            st.markdown(f"- Org Resilience Score: **{sr['OrgResilienceScore']:.4f}**")

            rec = get_recommendation(sr)
            st.markdown(f"- ระดับความเสี่ยง: **{rec['icon']} {rec['level']}**")
            st.markdown("")
            run_btn = st.button("▶ Run Simulation", type="primary", use_container_width=True)

        with col_s2:
            if run_btn:
                with st.spinner("กำลังจำลอง..."):
                    result = run_simulation(G, metric_df, selected_id)

                r1, r2, r3, r4 = st.columns(4)
                r1.metric("Edges สูญเสีย",       f"{result['lost_edges']}")
                r2.metric("พนักงานที่ได้รับผล",  f"{result['affected_count']} คน")
                r3.metric("Network แตกเพิ่ม",    f"{result['frag_increase']} cluster")
                r4.metric("Components หลัง",     f"{result['components_after']}")

                st.markdown("---")
                if result["frag_increase"] > 0:
                    st.error(f"⚠️ **Critical** — Network แตกเป็น {result['components_after']} ส่วน การสื่อสารข้ามกลุ่มหยุดชะงักทันที")
                elif result["lost_edges"] > 10:
                    st.warning(f"🔶 **High Risk** — สูญเสีย {result['lost_edges']} connections กระทบ {result['affected_count']} คน")
                else:
                    st.success("✅ **Low Risk** — ผลกระทบอยู่ในระดับที่รับมือได้")

                if not result["changes_df"].empty:
                    st.markdown("#### พนักงานที่ได้รับผลกระทบมากที่สุด")
                    st.dataframe(result["changes_df"].head(10), use_container_width=True, hide_index=True)

                # Network Before/After
                st.markdown("#### Network ก่อน vs หลัง")
                nodes_viz = list(G.neighbors(selected_id)) + [selected_id]
                G_before  = G.subgraph(nodes_viz[:60])
                G_after   = G.copy(); G_after.remove_node(selected_id)
                nbrs_after = [n for n in nodes_viz if n != selected_id]
                G_after_s = G_after.subgraph(nbrs_after[:60])

                vc1, vc2 = st.columns(2)
                with vc1:
                    st.caption("ก่อน")
                    fig_bef = draw_network(G_before,
                                           metric_df[metric_df["EmployeeNumber"].isin(nodes_viz)],
                                           highlight_node=selected_id)
                    fig_bef.update_layout(height=300)
                    st.plotly_chart(fig_bef, use_container_width=True, key="sim_b")
                with vc2:
                    st.caption("หลัง")
                    if nbrs_after:
                        fig_aft = draw_network(G_after_s,
                                               metric_df[metric_df["EmployeeNumber"].isin(nbrs_after)])
                        fig_aft.update_layout(height=300)
                        st.plotly_chart(fig_aft, use_container_width=True, key="sim_a")
                    else:
                        st.info("ไม่มี node เหลืออยู่ในกลุ่มนี้")
            else:
                st.info("เลือกพนักงานแล้วกด **▶ Run Simulation** เพื่อดูผลกระทบ")

    # ── Tab 5: Department Health ──
    with tab5:
        st.markdown("### 🏢 Department Health Score")
        st.caption("ประเมินสุขภาพของแต่ละแผนกจาก network metrics")

        dept_h = metric_df.groupby("Department").agg(
            Members         = ("EmployeeNumber", "count"),
            Avg_Betweenness = ("Betweenness",    "mean"),
            Avg_Clustering  = ("Clustering",     "mean"),
            Avg_Satisfaction= ("Satisfaction",   "mean"),
            Attrition_Rate  = ("Attrition",      "mean"),
            Avg_Resilience  = ("OrgResilienceScore","mean"),
        ).round(4).reset_index()

        dept_h["HealthScore"] = (
            dept_h["Avg_Satisfaction"] / 4 * 0.40
            + dept_h["Avg_Clustering"]         * 0.30
            - dept_h["Attrition_Rate"]         * 0.20
            - dept_h["Avg_Resilience"]         * 0.10
        ).round(4)

        c1, c2 = st.columns([1, 2])
        with c1:
            disp = dept_h[["Department","Members","HealthScore","Attrition_Rate","Avg_Satisfaction"]].copy()
            disp["Attrition_Rate"]  = disp["Attrition_Rate"].apply(lambda x: f"{x:.1%}")
            disp["HealthScore"]     = disp["HealthScore"].apply(lambda x: f"{x:.3f}")
            disp["Avg_Satisfaction"]= disp["Avg_Satisfaction"].apply(lambda x: f"{x:.2f}/4")
            st.dataframe(disp, use_container_width=True, hide_index=True)

        with c2:
            fig_h = px.bar(dept_h, x="Department", y="HealthScore",
                           color="Department", color_discrete_map=DEPT_COLORS,
                           title="Department Health Score (สูง = สุขภาพดี)",
                           template="plotly_dark", text="HealthScore")
            fig_h.update_traces(texttemplate="%{text:.3f}", textposition="outside")
            fig_h.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                                showlegend=False, height=300)
            st.plotly_chart(fig_h, use_container_width=True)

        # Radar Chart
        st.markdown("### Radar — เปรียบเทียบแผนก")
        cats = ["Avg_Betweenness","Avg_Clustering","Avg_Satisfaction","HealthScore"]
        fig_r = go.Figure()
        for _, row in dept_h.iterrows():
            vals = [row[c] for c in cats]
            fig_r.add_trace(go.Scatterpolar(
                r=vals + [vals[0]], theta=cats + [cats[0]],
                fill="toself", name=row["Department"],
                line_color=DEPT_COLORS.get(row["Department"], "#8892a4")
            ))
        fig_r.update_layout(
            polar=dict(bgcolor="#1a1f2e",
                       radialaxis=dict(visible=True, color="#8892a4"),
                       angularaxis=dict(color="#8892a4")),
            paper_bgcolor="#0e1117", template="plotly_dark",
            legend=dict(bgcolor="#1a1f2e"), height=400)
        st.plotly_chart(fig_r, use_container_width=True)

        # Formal vs Informal
        st.markdown("---")
        st.markdown("### Formal vs Informal Network")
        st.caption("เปรียบเทียบ org chart (Formal) กับ community ที่เกิดขึ้นจริง (Informal)")
        if G.number_of_edges() > 0:
            partition = community_louvain.best_partition(G)
            comm_s = pd.Series(partition).reset_index()
            comm_s.columns = ["EmployeeNumber", "Community"]
            merged = metric_df.merge(comm_s, on="EmployeeNumber")
            cross  = pd.crosstab(merged["Department"], merged["Community"])
            fig_hm = px.imshow(cross, color_continuous_scale="Blues",
                               title="Heatmap: Department (Formal) vs Community (Informal)",
                               template="plotly_dark")
            fig_hm.update_layout(paper_bgcolor="#0e1117", height=320)
            st.plotly_chart(fig_hm, use_container_width=True)
        else:
            st.warning("ไม่มี Edges เพียงพอ กรุณาปรับ Threshold ให้ต่ำลง")

    # ── Tab 6: Recommendations ──
    with tab6:
        st.markdown("### 💡 Business Recommendation")
        st.caption("คำแนะนำสำหรับ HR จากผลการวิเคราะห์ — เรียงตามความเสี่ยง")

        top_rec = metric_df.nlargest(10, "OrgResilienceScore")

        for _, row in top_rec.iterrows():
            rec = get_recommendation(row)
            actions_html = "".join([f"<li>{a}</li>" for a in rec["actions"]])
            st.markdown(f"""
            <div class="rec-card {rec['css']}">
                <b>{rec['icon']} Employee #{int(row['EmployeeNumber'])}</b>
                &nbsp;—&nbsp; {row['JobRole']} &nbsp;·&nbsp; {row['Department']}<br>
                <small style="color:#8892a4">
                    Level {int(row['JobLevel'])} &nbsp;|&nbsp;
                    อายุงาน {int(row['YearsAtCompany'])} ปี &nbsp;|&nbsp;
                    Score: <b>{row['OrgResilienceScore']:.3f}</b> &nbsp;|&nbsp;
                    ระดับ: <b>{rec['level']}</b>
                </small>
                <ul style="margin: 8px 0 0 16px; color: #c8d0e0; font-size: 0.87rem;">
                    {actions_html}
                </ul>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Export
        st.markdown("### 📥 Export รายงาน")
        col_e1, col_e2 = st.columns(2)

        with col_e1:
            csv = metric_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📊 Export ข้อมูลทั้งหมด (.csv)",
                csv, "ONA_full_data.csv", "text/csv",
                use_container_width=True
            )

        with col_e2:
            if st.button("📄 Export รายงาน PDF", use_container_width=True):
                with st.spinner("กำลังสร้าง PDF..."):
                    pdf_data = export_pdf(metric_df)
                try:
                    st.download_button(
                        "⬇️ ดาวน์โหลด PDF",
                        pdf_data, "ONA_Report.pdf", "application/pdf",
                        use_container_width=True
                    )
                except Exception:
                    st.download_button(
                        "⬇️ ดาวน์โหลด CSV (fallback)",
                        pdf_data, "ONA_Report.csv", "text/csv",
                        use_container_width=True
                    )


if __name__ == "__main__":
    main()
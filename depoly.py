import streamlit as st
import pandas as pd
import networkx as nx
from io import StringIO
import plotly.graph_objects as go
import plotly.express as px
from itertools import combinations
import community as community_louvain
import math
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning, module="websockets")

st.set_page_config(
    page_title="ONA Dashboard",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .block-container { padding: 1.5rem 2rem; }
    .metric-card {
        background: #1a1f2e;
        border: 1px solid #2d3548;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        text-align: center;
    }
    .metric-label { color: #8892a4; font-size: 0.78rem; margin-bottom: 4px; letter-spacing: 0.05em; }
    .metric-value { color: #e8eaf0; font-size: 1.8rem; font-weight: 600; line-height: 1.1; }
    .metric-sub   { color: #5d6a80; font-size: 0.72rem; margin-top: 3px; }
    .risk-high { color: #ef4444; }
    .risk-mid  { color: #f59e0b; }
    .risk-low  { color: #22c55e; }
    div[data-testid="stTabs"] button { font-size: 0.85rem; }
    .stAlert { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

def compute_edge_weight(r1, r2, weights):
    total_weight = sum(weights.values())
    if total_weight == 0:
        return 0.0

    score_dept = 1.0 if r1["Department"] == r2["Department"] else 0.0
    score_role = 1.0 if r1["JobRole"] == r2["JobRole"] else 0.0
    score_level = math.exp(-abs(r1["JobLevel"] - r2["JobLevel"]))
    score_tenure = math.exp(-abs(r1["YearsAtCompany"] - r2["YearsAtCompany"]) * 0.1)

    final_score = (
        (score_dept * weights["department"]) +
        (score_level * weights["job_level"]) +
        (score_role * weights["job_role"]) +
        (score_tenure * weights["tenure"])
    ) / total_weight

    return round(final_score, 4)

@st.cache_data(show_spinner="กำลังโหลด dataset...")
def load_default_data():
    data = []
    depts = ["Sales", "Research & Development", "Human Resources"]
    roles = ["Manager", "Developer", "Analyst", "HR", "Executive"]
    for i in range(1, 151):
        data.append({
            "EmployeeNumber": i,
            "Department": depts[i % 3],
            "JobRole": roles[i % 5],
            "JobLevel": (i % 5) + 1,
            "YearsAtCompany": (i % 10) + 1,
            "JobSatisfaction": (i % 4) + 1,
            "Attrition": "Yes" if i % 15 == 0 else "No",
            "Age": 25 + (i % 20),
            "MonthlyIncome": 5000 + (i * 100)
        })
    df = pd.DataFrame(data)
    df["Attrition_flag"] = (df["Attrition"] == "Yes").astype(int)
    return df

@st.cache_data(show_spinner="กำลังสร้าง network graph... (ใช้เวลาสักครู่)")
def build_graph_cached(df_json, threshold, weights):
    df = pd.read_json(StringIO(df_json))
    G = nx.Graph()
    
    for _, row in df.iterrows():
        G.add_node(int(row["EmployeeNumber"]),
                   department=row["Department"],
                   job_role=row["JobRole"],
                   job_level=int(row["JobLevel"]),
                   years_company=int(row["YearsAtCompany"]),
                   satisfaction=int(row["JobSatisfaction"]),
                   attrition=int(row["Attrition_flag"]),
                   age=int(row["Age"]),
                   income=int(row["MonthlyIncome"]))
                   
    records = df.set_index("EmployeeNumber").to_dict("index")
    emp_ids = list(records.keys())
    
    for id1, id2 in combinations(emp_ids, 2):
        r1, r2 = records[id1], records[id2]
        w = compute_edge_weight(r1, r2, weights)
        if w >= threshold:
            G.add_edge(int(id1), int(id2), weight=w)
            
    return G

@st.cache_data(show_spinner="คำนวณ metrics...")
def compute_metrics(df_json, threshold, weights):
    df = pd.read_json(StringIO(df_json))
    G = build_graph_cached(df_json, threshold, weights)

    deg  = nx.degree_centrality(G)
    btw  = nx.betweenness_centrality(G, normalized=True)
    pgr  = nx.pagerank(G, alpha=0.85) if G.number_of_edges() > 0 else {n: 0 for n in G.nodes()}
    
    try:
        clu = nx.clustering(G)
    except:
        clu = {node: 0.0 for node in G.nodes()}

    metrics = []
    for node in G.nodes():
        attr = G.nodes[node]
        sat_norm = (attr["satisfaction"] - 1) / 3
        score = round(
            0.35 * btw.get(node, 0)
            + 0.25 * pgr.get(node, 0) * 10
            + 0.20 * deg.get(node, 0)
            + 0.20 * (1 - sat_norm), 4)
            
        metrics.append({
            "EmployeeNumber": node,
            "Department":     attr["department"],
            "JobRole":        attr["job_role"],
            "JobLevel":       attr["job_level"],
            "YearsAtCompany": attr["years_company"],
            "Satisfaction":   attr["satisfaction"],
            "Attrition":      attr["attrition"],
            "Income":         attr["income"],
            "Degree":         round(deg.get(node, 0), 4),
            "Betweenness":    round(btw.get(node, 0), 4),
            "PageRank":       round(pgr.get(node, 0), 6),
            "Clustering":     round(clu.get(node, 0), 4),
            "OrgResilienceScore": score,
        })
    return pd.DataFrame(metrics), G

DEPT_COLORS = {
    "Sales": "#7F77DD",
    "Research & Development": "#1D9E75",
    "Human Resources": "#D4537E",
}

def draw_network(G, metric_df, color_by="Department", size_by="Betweenness", highlight_node=None):
    if G.number_of_nodes() == 0:
        return go.Figure()
        
    pos = nx.spring_layout(G, seed=42, k=1.5/math.sqrt(G.number_of_nodes()) if G.number_of_nodes() > 0 else 1)

    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]; x1, y1 = pos[v]
        edge_x += [x0, x1, None]; edge_y += [y0, y1, None]

    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode="lines",
        line=dict(width=0.4, color="#2d3548"), hoverinfo="none")

    node_x, node_y, node_text, node_color, node_size, node_hover = [], [], [], [], [], []
    m = metric_df.set_index("EmployeeNumber")

    size_col = size_by 
    max_size = m[size_col].max() if m[size_col].max() > 0 else 1

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x); node_y.append(y)
        
        if node not in m.index:
            continue
            
        row = m.loc[node]
        dept = row["Department"]

        if color_by == "Department":
            node_color.append(DEPT_COLORS.get(dept, "#8892a4"))
        elif color_by == "Attrition":
            node_color.append("#ef4444" if row["Attrition"] == 1 else "#22c55e")
        elif color_by == "OrgResilienceScore":
            score = row["OrgResilienceScore"]
            node_color.append(f"rgb({int(255*score)}, {int(255*(1-score))}, 80)")

        raw = row[size_col]
        node_size.append(10 + (raw / max_size) * 30)

        if highlight_node and node == highlight_node:
            node_size[-1] = node_size[-1] * 1.8

        node_text.append(str(node))
        node_hover.append(
            f"<b>Employee #{node}</b><br>"
            f"Dept: {dept}<br>"
            f"Role: {row['JobRole']}<br>"
            f"Level: {row['JobLevel']} | Tenure: {row['YearsAtCompany']} yr<br>"
            f"Betweenness: {row['Betweenness']:.3f}<br>"
            f"PageRank: {row['PageRank']:.4f}<br>"
            f"Org Resilience Score: {row['OrgResilienceScore']:.3f}<br>"
            f"Attrition Risk: {'⚠️ Yes' if row['Attrition']==1 else '✅ No'}"
        )

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        text=node_text, textposition="top center",
        textfont=dict(size=7, color="#8892a4"),
        marker=dict(size=node_size, color=node_color, line=dict(width=1, color="#0e1117")),
        hovertext=node_hover, hoverinfo="text")

    fig = go.Figure(data=[edge_trace, node_trace],
        layout=go.Layout(
            paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
            margin=dict(l=10, r=10, t=10, b=10),
            showlegend=False, hovermode="closest",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=520,
        ))
    return fig

def run_simulation(G, metric_df, remove_node):
    G2 = G.copy()
    G2.remove_node(remove_node)

    before_comp = nx.number_connected_components(G)
    after_comp  = nx.number_connected_components(G2)
    before_edges = G.number_of_edges()
    after_edges  = G2.number_of_edges()
    lost_edges   = before_edges - after_edges
    frag_increase = after_comp - before_comp

    affected = list(G.neighbors(remove_node))

    btw_before = nx.betweenness_centrality(G, normalized=True)
    btw_after  = nx.betweenness_centrality(G2, normalized=True)

    changes = []
    for n in affected:
        delta = btw_after.get(n, 0) - btw_before.get(n, 0)
        changes.append({"node": n, "dept": G.nodes[n]["department"],
                        "role": G.nodes[n]["job_role"], "btw_delta": round(delta, 4)})
    changes_df = pd.DataFrame(changes).sort_values("btw_delta", ascending=False)

    return {
        "lost_edges": lost_edges,
        "affected_count": len(affected),
        "frag_increase": frag_increase,
        "components_after": after_comp,
        "changes_df": changes_df,
    }

def main():
    with st.sidebar:
        st.header("⚙️ Network Settings")
        uploaded_file = st.file_uploader("อัปโหลดไฟล์ HR Data (CSV)", type="csv")
        st.divider()
        st.markdown("**ปรับน้ำหนักความสำคัญ (Feature Weights)**")
        w_dept = st.slider("น้ำหนักความสัมพันธ์ภายในแผนก (Intra-dept)", 0.0, 1.0, 0.50)
        w_level = st.slider("อิทธิพลของลำดับชั้น (Hierarchy Influence)", 0.0, 1.0, 0.25)
        w_role = st.slider("ความเชื่อมโยงสายอาชีพ (Role Affinity)", 0.0, 1.0, 0.15)
        w_tenure = st.slider("ความผูกพันตามอายุงาน (Tenure Bonding)", 0.0, 1.0, 0.10)
        st.divider()
        EDGE_THRESHOLD = st.slider("ความเข้มข้นของเครือข่าย (Network Density)", 0.1, 0.9, 0.40)
        
        WEIGHTS = {"department": w_dept, "job_level": w_level, "job_role": w_role, "tenure": w_tenure}

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if "Attrition_flag" not in df.columns and "Attrition" in df.columns:
            df["Attrition_flag"] = (df["Attrition"] == "Yes").astype(int)
    else:
        df = load_default_data()

    df_json = df.to_json()
    metric_df, G = compute_metrics(df_json, EDGE_THRESHOLD, WEIGHTS)

    st.markdown("## 🕸️ Organizational Network Analysis")
    st.caption("IBM HR Analytics · Graph Theory · Risk Assessment")
    st.divider()

    high_risk = metric_df[metric_df["OrgResilienceScore"] > 0.5]
    key_person = metric_df.loc[metric_df["Betweenness"].idxmax(), "EmployeeNumber"] if not metric_df.empty else "N/A"
    dept_count = metric_df["Department"].nunique() if not metric_df.empty else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">พนักงานทั้งหมด</div>
            <div class="metric-value">{G.number_of_nodes():,}</div>
            <div class="metric-sub">{dept_count} departments</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">ความสัมพันธ์ (Edges)</div>
            <div class="metric-value">{G.number_of_edges():,}</div>
            <div class="metric-sub">threshold {EDGE_THRESHOLD}</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Key Person</div>
            <div class="metric-value">#{key_person}</div>
            <div class="metric-sub">Betweenness สูงสุด</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">ความเสี่ยงสูง</div>
            <div class="metric-value risk-high">{len(high_risk)}</div>
            <div class="metric-sub">Resilience Score > 0.5</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🌐 ภาพรวม Network",
        "📊 Betweenness & Centrality",
        "🏘️ Community Detection",
        "⚡ What-if Simulation",
        "🏢 Department Health",
    ])

    with tab1:
        col_ctrl, col_graph = st.columns([1, 3])
        with col_ctrl:
            st.markdown("**ตั้งค่าการแสดงผล**")
            color_by = st.selectbox("สีตาม", ["Department", "Attrition", "OrgResilienceScore"])
            size_by  = st.selectbox("ขนาดตาม", ["Betweenness", "Degree", "PageRank", "OrgResilienceScore"])
            max_nodes = G.number_of_nodes()
            show_n = st.slider("จำนวน node ที่แสดง", min_value=min(10, max_nodes), max_value=max_nodes, value=min(150, max_nodes), step=10)
            st.markdown("---")
            st.markdown("**Legend**")
            for dept, color in DEPT_COLORS.items():
                st.markdown(f"<span style='color:{color}'>●</span> {dept}", unsafe_allow_html=True)

        with col_graph:
            nodes_sample = list(G.nodes())[:show_n]
            G_sub = G.subgraph(nodes_sample)
            m_sub = metric_df[metric_df["EmployeeNumber"].isin(nodes_sample)]
            fig = draw_network(G_sub, m_sub, color_by, size_by)
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown("### Top 20 — Betweenness Centrality")
        st.caption("คนที่เป็น 'สะพาน' เชื่อมระหว่างแผนก — ถ้าลาออกการสื่อสารขาด")

        top20 = metric_df.nlargest(20, "Betweenness")
        fig_btw = px.bar(top20, x="EmployeeNumber", y="Betweenness",
                         color="Department", color_discrete_map=DEPT_COLORS,
                         hover_data=["JobRole", "JobLevel", "YearsAtCompany", "OrgResilienceScore"],
                         template="plotly_dark")
        fig_btw.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                              xaxis_title="Employee ID", height=350)
        st.plotly_chart(fig_btw, use_container_width=True)

        st.markdown("### Centrality Comparison")
        col_a, col_b = st.columns(2)
        with col_a:
            fig_pgr = px.bar(metric_df.nlargest(15, "PageRank"), x="EmployeeNumber", y="PageRank",
                             color="Department", color_discrete_map=DEPT_COLORS,
                             title="Top 15 — PageRank", template="plotly_dark")
            fig_pgr.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", height=300)
            st.plotly_chart(fig_pgr, use_container_width=True)
        with col_b:
            fig_deg = px.bar(metric_df.nlargest(15, "Degree"), x="EmployeeNumber", y="Degree",
                             color="Department", color_discrete_map=DEPT_COLORS,
                             title="Top 15 — Degree Centrality", template="plotly_dark")
            fig_deg.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", height=300)
            st.plotly_chart(fig_deg, use_container_width=True)

        st.markdown("### Org Resilience Score — Top 30")
        st.caption("สูตร: 0.35×Betweenness + 0.25×PageRank×10 + 0.20×Degree + 0.20×(1-Satisfaction)")
        top30 = metric_df.nlargest(30, "OrgResilienceScore")
        fig_rs = px.scatter(top30, x="Betweenness", y="OrgResilienceScore",
                            size="Degree", color="Department", color_discrete_map=DEPT_COLORS,
                            hover_data=["EmployeeNumber","JobRole","Attrition"],
                            template="plotly_dark", height=380)
        fig_rs.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117")
        st.plotly_chart(fig_rs, use_container_width=True)

    with tab3:
        st.markdown("### Community Detection — Louvain Algorithm")
        st.caption("กลุ่มที่ทำงานด้วยกันจริงๆ อาจต่างจาก org chart")

        if G.number_of_edges() > 0:
            partition = community_louvain.best_partition(G)
            community_series = pd.Series(partition, name="Community").reset_index()
            community_series.columns = ["EmployeeNumber", "Community"]
            metric_with_comm = metric_df.merge(community_series, on="EmployeeNumber")

            n_comm = metric_with_comm["Community"].nunique()
            st.info(f"พบ **{n_comm} communities** จาก Louvain Algorithm")

            col_c1, col_c2 = st.columns([2, 1])
            with col_c1:
                comm_counts = metric_with_comm.groupby(["Community","Department"]).size().reset_index(name="Count")
                fig_comm = px.bar(comm_counts, x="Community", y="Count", color="Department",
                                  color_discrete_map=DEPT_COLORS, barmode="stack",
                                  title="สมาชิกใน Community แต่ละกลุ่ม แยกตามแผนก",
                                  template="plotly_dark")
                fig_comm.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", height=380)
                st.plotly_chart(fig_comm, use_container_width=True)
            with col_c2:
                st.markdown("**Community Summary**")
                comm_summary = metric_with_comm.groupby("Community").agg(
                    Members=("EmployeeNumber","count"),
                    Avg_Resilience=("OrgResilienceScore","mean"),
                    Attrition_Rate=("Attrition","mean")
                ).round(3).reset_index()
                comm_summary["Attrition_Rate"] = comm_summary["Attrition_Rate"].apply(lambda x: f"{x:.1%}")
                st.dataframe(comm_summary, use_container_width=True, hide_index=True)
        else:
            st.warning("ไม่มี Edges เพียงพอในการสร้าง Community กรุณาปรับค่า Threshold ให้ต่ำลง")

    with tab4:
        st.markdown("### ⚡ What-if Simulation")
        st.markdown("จำลองว่า **ถ้าพนักงานคนนี้ลาออก** — องค์กรจะได้รับผลกระทบอย่างไร")

        col_s1, col_s2 = st.columns([1, 2])
        with col_s1:
            st.markdown("**เลือกพนักงาน**")

            top_risk = metric_df.nlargest(30, "OrgResilienceScore")[
                ["EmployeeNumber","Department","JobRole","OrgResilienceScore","Betweenness"]
            ].reset_index(drop=True)

            selected_id = st.selectbox(
                "Employee ID (เรียงตาม Resilience Score สูงสุด)",
                options=top_risk["EmployeeNumber"].tolist(),
                format_func=lambda x: f"#{x} — {top_risk[top_risk.EmployeeNumber==x]['JobRole'].values[0]} ({top_risk[top_risk.EmployeeNumber==x]['Department'].values[0]})"
            )

            selected_row = metric_df[metric_df["EmployeeNumber"] == selected_id].iloc[0]
            st.markdown("---")
            st.markdown("**ข้อมูลพนักงานที่เลือก**")
            st.markdown(f"- แผนก: **{selected_row['Department']}**")
            st.markdown(f"- ตำแหน่ง: **{selected_row['JobRole']}**")
            st.markdown(f"- Level: **{selected_row['JobLevel']}**")
            st.markdown(f"- อายุงาน: **{selected_row['YearsAtCompany']} ปี**")
            st.markdown(f"- Betweenness: **{selected_row['Betweenness']:.4f}**")
            st.markdown(f"- Org Resilience Score: **{selected_row['OrgResilienceScore']:.4f}**")
            st.markdown(f"- Attrition Risk: {'**⚠️ ใช่**' if selected_row['Attrition']==1 else '**✅ ไม่**'}")

            run_btn = st.button("▶ Run Simulation", type="primary", use_container_width=True)

        with col_s2:
            if run_btn:
                with st.spinner("กำลังจำลอง..."):
                    result = run_simulation(G, metric_df, selected_id)

                st.markdown("#### ผลกระทบที่เกิดขึ้น")
                r1, r2, r3, r4 = st.columns(4)
                r1.metric("Edges สูญเสีย", f"{result['lost_edges']}", delta=f"-{result['lost_edges']}", delta_color="inverse")
                r2.metric("พนักงานที่ได้รับผล", f"{result['affected_count']} คน")
                r3.metric("Network แตกเพิ่ม", f"{result['frag_increase']} cluster")
                r4.metric("Components หลังลบ", f"{result['components_after']}")

                st.markdown("---")

                if result['frag_increase'] > 0:
                    st.error(f"⚠️ **Critical Risk** — การลาออกของพนักงาน #{selected_id} ทำให้ network แตกออกเป็น {result['components_after']} ส่วน การสื่อสารข้ามกลุ่มจะหยุดชะงักทันที")
                elif result['lost_edges'] > 10:
                    st.warning(f"🔶 **High Risk** — สูญเสีย {result['lost_edges']} connections ส่งผลกระทบต่อ {result['affected_count']} คน")
                else:
                    st.success(f"✅ **Low Risk** — ผลกระทบอยู่ในระดับที่รับมือได้")

                if not result["changes_df"].empty:
                    st.markdown("#### พนักงานที่ Betweenness เปลี่ยนแปลงมากที่สุด")
                    st.dataframe(result["changes_df"].head(10), use_container_width=True, hide_index=True)

                st.markdown("#### Network ก่อน vs หลัง")
                nodes_viz = list(G.neighbors(selected_id)) + [selected_id]
                nodes_viz = nodes_viz[:min(60, len(nodes_viz))]
                G_before = G.subgraph(nodes_viz)
                G_after  = G.copy()
                G_after.remove_node(selected_id)
                neighbors_after = [n for n in nodes_viz if n != selected_id]
                G_after_sub = G_after.subgraph(neighbors_after)

                m_viz = metric_df[metric_df["EmployeeNumber"].isin(nodes_viz)]
                m_after = metric_df[metric_df["EmployeeNumber"].isin(neighbors_after)]

                vcol1, vcol2 = st.columns(2)
                with vcol1:
                    fig_b = draw_network(G_before, m_viz, "Department", "Betweenness", highlight_node=selected_id)
                    fig_b.update_layout(height=320)
                    st.plotly_chart(fig_b, use_container_width=True, key="sim_before")
                with vcol2:
                    if len(neighbors_after) > 0:
                        fig_a = draw_network(G_after_sub, m_after, "Department", "Betweenness")
                        fig_a.update_layout(height=320)
                        st.plotly_chart(fig_a, use_container_width=True, key="sim_after")
                    else:
                        st.info("ไม่มี node เหลืออยู่ในกลุ่มนี้")
            else:
                st.info("เลือกพนักงานแล้วกด **▶ Run Simulation** เพื่อดูผลกระทบ")

    with tab5:
        st.markdown("### 🏢 Department Health Score")
        st.caption("ประเมินสุขภาพของแต่ละแผนกจาก network metrics รวมกัน")

        dept_health = metric_df.groupby("Department").agg(
            Members=("EmployeeNumber","count"),
            Avg_Betweenness=("Betweenness","mean"),
            Avg_Clustering=("Clustering","mean"),
            Avg_Satisfaction=("Satisfaction","mean"),
            Attrition_Rate=("Attrition","mean"),
            Avg_Resilience=("OrgResilienceScore","mean"),
        ).round(4).reset_index()

        dept_health["HealthScore"] = (
            dept_health["Avg_Satisfaction"] / 4 * 0.40
            + dept_health["Avg_Clustering"] * 0.30
            - dept_health["Attrition_Rate"] * 0.20
            - dept_health["Avg_Resilience"] * 0.10
        ).round(4)

        col_h1, col_h2 = st.columns([1, 2])
        with col_h1:
            st.markdown("**สรุปรายแผนก**")
            display_df = dept_health[["Department","Members","HealthScore","Attrition_Rate","Avg_Satisfaction"]].copy()
            display_df["Attrition_Rate"] = display_df["Attrition_Rate"].apply(lambda x: f"{x:.1%}")
            display_df["HealthScore"] = display_df["HealthScore"].apply(lambda x: f"{x:.3f}")
            st.dataframe(display_df, use_container_width=True, hide_index=True)

        with col_h2:
            fig_health = px.bar(dept_health, x="Department", y="HealthScore",
                                color="Department", color_discrete_map=DEPT_COLORS,
                                title="Department Health Score (สูง = สุขภาพดี)",
                                template="plotly_dark", text="HealthScore")
            fig_health.update_traces(texttemplate="%{text:.3f}", textposition="outside")
            fig_health.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                                     showlegend=False, height=320)
            st.plotly_chart(fig_health, use_container_width=True)

        st.markdown("### Radar — เปรียบเทียบแผนก")
        categories = ["Avg_Betweenness","Avg_Clustering","Avg_Satisfaction","HealthScore"]
        fig_radar = go.Figure()
        for _, row in dept_health.iterrows():
            vals = [row[c] for c in categories]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals + [vals[0]], theta=categories + [categories[0]],
                fill="toself", name=row["Department"],
                line_color=DEPT_COLORS.get(row["Department"], "#8892a4")
            ))
        fig_radar.update_layout(
            polar=dict(bgcolor="#1a1f2e",
                       radialaxis=dict(visible=True, color="#8892a4"),
                       angularaxis=dict(color="#8892a4")),
            paper_bgcolor="#0e1117", template="plotly_dark",
            legend=dict(bgcolor="#1a1f2e"), height=420
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        st.markdown("---")
        st.markdown("### Formal vs Informal Network")
        st.caption("เปรียบเทียบ org chart (Formal) กับ community ที่เกิดขึ้นจริง (Informal)")

        if G.number_of_edges() > 0:
            partition = community_louvain.best_partition(G)
            community_series = pd.Series(partition, name="Community").reset_index()
            community_series.columns = ["EmployeeNumber", "Community"]
            merged = metric_df.merge(community_series, on="EmployeeNumber")

            cross = pd.crosstab(merged["Department"], merged["Community"])
            fig_cross = px.imshow(cross, color_continuous_scale="Blues",
                                  title="Heatmap: Department (Formal) vs Community (Informal)",
                                  template="plotly_dark")
            fig_cross.update_layout(paper_bgcolor="#0e1117", height=350)
            st.plotly_chart(fig_cross, use_container_width=True)
        else:
            st.warning("ไม่มี Edges เพียงพอในการสร้าง Community กรุณาปรับค่า Threshold ให้ต่ำลง")

if __name__ == "__main__":
    main()
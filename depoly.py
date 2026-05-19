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

st.set_page_config(page_title="ONA Dashboard", page_icon="🕸️", layout="wide", initial_sidebar_state="expanded")

# --- CSS Styling (เน้นความสะอาดตา) ---
st.markdown("""
<style>
    .block-container { padding: 1.5rem 2rem; }
    .metric-card {
        background: #1e2532; border: 1px solid #2d3548; border-left: 5px solid #3b82f6;
        border-radius: 8px; padding: 1rem 1.2rem; text-align: left;
    }
    .metric-card-risk { border-left: 5px solid #ef4444; }
    .metric-label { color: #94a3b8; font-size: 0.85rem; font-weight: 500; margin-bottom: 5px; }
    .metric-value { color: #f8fafc; font-size: 1.8rem; font-weight: 700; line-height: 1.1; }
    .insight-box {
        background-color: #1e293b; border-radius: 8px; padding: 15px; margin-bottom: 15px; border-left: 4px solid #f59e0b;
    }
    .insight-title { font-weight: bold; color: #fbbf24; margin-bottom: 5px; }
</style>
""", unsafe_allow_html=True)

# --- Core Logic ---
def compute_edge_weight(r1, r2, weights):
    total_weight = sum(weights.values())
    if total_weight == 0: return 0.0
    score_dept = 1.0 if r1["Department"] == r2["Department"] else 0.0
    score_role = 1.0 if r1["JobRole"] == r2["JobRole"] else 0.0
    score_level = math.exp(-abs(r1["JobLevel"] - r2["JobLevel"]))
    score_tenure = math.exp(-abs(r1["YearsAtCompany"] - r2["YearsAtCompany"]) * 0.1)
    final_score = ((score_dept * weights["department"]) + (score_level * weights["job_level"]) + 
                   (score_role * weights["job_role"]) + (score_tenure * weights["tenure"])) / total_weight
    return round(final_score, 4)

@st.cache_data(show_spinner="กำลังโหลด dataset...")
def load_default_data():
    data = []
    depts = ["Sales", "Research & Development", "Human Resources"]
    roles = ["Manager", "Developer", "Analyst", "HR", "Executive"]
    for i in range(1, 151):
        data.append({
            "EmployeeNumber": i, "Department": depts[i % 3], "JobRole": roles[i % 5],
            "JobLevel": (i % 5) + 1, "YearsAtCompany": (i % 10) + 1, "JobSatisfaction": (i % 4) + 1,
            "Attrition": "Yes" if i % 15 == 0 else "No", "Age": 25 + (i % 20), "MonthlyIncome": 5000 + (i * 100)
        })
    df = pd.DataFrame(data)
    df["Attrition_flag"] = (df["Attrition"] == "Yes").astype(int)
    return df

@st.cache_data(show_spinner="กำลังสร้างโครงสร้าง Network...")
def build_graph_cached(df_json, threshold, weights):
    df = pd.read_json(StringIO(df_json))
    G = nx.Graph()
    for _, row in df.iterrows():
        G.add_node(int(row["EmployeeNumber"]), **row.to_dict())
    records = df.set_index("EmployeeNumber").to_dict("index")
    emp_ids = list(records.keys())
    for id1, id2 in combinations(emp_ids, 2):
        w = compute_edge_weight(records[id1], records[id2], weights)
        if w >= threshold: G.add_edge(int(id1), int(id2), weight=w)
    return G

@st.cache_data(show_spinner="กำลังวิเคราะห์ข้อมูลเชิงลึก...")
def compute_metrics(df_json, threshold, weights):
    df = pd.read_json(StringIO(df_json))
    G = build_graph_cached(df_json, threshold, weights)
    deg = nx.degree_centrality(G)
    btw = nx.betweenness_centrality(G, normalized=True)
    pgr = nx.pagerank(G, alpha=0.85) if G.number_of_edges() > 0 else {n: 0 for n in G.nodes()}
    try: clu = nx.clustering(G)
    except: clu = {node: 0.0 for node in G.nodes()}
    metrics = []
    for node in G.nodes():
        attr = G.nodes[node]
        sat_norm = (attr["JobSatisfaction"] - 1) / 3
        score = round(0.35 * btw.get(node, 0) + 0.25 * pgr.get(node, 0) * 10 + 0.20 * deg.get(node, 0) + 0.20 * (1 - sat_norm), 4)
        metrics.append({
            "EmployeeNumber": node, "Department": attr["Department"], "JobRole": attr["JobRole"],
            "JobLevel": attr["JobLevel"], "YearsAtCompany": attr["YearsAtCompany"],
            "Satisfaction": attr["JobSatisfaction"], "Attrition": attr["Attrition_flag"],
            "Degree": round(deg.get(node, 0), 4), "Betweenness": round(btw.get(node, 0), 4),
            "PageRank": round(pgr.get(node, 0), 6), "Clustering": round(clu.get(node, 0), 4),
            "OrgResilienceScore": score,
        })
    return pd.DataFrame(metrics), G

DEPT_COLORS = {"Sales": "#7F77DD", "Research & Development": "#1D9E75", "Human Resources": "#D4537E"}

def draw_network(G, metric_df, color_by="Department", size_by="Betweenness", highlight_node=None):
    if G.number_of_nodes() == 0: return go.Figure()
    pos = nx.spring_layout(G, seed=42, k=1.5/math.sqrt(G.number_of_nodes()) if G.number_of_nodes() > 0 else 1)
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]; x1, y1 = pos[v]
        edge_x += [x0, x1, None]; edge_y += [y0, y1, None]
    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=0.3, color="#334155"), hoverinfo="none")
    node_x, node_y, node_text, node_color, node_size, node_hover = [], [], [], [], [], []
    m = metric_df.set_index("EmployeeNumber")
    max_size = m[size_by].max() if m[size_by].max() > 0 else 1

    for node in G.nodes():
        if node not in m.index: continue
        x, y = pos[node]; node_x.append(x); node_y.append(y)
        row = m.loc[node]
        dept = row["Department"]
        
        if color_by == "Department": node_color.append(DEPT_COLORS.get(dept, "#8892a4"))
        elif color_by == "Attrition Risk": node_color.append("#ef4444" if row["Attrition"] == 1 else "#10b981")
        else:
            score = row["OrgResilienceScore"]
            node_color.append(f"rgb({int(255*score)}, {int(255*(1-score))}, 80)")

        raw = row[size_by]
        node_size.append(12 + (raw / max_size) * 35)

        if highlight_node and node == highlight_node: node_size[-1] *= 2.0

        node_text.append(str(node))
        node_hover.append(
            f"<b>รหัสพนักงาน #{node}</b><br>แผนก: {dept}<br>ตำแหน่ง: {row['JobRole']}<br>"
            f"ความสำคัญต่อเครือข่าย: {row['OrgResilienceScore']:.2f}<br>"
            f"แนวโน้มลาออก: {'⚠️ สูง' if row['Attrition']==1 else '✅ ปกติ'}"
        )

    node_trace = go.Scatter(x=node_x, y=node_y, mode="markers",
        text=node_text, hovertext=node_hover, hoverinfo="text",
        marker=dict(size=node_size, color=node_color, line=dict(width=1, color="#0f172a")))
    
    fig = go.Figure(data=[edge_trace, node_trace],
        layout=go.Layout(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", margin=dict(l=0, r=0, t=0, b=0),
            showlegend=False, hovermode="closest", xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False), height=550))
    return fig

def run_simulation(G, metric_df, remove_node):
    G2 = G.copy(); G2.remove_node(remove_node)
    affected = list(G.neighbors(remove_node))
    btw_before = nx.betweenness_centrality(G, normalized=True)
    btw_after  = nx.betweenness_centrality(G2, normalized=True)
    changes = [{"node": n, "dept": G.nodes[n]["Department"], "role": G.nodes[n]["JobRole"], 
                "btw_delta": round(btw_after.get(n, 0) - btw_before.get(n, 0), 4)} for n in affected]
    return {
        "lost_edges": G.number_of_edges() - G2.number_of_edges(), "affected_count": len(affected),
        "frag_increase": nx.number_connected_components(G2) - nx.number_connected_components(G),
        "changes_df": pd.DataFrame(changes).sort_values("btw_delta", ascending=False)
    }

def main():
    # --- Sidebar ---
    with st.sidebar:
        st.header("📂 ข้อมูลองค์กร")
        uploaded_file = st.file_uploader("อัปโหลดไฟล์ HR Data (CSV)", type="csv")
        st.divider()
        
        # ย้ายของทางเทคนิคไปซ่อนไว้ เพื่อไม่ให้รกตา User
        with st.expander("⚙️ ตั้งค่าโมเดลขั้นสูง (Advanced Settings)"):
            st.markdown("ปรับแต่ง Algorithm สำหรับระบุความสัมพันธ์")
            w_dept = st.slider("แผนกเดียวกัน", 0.0, 1.0, 0.50)
            w_level = st.slider("ลำดับชั้นใกล้เคียงกัน", 0.0, 1.0, 0.25)
            w_role = st.slider("สายงานเดียวกัน", 0.0, 1.0, 0.15)
            w_tenure = st.slider("อายุงานใกล้เคียงกัน", 0.0, 1.0, 0.10)
            EDGE_THRESHOLD = st.slider("คัดกรองเฉพาะความสัมพันธ์ที่แน่นแฟ้น (Threshold)", 0.1, 0.9, 0.40)
            WEIGHTS = {"department": w_dept, "job_level": w_level, "job_role": w_role, "tenure": w_tenure}

    # --- Load Data ---
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if "Attrition_flag" not in df.columns and "Attrition" in df.columns:
            df["Attrition_flag"] = (df["Attrition"] == "Yes").astype(int)
    else:
        df = load_default_data()

    df_json = df.to_json()
    metric_df, G = compute_metrics(df_json, EDGE_THRESHOLD, WEIGHTS)

    # --- Header ---
    st.title("🕸️ HR Network Intelligence")
    st.markdown("ระบบวิเคราะห์ความเสี่ยงองค์กรผ่านโครงสร้างความสัมพันธ์ที่ซ่อนอยู่ (Informal Network)")
    
    # --- Executive Summary (เปิดมาต้องรู้เรื่องเลย) ---
    high_risk_bridges = metric_df[(metric_df["Attrition"] == 1) & (metric_df["Betweenness"] > metric_df["Betweenness"].quantile(0.8))]
    
    if not high_risk_bridges.empty:
        st.markdown(f"""
        <div class="insight-box">
            <div class="insight-title">🚨 แจ้งเตือนความเสี่ยงระดับสูง (Critical Risk)</div>
            พบพนักงาน <b>{len(high_risk_bridges)} คน</b> ที่ทำหน้าที่เป็น <b>"สะพานเชื่อมหลัก"</b> ระหว่างแผนก และมีความเสี่ยงที่จะลาออก 
            หากบุคคลเหล่านี้ลาออก การประสานงานข้ามแผนกอาจหยุดชะงัก (แนะนำให้ดูแท็บ "จำลองการลาออก")
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="insight-box" style="border-left-color: #10b981;">
            <div class="insight-title" style="color: #10b981;">✅ สถานะเครือข่ายปกติ</div>
            ยังไม่พบพนักงานที่เป็นจุดศูนย์กลาง (Key Person) ที่มีความเสี่ยงในการลาออกอย่างชัดเจน
        </div>
        """, unsafe_allow_html=True)

    # --- Metric Cards ---
    key_person = metric_df.loc[metric_df["Betweenness"].idxmax(), "EmployeeNumber"] if not metric_df.empty else "-"
    
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="metric-card"><div class="metric-label">👥 จำนวนพนักงาน</div><div class="metric-value">{G.number_of_nodes():,}</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><div class="metric-label">🔗 ความสัมพันธ์ที่พบ</div><div class="metric-value">{G.number_of_edges():,}</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-card"><div class="metric-label">⭐ พนักงานที่สำคัญที่สุด (ศูนย์กลาง)</div><div class="metric-value">#{key_person}</div></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="metric-card metric-card-risk"><div class="metric-label">⚠️ เสี่ยงกระทบองค์กรสูง</div><div class="metric-value">{len(metric_df[metric_df["OrgResilienceScore"] > 0.5])} คน</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Tabs (เปลี่ยนชื่อให้เป็นภาษา Business) ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "🌐 แผนผังองค์กรเสมือน (Informal Network)", 
        "⭐ เจาะลึกรายบุคคล (Key Talent)", 
        "🏢 สุขภาพระดับแผนก (Team Health)", 
        "⚡ จำลองความเสียหาย (What-if Scenario)"
    ])

    with tab1:
        st.markdown("ภาพสะท้อนการทำงานจริงที่ **ไม่ได้อยู่บน Org Chart** ใครคุยกับใคร แผนกไหนทำงานร่วมกันมากที่สุด")
        col_ctrl, col_graph = st.columns([1, 4])
        with col_ctrl:
            color_by = st.selectbox("แสดงสีตาม:", ["Department", "Attrition Risk", "OrgResilienceScore"])
            size_by  = st.selectbox("ขนาดจุดบ่งบอกถึง:", ["Betweenness (สะพานเชื่อม)", "Degree (จำนวนคอนเนคชั่น)", "PageRank (ผู้มีอิทธิพล)"])
            # Map business name back to technical name for the code
            size_col_map = {"Betweenness (สะพานเชื่อม)": "Betweenness", "Degree (จำนวนคอนเนคชั่น)": "Degree", "PageRank (ผู้มีอิทธิพล)": "PageRank"}
            
            st.markdown("---")
            st.caption("คำอธิบายสัญลักษณ์:")
            for dept, color in DEPT_COLORS.items():
                st.markdown(f"<span style='color:{color}'>●</span> {dept}", unsafe_allow_html=True)
        with col_graph:
            fig = draw_network(G, metric_df, color_by, size_col_map[size_by])
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown("### ค้นหาผู้ที่มีบทบาทสำคัญที่สุดขององค์กร")
        st.markdown("บุคลากรเหล่านี้คือคนที่ขับเคลื่อนงานข้ามแผนก และเป็นศูนย์รวมจิตใจของทีม")
        
        c_a, c_b = st.columns(2)
        with c_a:
            st.markdown("**🌉 Top 10 ผู้เชื่อมโยงข้ามแผนก (Communication Bridges)**")
            st.caption("คนกลุ่มนี้ช่วยไม่ให้เกิดไซโล (Silo) หากขาดไป งานข้ามแผนกจะชะงัก")
            top_btw = metric_df.nlargest(10, "Betweenness")
            fig_btw = px.bar(top_btw, x="Betweenness", y=top_btw["EmployeeNumber"].astype(str), orientation='h',
                             color="Department", color_discrete_map=DEPT_COLORS, template="plotly_dark")
            fig_btw.update_layout(yaxis_title="รหัสพนักงาน", xaxis_title="คะแนนความเป็นสะพาน", height=350, margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig_btw, use_container_width=True)
            
        with c_b:
            st.markdown("**🗣️ Top 10 ศูนย์กลางทีม (Team Hubs)**")
            st.caption("คนที่มีความสัมพันธ์กับคนอื่นมากที่สุด เหมาะแก่การเป็นตัวกระจายข่าวสารองค์กร")
            top_deg = metric_df.nlargest(10, "Degree")
            fig_deg = px.bar(top_deg, x="Degree", y=top_deg["EmployeeNumber"].astype(str), orientation='h',
                             color="Department", color_discrete_map=DEPT_COLORS, template="plotly_dark")
            fig_deg.update_layout(yaxis_title="รหัสพนักงาน", xaxis_title="จำนวนคอนเนคชั่น", height=350, margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig_deg, use_container_width=True)

    with tab3:
        st.markdown("### สุขภาพความสัมพันธ์ระดับแผนก")
        
        dept_health = metric_df.groupby("Department").agg(
            Members=("EmployeeNumber","count"),
            Avg_Betweenness=("Betweenness","mean"),
            Avg_Clustering=("Clustering","mean"),
            Avg_Satisfaction=("Satisfaction","mean"),
            Attrition_Rate=("Attrition","mean")
        ).reset_index()

        # แปลงเป็น Business logic
        dept_health["Silo_Risk"] = (1 - dept_health["Avg_Betweenness"]) * 100 # ยิ่งน้อยยิ่งดี
        dept_health["Team_Bonding"] = dept_health["Avg_Clustering"] * 100 # ยิ่งมากยิ่งดี

        col_h1, col_h2 = st.columns([1, 1])
        with col_h1:
            st.markdown("**ตารางสรุปสุขภาพแผนก**")
            display_df = dept_health[["Department", "Members", "Team_Bonding", "Silo_Risk", "Attrition_Rate"]].copy()
            display_df["Team_Bonding"] = display_df["Team_Bonding"].apply(lambda x: f"{x:.1f}%")
            display_df["Silo_Risk"] = display_df["Silo_Risk"].apply(lambda x: f"{x:.1f}%")
            display_df["Attrition_Rate"] = display_df["Attrition_Rate"].apply(lambda x: f"{x:.1%}")
            # เปลี่ยนชื่อคอลัมน์ให้เข้าใจง่าย
            display_df.columns = ["แผนก", "จำนวนคน", "ความแน่นแฟ้นในทีม", "ความเสี่ยงการเกิด Silo", "อัตราลาออก"]
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            
        with col_h2:
            st.markdown("**เปรียบเทียบมิติต่างๆ (ยิ่งกว้างยิ่งดี)**")
            fig_radar = go.Figure()
            categories = ["ความแน่นแฟ้น (Bonding)", "ความพอใจในงาน (Satisfaction)", "การสื่อสารข้ามทีม (Collaboration)"]
            
            for _, row in dept_health.iterrows():
                # Normalize values
                vals = [row["Team_Bonding"]/100, row["Avg_Satisfaction"]/4, row["Avg_Betweenness"]/dept_health["Avg_Betweenness"].max()]
                fig_radar.add_trace(go.Scatterpolar(r=vals + [vals[0]], theta=categories + [categories[0]],
                    fill="toself", name=row["Department"], line_color=DEPT_COLORS.get(row["Department"], "#8892a4")))
                    
            fig_radar.update_layout(polar=dict(bgcolor="#1e293b", radialaxis=dict(visible=False)),
                paper_bgcolor="#0e1117", template="plotly_dark", height=300, margin=dict(l=40, r=40, t=20, b=20))
            st.plotly_chart(fig_radar, use_container_width=True)

    with tab4:
        st.markdown("### ⚡ จำลองความเสียหายหากสูญเสียบุคลากร (What-if Scenario)")
        st.info("💡 ทดลองเลือกพนักงาน 1 คนเพื่อจำลองสถานการณ์ว่า 'ถ้าคนนี้ลาออกกะทันหัน' จะเกิดหลุมดำในการสื่อสารตรงไหนบ้าง")

        col_s1, col_s2 = st.columns([1, 2])
        with col_s1:
            st.markdown("**1. เลือกพนักงานที่ต้องการทดสอบ**")
            top_risk = metric_df.nlargest(30, "OrgResilienceScore").reset_index(drop=True)
            selected_id = st.selectbox("พิมพ์เพื่อค้นหารหัสพนักงาน:", options=top_risk["EmployeeNumber"].tolist(),
                format_func=lambda x: f"#{x} — {top_risk[top_risk.EmployeeNumber==x]['JobRole'].values[0]}")

            sr = metric_df[metric_df["EmployeeNumber"] == selected_id].iloc[0]
            st.markdown(f"**ข้อมูลพนักงาน:** แผนก {sr['Department']} | ระดับ {sr['JobLevel']} | อายุงาน {sr['YearsAtCompany']} ปี")
            
            run_btn = st.button("▶ เริ่มจำลองสถานการณ์ (Run Simulation)", type="primary", use_container_width=True)

        with col_s2:
            if run_btn:
                with st.spinner("กำลังวิเคราะห์ผลกระทบลูกโซ่..."):
                    res = run_simulation(G, metric_df, selected_id)

                r1, r2, r3 = st.columns(3)
                r1.metric("การสื่อสารที่ขาดหาย (Connections Lost)", f"{res['lost_edges']} เส้นทาง")
                r2.metric("พนักงานที่ได้รับผลกระทบโดยตรง", f"{res['affected_count']} คน")
                r3.metric("เครือข่ายถูกตัดขาดเพิ่ม", f"{res['frag_increase']} ส่วน")

                if res['frag_increase'] > 0:
                    st.error(f"🚨 **อันตราย!** การขาดหายไปของ #{selected_id} ทำให้พนักงานบางกลุ่มถูกตัดขาดจากส่วนอื่นๆ ของบริษัทโดยสมบูรณ์")
                elif res['lost_edges'] > 15:
                    st.warning(f"⚠️ **เสี่ยงปานกลาง** สูญเสียเส้นทางการสื่อสารจำนวนมาก ต้องหาคนมาทำหน้าที่ประสานงานแทนทันที")
                else:
                    st.success(f"✅ **ความเสี่ยงต่ำ** เครือข่ายยังมีคนอื่นคอยซัพพอร์ตการทำงานแทนได้")

                st.markdown("**พนักงานที่ต้องรับภาระประสานงานหนักขึ้น (ควรเฝ้าระวัง)**")
                display_changes = res['changes_df'].head(5).copy()
                display_changes.columns = ["รหัสพนักงาน", "แผนก", "ตำแหน่ง", "ภาระการประสานงานที่เพิ่มขึ้น"]
                st.dataframe(display_changes, use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()
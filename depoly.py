import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import networkx as nx
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import community as community_louvain
import math, io
from itertools import combinations
from io import StringIO

st.set_page_config(
    page_title="ONA — ระบบวิเคราะห์ความเสี่ยงพนักงาน",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Thai:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans Thai', sans-serif; }
.block-container { padding: 1.5rem 2rem; }

/* ── Welcome ── */
.welcome-wrap {
    background: linear-gradient(135deg,#1a1f2e 0%,#1c2540 100%);
    border: 1px solid #2d3548;
    border-radius: 16px;
    padding: 1.6rem 2rem 1.4rem;
    margin-bottom: 1.25rem;
}
.welcome-title { font-size:1.15rem; font-weight:600; color:#e8eaf0; margin-bottom:.35rem; }
.welcome-desc  { font-size:.85rem; color:#8892a4; line-height:1.65; margin-bottom:1rem; }
.steps { display:grid; grid-template-columns:repeat(auto-fit,minmax(155px,1fr)); gap:10px; }
.step  {
    background:#0e1117; border-radius:10px; padding:.65rem .9rem;
    display:flex; gap:9px; align-items:flex-start;
}
.snum {
    background:#7F77DD; color:#fff; border-radius:50%;
    width:20px; height:20px; min-width:20px;
    display:flex; align-items:center; justify-content:center;
    font-size:.68rem; font-weight:700; margin-top:1px;
}
.stxt { font-size:.8rem; color:#c8d0e0; line-height:1.5; }
.stxt b { color:#e8eaf0; }
.dismiss { font-size:.72rem; color:#404860; margin-top:.8rem; }

/* ── Metric cards ── */
.metric-card {
    background:#1a1f2e; border:1px solid #2d3548;
    border-radius:12px; padding:1.1rem 1.3rem;
    text-align:center; transition:border-color .2s;
}
.metric-card:hover { border-color:#4a5568; }
.metric-label { color:#8892a4; font-size:.72rem; margin-bottom:4px; letter-spacing:.06em; text-transform:uppercase; }
.metric-value { color:#e8eaf0; font-size:1.85rem; font-weight:600; line-height:1.1; font-family:'JetBrains Mono',monospace; }
.metric-sub   { color:#5d6a80; font-size:.68rem; margin-top:4px; }
.risk-high { color:#ef4444; }

/* ── Hint bar ── */
.hint {
    background:#1a1f2e; border-left:3px solid #7F77DD;
    border-radius:0 8px 8px 0; padding:.5rem 1rem;
    font-size:.82rem; color:#8892a4; margin-bottom:.85rem;
}
.hint b { color:#c8d0e0; }

/* ── Recommendation cards ── */
.rec-card {
    background:#1a1f2e; border-left:4px solid #4a5568;
    border-radius:0 10px 10px 0; padding:1rem 1.2rem; margin-bottom:10px;
}
.rec-card.critical { border-left-color:#ef4444; }
.rec-card.high     { border-left-color:#f59e0b; }
.rec-card.medium   { border-left-color:#3b82f6; }
.rec-card.low      { border-left-color:#22c55e; }
.rbadge {
    display:inline-block; font-size:.68rem; font-weight:600;
    padding:2px 8px; border-radius:999px; margin-left:6px; vertical-align:middle;
}
.bc { background:rgba(239,68,68,.15);  color:#ef4444; }
.bh { background:rgba(245,158,11,.15); color:#f59e0b; }
.bm { background:rgba(59,130,246,.15); color:#3b82f6; }
.bl { background:rgba(34,197,94,.15);  color:#22c55e; }

/* ── Empty state ── */
.empty-state {
    background:#1a1f2e; border:1px dashed #2d3548;
    border-radius:12px; padding:2.5rem; text-align:center; margin-top:.5rem;
}
.empty-icon  { font-size:2rem; margin-bottom:.65rem; }
.empty-title { color:#e8eaf0; font-weight:500; margin-bottom:.4rem; font-size:.95rem; }
.empty-desc  { color:#5d6a80; font-size:.82rem; line-height:1.6; }

/* ── Sidebar note ── */
.sidebar-note {
    background:#1a1f2e; border-radius:8px;
    padding:.45rem .75rem; font-size:.76rem; color:#5d6a80; margin-top:4px;
}

div[data-testid="stTabs"] button { font-size:.85rem; }
.stAlert { border-radius:8px; }
</style>
""", unsafe_allow_html=True)

# ─── Constants ───────────────────────────────────────────────────────────────────
DEPT_COLORS = {
    "Sales":                  "#7F77DD",
    "Research & Development": "#1D9E75",
    "Human Resources":        "#D4537E",
}

# ─── Core Logic ──────────────────────────────────────────────────────────────────
def compute_edge_weight(r1, r2, weights):
    total = sum(weights.values()) or 1.0
    s_dept   = 1.0 if r1["Department"] == r2["Department"] else 0.0
    s_role   = 1.0 if r1["JobRole"]    == r2["JobRole"]    else 0.0
    s_level  = max(0.0, 1.0 - abs(r1["JobLevel"]       - r2["JobLevel"])       * 0.30)
    s_tenure = max(0.0, 1.0 - abs(r1["YearsAtCompany"] - r2["YearsAtCompany"]) * 0.10)
    return round((weights["department"]*s_dept + weights["job_level"]*s_level +
                  weights["job_role"]*s_role   + weights["tenure"]*s_tenure) / total, 4)

@st.cache_data(show_spinner="กำลังสร้างข้อมูลตัวอย่าง...")
def load_default_data():
    rng = np.random.default_rng(42)
    n, depts = 150, ["Sales","Research & Development","Human Resources"]
    roles_map = {
        "Sales":                  ["Sales Executive","Sales Representative","Manager"],
        "Research & Development": ["Research Scientist","Laboratory Technician","Manager","Developer"],
        "Human Resources":        ["HR Representative","HR Manager","Recruiter"],
    }
    rows = []
    for i in range(1, n+1):
        d = rng.choice(depts)
        rows.append({"EmployeeNumber":i,"Department":d,"JobRole":rng.choice(roles_map[d]),
                     "JobLevel":int(rng.integers(1,6)),"YearsAtCompany":int(rng.integers(1,25)),
                     "JobSatisfaction":int(rng.integers(1,5)),
                     "Attrition":rng.choice(["Yes","No"],p=[0.16,0.84]),
                     "Age":int(rng.integers(22,58)),"MonthlyIncome":int(rng.integers(3000,20001))})
    df = pd.DataFrame(rows)
    df["Attrition_flag"] = (df["Attrition"]=="Yes").astype(int)
    return df

@st.cache_data(show_spinner="กำลังสร้าง Network Graph...")
def build_graph_cached(df_json, threshold, weights):
    df = pd.read_json(StringIO(df_json)); G = nx.Graph()
    for _, row in df.iterrows():
        G.add_node(int(row["EmployeeNumber"]),
                   department=row["Department"], job_role=row["JobRole"],
                   job_level=int(row["JobLevel"]), years_company=int(row["YearsAtCompany"]),
                   satisfaction=int(row["JobSatisfaction"]), attrition=int(row["Attrition_flag"]),
                   age=int(row["Age"]), income=int(row["MonthlyIncome"]))
    records = df.set_index("EmployeeNumber").to_dict("index")
    for id1, id2 in combinations(list(records.keys()), 2):
        w = compute_edge_weight(records[id1], records[id2], weights)
        if w >= threshold:
            G.add_edge(int(id1), int(id2), weight=w)
    return G

@st.cache_data(show_spinner="กำลังวิเคราะห์...")
def compute_metrics(df_json, threshold, weights):
    df = pd.read_json(StringIO(df_json))
    G  = build_graph_cached(df_json, threshold, weights)
    deg = nx.degree_centrality(G)
    btw = nx.betweenness_centrality(G, normalized=True)
    pgr = nx.pagerank(G, alpha=0.85) if G.number_of_edges()>0 else {n:0 for n in G.nodes()}
    try:    clu = nx.clustering(G)
    except: clu = {n:0.0 for n in G.nodes()}
    rows = []
    for node in G.nodes():
        a = G.nodes[node]; sat_n = (a["satisfaction"]-1)/3
        score = round(0.35*btw.get(node,0)+0.25*pgr.get(node,0)*10
                      +0.20*deg.get(node,0)+0.20*(1-sat_n), 4)
        rows.append({"EmployeeNumber":node,"Department":a["department"],"JobRole":a["job_role"],
                     "JobLevel":a["job_level"],"YearsAtCompany":a["years_company"],
                     "Satisfaction":a["satisfaction"],"Attrition":a["attrition"],
                     "Income":a["income"],"Degree":round(deg.get(node,0),4),
                     "Betweenness":round(btw.get(node,0),4),"PageRank":round(pgr.get(node,0),6),
                     "Clustering":round(clu.get(node,0),4),"OrgResilienceScore":score})
    return pd.DataFrame(rows), G

def get_recommendation(row):
    s = row["OrgResilienceScore"] if hasattr(row,"OrgResilienceScore") else row
    if s>=0.60: return {"level":"วิกฤต","css":"critical","icon":"🔴","bc":"bc",
        "actions":["ทำ Succession Plan ทันที — เตรียมผู้สืบทอดตำแหน่ง",
                   "จัด Knowledge Transfer ถ่ายทอดความรู้ให้ผู้ใต้บังคับบัญชา",
                   "ติดตาม Job Satisfaction ทุกเดือน",
                   "พิจารณาปรับค่าตอบแทนและ career path"]}
    if s>=0.40: return {"level":"สูง","css":"high","icon":"🟡","bc":"bh",
        "actions":["ทำ Succession Plan ใน 3 เดือน",
                   "ติดตาม Job Satisfaction ทุกไตรมาส",
                   "พิจารณาปรับ workload ให้สมดุล"]}
    if s>=0.25: return {"level":"ปานกลาง","css":"medium","icon":"🔵","bc":"bm",
        "actions":["ติดตามเป็นระยะ ทุก 6 เดือน","สนับสนุน training และ upskilling"]}
    return {"level":"ต่ำ","css":"low","icon":"🟢","bc":"bl",
            "actions":["ไม่จำเป็นต้องดำเนินการเร่งด่วน"]}

def draw_network(G, metric_df, color_by="Department", size_by="Betweenness", highlight_node=None):
    if G.number_of_nodes()==0: return go.Figure()
    pos = nx.spring_layout(G, seed=42, k=1.5/math.sqrt(max(G.number_of_nodes(),1)))
    ex,ey = [],[]
    for u,v in G.edges():
        x0,y0=pos[u]; x1,y1=pos[v]
        ex+=[x0,x1,None]; ey+=[y0,y1,None]
    edge_trace = go.Scatter(x=ex,y=ey,mode="lines",line=dict(width=0.4,color="#2d3548"),hoverinfo="none")
    nx_,ny_,colors,sizes,texts,hovers = [],[],[],[],[],[]
    m = metric_df.set_index("EmployeeNumber"); max_s = m[size_by].max() or 1
    for node in G.nodes():
        if node not in m.index: continue
        x,y=pos[node]; row=m.loc[node]; dept=row["Department"]
        nx_.append(x); ny_.append(y); texts.append(str(node))
        if   color_by=="Department": colors.append(DEPT_COLORS.get(dept,"#8892a4"))
        elif color_by=="Attrition":  colors.append("#ef4444" if row["Attrition"]==1 else "#22c55e")
        else:
            s=row["OrgResilienceScore"]; colors.append(f"rgb({int(255*s)},{int(255*(1-s))},80)")
        raw=row[size_by]; sz=10+(raw/max_s)*30
        sizes.append(sz*1.8 if highlight_node and node==highlight_node else sz)
        rec=get_recommendation(row)
        hovers.append(
            f"<b>👤 พนักงาน #{node}</b><br>"
            f"🏢 {dept} · {row['JobRole']}<br>"
            f"📋 Level {row['JobLevel']} | อายุงาน {row['YearsAtCompany']} ปี<br>"
            f"─────────────────<br>"
            f"⚡ Resilience Score: <b>{row['OrgResilienceScore']:.3f}</b><br>"
            f"📊 Betweenness: {row['Betweenness']:.3f}<br>"
            f"🚨 ระดับความเสี่ยง: <b>{rec['level']}</b>"
        )
    node_trace = go.Scatter(x=nx_,y=ny_,mode="markers+text",text=texts,
                            textposition="top center",textfont=dict(size=7,color="#8892a4"),
                            marker=dict(size=sizes,color=colors,line=dict(width=1,color="#0e1117")),
                            hovertext=hovers,hoverinfo="text")
    return go.Figure(data=[edge_trace,node_trace],
                     layout=go.Layout(paper_bgcolor="#0e1117",plot_bgcolor="#0e1117",
                                      margin=dict(l=10,r=10,t=10,b=10),
                                      showlegend=False,hovermode="closest",height=520,
                                      xaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
                                      yaxis=dict(showgrid=False,zeroline=False,showticklabels=False)))

def run_simulation(G, metric_df, remove_node):
    G2=G.copy(); G2.remove_node(remove_node)
    btw_b=nx.betweenness_centrality(G, normalized=True)
    btw_a=nx.betweenness_centrality(G2,normalized=True)
    affected=list(G.neighbors(remove_node))
    changes=[{"node":n,"แผนก":G.nodes[n]["department"],"ตำแหน่ง":G.nodes[n]["job_role"],
               "ภาระงานเพิ่มขึ้น":round(btw_a.get(n,0)-btw_b.get(n,0),4)} for n in affected]
    return {"lost_edges":G.number_of_edges()-G2.number_of_edges(),
            "affected_count":len(affected),
            "frag_increase":nx.number_connected_components(G2)-nx.number_connected_components(G),
            "components_after":nx.number_connected_components(G2),
            "changes_df":pd.DataFrame(changes).sort_values("ภาระงานเพิ่มขึ้น",ascending=False)}

def export_pdf(metric_df):
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate,Table,TableStyle,Paragraph,Spacer
        from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
        from reportlab.lib.units import cm
        buf=io.BytesIO()
        doc=SimpleDocTemplate(buf,pagesize=A4,leftMargin=2*cm,rightMargin=2*cm,topMargin=2*cm,bottomMargin=2*cm)
        styles=getSampleStyleSheet(); story=[]
        story.append(Paragraph("ONA Report — รายงานความเสี่ยงบุคลากร",
                                ParagraphStyle("t",parent=styles["Heading1"],fontSize=16,spaceAfter=6)))
        story.append(Paragraph("Organizational Network Analysis | IBM HR Dataset",styles["Normal"]))
        story.append(Spacer(1,.5*cm))
        high=len(metric_df[metric_df["OrgResilienceScore"]>=0.40])
        story.append(Paragraph(f"พนักงานทั้งหมด: {len(metric_df)} คน | ต้องดูแลเร่งด่วน: {high} คน",styles["Normal"]))
        story.append(Spacer(1,.4*cm))
        story.append(Paragraph("Top 20 — พนักงานที่มีความเสี่ยงสูงสุด",styles["Heading2"]))
        story.append(Spacer(1,.2*cm))
        top20=metric_df.nlargest(20,"OrgResilienceScore")[["EmployeeNumber","Department","JobRole","JobLevel","OrgResilienceScore","Betweenness"]]
        data=[["#","แผนก","ตำแหน่ง","Level","Score","Betweenness"]]
        for _,r in top20.iterrows():
            data.append([str(int(r["EmployeeNumber"])),r["Department"][:20],r["JobRole"][:20],
                         str(int(r["JobLevel"])),f"{r['OrgResilienceScore']:.3f}",f"{r['Betweenness']:.3f}"])
        tbl=Table(data,colWidths=[1.5*cm,4.5*cm,4.5*cm,1.5*cm,2*cm,2.5*cm])
        tbl.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#1a1f2e")),("TEXTCOLOR",(0,0),(-1,0),colors.white),
            ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),8),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#f8f9fa")]),
            ("GRID",(0,0),(-1,-1),.5,colors.HexColor("#dee2e6")),
            ("ALIGN",(0,0),(-1,-1),"CENTER"),("VALIGN",(0,0),(-1,-1),"MIDDLE"),("ROWHEIGHT",(0,0),(-1,-1),18),
        ]))
        story.append(tbl); doc.build(story); buf.seek(0)
        return buf.getvalue()
    except ImportError:
        buf2=io.StringIO(); metric_df.nlargest(20,"OrgResilienceScore").to_csv(buf2,index=False)
        return buf2.getvalue().encode("utf-8")


# ══════════════════════════════════════════════════════════════════════════════════
def main():
    if "welcomed" not in st.session_state:
        st.session_state.welcomed = False

    # ══ SIDEBAR ══════════════════════════════════════════════════════════════════
    with st.sidebar:
        st.markdown("## 🕸️ ONA Dashboard")
        st.caption("ระบบวิเคราะห์ความเสี่ยงบุคลากร")
        st.divider()

        uploaded_file = st.file_uploader("📂 อัปโหลดไฟล์ข้อมูลพนักงาน (CSV)", type="csv")
        st.markdown('<div class="sidebar-note">💡 ยังไม่มีไฟล์? กดข้ามได้เลย ระบบมีข้อมูลตัวอย่าง 150 คนให้</div>',
                    unsafe_allow_html=True)
        st.divider()

        # Expert mode toggle — ซ่อน slider ไว้ก่อน ไม่ให้ user ธรรมดางง
        expert = st.toggle("⚙️ ปรับค่าขั้นสูง", value=False,
                           help="เปิดเพื่อปรับน้ำหนักความสัมพันธ์ด้วยตัวเอง ถ้าไม่เปิดระบบจะใช้ค่าที่เหมาะสมที่สุดให้")
        if expert:
            st.markdown("**ปรับน้ำหนักความสัมพันธ์**")
            w_dept   = st.slider("แผนกเดียวกัน",     0.0,1.0,0.50,0.05)
            w_level  = st.slider("ระดับงานใกล้กัน",  0.0,1.0,0.25,0.05)
            w_role   = st.slider("ตำแหน่งเดียวกัน",  0.0,1.0,0.15,0.05)
            w_tenure = st.slider("อายุงานใกล้เคียง", 0.0,1.0,0.10,0.05)
            threshold = st.slider("ความหนาแน่น Network", 0.10,0.90,0.40,0.05)
        else:
            w_dept,w_level,w_role,w_tenure,threshold = 0.50,0.25,0.15,0.10,0.40
            st.markdown('<div class="sidebar-note">✅ ใช้ค่าเริ่มต้นที่ปรับแต่งมาแล้ว พร้อมใช้งานได้ทันที</div>',
                        unsafe_allow_html=True)

        WEIGHTS = {"department":w_dept,"job_level":w_level,"job_role":w_role,"tenure":w_tenure}
        st.divider()
        st.markdown("**สีของแผนก**")
        for dept,color in DEPT_COLORS.items():
            st.markdown(f"<span style='color:{color}'>●</span> {dept}", unsafe_allow_html=True)

    # ── Load Data ──────────────────────────────────────────────────────────────
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if "Attrition_flag" not in df.columns and "Attrition" in df.columns:
            df["Attrition_flag"] = (df["Attrition"]=="Yes").astype(int)
        data_label = f"📂 {uploaded_file.name}"
    else:
        df = load_default_data()
        data_label = "🧪 ข้อมูลตัวอย่าง 150 คน"

    df_json = df.to_json()
    metric_df, G = compute_metrics(df_json, threshold, WEIGHTS)

    # ══ HEADER ════════════════════════════════════════════════════════════════
    h1,h2 = st.columns([3,1])
    with h1:
        st.markdown("## 🕸️ ระบบวิเคราะห์ความเสี่ยงบุคลากรองค์กร")
        st.caption(f"Organizational Network Analysis · {data_label}")
    with h2:
        btn_label = "✕ ปิดคำแนะนำ" if not st.session_state.welcomed else "📖 วิธีใช้งาน"
        if st.button(btn_label, use_container_width=True):
            st.session_state.welcomed = not st.session_state.welcomed

    # ══ WELCOME BANNER ════════════════════════════════════════════════════════
    if not st.session_state.welcomed:
        st.markdown("""
        <div class="welcome-wrap">
            <div class="welcome-title">👋 ยินดีต้อนรับ — ระบบพร้อมใช้งานแล้ว ไม่ต้องตั้งค่าอะไรเพิ่ม</div>
            <div class="welcome-desc">
                Dashboard นี้ช่วย HR วิเคราะห์ว่า <b>พนักงานคนไหนสำคัญที่สุดต่อองค์กร</b>
                และถ้าคนนั้นลาออก จะกระทบใครบ้าง — เริ่มได้เลยจากขั้นตอนด้านล่าง
            </div>
            <div class="steps">
                <div class="step">
                    <div class="snum">1</div>
                    <div class="stxt"><b>ดูคำแนะนำ HR</b><br>Tab แรก "📋 คำแนะนำ HR" — บอกทันทีว่าต้องทำอะไรกับใคร</div>
                </div>
                <div class="step">
                    <div class="snum">2</div>
                    <div class="stxt"><b>จำลองการลาออก</b><br>Tab "⚡ จำลอง" — เลือกคนแล้วดูว่าถ้าลาออกจะพังแค่ไหน</div>
                </div>
                <div class="step">
                    <div class="snum">3</div>
                    <div class="stxt"><b>ดูภาพ Network</b><br>Tab "🌐 แผนที่องค์กร" — วงใหญ่ = คนสำคัญ hover เพื่อดูรายละเอียด</div>
                </div>
                <div class="step">
                    <div class="snum">4</div>
                    <div class="stxt"><b>อัปโหลดข้อมูล</b><br>ไม่จำเป็น — ใช้ข้อมูลตัวอย่างได้เลยหรือวาง CSV ที่ sidebar</div>
                </div>
            </div>
            <div class="dismiss">กด "✕ ปิดคำแนะนำ" ที่มุมบนขวาเพื่อซ่อน</div>
        </div>
        """, unsafe_allow_html=True)

    # ══ SUMMARY CARDS ═════════════════════════════════════════════════════════
    high_risk = metric_df[metric_df["OrgResilienceScore"]>0.50]
    critical  = metric_df[metric_df["OrgResilienceScore"]>0.60]
    kp = metric_df.loc[metric_df["Betweenness"].idxmax()] if not metric_df.empty else None
    kp_val  = f"#{int(kp['EmployeeNumber'])}" if kp is not None else "N/A"
    kp_role = kp["JobRole"] if kp is not None else ""

    c1,c2,c3,c4 = st.columns(4)
    for col,lbl,val,sub,cls in [
        (c1,"พนักงานทั้งหมด",   f"{G.number_of_nodes():,}", f"{metric_df['Department'].nunique()} แผนก",""),
        (c2,"ความสัมพันธ์ในองค์กร",f"{G.number_of_edges():,}","เส้นเชื่อมระหว่างพนักงาน",""),
        (c3,"บุคคลสำคัญที่สุด", kp_val,  kp_role,""),
        (c4,"ต้องดูแลเร่งด่วน", str(len(critical)), f"รวมความเสี่ยงสูง {len(high_risk)} คน","risk-high"),
    ]:
        with col:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">{lbl}</div>
                <div class="metric-value {cls}">{val}</div>
                <div class="metric-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ══ TABS — HR-first order ═════════════════════════════════════════════════
    tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs([
        "📋 คำแนะนำ HR",        # ← สิ่งที่ HR ต้องการก่อน
        "⚡ จำลองการลาออก",
        "🌐 แผนที่องค์กร",
        "📊 การวิเคราะห์เชิงลึก",
        "🏘️ กลุ่มในองค์กร",
        "🏢 สุขภาพแผนก",
    ])

    # ══════════════════════════════════════════════════════════════
    # TAB 1 — คำแนะนำ HR  (เข้าใจง่าย ใช้ได้ทันที)
    # ══════════════════════════════════════════════════════════════
    with tab1:
        st.markdown('<div class="hint">💡 <b>ใช้งานอย่างไร:</b> รายการด้านล่างเรียงจากเร่งด่วนมากที่สุด — กรองตามแผนกหรือระดับความเสี่ยงได้ที่ด้านบน แล้ว Export เป็น PDF ส่งทีมได้เลย</div>', unsafe_allow_html=True)

        # Filter bar
        f1,f2,f3 = st.columns([1,1,2])
        with f1:
            lvl = st.selectbox("ระดับความเสี่ยง",["ทั้งหมด","🔴 วิกฤต","🟡 สูง","🔵 ปานกลาง","🟢 ต่ำ"])
        with f2:
            dept_f = st.selectbox("แผนก",["ทั้งหมด"]+sorted(metric_df["Department"].unique().tolist()))
        with f3:
            top_n = st.slider("จำนวนที่แสดง", 5, 30, 10)

        filtered = metric_df.copy()
        if dept_f!="ทั้งหมด":
            filtered = filtered[filtered["Department"]==dept_f]
        lvl_map = {"🔴 วิกฤต":(0.60,1.0),"🟡 สูง":(0.40,0.60),"🔵 ปานกลาง":(0.25,0.40),"🟢 ต่ำ":(0,0.25)}
        if lvl!="ทั้งหมด":
            lo,hi = lvl_map[lvl]
            filtered = filtered[(filtered["OrgResilienceScore"]>=lo)&(filtered["OrgResilienceScore"]<hi)]

        recs = filtered.nlargest(top_n,"OrgResilienceScore")

        if recs.empty:
            st.info("ไม่พบพนักงานในระดับที่เลือก")
        else:
            for _,row in recs.iterrows():
                rec = get_recommendation(row)
                acts = "".join([f"<li>{a}</li>" for a in rec["actions"]])
                stars = "⭐"*int(row["Satisfaction"])+"☆"*(4-int(row["Satisfaction"]))
                st.markdown(f"""
                <div class="rec-card {rec['css']}">
                    <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:6px;">
                        <div>
                            <b style="color:#e8eaf0;font-size:.95rem;">{rec['icon']} พนักงาน #{int(row['EmployeeNumber'])}</b>
                            <span class="rbadge {rec['bc']}">{rec['level']}</span><br>
                            <span style="color:#8892a4;font-size:.82rem;">
                                {row['JobRole']} · {row['Department']} ·
                                Level {int(row['JobLevel'])} · อายุงาน {int(row['YearsAtCompany'])} ปี
                            </span>
                        </div>
                        <div style="text-align:right;font-size:.78rem;color:#5d6a80;">
                            คะแนนความเสี่ยง<br>
                            <b style="color:#e8eaf0;font-size:1.05rem;">{row['OrgResilienceScore']:.3f}</b><br>
                            ความพึงพอใจ {stars}
                        </div>
                    </div>
                    <div style="margin-top:.6rem;font-size:.82rem;color:#8892a4;font-weight:500;">สิ่งที่ HR ควรทำ:</div>
                    <ul style="margin:4px 0 0 16px;color:#c8d0e0;font-size:.85rem;line-height:1.75;">{acts}</ul>
                </div>
                """, unsafe_allow_html=True)

        st.divider()
        st.markdown("### 📥 Export รายงาน")
        e1,e2 = st.columns(2)
        with e1:
            csv = metric_df.to_csv(index=False).encode("utf-8")
            st.download_button("📊 ดาวน์โหลดข้อมูลทั้งหมด (.csv)",
                               csv,"ONA_full_data.csv","text/csv",use_container_width=True)
        with e2:
            if st.button("📄 สร้างรายงาน PDF",use_container_width=True):
                with st.spinner("กำลังสร้างรายงาน..."):
                    pdf_data = export_pdf(metric_df)
                st.download_button("⬇️ ดาวน์โหลด PDF",pdf_data,"ONA_Report.pdf","application/pdf",use_container_width=True)

    # ══════════════════════════════════════════════════════════════
    # TAB 2 — จำลองการลาออก
    # ══════════════════════════════════════════════════════════════
    with tab2:
        st.markdown('<div class="hint">💡 <b>ใช้งานอย่างไร:</b> เลือกชื่อพนักงานจากรายการ แล้วกด "จำลองผลกระทบ" — ระบบจะแสดงว่าถ้าคนนี้ลาออก ใครต้องรับภาระงานแทน และองค์กรสะเทือนแค่ไหน</div>', unsafe_allow_html=True)

        cs1,cs2 = st.columns([1,2])
        with cs1:
            top_r = metric_df.nlargest(30,"OrgResilienceScore").reset_index(drop=True)

            def fmt(eid):
                r = top_r[top_r.EmployeeNumber==eid]
                if r.empty: return str(eid)
                s = r["OrgResilienceScore"].values[0]
                icon = "🔴" if s>=0.60 else "🟡" if s>=0.40 else "🔵" if s>=0.25 else "🟢"
                return f"{icon} #{eid} — {r['JobRole'].values[0]} ({r['Department'].values[0]})"

            sel_id = st.selectbox("เลือกพนักงาน", options=top_r["EmployeeNumber"].tolist(), format_func=fmt)
            sr = metric_df[metric_df["EmployeeNumber"]==sel_id].iloc[0]
            rec = get_recommendation(sr)

            st.markdown("---")
            st.markdown(f"""
            <div style="background:#1a1f2e;border-radius:10px;padding:.9rem 1rem;">
                <div style="font-size:.78rem;color:#8892a4;margin-bottom:.5rem;text-transform:uppercase;letter-spacing:.05em;">ข้อมูลพนักงาน</div>
                <div style="color:#e8eaf0;font-size:.88rem;line-height:1.9;">
                    🏢 {sr['Department']}<br>
                    💼 {sr['JobRole']} (Level {int(sr['JobLevel'])})<br>
                    📅 อายุงาน {int(sr['YearsAtCompany'])} ปี<br>
                    🚨 ระดับความเสี่ยง: <b>{rec['icon']} {rec['level']}</b><br>
                    📊 คะแนน: <b>{sr['OrgResilienceScore']:.4f}</b>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("")
            run_btn = st.button("⚡ จำลองผลกระทบ", type="primary", use_container_width=True)

        with cs2:
            if run_btn:
                with st.spinner("กำลังจำลอง..."):
                    result = run_simulation(G, metric_df, sel_id)

                r1,r2,r3,r4 = st.columns(4)
                r1.metric("ความสัมพันธ์ที่หายไป", f"{result['lost_edges']}")
                r2.metric("คนที่ได้รับผลกระทบ",   f"{result['affected_count']} คน")
                r3.metric("กลุ่มแตกออกเพิ่ม",     f"{result['frag_increase']} กลุ่ม")
                r4.metric("กลุ่มทั้งหมดหลังจากนั้น",f"{result['components_after']}")

                st.markdown("---")
                if result["frag_increase"]>0:
                    st.error(f"🚨 **ผลกระทบรุนแรง** — ทีมแตกออกเป็น {result['components_after']} กลุ่มที่ไม่เชื่อมถึงกัน บางทีมจะขาดการสื่อสารกับองค์กรทันที")
                elif result["lost_edges"]>10:
                    st.warning(f"⚠️ **ผลกระทบสูง** — การทำงานร่วมกัน {result['lost_edges']} รายการจะหยุดชะงัก กระทบพนักงาน {result['affected_count']} คนโดยตรง")
                else:
                    st.success(f"✅ **ผลกระทบต่ำ** — องค์กรยังทำงานต่อได้ ผลกระทบจำกัดอยู่ที่ {result['affected_count']} คน")

                st.markdown("#### สิ่งที่ HR ควรทำต่อ")
                for a in rec["actions"]:
                    st.markdown(f"- {a}")

                if not result["changes_df"].empty:
                    st.markdown("#### พนักงานที่จะต้องรับภาระงานเพิ่ม")
                    st.caption("คนที่อยู่ในรายการนี้คือคนที่จะต้องทำหน้าที่ประสานงานแทน — HR ควรติดตามดูแลเป็นพิเศษ")
                    st.dataframe(result["changes_df"].head(8),use_container_width=True,hide_index=True)

                st.markdown("#### แผนที่องค์กร ก่อน vs หลัง")
                nv = list(G.neighbors(sel_id))+[sel_id]
                G_b = G.subgraph(nv[:60])
                G_ac = G.copy(); G_ac.remove_node(sel_id)
                nba = [n for n in nv if n!=sel_id]
                G_as = G_ac.subgraph(nba[:60])
                vc1,vc2 = st.columns(2)
                with vc1:
                    st.caption(f"✅ ก่อน — #{sel_id} ยังอยู่ (วงใหญ่)")
                    fb=draw_network(G_b,metric_df[metric_df["EmployeeNumber"].isin(nv)],highlight_node=sel_id)
                    fb.update_layout(height=280); st.plotly_chart(fb,use_container_width=True,key="sb")
                with vc2:
                    st.caption(f"❌ หลัง — ลบ #{sel_id} ออกแล้ว")
                    if nba:
                        fa=draw_network(G_as,metric_df[metric_df["EmployeeNumber"].isin(nba)])
                        fa.update_layout(height=280); st.plotly_chart(fa,use_container_width=True,key="sa")
                    else:
                        st.info("ไม่มี node เหลืออยู่")
            else:
                st.markdown("""
                <div class="empty-state">
                    <div class="empty-icon">⚡</div>
                    <div class="empty-title">เลือกพนักงาน แล้วกด "จำลองผลกระทบ"</div>
                    <div class="empty-desc">ระบบจะแสดงว่าถ้าพนักงานคนนี้ลาออก<br>
                    องค์กรจะสูญเสียการเชื่อมต่ออะไรบ้าง และใครต้องรับงานแทน</div>
                </div>
                """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════
    # TAB 3 — แผนที่องค์กร
    # ══════════════════════════════════════════════════════════════
    with tab3:
        st.markdown('<div class="hint">💡 <b>วิธีอ่านแผนที่:</b> <b>วงกลมใหญ่</b> = พนักงานสำคัญที่ HR ควรดูแลเป็นพิเศษ | <b>เส้น</b> = ความสัมพันธ์ในการทำงาน | <b>hover</b> ที่วงกลมเพื่อดูข้อมูลและระดับความเสี่ยง</div>', unsafe_allow_html=True)

        cc1,cc2 = st.columns([1,3])
        with cc1:
            st.markdown("**ตั้งค่าการแสดงผล**")
            color_by = st.selectbox("แสดงสีตาม",
                                    ["Department","Attrition","OrgResilienceScore"],
                                    format_func=lambda x: {"Department":"แผนก","Attrition":"ความเสี่ยงลาออก","OrgResilienceScore":"คะแนนความเสี่ยง"}.get(x,x))
            size_by  = st.selectbox("ขนาดวงกลมตาม",
                                    ["Betweenness","Degree","PageRank","OrgResilienceScore"],
                                    format_func=lambda x: {"Betweenness":"ความสำคัญเป็นสะพาน","Degree":"จำนวนคนที่ทำงานด้วย","PageRank":"อิทธิพลในองค์กร","OrgResilienceScore":"คะแนนความเสี่ยง"}.get(x,x))
            mn = G.number_of_nodes()
            sn = st.slider("จำนวนพนักงานที่แสดง", min(10,mn), mn, min(150,mn), 10)
        with cc2:
            Gs = G.subgraph(list(G.nodes())[:sn])
            ms = metric_df[metric_df["EmployeeNumber"].isin(list(G.nodes())[:sn])]
            st.plotly_chart(draw_network(Gs,ms,color_by,size_by), use_container_width=True)

    # ══════════════════════════════════════════════════════════════
    # TAB 4 — การวิเคราะห์เชิงลึก
    # ══════════════════════════════════════════════════════════════
    with tab4:
        st.markdown('<div class="hint">💡 <b>สำหรับ:</b> ผู้ที่ต้องการดูตัวเลขและกราฟเชิงสถิติโดยละเอียด — ถ้าต้องการแค่คำแนะนำ ดูที่ Tab "📋 คำแนะนำ HR" แทนได้เลย</div>', unsafe_allow_html=True)

        st.markdown("### Top 20 — พนักงานที่เป็น 'สะพาน' ขององค์กร (Betweenness)")
        st.caption("พนักงานที่แท่งสูง = ถ้าลาออกแล้วการสื่อสารระหว่างทีมจะขาดทันที")
        top20=metric_df.nlargest(20,"Betweenness")
        fig_b=px.bar(top20,x="EmployeeNumber",y="Betweenness",color="Department",
                     color_discrete_map=DEPT_COLORS,hover_data=["JobRole","JobLevel","OrgResilienceScore"],
                     template="plotly_dark")
        fig_b.update_layout(paper_bgcolor="#0e1117",plot_bgcolor="#0e1117",height=320,
                            xaxis_title="รหัสพนักงาน",yaxis_title="Betweenness Centrality")
        st.plotly_chart(fig_b,use_container_width=True)

        ca,cb = st.columns(2)
        with ca:
            fig_p=px.bar(metric_df.nlargest(15,"PageRank"),x="EmployeeNumber",y="PageRank",
                         color="Department",color_discrete_map=DEPT_COLORS,
                         title="Top 15 — PageRank (อิทธิพลในองค์กร)",template="plotly_dark")
            fig_p.update_layout(paper_bgcolor="#0e1117",plot_bgcolor="#0e1117",height=280)
            st.plotly_chart(fig_p,use_container_width=True)
        with cb:
            fig_d=px.bar(metric_df.nlargest(15,"Degree"),x="EmployeeNumber",y="Degree",
                         color="Department",color_discrete_map=DEPT_COLORS,
                         title="Top 15 — Degree (จำนวนคนที่ทำงานด้วย)",template="plotly_dark")
            fig_d.update_layout(paper_bgcolor="#0e1117",plot_bgcolor="#0e1117",height=280)
            st.plotly_chart(fig_d,use_container_width=True)

        st.markdown("### คะแนนความเสี่ยงรวม (Org Resilience Score) — Top 30")
        st.caption("สูตร: 0.35×Betweenness + 0.25×PageRank×10 + 0.20×Degree + 0.20×(1-Satisfaction)")
        fig_s=px.scatter(metric_df.nlargest(30,"OrgResilienceScore"),
                         x="Betweenness",y="OrgResilienceScore",size="Degree",
                         color="Department",color_discrete_map=DEPT_COLORS,
                         hover_data=["EmployeeNumber","JobRole","Attrition"],
                         template="plotly_dark",height=360)
        fig_s.update_layout(paper_bgcolor="#0e1117",plot_bgcolor="#0e1117")
        st.plotly_chart(fig_s,use_container_width=True)

    # ══════════════════════════════════════════════════════════════
    # TAB 5 — กลุ่มในองค์กร
    # ══════════════════════════════════════════════════════════════
    with tab5:
        st.markdown('<div class="hint">💡 <b>วิธีอ่าน:</b> ระบบค้นหากลุ่มที่ทำงานด้วยกันจริงๆ — ถ้าแท่งสีปนกัน แสดงว่ากลุ่มนั้นทำงานข้ามแผนก ซึ่งอาจต่างจาก org chart</div>', unsafe_allow_html=True)

        if G.number_of_edges()>0:
            partition=community_louvain.best_partition(G)
            cs2=pd.Series(partition).reset_index(); cs2.columns=["EmployeeNumber","Community"]
            mwc=metric_df.merge(cs2,on="EmployeeNumber")
            st.info(f"พบ **{mwc['Community'].nunique()} กลุ่ม** ในองค์กร — กลุ่มที่ทำงานด้วยกันจริงๆ จาก Louvain Algorithm")
            c1,c2=st.columns([2,1])
            with c1:
                cnt=mwc.groupby(["Community","Department"]).size().reset_index(name="Count")
                fig_c=px.bar(cnt,x="Community",y="Count",color="Department",
                             color_discrete_map=DEPT_COLORS,barmode="stack",
                             title="สมาชิกในแต่ละกลุ่ม แยกตามแผนก",template="plotly_dark")
                fig_c.update_layout(paper_bgcolor="#0e1117",plot_bgcolor="#0e1117",height=360,
                                    xaxis_title="กลุ่ม",yaxis_title="จำนวนสมาชิก")
                st.plotly_chart(fig_c,use_container_width=True)
            with c2:
                st.markdown("**สรุปแต่ละกลุ่ม**")
                cs3=mwc.groupby("Community").agg(สมาชิก=("EmployeeNumber","count"),
                    คะแนนเฉลี่ย=("OrgResilienceScore","mean"),อัตราลาออก=("Attrition","mean")).round(3).reset_index()
                cs3["อัตราลาออก"]=cs3["อัตราลาออก"].apply(lambda x:f"{x:.0%}")
                st.dataframe(cs3,use_container_width=True,hide_index=True)
        else:
            st.warning("ไม่มีความสัมพันธ์เพียงพอ กรุณาเปิด '⚙️ ปรับค่าขั้นสูง' ที่ sidebar แล้วลด Threshold ลง")

    # ══════════════════════════════════════════════════════════════
    # TAB 6 — สุขภาพแผนก
    # ══════════════════════════════════════════════════════════════
    with tab6:
        st.markdown('<div class="hint">💡 <b>วิธีอ่าน:</b> Health Score ยิ่งสูง = แผนกสุขภาพดี | ยิ่งต่ำ = แผนกพึ่งพาคนคนเดียวมากเกินไป หรือมีพนักงานลาออกสูง</div>', unsafe_allow_html=True)

        dh=metric_df.groupby("Department").agg(
            สมาชิก=("EmployeeNumber","count"),Avg_Betweenness=("Betweenness","mean"),
            Avg_Clustering=("Clustering","mean"),ความพึงพอใจเฉลี่ย=("Satisfaction","mean"),
            อัตราลาออก=("Attrition","mean"),Avg_Resilience=("OrgResilienceScore","mean"),
        ).round(4).reset_index()
        dh["HealthScore"]=(dh["ความพึงพอใจเฉลี่ย"]/4*0.40+dh["Avg_Clustering"]*0.30
                           -dh["อัตราลาออก"]*0.20-dh["Avg_Resilience"]*0.10).round(4)

        d1,d2=st.columns([1,2])
        with d1:
            disp=dh[["Department","สมาชิก","HealthScore","อัตราลาออก","ความพึงพอใจเฉลี่ย"]].copy()
            disp["อัตราลาออก"]=disp["อัตราลาออก"].apply(lambda x:f"{x:.1%}")
            disp["HealthScore"]=disp["HealthScore"].apply(lambda x:f"{x:.3f}")
            disp["ความพึงพอใจเฉลี่ย"]=disp["ความพึงพอใจเฉลี่ย"].apply(lambda x:f"{x:.2f}/4")
            st.dataframe(disp,use_container_width=True,hide_index=True)
        with d2:
            fig_h=px.bar(dh,x="Department",y="HealthScore",color="Department",
                         color_discrete_map=DEPT_COLORS,title="คะแนนสุขภาพแผนก (สูง = สุขภาพดี)",
                         template="plotly_dark",text="HealthScore")
            fig_h.update_traces(texttemplate="%{text:.3f}",textposition="outside")
            fig_h.update_layout(paper_bgcolor="#0e1117",plot_bgcolor="#0e1117",showlegend=False,height=300)
            st.plotly_chart(fig_h,use_container_width=True)

        st.markdown("### เปรียบเทียบแผนกแบบ Radar")
        cats=["Avg_Betweenness","Avg_Clustering","ความพึงพอใจเฉลี่ย","HealthScore"]
        fig_r=go.Figure()
        for _,row in dh.iterrows():
            vals=[row[c] for c in cats]
            fig_r.add_trace(go.Scatterpolar(r=vals+[vals[0]],theta=cats+[cats[0]],
                fill="toself",name=row["Department"],line_color=DEPT_COLORS.get(row["Department"],"#8892a4")))
        fig_r.update_layout(polar=dict(bgcolor="#1a1f2e",radialaxis=dict(visible=True,color="#8892a4"),
                            angularaxis=dict(color="#8892a4")),paper_bgcolor="#0e1117",
                            template="plotly_dark",legend=dict(bgcolor="#1a1f2e"),height=380)
        st.plotly_chart(fig_r,use_container_width=True)

        st.markdown("---")
        st.markdown("### Formal vs Informal Network")
        st.caption("เปรียบเทียบ org chart จริง (แถว) กับกลุ่มที่เกิดขึ้นจริง (คอลัมน์) — ช่องเข้ม = มีพนักงานอยู่มาก")
        if G.number_of_edges()>0:
            partition=community_louvain.best_partition(G)
            cs4=pd.Series(partition).reset_index(); cs4.columns=["EmployeeNumber","Community"]
            merged=metric_df.merge(cs4,on="EmployeeNumber")
            cross=pd.crosstab(merged["Department"],merged["Community"])
            fig_hm=px.imshow(cross,color_continuous_scale="Blues",
                             title="Heatmap: โครงสร้างองค์กรจริง vs กลุ่มที่เกิดขึ้นจริง",
                             template="plotly_dark")
            fig_hm.update_layout(paper_bgcolor="#0e1117",height=300,
                                 xaxis_title="กลุ่มที่เกิดขึ้นจริง",yaxis_title="แผนก")
            st.plotly_chart(fig_hm,use_container_width=True)
        else:
            st.warning("ไม่มีความสัมพันธ์เพียงพอ กรุณาปรับค่า Threshold ที่ sidebar")


if __name__ == "__main__":
    main()
# -*- coding: utf-8 -*-
"""
模型選擇地圖 × 專案成果儀表板
黃德麟 · 資料分析師
所有數字皆為本機實跑、追得到出處
"""
import streamlit as st
import plotly.graph_objects as go
import pandas as pd

st.set_page_config(page_title="模型選擇地圖 · 黃德麟", page_icon="🧭", layout="wide")

# ── 與作品集 index.html 同一組 STEADY 棕色調色盤 ──
ACCENT, GOLD, TEAL, RUST = "#6e4526", "#916a28", "#2f5250", "#a33b2a"
BROWN2, TAN, INK, MUTED = "#8a5a30", "#c3b7a0", "#37302a", "#5f5344"
C1, C2, C3, C4 = ACCENT, RUST, TEAL, GOLD  # 保留舊變數名相容，改指向新色碼
GRAY, LIGHTBLUE = TAN, GOLD

st.markdown("""
<style>
  html, body, [class*="css"] { font-family:"Noto Sans TC","Microsoft JhengHei",sans-serif; }
  [data-testid="stMetric"] { background:#ffffff; border:1px solid #e6ddce; border-radius:12px; padding:14px 16px; }
  [data-testid="stMetricLabel"] { color:#5f5344; }
  [data-testid="stMetricValue"] { color:#6e4526; }
  .stTabs [data-baseweb="tab"] { color:#5f5344; font-weight:700; }
  .stTabs [aria-selected="true"] { color:#6e4526 !important; }
  .stTabs [data-baseweb="tab-highlight"] { background-color:#6e4526 !important; }
  div[data-testid="stDataFrame"] { border:1px solid #e6ddce; border-radius:12px; overflow:hidden; }
</style>
""", unsafe_allow_html=True)

st.title("🧭 什麼狀況用什麼模型？")
st.caption("資料分析專案成果儀表板 · 黃德麟 · 數字全部來自本機實跑、追得到出處")

tab1, tab2, tab3 = st.tabs(["📍 模型選擇地圖", "📊 專案成果真數字", "🧠 決策原則（怎麼選）"])

# ════════ Tab 1 · 模型選擇地圖 ════════
SCENARIOS = {
    "我想知道哪些客戶快流失了": dict(
        model="監督式分類 · 決策樹", day="客戶流失預測", family="監督式學習（y=離散標籤）",
        why="有歷史答案欄（流失/沒流失），y 是類別 → 分類問題。",
        result="Test Recall 75.9%、AUC 0.811。反直覺實證：baseline「永遠猜留客」Accuracy 80.7% 竟贏過決策樹模型 75.3%，但 Recall=0%——一個流失客戶都抓不到。換算成錢：29 位實際流失客戶消費金額合計 NT\\$592,602，模型抓到的 22 人只佔 NT\\$394,456（66.6%）——漏掉的 7 人藏著更高比例的高價值客戶。",
        lesson="類別不平衡時 Accuracy 會騙人，業務要看 Recall（漏掉真流失的代價 >> 誤判的挽留成本）。"),
    "我想知道下個月會賣多少": dict(
        model="時間序列回歸 · Prophet + Baseline", day="銷量預測", family="監督式學習（y=連續數值＋時間軸）",
        why="y 是連續數字且有時間結構 → 回歸/時序問題。",
        result="Prophet 勝 3/5 個 SKU（優於 Baseline ≥2pp 才算贏）；SKU_001 解讀出雙十一單月加成 +552%、年趨勢 +13.5%。",
        lesson="兩條紀律：①Baseline first——最笨的「去年同月」就贏過一半場景；②時序絕不能隨機 shuffle 切分（data leakage：正確切 12.6% vs 錯誤切 28.1%）。"),
    "我想知道什麼商品會被一起買": dict(
        model="關聯規則 · Apriori", day="購物籃關聯分析", family="非監督式學習（無 y，找共現結構）",
        why="沒有答案欄，要從交易紀錄自己挖「誰跟誰常一起出現」。",
        result="Top 規則：尿布↔啤酒 Lift 7.32、紅酒↔起司 7.20、醬油→米 5.18（3,300 筆交易實跑）。實測 Lift 遠高於資料本身的設計假設（>1.5），給定的參考值也要自己驗證。",
        lesson="Lift 一票否決：Confidence 93% 的規則若 Lift≈1（如塑膠袋），是「人人都買」不是「關聯」。關聯≠因果。"),
    "我想把客戶分成幾種人": dict(
        model="非監督分群 · K-means", day="客戶分群", family="非監督式學習（無 y，找相似群）",
        why="沒有預先定義的類別，讓演算法依 RFM 相似度自己分群。",
        result="1,500 位客戶分 4 群：VIP高頻高額 134 / 穩定中段 750 / 流失高風險 504 / 沉睡 112。不標準化會崩潰成 [300,1,1,1198]——兩個離群 VIP 各自成群。",
        lesson="①先 Log+標準化，否則 Monetary 吃掉 99% 的距離；②k 聽業務的：Silhouette 最高是 k=2，但業務只能管 4-6 群 → 選 k=4。"),
    "我想聽懂客訴在罵什麼": dict(
        model="LLM 文本分類（Gemini API）", day="客訴文本分析", family="LLM 應用（非結構化文本）",
        why="客訴是自由文字（含 emoji、錯字），傳統模型要先大量標註，LLM zero-shot 直接分。",
        result="200 筆客訴：情感四級+痛點分類，準確率 60%（誠實低於預期 75-85%，負向vs強烈負向邊界模糊）。交叉表挖出盲點：貨損類「強烈負向」佔比 25% 全類最高（雖然筆數不是最多）。",
        lesson="模型不完美時交叉分析仍能創造業務價值；「先洗乾淨才能跑」的傳統 SOP 被 LLM 翻轉。"),
    "我想知道車怎麼跑最省": dict(
        model="最佳化求解 · OR-Tools（CVRPTW）", day="配送路徑最佳化", family="作業研究／最佳化（不是預測，是求解）",
        why="這不是「預測會發生什麼」，是「在容量+時窗限制下找最好的方案」——NP-hard，20 站就有 20! 種排列。",
        result="人工 baseline NT\\$3,148 看似能跑、實藏 4 條時窗違反+2 台超工時（不可行）；OR-Tools NT\\$3,044、0 違反（可行），省 3.3%。",
        lesson="別只賣省錢，賣「0 違規能跑的解」——從不可行到可行的價值遠大於那 3.3%。單次跑分不能當定論（GLS vs SA 差 0.03% 是巧合，隨機演算法本就有 5-15% 變異屬正常）。"),
    "我想一張畫面看完全部模型": dict(
        model="資料整合 · SQLite + Streamlit + MDM 治理", day="系統整合", family="資料工程／治理（不是模型，是地基）",
        why="四個模型輸出散在四個 CSV，主管要的是「這個客戶綜合怎麼處理」——用 customer_id 一把鍵 JOIN 出客戶 360。",
        result="50 客戶×4 模型整合 6 張表；LEFT JOIN 保住 30 位散戶（vehicle_id=NULL 是業務事實不是資料壞掉）；MDM 8 欄位各指派唯一 Master。",
        lesson="整合難在治理不在技術：Master 是「權威來源」不是「資料量最大」（客戶地址 Master=CRM 不是 ERP）；越即時≠越好（對數成本曲線，Real-time=15×）。"),
    "我想讓機器自己盯庫存自己發警報": dict(
        model="AI Agent · Claude Code（tool use）", day="AI Agent 自動化", family="Agent 自動化（會「做」不只會「說」）",
        why="LLM 會說、Agent 會做——給工具（Read/Write/Bash）+ 工作手冊（CLAUDE.md）+ 門禁卡（settings.local.json），它就能自己讀 CSV→判斷→寫警報檔。",
        result="30 SKU 三級分類（緊急5/普通8/正常17）Agent 自動產出 alert 檔；越權測試：叫它刪 inventory.csv 被 deny list 攔下。",
        lesson="用 Agent 前三道門檻：需要自主決策嗎？要讀寫真實資料嗎？風險可控嗎？越給自由越要 guardrail。"),
    "我想先看清現況再說": dict(
        model="描述統計 + RFM / ABC / EIQ / IQR", day="基礎資料分析", family="描述統計（一切的地基）",
        why="四階分析階梯的第一階：沒把「發生什麼」做好，就沒資格談「預測什麼」。",
        result="OTD 79.6% 揪出病灶路線 R-03（96% 異常集中）；儲位重排 6.7 格→2.6 格省 61.6%；三表 Merge 抓出 12 倍採購積壓；RFM 八分群、Top20% 客戶實際只佔 30% 運費（80/20 是迷思要實算）。",
        lesson="平均數會騙人（用中位數）、80/20 是啟發不是鐵律、異常用 IQR±1.5 倍——每個數字都要自己算過。"),
}

with tab1:
    st.subheader("選一個你想解決的問題")
    q = st.selectbox("你的業務問題是：", list(SCENARIOS.keys()))
    s = SCENARIOS[q]
    a, b = st.columns([1, 2])
    with a:
        st.metric("該用的方法", s["model"])
        st.caption(f'{s["day"]}｜{s["family"]}')
    with b:
        st.markdown(f'**為什麼是它**：{s["why"]}')
        st.markdown(f'**我的實跑結果**：{s["result"]}')
        st.info(f'💡 帶走的教訓：{s["lesson"]}')
    st.divider()
    st.subheader("完整對照表（九個專案）")
    df_map = pd.DataFrame([
        dict(業務問題=k, 方法=v["model"], 對應=v["day"], 家族=v["family"]) for k, v in SCENARIOS.items()
    ])
    st.dataframe(df_map, use_container_width=True, hide_index=True)

# ════════ Tab 2 · 真數字成果 ════════
with tab2:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("雙十一單月加成", "+552%", "Prophet 銷量解讀")
    m2.metric("流失預測 Recall", "75.9%", "baseline 陷阱實證")
    m3.metric("最強關聯規則 Lift", "7.32", "尿布 ↔ 啤酒")
    m4.metric("路徑成本", "$3,044", "-3.3%，0 違規", delta_color="inverse")
    st.divider()

    colA, colB = st.columns(2)
    with colA:
        st.markdown("**用對指標驗收模型：baseline Accuracy 贏、Recall 掛零**")
        fig = go.Figure()
        fig.add_bar(name="baseline（永遠猜留客）", x=["Accuracy", "Recall"], y=[80.7, 0],
                    marker_color=GOLD, text=["80.7%", "0%"], textposition="outside")
        fig.add_bar(name="決策樹模型", x=["Accuracy", "Recall"], y=[75.3, 75.9],
                    marker_color=C1, text=["75.3%", "75.9%"], textposition="outside")
        fig.update_layout(barmode="group", height=320, margin=dict(t=10, b=10),
                          yaxis=dict(range=[0, 100], title="%"), legend=dict(orientation="h", y=1.12),
                          plot_bgcolor="white", bargap=0.35)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("類別不平衡下 Accuracy 會誤導——決策樹模型 Accuracy 較低，卻抓得到 76% 的真流失客戶。")

    with colB:
        st.markdown("**購物籃關聯：Top 3 規則（Lift）**")
        rules = ["醬油 → 米", "紅酒 ↔ 起司", "尿布 ↔ 啤酒"]
        lifts = [5.18, 7.20, 7.32]
        fig = go.Figure(go.Bar(x=lifts, y=rules, orientation="h", marker_color=[GOLD, BROWN2, ACCENT],
                               text=[f"{v:.2f}" for v in lifts], textposition="outside"))
        fig.add_vline(x=1, line_dash="dash", line_color=MUTED,
                      annotation_text="Lift=1 統計獨立", annotation_position="bottom right")
        fig.update_layout(height=320, margin=dict(t=10, b=10), xaxis=dict(range=[0, 8.6], title="Lift"),
                          plot_bgcolor="white", bargap=0.4)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("實測 Lift 遠高於資料本身的設計假設（>1.5）——給定的參考值也要自己驗證。")

    colC, colD = st.columns(2)
    with colC:
        st.markdown("**配送路徑最佳化：重點不是省 3.3%，是從「不可行」到「可行」**")
        fig = go.Figure(go.Bar(
            x=["人工 baseline", "OR-Tools"], y=[3148, 3044],
            marker_color=[RUST, ACCENT],
            text=["$3,148<br>⚠ 4 條時窗違反（不可行）", "$3,044<br>✓ 0 違反（可行）"],
            textposition="outside"))
        fig.update_layout(height=340, margin=dict(t=30, b=10), yaxis=dict(range=[0, 3900], title="總成本 (NT$)"),
                          plot_bgcolor="white", bargap=0.45)
        st.plotly_chart(fig, use_container_width=True)

    with colD:
        st.markdown("**客戶分群：K-means 四群客戶（1,500 位）**")
        segs = ["VIP 高頻高額", "穩定中段", "流失高風險", "沉睡客戶"]
        counts = [134, 750, 504, 112]
        fig = go.Figure(go.Bar(x=segs, y=counts, marker_color=[GOLD, TEAL, RUST, TAN],
                               text=counts, textposition="outside"))
        fig.update_layout(height=340, margin=dict(t=10, b=10), yaxis=dict(title="人數"),
                          plot_bgcolor="white", bargap=0.4)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Silhouette 最高其實是 k=2，但業務只能管 4-6 群 → 選 k=4（ML×業務雙約束）。")

    st.markdown("**倉儲儲位重排：平均揀貨移動距離**")
    fig = go.Figure(go.Bar(x=["重排前", "重排後"], y=[6.7, 2.6], marker_color=[LIGHTBLUE, C1],
                           text=["6.7 格", "2.6 格（-61.6%）"], textposition="outside"))
    fig.update_layout(height=280, margin=dict(t=10, b=10), yaxis=dict(range=[0, 8.6], title="平均格數"),
                      plot_bgcolor="white", bargap=0.55)
    st.plotly_chart(fig, use_container_width=True)

# ════════ Tab 3 · 決策原則 ════════
with tab3:
    st.subheader("拿到一個業務問題，照這個順序問自己")
    st.markdown("""
| 問題 | 答案 A | 答案 B |
|---|---|---|
| **Q0. 現況看清楚了嗎？** | 還沒 → **先做描述統計**（中位數/IQR/RFM/ABC），沒有第一階就沒有上面全部 | 清楚了 → 往下 |
| **Q1. 資料是自由文字嗎？** | 是 → **LLM 文本分類** | 不是 → 往下 |
| **Q2. 有沒有歷史「答案欄 y」？** | 有 → 監督式：y 離散 → **分類**；y 連續 → **回歸/時序** | 沒有 → 非監督：找共現 → **Apriori**；找相似 → **K-means** |
| **Q3. 要的不是預測，是「限制下的最佳方案」？** | 是 → **最佳化求解 OR-Tools** | 不是 → 往下 |
| **Q4. 模型都有了但散在四處？** | 是 → **整合 SQLite+BI**，先治理（誰是 Master）再談技術 | — |
| **Q5. 要機器自動執行、不只給人看？** | 是 → **AI Agent**，過三道門檻再上，deny list 不能少 | 否 → BI 呈現給人決策就好 |
""")
    st.divider()
    st.subheader("九個專案最值錢的五個反直覺")
    st.markdown("""
1. **Accuracy 會騙人**——baseline 80.7% 贏決策樹模型，但 Recall=0%
2. **給定的參考值也要驗證**——Lift 實測 7.32 vs 設計假設 >1.5；資料庫大小實測 45KB vs 預期 12-15KB，照實填+註記
3. **別只賣省錢，賣「可行」**——NT\\$3,148 的排法根本跑不動
4. **越即時≠越好**——對數成本曲線，99% 場景日批次就夠
5. **越給 AI 自由，越要 guardrail**——Agent 強在「做」，所以 deny list 是配套不是選配
""")
    st.caption("黃德麟 · 資料分析師——用對的方法解決對的問題，每個結論都追得到出處。")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO

# 设置页面标题和布局
st.set_page_config(page_title="通用数据可视化平台", layout="wide")
st.title("📊 数据可视化平台")

# ======================
# 数据上传与预处理
# ======================
uploaded_file = st.sidebar.file_uploader(
    "上传数据文件 (CSV/Excel)", 
    type=["csv", "xlsx"]
)

# 动态生成示例数据（如果用户未上传）
@st.cache_data
def generate_demo_data():
    date_rng = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
    return pd.DataFrame({
        "日期": date_rng,
        "销售额": np.random.normal(1000, 200, len(date_rng)).cumsum(),
        "产品类别": np.random.choice(["A", "B", "C"], len(date_rng)),
        "客户评分": np.random.randint(1, 6, len(date_rng)),
        "广告费用": np.abs(np.random.normal(500, 100, len(date_rng)))
    })

df = generate_demo_data() if uploaded_file is None else pd.read_csv(uploaded_file)

# 显示数据预览
with st.expander("数据预览 (前5行)"):
    st.dataframe(df.head())

# ======================
# 图表配置侧边栏
# ======================
st.sidebar.header("图表配置")
chart_type = st.sidebar.selectbox(
    "选择图表类型",
    ["折线图", "柱状图", "散点图", "直方图", "箱线图", "饼图"]
)

library = st.sidebar.radio("选择可视化库", ["Plotly", "Matplotlib"])

# 动态参数配置（根据图表类型变化）
params = {}
if chart_type in ["折线图", "柱状图", "散点图"]:
    params["x"] = st.sidebar.selectbox("X轴字段", df.columns)
    params["y"] = st.sidebar.selectbox("Y轴字段", df.columns)
    if chart_type == "散点图":
        params["color"] = st.sidebar.selectbox("颜色字段", [None] + list(df.columns))
        params["size"] = st.sidebar.selectbox("大小字段", [None] + list(df.columns))

elif chart_type == "直方图":
    params["column"] = st.sidebar.selectbox("选择数值列", df.select_dtypes(include=np.number).columns)
    params["bins"] = st.sidebar.slider("分箱数量", 5, 50, 20)

elif chart_type == "箱线图":
    params["category"] = st.sidebar.selectbox("分类字段", df.select_dtypes(exclude=np.number).columns)
    params["value"] = st.sidebar.selectbox("数值字段", df.select_dtypes(include=np.number).columns)

elif chart_type == "饼图":
    params["names"] = st.sidebar.selectbox("分类字段", df.select_dtypes(exclude=np.number).columns)
    params["values"] = st.sidebar.selectbox("数值字段", df.select_dtypes(include=np.number).columns)

# ======================
# 图表生成核心逻辑
# ======================
def generate_chart(df, chart_type, library, params):
    try:
        if library == "Plotly":
            if chart_type == "折线图":
                fig = px.line(df, x=params["x"], y=params["y"], title=f"{params['y']} 趋势")
            elif chart_type == "柱状图":
                fig = px.bar(df, x=params["x"], y=params["y"], title=f"{params['y']} 分布")
            elif chart_type == "散点图":
                fig = px.scatter(df, x=params["x"], y=params["y"], 
                               color=params["color"], size=params["size"],
                               title="散点关系分析")
            elif chart_type == "直方图":
                fig = px.histogram(df, x=params["column"], nbins=params["bins"],
                                 title=f"{params['column']} 分布直方图")
            elif chart_type == "箱线图":
                fig = px.box(df, x=params["category"], y=params["value"],
                           title=f"{params['value']} 箱线分析")
            elif chart_type == "饼图":
                fig = px.pie(df, names=params["names"], values=params["values"],
                           title=f"{params['names']} 占比分析")
            return fig
        
        elif library == "Matplotlib":
            fig, ax = plt.subplots(figsize=(8,4))
            if chart_type == "折线图":
                df.plot(x=params["x"], y=params["y"], ax=ax, linestyle='--', marker='o')
                ax.set_title(f"{params['y']} 趋势 (Matplotlib)")
            elif chart_type == "柱状图":
                df[params["y"]].plot(kind='bar', ax=ax, color='skyblue')
                ax.set_title(f"{params['y']} 分布 (Matplotlib)")
            elif chart_type == "散点图":
                ax.scatter(df[params["x"]], df[params["y"]], 
                          c=df[params["color"]] if params["color"] else None,
                          s=df[params["size"]]*10 if params["size"] else 20)
                ax.set_title("散点关系分析 (Matplotlib)")
            elif chart_type == "直方图":
                ax.hist(df[params["column"]], bins=params["bins"], color='green', alpha=0.7)
                ax.set_title(f"{params['column']} 分布直方图 (Matplotlib)")
            elif chart_type == "箱线图":
                df.boxplot(column=params["value"], by=params["category"], ax=ax)
                ax.set_title(f"{params['value']} 箱线分析 (Matplotlib)")
            elif chart_type == "饼图":
                df.groupby(params["names"])[params["values"]].sum().plot(
                    kind='pie', autopct='%1.1f%%', ax=ax
                )
                ax.set_title(f"{params['names']} 占比分析 (Matplotlib)")
            return fig
            
    except Exception as e:
        st.error(f"图表生成错误: {str(e)}")
        return None

# ======================
# 主界面渲染
# ======================
if st.sidebar.button("生成图表"):
    chart = generate_chart(df, chart_type, library, params)
    if chart:
        if library == "Plotly":
            st.plotly_chart(chart, use_container_width=True)
        elif library == "Matplotlib":
            st.pyplot(chart)
else:
    st.info("👈 请在左侧配置图表参数后点击【生成图表】")
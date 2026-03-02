"""Professional light theme utilities for Streamlit pages."""

CUSTOM_CSS = """
<style>
    :root {
        --bg-main: #FFFFFF;
        --bg-soft: #F6F8FB;
        --bg-soft-2: #F2F5FA;
        --border: #E5EAF0;
        --border-strong: #D7E0EB;
        --text-main: #1F2937;
        --text-muted: #5F6B7A;
        --primary: #0B5ED7;
        --success: #0F9D58;
        --danger: #D93025;
        --shadow-soft: 0 1px 2px rgba(15, 23, 42, 0.04), 0 6px 18px rgba(15, 23, 42, 0.03);
        --radius-card: 12px;
    }

    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }

    h1, h2, h3, h4, h5, h6 {
        color: var(--text-main) !important;
        letter-spacing: -0.01em;
        line-height: 1.15;
    }

    p, li, label, span {
        color: var(--text-main);
    }

    p, li {
        line-height: 1.55;
    }

    small, [data-testid="stCaptionContainer"] {
        color: var(--text-muted) !important;
    }

    code {
        background: #F3F6FB;
        border: 1px solid #E7EDF4;
        border-radius: 6px;
        padding: 0.08rem 0.32rem;
        font-size: 0.92em;
    }
    pre code {
        background: transparent;
        border: none;
        padding: 0;
    }

    [data-testid="stAlert"] {
        border-radius: var(--radius-card);
        border: 1px solid var(--border) !important;
    }

    /* KPI cards */
    .stMetric {
        background: var(--bg-soft);
        border: 1px solid var(--border);
        border-radius: var(--radius-card);
        padding: 0.9rem 1rem;
        box-shadow: var(--shadow-soft);
        min-height: 5.15rem;
    }
    .stMetric label {
        color: var(--text-muted) !important;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: var(--text-main) !important;
        font-weight: 700;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #F7FAFF;
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span {
        color: var(--text-main) !important;
    }

    /* Plotly/chart containers */
    [data-testid="stPlotlyChart"],
    [data-testid="stVegaLiteChart"],
    [data-testid="stCustomComponentV2"] {
        background: var(--bg-main);
        border: 1px solid var(--border);
        border-radius: var(--radius-card);
        box-shadow: var(--shadow-soft);
        padding: 0.35rem 0.35rem 0.15rem;
        margin-bottom: 0.25rem;
    }

    /* Dataframe / table border polish */
    [data-testid="stDataFrame"],
    [data-testid="stTable"] {
        border: 1px solid var(--border);
        border-radius: var(--radius-card);
        overflow: hidden;
        box-shadow: var(--shadow-soft);
        background: var(--bg-main);
    }

    [data-testid="stDataFrame"] [role="grid"] {
        border-radius: var(--radius-card);
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 0.45rem 0.9rem;
    }
    .stTabs [role="tabpanel"] {
        padding-top: 0.35rem;
    }

    /* Segmented controls / pills */
    [data-testid="stSegmentedControl"] {
        margin-bottom: 0.35rem;
    }
    [data-testid="stSegmentedControl"] [role="radiogroup"] {
        background: var(--bg-soft);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 2px;
    }
    [data-testid="stSegmentedControl"] button {
        border-radius: 8px !important;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 9px;
        border-color: var(--border-strong);
    }
    .stButton > button[kind="secondary"] {
        background: var(--bg-main);
    }

    /* Expander */
    [data-testid="stExpander"] {
        border: 1px solid var(--border);
        border-radius: var(--radius-card);
        box-shadow: var(--shadow-soft);
        background: var(--bg-main);
        overflow: hidden;
    }
    .streamlit-expanderHeader {
        color: var(--text-main) !important;
        font-weight: 600;
        background: linear-gradient(180deg, #FBFCFE 0%, #F7FAFF 100%);
        border-bottom: 1px solid #EFF3F8;
    }

    /* Images */
    [data-testid="stImage"] img {
        border-radius: var(--radius-card);
        border: 1px solid var(--border);
    }

    hr {
        border: none;
        border-top: 1px solid var(--border);
        margin: 1rem 0 0.9rem;
    }

    /* Plotly backgrounds should stay white */
    .js-plotly-plot .plotly .main-svg {
        background: transparent !important;
    }
    .js-plotly-plot .plotly .xtick text,
    .js-plotly-plot .plotly .ytick text {
        fill: #5F6B7A !important;
    }
</style>
"""


def inject_custom_css():
    """Inject custom CSS into the Streamlit app."""
    import streamlit as st

    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


PLOTLY_TEMPLATE = {
    "layout": {
        "paper_bgcolor": "white",
        "plot_bgcolor": "white",
        "font": {"color": "#1F2937", "family": "Arial, sans-serif"},
        "xaxis": {
            "gridcolor": "#E6ECF2",
            "zerolinecolor": "#CBD5E1",
            "linecolor": "#CBD5E1",
            "automargin": True,
            "ticks": "outside",
            "tickfont": {"color": "#5F6B7A"},
        },
        "yaxis": {
            "gridcolor": "#E6ECF2",
            "zerolinecolor": "#CBD5E1",
            "linecolor": "#CBD5E1",
            "automargin": True,
            "ticks": "outside",
            "tickfont": {"color": "#5F6B7A"},
        },
        "colorway": [
            "#0B5ED7",
            "#198754",
            "#F59F00",
            "#DC3545",
            "#6F42C1",
            "#0DCAF0",
            "#FD7E14",
            "#20C997",
        ],
        "margin": {"l": 60, "r": 20, "t": 52, "b": 48},
        "legend": {
            "bgcolor": "rgba(255,255,255,0.9)",
            "bordercolor": "#E5EAF0",
            "borderwidth": 1,
            "font": {"color": "#1F2937"},
        },
        "hoverlabel": {"bgcolor": "white", "bordercolor": "#D7E0EB", "font": {"color": "#1F2937"}},
    }
}

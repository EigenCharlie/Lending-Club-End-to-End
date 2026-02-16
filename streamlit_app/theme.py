"""Professional light theme utilities for Streamlit pages."""

CUSTOM_CSS = """
<style>
    :root {
        --bg-main: #FFFFFF;
        --bg-soft: #F6F8FB;
        --border: #E5EAF0;
        --text-main: #1F2937;
        --text-muted: #5F6B7A;
        --primary: #0B5ED7;
        --success: #0F9D58;
        --danger: #D93025;
    }

    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }

    h1, h2, h3, h4, h5, h6 {
        color: var(--text-main) !important;
        letter-spacing: -0.01em;
    }

    p, li, label, span {
        color: var(--text-main);
    }

    /* KPI cards */
    .stMetric {
        background: var(--bg-soft);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 0.9rem 1rem;
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

    /* Dataframe border polish */
    [data-testid="stDataFrame"] {
        border: 1px solid var(--border);
        border-radius: 8px;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 0.45rem 0.9rem;
    }

    /* Expander */
    .streamlit-expanderHeader {
        color: var(--text-main) !important;
        font-weight: 600;
    }

    /* Plotly backgrounds should stay white */
    .js-plotly-plot .plotly .main-svg {
        background: transparent !important;
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
        },
        "yaxis": {
            "gridcolor": "#E6ECF2",
            "zerolinecolor": "#CBD5E1",
            "linecolor": "#CBD5E1",
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
        "legend": {"bgcolor": "rgba(255,255,255,0.85)", "bordercolor": "#E5EAF0", "borderwidth": 1},
    }
}


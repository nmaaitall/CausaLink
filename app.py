import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pgmpy.estimators import PC
import networkx as nx
from io import BytesIO
from datetime import datetime

st.set_page_config(page_title="CausaLink", page_icon="🔗", layout="wide")


def export_graph_as_png(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    return buf


st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        text-align: center;
        color: #7f7f7f;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
        border-radius: 0.5rem;
        padding: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<p class="main-header">CausaLink</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Causal Inference Platform - Discover True Causal Relationships</p>',
            unsafe_allow_html=True)

st.sidebar.header("Navigation")
page = st.sidebar.radio("Select Page", ["Data Upload", "Data Exploration", "Causal Analysis"])

if 'df' in st.session_state and st.session_state['df'] is not None:
    st.sidebar.success("Dataset loaded")
    st.sidebar.metric("Rows", f"{st.session_state['df'].shape[0]:,}")
    st.sidebar.metric("Columns", st.session_state['df'].shape[1])
    numeric_count = len(st.session_state['df'].select_dtypes(include=[np.number]).columns)
    st.sidebar.metric("Numeric Columns", numeric_count)

if page == "Data Upload":
    st.header("Step 1: Upload Your Dataset")

    with st.expander("Dataset Requirements", expanded=False):
        st.write("- File format: CSV")
        st.write("- Maximum 10,000 rows")
        st.write("- Maximum 50 columns")
        st.write("- Text columns will be automatically converted to numeric codes")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, sep=';')
        except:
            df = pd.read_csv(uploaded_file, sep=',')

        df = df.dropna(axis=1, how='all')

        original_types = df.dtypes.to_dict()

        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except:
                    pass

        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype('category').cat.codes.astype('float64')
                df[col] = df[col].replace(-1.0, np.nan)

        if df.shape[0] > 10000:
            st.error(f"Dataset too large! Your file has {df.shape[0]:,} rows. Maximum allowed is 10,000 rows.")
            st.info("Please filter your data and try again with a smaller file.")
        elif df.shape[1] > 50:
            st.error(f"Too many columns! Your file has {df.shape[1]} columns. Maximum allowed is 50 columns.")
            st.info("Please select fewer columns and try again.")
        else:
            st.session_state['df'] = df
            st.session_state['original_df'] = df.copy()
            st.session_state['original_types'] = original_types

            st.success("File uploaded successfully! Text columns converted to numeric codes.")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Rows", f"{df.shape[0]:,}")
            with col2:
                st.metric("Total Columns", df.shape[1])
            with col3:
                st.metric("Missing Values", f"{df.isnull().sum().sum():,}")
            with col4:
                numeric_count = len(df.select_dtypes(include=[np.number]).columns)
                st.metric("Numeric Columns", numeric_count)

            st.write("Dataset Preview:")
            st.dataframe(
                df.head(20),
                use_container_width=True,
                height=400
            )

            st.write("Column Information:")
            col_info = []
            for col in df.columns:
                original_type = str(original_types.get(col, 'unknown'))
                current_type = str(df[col].dtype)
                converted = "(converted)" if original_type == 'object' and current_type != 'object' else ""

                col_info.append({
                    'Column': col,
                    'Original Type': original_type,
                    'Current Type': current_type + " " + converted,
                    'Non-Null': df[col].count(),
                    'Null': df[col].isnull().sum(),
                    'Unique': df[col].nunique()
                })

            col_info_df = pd.DataFrame(col_info)
            st.dataframe(col_info_df, use_container_width=True, height=400)

            if any('converted' in str(x) for x in col_info_df['Current Type']):
                st.info("Text columns were automatically converted to numeric codes for analysis.")

    elif 'df' in st.session_state and st.session_state['df'] is not None:
        st.info("Dataset already loaded. Upload a new file to replace it.")

        df = st.session_state['df']

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Rows", f"{df.shape[0]:,}")
        with col2:
            st.metric("Total Columns", df.shape[1])
        with col3:
            st.metric("Missing Values", f"{df.isnull().sum().sum():,}")
        with col4:
            numeric_count = len(df.select_dtypes(include=[np.number]).columns)
            st.metric("Numeric Columns", numeric_count)

        st.write("Current Dataset Preview:")
        st.dataframe(df.head(20), use_container_width=True, height=400)

    else:
        st.info("Please upload your CSV file to begin analysis")

elif page == "Data Exploration":
    st.header("Step 2: Explore Your Data")

    if 'df' not in st.session_state or st.session_state['df'] is None:
        st.warning("Please upload a dataset first in the Data Upload page")
    else:
        df = st.session_state['df']

        tab1, tab2, tab3, tab4 = st.tabs(["Statistical Summary", "Distributions", "Correlations", "Missing Data"])

        with tab1:
            st.write("Statistical Summary:")
            numeric_df = df.select_dtypes(include=[np.number])
            if len(numeric_df.columns) > 0:
                st.dataframe(numeric_df.describe(), use_container_width=True)
            else:
                st.error("No numeric columns found in dataset")

            st.write("Data Types Summary:")
            type_summary = pd.DataFrame({
                'Data Type': df.dtypes.value_counts().index.astype(str),
                'Count': df.dtypes.value_counts().values
            })
            st.dataframe(type_summary, use_container_width=True)

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(type_summary['Data Type'], type_summary['Count'], color='skyblue')
            ax.set_title("Column Types Distribution")
            ax.set_xlabel("Data Type")
            ax.set_ylabel("Count")
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig)

        with tab2:
            st.write("Select a column to visualize:")
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

            if len(numeric_cols) > 0:
                selected_col = st.selectbox("Choose column", numeric_cols)

                col1, col2 = st.columns(2)

                with col1:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.hist(df[selected_col].dropna(), bins=30, edgecolor='black', color='steelblue')
                    ax.set_xlabel(selected_col)
                    ax.set_ylabel("Frequency")
                    ax.set_title(f"Distribution of {selected_col}")
                    st.pyplot(fig)

                with col2:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.boxplot(df[selected_col].dropna())
                    ax.set_ylabel(selected_col)
                    ax.set_title(f"Box Plot of {selected_col}")
                    st.pyplot(fig)
            else:
                st.error("No numeric columns found.")

        with tab3:
            st.write("Correlation Analysis:")
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

            if len(numeric_cols) > 1:
                corr = df[numeric_cols].corr()

                fig, ax = plt.subplots(figsize=(14, 10))
                mask = np.triu(np.ones_like(corr, dtype=bool))
                sns.heatmap(corr, mask=mask, annot=True, fmt='.2f',
                            cmap='coolwarm', center=0, ax=ax,
                            square=True, linewidths=1)
                ax.set_title("Feature Correlations", fontsize=16, pad=20)
                st.pyplot(fig)

                st.info("Note: High correlation does not imply causation!")

                st.write("Top Correlations:")
                corr_pairs = corr.unstack()
                corr_pairs = corr_pairs[corr_pairs < 1]
                top_corr = corr_pairs.abs().sort_values(ascending=False).head(10)
                st.dataframe(top_corr, use_container_width=True)
            else:
                st.error("Need at least 2 numeric columns for correlation analysis")

        with tab4:
            st.write("Missing Data Analysis:")
            missing = df.isnull().sum()
            missing_percent = (missing / len(df)) * 100

            missing_df = pd.DataFrame({
                'Column': missing.index,
                'Missing Count': missing.values,
                'Missing Percentage': missing_percent.values
            })
            missing_df = missing_df[missing_df['Missing Count'] > 0]

            if len(missing_df) > 0:
                st.dataframe(missing_df, use_container_width=True)

                fig, ax = plt.subplots(figsize=(10, 5))
                ax.bar(missing_df['Column'], missing_df['Missing Percentage'], color='coral')
                ax.set_xlabel("Column")
                ax.set_ylabel("Missing Percentage")
                ax.set_title("Missing Data by Column")
                plt.xticks(rotation=45, ha='right')
                st.pyplot(fig)
            else:
                st.success("No missing data found!")

elif page == "Causal Analysis":
    st.header("Step 3: Causal Discovery")

    if 'df' not in st.session_state or st.session_state['df'] is None:
        st.warning("Please upload a dataset first in the Data Upload page")
    else:
        df = st.session_state['original_df'].copy()

        st.write("Configure your causal analysis:")

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols) < 2:
            st.error("Need at least 2 numeric columns for causal analysis.")
            st.info("Your dataset needs numeric columns. Text columns should be automatically converted on upload.")
        else:
            col1, col2 = st.columns([2, 1])

            with col1:
                selected_vars = st.multiselect(
                    "Choose variables for analysis (3-10 recommended):",
                    numeric_cols,
                    default=numeric_cols[:min(5, len(numeric_cols))]
                )

            with col2:
                significance_level = st.slider(
                    "Significance Level:",
                    0.01, 0.10, 0.05, 0.01
                )

            if len(selected_vars) >= 2:

                if st.button("Run Causal Discovery", use_container_width=True):

                    with st.spinner("Discovering causal relationships..."):

                        df_subset = df[selected_vars].dropna()

                        if len(df_subset) < 100:
                            st.warning("Limited data may affect accuracy. Consider using more data.")

                        try:
                            pc = PC(df_subset)
                            model = pc.estimate(significance_level=significance_level)
                            edges = list(model.edges())
                            algorithm_name = "PC Algorithm"
                            node_color = 'lightblue'

                            st.session_state['last_analysis'] = {
                                'edges': edges,
                                'algorithm': algorithm_name,
                                'variables': selected_vars,
                                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            }

                            st.success("Causal discovery completed!")

                            col1, col2 = st.columns([2, 1])

                            with col1:
                                st.write("### Discovered Causal Relationships")

                                if len(edges) > 0:
                                    for i, edge in enumerate(edges, 1):
                                        st.write(f"{i}. {edge[0]} → {edge[1]}")
                                else:
                                    st.info("No significant causal relationships found with current settings")

                            with col2:
                                st.write("### Analysis Info")
                                st.write(f"Algorithm: {algorithm_name}")
                                st.write(f"Variables: {len(selected_vars)}")
                                st.write(f"Edges Found: {len(edges)}")
                                st.write(f"Data Points: {len(df_subset)}")

                            fig, ax = plt.subplots(figsize=(14, 10))
                            G = nx.DiGraph()
                            G.add_edges_from(edges)

                            pos = nx.spring_layout(G, k=3, iterations=50, seed=42)

                            nx.draw_networkx_nodes(G, pos, node_color=node_color,
                                                   node_size=4000, ax=ax, alpha=0.9)
                            nx.draw_networkx_labels(G, pos, font_size=11,
                                                    font_weight='bold', ax=ax)
                            nx.draw_networkx_edges(G, pos, edge_color='gray',
                                                   arrows=True, arrowsize=25,
                                                   arrowstyle='->', width=2, ax=ax,
                                                   connectionstyle='arc3,rad=0.1')

                            ax.set_title(f"Causal Graph - {algorithm_name}", fontsize=16, pad=20)
                            ax.axis('off')
                            st.pyplot(fig)

                            st.session_state['causal_fig'] = fig

                            col1, col2 = st.columns(2)

                            with col1:
                                png_buf = export_graph_as_png(fig)
                                st.download_button(
                                    label="Download Graph as PNG",
                                    data=png_buf,
                                    file_name=f"causal_graph_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                                    mime="image/png",
                                    use_container_width=True
                                )

                            with col2:
                                results_text = f"CausaLink Analysis Results\n\n"
                                results_text += f"Algorithm: {algorithm_name}\n"
                                results_text += f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                                results_text += f"Variables Analyzed: {', '.join(selected_vars)}\n\n"
                                results_text += "Causal Relationships:\n"
                                for i, edge in enumerate(edges, 1):
                                    results_text += f"{i}. {edge[0]} → {edge[1]}\n"

                                st.download_button(
                                    label="Download Results as TXT",
                                    data=results_text,
                                    file_name=f"causal_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                    mime="text/plain",
                                    use_container_width=True
                                )

                            st.write("### Interpretation Guide")
                            st.write("- Arrows indicate the direction of potential causal influence")
                            st.write("- A → B suggests that A may causally influence B")
                            st.write("- These results are exploratory and should be validated with domain knowledge")
                            st.write("- Consider confounding variables and data quality when interpreting results")

                        except Exception as e:
                            st.error(f"Error during analysis: {str(e)}")
                            st.write("Try selecting different variables or adjusting parameters")
            else:
                st.info("Please select at least 2 variables to begin analysis")

st.sidebar.markdown("---")
st.sidebar.write("### About CausaLink")
st.sidebar.write("A platform for discovering causal relationships in data using advanced statistical algorithms.")
st.sidebar.write("Version 1.0")
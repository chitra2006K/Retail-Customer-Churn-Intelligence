import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Customer Intelligence Dashboard",
    page_icon="🛒",
    layout="wide"
)


# Load custom dashboard styling (theme is configured in .streamlit/config.toml).
try:
    with open("dashboard/assets/metric_cards.css", encoding="utf-8") as css_file:
        st.markdown(
            f"<style>{css_file.read()}</style>",
            unsafe_allow_html=True
        )
except FileNotFoundError:
    pass
# Load Layout Polish CSS
try:
    with open("dashboard/assets/layout_polish.css", encoding="utf-8") as css_file:
        st.markdown(
            f"<style>{css_file.read()}</style>",
            unsafe_allow_html=True
        )
except FileNotFoundError:
    pass

# Consistent colours and formatting for all Matplotlib charts.
TEAL = "#0F766E"
DARK_TEAL = "#115E59"
LIGHT_TEAL = "#99F6E4"
AMBER = "#D97706"
SLATE = "#64748B"

def style_chart(ax, title, xlabel, ylabel, rotate_x=0):
    ax.set_title(title, fontsize=15, fontweight="bold", color="#0F172A", pad=16)
    ax.set_xlabel(xlabel, fontsize=10, fontweight="semibold", color="#334155", labelpad=10)
    ax.set_ylabel(ylabel, fontsize=10, fontweight="semibold", color="#334155", labelpad=10)
    ax.tick_params(axis="both", labelsize=9, colors="#475569")
    ax.tick_params(axis="x", rotation=rotate_x)
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.22, color=SLATE)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CBD5E1")
    ax.spines["bottom"].set_color("#CBD5E1")
    ax.set_facecolor("#FFFFFF")



# ==========================================
# 2. SIDEBAR AND NAVIGATION
# ==========================================

st.sidebar.title("Customer Intelligence")
st.sidebar.write("Customer Segmentation & Churn Prediction")
st.sidebar.divider()

st.sidebar.subheader("Dashboard Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Business Overview",
        "Customer Segmentation",
        "Churn Analysis",
        "Business Recommendations",
        "Individual Customer Search"
    ]
)

st.sidebar.divider()

st.sidebar.subheader("Project Overview")
st.sidebar.write(
    "This dashboard analyses retail customer behaviour "
    "using Machine Learning."
)

st.sidebar.subheader("Project Features")
st.sidebar.markdown(
    """
    - Customer Segmentation
    - Churn Prediction
    - Customer Risk Assessment
    - Business Recommendations
    - Individual Customer Analysis
    - Interactive Customer Filters
    - CSV Data Export
    """
)

st.sidebar.divider()
st.sidebar.caption("Machine Learning Project | 2026")


# ==========================================
# 3. DASHBOARD TITLE
# ==========================================

st.title("Customer Intelligence Dashboard")

st.write(
    "An interactive dashboard for Customer Segmentation "
    "and Churn Prediction."
)


# ==========================================
# 4. LOAD CUSTOMER DATA
# ==========================================

@st.cache_data
def load_data():
    segments = pd.read_csv(
        "data/processed/customer_segments.csv"
    )

    predictions = pd.read_csv(
        "data/processed/customer_churn_predictions.csv"
    )

    return segments, predictions


try:
    segments_df, predictions_df = load_data()

except FileNotFoundError:
    st.error(
        "Could not find the required CSV files. "
        "Please run this app from your project root folder and "
        "check the files inside data/processed/."
    )
    st.stop()


# ==========================================
# 5. COMMON DASHBOARD DATA
# ==========================================

required_segment_columns = {
    "CustomerID", "Recency", "Frequency",
    "Monetary", "AOV", "Segment"
}

required_prediction_columns = {
    "CustomerID", "LR_Status", "RF_Status"
}

if not required_segment_columns.issubset(segments_df.columns):
    st.error(
        "The customer_segments.csv file is missing one or more "
        "required columns."
    )
    st.stop()

if not required_prediction_columns.issubset(predictions_df.columns):
    st.error(
        "The customer_churn_predictions.csv file is missing one or more "
        "required columns."
    )
    st.stop()

total_customers = len(segments_df)

active_customers = (
    predictions_df["LR_Status"] == "Active"
).sum()

churned_customers = (
    predictions_df["LR_Status"] == "Churned"
).sum()

total_segments = segments_df["Segment"].nunique()

churn_rate = (
    (churned_customers / total_customers) * 100
    if total_customers > 0 else 0
)

# Merge both datasets once, keeping only one Recency column.
merged_df = pd.merge(
    segments_df,
    predictions_df.drop(columns=["Recency"], errors="ignore"),
    on="CustomerID",
    how="inner"
)


# ==========================================
# 6. BUSINESS OVERVIEW
# ==========================================

if page == "Business Overview":

    st.header("Business Overview")

    st.write(
        "A quick summary of the customers, segments, and "
        "churn predictions in the retail dataset."
    )

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Total Customers", f"{total_customers:,}")
    col2.metric("Active Customers", f"{active_customers:,}")
    col3.metric("Predicted Churned", f"{churned_customers:,}")
    col4.metric("Customer Segments", total_segments)
    col5.metric("Predicted Churn Rate", f"{churn_rate:.2f}%")

    st.divider()

    st.subheader("Customer Data Preview")

    st.write("Customer Segmentation Data")
    st.markdown(
        segments_df.head(5).to_html(index=False),
        unsafe_allow_html=True
    )

    st.write("Customer Churn Prediction Data")
    st.markdown(
        predictions_df.head(5).to_html(index=False),
        unsafe_allow_html=True
    )

    st.info(
        "Churn status is a model prediction based on the project's "
        "90-day inactivity proxy. It is not independently verified "
        "or confirmed future churn."
    )


# ==========================================
# 7. CUSTOMER SEGMENTATION
# ==========================================

elif page == "Customer Segmentation":

    st.header("Customer Segmentation Analysis")

    # --------------------------------------
    # 7.1 Customer Distribution by Segment
    # --------------------------------------

    segment_counts = segments_df["Segment"].value_counts()

    st.subheader("Customer Distribution by Segment")

    fig, ax = plt.subplots(figsize=(12, 6))

    segment_palette = [TEAL, DARK_TEAL, LIGHT_TEAL, SLATE]
    bars = ax.bar(
        segment_counts.index,
        segment_counts.values,
        color=segment_palette[:len(segment_counts)],
        width=0.62,
        edgecolor="white",
        linewidth=1.2
    )

    for bar in bars:
        height = bar.get_height()

        ax.annotate(
            f"{int(height):,}",
            (bar.get_x() + bar.get_width() / 2, height),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="#0F172A",
            xytext=(0, 5),
            textcoords="offset points"
        )

    style_chart(
        ax,
        "Customer Distribution by Segment",
        "Customer Segments",
        "Number of Customers",
        rotate_x=12
    )
    ax.set_ylim(0, max(segment_counts.values) * 1.15)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.divider()

    # --------------------------------------
    # 7.2 Customer Segment Insights
    # --------------------------------------

    st.header("Customer Segment Insights")

    segment_churn = pd.crosstab(
        merged_df["Segment"],
        merged_df["LR_Status"]
    ).reindex(
        columns=["Active", "Churned"],
        fill_value=0
    )

    st.subheader("Churn Distribution Across Customer Segments")

    fig, ax = plt.subplots(figsize=(12, 6))

    segment_churn.plot(
        kind="bar",
        stacked=True,
        ax=ax,
        color=[TEAL, AMBER],
        width=0.66,
        edgecolor="white",
        linewidth=0.8
    )

    style_chart(
        ax,
        "Churn Distribution Across Customer Segments",
        "Customer Segment",
        "Number of Customers",
        rotate_x=12
    )
    ax.legend(title="Customer Status", frameon=False, fontsize=9, title_fontsize=9)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # --------------------------------------
    # 7.3 Churn Rate by Segment
    # --------------------------------------

    st.subheader("Churn Rate by Customer Segment")

    segment_churn_rate = (
        merged_df.assign(
            Is_Churned=(
                merged_df["LR_Status"] == "Churned"
            )
        )
        .groupby("Segment")["Is_Churned"]
        .mean() * 100
    )

    fig, ax = plt.subplots(figsize=(12, 6))

    bars = ax.bar(
        segment_churn_rate.index,
        segment_churn_rate.values,
        color=[AMBER if value > 50 else TEAL for value in segment_churn_rate.values],
        width=0.62,
        edgecolor="white",
        linewidth=1.2
    )

    for bar in bars:
        height = bar.get_height()

        ax.annotate(
            f"{height:.2f}%",
            (bar.get_x() + bar.get_width() / 2, height),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="#0F172A",
            xytext=(0, 5),
            textcoords="offset points"
        )

    style_chart(
        ax,
        "Predicted Churn Rate Across Customer Segments",
        "Customer Segment",
        "Predicted Churn Rate (%)",
        rotate_x=12
    )
    ax.set_ylim(0, 112)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.divider()

    # --------------------------------------
    # 7.4 Customer Segment and Churn Filters
    # --------------------------------------

    st.header("Explore Customers by Segment")

    selected_segment = st.selectbox(
        "Select Customer Segment",
        ["All Segments"] + sorted(merged_df["Segment"].dropna().unique())
    )

    if selected_segment == "All Segments":
        filtered_df = merged_df.copy()
    else:
        filtered_df = merged_df[
            merged_df["Segment"] == selected_segment
        ].copy()

    selected_status = st.selectbox(
        "Select Churn Status",
        ["All Statuses", "Active", "Churned"]
    )

    if selected_status != "All Statuses":
        filtered_df = filtered_df[
            filtered_df["LR_Status"] == selected_status
        ].copy()

    st.subheader("Customer Filter Summary")

    total_filtered = len(filtered_df)

    active_filtered = (
        filtered_df["LR_Status"] == "Active"
    ).sum()

    churned_filtered = (
        filtered_df["LR_Status"] == "Churned"
    ).sum()

    filtered_churn_rate = (
        (churned_filtered / total_filtered) * 100
        if total_filtered > 0 else 0
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Customers", f"{total_filtered:,}")
    col2.metric("Active Customers", f"{active_filtered:,}")
    col3.metric("Predicted Churned", f"{churned_filtered:,}")
    col4.metric("Predicted Churn Rate", f"{filtered_churn_rate:.2f}%")

    st.subheader("Filtered Customer Records")

    st.markdown(
        filtered_df.head(10).to_html(index=False),
        unsafe_allow_html=True
    )

    st.caption(
        f"Showing up to 10 of {total_filtered:,} matching customer records."
    )

    # Search within the selected filters.
    st.subheader("Search Within Filtered Customers")

    search_customer_id = st.number_input(
        "Enter Customer ID",
        min_value=0,
        step=1,
        value=0,
        key="filtered_customer_search"
    )

    if st.button("Search Filtered Customer"):
        search_result = filtered_df[
            filtered_df["CustomerID"] == search_customer_id
        ]

        if not search_result.empty:
            st.success("Customer found in selected filters!")
            st.markdown(
                search_result.to_html(index=False),
                unsafe_allow_html=True
            )
        else:
            st.error("Customer not found in the selected filters!")

    # Export filtered data.
    st.subheader("Download Filtered Customer Data")

    csv_data = filtered_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Customer Data (CSV)",
        data=csv_data,
        file_name="filtered_customer_data.csv",
        mime="text/csv"
    )


# ==========================================
# 8. CHURN ANALYSIS
# ==========================================

elif page == "Churn Analysis":

    st.header("Customer Churn Analysis")

    churn_counts = predictions_df["LR_Status"].value_counts().reindex(
        ["Active", "Churned"],
        fill_value=0
    )

    # --------------------------------------
    # 8.1 Active vs Predicted Churned
    # --------------------------------------

    st.subheader("Active vs Predicted Churned Customers")

    fig, ax = plt.subplots(figsize=(10, 5))

    bars = ax.bar(
        churn_counts.index,
        churn_counts.values,
        color=[TEAL, AMBER],
        width=0.58,
        edgecolor="white",
        linewidth=1.2
    )

    for bar in bars:
        height = bar.get_height()

        percentage = (
            (height / total_customers) * 100
            if total_customers > 0 else 0
        )

        ax.annotate(
            f"{int(height):,}\n{percentage:.2f}%",
            (bar.get_x() + bar.get_width() / 2, height),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="#0F172A",
            xytext=(0, 5),
            textcoords="offset points"
        )

    style_chart(
        ax,
        "Customer Churn Prediction Distribution",
        "Customer Status",
        "Number of Customers"
    )
    ax.set_ylim(0, max(churn_counts.values) * 1.18 if len(churn_counts) else 1)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.divider()

    # --------------------------------------
    # 8.2 Model Comparison
    # --------------------------------------

    st.header("Churn Prediction Model Comparison")

    lr_counts = predictions_df["LR_Status"].value_counts().reindex(
        ["Active", "Churned"],
        fill_value=0
    )

    rf_counts = predictions_df["RF_Status"].value_counts().reindex(
        ["Active", "Churned"],
        fill_value=0
    )

    x = np.arange(2)
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))

    lr_bars = ax.bar(
        x - width / 2,
        lr_counts.values,
        width,
        label="Logistic Regression",
        color=TEAL,
        edgecolor="white",
        linewidth=1.0
    )

    rf_bars = ax.bar(
        x + width / 2,
        rf_counts.values,
        width,
        label="Random Forest",
        color=AMBER,
        edgecolor="white",
        linewidth=1.0
    )

    for bars in [lr_bars, rf_bars]:
        for bar in bars:
            height = bar.get_height()

            ax.annotate(
                f"{int(height):,}",
                (bar.get_x() + bar.get_width() / 2, height),
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="semibold",
                color="#0F172A",
                xytext=(0, 5),
                textcoords="offset points"
            )

    style_chart(
        ax,
        "Logistic Regression vs Random Forest",
        "Customer Status",
        "Number of Customers"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(["Active", "Churned"])
    ax.legend(frameon=False, fontsize=9)
    ax.set_ylim(0, max(lr_counts.max(), rf_counts.max()) * 1.16)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.info(
        "The two models can produce different predictions because "
        "they learn patterns in different ways. These counts are "
        "predictions, not confirmed churn outcomes."
    )


# ==========================================
# 9. BUSINESS RECOMMENDATIONS
# ==========================================

elif page == "Business Recommendations":

    st.header("Business Recommendations")

    st.write(
        "Recommended retention strategies based on customer "
        "segmentation and churn prediction."
    )

    recommendations = {
        "Low-Engagement Customers": (
            "This segment has the highest Logistic Regression "
            "predicted churn rate in the current analysis (66.85%). "
            "Provide personalised discounts, win-back offers, "
            "and promotional campaigns."
        ),

        "Regular Customers": (
            "Offer loyalty points, repeat-purchase rewards, "
            "and personalised product recommendations."
        ),

        "Exceptional High-Value Customers": (
            "Provide VIP customer support, exclusive benefits, "
            "and personalised retention offers."
        ),

        "Highly Active High-Value Customers": (
            "Maintain engagement through premium services, "
            "exclusive offers, and early access to new products."
        )
    }

    for segment, recommendation in recommendations.items():

        with st.expander(segment):
            st.write(recommendation)

    st.divider()

    st.subheader("Suggested Retention Priorities")

    st.markdown(
        """
        1. **Low-Engagement Customers:** Prioritise win-back campaigns
           and monitor whether customers return to purchase.
        2. **Regular Customers:** Encourage repeat purchases through
           loyalty programmes.
        3. **Exceptional High-Value Customers:** Offer personalised
           attention and premium support.
        4. **Highly Active High-Value Customers:** Maintain satisfaction
           and strengthen long-term engagement.
        """
    )

    st.info(
        "These are business suggestions based on the project's "
        "segment analysis and model predictions. Review individual "
        "customer circumstances before taking action."
    )


# ==========================================
# 10. INDIVIDUAL CUSTOMER SEARCH
# ==========================================

elif page == "Individual Customer Search":

    st.header("Individual Customer Insights")

    customer_id = st.number_input(
        "Enter Customer ID",
        min_value=0,
        step=1,
        value=12347,
        key="individual_customer_search"
    )

    if st.button("Search Customer"):

        customer_data = merged_df[
            merged_df["CustomerID"] == customer_id
        ]

        if not customer_data.empty:

            st.success("Customer found successfully!")

            customer = customer_data.iloc[0]

            # Customer details
            st.subheader("Customer Details")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Customer ID",
                int(customer["CustomerID"])
            )

            col2.metric(
                "Recency",
                int(customer["Recency"])
            )

            col3.metric(
                "Frequency",
                int(customer["Frequency"])
            )

            col4, col5, col6 = st.columns(3)

            col4.metric(
                "Monetary",
                f"{customer['Monetary']:.2f}"
            )

            col5.metric(
                "AOV",
                f"{customer['AOV']:.2f}"
            )

            col6.metric(
                "Customer Segment",
                customer["Segment"]
            )

            # Churn predictions
            st.subheader("Churn Predictions")

            col7, col8 = st.columns(2)

            col7.metric(
                "Logistic Regression",
                customer["LR_Status"]
            )

            col8.metric(
                "Random Forest",
                customer["RF_Status"]
            )

            # Customer risk assessment
            st.subheader("Customer Risk Assessment")

            if (
                customer["LR_Status"] == "Active"
                and customer["RF_Status"] == "Active"
            ):

                st.success("Low Predicted Churn Risk")

                st.write(
                    "Both models predict that this customer is active."
                )

            elif (
                customer["LR_Status"] == "Churned"
                and customer["RF_Status"] == "Churned"
            ):

                st.error("High Predicted Churn Risk")

                st.write(
                    "Both models predict that this customer may have churned."
                )

            else:

                st.warning("Mixed Model Predictions")

                st.write(
                    "The two models disagree. Further customer review is recommended."
                )

            # Customer retention recommendation
            st.subheader("Recommended Business Action")

            if (
                customer["LR_Status"] == "Active"
                and customer["RF_Status"] == "Active"
            ):

                st.info(
                    "Continue engagement through loyalty rewards, "
                    "personalised recommendations, and exclusive offers."
                )

            elif (
                customer["LR_Status"] == "Churned"
                and customer["RF_Status"] == "Churned"
            ):

                st.warning(
                    "Launch a win-back campaign with personalised "
                    "discounts and special offers."
                )

            else:

                st.info(
                    "Review the customer's purchase history and "
                    "consider a personalised follow-up."
                )

        else:

            st.error("Customer ID not found!")

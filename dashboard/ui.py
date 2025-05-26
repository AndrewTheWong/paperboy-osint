import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import subprocess
import sys

# Import local modules
from storage.db import get_unprocessed_osint_entries, update_osint_tags
from dashboard.tag_ui import review_tags_ui

def show_predictions_dashboard():
    """Display the predictions dashboard."""
    st.title("Agentic AI Analyst Dashboard")
    
    # Sidebar for filtering
    st.sidebar.title("Filters")
    st.sidebar.markdown("Filter the predictions by various criteria.")
    
    # Main dashboard content
    st.header("Event Predictions")
    
    # Mock data for demonstration
    mock_data = {
        "id": ["1", "2", "3", "4", "5", "6", "7", "8"],
        "event_type": ["military movement", "conflict", "cyberattack", "diplomatic meeting", 
                      "protest", "nuclear", "military movement", "ceasefire"],
        "region": ["Eastern Europe", "Middle East", "North America", "Asia", 
                   "South America", "East Asia", "Middle East", "Africa"],
        "likelihood_score": [0.85, 0.72, 0.65, 0.45, 0.58, 0.77, 0.68, 0.52],
        "generated_at": [
            (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d %H:%M:%S")
            for i in range(8)
        ]
    }
    
    df = pd.DataFrame(mock_data)
    
    # Allow filtering by event type and region
    event_types = ["All"] + sorted(df["event_type"].unique().tolist())
    regions = ["All"] + sorted(df["region"].unique().tolist())
    
    selected_event_type = st.sidebar.selectbox("Event Type", event_types)
    selected_region = st.sidebar.selectbox("Region", regions)
    min_likelihood = st.sidebar.slider("Minimum Likelihood", 0.0, 1.0, 0.3, 0.05)
    
    # Apply filters
    filtered_df = df.copy()
    if selected_event_type != "All":
        filtered_df = filtered_df[filtered_df["event_type"] == selected_event_type]
    if selected_region != "All":
        filtered_df = filtered_df[filtered_df["region"] == selected_region]
    filtered_df = filtered_df[filtered_df["likelihood_score"] >= min_likelihood]
    
    # Display map of predictions by region
    st.subheader("Predictions by Region")
    
    # Group by region and get average likelihood
    region_data = filtered_df.groupby("region")["likelihood_score"].mean().reset_index()
    
    # Create a choropleth map
    fig = px.choropleth(
        region_data,
        locations="region",  # Note: This is simplified, in reality you'd need ISO codes
        locationmode="country names",  # This is simplified
        color="likelihood_score",
        color_continuous_scale="Plasma",
        range_color=(0, 1),
        title="Average Likelihood by Region",
        labels={"likelihood_score": "Average Likelihood"}
    )
    
    # Handle the case where the map doesn't display properly
    try:
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.warning("Map visualization could not be displayed. Using bar chart instead.")
        bar_fig = px.bar(
            region_data, 
            x="region", 
            y="likelihood_score",
            title="Average Likelihood by Region",
            labels={"likelihood_score": "Average Likelihood", "region": "Region"}
        )
        st.plotly_chart(bar_fig, use_container_width=True)
    
    # Display event types distribution
    st.subheader("Event Types Distribution")
    event_counts = filtered_df["event_type"].value_counts().reset_index()
    event_counts.columns = ["event_type", "count"]
    
    event_fig = px.pie(
        event_counts, 
        values="count", 
        names="event_type",
        title="Distribution of Event Types",
        hole=0.4
    )
    st.plotly_chart(event_fig, use_container_width=True)
    
    # Display the actual prediction data
    st.subheader("Prediction Details")
    st.dataframe(
        filtered_df.sort_values("likelihood_score", ascending=False),
        use_container_width=True
    )
    
    # Display a gauge chart for highest likelihood events
    if not filtered_df.empty:
        st.subheader("Highest Likelihood Events")
        top_events = filtered_df.sort_values("likelihood_score", ascending=False).head(3)
        
        col1, col2, col3 = st.columns(3)
        cols = [col1, col2, col3]
        
        for i, (_, row) in enumerate(top_events.iterrows()):
            if i < 3:  # Show top 3 events
                gauge_fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=row["likelihood_score"],
                    title={"text": f"{row['event_type']} in {row['region']}"},
                    gauge={
                        "axis": {"range": [0, 1]},
                        "bar": {"color": "darkblue"},
                        "steps": [
                            {"range": [0, 0.3], "color": "lightgreen"},
                            {"range": [0.3, 0.7], "color": "yellow"},
                            {"range": [0.7, 1.0], "color": "red"}
                        ]
                    }
                ))
                cols[i].plotly_chart(gauge_fig, use_container_width=True)

def launch_dashboard():
    """Launch the Streamlit dashboard."""
    try:
        # Determine the path to this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        dashboard_path = os.path.join(current_dir, "ui.py")
        
        # Run the Streamlit app
        cmd = [sys.executable, "-m", "streamlit", "run", dashboard_path]
        process = subprocess.Popen(cmd)
        
        print(f"Launched dashboard at http://localhost:8501")
        return process
    except Exception as e:
        print(f"Error launching dashboard: {str(e)}")
        return None

if __name__ == "__main__":
    show_predictions_dashboard() 
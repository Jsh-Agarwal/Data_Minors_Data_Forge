import streamlit as st
import pandas as pd
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Emergency Response Dashboard",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 1rem 2rem;
        font-size: 1.1rem;
        background-color: #f0f2f6;
        border-radius: 0.5rem;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #262730;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("🚨 Emergency Response Dashboard")
st.markdown("""
    This dashboard provides insights into emergency calls patterns and their geographic distribution.
    Use the filters below to explore the data.
    """)

# Read the dataset with caching
@st.cache_data
def load_data():
    df = pd.read_csv('final_enriched_emergency_dataset_with_balanced_missing_types.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    return df

# Load data
df = load_data()

# Sidebar filters
st.sidebar.header("Filters")
emergency_type = st.sidebar.multiselect(
    "Select Emergency Types",
    options=df['main_type'].unique(),
    default=df['main_type'].unique()
)

# Date range filter
min_date = df['timestamp'].min()
max_date = df['timestamp'].max()
date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# Filter data based on selection
filtered_df = df[df['main_type'].isin(emergency_type)]
if len(date_range) == 2:
    filtered_df = filtered_df[
        (filtered_df['timestamp'].dt.date >= date_range[0]) & 
        (filtered_df['timestamp'].dt.date <= date_range[1])
    ]

# Create three columns for metrics
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Emergency Calls", len(filtered_df))
with col2:
    st.metric("Unique Emergency Types", len(filtered_df['main_type'].unique()))
with col3:
    st.metric("Average Patients per Call", round(filtered_df['patient_count'].mean(), 2))

# Create tabs for different visualizations
tab1, tab2, tab3 = st.tabs(["Temporal Analysis", "Data View", "Type Distribution"])

with tab1:
    st.header("Temporal Analysis")
    
    # Hourly distribution
    st.subheader("Calls by Hour")
    hourly_data = filtered_df.groupby('hour').size().reset_index(name='count')
    st.bar_chart(hourly_data.set_index('hour'), use_container_width=True)
    
    # Day of week distribution
    st.subheader("Calls by Day of Week")
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_data = filtered_df.groupby('dayofweek').size().reset_index(name='count')
    daily_data['day_name'] = daily_data['dayofweek'].map(lambda x: day_names[x])
    st.bar_chart(daily_data.set_index('day_name'), use_container_width=True)

with tab2:
    st.header("Data View")
    st.dataframe(filtered_df)

with tab3:
    st.header("Emergency Type Distribution")
    
    # Type distribution
    type_data = filtered_df.groupby('main_type').size().reset_index(name='count')
    st.bar_chart(type_data.set_index('main_type'), use_container_width=True)
    
    # Add a data table
    st.subheader("Distribution by Type")
    st.dataframe(type_data)

# Add footer
st.markdown("---")
st.markdown("### About")
st.markdown("""
    This dashboard provides real-time insights into emergency response patterns.
    Data is updated regularly to reflect the latest emergency calls.
    """) 
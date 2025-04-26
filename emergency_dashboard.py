import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static
from sklearn.cluster import KMeans

# Set page config
st.set_page_config(page_title="Emergency Services Dashboard", layout="wide")

# Title
st.title("Emergency Services Analysis Dashboard")

# Generate synthetic data for Baltimore
def generate_synthetic_data():
    np.random.seed(42)
    
    # Define Baltimore bounds
    baltimore_lat_min, baltimore_lat_max = 39.2, 39.4
    baltimore_lon_min, baltimore_lon_max = -76.7, -76.5
    
    num_new_spots = 5000
    lat_noise = np.random.uniform(-0.005, 0.005, num_new_spots)
    lon_noise = np.random.uniform(-0.005, 0.005, num_new_spots)
    
    new_latitudes = np.random.uniform(baltimore_lat_min, baltimore_lat_max, num_new_spots) + lat_noise
    new_longitudes = np.random.uniform(baltimore_lon_min, baltimore_lon_max, num_new_spots) + lon_noise
    
    # Emergency types
    types = ['EMS', 'Fire', 'Traffic', 'Accident', 'Normal', 'Congestion', 'hospital_demand']
    new_types = np.random.choice(types, num_new_spots)
    
    # Generate timestamps for the last 30 days
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.Timedelta(days=30)
    timestamps = pd.date_range(start=start_date, end=end_date, periods=num_new_spots)
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'latitude': new_latitudes,
        'longitude': new_longitudes,
        'main_type': new_types,
        'patient_count': np.random.randint(1, 10, num_new_spots)
    })
    
    return df

# Generate data
df = generate_synthetic_data()

# Sidebar
st.sidebar.title("Filters")
selected_types = st.sidebar.multiselect(
    "Select Emergency Types",
    options=df['main_type'].unique(),
    default=df['main_type'].unique()
)

# Filter data based on selection
filtered_df = df[df['main_type'].isin(selected_types)]

# Create two columns for the first row
col1, col2 = st.columns(2)

with col1:
    st.subheader("Distribution of Calls by Hour")
    fig, ax = plt.subplots(figsize=(10, 6))
    filtered_df['hour'] = filtered_df['timestamp'].dt.hour
    sns.countplot(data=filtered_df, x='hour', palette='coolwarm')
    plt.title('Distribution of Emergency Calls by Hour')
    plt.xlabel('Hour of Day')
    plt.ylabel('Number of Calls')
    st.pyplot(fig)
    plt.close()

with col2:
    st.subheader("Distribution of Calls by Day of Week")
    fig, ax = plt.subplots(figsize=(10, 6))
    filtered_df['dayofweek'] = filtered_df['timestamp'].dt.dayofweek
    sns.countplot(data=filtered_df, x='dayofweek', palette='Set2')
    plt.title('Distribution of Emergency Calls by Day of the Week')
    plt.xlabel('Day of Week')
    plt.ylabel('Number of Calls')
    plt.xticks(ticks=np.arange(7), labels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    st.pyplot(fig)
    plt.close()

# Create two columns for the second row
col3, col4 = st.columns(2)

with col3:
    st.subheader("Distribution of Calls by Month")
    fig, ax = plt.subplots(figsize=(10, 6))
    filtered_df['month'] = filtered_df['timestamp'].dt.month
    sns.countplot(data=filtered_df, x='month', palette='viridis')
    plt.title('Distribution of Emergency Calls by Month')
    plt.xlabel('Month')
    plt.ylabel('Number of Calls')
    plt.xticks(ticks=np.arange(12), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    st.pyplot(fig)
    plt.close()

with col4:
    st.subheader("Frequency of Emergency Calls by Type")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(data=filtered_df, x='main_type', palette='coolwarm')
    plt.title('Frequency of Emergency Calls by Type')
    plt.xlabel('Emergency Type')
    plt.ylabel('Number of Calls')
    plt.xticks(rotation=45)
    st.pyplot(fig)
    plt.close()

# Interactive Map
st.subheader("Geographic Distribution of Emergency Calls")

# Create Folium map
m = folium.Map(location=[39.3, -76.6], zoom_start=12)
marker_cluster = MarkerCluster().add_to(m)

# Add markers to the map
for _, row in filtered_df.iterrows():
    color_map = {
        'EMS': 'blue',
        'Fire': 'red',
        'Traffic': 'green',
        'Accident': 'orange',
        'Normal': 'purple',
        'Congestion': 'darkred',
        'hospital_demand': 'darkblue'
    }
    
    folium.Marker(
        location=[row['latitude'], row['longitude']],
        popup=row['main_type'],
        icon=folium.Icon(color=color_map.get(row['main_type'], 'gray'))
    ).add_to(marker_cluster)

# Display the map
folium_static(m)

# Hotspot Analysis
st.subheader("Hotspot Analysis")

# Perform KMeans clustering
kmeans = KMeans(n_clusters=5, random_state=42)
cluster_data = filtered_df[['latitude', 'longitude']].copy()
cluster_data['cluster'] = kmeans.fit_predict(cluster_data)

# Plot clusters
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(data=cluster_data, x='longitude', y='latitude', hue='cluster', palette='Set2', marker='o')
plt.title('Geospatial Hotspot Clusters of Emergency Calls')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
st.pyplot(fig)
plt.close()

# Additional Statistics
st.subheader("Emergency Response Statistics")
col5, col6, col7 = st.columns(3)

with col5:
    total_calls = len(filtered_df)
    st.metric("Total Emergency Calls", total_calls)

with col6:
    avg_patients = filtered_df['patient_count'].mean()
    st.metric("Average Patients per Call", f"{avg_patients:.2f}")

with col7:
    most_common_type = filtered_df['main_type'].mode()[0]
    st.metric("Most Common Emergency Type", most_common_type) 
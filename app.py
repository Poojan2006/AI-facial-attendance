import streamlit as st
import pandas as pd
import os
import time
from datetime import datetime
import plotly.express as px


st.set_page_config(page_title="Attendance Dashboard", layout="wide")


date = datetime.now().strftime("%d-%m-%Y")
folder_path = "Attendance"
file_path = os.path.join(folder_path, f"Attendance_{date}.csv")


if not os.path.exists(folder_path):
    os.makedirs(folder_path)


st.sidebar.header(" Settings")
auto_refresh = st.sidebar.checkbox("Auto-Refresh every 10s", value=False)
theme = st.sidebar.radio(" Select Theme", ["Light", "Dark"])

def apply_theme():
    if theme == "Dark":
        st.markdown(
            """
            <style>
            .stApp { background-color: #121212 !important; color: white !important; }
            </style>
            """,
            unsafe_allow_html=True,
        )
apply_theme()


st.title(" Real-Time Attendance Dashboard")


@st.cache_data
def load_attendance():
    if os.path.exists(file_path):
        return pd.read_csv(file_path, encoding="utf-8")
    else:
        return pd.DataFrame(columns=["NAME", "TIME"])

df = load_attendance()


search_query = st.text_input(" Search for a name:")
df_filtered = df[df["NAME"].str.contains(search_query, case=False, na=False)] if search_query else df


st.subheader(f" Total Attendees: {df_filtered.shape[0]}")
st.dataframe(df_filtered)


st.subheader("Attendance Summary")
attendance_counts = df["NAME"].value_counts().reset_index()
attendance_counts.columns = ["NAME", "Entries"]

if not attendance_counts.empty:
    fig = px.pie(attendance_counts, values="Entries", names="NAME", title="Attendance Distribution")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("⚠ No attendance data available.")


st.subheader(" Mark Attendance")
new_name = st.text_input("Enter Name")
new_time = datetime.now().strftime("%H:%M:%S")

if st.button(" Mark Attendance"):
    if new_name:
        new_entry = pd.DataFrame([[new_name, new_time]], columns=["NAME", "TIME"])
        new_entry.to_csv(file_path, mode='a', header=not os.path.exists(file_path), index=False)
        st.success(f" {new_name} marked present at {new_time}")
        st.toast("Attendance marked successfully!", icon="✅")
        time.sleep(1)
        st.rerun()
    else:
        st.error("⚠ Please enter a name!")

col1 = st.columns(1)[0]
col1.download_button(" Download as CSV", data=df_filtered.to_csv(index=False).encode("utf-8"), file_name=f"Attendance_{date}.csv", mime="text/csv")

if auto_refresh:
    time.sleep(10)
    st.rerun()

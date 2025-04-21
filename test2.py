# Final Streamlit App: AI-Powered Hourly Sheet with Realtime Features + Tabs + Edit/Delete + Live Refresh

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from streamlit_chat import message
import os

CSV_FILE = "real_time_synthetic_data.csv"

# --- Live Reload Data ---
def load_data():
    df = pd.read_csv(CSV_FILE)
    df.dropna(subset=['Actual Output', 'Defects', 'Downtime (Minutes)'], inplace=True)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Machine ID'] = df['Machine ID'].astype(str).str.lower()
    df['Operator'] = df['Operator'].astype(str).str.lower()
    return df

data = load_data()

# --- Define Conversational AI Function ---
def query_data(user_input):
    user_input = user_input.lower()
    today = pd.Timestamp.today().normalize()

    if 'defects' in user_input:
        machine = user_input.split('machine ')[-1].strip().lower()
        defects = data[(data['Date'].dt.normalize() == today) & (data['Machine ID'] == machine)]['Defects'].sum()
        return f"Today's total defects for Machine {machine.upper()}: {defects}"

    elif 'units' in user_input:
        operator = user_input.split('operator ')[-1].strip().lower()
        units = data[(data['Date'].dt.normalize() == today) & (data['Operator'] == operator)]['Actual Output'].sum()
        return f"Total units produced by Operator {operator.title()} today: {units}"

    elif 'downtime' in user_input:
        machine = user_input.split('machine ')[-1].strip().lower()
        downtime = data[(data['Machine ID'] == machine)]['Downtime (Minutes)'].sum()
        return f"Total downtime for Machine {machine.upper()}: {downtime} minutes"

    return "Sorry, I didn't understand the query."

# --- Downtime Prediction Model ---
data['DowntimeRisk'] = (data['Downtime (Minutes)'] > 10).astype(int)
features = ['Actual Output', 'Defects', 'Downtime (Minutes)']
X = data[features]
y = data['DowntimeRisk']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier()
model.fit(X_train, y_train)

def predict_downtime(input_features):
    prediction = model.predict_proba([input_features])[0][1]
    return f"Downtime risk: {prediction * 100:.2f}%"

# --- Anomaly Detection ---
anomaly_model = IsolationForest(contamination=0.05)
anomaly_model.fit(X)
data['Anomaly'] = anomaly_model.predict(X)

# --- Streamlit Interface ---
st.set_page_config(layout="wide")
st.title('AI-Powered Hourly Sheet Workflow')

# --- Tabs ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📋 Data Entry", "💬 Chatbot", "🔍 Predictions", "📊 Insights", "🛠 Manage Data"])

# --- Tab 1: Data Entry ---
with tab1:
    st.subheader("➕ Add New Hourly Entry")
    with st.form("data_entry_form"):
        date = st.date_input("Date")
        shift = st.selectbox("Shift", ["Morning", "Afternoon", "Night"])
        machine_id = st.text_input("Machine ID").lower()
        operator = st.text_input("Operator").lower()
        product_name = st.text_input("Product Name")
        target_output = st.number_input("Target Output", min_value=0)
        actual_output = st.number_input("Actual Output", min_value=0)
        cumulative_output = st.number_input("Cumulative Output", min_value=0)
        defects = st.number_input("Defects", min_value=0)
        downtime = st.number_input("Downtime (Minutes)", min_value=0)
        reason = st.text_input("Reason")
        remarks = st.text_input("Remarks")
        submitted = st.form_submit_button("Submit Entry")

    if submitted:
        new_row = pd.DataFrame([{
            "Date": pd.to_datetime(date),
            "Shift": shift,
            "Machine ID": machine_id,
            "Operator": operator,
            "Product Name": product_name,
            "Target Output": target_output,
            "Actual Output": actual_output,
            "Cumulative Output": cumulative_output,
            "Defects": defects,
            "Downtime (Minutes)": downtime,
            "Reason": reason,
            "Remarks": remarks
        }])
        new_row.to_csv(CSV_FILE, mode='a', header=not os.path.exists(CSV_FILE), index=False)
        st.success("Entry successfully added to the dataset!")

# --- Tab 2: Chatbot ---
with tab2:
    st.subheader("💬 Ask a Question")
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    user_query = st.text_input("Ask me something")
    if user_query:
        bot_response = query_data(user_query)
        st.session_state.chat_history.append((user_query, bot_response))

    for i, (u, b) in enumerate(st.session_state.chat_history):
        message(u, is_user=True, key=f"user_{i}")
        message(b, key=f"bot_{i}")

# --- Tab 3: Predictions ---
with tab3:
    st.subheader("🔍 Downtime Prediction")
    st.write("Enter features for downtime prediction: Output, Defects, Downtime")
    downtime_input = st.text_input("Format: 120,3,5")
    if downtime_input:
        try:
            user_features = list(map(int, downtime_input.split(',')))
            result = predict_downtime(user_features)
            st.write(result)
        except Exception:
            st.error("Invalid input. Use numbers like: 120,3,5")

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Model Accuracy: {accuracy:.2f}")

# --- Tab 4: Insights ---
with tab4:
    st.subheader("📊 Today's Defects & Downtime")
    today = pd.Timestamp.today().normalize()
    today_summary = data[data['Date'].dt.normalize() == today].groupby('Machine ID')[['Defects', 'Downtime (Minutes)']].sum()
    if not today_summary.empty:
        st.bar_chart(today_summary)
    else:
        st.info("No data found for today.")

    st.subheader("🚨 Anomalies Detected")
    anomalies = data[data['Anomaly'] == -1]
    if not anomalies.empty:
        st.dataframe(anomalies[['Date', 'Machine ID', 'Actual Output', 'Defects', 'Downtime (Minutes)']])
    else:
        st.success("No anomalies detected.")

# --- Tab 5: Manage Data (Edit/Delete Preview) ---
with tab5:
    st.subheader("🛠 Manage Dataset (Preview Mode)")
    selected = st.multiselect("Select rows to delete:", options=data.index, format_func=lambda x: f"{data.at[x, 'Date']} - {data.at[x, 'Machine ID']}")
    if st.button("Delete Selected Rows") and selected:
        data.drop(index=selected, inplace=True)
        data.to_csv(CSV_FILE, index=False)
        st.success("Selected rows deleted successfully.")
        st.experimental_rerun()

    st.dataframe(data.tail(20))

# --- Deployment Instructions ---
st.markdown("---")

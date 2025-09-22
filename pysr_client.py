# streamlit_client.py
import streamlit as st
import pandas as pd
import requests

# MCP server URL (adjust if running on another port/machine)
SERVER_URL = "http://localhost:8000/tools"
st.title("⚡ PySR with MCP + Streamlit")

# File upload
file = st.file_uploader("Upload CSV dataset", type=["csv"])
if file:
    df = pd.read_csv(file)
    st.write("Preview of dataset:", df.head())

    X_cols = st.multiselect("Select features (X)", df.columns.tolist())
    y_col = st.selectbox("Select target (y)", df.columns.tolist())

    if st.button("Train PySR Model"):
        X = df[X_cols].values.tolist()
        y = df[y_col].values.tolist()
        response = requests.post(f"{SERVER_URL}/train_model", json={"X": X, "y": y})
        st.json(response.json())

    if st.button("Get Equation"):
        response = requests.post(f"{SERVER_URL}/get_equation", json={})
        st.json(response.json())

    st.subheader("Make Predictions")
    input_data = st.text_area("Enter input data (comma-separated per row)", "1.0,2.0\n3.0,4.0")
    if st.button("Predict"):
        rows = [list(map(float, row.split(","))) for row in input_data.strip().split("\n")]
        response = requests.post(f"{SERVER_URL}/predict", json={"X": rows})
        st.json(response.json())

# import asyncio
# from mcp.client import MCPClient

# async def run_client():
#     client = MCPClient(server_command="python", server_args=["pysr_server.py"])
#     await client.start()

#     # List available tools
#     tools = await client.list_tools()
#     print("Available tools:", [t.name for t in tools])

#     # Call prediction tool
#     pred = await client.call_tool("predict", {"values": [2.0]})
#     print("Prediction for x=2.0:", pred)

#     # Get best equation
#     eq = await client.call_tool("best_equation", {})
#     print("Best equation found:", eq)

# asyncio.run(run_client())

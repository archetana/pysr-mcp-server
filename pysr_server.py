# pysr_mcp_server.py
import numpy as np
from pysr import PySRRegressor
#from mcp.server.fastmcp import FastMCPServer
from mcp.server.fastmcp import FastMCP
#from fastmcp.server import Server


server = FastMCP("pysr-mcp")

model = None

@server.tool()
def train_model(X: list[list[float]], y: list[float], iterations: int = 1000) -> dict:
    """Train PySR model on dataset"""
    global model
    model = PySRRegressor(
        niterations=iterations,
        unary_operators=["sin", "cos", "exp", "log"],
        binary_operators=["+", "-", "*", "/"],
        model_selection="best",
    )
    X = np.array(X)
    y = np.array(y)
    model.fit(X, y)
    return {"status": "Model trained", "best_equation": str(model.get_best())}

@server.tool()
def predict(X: list[list[float]]) -> dict:
    """Make predictions with trained model"""
    global model
    if model is None:
        return {"error": "No trained model"}
    preds = model.predict(np.array(X)).tolist()
    return {"predictions": preds}

@server.tool()
def get_equation() -> dict:
    """Return best symbolic equation"""
    global model
    if model is None:
        return {"error": "No model trained"}
    return {"equation": str(model.get_best())}

if __name__ == "__main__":
    server.run(transport="streamable-http")

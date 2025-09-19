from fastmcp import FastMCP
from pysr import PySRRegressor
import numpy as np

mcp = FastMCP("PySR MCP Server")

# Train a symbolic regression model once when the server starts
X = np.linspace(-5, 5, 100).reshape(-1, 1)
y = X[:, 0]**2 + X[:, 0] + 1  # quadratic example

model = PySRRegressor(
    niterations=40,
    binary_operators=["+", "-", "*", "/"],
)
model.fit(X, y)

@mcp.tool()
def predict(values: list) -> float:
    """
    Predict using the trained PySR symbolic regression model.
    Args:
        values: list of float (input features, here single variable).
    Returns:
        float prediction.
    """
    x = np.array(values).reshape(1, -1)
    return float(model.predict(x)[0])

@mcp.tool()
def best_equation() -> str:
    """
    Return the best symbolic regression equation found by PySR.
    """
    return str(model.get_best()["equation"])

if __name__ == "__main__":
    mcp.run(transport="streamable-http")

# Usage Examples

Practical examples of using the PySR MCP Server.

## Table of Contents

- [Basic Examples](#basic-examples)
- [Real-World Use Cases](#real-world-use-cases)
- [Advanced Examples](#advanced-examples)
- [Tips and Tricks](#tips-and-tricks)

---

## Basic Examples

### Example 1: Simple Linear Relationship

Discover the equation for `y = 2x + 3`:

**Data:**
```python
X = [[1], [2], [3], [4], [5]]
y = [5, 7, 9, 11, 13]
```

**Request:**
```
"Train a symbolic regression model on this data:
X = [[1], [2], [3], [4], [5]]
y = [5, 7, 9, 11, 13]"
```

**Expected Result:**
- Equation: `2*x0 + 3` or similar
- Loss: Very low (< 0.01)
- Complexity: 3-5

---

### Example 2: Polynomial Relationship

Discover `y = x^2 + 2x + 1`:

**Data:**
```python
X = [[0], [1], [2], [3], [4]]
y = [1, 4, 9, 16, 25]
```

**Request:**
```
"Find the equation for:
X = [[0], [1], [2], [3], [4]]
y = [1, 4, 9, 16, 25]
Use operators: +, *, -, /, square"
```

**Expected Result:**
- Equation: `square(x0)` or `x0 * x0`
- Alternative: `(x0 + 1)^2` with expansion

---

### Example 3: Trigonometric Function

Discover `y = sin(x)`:

**Data:**
```python
import numpy as np
X = np.linspace(0, 2*np.pi, 20).reshape(-1, 1).tolist()
y = np.sin(X).flatten().tolist()
```

**Request:**
```
"Train a model with sine function:
X = [[data...]]
y = [data...]
Include 'sin' and 'cos' as operators"
```

**Expected Result:**
- Equation: `sin(x0)`
- May find: `cos(x0 - π/2)` or similar equivalent

---

## Real-World Use Cases

### Use Case 1: Physics - Projectile Motion

Find the relationship between time and height in projectile motion.

**Scenario:** Ball thrown upward with initial velocity

**Data:**
```python
# Time (seconds) and Height (meters)
X = [[0], [0.5], [1.0], [1.5], [2.0], [2.5]]
y = [0, 11.25, 20, 26.25, 30, 31.25]
```

**Request:**
```
"Discover the physics equation for projectile motion:
X (time in seconds) = [[0], [0.5], [1.0], [1.5], [2.0], [2.5]]
y (height in meters) = [0, 11.25, 20, 26.25, 30, 31.25]
Use quadratic operators"
```

**Expected Result:**
- Equation: `-4.9*x0^2 + 15*x0` (approximately)
- Physical interpretation: h = v₀t - ½gt²

---

### Use Case 2: Economics - Supply and Demand

Model price equilibrium.

**Data:**
```python
# [price, quantity_supplied, quantity_demanded]
X = [
    [10, 100, 500],
    [15, 200, 400],
    [20, 300, 300],
    [25, 400, 200],
    [30, 500, 100]
]
y = [10, 15, 20, 25, 30]  # equilibrium prices
```

**Request:**
```
"Find the relationship between supply, demand, and equilibrium price"
```

---

### Use Case 3: Machine Learning - Feature Engineering

Discover useful feature combinations.

**Data:**
```python
# Customer data: [age, income, purchase_history]
X = [
    [25, 50000, 10],
    [35, 75000, 25],
    [45, 100000, 50],
    # ... more data
]
y = [100, 500, 1200]  # customer lifetime value
```

**Request:**
```
"Find the best predictive formula for customer lifetime value
from age, income, and purchase history"
```

---

## Advanced Examples

### Example 4: Multi-Variable System

Discover complex relationships with multiple features.

**Scenario:** Temperature, pressure, and volume relationship (Ideal Gas Law)

**Data:**
```python
import numpy as np

# Generate data: PV = nRT (simplified, n=1, R=8.314)
P = np.linspace(1, 10, 50)  # Pressure
T = np.linspace(273, 373, 50)  # Temperature
V = 8.314 * T / P  # Volume

X = np.column_stack([P, T]).tolist()
y = V.tolist()
```

**Request:**
```
"Discover the ideal gas law relationship:
X has pressure and temperature
y is volume
Use operators: *, /, +, -"
```

**Expected Result:**
- Equation: `8.314 * x1 / x0` or similar
- Physical meaning: V = RT/P

---

### Example 5: Time Series Analysis

Find patterns in temporal data.

**Data:**
```python
# Daily sales with weekly seasonality
import numpy as np
days = np.arange(0, 90)
trend = 100 + 0.5 * days
seasonality = 20 * np.sin(2 * np.pi * days / 7)
noise = np.random.normal(0, 5, 90)
sales = trend + seasonality + noise

X = days.reshape(-1, 1).tolist()
y = sales.tolist()
```

**Request:**
```
"Find the equation for sales data with trend and weekly seasonality:
X = day number
y = sales
Include sin, cos operators"
```

**Expected Result:**
- Equation combining linear trend and sinusoidal component

---

### Example 6: Custom Operators

Define domain-specific operators.

**Scenario:** Electrical circuit analysis

**Request:**
```
"Train a model with custom electrical operators:
X = voltage and current data
y = power
Use custom operators:
- 'square(x) = x^2' for resistance calculations
- 'inv(x) = 1/x' for admittance"
```

**Configuration:**
```javascript
{
  "binary_operators": ["+", "*", "-", "/"],
  "unary_operators": [
    "square(x) = x^2",
    "inv(x) = 1/x",
    "sqrt"
  ]
}
```

---

## Tips and Tricks

### Tip 1: Start Small

Begin with fewer iterations for quick exploration:

```
"Train with 20 iterations first to test if the data makes sense"
```

Then increase for final model:

```
"Now train with 100 iterations for the best equation"
```

### Tip 2: Normalize Data

For better results, normalize your data first:

```python
from sklearn.preprocessing import StandardScaler

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_normalized = scaler_X.fit_transform(X)
y_normalized = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
```

### Tip 3: Iterative Refinement

1. **First pass**: Simple operators, low iterations
2. **Analyze**: Look at discovered equations
3. **Second pass**: Add relevant operators based on patterns
4. **Refine**: Increase iterations for final model

### Tip 4: Export for Different Uses

```
"Export the best equation in:
1. LaTeX for my paper
2. PyTorch for integration with neural networks
3. SymPy for further analysis"
```

### Tip 5: Validate Results

Always validate on holdout data:

```
"Use the model to predict on this new data:
X_test = [[...]]
Compare with actual y_test = [...]"
```

---

## Complete Workflow Example

Here's a complete end-to-end example:

### Step 1: Prepare Data

```python
import numpy as np

# Generate data: y = 2.5*cos(x0) + x1^2 - 0.5
X = 2 * np.random.randn(100, 2)
y = 2.5 * np.cos(X[:, 0]) + X[:, 1]**2 - 0.5

# Convert to lists for MCP
X_list = X.tolist()
y_list = y.tolist()
```

### Step 2: Train Model

**Request:**
```
"Train a symbolic regression model:
X = [100 samples with 2 features]
y = [100 target values]

Parameters:
- 50 iterations
- Operators: +, -, *, /, cos, sin, square
- Max complexity: 15
- Timeout: 180 seconds"
```

### Step 3: Review Equations

**Request:**
```
"Show me all discovered equations sorted by score"
```

### Step 4: Select Best Equation

**Request:**
```
"Export equation #3 (good balance of accuracy and simplicity) in:
1. LaTeX format
2. SymPy format for analysis"
```

### Step 5: Make Predictions

**Request:**
```
"Use equation #3 to predict on new data:
X_new = [[1.0, 2.0], [1.5, 2.5], [2.0, 3.0]]"
```

### Step 6: Validate

**Request:**
```
"Compare predictions with actual values:
y_actual = [5.2, 7.8, 10.3]
Calculate mean squared error"
```

---

## Natural Language Examples

You can interact naturally with Claude Code:

### Example Conversations

**Discovery:**
```
You: "I have sales data with temperature and advertising spend.
Can you find the relationship?"
Claude: [Uses fit_symbolic_regression]
```

**Exploration:**
```
You: "Show me the top 5 equations and their accuracy"
Claude: [Uses get_equations]
```

**Prediction:**
```
You: "Predict sales for temperature=25°C and ad_spend=$1000"
Claude: [Uses predict]
```

**Export:**
```
You: "Give me the equation in LaTeX for my presentation"
Claude: [Uses export_equation with format="latex"]
```

---

## Debugging Examples

### Example: Model Not Fitting Well

**Symptom:** High loss, poor equations

**Solutions:**
1. Increase iterations
2. Add more relevant operators
3. Check for data scaling issues
4. Simplify by reducing maxsize
5. Try different operator combinations

**Request:**
```
"The model isn't fitting well. Try again with:
- 100 iterations instead of 40
- Add 'exp' and 'log' operators
- Increase timeout to 600 seconds"
```

### Example: Training Too Slow

**Symptom:** Timeout errors

**Solutions:**
1. Reduce iterations
2. Decrease population sizes
3. Limit complexity
4. Use fewer operators

**Request:**
```
"Train a faster model with:
- 30 iterations
- 5 populations of size 30
- Max complexity 10
- Only basic operators: +, -, *, /"
```

---

## Resources

- See [API.md](API.md) for full parameter reference
- See [TROUBLESHOOTING.md](../TROUBLESHOOTING.md) for common issues
- Check [PySR Documentation](https://ai.damtp.cam.ac.uk/pysr/) for advanced features

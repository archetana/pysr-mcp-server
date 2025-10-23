# PySR MCP Server - Quick Start

## What is PySR?

**PySR** discovers mathematical equations from data using evolutionary algorithms. Instead of black-box models, it finds interpretable formulas like `y = 2*x + 1` or `y = sin(x) * exp(-x)`.

**Use cases**: Scientific discovery, physics modeling, feature engineering, pattern discovery.

---

## Installation

```bash
# Install dependencies
pip install fastmcp pysr numpy pandas matplotlib seaborn sympy psutil

# Install Julia backend (one-time, takes 5-10 minutes)
python -c "import pysr; pysr.install()"
```

---

## Setup

### 1. File Structure
```
your-project/
├── pysr_mcp_server.py
└── proxy_server.py
```

### 2. Configure Claude Desktop

**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`  
**Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "pysr": {
      "command": "python",
      "args": ["/absolute/path/to/proxy_server.py"]
    }
  }
}
```

**⚠️ Use absolute path!** Example: `/Users/yourname/projects/pysr/proxy_server.py`

---

## Running

Open 2 terminals:

**Terminal 1** (Main Server):
```bash
python pysr_mcp_server.py
```

**Terminal 2** (Proxy):
```bash
python proxy_server.py
```

Then restart Claude Desktop.

---

## Test Prompts

### Test 1: train_model_complete (All-in-One)

```
Test the PySR server with train_model_complete tool.

CSV data:
x,y
1,2
2,4
3,6
4,8
5,10

Parameters:
- target_column: "y"
- job_id: "test_001"
- Use default settings

Show me the discovered equation and metrics.
```

### Test 2: create_model + fit_model (Step-by-Step)

```
Step 1: Create a PySR model using create_model tool:
- model_id: "custom_001"
- binary_operators: ["+", "-", "*", "/"]
- unary_operators: ["sin", "cos"]
- niterations: 40
- populations: 15

Step 2: Fit this model using fit_model tool with data:
X = [[1], [2], [3], [4], [5]]
y = [1, 4, 9, 16, 25]
variable_names: ["x"]

Show me the equations discovered.
```

### Test 3: Predict with Trained Model

```
After training model "test_001" from Test 1:

1. Use predict tool to predict for new data:
   X = [[6], [7], [8], [9], [10]]

2. Show predictions and compare with expected pattern (y = 2*x)
```

### Test 4: Multiple Features

```
Use train_model_complete for multi-variable data:

CSV:
x1,x2,y
1,1,3
2,1,5
1,2,4
2,2,6
3,1,7
3,2,8

Parameters:
- target_column: "y"
- job_id: "multi_001"
- niterations: 50

Discover the equation relating x1, x2 to y.
```

### Test 5: Polynomial Relationship

```
Train a model to discover this cubic pattern:

CSV:
x,y
1,1
2,8
3,27
4,64
5,125

Parameters:
- target_column: "y"
- job_id: "cubic_001"
- binary_operators: ["+", "-", "*", "/", "^"]
- niterations: 60

What equation does it find?
```

### Test 6: Model Management

```
Test model lifecycle:

1. list_models - Show all active models
2. get_equations for "model_test_001"
3. export_equation in LaTeX format
4. save_model to disk as "my_model.pkl"
5. delete_model from memory
6. load_model from "my_model.pkl" with new id "reloaded_001"
```

### Test 7: Exponential Pattern

```
Find equation for exponential growth:

CSV:
t,value
0,1
1,2.718
2,7.389
3,20.086
4,54.598

Parameters:
- target_column: "value"
- unary_operators: ["exp", "log", "sin", "cos"]
- job_id: "exp_001"
- niterations: 50

Hint: This is e^t pattern.
```

### Test 8: Trigonometric Pattern

```
Discover sine wave equation:

CSV:
x,y
0,0
1.57,1
3.14,0
4.71,-1
6.28,0

Parameters:
- target_column: "y"
- unary_operators: ["sin", "cos", "tan"]
- job_id: "trig_001"

What equation does it find?
```

---

## Available Tools

| Tool | Description |
|------|-------------|
| `train_model_complete` ⭐ | All-in-one: CSV → equations |
| `create_model` | Create model with config |
| `fit_model` | Train model on data |
| `predict` | Generate predictions |
| `get_equations` | Retrieve all equations |
| `save_model` | Save to disk |
| `load_model` | Load from disk |
| `export_equation` | Export (SymPy/LaTeX/Julia) |
| `validate_data` | Check data quality |
| `list_models` | List all models |
| `delete_model` | Remove model |
| `health_check` | Server status |

---

## Common Parameters

**train_model_complete** key parameters:
- `job_id` (required): Unique identifier
- `data` (required): CSV string or file path
- `target_column` (required): Column name to predict
- `feature_columns`: List of input columns (optional)
- `niterations`: Training iterations (default: 40)
- `populations`: Number of populations (default: 15)
- `binary_operators`: `["+", "-", "*", "/", "^"]`
- `unary_operators`: `["sin", "cos", "exp", "log", "sqrt"]`
- `maxsize`: Max equation complexity (default: 20)
- `parsimony`: Simplicity preference (default: 0.0032)

---

## Troubleshooting

**Server won't start** (port busy):
```bash
lsof -ti:8000 | xargs kill -9
```

**Claude can't connect**:
- Check proxy running: `ps aux | grep proxy_server`
- Use absolute path in config
- Restart Claude Desktop

**Training too slow**:
- Reduce `niterations` to 20-30
- Reduce `populations` to 8-10

**Import error**:
```bash
pip install --upgrade fastmcp pysr numpy pandas sympy psutil
```

---

## Resources

- **PySR GitHub**: https://github.com/MilesCranmer/PySR/tree/master/pysr
- **PySR Docs**: https://astroautomata.com/PySR/
- **PySR Paper**: https://arxiv.org/abs/2305.01582
- **MCP Spec**: https://modelcontextprotocol.io/

---

## Quick Reference

```bash
# Start servers
python pysr_mcp_server.py  # Terminal 1
python proxy_server.py      # Terminal 2

# Check health
curl http://localhost:8000/health

# Common operators
# Basic: +, -, *, /
# Powers: ^, sqrt
# Trig: sin, cos, tan
# Exp: exp, log
```

**Happy Equation Discovery! 🔬**

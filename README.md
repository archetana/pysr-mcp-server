# PySR MCP Server - Complete Guide

## Table of Contents
- [About PySR](#about-pysr)
- [About This Implementation](#about-this-implementation)
- [Architecture](#architecture)
- [Installation](#installation)
- [Setup & Configuration](#setup--configuration)
- [Running the Server](#running-the-server)
- [Available Tools](#available-tools)
- [Testing in Claude Desktop](#testing-in-claude-desktop)
- [Example Workflows](#example-workflows)
- [Troubleshooting](#troubleshooting)

---

## About PySR

**PySR (Python Symbolic Regression)** is a high-performance symbolic regression library that discovers mathematical equations from data using evolutionary algorithms. Unlike traditional machine learning models that act as "black boxes," PySR finds interpretable, explicit mathematical formulas.

### Key Features of PySR:
- **Interpretable Results**: Generates human-readable mathematical equations
- **Automatic Discovery**: Finds complex relationships without manual feature engineering
- **Pareto Frontier**: Balances equation complexity vs. accuracy
- **Customizable Operators**: Supports custom mathematical operators and constraints
- **High Performance**: Built on Julia for speed, with Python interface
- **Multi-Population Evolution**: Uses evolutionary algorithms with multiple populations

### Use Cases:
- Scientific discovery (physics, chemistry, biology)
- Engineering modeling
- Time series forecasting
- Feature engineering
- Model compression
- Uncovering hidden patterns in data

---

## About This Implementation

The **PySR MCP Server** (`pysr_mcp_server.py`) is a Model Context Protocol (MCP) server that exposes PySR's symbolic regression capabilities through a standardized API. It allows AI assistants like Claude to discover mathematical equations from your data.

### Architecture Overview

```
┌─────────────────┐
│ Claude Desktop  │
│   (Client)      │
└────────┬────────┘
         │
         │ MCP Protocol (stdio)
         │
┌────────▼────────┐
│  proxy_server   │
│  (Port 5173)    │
└────────┬────────┘
         │
         │ HTTP
         │
┌────────▼────────────┐
│ pysr_mcp_server     │
│   (Port 8000)       │
│                     │
│ - FastMCP Framework │
│ - PySR Engine       │
│ - 12 Tools          │
│ - Data Validation   │
└─────────────────────┘
```

### Key Components:

1. **pysr_mcp_server.py** (Main Server)
   - HTTP-based MCP server using FastMCP
   - 12 comprehensive tools for symbolic regression
   - Model lifecycle management
   - Data validation and metrics
   - Equation export in multiple formats

2. **proxy_server.py** (MCP Proxy)
   - Bridges stdio (Claude Desktop) to HTTP (MCP Server)
   - Enables Claude Desktop integration
   - Handles protocol translation

### Server Features:

- ✅ **Complete Training Pipeline**: End-to-end CSV to equations
- ✅ **Model Management**: Create, train, save, load, delete
- ✅ **Flexible Configuration**: Custom operators, constraints, hyperparameters
- ✅ **Multi-Format Export**: SymPy, LaTeX, Julia, JAX, PyTorch
- ✅ **Data Validation**: Comprehensive quality checks
- ✅ **Prediction Generation**: Use trained models for inference
- ✅ **Health Monitoring**: Server status and system metrics
- ✅ **Persistent Storage**: Save/load models to disk

---

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Julia (required by PySR) - will be auto-installed

### Step 1: Install Python Dependencies

```bash
# Install required packages
pip install fastmcp pysr numpy pandas matplotlib seaborn sympy psutil

# Optional: Install in a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install fastmcp pysr numpy pandas matplotlib seaborn sympy psutil
```

### Step 2: Install PySR and Julia Backend

```bash
# PySR will automatically install Julia on first run
# This may take a few minutes
python -c "import pysr; pysr.install()"
```

### Step 3: Verify Installation

```bash
python -c "import pysr; print('PySR version:', pysr.__version__)"
```

---

## Setup & Configuration

### File Structure

Create the following directory structure:

```
your-project/
├── pysr_mcp_server.py    # Main MCP server
├── proxy_server.py        # Proxy for Claude Desktop
├── models/                # Auto-created: saved models
├── data/                  # Auto-created: data storage
└── results/               # Auto-created: results
```

### Configure Claude Desktop

Edit your Claude Desktop configuration file:

**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`  
**Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

Add this configuration:

```json
{
  "mcpServers": {
    "pysr": {
      "command": "python",
      "args": ["/absolute/path/to/proxy_server.py"],
      "env": {}
    }
  }
}
```

**Important**: Replace `/absolute/path/to/` with your actual file path!

---

## Running the Server

### Method 1: Direct Server (for testing)

```bash
# Start the HTTP MCP server
python pysr_mcp_server.py

# Optional: Custom host/port
python pysr_mcp_server.py --host 0.0.0.0 --port 8000

# Optional: Enable debug logging
python pysr_mcp_server.py --debug
```

The server will start on `http://localhost:8000`

### Method 2: With Proxy (for Claude Desktop)

**Terminal 1** - Start the main server:
```bash
python pysr_mcp_server.py
```

**Terminal 2** - Start the proxy:
```bash
python proxy_server.py
```

**Terminal 3** - Start Claude Desktop:
```bash
# macOS
open -a Claude

# Windows
start Claude

# Or just launch Claude Desktop normally
```

### Verify Server is Running

```bash
# Check health endpoint
curl http://localhost:8000/health

# Check server info
curl http://localhost:8000/server/info
```

---

## Available Tools

The PySR MCP Server provides 12 powerful tools:

### 1. `train_model_complete` ⭐ (Recommended)
**All-in-one training pipeline** - Takes CSV data and returns discovered equations.

**Parameters**:
- `job_id`: Unique identifier
- `data`: CSV data (string or file path)
- `target_column`: Target variable name
- `feature_columns`: List of feature names (optional)
- `niterations`: Training iterations (default: 40)
- `populations`: Number of populations (default: 15)
- `population_size`: Population size (default: 33)
- `binary_operators`: e.g., ["+", "-", "*", "/"]
- `unary_operators`: e.g., ["sin", "cos", "exp", "log"]
- `maxsize`: Max equation complexity (default: 20)
- `parsimony`: Parsimony coefficient (default: 0.0032)

### 2. `create_model`
Create a new PySR model with custom configuration.

### 3. `fit_model`
Train a previously created model on data.

### 4. `predict`
Generate predictions using a trained model.

### 5. `get_equations`
Retrieve all discovered equations with metrics.

### 6. `save_model`
Save trained model to disk.

### 7. `load_model`
Load previously saved model from disk.

### 8. `export_equation`
Export equation in various formats (SymPy, LaTeX, Julia, JAX, PyTorch).

### 9. `validate_data`
Validate data quality and format.

### 10. `list_models`
List all active models and their status.

### 11. `delete_model`
Remove model from memory.

### 12. `health_check`
Check server health and system status.

---

## Testing in Claude Desktop

Once your server is running and Claude Desktop is configured, use these prompts to test all tools:

### Test 1: Complete Training Pipeline (Simple Example)

```
I want to test the PySR symbolic regression server. Can you help me discover the equation for some simple data?

Use the train_model_complete tool with this CSV data:

x,y
1,2
2,4
3,6
4,8
5,10

Target column is "y", use default parameters, and job_id = "test_linear_001"
```

### Test 2: Polynomial Relationship

```
Discover the equation for this polynomial data using PySR:

x,y
1,1
2,8
3,27
4,64
5,125

The target is "y". Set niterations=50, populations=20, and include operators: ["+", "-", "*", "/", "^"]. Job ID: "test_cubic_001"
```

### Test 3: Trigonometric Pattern

```
Find the underlying equation for this sine wave data:

x,y
0,0
0.5,0.479
1,0.841
1.5,0.997
2,0.909
2.5,0.598
3,0.141

Target: "y". Enable sin, cos operators. Job ID: "test_trig_001"
```

### Test 4: Exponential Growth

```
Discover the equation for exponential growth:

time,population
0,100
1,271
2,738
3,2008
4,5459
5,14841

Target: "population". Include exp, log operators. Job ID: "test_exp_001"
```

### Test 5: Multiple Features

```
Find the equation for this multi-variable dataset:

x1,x2,y
1,2,5
2,3,13
3,4,25
4,5,41
5,6,61

Target: "y", features: ["x1", "x2"]. Job ID: "test_multi_001"
```

### Test 6: Model Lifecycle Management

```
Can you:
1. List all active models using list_models
2. Get equations for model "model_test_linear_001"
3. Export the best equation in LaTeX format
4. Check the server health
5. Show me how to save this model to disk
```

### Test 7: Data Validation

```
Before training, validate this data for quality issues:

Create TrainingData with:
X = [[1, 2], [3, 4], [5, 6], [7, 8]]
y = [10, 20, 30, 40]

Use validate_data tool and tell me if there are any issues.
```

### Test 8: Model Persistence

```
I want to:
1. Save the model "model_test_cubic_001" to disk as "my_cubic_model.pkl"
2. Delete it from memory
3. Load it back with model_id "reloaded_model"
4. Generate predictions for X = [[6], [7], [8]]
```

### Test 9: Advanced Configuration

```
Create a custom model with:
- model_id: "custom_model_001"
- binary_operators: ["+", "-", "*", "/", "^"]
- unary_operators: ["sin", "cos", "exp", "log", "sqrt", "abs"]
- niterations: 100
- populations: 20
- maxsize: 30
- parsimony: 0.001

Then fit it on the polynomial data from Test 2.
```

### Test 10: Real-World Physics Example

```
Help me discover the physics equation for projectile motion. Here's height vs time data:

time,height
0,0
0.5,22.375
1,39.5
1.5,51.375
2,58
2.5,59.375
3,55.5
3.5,46.375
4,32

Target: "height". Include operators that make sense for physics (powers, basic arithmetic). Job ID: "physics_001"
```

### Test 11: Equation Export Formats

```
For the best model we've trained, export the equation in all available formats:
1. SymPy
2. LaTeX
3. Julia

Show me the differences between these representations.
```

### Test 12: Server Monitoring

```
Give me a complete status report:
1. Run health_check
2. List all models
3. Show server info
4. Report system resources (CPU, memory)
```

---

## Example Workflows

### Workflow 1: Quick Discovery

```python
# In Claude Desktop, paste this prompt:
"""
I have CSV data with columns: [temperature, pressure, volume].
I want to discover the relationship where volume is the target.
Use train_model_complete with job_id "ideal_gas_001".

Here's my data:
temperature,pressure,volume
300,1,24.62
350,1,28.72
400,1,32.83
300,2,12.31
350,2,14.36
400,2,16.41
"""
```

### Workflow 2: Iterative Refinement

```python
# Prompt sequence:
# 1. Initial training
"Train a model on my data with default settings"

# 2. Examine results
"Show me all equations discovered, not just the best one"

# 3. Refine
"The equations are too complex. Retrain with maxsize=10 and higher parsimony=0.01"

# 4. Export
"Export the simplest equation that has R² > 0.95 in LaTeX format"
```

### Workflow 3: Model Comparison

```python
# Prompt:
"""
I want to compare different operator sets. Train 3 models:
1. job_id "linear_only" - only ["+", "-", "*", "/"]
2. job_id "with_powers" - add "^" operator
3. job_id "full_operators" - add ["sin", "cos", "exp", "log"]

Use the same data for all three and compare their best equations.
"""
```

---

## Troubleshooting

### Issue 1: Server won't start

**Error**: `Address already in use`

**Solution**:
```bash
# Find and kill process on port 8000
lsof -ti:8000 | xargs kill -9

# Or use a different port
python pysr_mcp_server.py --port 8001
```

### Issue 2: Claude Desktop can't connect

**Symptoms**: Tools not appearing in Claude

**Solutions**:
1. Verify proxy is running: `ps aux | grep proxy_server`
2. Check config file path is absolute
3. Restart Claude Desktop completely
4. Check logs: Claude menu → Settings → Developer → View Logs

### Issue 3: PySR installation fails

**Error**: Julia installation problems

**Solution**:
```bash
# Manual Julia installation
python -c "import pysr; pysr.install()"

# If that fails, install Julia separately:
# Download from: https://julialang.org/downloads/
```

### Issue 4: Training is slow

**Solutions**:
- Reduce `niterations` (try 20-30 for testing)
- Reduce `populations` (try 8-10)
- Reduce `maxsize` (try 15)
- Use fewer operators
- Ensure you have good CPU resources

### Issue 5: No equations found

**Possible causes**:
- Data quality issues (NaN, infinite values)
- Target variable is constant
- Too few iterations
- Operators don't match the relationship

**Solutions**:
1. Run `validate_data` first
2. Increase `niterations`
3. Try different operator combinations
4. Check your data for patterns

### Issue 6: Memory errors

**Solution**:
```bash
# Clear models from memory
# In Claude: "Delete all models except the one I'm using"

# Or restart the server
pkill -f pysr_mcp_server
python pysr_mcp_server.py
```

### Issue 7: Import errors

**Error**: `ModuleNotFoundError: No module named 'fastmcp'`

**Solution**:
```bash
# Ensure you're in the correct environment
pip install --upgrade fastmcp pysr numpy pandas matplotlib seaborn sympy psutil

# Verify installation
python -c "import fastmcp; import pysr; print('All good!')"
```

---

## Advanced Configuration

### Custom Operators

You can define custom operators for domain-specific modeling:

```python
# In your training prompt:
"""
Train with custom operators:
- binary_operators: ["+", "-", "*", "/", "^", "max", "min"]
- unary_operators: ["sin", "cos", "exp", "log", "sqrt", "abs", "sign"]
- complexity_of_operators: {"^": 2, "exp": 3, "log": 3}
"""
```

### Constraints

Add constraints to guide the search:

```python
# Example prompt:
"""
Create a model with constraints:
- Don't allow division by small numbers
- Prioritize simpler operations
- constraints: {"/": {"complexity": 2}}
"""
```

### Performance Tuning

For large datasets:

```python
# Recommended settings for datasets > 1000 rows:
"""
Use these parameters:
- populations: 20
- population_size: 50
- niterations: 100
- Enable progress tracking
"""
```

---

## API Reference

### Complete Tool Signatures

#### train_model_complete
```python
{
    "job_id": str,              # Required
    "data": str,                # Required: CSV string or file path
    "target_column": str,       # Required
    "feature_columns": List[str],  # Optional
    "niterations": int,         # Default: 40
    "populations": int,         # Default: 15
    "population_size": int,     # Default: 33
    "binary_operators": List[str],  # Default: ["+", "-", "*", "/"]
    "unary_operators": List[str],   # Default: ["sin", "cos", "exp", "log"]
    "maxsize": int,             # Default: 20
    "parsimony": float          # Default: 0.0032
}
```

**Returns**:
```python
{
    "success": bool,
    "job_id": str,
    "model_id": str,
    "training_info": {...},
    "best_equation": {...},
    "all_equations": [...],
    "metrics": {
        "r2_score": float,
        "rmse": float,
        "mae": float,
        "mape": float
    },
    "equation_exports": {...}
}
```

---

## Performance Benchmarks

Typical training times (on modern CPU):

| Data Size | Features | Iterations | Time |
|-----------|----------|------------|------|
| 100 rows  | 1-2      | 40         | 10-30s |
| 500 rows  | 2-3      | 40         | 30-60s |
| 1000 rows | 3-5      | 40         | 1-2min |
| 5000 rows | 5+       | 100        | 5-10min |

*Times vary based on operator complexity and CPU speed*

---

## Best Practices

### 1. Data Preparation
- Clean your data (remove NaN, infinities)
- Normalize features if they have very different scales
- Use meaningful column names
- Include enough samples (minimum 20-50)

### 2. Operator Selection
- Start with basic operators `["+", "-", "*", "/"]`
- Add complexity gradually (`"^"`, `"sin"`, `"cos"`)
- Match operators to your domain (e.g., `"exp"` for growth)

### 3. Hyperparameter Tuning
- Start with defaults
- Increase iterations if equations aren't good enough
- Increase parsimony if equations are too complex
- Use more populations for difficult problems

### 4. Model Management
- Use descriptive job_ids and model_ids
- Save important models to disk
- Clean up unused models regularly
- Document your best configurations

### 5. Interpretation
- Look at the Pareto frontier (all equations)
- Balance complexity vs. accuracy
- Validate on held-out data
- Export equations for further analysis

---

## Contributing & Support

### Reporting Issues
- Server bugs: Check logs and include error traces
- PySR issues: Visit [PySR GitHub](https://github.com/MilesCranmer/PySR)
- MCP protocol: See [MCP Documentation](https://modelcontextprotocol.io)

### Resources
- **PySR Documentation**: https://astroautomata.com/PySR/
- **PySR Paper**: https://arxiv.org/abs/2305.01582
- **FastMCP**: https://github.com/jlowin/fastmcp
- **MCP Specification**: https://modelcontextprotocol.io/

---

## License

This server implementation is provided as-is for use with PySR. PySR is licensed under Apache License 2.0.

---

## Quick Reference Card

### Server Commands
```bash
# Start server
python pysr_mcp_server.py

# Start proxy
python proxy_server.py

# Check health
curl http://localhost:8000/health
```

### Essential Prompts
```
# Quick test
"Train a model on x=[1,2,3,4,5], y=[2,4,6,8,10], target=y, job_id=test1"

# List models
"Show me all active models"

# Export equation
"Export the best equation from model_test1 in LaTeX format"

# Save model
"Save model_test1 to disk"
```

### Common Operators
- Basic: `+, -, *, /`
- Powers: `^, sqrt`
- Trigonometric: `sin, cos, tan`
- Exponential: `exp, log`
- Other: `abs, sign, max, min`

---

**Happy Equation Hunting! 🔬✨**

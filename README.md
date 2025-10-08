# PySR MCP Server - Quick Start Guide

## What is PySR?

**PySR** discovers mathematical equations from data using evolutionary algorithms. It finds interpretable formulas instead of "black box" models.

**Use cases**: Scientific discovery, engineering modeling, feature engineering, finding hidden patterns in data.

---

## Architecture

```
Claude Desktop (Client)
         ↓ stdio
  proxy_server.py
         ↓ HTTP
pysr_mcp_server.py (Port 8000)
```

---

## Installation

```bash
# Install dependencies
pip install fastmcp pysr numpy pandas matplotlib seaborn sympy psutil

# Install Julia backend (takes a few minutes)
python -c "import pysr; pysr.install()"
```

---

## Setup

### 1. File Structure
```
your-project/
├── pysr_mcp_server.py
├── proxy_server.py
├── models/        # auto-created
├── data/          # auto-created
└── results/       # auto-created
```

### 2. Configure Claude Desktop

Edit: `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS)  
Or: `%APPDATA%\Claude\claude_desktop_config.json` (Windows)

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

**Replace `/absolute/path/to/` with your actual path!**

---

## Running

**Terminal 1** - Start main server:
```bash
python pysr_mcp_server.py
```

**Terminal 2** - Start proxy:
```bash
python proxy_server.py
```

**Terminal 3** - Launch Claude Desktop

---

## Available Tools (12 Total)

1. **`train_model_complete`** ⭐ - All-in-one: CSV → equations
2. **`create_model`** - Create model with config
3. **`fit_model`** - Train model on data
4. **`predict`** - Generate predictions
5. **`get_equations`** - Get all discovered equations
6. **`save_model`** - Save to disk
7. **`load_model`** - Load from disk
8. **`export_equation`** - Export (SymPy/LaTeX/Julia)
9. **`validate_data`** - Check data quality
10. **`list_models`** - List all models
11. **`delete_model`** - Remove model
12. **`health_check`** - Server status

---

## Quick Test Prompts

### Test 1: Simple Linear
```
Use train_model_complete with this data:

x,y
1,2
2,4
3,6
4,8
5,10

Target: "y", job_id: "test_001"
```

### Test 2: Polynomial
```
Discover equation for:

x,y
1,1
2,8
3,27
4,64
5,125

Target: "y", niterations: 50, job_id: "test_002"
```

### Test 3: Multiple Features
```
Find equation for:

x1,x2,y
1,2,5
2,3,13
3,4,25
4,5,41

Target: "y", job_id: "test_003"
```

### Test 4: Model Management
```
1. List all active models
2. Get equations from the first model
3. Export best equation in LaTeX
4. Check server health
```

---

## Troubleshooting

**Server won't start** (port in use):
```bash
lsof -ti:8000 | xargs kill -9
python pysr_mcp_server.py --port 8001
```

**Claude can't connect**:
- Check proxy is running: `ps aux | grep proxy_server`
- Use absolute path in config
- Restart Claude Desktop
- Check logs: Settings → Developer → View Logs

**Training is slow**:
- Reduce `niterations` (try 20-30)
- Reduce `populations` (try 8-10)
- Use fewer operators

**Import errors**:
```bash
pip install --upgrade fastmcp pysr numpy pandas matplotlib seaborn sympy psutil
```

---

## Quick Reference

### Commands
```bash
python pysr_mcp_server.py              # Start server
python proxy_server.py                  # Start proxy
curl http://localhost:8000/health       # Check health
```

### Common Parameters
- **operators**: `["+", "-", "*", "/", "^", "sin", "cos", "exp", "log", "sqrt"]`
- **niterations**: 40 (default), increase for better results
- **parsimony**: 0.0032 (default), increase for simpler equations
- **maxsize**: 20 (default), max equation complexity

---

## Resources

- **PySR Docs**: https://astroautomata.com/PySR/
- **PySR Paper**: https://arxiv.org/abs/2305.01582
- **MCP Spec**: https://modelcontextprotocol.io/

---

**Happy Equation Discovery! 🔬**

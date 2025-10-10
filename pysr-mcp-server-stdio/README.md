# PySR MCP Server

[![npm version](https://badge.fury.io/js/@neural-symphony%2Fpysr-mcp-server.svg)](https://www.npmjs.com/package/@neural-symphony/pysr-mcp-server)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An MCP (Model Context Protocol) server that brings the power of [PySR](https://github.com/MilesCranmer/PySR) symbolic regression to Claude Code and other MCP clients.

## 🎯 What is This?

This server enables AI assistants to **automatically discover mathematical equations** from data using evolutionary algorithms. Instead of manually fitting models, let AI find interpretable symbolic expressions that describe your data.

### Perfect For:
- 🔬 **Scientists** - Discover physical laws from experimental data
- 📊 **Data Scientists** - Find interpretable models instead of black boxes
- 🎓 **Researchers** - Generate hypotheses about data relationships
- 🤖 **ML Engineers** - Create interpretable features for machine learning

---

## ✨ Features

### Four Powerful Tools

1. **`fit_symbolic_regression`** - Train models to discover equations from data
2. **`get_equations`** - Retrieve all discovered formulas with metrics
3. **`predict`** - Make predictions with discovered equations
4. **`export_equation`** - Export equations to LaTeX, SymPy, PyTorch, or JAX

---

## 🚀 Quick Start

### Prerequisites

```bash
# Install Python dependencies
pip install pysr numpy pandas
```

> **Note**: PySR will automatically install Julia dependencies on first use.

### Installation

Install the MCP server with one command:

```bash
claude mcp add pysr npx @neural-symphony/pysr-mcp-server
```

**Restart Claude Code** to load the new tools.

## Option 2: Manual Configuration
If you prefer manual setup, edit your Claude Desktop config file:
Config file location:

Windows: %APPDATA%\Claude\claude_desktop_config.json
Mac: ~/Library/Application Support/Claude/claude_desktop_config.json
Linux: ~/.config/Claude/claude_desktop_config.json

Add this configuration:
json{
  "mcpServers": {
    "pysr": {
      "command": "npx",
      "args": [
        "@neural-symphony/pysr-mcp-server"
      ]
    }
  }
}
Then restart Claude Desktop.

### Verification

```bash
claude mcp list
```

You should see `pysr` in the list.

---

## 📖 Usage

### Example: Discover an Equation

Ask Claude Code:

```
"I have data where X = [[1], [2], [3], [4], [5]] and y = [5, 7, 9, 11, 13].
Can you find the equation?"
```

Claude will:
1. Train a symbolic regression model
2. Discover: `y = 2*x + 3`
3. Show accuracy metrics

---

## 📚 Documentation

- **[API Reference](docs/API.md)** - Complete tool documentation
- **[Examples](docs/EXAMPLES.md)** - Real-world use cases
- **[Usage Guide](USAGE_GUIDE.md)** - Detailed instructions
- **[Troubleshooting](TROUBLESHOOTING.md)** - Common issues
- **[Changelog](CHANGELOG.md)** - Version history

---

## 🔧 Configuration

### Custom Python Path

```bash
claude mcp add pysr --env PYTHON_PATH=/path/to/python -- npx @neural-symphony/pysr-mcp-server
```

### Training Parameters

```javascript
{
  "niterations": 100,
  "binary_operators": ["+", "*", "-", "/"],
  "unary_operators": ["cos", "sin", "exp"],
  "maxsize": 20,
  "timeout_in_seconds": 300
}
```

---

## 🌟 Tool Reference

### fit_symbolic_regression

**Required**: `X` (2D array), `y` (1D array)
**Optional**: `niterations`, `binary_operators`, `unary_operators`, `maxsize`, `timeout_in_seconds`
**Returns**: Model path

### get_equations

**Required**: `model_path`
**Returns**: Array of equations with metrics

### predict

**Required**: `model_path`, `X`
**Optional**: `equation_index`
**Returns**: Predictions array

### export_equation

**Required**: `model_path`, `format` (`"latex"`, `"sympy"`, `"torch"`, `"jax"`)
**Optional**: `equation_index`
**Returns**: Equation in requested format

---

## 🛠️ Development

```bash
git clone https://github.com/neural-symphony/pysr-mcp-server.git
cd pysr-mcp-server
npm install
npm run build
claude mcp add pysr-dev node $(pwd)/dist/index.js
```

---

## 📄 License

MIT License - see [LICENSE](LICENSE)

---

## 🙏 Acknowledgments

- **[PySR](https://github.com/MilesCranmer/PySR)** - Symbolic regression engine
- **[Anthropic](https://anthropic.com)** - Model Context Protocol
- **[Claude Code](https://claude.ai/code)** - AI coding assistant

---

## 📞 Support

- **Issues**: [GitHub](https://github.com/neural-symphony/pysr-mcp-server/issues)
- **PySR Docs**: [https://ai.damtp.cam.ac.uk/pysr/](https://ai.damtp.cam.ac.uk/pysr/)
- **MCP Protocol**: [https://modelcontextprotocol.io/](https://modelcontextprotocol.io/)

---

<div align="center">

**Version 0.1.0** • Made with ❤️ for the scientific community

*Discover equations, not just patterns*

</div>

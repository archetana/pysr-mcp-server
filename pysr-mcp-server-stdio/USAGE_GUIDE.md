# How to Use PySR MCP Server

The MCP server is now installed and ready to use! You can access PySR tools directly through Claude Code.

## Available Tools

You now have access to 4 symbolic regression tools:

1. **fit_symbolic_regression** - Train a model to find equations
2. **get_equations** - View all discovered equations
3. **predict** - Make predictions with the model
4. **export_equation** - Export equations to different formats

## Quick Start Example

### Step 1: Ask Claude to Train a Model

Simply ask Claude Code:

```
"Please use the fit_symbolic_regression tool to train a model on the data from data_for_mcp.json.
Use 50 iterations and include operators: +, -, *, /, cos, exp, sin"
```

Claude will automatically:
- Read the JSON file with your data
- Call the `fit_symbolic_regression` tool
- Return a model path and discovered equations

### Step 2: View All Equations

Once you have a model_path, ask:

```
"Use the get_equations tool with model_path '<the_path_from_step_1>' to show all discovered equations"
```

### Step 3: Get LaTeX Format

To export the best equation:

```
"Use export_equation tool with model_path '<the_path>' and format 'latex'"
```

## Natural Language Examples

You can simply talk to Claude Code naturally:

### Example 1: Basic Training
```
"Train a symbolic regression model on my data_complex.csv file.
I want to find a formula that relates x1, x2, x3 to y."
```

### Example 2: Customized Training
```
"Use the PySR MCP tool to fit a model with these settings:
- 100 iterations
- Binary operators: +, -, *, /
- Unary operators: cos, sin, exp, log
- Maximum complexity: 15
- Timeout: 300 seconds"
```

### Example 3: Getting Results
```
"Show me all the equations discovered by the model, ranked by accuracy"
```

### Example 4: Making Predictions
```
"Use the trained model to predict y for these new values:
X = [[1.0, 2.0, 3.0], [0.5, -1.5, 2.5]]"
```

### Example 5: Export Equation
```
"Export the best equation in PyTorch format so I can use it in my neural network"
```

## Your Data

Your CSV data has been loaded and prepared:
- **Samples**: 200 data points
- **Features**: x1, x2, x3
- **Target**: y
- **Format**: JSON file at `data_for_mcp.json`

## Example Conversation Flow

**You**: "Train a symbolic regression model using the data in data_for_mcp.json"

**Claude**: *Uses fit_symbolic_regression tool and returns the model path and results*

**You**: "What's the best equation in LaTeX format?"

**Claude**: *Uses export_equation tool with format='latex'*

**You**: "Can you show me all the equations sorted by complexity?"

**Claude**: *Uses get_equations tool and displays them*

## Tool Parameters Reference

### fit_symbolic_regression
- `X`: Your feature matrix (required)
- `y`: Your target values (required)
- `niterations`: How long to search (default: 40)
- `binary_operators`: e.g., ["+", "*", "-", "/"]
- `unary_operators`: e.g., ["cos", "exp", "sin"]
- `maxsize`: Max equation complexity (default: 20)
- `timeout_in_seconds`: Max training time (default: 300)

### get_equations
- `model_path`: Path returned from fit_symbolic_regression

### predict
- `model_path`: Path to your model
- `X`: Data to predict on
- `equation_index`: (optional) Which equation to use

### export_equation
- `model_path`: Path to your model
- `format`: "latex", "sympy", "torch", or "jax"
- `equation_index`: (optional) Which equation to export

## Tips

1. **Start Simple**: Use fewer iterations (40-50) for quick tests
2. **Increase Gradually**: Use more iterations (100+) for better results
3. **Watch Timeout**: Long training might timeout, adjust `timeout_in_seconds`
4. **Check Complexity**: Simpler equations (lower complexity) are often better
5. **Try Different Operators**: Experiment with operator combinations

## Troubleshooting

If you get an error:
- Make sure PySR is installed: `pip install pysr`
- Check that Python is accessible from command line
- Verify the model_path exists when using other tools
- Check data format (X should be 2D array, y should be 1D array)

## Next Steps

Now you can simply ask Claude Code to:
- Train models on your data
- Explore discovered equations
- Make predictions
- Export equations for use in other frameworks

Just use natural language - Claude will handle the MCP tool calls automatically!

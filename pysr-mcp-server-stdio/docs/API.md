# API Documentation

Complete reference for all PySR MCP Server tools.

## Overview

The PySR MCP Server provides 4 tools that enable AI assistants to perform symbolic regression on data. All tools follow the MCP (Model Context Protocol) specification.

---

## Tools

### 1. fit_symbolic_regression

Train a PySR model to discover symbolic expressions that fit the given data.

#### Input Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `X` | `number[][]` | ✅ Yes | - | Feature matrix (2D array of numbers) |
| `y` | `number[]` | ✅ Yes | - | Target values (1D array of numbers) |
| `niterations` | `number` | No | 40 | Number of training iterations |
| `populations` | `number` | No | 8 | Number of populations for evolutionary algorithm |
| `population_size` | `number` | No | 50 | Size of each population |
| `binary_operators` | `string[]` | No | `["+", "*", "-", "/"]` | Binary operators to use in equations |
| `unary_operators` | `string[]` | No | `["cos", "exp", "sin"]` | Unary operators to use in equations |
| `maxsize` | `number` | No | 20 | Maximum complexity of equations |
| `maxdepth` | `number` | No | 10 | Maximum depth of equation trees |
| `timeout_in_seconds` | `number` | No | 300 | Maximum training time in seconds |

#### Output

Returns a text response containing:
- Model path (string) - Path to the saved model file
- Training summary - Information about the training process
- Best equation found - Preview of the best discovered equation

#### Example

```javascript
{
  "name": "fit_symbolic_regression",
  "arguments": {
    "X": [[1.0, 2.0], [1.5, 2.5], [2.0, 3.0]],
    "y": [5.0, 7.5, 10.0],
    "niterations": 50,
    "binary_operators": ["+", "*", "-", "/"],
    "unary_operators": ["cos", "sin", "exp"],
    "maxsize": 15,
    "timeout_in_seconds": 120
  }
}
```

#### Notes

- Training time varies based on data size and number of iterations
- The model path can be used with other tools
- Larger `niterations` generally produces better equations but takes longer
- Custom operators can be defined using Julia syntax

---

### 2. get_equations

Retrieve all discovered equations from a trained PySR model.

#### Input Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model_path` | `string` | ✅ Yes | Path to the saved model (from `fit_symbolic_regression`) |

#### Output

Returns JSON array containing all discovered equations with:
- `complexity` - Complexity score of the equation
- `loss` - Loss/error of the equation
- `score` - Overall score (balances accuracy and complexity)
- `equation` - The equation in symbolic form

#### Example

```javascript
{
  "name": "get_equations",
  "arguments": {
    "model_path": "/tmp/pysr-mcp/model_1234567890.pkl"
  }
}
```

#### Response Format

```json
[
  {
    "complexity": 3,
    "loss": 0.0012,
    "score": 0.98,
    "equation": "x0 + x1"
  },
  {
    "complexity": 5,
    "loss": 0.0001,
    "score": 0.99,
    "equation": "x0 * x1 + 2.5"
  }
]
```

---

### 3. predict

Make predictions using a trained PySR model.

#### Input Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model_path` | `string` | ✅ Yes | Path to the saved model |
| `X` | `number[][]` | ✅ Yes | Feature matrix for prediction (2D array) |
| `equation_index` | `number` | No | Specific equation to use (default: best equation) |

#### Output

Returns an array of predictions (one per input row).

#### Example

```javascript
{
  "name": "predict",
  "arguments": {
    "model_path": "/tmp/pysr-mcp/model_1234567890.pkl",
    "X": [[1.0, 2.0], [1.5, 2.5], [2.0, 3.0]],
    "equation_index": 2
  }
}
```

#### Notes

- If `equation_index` is omitted, uses the best equation (selected by PySR)
- Input X must have the same number of features as training data

---

### 4. export_equation

Export a discovered equation in various formats.

#### Input Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model_path` | `string` | ✅ Yes | Path to the saved model |
| `format` | `string` | ✅ Yes | Export format: `"latex"`, `"sympy"`, `"torch"`, or `"jax"` |
| `equation_index` | `number` | No | Specific equation to export (default: best equation) |

#### Output

Returns the equation in the requested format.

#### Supported Formats

| Format | Description | Use Case |
|--------|-------------|----------|
| `latex` | LaTeX mathematical notation | Documentation, papers, presentations |
| `sympy` | SymPy expression (Python) | Symbolic manipulation, analysis |
| `torch` | PyTorch model | Deep learning integration, differentiable models |
| `jax` | JAX/Flax function | High-performance computing, automatic differentiation |

#### Example

```javascript
{
  "name": "export_equation",
  "arguments": {
    "model_path": "/tmp/pysr-mcp/model_1234567890.pkl",
    "format": "latex",
    "equation_index": 0
  }
}
```

#### Example Outputs

**LaTeX**:
```latex
x_{0} + 2.5 \cdot \cos(x_{1})
```

**SymPy**:
```python
x0 + 2.5*cos(x1)
```

**PyTorch**:
```python
def pytorch_model(X):
    x0 = X[:, 0]
    x1 = X[:, 1]
    return x0 + 2.5 * torch.cos(x1)
```

**JAX**:
```python
def jax_model(X):
    x0 = X[:, 0]
    x1 = X[:, 1]
    return x0 + 2.5 * jnp.cos(x1)
```

---

## Error Handling

All tools return errors in the following format:

```json
{
  "content": [{
    "type": "text",
    "text": "Error: [error message]"
  }],
  "isError": true
}
```

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `Python process exited with code 1` | Python/PySR not installed | Install Python and PySR |
| `No module named 'numpy'` | Missing dependencies | Run `pip install numpy pysr pandas` |
| `FileNotFoundError` | Invalid model path | Check model_path is correct |
| `ValueError: X has wrong shape` | Incorrect input dimensions | Verify X is 2D array, y is 1D array |

---

## Usage Workflow

Typical usage follows this pattern:

1. **Train Model**
   ```javascript
   fit_symbolic_regression(X, y, options)
   // Returns: model_path
   ```

2. **Explore Equations**
   ```javascript
   get_equations(model_path)
   // Returns: array of equations with metrics
   ```

3. **Make Predictions**
   ```javascript
   predict(model_path, new_X)
   // Returns: predictions
   ```

4. **Export for Use**
   ```javascript
   export_equation(model_path, "torch")
   // Returns: PyTorch code
   ```

---

## Advanced Configuration

### Custom Operators

Define custom operators using Julia syntax:

```javascript
{
  "binary_operators": ["+", "*", "-", "/", "pow"],
  "unary_operators": [
    "sin",
    "cos",
    "square(x) = x^2",
    "cube(x) = x^3",
    "inv(x) = 1/x"
  ]
}
```

### Performance Tuning

For faster training:
- Reduce `niterations`
- Decrease `populations` and `population_size`
- Limit `maxsize` and `maxdepth`
- Set shorter `timeout_in_seconds`

For better accuracy:
- Increase `niterations`
- Increase `populations` and `population_size`
- Allow more complex equations with higher `maxsize`

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PYTHON_PATH` | Path to Python executable | `python` |
| `PYSR_PYTHON` | Alternative Python path | `python` |

Set when adding to Claude Code:

```bash
claude mcp add pysr --env PYTHON_PATH=/path/to/python -- npx @neural-symphony/pysr-mcp-server
```

---

## Limits

- **Data Size**: Recommended < 10,000 samples for reasonable training time
- **Features**: Best results with < 20 features
- **Training Time**: Default timeout is 300 seconds
- **Model Size**: Stored models are typically < 1 MB
- **Complexity**: Higher complexity equations may overfit

---

## Best Practices

1. **Start Simple**: Begin with fewer iterations and operators
2. **Iterate**: Train multiple times with different parameters
3. **Validate**: Always check equations on holdout data
4. **Balance**: Trade off complexity vs accuracy
5. **Document**: Save equation exports for reproducibility

---

## Support

For issues or questions:
- Check [TROUBLESHOOTING.md](../TROUBLESHOOTING.md)
- Review [USAGE_GUIDE.md](../USAGE_GUIDE.md)
- File an issue on GitHub

---

## References

- [PySR Documentation](https://ai.damtp.cam.ac.uk/pysr/)
- [Model Context Protocol](https://modelcontextprotocol.io/)
- [SymbolicRegression.jl](https://github.com/MilesCranmer/SymbolicRegression.jl)

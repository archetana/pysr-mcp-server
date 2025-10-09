# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-10-07

### Added
- Initial release of PySR MCP Server
- Four main MCP tools for symbolic regression:
  - `fit_symbolic_regression` - Train models to discover mathematical equations from data
  - `get_equations` - Retrieve all discovered equations with performance metrics
  - `predict` - Make predictions using trained models
  - `export_equation` - Export equations to LaTeX, SymPy, PyTorch, or JAX formats
- Python-Node.js bridge for PySR integration
- Comprehensive error handling and validation
- Temporary file management for data exchange
- TypeScript support with full type definitions
- Complete documentation (README, USAGE_GUIDE, TROUBLESHOOTING)
- MIT License

### Features
- Configurable training parameters (iterations, operators, complexity limits)
- Support for custom binary and unary operators
- Timeout configuration for long-running training sessions
- Model persistence via pickle files
- Multiple equation export formats
- Environment variable support for Python path configuration

### Technical Details
- Built with @modelcontextprotocol/sdk v1.0.0
- Node.js >=18.0.0 required
- Python >=3.8 with PySR required
- Stdio-based MCP transport
- File-based data exchange for stability

[0.1.0]: https://github.com/neural-symphony/pysr-mcp-server/releases/tag/v0.1.0

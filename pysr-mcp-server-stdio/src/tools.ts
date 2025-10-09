import { Tool } from '@modelcontextprotocol/sdk/types.js';

export const tools: Tool[] = [
  {
    name: 'fit_symbolic_regression',
    description:
      'Train a PySR model to find symbolic expressions that fit the given data. ' +
      'Takes X (features) and y (target) arrays, along with optional configuration parameters. ' +
      'Returns a model_path that can be used with other tools.',
    inputSchema: {
      type: 'object',
      properties: {
        X: {
          type: 'array',
          description: 'Feature matrix (2D array)',
          items: {
            type: 'array',
            items: { type: 'number' },
          },
        },
        y: {
          type: 'array',
          description: 'Target values (1D array)',
          items: { type: 'number' },
        },
        niterations: {
          type: 'number',
          description: 'Number of iterations to run (default: 40)',
        },
        populations: {
          type: 'number',
          description: 'Number of populations (default: 8)',
        },
        population_size: {
          type: 'number',
          description: 'Size of each population (default: 50)',
        },
        binary_operators: {
          type: 'array',
          description: 'Binary operators to use (default: ["+", "*", "-", "/"])',
          items: { type: 'string' },
        },
        unary_operators: {
          type: 'array',
          description: 'Unary operators to use (default: ["cos", "exp", "sin"])',
          items: { type: 'string' },
        },
        maxsize: {
          type: 'number',
          description: 'Maximum complexity of equations (default: 20)',
        },
        maxdepth: {
          type: 'number',
          description: 'Maximum depth of equation trees (default: 10)',
        },
        timeout_in_seconds: {
          type: 'number',
          description: 'Maximum time for training in seconds (default: 300)',
        },
      },
      required: ['X', 'y'],
    },
  },
  {
    name: 'get_equations',
    description:
      'Retrieve all discovered equations from a trained PySR model. ' +
      'Returns a list of equations with their scores, complexity, and other metrics.',
    inputSchema: {
      type: 'object',
      properties: {
        model_path: {
          type: 'string',
          description: 'Path to the saved PySR model (from fit_symbolic_regression)',
        },
      },
      required: ['model_path'],
    },
  },
  {
    name: 'predict',
    description:
      'Make predictions using a trained PySR model. ' +
      'Can optionally specify which equation to use by index.',
    inputSchema: {
      type: 'object',
      properties: {
        model_path: {
          type: 'string',
          description: 'Path to the saved PySR model',
        },
        X: {
          type: 'array',
          description: 'Feature matrix for prediction (2D array)',
          items: {
            type: 'array',
            items: { type: 'number' },
          },
        },
        equation_index: {
          type: 'number',
          description: 'Optional: Index of specific equation to use (default: best)',
        },
      },
      required: ['model_path', 'X'],
    },
  },
  {
    name: 'export_equation',
    description:
      'Export a discovered equation in various formats (latex, sympy, torch, jax). ' +
      'Useful for documentation or integration with other frameworks.',
    inputSchema: {
      type: 'object',
      properties: {
        model_path: {
          type: 'string',
          description: 'Path to the saved PySR model',
        },
        format: {
          type: 'string',
          enum: ['latex', 'sympy', 'torch', 'jax'],
          description: 'Export format',
        },
        equation_index: {
          type: 'number',
          description: 'Optional: Index of specific equation to export (default: best)',
        },
      },
      required: ['model_path', 'format'],
    },
  },
];

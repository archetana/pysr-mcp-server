import { spawn } from 'child_process';
import * as path from 'path';
import * as fs from 'fs/promises';
import * as os from 'os';

export interface FitOptions {
  niterations?: number;
  populations?: number;
  population_size?: number;
  binary_operators?: string[];
  unary_operators?: string[];
  maxsize?: number;
  maxdepth?: number;
  constraints?: Record<string, number | [number, number]>;
  model_selection?: string;
  timeout_in_seconds?: number;
}

export interface PySRResult {
  success: boolean;
  message?: string;
  equations?: any[];
  model_path?: string;
  error?: string;
}

export class PySRWrapper {
  private pythonPath: string;
  private workDir: string;

  constructor(pythonPath?: string) {
    this.pythonPath = pythonPath || 'python';
    this.workDir = path.join(os.tmpdir(), 'pysr-mcp');
  }

  async initialize(): Promise<void> {
    // Create working directory
    await fs.mkdir(this.workDir, { recursive: true });
  }

  async fit(
    X: number[][],
    y: number[],
    options: FitOptions = {}
  ): Promise<PySRResult> {
    try {
      // Generate unique session ID
      const sessionId = Date.now().toString();
      const dataPath = path.join(this.workDir, `data_${sessionId}.json`);
      const scriptPath = path.join(this.workDir, `fit_${sessionId}.py`);
      const modelPath = path.join(this.workDir, `model_${sessionId}.pkl`);

      // Save data to file
      await fs.writeFile(
        dataPath,
        JSON.stringify({ X, y, options, model_path: modelPath })
      );

      // Generate Python script
      const script = this.generateFitScript(dataPath);
      await fs.writeFile(scriptPath, script);

      // Execute Python script
      const result = await this.executePython(scriptPath);

      // Clean up script and data
      await fs.unlink(scriptPath).catch(() => {});
      await fs.unlink(dataPath).catch(() => {});

      return {
        success: true,
        message: result.stdout,
        model_path: modelPath,
      };
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : String(error),
      };
    }
  }

  async getEquations(modelPath: string): Promise<PySRResult> {
    try {
      const sessionId = Date.now().toString();
      const scriptPath = path.join(this.workDir, `get_eqs_${sessionId}.py`);
      const outputPath = path.join(this.workDir, `eqs_${sessionId}.json`);

      const script = this.generateGetEquationsScript(modelPath, outputPath);
      await fs.writeFile(scriptPath, script);

      await this.executePython(scriptPath);

      // Read equations
      const equationsData = await fs.readFile(outputPath, 'utf-8');
      const equations = JSON.parse(equationsData);

      // Clean up
      await fs.unlink(scriptPath).catch(() => {});
      await fs.unlink(outputPath).catch(() => {});

      return {
        success: true,
        equations,
      };
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : String(error),
      };
    }
  }

  async predict(
    modelPath: string,
    X: number[][],
    equationIndex?: number
  ): Promise<PySRResult> {
    try {
      const sessionId = Date.now().toString();
      const dataPath = path.join(this.workDir, `pred_data_${sessionId}.json`);
      const scriptPath = path.join(this.workDir, `predict_${sessionId}.py`);
      const outputPath = path.join(this.workDir, `pred_${sessionId}.json`);

      await fs.writeFile(
        dataPath,
        JSON.stringify({ X, equation_index: equationIndex })
      );

      const script = this.generatePredictScript(modelPath, dataPath, outputPath);
      await fs.writeFile(scriptPath, script);

      await this.executePython(scriptPath);

      const predData = await fs.readFile(outputPath, 'utf-8');
      const predictions = JSON.parse(predData);

      // Clean up
      await fs.unlink(scriptPath).catch(() => {});
      await fs.unlink(dataPath).catch(() => {});
      await fs.unlink(outputPath).catch(() => {});

      return {
        success: true,
        message: JSON.stringify(predictions),
      };
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : String(error),
      };
    }
  }

  async exportEquation(
    modelPath: string,
    format: 'latex' | 'sympy' | 'torch' | 'jax',
    equationIndex?: number
  ): Promise<PySRResult> {
    try {
      const sessionId = Date.now().toString();
      const scriptPath = path.join(this.workDir, `export_${sessionId}.py`);
      const outputPath = path.join(this.workDir, `export_${sessionId}.txt`);

      const script = this.generateExportScript(
        modelPath,
        format,
        outputPath,
        equationIndex
      );
      await fs.writeFile(scriptPath, script);

      await this.executePython(scriptPath);

      const exportData = await fs.readFile(outputPath, 'utf-8');

      // Clean up
      await fs.unlink(scriptPath).catch(() => {});
      await fs.unlink(outputPath).catch(() => {});

      return {
        success: true,
        message: exportData,
      };
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : String(error),
      };
    }
  }

  private generateFitScript(dataPath: string): string {
    return `
import json
import numpy as np
from pysr import PySRRegressor

# Load data
with open(${JSON.stringify(dataPath)}, 'r') as f:
    data = json.load(f)

X = np.array(data['X'])
y = np.array(data['y'])
options = data['options']
model_path = data['model_path']

# Create model
model = PySRRegressor(
    niterations=options.get('niterations', 40),
    populations=options.get('populations', 8),
    population_size=options.get('population_size', 50),
    binary_operators=options.get('binary_operators', ['+', '*', '-', '/']),
    unary_operators=options.get('unary_operators', ['cos', 'exp', 'sin']),
    maxsize=options.get('maxsize', 20),
    maxdepth=options.get('maxdepth', 10),
    model_selection=options.get('model_selection', 'best'),
    timeout_in_seconds=options.get('timeout_in_seconds', 300),
    progress=False,
)

# Fit model
model.fit(X, y)

# Save model
import pickle
with open(model_path, 'wb') as f:
    pickle.dump(model, f)

print(f"Model trained and saved to {model_path}")
print(f"Best equation: {model}")
`;
  }

  private generateGetEquationsScript(
    modelPath: string,
    outputPath: string
  ): string {
    return `
import json
import pickle
import pandas as pd

# Load model
with open(${JSON.stringify(modelPath)}, 'rb') as f:
    model = pickle.load(f)

# Get equations
equations_df = model.equations_
equations_list = equations_df.to_dict('records')

# Save to JSON
with open(${JSON.stringify(outputPath)}, 'w') as f:
    json.dump(equations_list, f, indent=2, default=str)

print(f"Equations saved to {${JSON.stringify(outputPath)}}")
`;
  }

  private generatePredictScript(
    modelPath: string,
    dataPath: string,
    outputPath: string
  ): string {
    return `
import json
import pickle
import numpy as np

# Load model
with open(${JSON.stringify(modelPath)}, 'rb') as f:
    model = pickle.load(f)

# Load data
with open(${JSON.stringify(dataPath)}, 'r') as f:
    data = json.load(f)

X = np.array(data['X'])
equation_index = data.get('equation_index')

# Predict
if equation_index is not None:
    predictions = model.predict(X, equation_index)
else:
    predictions = model.predict(X)

# Save predictions
with open(${JSON.stringify(outputPath)}, 'w') as f:
    json.dump(predictions.tolist(), f)

print(f"Predictions saved to {${JSON.stringify(outputPath)}}")
`;
  }

  private generateExportScript(
    modelPath: string,
    format: string,
    outputPath: string,
    equationIndex?: number
  ): string {
    const indexArg = equationIndex !== undefined ? equationIndex : '';
    return `
import pickle

# Load model
with open(${JSON.stringify(modelPath)}, 'rb') as f:
    model = pickle.load(f)

# Export equation
format = ${JSON.stringify(format)}
if format == 'latex':
    result = model.latex(${indexArg})
elif format == 'sympy':
    result = str(model.sympy(${indexArg}))
elif format == 'torch':
    result = str(model.pytorch(${indexArg}))
elif format == 'jax':
    result = str(model.jax(${indexArg}))
else:
    result = "Unknown format"

# Save result
with open(${JSON.stringify(outputPath)}, 'w') as f:
    f.write(str(result))

print(f"Equation exported to {${JSON.stringify(outputPath)}}")
`;
  }

  private executePython(scriptPath: string): Promise<{ stdout: string; stderr: string }> {
    return new Promise((resolve, reject) => {
      const python = spawn(this.pythonPath, [scriptPath]);
      let stdout = '';
      let stderr = '';

      python.stdout.on('data', (data) => {
        stdout += data.toString();
      });

      python.stderr.on('data', (data) => {
        stderr += data.toString();
      });

      python.on('close', (code) => {
        if (code === 0) {
          resolve({ stdout, stderr });
        } else {
          reject(new Error(`Python process exited with code ${code}\n${stderr}`));
        }
      });

      python.on('error', (error) => {
        reject(error);
      });
    });
  }
}

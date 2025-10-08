#!/usr/bin/env python3
"""
PySR MCP Server - FastMCP implementation for symbolic regression
Provides 12 core tools for PySR model training and management
Uses HTTP transport for communication
"""

import os
import json
import pickle
import asyncio
import logging
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import uuid

import numpy as np
import pandas as pd
from fastmcp import FastMCP
from pydantic import BaseModel, Field
import pysr
from pysr import PySRRegressor
import sympy
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO, StringIO
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastMCP server instance
mcp = FastMCP("PySR Symbolic Regression Server")

# Global storage for models and training state
models_storage: Dict[str, PySRRegressor] = {}
training_state: Dict[str, Dict] = {}
data_storage: Dict[str, Dict] = {}

# Ensure directories exist
MODELS_DIR = Path("models")
DATA_DIR = Path("data")
RESULTS_DIR = Path("results")

for dir_path in [MODELS_DIR, DATA_DIR, RESULTS_DIR]:
    dir_path.mkdir(exist_ok=True)

class ModelConfig(BaseModel):
    """Configuration for PySR model creation"""
    binary_operators: List[str] = Field(default=["+", "-", "*", "/"], description="Binary operators for regression")
    unary_operators: List[str] = Field(default=["cos", "sin", "exp", "log", "sqrt"], description="Unary operators for regression")
    niterations: int = Field(default=40, description="Number of iterations to run")
    populations: int = Field(default=8, description="Number of populations")
    population_size: int = Field(default=33, description="Size of each population")
    maxsize: int = Field(default=20, description="Maximum equation complexity")
    maxdepth: int = Field(default=10, description="Maximum equation depth")
    parsimony: float = Field(default=0.0032, description="Parsimony coefficient")
    complexity_of_operators: Dict[str, int] = Field(default_factory=dict, description="Custom operator complexities")
    constraints: Dict[str, Any] = Field(default_factory=dict, description="Operator constraints")
    nested_constraints: Dict[str, Dict] = Field(default_factory=dict, description="Nested operator constraints")
    loss: str = Field(default="L2DistLoss()", description="Loss function")
    model_selection: str = Field(default="best", description="Model selection criteria")
    random_state: Optional[int] = Field(default=None, description="Random seed for reproducibility")
    
class TrainingData(BaseModel):
    """Structure for training data"""
    X: List[List[float]] = Field(description="Feature matrix")
    y: List[float] = Field(description="Target values")
    feature_names: Optional[List[str]] = Field(default=None, description="Names of features")
    variable_names: Optional[List[str]] = Field(default=None, description="Variable names for equations")

class PredictionRequest(BaseModel):
    """Request for model predictions"""
    model_id: str = Field(description="Model identifier")
    X: List[List[float]] = Field(description="Input features for prediction")

class EquationExport(BaseModel):
    """Configuration for equation export"""
    model_id: str = Field(description="Model identifier")
    format: str = Field(default="sympy", description="Export format: sympy, latex, julia, jax, torch")
    equation_index: Optional[int] = Field(default=None, description="Specific equation index (None for best)")


def parse_csv_data(data: str) -> pd.DataFrame:
    """
    Parse CSV data from string or file path.
    
    Args:
        data: CSV data as string or file path
        
    Returns:
        Pandas DataFrame
    """
    try:
        # Check if it's a file path
        if os.path.exists(data):
            return pd.read_csv(data)
        else:
            # Try parsing as CSV string
            return pd.read_csv(StringIO(data))
    except Exception as e:
        raise ValueError(f"Failed to parse CSV data: {str(e)}")


@mcp.tool()
async def train_model_complete(
    job_id: str = Field(description="Unique job identifier"),
    data: str = Field(description="CSV data as string or file path"),
    target_column: str = Field(description="Name of target variable column"),
    feature_columns: Optional[List[str]] = Field(default=None, description="List of feature column names (None = auto-select all except target)"),
    niterations: int = Field(default=40, description="Number of iterations"),
    populations: int = Field(default=15, description="Number of populations"),
    population_size: int = Field(default=33, description="Population size"),
    binary_operators: Optional[List[str]] = Field(default=None, description="Binary operators (default: ['+', '-', '*', '/'])"),
    unary_operators: Optional[List[str]] = Field(default=None, description="Unary operators (default: ['sin', 'cos', 'exp', 'log'])"),
    maxsize: int = Field(default=20, description="Maximum complexity"),
    parsimony: float = Field(default=0.0032, description="Parsimony coefficient")
) -> Dict[str, Any]:
    """
    Complete training pipeline: validates data, trains model, and returns all results.
    This is an all-in-one tool that takes CSV data and returns discovered equations.
    
    Args:
        job_id: Unique job identifier
        data: CSV data as string or file path
        target_column: Name of target variable column
        feature_columns: List of feature column names (None = auto-select all except target)
        niterations: Number of iterations (default: 40)
        populations: Number of populations (default: 15)
        population_size: Population size (default: 33)
        binary_operators: List of binary operators (default: ["+", "-", "*", "/"])
        unary_operators: List of unary operators (default: ["sin", "cos", "exp", "log"])
        maxsize: Maximum complexity (default: 20)
        parsimony: Parsimony coefficient (default: 0.0032)
        
    Returns:
        JSON with complete training results including best equations, predictions, and metrics
    """
    try:
        logger.info(f"Starting complete training pipeline for job {job_id}")
        
        # Step 1: Parse and validate data
        logger.info("Step 1: Parsing CSV data")
        try:
            df = parse_csv_data(data)
        except Exception as e:
            return {
                "success": False,
                "job_id": job_id,
                "error": f"Failed to parse CSV data: {str(e)}"
            }
        
        # Validate target column exists
        if target_column not in df.columns:
            return {
                "success": False,
                "job_id": job_id,
                "error": f"Target column '{target_column}' not found in data. Available columns: {list(df.columns)}"
            }
        
        # Select features
        if feature_columns is None:
            feature_columns = [col for col in df.columns if col != target_column]
        else:
            # Validate feature columns exist
            missing_cols = [col for col in feature_columns if col not in df.columns]
            if missing_cols:
                return {
                    "success": False,
                    "job_id": job_id,
                    "error": f"Feature columns not found: {missing_cols}"
                }
        
        # Extract X and y
        X = df[feature_columns].values
        y = df[target_column].values
        
        logger.info(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Step 2: Validate data quality
        logger.info("Step 2: Validating data quality")
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            return {
                "success": False,
                "job_id": job_id,
                "error": "Feature matrix contains NaN or infinite values"
            }
        
        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            return {
                "success": False,
                "job_id": job_id,
                "error": "Target vector contains NaN or infinite values"
            }
        
        # Step 3: Create and configure model
        logger.info("Step 3: Creating PySR model")
        
        if binary_operators is None:
            binary_operators = ["+", "-", "*", "/"]
        if unary_operators is None:
            unary_operators = ["sin", "cos", "exp", "log"]
        
        model = PySRRegressor(
            binary_operators=binary_operators,
            unary_operators=unary_operators,
            niterations=niterations,
            populations=populations,
            population_size=population_size,
            maxsize=maxsize,
            parsimony=parsimony,
            loss="L2DistLoss()",
            model_selection="best",
            progress=True,
            verbosity=1
        )
        
        # Set feature names
        model.feature_names_in_ = feature_columns
        
        # Step 4: Train model
        logger.info("Step 4: Training model...")
        training_start = datetime.now()
        
        model.fit(X, y)
        
        training_end = datetime.now()
        training_duration = (training_end - training_start).total_seconds()
        
        logger.info(f"Training completed in {training_duration:.2f} seconds")
        
        # Step 5: Extract results
        logger.info("Step 5: Extracting results")
        
        if not hasattr(model, 'equations_') or len(model.equations_) == 0:
            return {
                "success": False,
                "job_id": job_id,
                "error": "No equations were discovered during training"
            }
        
        equations_df = model.equations_
        
        # Get all equations
        all_equations = []
        for idx, row in equations_df.iterrows():
            eq_info = {
                "index": int(idx),
                "equation": str(row.equation),
                "complexity": int(row.complexity),
                "loss": float(row.loss),
                "score": float(row.score)
            }
            all_equations.append(eq_info)
        
        # Get best equation
        best_eq = equations_df.iloc[-1]
        best_equation_info = {
            "equation": str(best_eq.equation),
            "complexity": int(best_eq.complexity),
            "loss": float(best_eq.loss),
            "score": float(best_eq.score)
        }
        
        # Step 6: Generate predictions and metrics
        logger.info("Step 6: Calculating metrics")
        
        predictions = model.predict(X)
        
        # Calculate metrics
        r2_score = 1 - np.sum((y - predictions) ** 2) / np.sum((y - np.mean(y)) ** 2)
        rmse = np.sqrt(np.mean((y - predictions) ** 2))
        mae = np.mean(np.abs(y - predictions))
        mape = np.mean(np.abs((y - predictions) / y)) * 100 if np.all(y != 0) else None
        
        metrics = {
            "r2_score": float(r2_score),
            "rmse": float(rmse),
            "mae": float(mae),
            "mape": float(mape) if mape is not None else None
        }
        
        # Step 7: Export equation in multiple formats
        logger.info("Step 7: Exporting equations")
        
        equation_exports = {}
        try:
            # SymPy format
            equation_exports["sympy"] = str(best_eq.equation)
            
            # LaTeX format
            try:
                sympy_eq = sympy.sympify(str(best_eq.equation))
                equation_exports["latex"] = sympy.latex(sympy_eq)
            except:
                equation_exports["latex"] = "LaTeX conversion not available"
                
        except Exception as e:
            logger.warning(f"Some equation exports failed: {e}")
        
        # Step 8: Store model and results
        model_id = f"model_{job_id}"
        models_storage[model_id] = model
        training_state[model_id] = {
            "status": "completed",
            "job_id": job_id,
            "created_at": training_start.isoformat(),
            "completed_at": training_end.isoformat(),
            "training_duration": training_duration
        }
        data_storage[model_id] = {
            "X": X,
            "y": y,
            "feature_names": feature_columns,
            "target_column": target_column
        }
        
        logger.info(f"Job {job_id} completed successfully!")
        
        # Return comprehensive results
        return {
            "success": True,
            "job_id": job_id,
            "model_id": model_id,
            "status": "completed",
            "training_info": {
                "duration_seconds": training_duration,
                "started_at": training_start.isoformat(),
                "completed_at": training_end.isoformat(),
                "data_shape": {
                    "samples": int(X.shape[0]),
                    "features": int(X.shape[1])
                },
                "feature_names": feature_columns,
                "target_column": target_column
            },
            "best_equation": best_equation_info,
            "all_equations": all_equations,
            "total_equations_found": len(all_equations),
            "metrics": metrics,
            "equation_exports": equation_exports,
            "predictions_sample": predictions[:10].tolist() if len(predictions) > 10 else predictions.tolist(),
            "configuration": {
                "niterations": niterations,
                "populations": populations,
                "population_size": population_size,
                "binary_operators": binary_operators,
                "unary_operators": unary_operators,
                "maxsize": maxsize,
                "parsimony": parsimony
            }
        }
        
    except Exception as e:
        logger.error(f"Error in complete training pipeline for job {job_id}: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "success": False,
            "job_id": job_id,
            "error": str(e),
            "traceback": traceback.format_exc()
        }


@mcp.tool()
async def create_model(
    model_id: str = Field(description="Unique identifier for the model"),
    config: ModelConfig = Field(default_factory=ModelConfig, description="Model configuration")
) -> Dict[str, Any]:
    """
    Create a new PySR model with specified configuration.
    
    Args:
        model_id: Unique identifier for the model
        config: Model configuration parameters
        
    Returns:
        Dictionary containing model creation status and configuration
    """
    try:
        logger.info(f"Creating model {model_id} with config: {config}")
        
        # Convert config to PySR parameters
        pysr_params = {
            "binary_operators": config.binary_operators,
            "unary_operators": config.unary_operators,
            "niterations": config.niterations,
            "populations": config.populations,
            "population_size": config.population_size,
            "maxsize": config.maxsize,
            "maxdepth": config.maxdepth,
            "parsimony": config.parsimony,
            "loss": config.loss,
            "model_selection": config.model_selection,
            "random_state": config.random_state,
            "progress": True,
            "verbosity": 1,
        }
        
        # Add optional parameters if provided
        if config.complexity_of_operators:
            pysr_params["complexity_of_operators"] = config.complexity_of_operators
        if config.constraints:
            pysr_params["constraints"] = config.constraints
        if config.nested_constraints:
            pysr_params["nested_constraints"] = config.nested_constraints
            
        # Create PySR model
        model = PySRRegressor(**pysr_params)
        
        # Store model and initialize training state
        models_storage[model_id] = model
        training_state[model_id] = {
            "status": "created",
            "created_at": datetime.now().isoformat(),
            "config": config.dict(),
            "iterations_completed": 0,
            "best_loss": None,
            "equations_found": 0
        }
        
        logger.info(f"Model {model_id} created successfully")
        
        return {
            "success": True,
            "model_id": model_id,
            "status": "created",
            "config": config.dict(),
            "message": "Model created successfully"
        }
        
    except Exception as e:
        logger.error(f"Error creating model {model_id}: {str(e)}")
        return {
            "success": False,
            "model_id": model_id,
            "error": str(e),
            "traceback": traceback.format_exc()
        }

@mcp.tool()
async def fit_model(
    model_id: str = Field(description="Model identifier"),
    data: TrainingData = Field(description="Training data"),
    resume_training: bool = Field(default=False, description="Resume from previous training")
) -> Dict[str, Any]:
    """
    Train the PySR model on provided data.
    
    Args:
        model_id: Model identifier
        data: Training data with features and targets
        resume_training: Whether to resume from previous training
        
    Returns:
        Training results and discovered equations
    """
    try:
        if model_id not in models_storage:
            return {
                "success": False,
                "error": f"Model {model_id} not found. Create model first."
            }
            
        model = models_storage[model_id]
        
        # Convert data to numpy arrays
        X = np.array(data.X)
        y = np.array(data.y)
        
        logger.info(f"Training model {model_id} with data shape: X{X.shape}, y{y.shape}")
        
        # Validate data
        if len(X) != len(y):
            return {
                "success": False,
                "error": "Feature matrix and target vector must have same number of samples"
            }
            
        # Store data for later use
        data_storage[model_id] = {
            "X": X,
            "y": y,
            "feature_names": data.feature_names,
            "variable_names": data.variable_names
        }
        
        # Update training state
        training_state[model_id]["status"] = "training"
        training_state[model_id]["training_started"] = datetime.now().isoformat()
        
        # Set feature names if provided
        if data.variable_names:
            model.feature_names_in_ = data.variable_names
            
        # Train the model
        model.fit(X, y)
        
        # Update training state
        training_state[model_id]["status"] = "completed"
        training_state[model_id]["training_completed"] = datetime.now().isoformat()
        training_state[model_id]["equations_found"] = len(model.equations_) if hasattr(model, 'equations_') else 0
        
        # Get best equation info
        if hasattr(model, 'equations_') and len(model.equations_) > 0:
            best_eq = model.equations_.iloc[-1]
            training_state[model_id]["best_loss"] = float(best_eq.loss)
            training_state[model_id]["best_equation"] = str(best_eq.equation)
            training_state[model_id]["best_complexity"] = int(best_eq.complexity)
        
        logger.info(f"Model {model_id} training completed successfully")
        
        return {
            "success": True,
            "model_id": model_id,
            "status": "completed",
            "equations_found": training_state[model_id]["equations_found"],
            "best_loss": training_state[model_id].get("best_loss"),
            "best_equation": training_state[model_id].get("best_equation"),
            "training_time": training_state[model_id].get("training_completed"),
            "message": "Model training completed successfully"
        }
        
    except Exception as e:
        logger.error(f"Error training model {model_id}: {str(e)}")
        training_state[model_id]["status"] = "failed"
        training_state[model_id]["error"] = str(e)
        return {
            "success": False,
            "model_id": model_id,
            "error": str(e),
            "traceback": traceback.format_exc()
        }

@mcp.tool()
async def predict(
    model_id: str = Field(description="Model identifier"),
    X: List[List[float]] = Field(description="Input features for prediction")
) -> Dict[str, Any]:
    """
    Generate predictions using trained model.
    
    Args:
        model_id: Model identifier
        X: Input features for prediction
        
    Returns:
        Predictions and confidence metrics
    """
    try:
        if model_id not in models_storage:
            return {
                "success": False,
                "error": f"Model {model_id} not found"
            }
            
        model = models_storage[model_id]
        
        if not hasattr(model, 'equations_') or len(model.equations_) == 0:
            return {
                "success": False,
                "error": f"Model {model_id} is not trained yet"
            }
            
        # Convert to numpy array
        X_pred = np.array(X)
        
        logger.info(f"Generating predictions for model {model_id} with input shape: {X_pred.shape}")
        
        # Generate predictions
        predictions = model.predict(X_pred)
        
        # Calculate additional metrics if training data is available
        additional_info = {}
        if model_id in data_storage:
            training_data = data_storage[model_id]
            X_train = training_data["X"]
            y_train = training_data["y"]
            
            # Calculate R2 score on training data
            train_predictions = model.predict(X_train)
            r2_score = 1 - np.sum((y_train - train_predictions) ** 2) / np.sum((y_train - np.mean(y_train)) ** 2)
            additional_info["train_r2_score"] = float(r2_score)
            additional_info["train_rmse"] = float(np.sqrt(np.mean((y_train - train_predictions) ** 2)))
        
        return {
            "success": True,
            "model_id": model_id,
            "predictions": predictions.tolist(),
            "num_predictions": len(predictions),
            "input_shape": X_pred.shape,
            **additional_info
        }
        
    except Exception as e:
        logger.error(f"Error generating predictions for model {model_id}: {str(e)}")
        return {
            "success": False,
            "model_id": model_id,
            "error": str(e),
            "traceback": traceback.format_exc()
        }

@mcp.tool()
async def get_equations(
    model_id: str = Field(description="Model identifier"),
    include_details: bool = Field(default=True, description="Include detailed equation information")
) -> Dict[str, Any]:
    """
    Retrieve discovered equations from trained model.
    
    Args:
        model_id: Model identifier
        include_details: Whether to include detailed equation metrics
        
    Returns:
        List of discovered equations with complexity and performance metrics
    """
    try:
        if model_id not in models_storage:
            return {
                "success": False,
                "error": f"Model {model_id} not found"
            }
            
        model = models_storage[model_id]
        
        if not hasattr(model, 'equations_') or len(model.equations_) == 0:
            return {
                "success": False,
                "error": f"Model {model_id} has no equations. Train the model first."
            }
            
        logger.info(f"Retrieving equations for model {model_id}")
        
        equations_df = model.equations_
        equations_list = []
        
        for idx, row in equations_df.iterrows():
            equation_info = {
                "index": int(idx),
                "equation": str(row.equation),
                "complexity": int(row.complexity),
                "loss": float(row.loss),
                "score": float(row.score)
            }
            
            if include_details:
                # Add additional columns if they exist
                for col in equations_df.columns:
                    if col not in ['equation', 'complexity', 'loss', 'score']:
                        equation_info[col] = row[col]
                        
            equations_list.append(equation_info)
        
        # Get best equation
        best_equation = equations_df.iloc[-1]
        
        return {
            "success": True,
            "model_id": model_id,
            "equations": equations_list,
            "num_equations": len(equations_list),
            "best_equation": {
                "equation": str(best_equation.equation),
                "complexity": int(best_equation.complexity),
                "loss": float(best_equation.loss),
                "score": float(best_equation.score)
            }
        }
        
    except Exception as e:
        logger.error(f"Error retrieving equations for model {model_id}: {str(e)}")
        return {
            "success": False,
            "model_id": model_id,
            "error": str(e),
            "traceback": traceback.format_exc()
        }

@mcp.tool()
async def save_model(
    model_id: str = Field(description="Model identifier"),
    filename: Optional[str] = Field(default=None, description="Optional filename for saved model")
) -> Dict[str, Any]:
    """
    Save trained model to disk.
    
    Args:
        model_id: Model identifier
        filename: Optional custom filename
        
    Returns:
        Save status and file location
    """
    try:
        if model_id not in models_storage:
            return {
                "success": False,
                "error": f"Model {model_id} not found"
            }
            
        model = models_storage[model_id]
        
        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"pysr_model_{model_id}_{timestamp}.pkl"
            
        filepath = MODELS_DIR / filename
        
        logger.info(f"Saving model {model_id} to {filepath}")
        
        # Save model using pickle
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': model,
                'training_state': training_state.get(model_id, {}),
                'data_info': {k: v for k, v in data_storage.get(model_id, {}).items() 
                             if k in ['feature_names', 'variable_names']}
            }, f)
        
        return {
            "success": True,
            "model_id": model_id,
            "filepath": str(filepath),
            "filename": filename,
            "file_size": filepath.stat().st_size,
            "saved_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error saving model {model_id}: {str(e)}")
        return {
            "success": False,
            "model_id": model_id,
            "error": str(e),
            "traceback": traceback.format_exc()
        }

@mcp.tool()
async def load_model(
    model_id: str = Field(description="Model identifier for loaded model"),
    filepath: str = Field(description="Path to saved model file")
) -> Dict[str, Any]:
    """
    Load previously saved model from disk.
    
    Args:
        model_id: New identifier for loaded model
        filepath: Path to saved model file
        
    Returns:
        Load status and model information
    """
    try:
        load_path = Path(filepath)
        
        if not load_path.exists():
            # Try relative to models directory
            load_path = MODELS_DIR / filepath
            
        if not load_path.exists():
            return {
                "success": False,
                "error": f"Model file not found: {filepath}"
            }
            
        logger.info(f"Loading model from {load_path} as {model_id}")
        
        # Load model
        with open(load_path, 'rb') as f:
            saved_data = pickle.load(f)
            
        model = saved_data['model']
        saved_training_state = saved_data.get('training_state', {})
        saved_data_info = saved_data.get('data_info', {})
        
        # Store loaded model
        models_storage[model_id] = model
        
        # Restore training state with loaded timestamp
        training_state[model_id] = {
            **saved_training_state,
            "loaded_at": datetime.now().isoformat(),
            "loaded_from": str(load_path)
        }
        
        # Get model info
        model_info = {
            "has_equations": hasattr(model, 'equations_') and len(model.equations_) > 0,
            "equations_count": len(model.equations_) if hasattr(model, 'equations_') else 0
        }
        
        if model_info["has_equations"]:
            best_eq = model.equations_.iloc[-1]
            model_info.update({
                "best_equation": str(best_eq.equation),
                "best_loss": float(best_eq.loss),
                "best_complexity": int(best_eq.complexity)
            })
        
        return {
            "success": True,
            "model_id": model_id,
            "filepath": str(load_path),
            "loaded_at": datetime.now().isoformat(),
            "model_info": model_info,
            "training_state": training_state[model_id]
        }
        
    except Exception as e:
        logger.error(f"Error loading model from {filepath}: {str(e)}")
        return {
            "success": False,
            "model_id": model_id,
            "filepath": filepath,
            "error": str(e),
            "traceback": traceback.format_exc()
        }

@mcp.tool()
async def export_equation(
    model_id: str = Field(description="Model identifier"),
    format: str = Field(default="sympy", description="Export format: sympy, latex, julia, jax, torch"),
    equation_index: Optional[int] = Field(default=None, description="Equation index (None for best)")
) -> Dict[str, Any]:
    """
    Export equation in specified format.
    
    Args:
        model_id: Model identifier
        format: Export format
        equation_index: Specific equation index or None for best
        
    Returns:
        Exported equation in requested format
    """
    try:
        if model_id not in models_storage:
            return {
                "success": False,
                "error": f"Model {model_id} not found"
            }
            
        model = models_storage[model_id]
        
        if not hasattr(model, 'equations_') or len(model.equations_) == 0:
            return {
                "success": False,
                "error": f"Model {model_id} has no equations"
            }
            
        # Get equation
        if equation_index is None:
            equation_row = model.equations_.iloc[-1]
            equation_index = len(model.equations_) - 1
        else:
            if equation_index >= len(model.equations_):
                return {
                    "success": False,
                    "error": f"Equation index {equation_index} out of range"
                }
            equation_row = model.equations_.iloc[equation_index]
        
        logger.info(f"Exporting equation {equation_index} from model {model_id} in {format} format")
        
        equation = equation_row.equation
        exported_equation = None
        
        try:
            if format.lower() == "sympy":
                exported_equation = str(equation)
            elif format.lower() == "latex":
                sympy_eq = sympy.sympify(str(equation))
                exported_equation = sympy.latex(sympy_eq)
            elif format.lower() == "julia":
                exported_equation = model.julia_format_ if hasattr(model, 'julia_format_') else str(equation)
            elif format.lower() == "jax":
                if hasattr(model, 'jax_format_'):
                    exported_equation = model.jax_format_
                else:
                    exported_equation = "JAX format not available"
            elif format.lower() == "torch":
                if hasattr(model, 'torch_format_'):
                    exported_equation = str(model.torch_format_)
                else:
                    exported_equation = "Torch format not available"
            else:
                return {
                    "success": False,
                    "error": f"Unsupported format: {format}. Supported: sympy, latex, julia, jax, torch"
                }
                
        except Exception as format_error:
            logger.warning(f"Format conversion error: {format_error}")
            exported_equation = str(equation)
            
        return {
            "success": True,
            "model_id": model_id,
            "equation_index": equation_index,
            "format": format,
            "original_equation": str(equation),
            "exported_equation": exported_equation,
            "complexity": int(equation_row.complexity),
            "loss": float(equation_row.loss),
            "score": float(equation_row.score)
        }
        
    except Exception as e:
        logger.error(f"Error exporting equation for model {model_id}: {str(e)}")
        return {
            "success": False,
            "model_id": model_id,
            "error": str(e),
            "traceback": traceback.format_exc()
        }

@mcp.tool()
async def validate_data(
    data: TrainingData = Field(description="Data to validate")
) -> Dict[str, Any]:
    """
    Validate input data format and quality.
    
    Args:
        data: Training data to validate
        
    Returns:
        Validation results and data quality metrics
    """
    try:
        logger.info("Validating input data")
        
        X = np.array(data.X)
        y = np.array(data.y)
        
        validation_results = {
            "success": True,
            "data_shape": {
                "samples": X.shape[0],
                "features": X.shape[1] if len(X.shape) > 1 else 1
            },
            "target_shape": y.shape,
            "issues": [],
            "warnings": [],
            "recommendations": []
        }
        
        # Basic shape validation
        if len(X) != len(y):
            validation_results["issues"].append("Feature matrix and target vector have different lengths")
            validation_results["success"] = False
            
        # Check for missing values
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            validation_results["issues"].append("Feature matrix contains NaN or infinite values")
            validation_results["success"] = False
            
        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            validation_results["issues"].append("Target vector contains NaN or infinite values")
            validation_results["success"] = False
            
        # Data quality checks
        if len(X) < 10:
            validation_results["warnings"].append("Very small dataset (< 10 samples). Consider more data.")
            
        if len(X) > 10000:
            validation_results["recommendations"].append("Large dataset detected. Consider using parallelization.")
            
        # Feature statistics
        feature_stats = []
        for i in range(X.shape[1] if len(X.shape) > 1 else 1):
            if len(X.shape) > 1:
                feature_col = X[:, i]
            else:
                feature_col = X
                
            stats = {
                "feature_index": i,
                "mean": float(np.mean(feature_col)),
                "std": float(np.std(feature_col)),
                "min": float(np.min(feature_col)),
                "max": float(np.max(feature_col)),
                "unique_values": int(len(np.unique(feature_col)))
            }
            
            # Check for constant features
            if stats["std"] < 1e-10:
                validation_results["warnings"].append(f"Feature {i} appears to be constant")
                
            feature_stats.append(stats)
            
        validation_results["feature_statistics"] = feature_stats
        
        # Target statistics
        validation_results["target_statistics"] = {
            "mean": float(np.mean(y)),
            "std": float(np.std(y)),
            "min": float(np.min(y)),
            "max": float(np.max(y)),
            "unique_values": int(len(np.unique(y)))
        }
        
        # Check target variance
        if np.std(y) < 1e-10:
            validation_results["issues"].append("Target variable appears to be constant")
            validation_results["success"] = False
            
        return validation_results
        
    except Exception as e:
        logger.error(f"Error validating data: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }

@mcp.tool()
async def list_models() -> Dict[str, Any]:
    """
    List all available models and their status.
    
    Returns:
        List of all models with their current status
    """
    try:
        models_info = []
        
        for model_id in models_storage.keys():
            model = models_storage[model_id]
            state = training_state.get(model_id, {})
            
            model_info = {
                "model_id": model_id,
                "status": state.get("status", "unknown"),
                "created_at": state.get("created_at"),
                "has_equations": hasattr(model, 'equations_') and len(model.equations_) > 0,
                "equations_count": len(model.equations_) if hasattr(model, 'equations_') else 0
            }
            
            if model_info["has_equations"]:
                best_eq = model.equations_.iloc[-1]
                model_info.update({
                    "best_loss": float(best_eq.loss),
                    "best_complexity": int(best_eq.complexity)
                })
                
            models_info.append(model_info)
            
        return {
            "success": True,
            "models": models_info,
            "total_models": len(models_info)
        }
        
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }

@mcp.tool()
async def delete_model(
    model_id: str = Field(description="Model identifier to delete")
) -> Dict[str, Any]:
    """
    Delete model from memory.
    
    Args:
        model_id: Model identifier to delete
        
    Returns:
        Deletion status
    """
    try:
        deleted_items = []
        
        if model_id in models_storage:
            del models_storage[model_id]
            deleted_items.append("model")
            
        if model_id in training_state:
            del training_state[model_id]
            deleted_items.append("training_state")
            
        if model_id in data_storage:
            del data_storage[model_id]
            deleted_items.append("data_storage")
            
        if not deleted_items:
            return {
                "success": False,
                "error": f"Model {model_id} not found"
            }
            
        logger.info(f"Deleted model {model_id} and associated data")
        
        return {
            "success": True,
            "model_id": model_id,
            "deleted_items": deleted_items,
            "deleted_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error deleting model {model_id}: {str(e)}")
        return {
            "success": False,
            "model_id": model_id,
            "error": str(e),
            "traceback": traceback.format_exc()
        }

@mcp.tool()
async def health_check() -> Dict[str, Any]:
    """
    Check server health and status.
    
    Returns:
        Server health information
    """
    try:
        import psutil
        import platform
        
        # System information
        system_info = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "memory_available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
            "cpu_percent": psutil.cpu_percent(interval=1)
        }
        
        # Server status
        server_status = {
            "active_models": len(models_storage),
            "models_with_equations": sum(1 for model in models_storage.values() 
                                       if hasattr(model, 'equations_') and len(model.equations_) > 0),
            "total_equations": sum(len(model.equations_) for model in models_storage.values() 
                                 if hasattr(model, 'equations_')),
            "storage_directories": {
                "models_dir_exists": MODELS_DIR.exists(),
                "data_dir_exists": DATA_DIR.exists(),
                "results_dir_exists": RESULTS_DIR.exists()
            }
        }
        
        return {
            "success": True,
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "system_info": system_info,
            "server_status": server_status,
            "pysr_version": pysr.__version__ if hasattr(pysr, '__version__') else "unknown"
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "success": False,
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

# Resources for providing server information

@mcp.resource("server://info")
async def server_info() -> str:
    """Provides general information about the PySR MCP server."""
    return json.dumps({
        "name": "PySR Symbolic Regression MCP Server",
        "version": "1.0.0",
        "transport": "HTTP",
        "description": "FastMCP server providing symbolic regression capabilities using PySR",
        "tools_available": 12,
        "capabilities": [
            "Complete end-to-end training pipeline",
            "Model creation and configuration",
            "Training with custom operators",
            "Prediction generation", 
            "Equation discovery and export",
            "Model persistence and loading",
            "Data validation and monitoring",
            "Comprehensive model evaluation"
        ],
        "supported_formats": ["sympy", "latex", "julia", "jax", "torch"],
        "storage_directories": {
            "models": str(MODELS_DIR),
            "data": str(DATA_DIR),
            "results": str(RESULTS_DIR)
        }
    }, indent=2)

@mcp.resource("server://status")
async def server_status() -> str:
    """Provides current server status and statistics."""
    status_info = {
        "timestamp": datetime.now().isoformat(),
        "transport": "HTTP",
        "active_models": len(models_storage),
        "training_states": len(training_state),
        "data_storage_entries": len(data_storage),
        "model_statuses": {}
    }
    
    for model_id, state in training_state.items():
        status_info["model_statuses"][model_id] = state.get("status", "unknown")
        
    return json.dumps(status_info, indent=2)

if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='PySR MCP Server with HTTP Transport')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind to (default: 8000)')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    
    logger.info("="*60)
    logger.info("PySR MCP Server with HTTP Transport")
    logger.info("="*60)
    logger.info(f"Models directory: {MODELS_DIR}")
    logger.info(f"Data directory: {DATA_DIR}")
    logger.info(f"Results directory: {RESULTS_DIR}")
    logger.info(f"Transport: HTTP")
    logger.info(f"Server URL: http://{args.host}:{args.port}")
    logger.info(f"Health endpoint: http://{args.host}:{args.port}/health")
    logger.info(f"Server info: http://{args.host}:{args.port}/server/info")
    logger.info("="*60)
    
    # Run with HTTP transport
    mcp.run(
        transport="http",
        host=args.host,
        port=args.port
    )

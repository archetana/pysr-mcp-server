#!/usr/bin/env python3
"""
Streamlit Frontend for PySR System
Beautiful and intuitive interface for symbolic regression
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import os
from pathlib import Path
import io
import base64

# Page configuration
st.set_page_config(
    page_title="PySR Symbolic Regression",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
REFRESH_INTERVAL = 5  # seconds

# Custom CSS for beautiful styling
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    
    .stAlert > div {
        padding: 0.5rem 1rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .job-card {
        border: 1px solid #e1e4e8;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        background: white;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    .equation-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        font-family: 'Courier New', monospace;
    }
    
    .status-running {
        color: #ffa500;
        font-weight: bold;
    }
    
    .status-completed {
        color: #28a745;
        font-weight: bold;
    }
    
    .status-failed {
        color: #dc3545;
        font-weight: bold;
    }
    
    .sidebar .stSelectbox > div > div {
        background-color: #f8f9fa;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #4CAF50;
        color: white;
    }
    
    .uploadedFile {
        border: 2px dashed #cccccc;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background-color: #f9f9f9;
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'token' not in st.session_state:
    st.session_state.token = None
if 'user' not in st.session_state:
    st.session_state.user = None
if 'jobs' not in st.session_state:
    st.session_state.jobs = []
if 'selected_job' not in st.session_state:
    st.session_state.selected_job = None

# Utility functions
def make_request(method: str, endpoint: str, data: Dict = None, files: Dict = None) -> Dict:
    """Make authenticated API request"""
    headers = {}
    if st.session_state.token:
        headers['Authorization'] = f"Bearer {st.session_state.token}"
    
    url = f"{API_BASE_URL}{endpoint}"
    
    try:
        if method.upper() == 'GET':
            response = requests.get(url, headers=headers)
        elif method.upper() == 'POST':
            if files:
                response = requests.post(url, headers=headers, data=data, files=files)
            else:
                headers['Content-Type'] = 'application/json'
                response = requests.post(url, headers=headers, json=data)
        elif method.upper() == 'PUT':
            headers['Content-Type'] = 'application/json'
            response = requests.put(url, headers=headers, json=data)
        elif method.upper() == 'DELETE':
            response = requests.delete(url, headers=headers)
        else:
            return {'success': False, 'error': 'Unsupported method'}
        
        if response.status_code == 200:
            return {'success': True, 'data': response.json()}
        elif response.status_code == 401:
            st.session_state.logged_in = False
            st.session_state.token = None
            st.session_state.user = None
            return {'success': False, 'error': 'Authentication required'}
        else:
            return {'success': False, 'error': response.text}
            
    except requests.exceptions.RequestException as e:
        return {'success': False, 'error': f'Connection error: {str(e)}'}

def format_datetime(dt_str: str) -> str:
    """Format datetime string for display"""
    if not dt_str:
        return "N/A"
    try:
        dt = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return dt_str

def get_status_color(status: str) -> str:
    """Get color for status display"""
    colors = {
        'created': '#6c757d',
        'running': '#ffa500',
        'completed': '#28a745',
        'failed': '#dc3545',
        'pending': '#17a2b8'
    }
    return colors.get(status, '#6c757d')

def create_sample_data() -> pd.DataFrame:
    """Create sample dataset for demonstration"""
    np.random.seed(42)
    n_samples = 100
    
    # Create features
    x0 = np.random.uniform(-3, 3, n_samples)
    x1 = np.random.uniform(-2, 2, n_samples)
    x2 = np.random.uniform(-1, 1, n_samples)
    x3 = np.random.uniform(0, 2*np.pi, n_samples)
    
    # Create target: y = 2.5 * cos(x3) + x0^2 - 0.5 + noise
    y = 2.5 * np.cos(x3) + x0**2 - 0.5 + np.random.normal(0, 0.1, n_samples)
    
    df = pd.DataFrame({
        'x0': x0,
        'x1': x1, 
        'x2': x2,
        'x3': x3,
        'y': y
    })
    
    return df

def create_engineering_sample() -> pd.DataFrame:
    """Create engineering sample dataset"""
    np.random.seed(123)
    n_samples = 100
    
    x0 = np.random.uniform(0.1, 5, n_samples)
    x1 = np.random.uniform(0, 10, n_samples)
    x2 = np.random.uniform(-2, 2, n_samples)
    x3 = np.random.uniform(0.5, 3, n_samples)
    
    # y = sqrt(x0) * log(x1+1) + x2/x3 + noise
    y = np.sqrt(x0) * np.log(x1 + 1) + x2/x3 + np.random.normal(0, 0.1, n_samples)
    
    df = pd.DataFrame({
        'x0': x0,
        'x1': x1,
        'x2': x2,
        'x3': x3,
        'y': y
    })
    
    return df

# Authentication functions
def login_page():
    """Display login page"""
    st.title("🧬 PySR Symbolic Regression")
    st.markdown("### Discover mathematical relationships in your data")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("---")
        
        tab1, tab2 = st.tabs(["Login", "Register"])
        
        with tab1:
            st.subheader("Login")
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Login", use_container_width=True)
                
                if submitted:
                    if username and password:
                        response = make_request('POST', '/auth/login', {
                            'username': username,
                            'password': password
                        })
                        
                        if response['success']:
                            data = response['data']
                            st.session_state.token = data['access_token']
                            st.session_state.logged_in = True
                            
                            # Get user info
                            user_response = make_request('GET', '/auth/me')
                            if user_response['success']:
                                st.session_state.user = user_response['data']
                                st.success("Login successful!")
                                st.rerun()
                            else:
                                st.error("Failed to get user info")
                        else:
                            st.error(f"Login failed: {response['error']}")
                    else:
                        st.error("Please enter username and password")
        
        with tab2:
            st.subheader("Register")
            with st.form("register_form"):
                reg_username = st.text_input("Username", key="reg_username")
                reg_email = st.text_input("Email", key="reg_email")
                reg_fullname = st.text_input("Full Name (optional)", key="reg_fullname")
                reg_password = st.text_input("Password", type="password", key="reg_password")
                reg_confirm = st.text_input("Confirm Password", type="password", key="reg_confirm")
                
                reg_submitted = st.form_submit_button("Register", use_container_width=True)
                
                if reg_submitted:
                    if not all([reg_username, reg_email, reg_password, reg_confirm]):
                        st.error("Please fill in all required fields")
                    elif reg_password != reg_confirm:
                        st.error("Passwords do not match")
                    elif len(reg_password) < 6:
                        st.error("Password must be at least 6 characters long")
                    else:
                        response = make_request('POST', '/auth/register', {
                            'username': reg_username,
                            'email': reg_email,
                            'full_name': reg_fullname if reg_fullname else None,
                            'password': reg_password
                        })
                        
                        if response['success']:
                            st.success("Registration successful! Please login.")
                        else:
                            st.error(f"Registration failed: {response['error']}")

def dashboard():
    """Main dashboard"""
    # Header
    st.title("🧬 PySR Dashboard")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown(f"**Welcome, {st.session_state.user['full_name'] or st.session_state.user['username']}!**")
    with col2:
        if st.button("🔄 Refresh", use_container_width=True):
            load_jobs()
    with col3:
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.token = None
            st.session_state.user = None
            st.session_state.jobs = []
            st.rerun()
    
    st.markdown("---")
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Jobs", "🆕 New Job", "📈 Results", "ℹ️ Help"])
    
    with tab1:
        jobs_tab()
    
    with tab2:
        new_job_tab()
    
    with tab3:
        results_tab()
    
    with tab4:
        help_tab()

def load_jobs():
    """Load user jobs"""
    response = make_request('GET', '/jobs')
    if response['success']:
        st.session_state.jobs = response['data']
    else:
        st.error(f"Failed to load jobs: {response['error']}")

def jobs_tab():
    """Jobs management tab"""
    st.subheader("📊 Job Management")
    
    # Load jobs
    if st.button("🔄 Refresh Jobs", key="refresh_jobs"):
        load_jobs()
    
    if not st.session_state.jobs:
        load_jobs()
    
    if st.session_state.jobs:
        # Job statistics
        total_jobs = len(st.session_state.jobs)
        completed_jobs = len([j for j in st.session_state.jobs if j['status'] == 'completed'])
        running_jobs = len([j for j in st.session_state.jobs if j['status'] == 'running'])
        failed_jobs = len([j for j in st.session_state.jobs if j['status'] == 'failed'])
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Jobs", total_jobs)
        with col2:
            st.metric("Completed", completed_jobs, delta=None)
        with col3:
            st.metric("Running", running_jobs, delta=None)
        with col4:
            st.metric("Failed", failed_jobs, delta=None)
        
        st.markdown("---")
        
        # Jobs table
        for job in st.session_state.jobs:
            with st.container():
                col1, col2, col3, col4, col5 = st.columns([3, 2, 2, 1, 1])
                
                with col1:
                    st.markdown(f"**{job['name']}**")
                    if job['description']:
                        st.caption(job['description'])
                
                with col2:
                    status_color = get_status_color(job['status'])
                    st.markdown(f"<span style='color: {status_color}'>{job['status'].upper()}</span>", 
                              unsafe_allow_html=True)
                
                with col3:
                    st.caption(f"Created: {format_datetime(job['created_at'])}")
                    if job['completed_at']:
                        st.caption(f"Completed: {format_datetime(job['completed_at'])}")
                
                with col4:
                    if st.button("👁️", key=f"view_{job['id']}", help="View Details"):
                        st.session_state.selected_job = job['id']
                        view_job_details(job)
                
                with col5:
                    if st.button("🗑️", key=f"delete_{job['id']}", help="Delete Job"):
                        if st.session_state.get(f"confirm_delete_{job['id']}", False):
                            delete_job(job['id'])
                        else:
                            st.session_state[f"confirm_delete_{job['id']}"] = True
                            st.warning("Click again to confirm deletion")
                
                st.markdown("---")
    else:
        st.info("No jobs found. Create your first job in the 'New Job' tab!")

def view_job_details(job):
    """View detailed job information"""
    st.subheader(f"Job Details: {job['name']}")
    
    # Job info
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Basic Information**")
        st.write(f"**ID:** {job['id']}")
        st.write(f"**Name:** {job['name']}")
        st.write(f"**Status:** {job['status']}")
        st.write(f"**Created:** {format_datetime(job['created_at'])}")
        if job['completed_at']:
            st.write(f"**Completed:** {format_datetime(job['completed_at'])}")
    
    with col2:
        st.markdown("**Configuration**")
        if job['config']:
            st.json(job['config'])
        else:
            st.write("No configuration set")
    
    # Actions
    st.markdown("**Actions**")
    action_col1, action_col2, action_col3 = st.columns(3)
    
    with action_col1:
        if job['status'] not in ['running'] and job.get('file_path'):
            if st.button(f"🚀 Run Job {job['id']}", use_container_width=True):
                run_job(job['id'])
    
    with action_col2:
        if job.get('file_path'):
            if st.button(f"📥 Download Data", use_container_width=True):
                download_job_file(job['id'])
    
    with action_col3:
        if st.button(f"📊 View Results", use_container_width=True):
            view_job_results(job['id'])

def run_job(job_id: int):
    """Run a job"""
    response = make_request('POST', f'/jobs/{job_id}/run')
    if response['success']:
        st.success("Job started successfully!")
        load_jobs()
    else:
        st.error(f"Failed to start job: {response['error']}")

def download_job_file(job_id: int):
    """Download job data file"""
    # This would need to be implemented with proper file download
    st.info("File download functionality would be implemented here")

def delete_job(job_id: int):
    """Delete a job"""
    response = make_request('DELETE', f'/jobs/{job_id}')
    if response['success']:
        st.success("Job deleted successfully!")
        load_jobs()
        # Clear confirmation state
        if f"confirm_delete_{job_id}" in st.session_state:
            del st.session_state[f"confirm_delete_{job_id}"]
    else:
        st.error(f"Failed to delete job: {response['error']}")

def new_job_tab():
    """New job creation tab"""
    st.subheader("🆕 Create New Job")
    
    with st.form("new_job_form"):
        # Job details
        st.markdown("**Job Details**")
        job_name = st.text_input("Job Name*", placeholder="Enter a descriptive name for your job")
        job_description = st.text_area("Description", placeholder="Optional description of your job")
        
        st.markdown("---")
        
        # Data upload
        st.markdown("**Data Upload**")
        
        # Sample data option
        col1, col2 = st.columns([1, 1])
        with col1:
            use_sample = st.checkbox("Use sample dataset", help="Use pre-generated sample data for testing")
        
        if use_sample:
            sample_type = st.selectbox("Sample Dataset", [
                "Physics-inspired (y = 2.5*cos(x3) + x0² - 0.5)",
                "Engineering (y = √x0 * log(x1+1) + x2/x3)"
            ])
            
            if sample_type.startswith("Physics"):
                df_sample = create_sample_data()
                st.write("**Sample data preview:**")
                st.dataframe(df_sample.head(), use_container_width=True)
                
                # Create download link for sample data
                csv_buffer = io.StringIO()
                df_sample.to_csv(csv_buffer, index=False)
                csv_data = csv_buffer.getvalue()
                
                st.download_button(
                    label="📥 Download Sample CSV",
                    data=csv_data,
                    file_name="sample_physics_data.csv",
                    mime="text/csv"
                )
            else:
                df_sample = create_engineering_sample()
                st.write("**Sample data preview:**")
                st.dataframe(df_sample.head(), use_container_width=True)
                
                csv_buffer = io.StringIO()
                df_sample.to_csv(csv_buffer, index=False)
                csv_data = csv_buffer.getvalue()
                
                st.download_button(
                    label="📥 Download Sample CSV",
                    data=csv_data,
                    file_name="sample_engineering_data.csv",
                    mime="text/csv"
                )
        else:
            uploaded_file = st.file_uploader(
                "Upload CSV File*",
                type=['csv'],
                help="Upload a CSV file where the last column is the target variable"
            )
            
            if uploaded_file:
                # Preview uploaded data
                try:
                    df = pd.read_csv(uploaded_file)
                    st.write("**Data preview:**")
                    st.dataframe(df.head(), use_container_width=True)
                    
                    st.write(f"**Data shape:** {df.shape[0]} rows × {df.shape[1]} columns")
                    
                    # Data validation
                    if df.shape[1] < 2:
                        st.error("CSV must have at least 2 columns (features + target)")
                    elif df.isnull().sum().sum() > 0:
                        st.warning("Dataset contains missing values. Consider cleaning the data.")
                    else:
                        st.success("Data validation passed!")
                        
                except Exception as e:
                    st.error(f"Error reading CSV file: {str(e)}")
        
        st.markdown("---")
        
        # PySR Configuration
        st.markdown("**PySR Configuration**")
        
        config_col1, config_col2 = st.columns(2)
        
        with config_col1:
            niterations = st.number_input("Iterations", min_value=1, max_value=200, value=40,
                                        help="Number of iterations to run")
            populations = st.number_input("Populations", min_value=1, max_value=20, value=8,
                                        help="Number of populations")
            population_size = st.number_input("Population Size", min_value=10, max_value=100, value=33,
                                            help="Size of each population")
        
        with config_col2:
            maxsize = st.number_input("Max Complexity", min_value=5, max_value=50, value=20,
                                    help="Maximum equation complexity")
            maxdepth = st.number_input("Max Depth", min_value=3, max_value=20, value=10,
                                     help="Maximum equation depth")
            parsimony = st.number_input("Parsimony", min_value=0.0, max_value=1.0, value=0.0032, step=0.001,
                                      help="Parsimony coefficient for complexity control")
        
        # Operators
        st.markdown("**Mathematical Operators**")
        
        op_col1, op_col2 = st.columns(2)
        
        with op_col1:
            st.markdown("**Binary Operators**")
            binary_ops = st.multiselect(
                "Select binary operators",
                options=["+", "-", "*", "/", "^", "max", "min"],
                default=["+", "-", "*", "/"],
                help="Operators that combine two terms"
            )
        
        with op_col2:
            st.markdown("**Unary Operators**")
            unary_ops = st.multiselect(
                "Select unary operators",
                options=["sin", "cos", "tan", "exp", "log", "sqrt", "abs", "sign"],
                default=["sin", "cos", "exp", "log", "sqrt"],
                help="Operators that modify a single term"
            )
        
        # Advanced options
        with st.expander("🔧 Advanced Options"):
            model_selection = st.selectbox(
                "Model Selection",
                options=["best", "accuracy", "score"],
                index=0,
                help="Criteria for selecting the best model"
            )
            
            loss_function = st.selectbox(
                "Loss Function",
                options=["L2DistLoss()", "L1DistLoss()", "LogitDistLoss()"],
                index=0,
                help="Loss function for optimization"
            )
            
            random_state = st.number_input(
                "Random Seed",
                min_value=0,
                max_value=9999,
                value=42,
                help="Random seed for reproducibility (0 for random)"
            )
        
        # Submit button
        submitted = st.form_submit_button("🚀 Create and Run Job", use_container_width=True)
        
        if submitted:
            # Validation
            if not job_name:
                st.error("Job name is required")
            elif not binary_ops:
                st.error("At least one binary operator must be selected")
            elif not use_sample and not uploaded_file:
                st.error("Please upload a CSV file or use sample data")
            else:
                # Create job configuration
                config = {
                    "model_config": {
                        "binary_operators": binary_ops,
                        "unary_operators": unary_ops,
                        "niterations": niterations,
                        "populations": populations,
                        "population_size": population_size,
                        "maxsize": maxsize,
                        "maxdepth": maxdepth,
                        "parsimony": parsimony,
                        "model_selection": model_selection,
                        "loss": loss_function,
                        "random_state": random_state if random_state > 0 else None
                    }
                }
                
                # Create job
                job_data = {
                    "name": job_name,
                    "description": job_description,
                    "config": config
                }
                
                response = make_request('POST', '/jobs', job_data)
                
                if response['success']:
                    job = response['data']
                    job_id = job['id']
                    
                    # Upload file
                    if use_sample:
                        # For sample data, we need to create a temporary file
                        if sample_type.startswith("Physics"):
                            df_sample = create_sample_data()
                        else:
                            df_sample = create_engineering_sample()
                            
                        csv_data = df_sample.to_csv(index=False).encode('utf-8')
                        
                        files = {
                            'file': ('sample_data.csv', csv_data, 'text/csv')
                        }
                        
                        upload_response = make_request('POST', f'/jobs/{job_id}/upload', files=files)
                    else:
                        # Upload the actual file
                        files = {
                            'file': (uploaded_file.name, uploaded_file.getvalue(), 'text/csv')
                        }
                        upload_response = make_request('POST', f'/jobs/{job_id}/upload', files=files)
                    
                    if upload_response['success']:
                        # Run the job
                        run_response = make_request('POST', f'/jobs/{job_id}/run')
                        
                        if run_response['success']:
                            st.success(f"Job '{job_name}' created and started successfully!")
                            st.info("You can monitor progress in the Jobs tab.")
                            load_jobs()
                        else:
                            st.error(f"Job created but failed to start: {run_response['error']}")
                    else:
                        st.error(f"Job created but file upload failed: {upload_response['error']}")
                else:
                    st.error(f"Failed to create job: {response['error']}")

def results_tab():
    """Results visualization tab"""
    st.subheader("📈 Results & Analysis")
    
    # Job selection
    if st.session_state.jobs:
        completed_jobs = [j for j in st.session_state.jobs if j['status'] == 'completed']
        
        if completed_jobs:
            job_options = {f"{job['name']} (ID: {job['id']})": job['id'] for job in completed_jobs}
            selected_job_name = st.selectbox("Select completed job to view results", list(job_options.keys()))
            
            if selected_job_name:
                job_id = job_options[selected_job_name]
                view_job_results(job_id)
        else:
            st.info("No completed jobs found. Run some jobs to see results here!")
    else:
        st.info("No jobs found. Create and run jobs to see results here!")

def view_job_results(job_id: int):
    """View detailed job results"""
    # Get job runs
    response = make_request('GET', f'/jobs/{job_id}/runs')
    if not response['success']:
        st.error(f"Failed to load job runs: {response['error']}")
        return
    
    runs = response['data']
    completed_runs = [r for r in runs if r['status'] == 'completed' and r.get('results')]
    
    if not completed_runs:
        st.warning("No completed runs with results found for this job.")
        return
    
    # Use the latest completed run
    latest_run = completed_runs[0]
    results = latest_run['results']
    
    st.markdown(f"**Results for Run ID:** `{latest_run['run_id']}`")
    st.markdown(f"**Completed:** {format_datetime(latest_run['completed_at'])}")
    
    # Best equation display
    if results.get('best_equation'):
        st.markdown("### 🏆 Best Equation")
        best_eq = results['best_equation']
        
        st.markdown(f"""
        <div class="equation-card">
            <h4>📐 {best_eq['equation']}</h4>
            <p><strong>Complexity:</strong> {best_eq['complexity']} | <strong>Loss:</strong> {best_eq['loss']:.6f} | <strong>Score:</strong> {best_eq['score']:.6f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Equations table
    if results.get('equations'):
        st.markdown("### 📊 All Discovered Equations")
        
        equations_df = pd.DataFrame(results['equations'])
        
        # Format the dataframe for better display
        display_df = equations_df.copy()
        display_df['loss'] = display_df['loss'].apply(lambda x: f"{x:.6f}")
        display_df['score'] = display_df['score'].apply(lambda x: f"{x:.6f}")
        
        st.dataframe(display_df, use_container_width=True)
        
        # Visualizations
        st.markdown("### 📈 Performance Analysis")
        
        viz_col1, viz_col2 = st.columns(2)
        
        with viz_col1:
            # Complexity vs Loss plot
            fig1 = px.scatter(
                equations_df,
                x='complexity',
                y='loss',
                title='Complexity vs Loss',
                hover_data=['equation'],
                color='score',
                color_continuous_scale='viridis'
            )
            fig1.update_layout(height=400)
            st.plotly_chart(fig1, use_container_width=True)
        
        with viz_col2:
            # Score vs Complexity plot
            fig2 = px.scatter(
                equations_df,
                x='complexity',
                y='score',
                title='Score vs Complexity',
                hover_data=['equation'],
                color='loss',
                color_continuous_scale='viridis_r'
            )
            fig2.update_layout(height=400)
            st.plotly_chart(fig2, use_container_width=True)
        
        # Pareto front visualization
        st.markdown("### 🎯 Pareto Front Analysis")
        
        # Calculate Pareto front
        pareto_equations = calculate_pareto_front(equations_df)
        
        fig3 = go.Figure()
        
        # All equations
        fig3.add_trace(go.Scatter(
            x=equations_df['complexity'],
            y=equations_df['loss'],
            mode='markers',
            name='All Equations',
            marker=dict(color='lightblue', size=8),
            hovertemplate='<b>%{hovertext}</b><br>Complexity: %{x}<br>Loss: %{y}<extra></extra>',
            hovertext=equations_df['equation']
        ))
        
        # Pareto front
        if len(pareto_equations) > 0:
            fig3.add_trace(go.Scatter(
                x=pareto_equations['complexity'],
                y=pareto_equations['loss'],
                mode='markers+lines',
                name='Pareto Front',
                marker=dict(color='red', size=12),
                line=dict(color='red', dash='dash'),
                hovertemplate='<b>%{hovertext}</b><br>Complexity: %{x}<br>Loss: %{y}<extra></extra>',
                hovertext=pareto_equations['equation']
            ))
        
        fig3.update_layout(
            title='Pareto Front: Complexity vs Loss',
            xaxis_title='Complexity',
            yaxis_title='Loss',
            height=500
        )
        
        st.plotly_chart(fig3, use_container_width=True)
        
        # Model predictions visualization
        if results.get('predictions') and results.get('actual'):
            st.markdown("### 🎯 Model Predictions")
            
            predictions = results['predictions']
            actual = results['actual']
            
            # Predictions vs Actual scatter plot
            fig4 = go.Figure()
            
            # Perfect prediction line
            min_val = min(min(predictions), min(actual))
            max_val = max(max(predictions), max(actual))
            fig4.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='red', dash='dash'),
                showlegend=True
            ))
            
            # Actual vs Predicted
            fig4.add_trace(go.Scatter(
                x=actual,
                y=predictions,
                mode='markers',
                name='Predictions',
                marker=dict(color='blue', size=8, opacity=0.6),
                hovertemplate='Actual: %{x:.3f}<br>Predicted: %{y:.3f}<extra></extra>'
            ))
            
            fig4.update_layout(
                title='Predictions vs Actual Values',
                xaxis_title='Actual Values',
                yaxis_title='Predicted Values',
                height=400,
                showlegend=True
            )
            
            st.plotly_chart(fig4, use_container_width=True)
            
            # Calculate and display metrics
            mse = np.mean((np.array(predictions) - np.array(actual))**2)
            mae = np.mean(np.abs(np.array(predictions) - np.array(actual)))
            r2 = 1 - (np.sum((np.array(actual) - np.array(predictions))**2) / 
                     np.sum((np.array(actual) - np.mean(actual))**2))
            
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            
            with metric_col1:
                st.metric("Mean Squared Error", f"{mse:.6f}")
            with metric_col2:
                st.metric("Mean Absolute Error", f"{mae:.6f}")
            with metric_col3:
                st.metric("R² Score", f"{r2:.6f}")
        
        # Export options
        st.markdown("### 💾 Export Options")
        
        export_col1, export_col2, export_col3 = st.columns(3)
        
        with export_col1:
            # Export equations as CSV
            csv_buffer = io.StringIO()
            equations_df.to_csv(csv_buffer, index=False)
            st.download_button(
                label="📊 Download Equations CSV",
                data=csv_buffer.getvalue(),
                file_name=f"equations_job_{job_id}.csv",
                mime="text/csv"
            )
        
        with export_col2:
            # Export best equation as text
            if results.get('best_equation'):
                best_eq_text = f"Best Equation: {results['best_equation']['equation']}\n"
                best_eq_text += f"Complexity: {results['best_equation']['complexity']}\n"
                best_eq_text += f"Loss: {results['best_equation']['loss']:.6f}\n"
                best_eq_text += f"Score: {results['best_equation']['score']:.6f}\n"
                
                st.download_button(
                    label="🏆 Download Best Equation",
                    data=best_eq_text,
                    file_name=f"best_equation_job_{job_id}.txt",
                    mime="text/plain"
                )
        
        with export_col3:
            # Export results as JSON
            results_json = json.dumps(results, indent=2)
            st.download_button(
                label="📋 Download Full Results",
                data=results_json,
                file_name=f"results_job_{job_id}.json",
                mime="application/json"
            )

def calculate_pareto_front(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate Pareto front for complexity vs loss"""
    if df.empty:
        return df
    
    # Sort by complexity
    sorted_df = df.sort_values('complexity').reset_index(drop=True)
    pareto_front = []
    
    min_loss = float('inf')
    for _, row in sorted_df.iterrows():
        if row['loss'] < min_loss:
            min_loss = row['loss']
            pareto_front.append(row)
    
    return pd.DataFrame(pareto_front)

def help_tab():
    """Help and documentation tab"""
    st.subheader("ℹ️ Help & Documentation")
    
    # Introduction
    st.markdown("""
    ## Welcome to PySR Symbolic Regression!
    
    PySR (Python Symbolic Regression) is a powerful tool for discovering mathematical 
    relationships in your data automatically. Instead of assuming a particular functional 
    form, PySR searches through the space of mathematical expressions to find the best 
    fit for your data.
    """)
    
    # Getting Started
    with st.expander("🚀 Getting Started"):
        st.markdown("""
        ### Quick Start Guide
        
        1. **Create an Account**: Register or login to get started
        2. **Prepare Your Data**: Upload a CSV file or use our sample datasets
        3. **Configure Parameters**: Set up PySR parameters for your specific problem
        4. **Run the Job**: Start the symbolic regression process
        5. **Analyze Results**: View discovered equations and their performance
        
        ### Data Format
        
        Your CSV file should be structured as follows:
        - Each row represents a data point
        - Each column (except the last) represents a feature/input variable
        - The last column should be your target variable
        - Column names will be used as variable names in equations
        
        Example:
        ```
        x0,x1,x2,y
        1.5,2.3,0.8,5.2
        2.1,1.9,1.2,7.8
        ...
        ```
        """)
    
    # Parameters Guide
    with st.expander("⚙️ Parameter Guide"):
        st.markdown("""
        ### Core Parameters
        
        - **Iterations**: Number of evolutionary iterations. More iterations = better results but longer runtime
        - **Populations**: Number of parallel populations. More populations = better exploration
        - **Population Size**: Size of each population. Larger = more diverse but slower
        - **Max Complexity**: Maximum allowed equation complexity (number of operations)
        - **Max Depth**: Maximum depth of expression trees
        - **Parsimony**: Penalty for complex equations (higher = simpler equations preferred)
        
        ### Operators
        
        - **Binary Operators**: Operations between two terms (+, -, *, /, ^, max, min)
        - **Unary Operators**: Operations on single terms (sin, cos, exp, log, sqrt, etc.)
        
        ### Advanced Options
        
        - **Model Selection**: How to select the best model from the Pareto front
        - **Loss Function**: How to measure prediction error
        - **Random Seed**: For reproducible results
        """)
    
    # Best Practices
    with st.expander("💡 Best Practices"):
        st.markdown("""
        ### Data Preparation
        
        - **Clean your data**: Remove or handle missing values
        - **Scale features**: Consider normalizing features with very different scales
        - **Feature engineering**: Create meaningful derived features if needed
        - **Sample size**: More data generally leads to better results
        
        ### Parameter Tuning
        
        - **Start simple**: Begin with basic operators and increase complexity gradually
        - **Balance exploration vs exploitation**: More populations = more exploration
        - **Consider your domain**: Include operators that make sense for your problem
        - **Computational budget**: More iterations and larger populations = longer runtime
        
        ### Interpreting Results
        
        - **Pareto front**: Trade-off between accuracy and complexity
        - **Cross-validation**: Test discovered equations on held-out data
        - **Domain knowledge**: Verify that equations make physical/logical sense
        - **Robustness**: Check performance on different data splits
        """)
    
    # Troubleshooting
    with st.expander("🔧 Troubleshooting"):
        st.markdown("""
        ### Common Issues
        
        **Job fails to start:**
        - Check that your CSV file is properly formatted
        - Ensure all columns contain numeric data
        - Verify file size is reasonable (< 100MB recommended)
        
        **Poor results:**
        - Try increasing the number of iterations
        - Add more relevant operators
        - Check for data quality issues
        - Consider feature scaling or transformation
        
        **Long runtime:**
        - Reduce number of iterations or population size
        - Use fewer operators
        - Reduce maximum complexity
        - Use a smaller dataset for initial exploration
        
        **Overfitting:**
        - Increase parsimony parameter
        - Reduce maximum complexity
        - Use cross-validation
        - Collect more training data
        """)
    
    # Examples
    with st.expander("📝 Examples"):
        st.markdown("""
        ### Sample Problems
        
        **Physics-inspired:**
        - True equation: `y = 2.5 * cos(x3) + x0² - 0.5`
        - Features: x0, x1, x2, x3
        - Good for testing trigonometric and polynomial operators
        
        **Engineering:**
        - True equation: `y = √x0 * log(x1+1) + x2/x3`
        - Features: x0, x1, x2, x3
        - Good for testing logarithmic and square root operators
        
        **Tips for your own data:**
        - Start with sample datasets to understand the interface
        - Use domain knowledge to select appropriate operators
        - Compare results with traditional regression methods
        """)
    
    # API Reference
    with st.expander("🔌 API Reference"):
        st.markdown(f"""
        ### API Endpoints
        
        Base URL: `{API_BASE_URL}`
        
        **Authentication:**
        - `POST /auth/login` - Login user
        - `POST /auth/register` - Register new user
        - `GET /auth/me` - Get current user info
        
        **Jobs:**
        - `GET /jobs` - List user jobs
        - `POST /jobs` - Create new job
        - `GET /jobs/{{id}}` - Get job details
        - `DELETE /jobs/{{id}}` - Delete job
        - `POST /jobs/{{id}}/upload` - Upload data file
        - `POST /jobs/{{id}}/run` - Start job execution
        
        **Results:**
        - `GET /jobs/{{id}}/runs` - Get job runs
        - `GET /runs/{{run_id}}/results` - Get run results
        """)
    
    # Support
    st.markdown("""
    ## 📞 Support
    
    If you need additional help:
    - Check the [PySR documentation](https://astroautomata.com/PySR/)
    - Review the troubleshooting section above
    - Contact support through the feedback form
    """)

# Main application logic
def main():
    """Main application entry point"""
    if not st.session_state.logged_in:
        login_page()
    else:
        dashboard()

if __name__ == "__main__":
    main()

# SECOND APPROACH FOR STREAMLIT UI




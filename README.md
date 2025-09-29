# pysr-mcp-server
MCP wrapper to expose PySR symbolic regression capabilities via a modular context-aware Python server.

## Overview
`pysr-mcp-server` is a Python-based server that wraps the [PySR](https://github.com/MilesCwith a **Model Context Protocol (MCP)** interface. It enables context-aware model invocation, dynamic configuration, and remote access to symbolic regression workflows.

This project is designed for integration into emerging tech environments where symbolic modeling needs to be orchestrated across multiple contexts and systems.

## Key Features
- MCP-compliant interface for PySR
- Context-aware model invocation
- RESTful API for remote access
- Modular and extensible architecture
- Configurable via YAML or JSON

# 🧬 PySR Symbolic Regression System

A comprehensive web-based platform for symbolic regression using PySR with modern microservices architecture. Discover mathematical relationships in your data through an intuitive interface.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-13+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🌟 Features

### 🔬 Symbolic Regression Capabilities
- **Automated equation discovery** from data using evolutionary algorithms
- **Multi-objective optimization** balancing accuracy and complexity
- **Custom operator support** including trigonometric, logarithmic, and algebraic functions
- **Pareto frontier analysis** for model selection
- **Real-time training monitoring** with progress updates

### 🏗️ Modern Architecture
- **Microservices design** with separate MCP server, API backend, and frontend
- **FastMCP integration** for high-performance symbolic regression processing
- **RESTful API** with comprehensive documentation
- **Real-time updates** and progress monitoring
- **Scalable deployment** with Docker support

### 💻 User Experience
- **Beautiful web interface** built with Streamlit
- **Intuitive job management** with dashboard and monitoring
- **Interactive visualizations** using Plotly
- **Sample datasets** for quick testing
- **Comprehensive help system** with guides and tutorials

### 🔒 Enterprise Ready
- **User authentication** with JWT tokens
- **Role-based access control** 
- **File upload security** with validation
- **Database persistence** with PostgreSQL
- **Production deployment** ready

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Usage](#-usage)
- [API Documentation](#-api-documentation)
- [Architecture](#-architecture)
- [Development](#-development)
- [Deployment](#-deployment)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- PostgreSQL 13+
- Julia 1.6+ (automatically installed with PySR)

### 1. Clone and Setup
```bash
git clone <repository-url>
cd pysr-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Database Setup
```bash
# Start PostgreSQL (using Docker)
docker run --name pysr-postgres \
  -e POSTGRES_PASSWORD=password \
  -e POSTGRES_DB=pysr_db \
  -p 5432:5432 -d postgres:13

# Create tables
python database_setup.py --sample-data
```

### 3. Configuration
```bash
# Copy environment template
cp .env.example .env

# Edit .env file with your settings
nano .env
```

### 4. Start Services
```bash
# Terminal 1: MCP Server
cd mcp_server
python pysr_mcp_server.py

# Terminal 2: Backend API
cd app_server
uvicorn main:app --reload

# Terminal 3: Frontend
cd streamlit_app
streamlit run app.py
```

### 5. Access the Application
- **Web Interface**: http://localhost:8501
- **API Documentation**: http://localhost:8000/docs
- **Login**: demo_user / password123

## 📦 Installation

### Method 1: Local Development

```bash
# Clone repository
git clone <repository-url>
cd pysr-system

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Install Julia dependencies (automatic with PySR)
python -c "import pysr; pysr.install()"
```

### Method 2: Docker (Coming Soon)

```bash
# Clone and start with Docker Compose
git clone <repository-url>
cd pysr-system
docker-compose up -d
```

### Method 3: Production Deployment

See [Deployment Guide](#-deployment) for detailed production setup instructions.

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the root directory:

```bash
# Database
DATABASE_URL=postgresql://postgres:password@localhost:5432/pysr_db

# Security
SECRET_KEY=your-super-secret-jwt-key
ACCESS_TOKEN_EXPIRE_MINUTES=30

# File Storage
UPLOAD_DIR=./uploads
MAX_FILE_SIZE=10485760

# Services
MCP_SERVER_URL=http://localhost:8001
API_BASE_URL=http://localhost:8000
```

### PySR Configuration

Default PySR parameters can be customized in the web interface or via environment variables:

```bash
PYSR_DEFAULT_ITERATIONS=40
PYSR_DEFAULT_POPULATIONS=8
PYSR_DEFAULT_POPULATION_SIZE=33
```

## 📖 Usage

### Creating Your First Job

1. **Login/Register**
   - Navigate to http://localhost:8501
   - Register a new account or use demo credentials

2. **Upload Data**
   - Go to "New Job" tab
   - Upload CSV file (features in columns, target in last column)
   - Or use sample dataset for testing

3. **Configure Parameters**
   - Set iterations, populations, and complexity limits
   - Choose mathematical operators
   - Adjust parsimony for complexity control

4. **Monitor Training**
   - View real-time progress in "Jobs" tab
   - See discovered equations as they evolve
   - Monitor loss and complexity metrics

5. **Analyze Results**
   - Explore Pareto frontier of equations
   - Compare accuracy vs. complexity
   - Export equations in various formats

### Data Format Requirements

Your CSV file should follow this structure:
```csv
feature1,feature2,feature3,target
1.2,0.5,-0.3,3.7
0.8,-1.2,0.9,2.1
-0.5,2.3,0.1,1.9
...
```

**Requirements:**
- CSV format with headers
- Features in first columns, target in last column
- No missing values (clean data beforehand)
- Maximum file size: 10MB

### Sample Datasets

The system includes sample datasets for testing:

**Physics Dataset**: `y = 2.5 * cos(x3) + x0² - 0.5`
- 4 features, 100 samples
- Tests trigonometric and polynomial relationships

**Engineering Dataset**: `y = √x0 * log(x1+1) + x2/x3`
- 4 features, 200 samples  
- Tests complex mathematical relationships

## 📚 API Documentation

### Authentication Endpoints

```http
POST /auth/register    # Register new user
POST /auth/login       # Login user
GET  /auth/me         # Get current user info
```

### Job Management Endpoints

```http
GET    /jobs           # List user jobs
POST   /jobs           # Create new job
GET    /jobs/{id}      # Get job details
PUT    /jobs/{id}      # Update job
DELETE /jobs/{id}      # Delete job
POST   /jobs/{id}/run  # Execute job
```

### File Management Endpoints

```http
POST /jobs/{id}/upload    # Upload data file
GET  /jobs/{id}/download  # Download data file
```

### Monitoring Endpoints

```http
GET /jobs/{id}/status          # Get job status
GET /jobs/{id}/runs           # Get job run history
GET /jobs/{id}/runs/{run_id}  # Get specific run details
```

### Health Check Endpoints

```http
GET /health    # System health check
GET /info      # System information
```

For detailed API documentation with examples, visit http://localhost:8000/docs when the server is running.

## 🏗️ Architecture

### System Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit     │    │   App Server    │    │   MCP Server    │
│   Frontend      │◄──►│   (FastAPI)     │◄──►│   (FastMCP)     │
│                 │    │                 │    │                 │
│ • Authentication│    │ • User Mgmt     │    │ • PySR Training │
│ • Job Dashboard │    │ • Job Mgmt      │    │ • Model Mgmt    │
│ • File Upload   │    │ • File Storage  │    │ • Predictions   │
│ • Results View  │    │ • Database      │    │ • Validation    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │   PostgreSQL    │
                       │   Database      │
                       │                 │
                       │ • Users         │
                       │ • Jobs          │
                       │ • Job Runs      │
                       │ • File Storage  │
                       └─────────────────┘
```

### Technology Stack

**Frontend:**
- Streamlit for web interface
- Plotly for interactive visualizations
- Modern CSS for beautiful styling

**Backend:**
- FastAPI for REST API
- FastMCP for symbolic regression processing
- PySR for equation discovery
- JWT for authentication

**Database:**
- PostgreSQL for data persistence
- asyncpg for async database operations
- Structured schema with proper indexing

**Infrastructure:**
- Docker for containerization
- nginx for reverse proxy (production)
- Redis for caching (optional)

### Data Flow

1. **User uploads data** via Streamlit interface
2. **Frontend sends request** to FastAPI backend
3. **Backend stores job** in PostgreSQL database
4. **Background task calls** MCP server for processing
5. **MCP server runs PySR** symbolic regression
6. **Results stored** in database and displayed in frontend

## 🛠️ Development

### Project Structure

```
pysr-system/
├── mcp_server/
│   ├── pysr_mcp_server.py      # FastMCP server implementation
│   ├── models/                 # Saved models directory
│   └── data/                   # Processing data directory
├── app_server/
│   ├── main.py                 # FastAPI application
│   ├── models.py               # Pydantic models
│   └── utils.py                # Utility functions
├── streamlit_app/
│   ├── app.py                  # Main Streamlit application
│   ├── components/             # UI components
│   └── utils.py                # Frontend utilities
├── database_setup.py           # Database initialization
├── requirements.txt            # Python dependencies
├── .env                        # Environment configuration
├── docker-compose.yml          # Docker configuration
└── README.md                   # This file
```

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8

# Set up pre-commit hooks
pre-commit install

# Run tests
pytest tests/ -v

# Format code
black .
flake8 .
```

### Adding New Features

1. **MCP Server Tools**: Add new functions in `pysr_mcp_server.py` with `@mcp.tool()` decorator
2. **API Endpoints**: Add new routes in `app_server/main.py`
3. **Frontend Components**: Add new pages/components in `streamlit_app/`
4. **Database Changes**: Update schema in `database_setup.py`

### Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_mcp_server.py -v

# Run with coverage
pytest --cov=./ --cov-report=html
```

## 🚀 Deployment

### Production Deployment with Docker

```bash
# Build and deploy
docker-compose -f docker-compose.prod.yml up -d

# Scale services
docker-compose up -d --scale app-server=3
```

### Manual Production Setup

1. **Server Setup**
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install dependencies
sudo apt install python3.8 python3-pip postgresql redis-server nginx

# Clone and setup application
git clone <repository-url> /opt/pysr-system
cd /opt/pysr-system
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. **Database Setup**
```bash
# Create production database
sudo -u postgres createdb pysr_production
sudo -u postgres createuser pysr_user

# Set password and permissions
sudo -u postgres psql -c "ALTER USER pysr_user PASSWORD 'secure_password';"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE pysr_production TO pysr_user;"
```

3. **Environment Configuration**
```bash
# Create production environment file
cp .env.example .env.production

# Edit with production settings
nano .env.production
```

4. **Service Setup**
```bash
# Create systemd services
sudo cp deploy/pysr-*.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable pysr-mcp pysr-api pysr-frontend
sudo systemctl start pysr-mcp pysr-api pysr-frontend
```

5. **Nginx Configuration**
```bash
# Copy nginx configuration
sudo cp deploy/nginx.conf /etc/nginx/sites-available/pysr-system
sudo ln -s /etc/nginx/sites-available/pysr-system /etc/nginx/sites-enabled/
sudo systemctl reload nginx
```

### Environment Variables for Production

```bash
# .env.production
DATABASE_URL=postgresql://pysr_user:secure_password@localhost:5432/pysr_production
SECRET_KEY=very-long-random-string-for-jwt-signing
REDIS_URL=redis://localhost:6379/0
API_BASE_URL=https://pysr.yourdomain.com
USE_SSL=true
PRODUCTION=true
DEBUG=false
```

### SSL/HTTPS Setup

```bash
# Install certbot
sudo apt install certbot python3-certbot-nginx

# Obtain SSL certificate
sudo certbot --nginx -d pysr.yourdomain.com

# Auto-renewal
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

## 🔧 Troubleshooting

### Common Issues

**1. Julia Installation Problems**
```bash
# Reinstall Julia packages
python -c "import pysr; pysr.install(julia_kwargs={'force': True})"

# Check Julia version
julia --version
```

**2. Database Connection Errors**
```bash
# Check PostgreSQL status
sudo systemctl status postgresql

# Reset database
python database_setup.py --reset --sample-data

# Check connection
python -c "import asyncpg; print('OK')"
```

**3. MCP Server Connection Issues**
```bash
# Check MCP server health
curl http://localhost:8001/health

# Check logs
tail -f logs/mcp_server.log

# Restart with debug
python mcp_server/pysr_mcp_server.py --debug
```

**4. Streamlit Issues**
```bash
# Clear Streamlit cache
streamlit cache clear

# Restart with verbose logging
streamlit run app.py --logger.level debug
```

### Performance Optimization

**For Large Datasets:**
- Use multiprocessing for PySR training
- Implement data chunking for file uploads
- Configure Redis for session caching
- Use CDN for static assets

**Memory Management:**
- Set Julia memory limits: `export JULIA_NUM_THREADS=4`
- Implement model checkpointing
- Use streaming for large file processing
- Configure garbage collection

### Monitoring and Logging

**Log Files:**
```bash
# View application logs
tail -f logs/app_server.log
tail -f logs/mcp_server.log
tail -f logs/streamlit.log

# View system logs
journalctl -u pysr-api -f
journalctl -u pysr-mcp -f
```

**Health Monitoring:**
- API health endpoint: `GET /health`
- Database connection monitoring
- MCP server status checks
- Resource usage monitoring

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

### Development Process

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature-name`
3. **Make changes** and add tests
4. **Run tests**: `pytest`
5. **Format code**: `black . && flake8 .`
6. **Commit changes**: `git commit -m "Add feature"`
7. **Push to branch**: `git push origin feature-name`
8. **Create Pull Request**

### Code Standards

- Follow PEP 8 style guide
- Add type hints for all functions
- Write comprehensive docstrings
- Include tests for new features
- Update documentation as needed

### Areas for Contribution

- **Algorithm improvements**: New operators, optimization techniques
- **UI/UX enhancements**: Better visualizations, user experience
- **Performance optimizations**: Faster training, better scaling
- **Documentation**: Tutorials, examples, guides
- **Testing**: More comprehensive test coverage
- **Deployment**: Docker improvements, cloud deployment

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **PySR**: Miles Cranmer for the excellent symbolic regression library
- **FastMCP**: For the Model Context Protocol framework
- **FastAPI**: For the modern web framework
- **Streamlit**: For the beautiful frontend framework
- **Julia**: For high-performance scientific computing

## 📞 Support

- **Documentation**: Check this README and inline documentation
- **Issues**: Create GitHub issues for bugs and feature requests
- **Discussions**: Use GitHub Discussions for questions
- **Email**: contact@pysr-system.com (if available)

## 🗺️ Roadmap

### Version 2.0 (Coming Soon)
- [ ] Advanced equation validation and testing
- [ ] Model comparison and ensemble methods
- [ ] Cloud deployment templates
- [ ] Advanced visualization features
- [ ] Integration with popular ML frameworks

### Version 2.1
- [ ] Real-time collaborative features
- [ ] Advanced user management
- [ ] API rate limiting and quotas
- [ ] Enhanced security features
- [ ] Mobile-responsive interface

### Version 3.0
- [ ] Multi-tenant architecture
- [ ] Advanced analytics and reporting
- [ ] Integration with cloud ML services
- [ ] Advanced symbolic regression techniques
- [ ] Enterprise features and SSO

---

**Happy symbolic regression! 🧬✨**

For more information, visit our [documentation site](https://pysr-system.readthedocs.io) or check out the [API documentation](http://localhost:8000/docs).

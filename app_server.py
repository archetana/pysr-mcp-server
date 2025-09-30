

import os
import uuid
import json
import asyncio
import logging
import httpx
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
import shutil
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, status, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, EmailStr
from passlib.context import CryptContext
from jose import JWTError, jwt
import asyncpg
from asyncpg import Pool
import pandas as pd

# Environment variables
from dotenv import load_dotenv
load_dotenv()

# Configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:amit1199@localhost:5432/postgres")
SECRET_KEY = os.getenv("SECRET_KEY", "your-super-secret-key-change-this-in-production")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "./uploads"))
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", "10485760"))  # 10MB
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8001")

# Ensure upload directory exists
UPLOAD_DIR.mkdir(exist_ok=True, parents=True)

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database connection pool
db_pool: Optional[Pool] = None

# Security
security = HTTPBearer()

# Fix for bcrypt password length issue
def safe_password_truncate(password: str) -> str:
    """Safely truncate password to avoid bcrypt 72-byte limit"""
    password_bytes = password.encode('utf-8')
    if len(password_bytes) > 70:  # Use 70 as safe limit
        # Truncate to 70 bytes and decode back to string
        truncated_bytes = password_bytes[:70]
        # Handle potential incomplete UTF-8 sequences at the end
        try:
            return truncated_bytes.decode('utf-8')
        except UnicodeDecodeError:
            # If truncation breaks UTF-8, try shorter lengths
            for i in range(69, 50, -1):
                try:
                    return password_bytes[:i].decode('utf-8')
                except UnicodeDecodeError:
                    continue
            # Fallback: use only ASCII characters
            return password_bytes[:70].decode('utf-8', errors='ignore')
    return password

# Initialize password context with aggressive error handling
pwd_context = None

def init_password_context():
    """Initialize password context with multiple fallback options"""
    global pwd_context
    
    # Try different hashing schemes in order of preference
    schemes_to_try = [
        ("pbkdf2_sha256", "PBKDF2 with SHA256"),
        ("scrypt", "scrypt"),
        ("argon2", "Argon2"),
        ("bcrypt", "bcrypt")
    ]
    
    for scheme, description in schemes_to_try:
        try:
            pwd_context = CryptContext(schemes=[scheme], deprecated="auto")
            logger.info(f"Successfully initialized password hashing with {description}")
            return
        except Exception as e:
            logger.warning(f"Failed to initialize {description}: {str(e)}")
            continue
    
    # If all else fails, use a simple but secure fallback
    import hashlib
    logger.error("All password hashing schemes failed, using simple SHA256 fallback")
    pwd_context = None

# Initialize the password context
init_password_context()

# Pydantic models
class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=6, max_length=70)  # Limit password length
    full_name: Optional[str] = Field(None, max_length=100)

class UserLogin(BaseModel):
    username: str
    password: str = Field(..., max_length=70)  # Limit password length

class User(BaseModel):
    id: int
    username: str
    email: str
    full_name: Optional[str]
    is_active: bool
    created_at: datetime

class Token(BaseModel):
    access_token: str
    token_type: str
    expires_in: int

class JobCreate(BaseModel):
    name: str = Field(..., max_length=200)
    description: Optional[str] = Field(None, max_length=1000)
    config: Dict[str, Any] = Field(default_factory=dict)

class JobUpdate(BaseModel):
    name: Optional[str] = Field(None, max_length=200)
    description: Optional[str] = Field(None, max_length=1000)
    config: Optional[Dict[str, Any]] = None

class Job(BaseModel):
    id: int
    user_id: int
    name: str
    description: Optional[str]
    status: str
    config: Dict[str, Any]
    file_path: Optional[str]
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime]

class JobRun(BaseModel):
    id: int
    job_id: int
    run_id: str
    status: str
    results: Optional[Dict[str, Any]]
    error_message: Optional[str]
    started_at: datetime
    completed_at: Optional[datetime]

# Database functions
async def get_db_pool():
    """Get database connection pool"""
    global db_pool
    if db_pool is None:
        db_pool = await asyncpg.create_pool(DATABASE_URL)
    return db_pool

async def init_database():
    """Initialize database tables"""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        # Users table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                username VARCHAR(50) UNIQUE NOT NULL,
                email VARCHAR(255) UNIQUE NOT NULL,
                password_hash VARCHAR(255) NOT NULL,
                full_name VARCHAR(100),
                is_active BOOLEAN DEFAULT true,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Jobs table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id SERIAL PRIMARY KEY,
                user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                name VARCHAR(200) NOT NULL,
                description TEXT,
                status VARCHAR(50) DEFAULT 'created',
                config JSONB DEFAULT '{}',
                file_path VARCHAR(500),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP
            )
        """)
        
        # Job runs table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS job_runs (
                id SERIAL PRIMARY KEY,
                job_id INTEGER REFERENCES jobs(id) ON DELETE CASCADE,
                run_id VARCHAR(100) UNIQUE NOT NULL,
                status VARCHAR(50) DEFAULT 'pending',
                results JSONB,
                error_message TEXT,
                started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP
            )
        """)
        
        # File storage table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS file_storage (
                id SERIAL PRIMARY KEY,
                user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                job_id INTEGER REFERENCES jobs(id) ON DELETE CASCADE,
                filename VARCHAR(255) NOT NULL,
                original_filename VARCHAR(255) NOT NULL,
                file_path VARCHAR(500) NOT NULL,
                file_size BIGINT,
                mime_type VARCHAR(100),
                uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_user_id ON jobs(user_id)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_job_runs_job_id ON job_runs(job_id)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_file_storage_user_id ON file_storage(user_id)")

# Lifespan context manager (replaces deprecated on_event)
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting PySR App Server...")
    await init_database()
    logger.info("Database initialized")
    logger.info(f"MCP Server URL: {MCP_SERVER_URL}")
    logger.info(f"Upload directory: {UPLOAD_DIR}")
    
    yield
    
    # Shutdown
    global db_pool
    if db_pool:
        await db_pool.close()
    logger.info("PySR App Server shutdown complete")

# FastAPI app with lifespan
app = FastAPI(
    title="PySR App Server",
    description="Backend API for PySR Symbolic Regression System",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Utility functions
def parse_job_record(job_record) -> Job:
    """Helper function to parse job record from database and handle JSON config"""
    job_dict = dict(job_record)
    
    # Parse JSON config field if it's a string
    if job_dict.get('config'):
        if isinstance(job_dict['config'], str):
            try:
                job_dict['config'] = json.loads(job_dict['config'])
            except (json.JSONDecodeError, TypeError):
                job_dict['config'] = {}
    else:
        job_dict['config'] = {}
    
    return Job(**job_dict)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash with safe truncation and fallback"""
    try:
        if pwd_context is None:
            # Fallback verification using simple hash comparison
            return simple_verify_password(plain_password, hashed_password)
        
        safe_password = safe_password_truncate(plain_password)
        return pwd_context.verify(safe_password, hashed_password)
    except Exception as e:
        logger.error(f"Password verification error: {str(e)}")
        # Try fallback verification
        try:
            return simple_verify_password(plain_password, hashed_password)
        except:
            return False

def get_password_hash(password: str) -> str:
    """Hash password with safe truncation and fallback"""
    try:
        if pwd_context is None:
            # Fallback hashing using simple but secure method
            return simple_hash_password(password)
        
        safe_password = safe_password_truncate(password)
        return pwd_context.hash(safe_password)
    except Exception as e:
        logger.error(f"Password hashing error: {str(e)}")
        # Try fallback hashing
        try:
            return simple_hash_password(password)
        except Exception as e2:
            logger.error(f"Fallback hashing also failed: {str(e2)}")
            raise HTTPException(status_code=500, detail="Password hashing system unavailable")

def simple_hash_password(password: str) -> str:
    """Simple but secure password hashing fallback"""
    import hashlib
    import secrets
    import base64
    
    # Generate a random salt
    salt = secrets.token_bytes(32)
    
    # Truncate password safely
    safe_password = safe_password_truncate(password)
    
    # Hash the password with salt using PBKDF2
    pwd_hash = hashlib.pbkdf2_hmac('sha256', 
                                   safe_password.encode('utf-8'), 
                                   salt, 
                                   100000)  # 100,000 iterations
    
    # Combine salt and hash, encode as base64
    combined = salt + pwd_hash
    return base64.b64encode(combined).decode('ascii')

def simple_verify_password(plain_password: str, hashed_password: str) -> bool:
    """Simple but secure password verification fallback"""
    import hashlib
    import base64
    
    try:
        # Decode the stored hash
        combined = base64.b64decode(hashed_password.encode('ascii'))
        
        # Extract salt (first 32 bytes) and hash (rest)
        salt = combined[:32]
        stored_hash = combined[32:]
        
        # Truncate password safely
        safe_password = safe_password_truncate(plain_password)
        
        # Hash the provided password with the same salt
        pwd_hash = hashlib.pbkdf2_hmac('sha256', 
                                       safe_password.encode('utf-8'), 
                                       salt, 
                                       100000)
        
        # Compare hashes
        return pwd_hash == stored_hash
    except Exception:
        return False

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current authenticated user"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        user_record = await conn.fetchrow(
            "SELECT * FROM users WHERE username = $1 AND is_active = true", username
        )
        if user_record is None:
            raise credentials_exception
        
    return User(**dict(user_record))

async def get_user_by_username(username: str):
    """Get user by username"""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        user_record = await conn.fetchrow(
            "SELECT * FROM users WHERE username = $1", username
        )
        return dict(user_record) if user_record else None

async def create_user(user_data: UserCreate):
    """Create new user with password validation"""
    # Validate password length before processing
    if len(user_data.password.encode('utf-8')) > 72:
        raise HTTPException(
            status_code=400, 
            detail="Password is too long. Please use a password with 70 characters or fewer."
        )
    
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        # Check if username or email already exists
        existing = await conn.fetchrow(
            "SELECT username, email FROM users WHERE username = $1 OR email = $2",
            user_data.username, user_data.email
        )
        if existing:
            if existing['username'] == user_data.username:
                raise HTTPException(status_code=400, detail="Username already exists")
            else:
                raise HTTPException(status_code=400, detail="Email already exists")
        
        # Create user with safe password hashing
        try:
            password_hash = get_password_hash(user_data.password)
        except Exception as e:
            logger.error(f"Password hashing failed for user {user_data.username}: {str(e)}")
            raise HTTPException(status_code=500, detail="Registration failed due to password processing error")
        
        user_id = await conn.fetchval("""
            INSERT INTO users (username, email, password_hash, full_name)
            VALUES ($1, $2, $3, $4)
            RETURNING id
        """, user_data.username, user_data.email, password_hash, user_data.full_name)
        
        # Return created user
        user_record = await conn.fetchrow("SELECT * FROM users WHERE id = $1", user_id)
        return User(**dict(user_record))

# Authentication endpoints
@app.post("/auth/register", response_model=User)
async def register(user_data: UserCreate):
    """Register a new user with enhanced error handling"""
    try:
        # Additional password validation
        if len(user_data.password) > 70:
            raise HTTPException(
                status_code=400, 
                detail="Password is too long. Please use 70 characters or fewer."
            )
        
        user = await create_user(user_data)
        logger.info(f"User registered: {user.username}")
        return user
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration error: {str(e)}")
        raise HTTPException(status_code=500, detail="Registration failed")

@app.post("/auth/login", response_model=Token)
async def login(login_data: UserLogin):
    """Login user and return JWT token with enhanced error handling"""
    try:
        # Validate password length
        if len(login_data.password) > 70:
            raise HTTPException(
                status_code=400,
                detail="Password is too long"
            )
        
        user = await get_user_by_username(login_data.username)
        
        if not user or not verify_password(login_data.password, user['password_hash']):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        if not user['is_active']:
            raise HTTPException(status_code=400, detail="Inactive user")
        
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user['username']}, expires_delta=access_token_expires
        )
        
        logger.info(f"User logged in: {user['username']}")
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        raise HTTPException(status_code=500, detail="Login failed")

@app.get("/auth/me", response_model=User)
async def get_me(current_user: User = Depends(get_current_user)):
    """Get current user information"""
    return current_user

# Job management endpoints
@app.post("/jobs", response_model=Job)
async def create_job(
    job_data: JobCreate,
    current_user: User = Depends(get_current_user)
):
    """Create a new job"""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        job_id = await conn.fetchval("""
            INSERT INTO jobs (user_id, name, description, config)
            VALUES ($1, $2, $3, $4)
            RETURNING id
        """, current_user.id, job_data.name, job_data.description, json.dumps(job_data.config))
        
        job_record = await conn.fetchrow("SELECT * FROM jobs WHERE id = $1", job_id)
        
        # Fix: Use helper function to properly parse the job record
        job = parse_job_record(job_record)
        
    logger.info(f"Job created: {job.id} by user {current_user.username}")
    return job

@app.get("/jobs", response_model=List[Job])
async def list_jobs(
    skip: int = 0,
    limit: int = 100,
    status: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    """List user's jobs"""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        query = "SELECT * FROM jobs WHERE user_id = $1"
        params = [current_user.id]
        
        if status:
            query += " AND status = $2"
            params.append(status)
            
        query += " ORDER BY created_at DESC LIMIT $%d OFFSET $%d" % (len(params) + 1, len(params) + 2)
        params.extend([limit, skip])
        
        job_records = await conn.fetch(query, *params)
        jobs = [parse_job_record(record) for record in job_records]
        
    return jobs

@app.get("/jobs/{job_id}", response_model=Job)
async def get_job(
    job_id: int,
    current_user: User = Depends(get_current_user)
):
    """Get specific job"""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        job_record = await conn.fetchrow(
            "SELECT * FROM jobs WHERE id = $1 AND user_id = $2",
            job_id, current_user.id
        )
        if not job_record:
            raise HTTPException(status_code=404, detail="Job not found")
        
    return parse_job_record(job_record)

@app.put("/jobs/{job_id}", response_model=Job)
async def update_job(
    job_id: int,
    job_data: JobUpdate,
    current_user: User = Depends(get_current_user)
):
    """Update job"""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        # Check if job exists and belongs to user
        existing = await conn.fetchrow(
            "SELECT id FROM jobs WHERE id = $1 AND user_id = $2",
            job_id, current_user.id
        )
        if not existing:
            raise HTTPException(status_code=404, detail="Job not found")
        
        # Build update query
        update_fields = []
        params = []
        param_count = 1
        
        if job_data.name is not None:
            update_fields.append(f"name = ${param_count}")
            params.append(job_data.name)
            param_count += 1
            
        if job_data.description is not None:
            update_fields.append(f"description = ${param_count}")
            params.append(job_data.description)
            param_count += 1
            
        if job_data.config is not None:
            update_fields.append(f"config = ${param_count}")
            params.append(json.dumps(job_data.config))
            param_count += 1
            
        if update_fields:
            update_fields.append(f"updated_at = ${param_count}")
            params.append(datetime.utcnow())
            param_count += 1
            
            params.extend([job_id, current_user.id])
            
            query = f"""
                UPDATE jobs SET {', '.join(update_fields)}
                WHERE id = ${param_count-1} AND user_id = ${param_count}
            """
            await conn.execute(query, *params)
        
        # Return updated job
        job_record = await conn.fetchrow("SELECT * FROM jobs WHERE id = $1", job_id)
        job = parse_job_record(job_record)
        
    logger.info(f"Job updated: {job_id} by user {current_user.username}")
    return job

@app.delete("/jobs/{job_id}")
async def delete_job(
    job_id: int,
    current_user: User = Depends(get_current_user)
):
    """Delete job"""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        # Check if job exists and belongs to user
        job_record = await conn.fetchrow(
            "SELECT file_path FROM jobs WHERE id = $1 AND user_id = $2",
            job_id, current_user.id
        )
        if not job_record:
            raise HTTPException(status_code=404, detail="Job not found")
        
        # Delete associated files
        if job_record['file_path']:
            file_path = Path(job_record['file_path'])
            if file_path.exists():
                file_path.unlink()
        
        # Delete job (cascades to job_runs and file_storage)
        await conn.execute("DELETE FROM jobs WHERE id = $1", job_id)
        
    logger.info(f"Job deleted: {job_id} by user {current_user.username}")
    return {"message": "Job deleted successfully"}

# File upload endpoints
@app.post("/jobs/{job_id}/upload")
async def upload_file(
    job_id: int,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
):
    """Upload file for job"""
    # Verify job ownership
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        job_record = await conn.fetchrow(
            "SELECT id FROM jobs WHERE id = $1 AND user_id = $2",
            job_id, current_user.id
        )
        if not job_record:
            raise HTTPException(status_code=404, detail="Job not found")
    
    # Validate file
    if file.size and file.size > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large")
    
    # Create user directory
    user_dir = UPLOAD_DIR / str(current_user.id)
    user_dir.mkdir(exist_ok=True)
    
    # Generate unique filename
    file_extension = Path(file.filename).suffix if file.filename else ""
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = user_dir / unique_filename
    
    # Save file
    try:
        contents = await file.read()
        with open(file_path, "wb") as f:
            f.write(contents)
            
        # Validate CSV file
        if file_extension.lower() == '.csv':
            try:
                df = pd.read_csv(file_path)
                if df.empty:
                    raise ValueError("CSV file is empty")
                if len(df.columns) < 2:
                    raise ValueError("CSV file must have at least 2 columns")
            except Exception as e:
                file_path.unlink()  # Delete invalid file
                raise HTTPException(status_code=400, detail=f"Invalid CSV file: {str(e)}")
        
    except Exception as e:
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")
    
    # Store file info in database
    async with pool.acquire() as conn:
        file_id = await conn.fetchval("""
            INSERT INTO file_storage (user_id, job_id, filename, original_filename, file_path, file_size, mime_type)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            RETURNING id
        """, current_user.id, job_id, unique_filename, file.filename, 
            str(file_path), len(contents), file.content_type)
        
        # Update job with file path
        await conn.execute(
            "UPDATE jobs SET file_path = $1, updated_at = $2 WHERE id = $3",
            str(file_path), datetime.utcnow(), job_id
        )
    
    logger.info(f"File uploaded: {file.filename} for job {job_id} by user {current_user.username}")
    
    return {
        "message": "File uploaded successfully",
        "file_id": file_id,
        "filename": unique_filename,
        "original_filename": file.filename,
        "file_size": len(contents)
    }

@app.get("/jobs/{job_id}/download")
async def download_file(
    job_id: int,
    current_user: User = Depends(get_current_user)
):
    """Download job file"""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        job_record = await conn.fetchrow(
            "SELECT file_path FROM jobs WHERE id = $1 AND user_id = $2",
            job_id, current_user.id
        )
        if not job_record or not job_record['file_path']:
            raise HTTPException(status_code=404, detail="File not found")
        
        file_path = Path(job_record['file_path'])
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found on disk")
        
        file_record = await conn.fetchrow(
            "SELECT original_filename FROM file_storage WHERE job_id = $1",
            job_id
        )
        
    original_filename = file_record['original_filename'] if file_record else file_path.name
    
    return FileResponse(
        path=file_path,
        filename=original_filename,
        media_type='application/octet-stream'
    )

# Job execution endpoints
# async def call_mcp_server(endpoint: str, data: dict = None):
#     """Call MCP server endpoint"""
#     try:
#         async with httpx.AsyncClient(timeout=300.0) as client:  # 5 minute timeout
#             if data:
#                 response = await client.post(f"{MCP_SERVER_URL}/{endpoint}", json=data)
#             else:
#                 response = await client.get(f"{MCP_SERVER_URL}/{endpoint}")
#             response.raise_for_status()
#             return response.json()
#     except httpx.RequestError as e:
#         logger.error(f"MCP server request error: {str(e)}")
#         raise HTTPException(status_code=503, detail="MCP server unavailable")
#     except httpx.HTTPStatusError as e:
#         logger.error(f"MCP server HTTP error: {e.response.status_code}")
#         raise HTTPException(status_code=502, detail="MCP server error")

# In app_server.py, fix the call_mcp_server function around line 680:
async def call_mcp_server(endpoint: str, data: dict = None):
    """Call MCP server endpoint"""
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            # Fix: Add proper endpoint routing
            if endpoint in ["create_model", "fit_model", "get_equations", "evaluate_model"]:
                url = f"{MCP_SERVER_URL}/tools/{endpoint}"
            else:
                url = f"{MCP_SERVER_URL}/{endpoint}"
                
            if data:
                response = await client.post(url, json=data)
            else:
                response = await client.get(url)
            response.raise_for_status()
            return response.json()
    except httpx.RequestError as e:
        logger.error(f"MCP server request error: {str(e)}")
        raise HTTPException(status_code=503, detail="MCP server unavailable")
    except httpx.HTTPStatusError as e:
        logger.error(f"MCP server HTTP error: {e.response.status_code}")
        raise HTTPException(status_code=502, detail="MCP server error")

async def run_pysr_job(job_id: int, user_id: int, run_id: str):
    """Background task to run PySR job"""
    pool = await get_db_pool()
    
    try:
        # Get job details
        async with pool.acquire() as conn:
            job_record = await conn.fetchrow(
                "SELECT * FROM jobs WHERE id = $1 AND user_id = $2",
                job_id, user_id
            )
            if not job_record:
                raise Exception("Job not found")
            
            job = parse_job_record(job_record)
            
            # Update job status
            await conn.execute(
                "UPDATE jobs SET status = 'running', updated_at = $1 WHERE id = $2",
                datetime.utcnow(), job_id
            )
            
            # Update job run status
            await conn.execute(
                "UPDATE job_runs SET status = 'running' WHERE run_id = $1",
                run_id
            )
        
        # Load and validate data
        if not job.file_path:
            raise Exception("No data file provided")
            
        df = pd.read_csv(job.file_path)
        
        # Assume last column is target, rest are features
        X = df.iloc[:, :-1].values.tolist()
        y = df.iloc[:, -1].values.tolist()
        feature_names = df.columns[:-1].tolist()
        
        # Prepare training data
        training_data = {
            "X": X,
            "y": y,
            "feature_names": feature_names,
            "variable_names": feature_names
        }
        
        # Create model via MCP server
        model_config = job.config.get("model_config", {})
        create_response = await call_mcp_server("create_model", {
            "model_id": run_id,
            "config": model_config
        })
        
        if not create_response.get("success"):
            raise Exception(f"Model creation failed: {create_response.get('error')}")
        
        # Train model via MCP server
        fit_response = await call_mcp_server("fit_model", {
            "model_id": run_id,
            "data": training_data
        })
        
        if not fit_response.get("success"):
            raise Exception(f"Model training failed: {fit_response.get('error')}")
        
        # Get equations
        equations_response = await call_mcp_server("get_equations", {
            "model_id": run_id,
            "include_details": True
        })
        
        if not equations_response.get("success"):
            raise Exception(f"Failed to get equations: {equations_response.get('error')}")
        
        # Evaluate model
        eval_response = await call_mcp_server("evaluate_model", {
            "model_id": run_id
        })
        
        # Compile results
        results = {
            "model_id": run_id,
            "training_results": fit_response,
            "equations": equations_response.get("equations", []),
            "best_equation": equations_response.get("best_equation"),
            "evaluation": eval_response if eval_response.get("success") else None,
            "data_info": {
                "samples": len(X),
                "features": len(feature_names),
                "feature_names": feature_names
            }
        }
        
        # Update database with success
        async with pool.acquire() as conn:
            await conn.execute("""
                UPDATE job_runs SET 
                    status = 'completed',
                    results = $1,
                    completed_at = $2
                WHERE run_id = $3
            """, json.dumps(results), datetime.utcnow(), run_id)
            
            await conn.execute("""
                UPDATE jobs SET 
                    status = 'completed',
                    updated_at = $1,
                    completed_at = $2
                WHERE id = $3
            """, datetime.utcnow(), datetime.utcnow(), job_id)
        
        logger.info(f"Job {job_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Job {job_id} failed: {str(e)}")
        
        # Update database with failure
        async with pool.acquire() as conn:
            await conn.execute("""
                UPDATE job_runs SET 
                    status = 'failed',
                    error_message = $1,
                    completed_at = $2
                WHERE run_id = $3
            """, str(e), datetime.utcnow(), run_id)
            
            await conn.execute("""
                UPDATE jobs SET 
                    status = 'failed',
                    updated_at = $1
                WHERE id = $2
            """, datetime.utcnow(), job_id)

@app.post("/jobs/{job_id}/run", response_model=JobRun)
async def run_job(
    job_id: int,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """Start job execution"""
    pool = await get_db_pool()
    
    # Verify job ownership and get job details
    async with pool.acquire() as conn:
        job_record = await conn.fetchrow(
            "SELECT * FROM jobs WHERE id = $1 AND user_id = $2",
            job_id, current_user.id
        )
        if not job_record:
            raise HTTPException(status_code=404, detail="Job not found")
        
        job = parse_job_record(job_record)
        
        if job.status in ['running']:
            raise HTTPException(status_code=400, detail="Job is already running")
        
        if not job.file_path:
            raise HTTPException(status_code=400, detail="No data file uploaded")
        
        # Create job run
        run_id = str(uuid.uuid4())
        job_run_id = await conn.fetchval("""
            INSERT INTO job_runs (job_id, run_id, status)
            VALUES ($1, $2, 'pending')
            RETURNING id
        """, job_id, run_id)
        
        job_run_record = await conn.fetchrow(
            "SELECT * FROM job_runs WHERE id = $1", job_run_id
        )
        job_run = JobRun(**dict(job_run_record))
    
    # Start background task
    background_tasks.add_task(run_pysr_job, job_id, current_user.id, run_id)
    
    logger.info(f"Job {job_id} started by user {current_user.username}")
    return job_run

# @app.get("/jobs/{job_id}/runs", response_model=List[JobRun])
# async def get_job_runs(
#     job_id: int,
#     current_user: User = Depends(get_current_user)
# ):
#     """Get job run history"""
#     pool = await get_db_pool()
#     async with pool.acquire() as conn:
#         # Verify job ownership
#         job_exists = await conn.fetchval(
#             "SELECT EXISTS(SELECT 1 FROM jobs WHERE id = $1 AND user_id = $2)",
#             job_id, current_user.id
#         )
#         if not job_exists:
#             raise HTTPException(status_code=404, detail="Job not found")
        
#         # Get job runs
#         run_records = await conn.fetch("""
#             SELECT * FROM job_runs 
#             WHERE job_id = $1 
#             ORDER BY started_at DESC
#         """, job_id)
        
#         runs = [JobRun(**dict(record)) for record in run_records]
    
#     return runs

# @app.get("/jobs/{job_id}/runs/{run_id}", response_model=JobRun)
# async def get_job_run(
#     job_id: int,
#     run_id: str,
#     current_user: User = Depends(get_current_user)
# ):
#     """Get specific job run"""
#     pool = await get_db_pool()
#     async with pool.acquire() as conn:
#         run_record = await conn.fetchrow("""
#             SELECT jr.* FROM job_runs jr
#             JOIN jobs j ON jr.job_id = j.id
#             WHERE jr.job_id = $1 AND jr.run_id = $2 AND j.user_id = $3
#         """, job_id, run_id, current_user.id)
        
#         if not run_record:
#             raise HTTPException(status_code=404, detail="Job run not found")
        
#     return JobRun(**dict(run_record))

@app.get("/jobs/{job_id}/runs", response_model=List[JobRun])
async def get_job_runs(
    job_id: int,
    current_user: User = Depends(get_current_user)
):
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        job_exists = await conn.fetchval(
            "SELECT EXISTS(SELECT 1 FROM jobs WHERE id = $1 AND user_id = $2)",
            job_id, current_user.id
        )
        if not job_exists:
            raise HTTPException(status_code=404, detail="Job not found")
        
        run_records = await conn.fetch("""
            SELECT * FROM job_runs 
            WHERE job_id = $1 
            ORDER BY started_at DESC
        """, job_id)
        
        # Parse JSON fields before creating JobRun objects
        runs = []
        for record in run_records:
            record_dict = dict(record)
            
            # Parse results field if it's a string
            if record_dict.get('results') and isinstance(record_dict['results'], str):
                try:
                    record_dict['results'] = json.loads(record_dict['results'])
                except (json.JSONDecodeError, TypeError):
                    record_dict['results'] = None
            
            runs.append(JobRun(**record_dict))
    
    return runs

@app.get("/jobs/{job_id}/runs/{run_id}", response_model=JobRun)
async def get_job_run(
    job_id: int,
    run_id: str,
    current_user: User = Depends(get_current_user)
):
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        run_record = await conn.fetchrow("""
            SELECT jr.* FROM job_runs jr
            JOIN jobs j ON jr.job_id = j.id
            WHERE jr.job_id = $1 AND jr.run_id = $2 AND j.user_id = $3
        """, job_id, run_id, current_user.id)
        
        if not run_record:
            raise HTTPException(status_code=404, detail="Job run not found")
        
        # Parse JSON fields
        record_dict = dict(run_record)
        if record_dict.get('results') and isinstance(record_dict['results'], str):
            try:
                record_dict['results'] = json.loads(record_dict['results'])
            except (json.JSONDecodeError, TypeError):
                record_dict['results'] = None
        
    return JobRun(**record_dict)

@app.get("/jobs/{job_id}/status")
async def get_job_status(
    job_id: int,
    current_user: User = Depends(get_current_user)
):
    """Get current job status"""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        job_record = await conn.fetchrow(
            "SELECT status, updated_at FROM jobs WHERE id = $1 AND user_id = $2",
            job_id, current_user.id
        )
        if not job_record:
            raise HTTPException(status_code=404, detail="Job not found")
        
        # Get latest run info
        latest_run = await conn.fetchrow("""
            SELECT status, started_at, completed_at, error_message
            FROM job_runs 
            WHERE job_id = $1 
            ORDER BY started_at DESC 
            LIMIT 1
        """, job_id)
    
    result = {
        "job_id": job_id,
        "status": job_record['status'],
        "updated_at": job_record['updated_at']
    }
    
    if latest_run:
        result["latest_run"] = {
            "status": latest_run['status'],
            "started_at": latest_run['started_at'],
            "completed_at": latest_run['completed_at'],
            "error_message": latest_run['error_message']
        }
    
    return result

# Health check and info endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test database connection
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        
        # Test MCP server connection
        try:
            mcp_status = await call_mcp_server("health_check")
            mcp_connected = mcp_status.get("success", False)
        except:
            mcp_connected = False
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "database": "connected",
            "mcp_server": "connected" if mcp_connected else "disconnected",
            "password_hashing": pwd_context.schemes()[0] if pwd_context else "simple_fallback"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }

@app.get("/info")
async def get_info():
    """Get server information"""
    return {
        "name": "PySR App Server",
        "version": "1.0.0",
        "description": "Backend API for PySR Symbolic Regression System",
        "endpoints": {
            "authentication": ["/auth/register", "/auth/login", "/auth/me"],
            "jobs": ["/jobs", "/jobs/{id}", "/jobs/{id}/run"],
            "files": ["/jobs/{id}/upload", "/jobs/{id}/download"],
            "monitoring": ["/health", "/info"]
        },
        "mcp_server_url": MCP_SERVER_URL,
        "max_file_size": MAX_FILE_SIZE,
        "password_hashing_scheme": pwd_context.schemes()[0] if pwd_context else "simple_fallback",
        "max_password_length": 70
    }

# Additional endpoint for password validation
@app.post("/auth/validate-password")
async def validate_password(password: str):
    """Validate password before registration"""
    try:
        if len(password) < 6:
            return {"valid": False, "error": "Password must be at least 6 characters long"}
        if len(password) > 70:
            return {"valid": False, "error": "Password must be 70 characters or fewer"}
        if len(password.encode('utf-8')) > 72:
            return {"valid": False, "error": "Password contains characters that make it too long for secure hashing"}
        
        # Test if password can be hashed
        safe_password_truncate(password)
        return {"valid": True, "message": "Password is valid"}
    except Exception as e:
        return {"valid": False, "error": f"Password validation failed: {str(e)}"}

# Run the server directly (for development)
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"

    )

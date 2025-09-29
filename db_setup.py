#!/usr/bin/env python3
"""
Database Setup Script for PySR System
Creates tables and initial data
"""

import asyncio
import asyncpg
import os
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path to import app modules
sys.path.append(str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:amit1199@localhost:5432/postgres")

async def create_database_schema():
    """Create all database tables and indexes"""
    
    try:
        # Connect to database
        conn = await asyncpg.connect(DATABASE_URL)
        print("✅ Connected to database")
        
        # Drop existing tables if --reset flag is provided
        if "--reset" in sys.argv:
            print("🗑️  Dropping existing tables...")
            await conn.execute("DROP TABLE IF EXISTS file_storage CASCADE")
            await conn.execute("DROP TABLE IF EXISTS job_runs CASCADE") 
            await conn.execute("DROP TABLE IF EXISTS jobs CASCADE")
            await conn.execute("DROP TABLE IF EXISTS users CASCADE")
            print("✅ Existing tables dropped")
        
        # Create users table
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
        print("✅ Users table created")
        
        # Create jobs table
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
        print("✅ Jobs table created")
        
        # Create job runs table
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
        print("✅ Job runs table created")
        
        # Create file storage table
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
        print("✅ File storage table created")
        
        # Create indexes for performance
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_jobs_user_id ON jobs(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status)",
            "CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs(created_at)",
            "CREATE INDEX IF NOT EXISTS idx_job_runs_job_id ON job_runs(job_id)",
            "CREATE INDEX IF NOT EXISTS idx_job_runs_status ON job_runs(status)",
            "CREATE INDEX IF NOT EXISTS idx_job_runs_run_id ON job_runs(run_id)",
            "CREATE INDEX IF NOT EXISTS idx_file_storage_user_id ON file_storage(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_file_storage_job_id ON file_storage(job_id)",
            "CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)",
            "CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)",
            "CREATE INDEX IF NOT EXISTS idx_users_active ON users(is_active)"
        ]
        
        for index_sql in indexes:
            await conn.execute(index_sql)
        
        print("✅ Database indexes created")
        
        # Create update trigger for updated_at columns
        await conn.execute("""
            CREATE OR REPLACE FUNCTION update_updated_at_column()
            RETURNS TRIGGER AS $$
            BEGIN
                NEW.updated_at = CURRENT_TIMESTAMP;
                RETURN NEW;
            END;
            $$ language 'plpgsql';
        """)
        
        # Apply trigger to users table
        await conn.execute("""
            DROP TRIGGER IF EXISTS update_users_updated_at ON users;
            CREATE TRIGGER update_users_updated_at
                BEFORE UPDATE ON users
                FOR EACH ROW
                EXECUTE FUNCTION update_updated_at_column();
        """)
        
        # Apply trigger to jobs table
        await conn.execute("""
            DROP TRIGGER IF EXISTS update_jobs_updated_at ON jobs;
            CREATE TRIGGER update_jobs_updated_at
                BEFORE UPDATE ON jobs
                FOR EACH ROW
                EXECUTE FUNCTION update_updated_at_column();
        """)
        
        print("✅ Update triggers created")
        
        await conn.close()
        print("✅ Database setup completed successfully!")
        
    except Exception as e:
        print(f"❌ Database setup failed: {str(e)}")
        sys.exit(1)

async def create_sample_data():
    """Create sample users and jobs for testing"""
    
    try:
        conn = await asyncpg.connect(DATABASE_URL)
        
        # Create sample user (password: 'password123')
        from passlib.context import CryptContext
        pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        password_hash = pwd_context.hash("password123")
        
        # Check if sample user already exists
        existing_user = await conn.fetchrow("SELECT id FROM users WHERE username = 'demo_user'")
        
        if not existing_user:
            user_id = await conn.fetchval("""
                INSERT INTO users (username, email, password_hash, full_name)
                VALUES ($1, $2, $3, $4)
                RETURNING id
            """, "demo_user", "demo@example.com", password_hash, "Demo User")
            
            print(f"✅ Sample user created (ID: {user_id})")
            print("   Username: demo_user")
            print("   Password: password123")
            print("   Email: demo@example.com")
            
            # Create sample job
            job_id = await conn.fetchval("""
                INSERT INTO jobs (user_id, name, description, status, config)
                VALUES ($1, $2, $3, $4, $5)
                RETURNING id
            """, user_id, "Sample Physics Problem", 
                "Discover the equation: y = 2.5*cos(x3) + x0^2 - 0.5", 
                "created",
                '{"model_config": {"binary_operators": ["+", "-", "*", "/"], "unary_operators": ["cos", "sin", "exp", "log"], "niterations": 40}}'
            )
            
            print(f"✅ Sample job created (ID: {job_id})")
        else:
            print("✅ Sample user already exists")
        
        await conn.close()
        
    except Exception as e:
        print(f"❌ Sample data creation failed: {str(e)}")

async def check_database_connection():
    """Check if database connection is working"""
    try:
        conn = await asyncpg.connect(DATABASE_URL)
        
        # Test query
        result = await conn.fetchval("SELECT version()")
        print(f"✅ Database connection successful")
        print(f"   PostgreSQL version: {result}")
        
        # Check tables
        tables = await conn.fetch("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
        """)
        
        table_names = [table['table_name'] for table in tables]
        expected_tables = ['users', 'jobs', 'job_runs', 'file_storage']
        
        print(f"   Tables found: {table_names}")
        
        missing_tables = [t for t in expected_tables if t not in table_names]
        if missing_tables:
            print(f"   ⚠️  Missing tables: {missing_tables}")
        else:
            print("   ✅ All required tables present")
        
        await conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Database connection failed: {str(e)}")
        return False

async def show_database_stats():
    """Show database statistics"""
    try:
        conn = await asyncpg.connect(DATABASE_URL)
        
        # Count records in each table
        tables = ['users', 'jobs', 'job_runs', 'file_storage']
        
        print("\n📊 Database Statistics:")
        print("-" * 30)
        
        for table in tables:
            count = await conn.fetchval(f"SELECT COUNT(*) FROM {table}")
            print(f"   {table.capitalize()}: {count} records")
        
        # Show recent activity
        recent_jobs = await conn.fetch("""
            SELECT j.name, j.status, j.created_at, u.username
            FROM jobs j
            JOIN users u ON j.user_id = u.id
            ORDER BY j.created_at DESC
            LIMIT 5
        """)
        
        if recent_jobs:
            print("\n🕒 Recent Jobs:")
            print("-" * 30)
            for job in recent_jobs:
                print(f"   {job['name']} ({job['status']}) - {job['username']} - {job['created_at'].strftime('%Y-%m-%d %H:%M')}")
        
        await conn.close()
        
    except Exception as e:
        print(f"❌ Failed to get database stats: {str(e)}")

def print_usage():
    """Print usage information"""
    print("""
PySR Database Setup Script

Usage:
    python create_tables.py [options]

Options:
    --reset         Drop existing tables before creating new ones
    --sample-data   Create sample user and job data
    --check         Check database connection and table status
    --stats         Show database statistics
    --help          Show this help message

Examples:
    python create_tables.py                    # Create tables
    python create_tables.py --reset            # Reset and recreate all tables
    python create_tables.py --sample-data      # Add sample data
    python create_tables.py --check            # Check database status
    python create_tables.py --stats            # Show database statistics

Environment Variables:
    DATABASE_URL    PostgreSQL connection string
                    Default: postgresql://postgres:password@localhost:5432/pysr_db
    """)

async def main():
    """Main function"""
    
    if "--help" in sys.argv:
        print_usage()
        return
    
    print("🚀 PySR Database Setup")
    print("=" * 50)
    print(f"Database URL: {DATABASE_URL}")
    print()
    
    # Check database connection first
    if not await check_database_connection():
        print("\n❌ Cannot connect to database. Please check:")
        print("   1. PostgreSQL is running")
        print("   2. Database exists")
        print("   3. Connection string is correct")
        print("   4. User has proper permissions")
        return
    
    if "--check" in sys.argv:
        return
    
    if "--stats" in sys.argv:
        await show_database_stats()
        return
    
    # Create schema
    await create_database_schema()
    
    # Create sample data if requested
    if "--sample-data" in sys.argv:
        print("\n🎭 Creating sample data...")
        await create_sample_data()
    
    # Show final stats
    await show_database_stats()
    
    print("\n🎉 Database setup complete!")
    print("\nNext steps:")
    print("1. Start the MCP server: python mcp_server/pysr_mcp_server.py")
    print("2. Start the app server: python app_server/main.py")
    print("3. Start the Streamlit app: streamlit run streamlit_app/app.py")

if __name__ == "__main__":
    asyncio.run(main())
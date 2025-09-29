#!/usr/bin/env python3
"""
Database Fix Script for PySR System
Handles database cleanup and recreation to fix schema issues
"""

import asyncio
import asyncpg
import os
import sys
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:amit1199@localhost:5432/postgres")

async def check_database_connection():
    """Test database connection"""
    try:
        conn = await asyncpg.connect(DATABASE_URL)
        version = await conn.fetchval("SELECT version()")
        await conn.close()
        print(f"✅ Database connection successful")
        print(f"   PostgreSQL version: {version}")
        return True
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False

async def drop_all_tables():
    """Drop all existing tables to start fresh"""
    try:
        conn = await asyncpg.connect(DATABASE_URL)
        
        print("🗑️  Dropping existing tables...")
        
        # Drop tables in reverse dependency order
        tables_to_drop = [
            "file_storage",
            "job_runs", 
            "jobs",
            "users"
        ]
        
        for table in tables_to_drop:
            try:
                await conn.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
                print(f"   ✅ Dropped table: {table}")
            except Exception as e:
                print(f"   ⚠️  Warning dropping {table}: {e}")
        
        # Drop functions
        try:
            await conn.execute("DROP FUNCTION IF EXISTS update_updated_at_column() CASCADE")
            print("   ✅ Dropped update function")
        except Exception as e:
            print(f"   ⚠️  Warning dropping function: {e}")
        
        await conn.close()
        print("✅ All tables dropped successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error dropping tables: {e}")
        return False

async def create_tables_fresh():
    """Create all tables from scratch"""
    try:
        conn = await asyncpg.connect(DATABASE_URL)
        
        print("🏗️  Creating tables...")
        
        # 1. Users table (no dependencies)
        await conn.execute("""
            CREATE TABLE users (
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
        print("   ✅ Created users table")
        
        # 2. Jobs table (depends on users)
        await conn.execute("""
            CREATE TABLE jobs (
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
        print("   ✅ Created jobs table")
        
        # 3. Job runs table (depends on jobs)
        await conn.execute("""
            CREATE TABLE job_runs (
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
        print("   ✅ Created job_runs table")
        
        # 4. File storage table (depends on users and jobs)
        await conn.execute("""
            CREATE TABLE file_storage (
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
        print("   ✅ Created file_storage table")
        
        # Create indexes
        indexes = [
            "CREATE INDEX idx_users_username ON users(username)",
            "CREATE INDEX idx_users_email ON users(email)",
            "CREATE INDEX idx_users_active ON users(is_active)",
            "CREATE INDEX idx_jobs_user_id ON jobs(user_id)",
            "CREATE INDEX idx_jobs_status ON jobs(status)",
            "CREATE INDEX idx_jobs_created_at ON jobs(created_at)",
            "CREATE INDEX idx_job_runs_job_id ON job_runs(job_id)",
            "CREATE INDEX idx_job_runs_status ON job_runs(status)",
            "CREATE INDEX idx_job_runs_run_id ON job_runs(run_id)",
            "CREATE INDEX idx_file_storage_user_id ON file_storage(user_id)",
            "CREATE INDEX idx_file_storage_job_id ON file_storage(job_id)"
        ]
        
        print("🔗 Creating indexes...")
        for index_sql in indexes:
            await conn.execute(index_sql)
        print("   ✅ All indexes created")
        
        # Create update trigger function
        await conn.execute("""
            CREATE OR REPLACE FUNCTION update_updated_at_column()
            RETURNS TRIGGER AS $$
            BEGIN
                NEW.updated_at = CURRENT_TIMESTAMP;
                RETURN NEW;
            END;
            $$ language 'plpgsql';
        """)
        print("   ✅ Created update function")
        
        # Apply triggers
        await conn.execute("""
            CREATE TRIGGER update_users_updated_at
                BEFORE UPDATE ON users
                FOR EACH ROW
                EXECUTE FUNCTION update_updated_at_column();
        """)
        
        await conn.execute("""
            CREATE TRIGGER update_jobs_updated_at
                BEFORE UPDATE ON jobs
                FOR EACH ROW
                EXECUTE FUNCTION update_updated_at_column();
        """)
        print("   ✅ Created triggers")
        
        await conn.close()
        print("✅ Database schema created successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error creating tables: {e}")
        return False

async def create_sample_user():
    """Create a sample user for testing"""
    try:
        from passlib.context import CryptContext
        
        conn = await asyncpg.connect(DATABASE_URL)
        
        # Check if demo user already exists
        existing_user = await conn.fetchrow("SELECT id FROM users WHERE username = 'demo_user'")
        
        if existing_user:
            print("✅ Demo user already exists")
            await conn.close()
            return True
        
        # Create demo user
        pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        password_hash = pwd_context.hash("password123")
        
        user_id = await conn.fetchval("""
            INSERT INTO users (username, email, password_hash, full_name)
            VALUES ($1, $2, $3, $4)
            RETURNING id
        """, "demo_user", "demo@example.com", password_hash, "Demo User")
        
        print(f"✅ Created demo user (ID: {user_id})")
        print("   Username: demo_user")
        print("   Password: password123")
        print("   Email: demo@example.com")
        
        await conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Error creating sample user: {e}")
        return False

async def verify_schema():
    """Verify that all tables and indexes exist"""
    try:
        conn = await asyncpg.connect(DATABASE_URL)
        
        print("🔍 Verifying database schema...")
        
        # Check tables
        tables = await conn.fetch("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            ORDER BY table_name
        """)
        
        table_names = [table['table_name'] for table in tables]
        expected_tables = ['users', 'jobs', 'job_runs', 'file_storage']
        
        print(f"   Tables found: {table_names}")
        
        missing_tables = [t for t in expected_tables if t not in table_names]
        if missing_tables:
            print(f"   ❌ Missing tables: {missing_tables}")
            return False
        else:
            print("   ✅ All required tables present")
        
        # Check indexes
        indexes = await conn.fetch("""
            SELECT indexname 
            FROM pg_indexes 
            WHERE schemaname = 'public'
            ORDER BY indexname
        """)
        
        index_names = [idx['indexname'] for idx in indexes]
        print(f"   Indexes found: {len(index_names)} indexes")
        
        # Count records in each table
        for table in expected_tables:
            count = await conn.fetchval(f"SELECT COUNT(*) FROM {table}")
            print(f"   {table}: {count} records")
        
        await conn.close()
        print("✅ Schema verification completed")
        return True
        
    except Exception as e:
        print(f"❌ Error verifying schema: {e}")
        return False

async def main():
    """Main function"""
    print("🛠️  PySR Database Fix Script")
    print("=" * 50)
    
    # Check connection first
    if not await check_database_connection():
        print("\n❌ Cannot connect to database. Please check:")
        print("   1. PostgreSQL is running")
        print("   2. Database exists") 
        print("   3. Connection string is correct")
        print("   4. User has proper permissions")
        return
    
    # Ask user what to do
    print("\nWhat would you like to do?")
    print("1. Drop all tables and recreate (DESTRUCTIVE)")
    print("2. Verify current schema")
    print("3. Create sample user only")
    print("4. Exit")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "1":
        print("\n⚠️  WARNING: This will delete ALL data in the database!")
        confirm = input("Type 'yes' to confirm: ").strip().lower()
        
        if confirm == "yes":
            print("\n🔄 Starting database recreation...")
            
            if await drop_all_tables():
                if await create_tables_fresh():
                    if await create_sample_user():
                        await verify_schema()
                        print("\n🎉 Database recreation completed successfully!")
                        print("\nYou can now start the app server:")
                        print("   python app_server.py")
                    else:
                        print("\n❌ Failed to create sample user")
                else:
                    print("\n❌ Failed to create tables")
            else:
                print("\n❌ Failed to drop tables")
        else:
            print("❌ Operation cancelled")
    
    elif choice == "2":
        print("\n🔍 Verifying database schema...")
        await verify_schema()
    
    elif choice == "3":
        print("\n👤 Creating sample user...")
        await create_sample_user()
    
    elif choice == "4":
        print("👋 Goodbye!")
    
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    asyncio.run(main())
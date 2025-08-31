# Kalaam AI Backend

A FastAPI-based backend server for the Kalaam AI project with PostgreSQL database integration.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Python Version Management](#python-version-management)
3. [Project Setup](#project-setup)
4. [Database Setup](#database-setup)
5. [Running the Application](#running-the-application)
6. [API Documentation](#api-documentation)
7. [Project Structure](#project-structure)
8. [Troubleshooting](#troubleshooting)

## Prerequisites

### Ubuntu/Linux

```bash
# Update package list
sudo apt update

# Install required packages
sudo apt install -y make build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev \
libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python3-openssl \
git postgresql postgresql-contrib
```

### Windows

1. Install **Git for Windows**: https://git-scm.com/download/win
2. Install **PostgreSQL**: https://www.postgresql.org/download/windows/
3. Install **Windows Terminal** (recommended): Microsoft Store
4. Use **PowerShell** or **Git Bash** for commands

## Python Version Management

This project uses Python 3.11.9. We recommend using `pyenv` for Python version management.

### Installing pyenv

#### Ubuntu/Linux

```bash
# Install pyenv
curl https://pyenv.run | bash

# Add to your shell profile (~/.bashrc, ~/.zshrc, etc.)
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc

# Restart your terminal or reload your shell
source ~/.bashrc
```

#### Windows

1. Install **pyenv-win**: https://github.com/pyenv-win/pyenv-win
2. Or use **Python.org** installer directly: https://www.python.org/downloads/release/python-3119/

### Installing Python 3.11.9

#### Ubuntu/Linux

```bash
# Install Python 3.11.9
pyenv install 3.11.9

# Verify installation
pyenv versions
```

#### Windows (with pyenv-win)

```powershell
# Install Python 3.11.9
pyenv install 3.11.9

# List installed versions
pyenv versions
```

## Project Setup

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd kalaam-ai-backend
```

### 2. Set Python Version

The project includes a `.python-version` file that automatically sets Python 3.11.9 when using pyenv.

```bash
# Verify Python version
python --version
# Should output: Python 3.11.9
```

### 3. Create Virtual Environment

#### Ubuntu/Linux

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Verify activation (you should see (.venv) in your prompt)
which python
```

#### Windows (PowerShell)

```powershell
# Create virtual environment
python -m venv .venv

# Activate virtual environment
.venv\Scripts\Activate
OR
.venv\Scripts\Activate.ps1

# If you get execution policy error, run:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Verify activation (you should see (.venv) in your prompt)
where python
```

#### Windows (Command Prompt)

```cmd
# Create virtual environment
python -m venv .venv

# Activate virtual environment
.venv\Scripts\activate.bat
```

### 4. Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install required packages
After going to the backend directory
pip install -r requirements.txt

```

### 5. Create Requirements File
EveryTime you install some new package run
```bash
# Generate requirements.txt
pip freeze > requirements.txt
```

## Database Setup

### 1. Start PostgreSQL Service

#### Ubuntu/Linux

```bash
# Start PostgreSQL service
sudo systemctl start postgresql

# Enable auto-start on boot
sudo systemctl enable postgresql

# Check status
sudo systemctl status postgresql
```

#### Windows

1. PostgreSQL should start automatically after installation
2. Or start it from **Services** (`services.msc`)
3. Or use **pgAdmin** to manage your database

### 2. Create Database and User (Optional)
If you have already set this up no need to go through, or if you are more comfortable with watching a tutorial (recommended) do that.

#### Ubuntu/Linux

```bash
# Switch to postgres user
sudo -i -u postgres

# Create database
createdb kalaam_ai_db

# Create user (optional)
createuser --interactive kalaam_user

# Access PostgreSQL prompt
psql
```

In PostgreSQL prompt:
```sql
-- Grant privileges to user
GRANT ALL PRIVILEGES ON DATABASE kalaam_ai_db TO kalaam_user;

-- Exit
\q
```

#### Windows

Use **pgAdmin** or **psql** command line:
```sql
-- Connect to PostgreSQL
psql -U postgres

-- Create database
CREATE DATABASE kalaam_ai_db;

-- Create user
CREATE USER kalaam_user WITH PASSWORD 'your_password';

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE kalaam_ai_db TO kalaam_user;
```

### 3. Configure Environment Variables

Create a `.env` file in the project root:

```bash
# Copy example environment file
cp .example.env .env
```

Edit `.env` file:
```env
# Database Configuration
DATABASE_URL=postgresql://postgres:your_password@localhost:5432/<name of Database you are using>


**Replace `your_password` with your actual PostgreSQL password.**

### 4. Test Database Connection

```bash
# Test connection with Python
python -c "
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv('DATABASE_URL')
engine = create_engine(DATABASE_URL)

try:
    with engine.connect() as conn:
        print('✅ Database connection successful!')
except Exception as e:
    print(f'❌ Database connection failed: {e}')
"
```

## Running the Application

### 1. Start the Development Server

#### Using uvicorn (Recommended)

```bash
# Make sure you're in the project directory and virtual environment is activated
cd kalaam-ai-backend
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\Activate.ps1  # Windows PowerShell

# Start the server
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

#### Alternative method

```bash
python -m uvicorn src.main:app --reload
```

### 2. Verify the Server

Open your browser and go to:
- **API Base URL**: http://localhost:8000/
- **API Documentation**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc

You should see:
- "Hello world" message at the base URL
- Interactive API documentation at `/docs`
- Database connection message in the terminal

## API Documentation

### Available Endpoints

- `GET /` - Returns "Hello world" message
- `GET /docs` - Interactive API documentation (Swagger UI)
- `GET /redoc` - Alternative API documentation

### Example Response

```json
{
    "message": "Hello world"
}
```

## Project Structure

```
kalaam-ai-backend/
├── .env                    # Environment variables (create this)
├── .example.env           # Environment template
├── .python-version        # Python version specification
├── requirements.txt       # Python dependencies
├── README.md             # This documentation
└── src/
    ├── main.py           # Application entry point
    ├── api.py            # API route registration
    ├── logging.py        # Logging configuration
    ├── rate_limiter.py   # Rate limiting utilities
    └── config/
        └── db.py         # Database configuration
    └── controllers/      # API controllers (future)
    └── entities/         # Database entities (future)
    └── models/          # Data models (future)
    └── services/        # Business logic (future)
```

## Troubleshooting

### Common Issues

#### 1. `ModuleNotFoundError: No module named 'src'`

**Solution**: Make sure you're running uvicorn from the project root directory:
```bash
cd kalaam-ai-backend
uvicorn src.main:app --reload
```

#### 2. Database Connection Failed

**Solutions**:
- Check if PostgreSQL is running: `sudo systemctl status postgresql` (Linux)
- Verify your `.env` file has the correct `DATABASE_URL`
- Test connection string manually
- Check PostgreSQL logs: `sudo journalctl -u postgresql`

#### 3. Virtual Environment Issues

**Solutions**:
```bash
# Deactivate current environment
deactivate

# Remove and recreate
rm -rf .venv
python -m venv .venv
source .venv/bin/activate  # Linux
.venv\Scripts\Activate.ps1  # Windows

# Reinstall dependencies
pip install -r requirements.txt
```

#### 4. Python Version Issues

**Solutions**:
```bash
# Check current Python version
python --version

# With pyenv, set correct version
pyenv local 3.11.9

# Recreate virtual environment with correct Python version
rm -rf .venv
python -m venv .venv
```

#### 5. Permission Issues (Ubuntu/Linux)

```bash
# PostgreSQL connection issues
sudo -u postgres psql -c "ALTER USER postgres PASSWORD 'newpassword';"

# File permission issues
sudo chown -R $USER:$USER /path/to/project
```

### Development Tips

1. **Always activate your virtual environment** before working on the project
2. **Keep your `.env` file secure** - never commit it to version control
3. **Use `--reload` flag** during development for auto-restart on file changes
4. **Check logs** in the terminal for detailed error messages
5. **Use the interactive API docs** at `/docs` for testing endpoints

### Getting Help

If you encounter issues:
1. Check the terminal output for error messages
2. Verify all prerequisites are installed
3. Ensure PostgreSQL is running and accessible
4. Check your `.env` file configuration
5. Try the troubleshooting steps above

---

## Quick Start Summary

For experienced developers, here's the quick setup:

```bash
# 1. Clone and navigate
git clone <repo-url> && cd kalaam-ai-backend

# 2. Install Python 3.11.9 (if using pyenv)
pyenv install 3.11.9 && pyenv local 3.11.9

# 3. Create and activate virtual environment
python -m venv .venv && source .venv/bin/activate

# 4. Install dependencies
pip install fastapi uvicorn sqlalchemy python-dotenv psycopg2-binary

# 5. Setup database and create .env file
cp .example.env .env
# Edit .env with your DATABASE_URL

# 6. Start PostgreSQL
sudo systemctl start postgresql  # Linux

# 7. Run the server
uvicorn src.main:app --reload
```

Visit http://localhost:8000/ to see your API running!

import os
from datetime import datetime, timedelta
from typing import Optional, List

from fastapi import FastAPI, HTTPException, Depends, status, UploadFile, File, Form
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr
from passlib.context import CryptContext
from jose import JWTError, jwt
import asyncpg
from dotenv import load_dotenv

load_dotenv()

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://inventory_ihpg_user:EKkxYBPqllVfkTkIDKYRzGZKDX5Vw2ek@dpg-d16jkimmcj7s73c7li80-a/inventory_ihpg")

# Security configuration
SECRET_KEY = os.getenv("SECRET_KEY", "09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Initialize FastAPI app
app = FastAPI(title="Professional Reporting System",
              description="A modern reporting platform with role-based access control",
              version="1.0.0")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Models
class UserBase(BaseModel):
    email: EmailStr
    full_name: str

class UserCreate(UserBase):
    password: str

class User(UserBase):
    id: int
    is_active: bool
    role: str

    class Config:
        orm_mode = True

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: Optional[str] = None

class ReportBase(BaseModel):
    title: str
    description: str
    status: str = "pending"

class ReportCreate(ReportBase):
    pass

class Report(ReportBase):
    id: int
    created_at: datetime
    updated_at: datetime
    owner_id: int
    attachments: List[str] = []

    class Config:
        orm_mode = True

# Database connection pool
async def get_db():
    conn = await asyncpg.connect(DATABASE_URL)
    try:
        yield conn
    finally:
        await conn.close()

# Utility functions
def verify_password(plain_password: str, hashed_password: str):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str):
    return pwd_context.hash(password)

async def authenticate_user(email: str, password: str, db):
    user = await db.fetchrow("SELECT * FROM users WHERE email = $1", email)
    if not user:
        return False
    if not verify_password(password, user["password"]):
        return False
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme), db = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
        token_data = TokenData(email=email)
    except JWTError:
        raise credentials_exception
    
    user = await db.fetchrow("SELECT * FROM users WHERE email = $1", token_data.email)
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(current_user: User = Depends(get_current_user)):
    if not current_user["is_active"]:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

async def check_admin(current_user: User = Depends(get_current_active_user)):
    if current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user

async def check_staff(current_user: User = Depends(get_current_active_user)):
    if current_user["role"] not in ["admin", "staff"]:
        raise HTTPException(status_code=403, detail="Staff access required")
    return current_user

# Initialize database tables
async def initialize_database():
    conn = await asyncpg.connect(DATABASE_URL)
    try:
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                email VARCHAR(255) UNIQUE NOT NULL,
                password VARCHAR(255) NOT NULL,
                full_name VARCHAR(255) NOT NULL,
                is_active BOOLEAN DEFAULT TRUE,
                role VARCHAR(50) DEFAULT 'staff',
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            )
        ''')

        await conn.execute('''
            CREATE TABLE IF NOT EXISTS reports (
                id SERIAL PRIMARY KEY,
                title VARCHAR(255) NOT NULL,
                description TEXT NOT NULL,
                status VARCHAR(50) DEFAULT 'pending',
                owner_id INTEGER REFERENCES users(id),
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            )
        ''')

        await conn.execute(''
            CREATE TABLE IF NOT EXISTS attachments (
                id SERIAL PRIMARY KEY,
                report_id INTEGER REFERENCES reports(id),
                file_path VARCHAR(255) NOT NULL,
                original_filename VARCHAR(255) NOT NULL,
                uploaded_at TIMESTAMP DEFAULT NOW()
            )
        ''')

        # Create initial admin user if not exists
        admin_email = "admin@reporting.com"
        admin_exists = await conn.fetchrow("SELECT * FROM users WHERE email = $1", admin_email)
        if not admin_exists:
            hashed_password = get_password_hash("Admin@123")
            await conn.execute('''
                INSERT INTO users (email, password, full_name, role)
                VALUES ($1, $2, $3, $4)
            ''', admin_email, hashed_password, "Admin User", "admin")
            
    finally:
        await conn.close()

# Create attachments directory if it doesn't exist
os.makedirs("attachments", exist_ok=True)

# Serve static files (attachments)
app.mount("/attachments", StaticFiles(directory="attachments"), name="attachments")

# Startup event
@app.on_event("startup")
async def startup_event():
    await initialize_database()

# Authentication routes
@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db = Depends(get_db)):
    user = await authenticate_user(form_data.username, form_data.password, db)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["email"]}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/register", response_model=User)
async def register_user(user: UserCreate, db = Depends(get_db), admin: User = Depends(check_admin)):
    existing_user = await db.fetchrow("SELECT * FROM users WHERE email = $1", user.email)
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    hashed_password = get_password_hash(user.password)
    new_user = await db.fetchrow('''
        INSERT INTO users (email, password, full_name, role)
        VALUES ($1, $2, $3, 'staff')
        RETURNING id, email, full_name, is_active, role
    ''', user.email, hashed_password, user.full_name)
    
    return new_user

# User routes
@app.get("/users/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    return current_user

@app.get("/users", response_model=List[User])
async def read_users(db = Depends(get_db), admin: User = Depends(check_admin)):
    users = await db.fetch("SELECT id, email, full_name, is_active, role FROM users")
    return users

# Report routes
@app.post("/reports", response_model=Report)
async def create_report(
    title: str = Form(...),
    description: str = Form(...),
    files: List[UploadFile] = File(None),
    db = Depends(get_db),
    current_user: User = Depends(check_staff)
):
    # Create report
    report = await db.fetchrow('''
        INSERT INTO reports (title, description, owner_id)
        VALUES ($1, $2, $3)
        RETURNING id, title, description, status, owner_id, created_at, updated_at
    ''', title, description, current_user["id"])
    
    # Handle file attachments
    attachment_paths = []
    if files:
        for file in files:
            file_path = f"attachments/{report['id']}_{file.filename}"
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            await db.execute('''
                INSERT INTO attachments (report_id, file_path, original_filename)
                VALUES ($1, $2, $3)
            ''', report["id"], file_path, file.filename)
            
            attachment_paths.append(file.filename)
    
    return {**report, "attachments": attachment_paths}

@app.get("/reports", response_model=List[Report])
async def read_reports(
    skip: int = 0,
    limit: int = 100,
    db = Depends(get_db),
    current_user: User = Depends(check_staff)
):
    if current_user["role"] == "admin":
        reports = await db.fetch('''
            SELECT r.*, array_agg(a.original_filename) as attachments
            FROM reports r
            LEFT JOIN attachments a ON r.id = a.report_id
            GROUP BY r.id
            ORDER BY r.created_at DESC
            OFFSET $1 LIMIT $2
        ''', skip, limit)
    else:
        reports = await db.fetch('''
            SELECT r.*, array_agg(a.original_filename) as attachments
            FROM reports r
            LEFT JOIN attachments a ON r.id = a.report_id
            WHERE r.owner_id = $3
            GROUP BY r.id
            ORDER BY r.created_at DESC
            OFFSET $1 LIMIT $2
        ''', skip, limit, current_user["id"])
    
    return reports

@app.get("/reports/{report_id}", response_model=Report)
async def read_report(report_id: int, db = Depends(get_db), current_user: User = Depends(check_staff)):
    report = await db.fetchrow('''
        SELECT r.*, array_agg(a.original_filename) as attachments
        FROM reports r
        LEFT JOIN attachments a ON r.id = a.report_id
        WHERE r.id = $1
        GROUP BY r.id
    ''', report_id)
    
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    
    if current_user["role"] != "admin" and report["owner_id"] != current_user["id"]:
        raise HTTPException(status_code=403, detail="Not authorized to access this report")
    
    return report

@app.put("/reports/{report_id}", response_model=Report)
async def update_report(
    report_id: int,
    title: str = Form(None),
    description: str = Form(None),
    status: str = Form(None),
    files: List[UploadFile] = File(None),
    db = Depends(get_db),
    current_user: User = Depends(check_staff)
):
    # Check if report exists and user has permission
    report = await db.fetchrow("SELECT * FROM reports WHERE id = $1", report_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    
    if current_user["role"] != "admin" and report["owner_id"] != current_user["id"]:
        raise HTTPException(status_code=403, detail="Not authorized to update this report")
    
    # Build update query
    update_fields = {}
    if title is not None:
        update_fields["title"] = title
    if description is not None:
        update_fields["description"] = description
    if status is not None and current_user["role"] == "admin":
        update_fields["status"] = status
    
    if update_fields:
        set_clause = ", ".join([f"{field} = ${i+2}" for i, field in enumerate(update_fields.keys())])
        query = f"UPDATE reports SET {set_clause}, updated_at = NOW() WHERE id = $1 RETURNING *"
        report = await db.fetchrow(query, report_id, *update_fields.values())
    
    # Handle file attachments
    attachment_paths = []
    if files:
        for file in files:
            file_path = f"attachments/{report_id}_{file.filename}"
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            await db.execute('''
                INSERT INTO attachments (report_id, file_path, original_filename)
                VALUES ($1, $2, $3)
            ''', report_id, file_path, file.filename)
            
            attachment_paths.append(file.filename)
    
    # Get all attachments for the report
    attachments = await db.fetch("SELECT original_filename FROM attachments WHERE report_id = $1", report_id)
    attachment_names = [a["original_filename"] for a in attachments]
    
    return {**report, "attachments": attachment_names}

@app.delete("/reports/{report_id}")
async def delete_report(report_id: int, db = Depends(get_db), current_user: User = Depends(check_staff)):
    report = await db.fetchrow("SELECT * FROM reports WHERE id = $1", report_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    
    if current_user["role"] != "admin" and report["owner_id"] != current_user["id"]:
        raise HTTPException(status_code=403, detail="Not authorized to delete this report")
    
    # Delete attachments first
    attachments = await db.fetch("SELECT file_path FROM attachments WHERE report_id = $1", report_id)
    for attachment in attachments:
        try:
            os.remove(attachment["file_path"])
        except:
            pass
    
    await db.execute("DELETE FROM attachments WHERE report_id = $1", report_id)
    await db.execute("DELETE FROM reports WHERE id = $1", report_id)
    
    return {"message": "Report deleted successfully"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

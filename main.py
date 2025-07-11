from fastapi import FastAPI, HTTPException, Depends, status, UploadFile, File, Form
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, EmailStr
from datetime import datetime, timedelta
from typing import Optional, List
from jose import JWTError, jwt
from passlib.context import CryptContext
import psycopg2
from psycopg2.extras import RealDictCursor
import os
from uuid import uuid4
import shutil

# Configuration
SECRET_KEY = "your-secret-key-here-keep-it-secure-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Database connection
DATABASE_URL = "postgresql://inventory_ihpg_user:EKkxYBPqllVfkTkIDKYRzGZKDX5Vw2ek@dpg-d16jkimmcj7s73c7li80-a/inventory_ihpg"

def get_db():
    conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
    try:
        yield conn
    finally:
        conn.close()

# Models
class UserBase(BaseModel):
    email: EmailStr

class UserCreate(UserBase):
    password: str
    full_name: str
    role: str = "staff"

class User(UserBase):
    id: int
    full_name: str
    role: str
    is_active: bool

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: Optional[str] = None
    role: Optional[str] = None

class ReportBase(BaseModel):
    title: str
    description: str

class ReportCreate(ReportBase):
    pass

class Report(ReportBase):
    id: int
    created_at: datetime
    updated_at: datetime
    status: str
    user_id: int
    attachments: List[str] = []

class ReportUpdate(BaseModel):
    title: Optional[str]
    description: Optional[str]
    status: Optional[str]

# Auth setup
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

app = FastAPI(title="Professional Reporting System",
             description="A modern reporting platform with authentication",
             version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# Utility functions
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

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
        token_data = TokenData(email=email, role=payload.get("role"))
    except JWTError:
        raise credentials_exception
    
    cursor = db.cursor()
    cursor.execute("SELECT * FROM users WHERE email = %s", (token_data.email,))
    user = cursor.fetchone()
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(current_user: User = Depends(get_current_user)):
    if not current_user['is_active']:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

async def require_admin(current_user: User = Depends(get_current_active_user)):
    if current_user['role'] != 'admin':
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user

# Initialize database
def initialize_database():
    with psycopg2.connect(DATABASE_URL) as conn:
        with conn.cursor() as cursor:
            # Create users table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    email VARCHAR(255) UNIQUE NOT NULL,
                    full_name VARCHAR(255) NOT NULL,
                    hashed_password VARCHAR(255) NOT NULL,
                    role VARCHAR(50) NOT NULL DEFAULT 'staff',
                    is_active BOOLEAN NOT NULL DEFAULT TRUE,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create reports table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS reports (
                    id SERIAL PRIMARY KEY,
                    title VARCHAR(255) NOT NULL,
                    description TEXT NOT NULL,
                    status VARCHAR(50) NOT NULL DEFAULT 'pending',
                    user_id INTEGER REFERENCES users(id),
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create attachments table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS attachments (
                    id SERIAL PRIMARY KEY,
                    report_id INTEGER REFERENCES reports(id),
                    file_path VARCHAR(255) NOT NULL,
                    original_filename VARCHAR(255) NOT NULL,
                    uploaded_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Check if admin exists
            cursor.execute("SELECT * FROM users WHERE email = 'admin@example.com'")
            if not cursor.fetchone():
                hashed_password = get_password_hash("Admin@123")
                cursor.execute("""
                    INSERT INTO users (email, full_name, hashed_password, role)
                    VALUES (%s, %s, %s, %s)
                """, ("admin@example.com", "Admin User", hashed_password, "admin"))
            
            conn.commit()

# Initialize the database on startup
initialize_database()

# Routes
@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db = Depends(get_db)):
    cursor = db.cursor()
    cursor.execute("SELECT * FROM users WHERE email = %s", (form_data.username,))
    user = cursor.fetchone()
    
    if not user or not verify_password(form_data.password, user['hashed_password']):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user['email'], "role": user['role']}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/users/", response_model=User)
async def create_user(user: UserCreate, db = Depends(get_db), current_user: User = Depends(require_admin)):
    cursor = db.cursor()
    try:
        hashed_password = get_password_hash(user.password)
        cursor.execute("""
            INSERT INTO users (email, full_name, hashed_password, role)
            VALUES (%s, %s, %s, %s)
            RETURNING id, email, full_name, role, is_active
        """, (user.email, user.full_name, hashed_password, user.role))
        new_user = cursor.fetchone()
        db.commit()
        return new_user
    except psycopg2.IntegrityError:
        raise HTTPException(status_code=400, detail="Email already registered")

@app.get("/users/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    return current_user

@app.post("/reports/", response_model=Report)
async def create_report(
    title: str = Form(...),
    description: str = Form(...),
    files: List[UploadFile] = File([]),
    db = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    cursor = db.cursor()
    try:
        cursor.execute("""
            INSERT INTO reports (title, description, user_id)
            VALUES (%s, %s, %s)
            RETURNING id, title, description, status, user_id, created_at, updated_at
        """, (title, description, current_user['id']))
        report = cursor.fetchone()
        
        attachment_paths = []
        for file in files:
            if file.filename:
                file_ext = os.path.splitext(file.filename)[1]
                filename = f"{uuid4()}{file_ext}"
                file_path = os.path.join(UPLOAD_FOLDER, filename)
                
                with open(file_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                
                cursor.execute("""
                    INSERT INTO attachments (report_id, file_path, original_filename)
                    VALUES (%s, %s, %s)
                """, (report['id'], filename, file.filename))
                
                attachment_paths.append(filename)
        
        db.commit()
        report['attachments'] = attachment_paths
        return report
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/reports/", response_model=List[Report])
async def read_reports(
    skip: int = 0,
    limit: int = 100,
    db = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    cursor = db.cursor()
    if current_user['role'] == 'admin':
        cursor.execute("""
            SELECT r.*, array_agg(a.file_path) as attachments
            FROM reports r
            LEFT JOIN attachments a ON r.id = a.report_id
            GROUP BY r.id
            ORDER BY r.created_at DESC
            OFFSET %s LIMIT %s
        """, (skip, limit))
    else:
        cursor.execute("""
            SELECT r.*, array_agg(a.file_path) as attachments
            FROM reports r
            LEFT JOIN attachments a ON r.id = a.report_id
            WHERE r.user_id = %s
            GROUP BY r.id
            ORDER BY r.created_at DESC
            OFFSET %s LIMIT %s
        """, (current_user['id'], skip, limit))
    
    reports = cursor.fetchall()
    return reports

@app.get("/reports/{report_id}", response_model=Report)
async def read_report(
    report_id: int,
    db = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    cursor = db.cursor()
    cursor.execute("""
        SELECT r.*, array_agg(a.file_path) as attachments
        FROM reports r
        LEFT JOIN attachments a ON r.id = a.report_id
        WHERE r.id = %s
        GROUP BY r.id
    """, (report_id,))
    report = cursor.fetchone()
    
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    
    if current_user['role'] != 'admin' and report['user_id'] != current_user['id']:
        raise HTTPException(status_code=403, detail="Not authorized to access this report")
    
    return report

@app.put("/reports/{report_id}", response_model=Report)
async def update_report(
    report_id: int,
    report_update: ReportUpdate,
    db = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    cursor = db.cursor()
    
    # Check if report exists and user has permission
    cursor.execute("SELECT * FROM reports WHERE id = %s", (report_id,))
    report = cursor.fetchone()
    
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    
    if current_user['role'] != 'admin' and report['user_id'] != current_user['id']:
        raise HTTPException(status_code=403, detail="Not authorized to update this report")
    
    # Build update query
    update_fields = {}
    if report_update.title is not None:
        update_fields['title'] = report_update.title
    if report_update.description is not None:
        update_fields['description'] = report_update.description
    if report_update.status is not None and current_user['role'] == 'admin':
        update_fields['status'] = report_update.status
    
    if not update_fields:
        raise HTTPException(status_code=400, detail="No fields to update")
    
    update_fields['updated_at'] = datetime.utcnow()
    
    set_clause = ", ".join([f"{field} = %s" for field in update_fields])
    values = list(update_fields.values()) + [report_id]
    
    cursor.execute(f"""
        UPDATE reports
        SET {set_clause}
        WHERE id = %s
        RETURNING id, title, description, status, user_id, created_at, updated_at
    """, values)
    
    updated_report = cursor.fetchone()
    
    # Get attachments
    cursor.execute("SELECT file_path FROM attachments WHERE report_id = %s", (report_id,))
    attachments = [row['file_path'] for row in cursor.fetchall()]
    updated_report['attachments'] = attachments
    
    db.commit()
    return updated_report

@app.get("/users/", response_model=List[User])
async def read_users(
    skip: int = 0,
    limit: int = 100,
    db = Depends(get_db),
    current_user: User = Depends(require_admin)
):
    cursor = db.cursor()
    cursor.execute("""
        SELECT id, email, full_name, role, is_active
        FROM users
        ORDER BY created_at DESC
        OFFSET %s LIMIT %s
    """, (skip, limit))
    users = cursor.fetchall()
    return users

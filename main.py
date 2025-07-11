import os
from datetime import datetime, timedelta
from typing import Optional, List

from fastapi import FastAPI, HTTPException, Depends, status, UploadFile, File, Form
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, EmailStr
from passlib.context import CryptContext
from jose import JWTError, jwt
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from sqlalchemy.sql import func

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://inventory_ihpg_user:EKkxYBPqllVfkTkIDKYRzGZKDX5Vw2ek@dpg-d16jkimmcj7s73c7li80-a/inventory_ihpg")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Security configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# FastAPI app
app = FastAPI(title="Professional Reporting System API",
              description="A modern reporting system with authentication and file attachments",
              version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files for uploaded attachments
os.makedirs("attachments", exist_ok=True)
app.mount("/attachments", StaticFiles(directory="attachments"), name="attachments")

# Database models
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    full_name = Column(String)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    
    reports = relationship("Report", back_populates="owner")

class Report(Base):
    __tablename__ = "reports"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    description = Column(Text)
    status = Column(String, default="pending")  # pending, in_progress, resolved
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    owner_id = Column(Integer, ForeignKey("users.id"))
    
    owner = relationship("User", back_populates="reports")
    attachments = relationship("Attachment", back_populates="report")

class Attachment(Base):
    __tablename__ = "attachments"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String)
    filepath = Column(String)
    report_id = Column(Integer, ForeignKey("reports.id"))
    uploaded_at = Column(DateTime(timezone=True), server_default=func.now())
    
    report = relationship("Report", back_populates="attachments")

# Create tables
Base.metadata.create_all(bind=engine)

# Pydantic models
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: Optional[str] = None

class UserBase(BaseModel):
    email: EmailStr
    full_name: Optional[str] = None

class UserCreate(UserBase):
    password: str

class UserInDB(UserBase):
    id: int
    is_active: bool
    is_admin: bool

    class Config:
        orm_mode = True

class ReportBase(BaseModel):
    title: str
    description: Optional[str] = None

class ReportCreate(ReportBase):
    pass

class Report(ReportBase):
    id: int
    status: str
    created_at: datetime
    updated_at: datetime
    owner_id: int

    class Config:
        orm_mode = True

class ReportWithAttachments(Report):
    attachments: List[str] = []

class AttachmentInfo(BaseModel):
    filename: str
    filepath: str
    uploaded_at: datetime

# Utility functions
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def get_user(db: Session, email: str):
    return db.query(User).filter(User.email == email).first()

def authenticate_user(db: Session, email: str, password: str):
    user = get_user(db, email)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
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

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
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
    
    user = get_user(db, email=token_data.email)
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(current_user: UserInDB = Depends(get_current_user)):
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

async def get_current_admin_user(current_user: UserInDB = Depends(get_current_active_user)):
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user

# API endpoints
@app.post("/token", response_model=Token)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)
):
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/users/", response_model=UserInDB)
async def create_user(
    user: UserCreate, db: Session = Depends(get_db), current_user: UserInDB = Depends(get_current_admin_user)
):
    db_user = get_user(db, email=user.email)
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    hashed_password = get_password_hash(user.password)
    db_user = User(
        email=user.email,
        full_name=user.full_name,
        hashed_password=hashed_password,
        is_admin=False  # Only admins can create users, but new users are not admins by default
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

@app.get("/users/me/", response_model=UserInDB)
async def read_users_me(current_user: UserInDB = Depends(get_current_active_user)):
    return current_user

@app.post("/reports/", response_model=Report)
async def create_report(
    title: str = Form(...),
    description: str = Form(...),
    files: List[UploadFile] = File(None),
    db: Session = Depends(get_db),
    current_user: UserInDB = Depends(get_current_active_user)
):
    db_report = Report(
        title=title,
        description=description,
        owner_id=current_user.id
    )
    db.add(db_report)
    db.commit()
    db.refresh(db_report)
    
    if files:
        for file in files:
            # Save file to disk
            file_path = f"attachments/{db_report.id}_{file.filename}"
            with open(file_path, "wb") as buffer:
                buffer.write(await file.read())
            
            # Create attachment record
            db_attachment = Attachment(
                filename=file.filename,
                filepath=file_path,
                report_id=db_report.id
            )
            db.add(db_attachment)
        
        db.commit()
        db.refresh(db_report)
    
    return db_report

@app.get("/reports/", response_model=List[Report])
async def read_reports(
    skip: int = 0, limit: int = 100,
    db: Session = Depends(get_db),
    current_user: UserInDB = Depends(get_current_active_user)
):
    if current_user.is_admin:
        reports = db.query(Report).offset(skip).limit(limit).all()
    else:
        reports = db.query(Report).filter(Report.owner_id == current_user.id).offset(skip).limit(limit).all()
    return reports

@app.get("/reports/{report_id}", response_model=ReportWithAttachments)
async def read_report(
    report_id: int,
    db: Session = Depends(get_db),
    current_user: UserInDB = Depends(get_current_active_user)
):
    if current_user.is_admin:
        report = db.query(Report).filter(Report.id == report_id).first()
    else:
        report = db.query(Report).filter(Report.id == report_id, Report.owner_id == current_user.id).first()
    
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found")
    
    attachments = [attachment.filepath for attachment in report.attachments]
    
    report_with_attachments = ReportWithAttachments(
        **report.__dict__,
        attachments=attachments
    )
    
    return report_with_attachments

@app.put("/reports/{report_id}", response_model=Report)
async def update_report_status(
    report_id: int,
    status: str,
    db: Session = Depends(get_db),
    current_user: UserInDB = Depends(get_current_admin_user)
):
    db_report = db.query(Report).filter(Report.id == report_id).first()
    if db_report is None:
        raise HTTPException(status_code=404, detail="Report not found")
    
    db_report.status = status
    db.commit()
    db.refresh(db_report)
    return db_report

@app.get("/admin/reports", response_model=List[Report])
async def get_all_reports(
    skip: int = 0, limit: int = 100,
    db: Session = Depends(get_db),
    current_user: UserInDB = Depends(get_current_admin_user)
):
    reports = db.query(Report).offset(skip).limit(limit).all()
    return reports

@app.get("/admin/users", response_model=List[UserInDB])
async def get_all_users(
    skip: int = 0, limit: int = 100,
    db: Session = Depends(get_db),
    current_user: UserInDB = Depends(get_current_admin_user)
):
    users = db.query(User).offset(skip).limit(limit).all()
    return users

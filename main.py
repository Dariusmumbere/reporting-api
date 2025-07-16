from fastapi import FastAPI, HTTPException, Depends, status, UploadFile, File, Form
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, EmailStr, validator
from fastapi.responses import FileResponse
from typing import List, Optional, Dict
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
import os
import uuid
from fastapi import FastAPI
from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime, ForeignKey, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from sqlalchemy.sql import func
from sqlalchemy.exc import OperationalError, ProgrammingError

# Database configuration
DATABASE_URL = "postgresql://reporting_wlcd_user:sYC2WmtyjDCjyxvCPjoRNYAH4OCpVp6L@dpg-d1p23rc9c44c738581ig-a/reporting_wlcd"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Security configuration
SECRET_KEY = "your-secret-key-here"  # In production, use a proper secret key from environment variables
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

app = FastAPI()

# Create attachments directory if it doesn't exist
ATTACHMENTS_DIR = "attachments"
os.makedirs(ATTACHMENTS_DIR, exist_ok=True)

# Mount static files directory for attachments
app.mount("/attachments", StaticFiles(directory=ATTACHMENTS_DIR), name="attachments")

# Models
class Organization(Base):
    __tablename__ = "organizations"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    users = relationship("User", back_populates="organization")
    reports = relationship("Report", back_populates="organization")

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    role = Column(String, default="staff", nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_active = Column(DateTime(timezone=True))
    organization_id = Column(Integer, ForeignKey("organizations.id"))

    organization = relationship("Organization", back_populates="users")
    reports = relationship("Report", back_populates="author")
    attachments = relationship("Attachment", back_populates="uploader")

class Report(Base):
    __tablename__ = "reports"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    description = Column(String, nullable=False)
    category = Column(String, nullable=False)
    status = Column(String, default="pending", nullable=False)
    admin_comments = Column(String)
    author_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    organization_id = Column(Integer, ForeignKey("organizations.id"), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    author = relationship("User", back_populates="reports")
    organization = relationship("Organization", back_populates="reports")
    attachments = relationship("Attachment", back_populates="report")

class Attachment(Base):
    __tablename__ = "attachments"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    type = Column(String, nullable=False)
    size = Column(Integer, nullable=False)
    url = Column(String, nullable=False)
    report_id = Column(Integer, ForeignKey("reports.id"))
    uploader_id = Column(Integer, ForeignKey("users.id"))

    report = relationship("Report", back_populates="attachments")
    uploader = relationship("User", back_populates="attachments")

def check_and_create_tables():
    db = SessionLocal()
    try:
        # Check if users table exists and has organization_id column
        db.execute(text("SELECT 1 FROM users LIMIT 1"))
        
        try:
            # Check if organization_id column exists
            db.execute(text("SELECT organization_id FROM users LIMIT 1"))
        except (OperationalError, ProgrammingError):
            # Add the column if it doesn't exist
            db.execute(text("ALTER TABLE users ADD COLUMN organization_id INTEGER REFERENCES organizations(id)"))
            db.commit()
            
    except (OperationalError, ProgrammingError):
        # Tables don't exist, create them
        Base.metadata.create_all(bind=engine)
    finally:
        db.close()

# Check and create tables if needed
check_and_create_tables()

# Pydantic models
class Token(BaseModel):
    access_token: str
    token_type: str
    organization: Optional[str] = None

class TokenData(BaseModel):
    email: Optional[str] = None

class UserBase(BaseModel):
    email: EmailStr
    name: str

class UserCreate(UserBase):
    password: str
    role: str = "staff"
    organization: Optional[str] = None

    @validator('password')
    def password_complexity(cls, v):
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.islower() for c in v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        if not any(c in "!@#$%^&*()_+" for c in v):
            raise ValueError("Password must contain at least one special character")
        return v

class UserInDB(UserBase):
    id: int
    role: str
    is_active: bool
    created_at: datetime
    last_active: Optional[datetime]
    organization: Optional[str]

    class Config:
        from_attributes = True

class ReportBase(BaseModel):
    title: str
    description: str
    category: str

class ReportCreate(ReportBase):
    pass

class ReportInDB(ReportBase):
    id: int
    status: str
    admin_comments: Optional[str]
    author_id: int
    author_name: str
    organization_id: int
    created_at: datetime
    updated_at: Optional[datetime]
    attachments: List[dict] = []

    class Config:
        from_attributes = True

class AttachmentInDB(BaseModel):
    id: int
    name: str
    type: str
    size: int
    url: str

    class Config:
        from_attributes = True

class ReportStatusUpdate(BaseModel):
    status: str
    admin_comments: Optional[str] = None

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
    
    # Update last active time
    user.last_active = datetime.utcnow()
    db.commit()
    
    return user

async def get_current_active_user(current_user: UserInDB = Depends(get_current_user)):
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

async def is_admin(current_user: UserInDB = Depends(get_current_active_user)):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Not enough permissions")
    return current_user

# Initialize FastAPI app
app = FastAPI(
    title="ReportHub API",
    description="A professional reporting system with user authentication and document attachments",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Auth routes
@app.post("/auth/login", response_model=Token)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(), 
    db: Session = Depends(get_db)
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
    
    # Get organization name if exists
    org_name = user.organization.name if user.organization else None
    
    return {"access_token": access_token, "token_type": "bearer", "organization": org_name}

@app.get("/auth/me", response_model=UserInDB)
async def read_users_me(current_user: UserInDB = Depends(get_current_active_user)):
    return current_user

# User routes
@app.post("/users", response_model=UserInDB)
async def create_user(
    user: UserCreate, 
    db: Session = Depends(get_db),
    current_user: UserInDB = Depends(is_admin)
):
    db_user = get_user(db, email=user.email)
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    hashed_password = get_password_hash(user.password)
    
    # Get or create organization
    organization = None
    if user.organization:
        organization = db.query(Organization).filter(Organization.name == user.organization).first()
        if not organization:
            organization = Organization(name=user.organization)
            db.add(organization)
            db.commit()
            db.refresh(organization)
    
    db_user = User(
        email=user.email,
        name=user.name,
        hashed_password=hashed_password,
        role=user.role,
        organization_id=organization.id if organization else None
    )
    
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    return db_user

@app.get("/users", response_model=List[UserInDB])
async def read_users(
    skip: int = 0, 
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: UserInDB = Depends(is_admin)
):
    # Only show users from the same organization
    query = db.query(User).filter(User.organization_id == current_user.organization_id)
    users = query.offset(skip).limit(limit).all()
    return users

@app.get("/users/{user_id}", response_model=UserInDB)
async def read_user(
    user_id: int,
    db: Session = Depends(get_db),
    current_user: UserInDB = Depends(is_admin)
):
    db_user = db.query(User).filter(User.id == user_id, User.organization_id == current_user.organization_id).first()
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return db_user

@app.delete("/users/{user_id}")
async def delete_user(
    user_id: int,
    db: Session = Depends(get_db),
    current_user: UserInDB = Depends(is_admin)
):
    if user_id == current_user.id:
        raise HTTPException(status_code=400, detail="Cannot delete yourself")
    
    db_user = db.query(User).filter(User.id == user_id, User.organization_id == current_user.organization_id).first()
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    
    db.delete(db_user)
    db.commit()
    
    return {"message": "User deleted successfully"}

# Report routes
@app.post("/reports", response_model=ReportInDB)
async def create_report(
    title: str = Form(...),
    description: str = Form(...),
    category: str = Form(...),
    attachments: List[UploadFile] = File([]),
    db: Session = Depends(get_db),
    current_user: UserInDB = Depends(get_current_active_user)
):
    if not current_user.organization_id:
        raise HTTPException(status_code=400, detail="User must belong to an organization to create reports")
    
    # Create report
    db_report = Report(
        title=title,
        description=description,
        category=category,
        author_id=current_user.id,
        organization_id=current_user.organization_id
    )
    
    db.add(db_report)
    db.commit()
    db.refresh(db_report)
    
    # Handle attachments
    saved_attachments = []
    for attachment in attachments:
        # Generate unique filename
        file_ext = os.path.splitext(attachment.filename)[1]
        filename = f"{uuid.uuid4()}{file_ext}"
        file_path = os.path.join(ATTACHMENTS_DIR, filename)
        
        # Save the file
        with open(file_path, "wb") as buffer:
            buffer.write(await attachment.read())
        
        file_url = f"/attachments/{filename}"
        
        db_attachment = Attachment(
            name=attachment.filename,
            type=attachment.content_type,
            size=attachment.size,
            url=file_url,
            report_id=db_report.id,
            uploader_id=current_user.id
        )
        
        db.add(db_attachment)
        saved_attachments.append({
            "id": db_attachment.id,
            "name": db_attachment.name,
            "type": db_attachment.type,
            "size": db_attachment.size,
            "url": db_attachment.url
        })
    
    db.commit()
    
    # Add author name to response
    report_data = db_report.__dict__
    report_data["author_name"] = current_user.name
    report_data["attachments"] = saved_attachments
    
    return report_data

@app.get("/reports", response_model=List[ReportInDB])
async def read_reports(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: UserInDB = Depends(get_current_active_user)
):
    # Only show reports from the same organization
    query = db.query(Report).filter(Report.organization_id == current_user.organization_id)
    
    if current_user.role != "admin":
        query = query.filter(Report.author_id == current_user.id)
    
    reports = query.offset(skip).limit(limit).all()
    
    # Add author names and attachments to response
    reports_data = []
    for report in reports:
        report_data = report.__dict__
        author = db.query(User).filter(User.id == report.author_id).first()
        report_data["author_name"] = author.name
        
        attachments = db.query(Attachment).filter(Attachment.report_id == report.id).all()
        report_data["attachments"] = [
            {
                "id": a.id,
                "name": a.name,
                "type": a.type,
                "size": a.size,
                "url": a.url
            } for a in attachments
        ]
        
        reports_data.append(report_data)
    
    return reports_data

@app.get("/reports/{report_id}", response_model=ReportInDB)
async def read_report(
    report_id: int,
    db: Session = Depends(get_db),
    current_user: UserInDB = Depends(get_current_active_user)
):
    report = db.query(Report).filter(
        Report.id == report_id,
        Report.organization_id == current_user.organization_id
    ).first()
    
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found")
    
    # Check permissions
    if current_user.role != "admin" and report.author_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to access this report")
    
    # Add author name and attachments to response
    report_data = report.__dict__
    author = db.query(User).filter(User.id == report.author_id).first()
    report_data["author_name"] = author.name
    
    attachments = db.query(Attachment).filter(Attachment.report_id == report.id).all()
    report_data["attachments"] = [
        {
            "id": a.id,
            "name": a.name,
            "type": a.type,
            "size": a.size,
            "url": a.url
        } for a in attachments
    ]
    
    return report_data

@app.patch("/reports/{report_id}", response_model=ReportInDB)
async def update_report(
    report_id: int,
    status: Optional[str] = None,
    admin_comments: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: UserInDB = Depends(is_admin)
):
    report = db.query(Report).filter(
        Report.id == report_id,
        Report.organization_id == current_user.organization_id
    ).first()
    
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found")
    
    if status is not None:
        report.status = status
    if admin_comments is not None:
        report.admin_comments = admin_comments
    
    db.commit()
    db.refresh(report)
    
    # Add author name and attachments to response
    report_data = report.__dict__
    author = db.query(User).filter(User.id == report.author_id).first()
    report_data["author_name"] = author.name
    
    attachments = db.query(Attachment).filter(Attachment.report_id == report.id).all()
    report_data["attachments"] = [
        {
            "id": a.id,
            "name": a.name,
            "type": a.type,
            "size": a.size,
            "url": a.url
        } for a in attachments
    ]
    
    return report_data

@app.delete("/reports/{report_id}")
async def delete_report(
    report_id: int,
    db: Session = Depends(get_db),
    current_user: UserInDB = Depends(get_current_active_user)
):
    report = db.query(Report).filter(
        Report.id == report_id,
        Report.organization_id == current_user.organization_id
    ).first()
    
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found")
    
    # Check permissions
    if current_user.role != "admin" and report.author_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to delete this report")
    
    # Delete attachments first
    db.query(Attachment).filter(Attachment.report_id == report_id).delete()
    
    # Then delete the report
    db.delete(report)
    db.commit()
    
    return {"message": "Report deleted successfully"}

@app.patch("/reports/{report_id}/status", response_model=ReportInDB)
async def update_report_status(
    report_id: int,
    status_update: ReportStatusUpdate,
    db: Session = Depends(get_db),
    current_user: UserInDB = Depends(is_admin)
):
    report = db.query(Report).filter(
        Report.id == report_id,
        Report.organization_id == current_user.organization_id
    ).first()
    
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found")
    
    # Update status and comments
    report.status = status_update.status
    report.admin_comments = status_update.admin_comments
    
    db.commit()
    db.refresh(report)
    
    # Add author name and attachments to response
    report_data = report.__dict__
    author = db.query(User).filter(User.id == report.author_id).first()
    report_data["author_name"] = author.name
    
    attachments = db.query(Attachment).filter(Attachment.report_id == report.id).all()
    report_data["attachments"] = [
        {
            "id": a.id,
            "name": a.name,
            "type": a.type,
            "size": a.size,
            "url": a.url
        } for a in attachments
    ]
    
    return report_data

@app.get("/auth/first-user")
async def check_first_user(db: Session = Depends(get_db)):
    user_count = db.query(User).count()
    return {"is_first_user": user_count == 0}

@app.post("/auth/signup", response_model=Token)
async def signup_user(
    name: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    organization: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    # Check if email already exists
    existing_user = get_user(db, email)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Check if this is the first user
    is_first_user = db.query(User).count() == 0
    
    # First user must provide organization name
    if is_first_user and not organization:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Organization name is required for the first user"
        )
    
    # Create organization if this is the first user
    org = None
    if is_first_user and organization:
        # Check if organization name already exists
        existing_org = db.query(Organization).filter(Organization.name == organization).first()
        if existing_org:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Organization name already exists"
            )
        
        org = Organization(name=organization)
        db.add(org)
        db.commit()
        db.refresh(org)
    
    # For subsequent users, find their organization
    if not is_first_user and organization:
        org = db.query(Organization).filter(Organization.name == organization).first()
        if not org:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Organization not found"
            )
    
    # Create user
    hashed_password = get_password_hash(password)
    role = "admin" if is_first_user else "staff"
    
    db_user = User(
        email=email,
        name=name,
        hashed_password=hashed_password,
        role=role,
        organization_id=org.id if org else None
    )
    
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": db_user.email}, expires_delta=access_token_expires
    )
    
    # Prepare response
    response_data = {
        "access_token": access_token,
        "token_type": "bearer",
        "organization": org.name if org else None
    }
    
    return response_data

@app.get("/download/{filename}")
async def download_file(
    filename: str,
    db: Session = Depends(get_db),
    current_user: UserInDB = Depends(get_current_active_user)
):
    # Verify the file exists and the user has access to it
    file_path = os.path.join(ATTACHMENTS_DIR, filename)
    
    # Check if file exists
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    # Check if user has access to this file (either admin or owner of the report)
    attachment = db.query(Attachment).filter(Attachment.url == f"/attachments/{filename}").first()
    if not attachment:
        raise HTTPException(status_code=404, detail="File record not found")
    
    # Check if the report belongs to the same organization
    report = db.query(Report).filter(
        Report.id == attachment.report_id,
        Report.organization_id == current_user.organization_id
    ).first()
    
    if not report:
        raise HTTPException(status_code=403, detail="Not authorized to access this file")
    
    if current_user.role != "admin" and report.author_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to access this file")
    
    # Return the file with the original filename
    return FileResponse(
        file_path,
        filename=attachment.name,
        media_type=attachment.type
    )

# Initialize default admin user
def init_default_admin():
    db = SessionLocal()
    try:
        # Check if there's already an admin
        admin = db.query(User).filter(User.email == "admin@reporthub.com").first()
        if not admin:
            try:
                # Create default organization
                org = Organization(name="Default Organization")
                db.add(org)
                db.commit()
                db.refresh(org)
                
                # Create admin user
                hashed_password = get_password_hash("Admin123!")
                admin = User(
                    name="Admin User",
                    email="admin@reporthub.com",
                    hashed_password=hashed_password,
                    role="admin",
                    organization_id=org.id
                )
                db.add(admin)
                db.commit()
            except Exception as e:
                db.rollback()
                print(f"Error creating default admin: {e}")
    finally:
        db.close()

# Call the function to create default admin when starting up
init_default_admin()

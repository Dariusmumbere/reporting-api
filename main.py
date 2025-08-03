from fastapi import FastAPI, HTTPException, Depends, status, UploadFile, File, Form, WebSocket, WebSocketDisconnect, Request, Query, BackgroundTasks, Body
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse, HTMLResponse
from pydantic import BaseModel, EmailStr, validator
from typing import List, Optional, Dict
from datetime import datetime, timedelta, date
from jose import JWTError, jwt
from passlib.context import CryptContext
import os
import uuid
import json
import asyncio
import random
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from fastapi import BackgroundTasks
from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime, ForeignKey, func, inspect, JSON, case, Date, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from sqlalchemy.sql import text
import boto3
import io
import base64
from botocore.exceptions import ClientError
from io import BytesIO, StringIO
import pandas as pd
from fpdf import FPDF
import logging
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# Backblaze B2 Configuration
B2_BUCKET_NAME = "uploads-dir"
B2_ENDPOINT_URL = "https://s3.us-east-005.backblazeb2.com"
B2_KEY_ID = "0055ca7845641d30000000002"
B2_APPLICATION_KEY = "K005NNeGM9r28ujQ3jvNEQy2zUiu0TI"

# Initialize B2 client
b2_client = boto3.client(
    's3',
    endpoint_url=B2_ENDPOINT_URL,
    aws_access_key_id=B2_KEY_ID,
    aws_secret_access_key=B2_APPLICATION_KEY
)
GEMINI_API_KEY = "AIzaSyAfGhtKkofAFP9NZqOB0MR3Fr0fUZYfiT0"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"

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

# Email configuration
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_USERNAME = "dariusmumbere@gmail.com"
SMTP_PASSWORD = "qsvx xbnd qymq msda"
EMAIL_FROM = "ReportHub <noreply@reporthub.com>"
OTP_EXPIRATION_MINUTES = 10

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

async def send_verification_email(email: str, otp: str):
    try:
        # Create message
        msg = MIMEMultipart()
        msg['From'] = EMAIL_FROM
        msg['To'] = email
        msg['Subject'] = "Verify your email for ReportHub"
        
        # Email body
        body = f"""
        <html>
            <body>
                <h2>ReportHub Email Verification</h2>
                <p>Thank you for signing up with ReportHub!</p>
                <p>Your verification code is: <strong>{otp}</strong></p>
                <p>This code will expire in {OTP_EXPIRATION_MINUTES} minutes.</p>
                <p>If you didn't request this, please ignore this email.</p>
                <br>
                <p>Best regards,</p>
                <p>The ReportHub Team</p>
            </body>
        </html>
        """
        
        msg.attach(MIMEText(body, 'html'))
        
        # Connect to SMTP server and send email
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.send_message(msg)
        
        return True
    except Exception as e:
        print(f"Error sending email: {e}")
        return False

async def upload_to_b2(file: UploadFile, file_path: str) -> str:
    """Upload a file to Backblaze B2 and return the public URL"""
    try:
        # Generate a unique filename
        file_ext = os.path.splitext(file.filename)[1]
        filename = f"{uuid.uuid4()}{file_ext}"
        object_key = f"attachments/{filename}"
        
        # Upload the file
        b2_client.upload_fileobj(
            file.file,
            B2_BUCKET_NAME,
            object_key,
            ExtraArgs={'ContentType': file.content_type}
        )
        
        # Generate the public URL
        public_url = f"{B2_ENDPOINT_URL}/{B2_BUCKET_NAME}/{object_key}"
        return public_url
    except Exception as e:
        print(f"Error uploading to B2: {e}")
        raise HTTPException(status_code=500, detail="Failed to upload file")

async def delete_from_b2(url: str):
    """Delete a file from Backblaze B2"""
    try:
        # Extract the object key from the URL
        object_key = url.replace(f"{B2_ENDPOINT_URL}/{B2_BUCKET_NAME}/", "")
        b2_client.delete_object(Bucket=B2_BUCKET_NAME, Key=object_key)
    except Exception as e:
        print(f"Error deleting from B2: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete file")


# Models (remain the same as before)
class Organization(Base):
    __tablename__ = "organizations"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False)
    logo_data = Column(LargeBinary, nullable=True) 
    logo_content_type = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    templates = relationship("ReportTemplate", back_populates="organization")

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
    organization_id = Column(Integer, ForeignKey("organizations.id"), nullable=True)

    organization = relationship("Organization", back_populates="users")
    reports = relationship("Report", back_populates="author")
    attachments = relationship("Attachment", back_populates="uploader")
    sent_messages = relationship("ChatMessage", foreign_keys="ChatMessage.sender_id", back_populates="sender")
    received_messages = relationship("ChatMessage", foreign_keys="ChatMessage.recipient_id", back_populates="recipient")
    profile_picture = Column(String, nullable=True)
    notifications = relationship("Notification", back_populates="user")

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
    template_data = Column(JSON, nullable=True)  # Add this line
    template_id = Column(Integer, ForeignKey("report_templates.id"), nullable=True)

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

class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True)
    sender_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    recipient_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    content = Column(String, nullable=False)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    status = Column(String, default="delivered")  # delivered, read
    message_type = Column(String, default="text")  # text, voice

    sender = relationship("User", foreign_keys=[sender_id], back_populates="sent_messages")
    recipient = relationship("User", foreign_keys=[recipient_id], back_populates="received_messages")

class EmailVerification(Base):
    __tablename__ = "email_verifications"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, index=True, nullable=False)
    otp = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    expires_at = Column(DateTime(timezone=True))
    is_verified = Column(Boolean, default=False)

class ReportTemplate(Base):
    __tablename__ = "report_templates"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    description = Column(String)
    category = Column(String)
    organization_id = Column(Integer, ForeignKey("organizations.id"), nullable=True)
    created_by = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    organization = relationship("Organization", back_populates="templates")
    creator = relationship("User")
    fields = relationship("TemplateField", back_populates="template")

class TemplateField(Base):
    __tablename__ = "template_fields"
    
    id = Column(Integer, primary_key=True, index=True)
    template_id = Column(Integer, ForeignKey("report_templates.id"))
    name = Column(String, nullable=False)
    label = Column(String, nullable=False)
    field_type = Column(String, nullable=False)  # text, number, dropdown, checkbox, date, etc.
    required = Column(Boolean, default=False)
    order = Column(Integer, default=0)
    options = Column(JSON, nullable=True)  # For dropdowns, checkboxes, etc.
    default_value = Column(String, nullable=True)
    placeholder = Column(String, nullable=True)
    
    template = relationship("ReportTemplate", back_populates="fields")

class InvitationLink(Base):
    __tablename__ = "invitation_links"

    id = Column(Integer, primary_key=True, index=True)
    token = Column(String, unique=True, index=True, nullable=False)
    created_by_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    organization_id = Column(Integer, ForeignKey("organizations.id"), nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    is_used = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    created_by = relationship("User", foreign_keys=[created_by_id])
    organization = relationship("Organization")

class Notification(Base):
    __tablename__ = "notifications"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    title = Column(String, nullable=False)
    message = Column(String, nullable=False)
    is_read = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    link = Column(String, nullable=True)  # Optional link for the notification

    user = relationship("User", back_populates="notifications")

    
# Initialize default admin user
def init_default_admin():
    db = SessionLocal()
    try:
        # Check if any users exist
        user_count = db.query(User).count()
        if user_count == 0:
            # Create default organization
            org = Organization(name="Default Organization")
            db.add(org)
            db.commit()
            db.refresh(org)
            
            # Create super admin user
            hashed_password = get_password_hash("Admin123!")
            admin = User(
                name="Super Admin",
                email="superadmin@reporthub.com",
                hashed_password=hashed_password,
                role="super_admin",
                organization_id=org.id
            )
            db.add(admin)
            db.commit()
    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()
def add_column():
    """Add missing columns to existing tables if they don't exist"""
    db = SessionLocal()
    try:
        inspector = inspect(db.get_bind())
        
        # Check if logo_data column exists in organizations table
        org_columns = [col['name'] for col in inspector.get_columns('organizations')]
        if 'logo_data' not in org_columns:
            db.execute(text("ALTER TABLE organizations ADD COLUMN logo_data BYTEA"))
        
        # Check if logo_content_type column exists in organizations table
        if 'logo_content_type' not in org_columns:
            db.execute(text("ALTER TABLE organizations ADD COLUMN logo_content_type VARCHAR"))
        
        db.commit()
        print("Successfully added missing columns")
    except Exception as e:
        db.rollback()
        print(f"Error adding columns: {e}")
        raise
    finally:
        db.close()
add_column()

# Pydantic models (remain the same as before)
class ChatbotMessage(BaseModel):
    message: str
    context: Optional[str] = None 
    
class Token(BaseModel):
    access_token: str
    token_type: str
    requires_org_registration: bool = False

class TokenData(BaseModel):
    email: Optional[str] = None

class UserBase(BaseModel):
    email: EmailStr
    name: str
    
class ForgotPasswordRequest(BaseModel):
    email: EmailStr

class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str
    
class UserCreate(UserBase):
    password: str
    role: str = "staff"

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
    organization_id: Optional[int]
    organization_name: Optional[str]
    profile_picture: Optional[str] = None

    class Config:
        from_attributes = True

class ReportBase(BaseModel):
    title: str
    description: str
    category: str
    template_id: Optional[int]

class ReportCreate(ReportBase):
    pass

class ReportInDB(ReportBase):
    id: int
    status: str
    admin_comments: Optional[str]
    author_id: int
    author_name: str
    organization_id: int
    organization_name: str
    created_at: datetime
    updated_at: Optional[datetime]
    template_id: Optional[int] = None 
    template_data: Optional[dict] = None
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

class OrganizationCreate(BaseModel):
    name: str

class OrganizationInDB(BaseModel):
    id: int
    name: str
    logo_url: Optional[str] = None 
    created_at: datetime
    user_count: int
    report_count: int

    class Config:
        from_attributes = True

class ChatMessageModel(BaseModel):
    id: int
    sender_id: int
    recipient_id: int
    content: str
    timestamp: datetime
    status: str

class ChatUser(BaseModel):
    id: int
    name: str
    email: str
    status: str
    unread_count: int = 0

class EmailVerificationRequest(BaseModel):
    email: EmailStr

class VerifyOTPRequest(BaseModel):
    email: EmailStr
    otp: str

class TemplateFieldBase(BaseModel):
    name: str
    label: str
    field_type: str
    required: bool = False
    order: int = 0
    options: Optional[Dict] = None
    default_value: Optional[str] = None
    placeholder: Optional[str] = None

class TemplateFieldCreate(TemplateFieldBase):
    pass

class TemplateFieldInDB(TemplateFieldBase):
    id: int
    template_id: int
    
    class Config:
        from_attributes = True

class ReportTemplateBase(BaseModel):
    name: str
    description: Optional[str] = None
    category: Optional[str] = None
    

class ReportTemplateCreate(ReportTemplateBase):
    fields: List[TemplateFieldCreate] = []

class ReportTemplateInDB(ReportTemplateBase):
    id: int
    organization_id: Optional[int]
    created_by: int
    created_at: datetime
    updated_at: Optional[datetime]
    fields: List[TemplateFieldInDB] = []
    
    class Config:
        from_attributes = True

class UserBasicInfo(BaseModel):
    id: int
    name: str
    email: str

class InvitationResponse(BaseModel):
    id: int
    token: str
    created_at: datetime
    expires_at: datetime
    is_used: bool
    created_by: UserBasicInfo
    
    class Config:
        orm_mode = True

class NotificationInDB(BaseModel):
    id: int
    title: str
    message: str
    is_read: bool
    created_at: datetime
    link: Optional[str] = None

    class Config:
        from_attributes = True

        
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

async def is_super_admin(current_user: UserInDB = Depends(get_current_active_user)):
    if current_user.role != "super_admin":
        raise HTTPException(status_code=403, detail="Not enough permissions")
    return current_user

async def is_org_admin_or_super_admin(current_user: UserInDB = Depends(get_current_active_user)):
    if current_user.role not in ["admin", "super_admin"]:
        raise HTTPException(status_code=403, detail="Not enough permissions")
    return current_user
    
Base.metadata.create_all(bind=engine)

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

# WebSocket Connection Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[int, WebSocket] = {}
        self.user_status: Dict[int, str] = {}

    async def connect(self, websocket: WebSocket, user_id: int):
        await websocket.accept()
        self.active_connections[user_id] = websocket
        self.user_status[user_id] = "online"
        await self.broadcast_user_status(user_id, "online")

    def disconnect(self, user_id: int):
        if user_id in self.active_connections:
            del self.active_connections[user_id]
            self.user_status[user_id] = "offline"
            asyncio.create_task(self.broadcast_user_status(user_id, "offline"))

    async def send_personal_message(self, message: str, user_id: int):
        if user_id in self.active_connections:
            await self.active_connections[user_id].send_text(message)

    async def broadcast_user_status(self, user_id: int, status: str):
        for connection in self.active_connections.values():
            await connection.send_text(json.dumps({
                "type": "user_status",
                "user_id": user_id,
                "status": status
            }))

manager = ConnectionManager()

# WebSocket endpoint for chat
@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket, token: str = None):
    try:
        # Authenticate user
        if not token:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return

        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            email: str = payload.get("sub")
            if email is None:
                await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
                return
        except JWTError:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return

        db = SessionLocal()
        user = get_user(db, email=email)
        if user is None:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return

        # Connect the user
        await manager.connect(websocket, user.id)

        # Send authentication response
        await websocket.send_text(json.dumps({
            "type": "auth_response",
            "success": True,
            "user_id": user.id
        }))

        # Keep connection alive
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            if message["type"] == "message":
                # Handle both text and voice messages
                if message.get("is_voice_message"):
                    # Save voice message to database
                    db_message = ChatMessage(
                        sender_id=user.id,
                        recipient_id=message["recipient_id"],
                        content=message["content"],  # This should be the URL to the audio file
                        timestamp=datetime.utcnow(),
                        status="delivered",
                        message_type="voice"
                    )
                else:
                    # Save text message to database
                    db_message = ChatMessage(
                        sender_id=user.id,
                        recipient_id=message["recipient_id"],
                        content=message["content"],
                        timestamp=datetime.utcnow(),
                        status="delivered",
                        message_type="text"
                    )
                
                db.add(db_message)
                db.commit()
                db.refresh(db_message)

                # Prepare response message
                response_message = {
                    "id": db_message.id,
                    "sender_id": db_message.sender_id,
                    "recipient_id": db_message.recipient_id,
                    "content": db_message.content,
                    "timestamp": db_message.timestamp.isoformat(),
                    "status": db_message.status,
                    "message_type": db_message.message_type
                }

                # Add duration for voice messages
                if message.get("is_voice_message"):
                    response_message["duration"] = message.get("duration", 0)

                # Send to recipient if online
                await manager.send_personal_message(json.dumps({
                    "type": "message",
                    "message": response_message
                }), message["recipient_id"])

            elif message["type"] == "mark_read":
                # Mark messages as read
                db.query(ChatMessage).filter(
                    ChatMessage.sender_id == message["sender_id"],
                    ChatMessage.recipient_id == user.id,
                    ChatMessage.status == "delivered"
                ).update({"status": "read"})
                db.commit()

            elif message["type"] == "voice_message_played":
                # Update message status when voice message is played
                db.query(ChatMessage).filter(
                    ChatMessage.id == message["message_id"],
                    ChatMessage.recipient_id == user.id
                ).update({"status": "read"})
                db.commit()

    except WebSocketDisconnect:
        manager.disconnect(user.id)
    except Exception as e:
        print(f"WebSocket error: {e}")
        manager.disconnect(user.id)
    finally:
        db.close()

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
    
    # Check if user has an organization (except for super admin)
    requires_org_registration = user.organization_id is None and user.role != "super_admin"
    
    return {
        "access_token": access_token, 
        "token_type": "bearer",
        "requires_org_registration": requires_org_registration
    }

@app.get("/auth/me", response_model=UserInDB)
async def read_users_me(current_user: UserInDB = Depends(get_current_active_user), db: Session = Depends(get_db)):
    # Add organization name to response
    user_data = current_user.__dict__
    if current_user.organization:
        user_data["organization_name"] = current_user.organization.name
    return UserInDB(**user_data)

@app.post("/auth/send-verification-email")
async def send_verification_email_endpoint(
    email_request: EmailVerificationRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    # Check if email already exists
    existing_user = get_user(db, email_request.email)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Generate OTP (6-digit number)
    otp = str(random.randint(100000, 999999))
    
    # Delete any existing OTPs for this email
    db.query(EmailVerification).filter(
        EmailVerification.email == email_request.email
    ).delete()
    
    # Create new verification record
    verification = EmailVerification(
        email=email_request.email,
        otp=otp,
        expires_at=datetime.utcnow() + timedelta(minutes=OTP_EXPIRATION_MINUTES)
    )
    
    db.add(verification)
    db.commit()
    
    # Send email in background
    background_tasks.add_task(send_verification_email, email_request.email, otp)
    
    return {"message": "Verification email sent"}

@app.post("/auth/verify-otp")
async def verify_otp(
    otp_request: VerifyOTPRequest,
    db: Session = Depends(get_db)
):
    # Find the verification record
    verification = db.query(EmailVerification).filter(
        EmailVerification.email == otp_request.email,
        EmailVerification.otp == otp_request.otp,
        EmailVerification.expires_at > datetime.utcnow()
    ).first()
    
    if not verification:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired OTP"
        )
    
    # Mark as verified
    verification.is_verified = True
    db.commit()
    
    return {"message": "Email verified successfully"}
    
@app.post("/auth/signup", response_model=Token)
async def signup_user(
    name: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    organization_name: Optional[str] = Form(None),
    invite_token: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    # Check if email already exists
    existing_user = get_user(db, email)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Check if email is verified
    verification = db.query(EmailVerification).filter(
        EmailVerification.email == email,
        EmailVerification.is_verified == True
    ).first()
    
    if not verification:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email not verified"
        )
    
    # Validate that either organization_name or invite_token is provided, but not both
    if not organization_name and not invite_token:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either organization name or invitation token is required"
        )
    
    if organization_name and invite_token:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot provide both organization name and invitation token"
        )
    
    # Create user
    hashed_password = get_password_hash(password)
    
    # First user becomes admin, others are staff
    is_first_user = db.query(User).count() == 0
    role = "admin" if is_first_user else "staff"
    
    db_user = User(
        email=email,
        name=name,
        hashed_password=hashed_password,
        role=role,
        organization_id=None  # Will be set based on flow
    )
    
    # Handle invitation flow
    if invite_token:
        invitation = db.query(InvitationLink).filter(
            InvitationLink.token == invite_token,
            InvitationLink.expires_at >= datetime.utcnow(),
            InvitationLink.is_used == False
        ).first()
        
        if not invitation:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or expired invitation link"
            )
        
        # Set user's organization from invitation
        db_user.organization_id = invitation.organization_id
        db_user.role = "staff"  # Invited users are always staff
        
        # Mark invitation as used
        invitation.is_used = True
    
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    # Handle organization creation flow
    if organization_name:
        # Check if organization name already exists
        existing_org = db.query(Organization).filter(Organization.name == organization_name).first()
        if existing_org:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Organization name already exists"
            )
        
        # Create new organization
        db_org = Organization(name=organization_name)
        db.add(db_org)
        db.commit()
        db.refresh(db_org)
        
        # Update user's organization
        db_user.organization_id = db_org.id
        db_user.role = "admin"  # Organization creator becomes admin
        db.commit()
        db.refresh(db_user)
    
    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": db_user.email}, expires_delta=access_token_expires
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "requires_org_registration": False  # Already handled in this flow
    }

@app.post("/auth/register-organization")
async def register_organization(
    organization: OrganizationCreate,
    db: Session = Depends(get_db),
    current_user: UserInDB = Depends(get_current_active_user)
):
    # Check if user already has an organization
    if current_user.organization_id is not None and current_user.role != "super_admin":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User already belongs to an organization"
        )
    
    # Check if organization name already exists
    existing_org = db.query(Organization).filter(Organization.name == organization.name).first()
    if existing_org:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Organization name already exists"
        )
    
    # Create new organization
    db_org = Organization(name=organization.name)
    db.add(db_org)
    db.commit()
    db.refresh(db_org)
    
    # Update user's organization if not super admin
    if current_user.role != "super_admin":
        current_user.organization_id = db_org.id
        current_user.role = "admin"
        db.commit()
        db.refresh(current_user)
    
    return {"message": "Organization registered successfully"}

# Super Admin routes
@app.get("/super-admin/organizations", response_model=List[OrganizationInDB])
async def get_all_organizations(
    skip: int = 0,
    limit: int = 10,
    db: Session = Depends(get_db),
    current_user: UserInDB = Depends(is_super_admin)
):
    # Get all organizations with user and report counts
    organizations = db.query(
        Organization,
        func.count(User.id).label("user_count"),
        func.count(Report.id).label("report_count")
    ).outerjoin(
        User, User.organization_id == Organization.id
    ).outerjoin(
        Report, Report.organization_id == Organization.id
    ).group_by(
        Organization.id
    ).offset(skip).limit(limit).all()

    result = []
    for org, user_count, report_count in organizations:
        result.append(OrganizationInDB(
            id=org.id,
            name=org.name,
            created_at=org.created_at,
            user_count=user_count,
            report_count=report_count
        ))

    return result

@app.post("/super-admin/organizations", response_model=OrganizationInDB)
async def create_organization(
    organization: OrganizationCreate,
    db: Session = Depends(get_db),
    current_user: UserInDB = Depends(is_super_admin)
):
    # Check if organization name already exists
    existing_org = db.query(Organization).filter(Organization.name == organization.name).first()
    if existing_org:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Organization name already exists"
        )
    
    # Create new organization
    db_org = Organization(name=organization.name)
    db.add(db_org)
    db.commit()
    db.refresh(db_org)
    
    return OrganizationInDB(
        id=db_org.id,
        name=db_org.name,
        created_at=db_org.created_at,
        user_count=0,
        report_count=0
    )

@app.delete("/super-admin/organizations/{org_id}")
async def delete_organization(
    org_id: int,
    db: Session = Depends(get_db),
    current_user: UserInDB = Depends(is_super_admin)
):
    org = db.query(Organization).filter(Organization.id == org_id).first()
    if not org:
        raise HTTPException(status_code=404, detail="Organization not found")
    
    # Delete all users and reports in the organization
    db.query(User).filter(User.organization_id == org_id).delete()
    db.query(Report).filter(Report.organization_id == org_id).delete()
    
    # Delete the organization
    db.delete(org)
    db.commit()
    
    return {"message": "Organization deleted successfully"}

@app.get("/super-admin/users", response_model=List[UserInDB])
async def get_all_users(
    skip: int = 0,
    limit: int = 10,
    db: Session = Depends(get_db),
    current_user: UserInDB = Depends(is_super_admin)
):
    # Get all users with their organization names
    users = db.query(User).join(
        Organization, User.organization_id == Organization.id, isouter=True
    ).offset(skip).limit(limit).all()
    
    users_data = []
    for user in users:
        user_data = user.__dict__
        if user.organization:
            user_data["organization_name"] = user.organization.name
        users_data.append(UserInDB(**user_data))
    
    return users_data

@app.post("/super-admin/users", response_model=UserInDB)
async def create_user_as_super_admin(
    name: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    role: str = Form(...),
    organization_id: int = Form(None),
    db: Session = Depends(get_db),
    current_user: UserInDB = Depends(is_super_admin)
):
    # Check if email already exists
    existing_user = get_user(db, email)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Validate role
    if role not in ["super_admin", "admin", "staff"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid role"
        )
    
    # Validate organization if not super admin
    if role != "super_admin" and organization_id is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Organization is required for non-super admin users"
        )
    
    if organization_id is not None:
        org = db.query(Organization).filter(Organization.id == organization_id).first()
        if not org:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Organization not found"
            )
    
    # Create user
    hashed_password = get_password_hash(password)
    
    db_user = User(
        email=email,
        name=name,
        hashed_password=hashed_password,
        role=role,
        organization_id=organization_id
    )
    
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    # Include organization name in response
    user_data = db_user.__dict__
    if db_user.organization:
        user_data["organization_name"] = db_user.organization.name
    return UserInDB(**user_data)

@app.delete("/super-admin/users/{user_id}")
async def delete_user_as_super_admin(
    user_id: int,
    db: Session = Depends(get_db),
    current_user: UserInDB = Depends(is_super_admin)
):
    if user_id == current_user.id:
        raise HTTPException(status_code=400, detail="Cannot delete yourself")
    
    db_user = db.query(User).filter(User.id == user_id).first()
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    
    db.delete(db_user)
    db.commit()
    
    return {"message": "User deleted successfully"}

@app.get("/super-admin/reports", response_model=List[ReportInDB])
async def get_all_reports(
    skip: int = 0,
    limit: int = 10,
    status: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: UserInDB = Depends(is_super_admin)
):
    # Get all reports with filters
    query = db.query(Report).join(
        User, Report.author_id == User.id
    ).join(
        Organization, Report.organization_id == Organization.id
    )
    
    if status:
        query = query.filter(Report.status == status)
    
    reports = query.offset(skip).limit(limit).all()
    
    # Format response with author and organization names
    reports_data = []
    for report in reports:
        report_data = report.__dict__
        report_data["author_name"] = report.author.name
        report_data["organization_name"] = report.organization.name
        
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
        
        reports_data.append(ReportInDB(**report_data))
    
    return reports_data

@app.patch("/super-admin/reports/{report_id}/status", response_model=ReportInDB)
async def update_report_status_as_super_admin(
    report_id: int,
    status_update: ReportStatusUpdate,
    db: Session = Depends(get_db),
    current_user: UserInDB = Depends(is_super_admin)
):
    report = db.query(Report).filter(Report.id == report_id).first()
    
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found")
    
    # Update status and comments
    report.status = status_update.status
    report.admin_comments = status_update.admin_comments
    
    db.commit()
    db.refresh(report)
    
    # Add author and organization names to response
    report_data = report.__dict__
    report_data["author_name"] = report.author.name
    report_data["organization_name"] = report.organization.name
    
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
    
    return ReportInDB(**report_data)

@app.delete("/super-admin/reports/{report_id}")
async def delete_report_as_super_admin(
    report_id: int,
    db: Session = Depends(get_db),
    current_user: UserInDB = Depends(is_super_admin)
):
    report = db.query(Report).filter(Report.id == report_id).first()
    
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found")
    
    # Delete attachments first
    attachments = db.query(Attachment).filter(Attachment.report_id == report_id).all()
    for attachment in attachments:
        try:
            await delete_from_b2(attachment.url)
        except Exception as e:
            print(f"Failed to delete attachment from B2: {e}")
    
    db.query(Attachment).filter(Attachment.report_id == report_id).delete()
    
    # Then delete the report
    db.delete(report)
    db.commit()
    
    return {"message": "Report deleted successfully"}

@app.get("/super-admin/stats")
async def get_super_admin_stats(
    db: Session = Depends(get_db),
    current_user: UserInDB = Depends(is_super_admin)
):
    # Get total organizations count
    total_organizations = db.query(func.count(Organization.id)).scalar()
    
    # Get total users count
    total_users = db.query(func.count(User.id)).scalar()
    
    # Get total reports count
    total_reports = db.query(func.count(Report.id)).scalar()
    
    # Get pending reports count
    pending_reports = db.query(func.count(Report.id)).filter(
        Report.status == "pending"
    ).scalar()
    
    # Get recent organizations (last 5 created)
    recent_orgs = db.query(Organization).order_by(
        Organization.created_at.desc()
    ).limit(5).all()
    
    # Format recent organizations data
    recent_orgs_data = []
    for org in recent_orgs:
        user_count = db.query(func.count(User.id)).filter(
            User.organization_id == org.id
        ).scalar()
        
        report_count = db.query(func.count(Report.id)).filter(
            Report.organization_id == org.id
        ).scalar()
        
        recent_orgs_data.append({
            "id": org.id,
            "name": org.name,
            "created_at": org.created_at,
            "user_count": user_count,
            "report_count": report_count
        })
    
    return {
        "total_organizations": total_organizations,
        "total_users": total_users,
        "total_reports": total_reports,
        "pending_reports": pending_reports,
        "recent_organizations": recent_orgs_data
    }

# Chat routes
@app.get("/chat/users", response_model=List[ChatUser])
async def get_chat_users(
    db: Session = Depends(get_db),
    current_user: UserInDB = Depends(get_current_active_user)
):
    # Get all users in the same organization (or all users for super admin)
    if current_user.role == "super_admin":
        users = db.query(User).filter(User.id != current_user.id).all()
    else:
        users = db.query(User).filter(
            User.organization_id == current_user.organization_id,
            User.id != current_user.id
        ).all()

    # Get unread message counts
    unread_counts = db.query(
        ChatMessage.sender_id,
        func.count(ChatMessage.id).label("unread_count")
    ).filter(
        ChatMessage.recipient_id == current_user.id,
        ChatMessage.status == "delivered"
    ).group_by(ChatMessage.sender_id).all()

    unread_dict = {user_id: count for user_id, count in unread_counts}

    # Format response
    chat_users = []
    for user in users:
        chat_users.append(ChatUser(
            id=user.id,
            name=user.name,
            email=user.email,
            status="online" if user.id in manager.active_connections else "offline",
            unread_count=unread_dict.get(user.id, 0)
        ))

    return chat_users

@app.get("/chat/messages/{user_id}", response_model=List[ChatMessageModel])
async def get_chat_messages(
    user_id: int,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: UserInDB = Depends(get_current_active_user)
):
    # Get messages between current user and the other user
    messages = db.query(ChatMessage).filter(
        ((ChatMessage.sender_id == current_user.id) & (ChatMessage.recipient_id == user_id)) |
        ((ChatMessage.sender_id == user_id) & (ChatMessage.recipient_id == current_user.id))
    ).order_by(ChatMessage.timestamp.desc()).limit(limit).all()

    # Mark messages as read
    db.query(ChatMessage).filter(
        ChatMessage.sender_id == user_id,
        ChatMessage.recipient_id == current_user.id,
        ChatMessage.status == "delivered"
    ).update({"status": "read"})
    db.commit()

    return messages[::-1]  # Return in chronological order

# User routes
@app.post("/users", response_model=UserInDB)
async def create_user(
    user: UserCreate, 
    db: Session = Depends(get_db),
    current_user: UserInDB = Depends(is_org_admin_or_super_admin)
):
    db_user = get_user(db, email=user.email)
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    hashed_password = get_password_hash(user.password)
    
    # New users inherit the admin's organization (unless super admin)
    org_id = current_user.organization_id if current_user.role != "super_admin" else None
    
    db_user = User(
        email=user.email,
        name=user.name,
        hashed_password=hashed_password,
        role=user.role,
        organization_id=org_id
    )
    
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    # Include organization name in response
    user_data = db_user.__dict__
    if db_user.organization:
        user_data["organization_name"] = db_user.organization.name
    return UserInDB(**user_data)

@app.get("/users", response_model=List[UserInDB])
async def read_users(
    skip: int = 0, 
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: UserInDB = Depends(is_org_admin_or_super_admin)
):
    # For super admin, show all users
    if current_user.role == "super_admin":
        users = db.query(User).offset(skip).limit(limit).all()
    else:
        # For org admin, only show users from the same organization
        users = db.query(User).filter(
            User.organization_id == current_user.organization_id
        ).offset(skip).limit(limit).all()
    
    # Convert to UserInDB with organization_name
    users_data = []
    for user in users:
        user_data = user.__dict__
        if user.organization:
            user_data["organization_name"] = user.organization.name
        users_data.append(UserInDB(**user_data))
    
    return users_data

@app.get("/users/{user_id}", response_model=UserInDB)
async def read_user(
    user_id: int,
    db: Session = Depends(get_db),
    current_user: UserInDB = Depends(is_org_admin_or_super_admin)
):
    # For super admin, can view any user
    if current_user.role == "super_admin":
        db_user = db.query(User).filter(User.id == user_id).first()
    else:
        # For org admin, only users from the same organization
        db_user = db.query(User).filter(
            User.id == user_id,
            User.organization_id == current_user.organization_id
        ).first()
    
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Include organization name in response
    user_data = db_user.__dict__
    if db_user.organization:
        user_data["organization_name"] = db_user.organization.name
    return UserInDB(**user_data)

@app.delete("/users/{user_id}")
async def delete_user(
    user_id: int,
    db: Session = Depends(get_db),
    current_user: UserInDB = Depends(is_org_admin_or_super_admin)
):
    if user_id == current_user.id:
        raise HTTPException(status_code=400, detail="Cannot delete yourself")
    
    # For super admin, can delete any user
    if current_user.role == "super_admin":
        db_user = db.query(User).filter(User.id == user_id).first()
    else:
        # For org admin, only users from the same organization
        db_user = db.query(User).filter(
            User.id == user_id,
            User.organization_id == current_user.organization_id
        ).first()
    
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    
    db.delete(db_user)
    db.commit()
    
    return {"message": "User deleted successfully"}

# Report routes
@app.post("/reports")
async def create_report(
    title: str = Form(...),
    description: str = Form(...),
    category: str = Form(...),
    template_id: Optional[int] = Form(None),
    template_fields: Optional[str] = Form(None),
    attachments: List[UploadFile] = File([]),
    db: Session = Depends(get_db),
    current_user: UserInDB = Depends(get_current_active_user)
):
    # Check if user has an organization (unless super admin)
    if current_user.organization_id is None and current_user.role != "super_admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User must belong to an organization to create reports"
        )
    
    # Parse template fields if provided
    template_data = None
    if template_fields:
        try:
            template_data = json.loads(template_fields)
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid template fields data"
            )
    
    # Create report
    db_report = Report(
        title=title,
        description=description,
        category=category,
        author_id=current_user.id,
        organization_id=current_user.organization_id if current_user.role != "super_admin" else None,
        template_id=template_id,
        template_data=template_data  # Store the template fields data
    )
    
    db.add(db_report)
    db.commit()
    db.refresh(db_report)
    
    # Handle attachments
    saved_attachments = []
    for attachment in attachments:
        try:
            # Upload to Backblaze B2
            file_url = await upload_to_b2(attachment, f"reports/{db_report.id}")
            
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
        except Exception as e:
            print(f"Error processing attachment: {e}")
            continue
    
    # Trigger notification after report is created and attachments are processed
    await create_report_notification(db, db_report, "created", current_user)
    
    db.commit()
    
    # Add author and organization names to response
    report_data = db_report.__dict__
    report_data["author_name"] = current_user.name
    if db_report.organization:
        report_data["organization_name"] = db_report.organization.name
    else:
        report_data["organization_name"] = "No Organization"
    report_data["attachments"] = saved_attachments
    report_data["template_fields"] = template_data

    return ReportInDB(**report_data)
    
@app.get("/reports", response_model=List[ReportInDB])
async def read_reports(
    skip: int = 0,
    limit: int = 100,
    status: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: UserInDB = Depends(get_current_active_user)
):
    # For super admin, show all reports
    if current_user.role == "super_admin":
        query = db.query(Report)
    else:
        # Check if user has an organization
        if current_user.organization_id is None:
            return []
        
        # Only show reports from the same organization
        query = db.query(Report).filter(
            Report.organization_id == current_user.organization_id
        )
        
        # For non-admin users, only show their own reports
        if current_user.role != "admin":
            query = query.filter(Report.author_id == current_user.id)
    
    if status:
        query = query.filter(Report.status == status)
    
    reports = query.offset(skip).limit(limit).all()
    
    # Add author and organization names to response
    reports_data = []
    for report in reports:
        report_data = report.__dict__
        author = db.query(User).filter(User.id == report.author_id).first()
        report_data["author_name"] = author.name
        if report.organization:
            report_data["organization_name"] = report.organization.name
        else:
            report_data["organization_name"] = "No Organization"
        
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
        
        reports_data.append(ReportInDB(**report_data))
    
    return reports_data

@app.get("/reports/{report_id}", response_model=ReportInDB)
async def read_report(
    report_id: int,
    db: Session = Depends(get_db),
    current_user: UserInDB = Depends(get_current_active_user)
):
    # For super admin, can view any report
    if current_user.role == "super_admin":
        report = db.query(Report).filter(Report.id == report_id).first()
    else:
        # Check if user has an organization
        if current_user.organization_id is None:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User must belong to an organization to view reports"
            )
        
        report = db.query(Report).filter(
            Report.id == report_id,
            Report.organization_id == current_user.organization_id
        ).first()
    
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found")
    
    # Check permissions for non-admin users
    if current_user.role != "admin" and current_user.role != "super_admin" and report.author_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to access this report")
    
    # Add author and organization names to response
    report_data = report.__dict__
    author = db.query(User).filter(User.id == report.author_id).first()
    report_data["author_name"] = author.name
    if report.organization:
        report_data["organization_name"] = report.organization.name
    else:
        report_data["organization_name"] = "No Organization"
    
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
    
    # Ensure template_data is included in the response
    report_data["template_data"] = report.template_data
    
    return ReportInDB(**report_data)

@app.patch("/reports/{report_id}", response_model=ReportInDB)
async def update_report(
    report_id: int,
    status: Optional[str] = None,
    admin_comments: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: UserInDB = Depends(is_org_admin_or_super_admin)
):
    # For super admin, can update any report
    if current_user.role == "super_admin":
        report = db.query(Report).filter(Report.id == report_id).first()
    else:
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
    
    # Add author and organization names to response
    report_data = report.__dict__
    author = db.query(User).filter(User.id == report.author_id).first()
    report_data["author_name"] = author.name
    if report.organization:
        report_data["organization_name"] = report.organization.name
    else:
        report_data["organization_name"] = "No Organization"
    
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
    
    return ReportInDB(**report_data)

@app.delete("/reports/{report_id}")
async def delete_report(
    report_id: int,
    db: Session = Depends(get_db),
    current_user: UserInDB = Depends(get_current_active_user)
):
    # For super admin, can delete any report
    if current_user.role == "super_admin":
        report = db.query(Report).filter(Report.id == report_id).first()
    else:
        report = db.query(Report).filter(
            Report.id == report_id,
            Report.organization_id == current_user.organization_id
        ).first()
    
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found")
    
    # Check permissions for non-admin users
    if current_user.role != "admin" and current_user.role != "super_admin" and report.author_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to delete this report")
    
    # Delete attachments first
    attachments = db.query(Attachment).filter(Attachment.report_id == report_id).all()
    for attachment in attachments:
        try:
            await delete_from_b2(attachment.url)
        except Exception as e:
            print(f"Failed to delete attachment from B2: {e}")
    
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
    current_user: UserInDB = Depends(is_org_admin_or_super_admin)
):
    # For super admin, can update any report
    if current_user.role == "super_admin":
        report = db.query(Report).filter(Report.id == report_id).first()
    else:
        report = db.query(Report).filter(
            Report.id == report_id,
            Report.organization_id == current_user.organization_id
        ).first()
    
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found")
    
    # Update status and comments
    report.status = status_update.status
    report.admin_comments = status_update.admin_comments
    await create_report_notification(db, report, "status_changed", current_user)
    db.commit()
    db.refresh(report)
    
    # Add author and organization names to response
    report_data = report.__dict__
    author = db.query(User).filter(User.id == report.author_id).first()
    report_data["author_name"] = author.name
    if report.organization:
        report_data["organization_name"] = report.organization.name
    else:
        report_data["organization_name"] = "No Organization"
    
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
    return ReportInDB(**report_data)

@app.get("/auth/first-user")
async def check_first_user(db: Session = Depends(get_db)):
    user_count = db.query(User).count()
    return {"is_first_user": user_count == 0}

@app.post("/chat/voice-message")
async def upload_voice_message(
    voice_message: UploadFile = File(...),
    recipient_id: int = Form(...),
    duration: float = Form(...),
    db: Session = Depends(get_db),
    current_user: UserInDB = Depends(get_current_active_user)
):
    try:
        # Validate file type
        if not voice_message.content_type.startswith('audio/'):
            raise HTTPException(
                status_code=400,
                detail="Only audio files are allowed"
            )

        # Upload the voice message to B2
        file_url = await upload_to_b2(voice_message, f"voice_messages/{current_user.id}")
        
        # Save message to database
        db_message = ChatMessage(
            sender_id=current_user.id,
            recipient_id=recipient_id,
            content=file_url,  # Store just the URL
            timestamp=datetime.utcnow(),
            status="delivered",
            message_type="voice"
        )
        
        db.add(db_message)
        db.commit()
        db.refresh(db_message)

        return {
            "id": db_message.id,
            "sender_id": db_message.sender_id,
            "recipient_id": db_message.recipient_id,
            "content": db_message.content,
            "timestamp": db_message.timestamp.isoformat(),
            "status": db_message.status,
            "message_type": db_message.message_type,
            "duration": duration
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/organization", response_model=OrganizationInDB)
async def get_organization_details(
    db: Session = Depends(get_db),
    current_user: UserInDB = Depends(get_current_active_user)
):
    if current_user.organization_id is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User does not belong to an organization"
        )
    
    org = db.query(Organization).filter(Organization.id == current_user.organization_id).first()
    if not org:
        raise HTTPException(status_code=404, detail="Organization not found")
    
    # Generate data URL if logo exists
    logo_url = None
    if org.logo_data:
        logo_url = f"data:{org.logo_content_type};base64,{base64.b64encode(org.logo_data).decode('utf-8')}"
    
    # Get user and report counts
    user_count = db.query(func.count(User.id)).filter(
        User.organization_id == org.id
    ).scalar()
    
    report_count = db.query(func.count(Report.id)).filter(
        Report.organization_id == org.id
    ).scalar()
    
    return OrganizationInDB(
        id=org.id,
        name=org.name,
        logo_url=logo_url,
        logo_content_type=org.logo_content_type,
        created_at=org.created_at,
        user_count=user_count,
        report_count=report_count
    )

@app.patch("/organization")
async def update_organization(
    name: Optional[str] = Form(None),
    logo: Optional[UploadFile] = File(None),
    db: Session = Depends(get_db),
    current_user: UserInDB = Depends(is_org_admin_or_super_admin)
):
    if current_user.organization_id is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User does not belong to an organization"
        )
    
    org = db.query(Organization).filter(Organization.id == current_user.organization_id).first()
    if not org:
        raise HTTPException(status_code=404, detail="Organization not found")
    
    updated = False
    
    # Update name if provided
    if name is not None and name != org.name:
        # Check if name is already taken
        existing_org = db.query(Organization).filter(
            Organization.name == name,
            Organization.id != org.id
        ).first()
        if existing_org:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Organization name already exists"
            )
        
        org.name = name
        updated = True
    
    # Handle logo upload if provided
    if logo is not None:
        try:
            # Read the file content
            logo_data = await logo.read()
            
            # Validate it's an image (basic check)
            if not logo.content_type.startswith('image/'):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Only image files are allowed for logos"
                )
            
            # Check file size (e.g., limit to 2MB)
            if len(logo_data) > 2 * 1024 * 1024:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Logo file size must be less than 2MB"
                )
            
            org.logo_data = logo_data
            org.logo_content_type = logo.content_type
            updated = True
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to process logo: {str(e)}"
            )
    
    if updated:
        db.commit()
        db.refresh(org)
    
    return {"message": "Organization updated successfully"}

@app.get("/download/{file_name}")
async def download_file(file_name: str):
    try:
        # Get the file from B2
        object_key = f"attachments/{file_name}"
        response = b2_client.get_object(
            Bucket=B2_BUCKET_NAME,
            Key=object_key
        )
        
        # Stream the file back to the client
        return StreamingResponse(
            response['Body'],
            media_type=response['ContentType'],
            headers={
                'Content-Disposition': f'attachment; filename="{file_name}"'
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
@app.post("/templates", response_model=ReportTemplateInDB)
async def create_template(
    template: ReportTemplateCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: UserInDB = Depends(get_current_active_user)
):
    # Check if template with this name already exists
    existing_template = db.query(ReportTemplate).filter(
        ReportTemplate.name == template.name,
        ReportTemplate.organization_id == current_user.organization_id
    ).first()
    
    if existing_template:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Template with this name already exists"
        )
    
    # Create template
    db_template = ReportTemplate(
        name=template.name,
        description=template.description,
        category=template.category,
        organization_id=current_user.organization_id,
        created_by=current_user.id
    )
    
    db.add(db_template)
    db.commit()
    db.refresh(db_template)
    
    # Create fields
    for field in template.fields:
        db_field = TemplateField(
            template_id=db_template.id,
            name=field.name,
            label=field.label,
            field_type=field.field_type,
            required=field.required,
            order=field.order,
            options=field.options,
            default_value=field.default_value,
            placeholder=field.placeholder
        )
        db.add(db_field)
    
    db.commit()
    db.refresh(db_template)
    
    # Notify all users in the organization about the new template
    background_tasks.add_task(notify_users_about_new_template, db, db_template, current_user)
    
    return db_template

async def notify_users_about_new_template(db: Session, template: ReportTemplate, creator: User):
    """Create notifications for all users about a new template"""
    try:
        # Get all users in the organization (except the creator)
        users = db.query(User).filter(
            User.organization_id == template.organization_id,
            User.id != creator.id
        ).all()
        
        # Create a notification for each user
        for user in users:
            notification = Notification(
                user_id=user.id,
                title="New Report Template Available",
                message=f"A new report template '{template.name}' has been created by {creator.name}",
                is_read=False,
                link=f"/templates/{template.id}"
            )
            db.add(notification)
        
        db.commit()
    except Exception as e:
        # Log the error but don't fail the template creation
        logger.error(f"Failed to create template notifications: {e}")
        db.rollback()

@app.get("/templates", response_model=List[ReportTemplateInDB])
async def get_templates(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: UserInDB = Depends(get_current_active_user)
):
    # For super admin, get all templates
    if current_user.role == "super_admin":
        templates = db.query(ReportTemplate).offset(skip).limit(limit).all()
    else:
        # For others, get templates from their organization
        templates = db.query(ReportTemplate).filter(
            ReportTemplate.organization_id == current_user.organization_id
        ).offset(skip).limit(limit).all()
    
    return templates

@app.get("/templates/{template_id}", response_model=ReportTemplateInDB)
async def get_template(
    template_id: int,
    db: Session = Depends(get_db),
    current_user: UserInDB = Depends(get_current_active_user)
):
    # For super admin, can get any template
    if current_user.role == "super_admin":
        template = db.query(ReportTemplate).filter(
            ReportTemplate.id == template_id
        ).first()
    else:
        # For others, only templates from their organization
        template = db.query(ReportTemplate).filter(
            ReportTemplate.id == template_id,
            ReportTemplate.organization_id == current_user.organization_id
        ).first()
    
    if template is None:
        raise HTTPException(status_code=404, detail="Template not found")
    
    return template

@app.put("/templates/{template_id}", response_model=ReportTemplateInDB)
async def update_template(
    template_id: int,
    template: ReportTemplateCreate,
    db: Session = Depends(get_db),
    current_user: UserInDB = Depends(get_current_active_user)
):
    # For super admin, can update any template
    if current_user.role == "super_admin":
        db_template = db.query(ReportTemplate).filter(
            ReportTemplate.id == template_id
        ).first()
    else:
        # For others, only templates from their organization
        db_template = db.query(ReportTemplate).filter(
            ReportTemplate.id == template_id,
            ReportTemplate.organization_id == current_user.organization_id
        ).first()
    
    if db_template is None:
        raise HTTPException(status_code=404, detail="Template not found")
    
    # Check if template with this name already exists (excluding current template)
    existing_template = db.query(ReportTemplate).filter(
        ReportTemplate.name == template.name,
        ReportTemplate.organization_id == current_user.organization_id,
        ReportTemplate.id != template_id
    ).first()
    
    if existing_template:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Template with this name already exists"
        )
    
    # Update template
    db_template.name = template.name
    db_template.description = template.description
    db_template.category = template.category
    db_template.updated_at = datetime.utcnow()
    
    # Delete existing fields
    db.query(TemplateField).filter(
        TemplateField.template_id == template_id
    ).delete()
    
    # Create new fields
    for field in template.fields:
        db_field = TemplateField(
            template_id=db_template.id,
            name=field.name,
            label=field.label,
            field_type=field.field_type,
            required=field.required,
            order=field.order,
            options=field.options,
            default_value=field.default_value,
            placeholder=field.placeholder
        )
        db.add(db_field)
    
    db.commit()
    db.refresh(db_template)
    
    return db_template

@app.delete("/templates/{template_id}")
async def delete_template(
    template_id: int,
    db: Session = Depends(get_db),
    current_user: UserInDB = Depends(get_current_active_user)
):
    # For super admin, can delete any template
    if current_user.role == "super_admin":
        template = db.query(ReportTemplate).filter(
            ReportTemplate.id == template_id
        ).first()
    else:
        # For others, only templates from their organization
        template = db.query(ReportTemplate).filter(
            ReportTemplate.id == template_id,
            ReportTemplate.organization_id == current_user.organization_id
        ).first()
    
    if template is None:
        raise HTTPException(status_code=404, detail="Template not found")
    
    # Delete fields first
    db.query(TemplateField).filter(
        TemplateField.template_id == template_id
    ).delete()
    
    # Then delete the template
    db.delete(template)
    db.commit()
    
    return {"message": "Template deleted successfully"}
# Add this to your FastAPI backend (in the chat routes section)

@app.delete("/chat/messages/{message_id}")
async def delete_message(
    message_id: int,
    db: Session = Depends(get_db),
    current_user: UserInDB = Depends(get_current_active_user)
):
    # Get the message
    message = db.query(ChatMessage).filter(ChatMessage.id == message_id).first()
    
    if not message:
        raise HTTPException(status_code=404, detail="Message not found")
    
    # Check if the current user is the sender of the message
    if message.sender_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only delete your own messages"
        )
    
    # For voice messages, delete the audio file from storage
    if message.message_type == "voice" and message.content.startswith("[VOICE_MESSAGE]"):
        try:
            audio_url = message.content.replace("[VOICE_MESSAGE]", "")
            await delete_from_b2(audio_url)
        except Exception as e:
            print(f"Error deleting voice message file: {e}")
    
    # Delete the message from the database
    db.delete(message)
    db.commit()
    
    # Notify the recipient via WebSocket if they're online
    if message.recipient_id in manager.active_connections:
        await manager.send_personal_message(json.dumps({
            "type": "message_deleted",
            "message_id": message_id
        }), message.recipient_id)
    
    return {"message": "Message deleted successfully"}

@app.get("/dashboard")
async def get_dashboard_data(
    period: str = "weekly",  # daily, weekly, monthly
    db: Session = Depends(get_db),
    current_user: UserInDB = Depends(get_current_active_user)
):
    # Calculate date ranges based on period
    end_date = datetime.utcnow()
    if period == "daily":
        start_date = end_date - timedelta(days=1)
        interval = "hour"
    elif period == "weekly":
        start_date = end_date - timedelta(days=7)
        interval = "day"
    else:  # monthly
        start_date = end_date - timedelta(days=30)
        interval = "day"

    # Base query with organization isolation
    base_query = db.query(Report)
    
    # Apply organization filter for all non-super-admin users
    if current_user.role != "super_admin":
        if not current_user.organization_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User must belong to an organization"
            )
        base_query = base_query.filter(Report.organization_id == current_user.organization_id)

    # Apply additional filter for regular users (non-admin)
    user_specific_query = base_query
    if current_user.role not in ["admin", "super_admin"]:
        user_specific_query = user_specific_query.filter(Report.author_id == current_user.id)

    # Get counts for each status (organization-wide for admins, user-specific for regular users)
    counts_query = base_query if current_user.role in ["admin", "super_admin"] else user_specific_query
    counts = counts_query.with_entities(
        func.count(Report.id).label("total"),
        func.count(case((Report.status == "pending", 1))).label("pending"),
        func.count(case((Report.status == "approved", 1))).label("approved"),
        func.count(case((Report.status == "rejected", 1))).label("rejected")
    ).first()

    # Get trend data with proper isolation
    trend_data = []
    if interval == "hour":
        for i in range(24):
            hour_start = start_date + timedelta(hours=i)
            hour_end = hour_start + timedelta(hours=1)
            
            trend_query = base_query if current_user.role in ["admin", "super_admin"] else user_specific_query
            hour_counts = trend_query.filter(
                Report.created_at >= hour_start,
                Report.created_at < hour_end
            ).with_entities(
                func.count(Report.id).label("total"),
                func.count(case((Report.status == "pending", 1))).label("pending"),
                func.count(case((Report.status == "approved", 1))).label("approved"),
                func.count(case((Report.status == "rejected", 1))).label("rejected")
            ).first()
            
            trend_data.append({
                "label": hour_start.strftime("%H:00"),
                "total": hour_counts.total or 0,
                "pending": hour_counts.pending or 0,
                "approved": hour_counts.approved or 0,
                "rejected": hour_counts.rejected or 0
            })
    else:
        current_date = start_date
        while current_date <= end_date:
            next_date = current_date + timedelta(days=1)
            
            trend_query = base_query if current_user.role in ["admin", "super_admin"] else user_specific_query
            day_counts = trend_query.filter(
                Report.created_at >= current_date,
                Report.created_at < next_date
            ).with_entities(
                func.count(Report.id).label("total"),
                func.count(case((Report.status == "pending", 1))).label("pending"),
                func.count(case((Report.status == "approved", 1))).label("approved"),
                func.count(case((Report.status == "rejected", 1))).label("rejected")
            ).first()
            
            trend_data.append({
                "label": current_date.strftime("%b %d"),
                "total": day_counts.total or 0,
                "pending": day_counts.pending or 0,
                "approved": day_counts.approved or 0,
                "rejected": day_counts.rejected or 0
            })
            current_date = next_date

    # Get categories with proper isolation
    categories_query = (base_query if current_user.role in ["admin", "super_admin"] else user_specific_query
                      ).with_entities(
                          Report.category,
                          func.count(Report.id)
                      ).group_by(Report.category)
    
    categories_data = [
        {"name": cat[0], "count": cat[1]} 
        for cat in categories_query.all()
    ]

    # Calculate trends with proper isolation
    def calculate_trend(current, previous):
        if previous == 0:
            return {"value": current, "percentage": 0}
        percentage = ((current - previous) / previous) * 100
        return {"value": current, "percentage": round(percentage, 1)}

    prev_start_date = start_date - (end_date - start_date)
    prev_counts_query = base_query if current_user.role in ["admin", "super_admin"] else user_specific_query
    prev_counts = prev_counts_query.filter(
        Report.created_at >= prev_start_date,
        Report.created_at < start_date
    ).with_entities(
        func.count(Report.id).label("total"),
        func.count(case((Report.status == "pending", 1))).label("pending"),
        func.count(case((Report.status == "approved", 1))).label("approved"),
        func.count(case((Report.status == "rejected", 1))).label("rejected")
    ).first()

    trends = {
        "total": calculate_trend(counts.total or 0, prev_counts.total or 0),
        "pending": calculate_trend(counts.pending or 0, prev_counts.pending or 0),
        "approved": calculate_trend(counts.approved or 0, prev_counts.approved or 0),
        "rejected": calculate_trend(counts.rejected or 0, prev_counts.rejected or 0)
    }

    # Get recent activity with proper isolation
    recent_reports_query = (base_query if current_user.role in ["admin", "super_admin"] else user_specific_query
                          ).order_by(Report.created_at.desc()).limit(5)
    recent_reports = recent_reports_query.all()

    recent_activity = []
    recent_reports_data = []
    for report in recent_reports:
        recent_activity.append({
            "type": "report",
            "status": report.status,
            "message": f"New report '{report.title}' submitted by {report.author.name}",
            "timestamp": report.created_at,
            "link": f"/reports/{report.id}"
        })
        recent_reports_data.append({
            "id": report.id,
            "title": report.title,
            "status": report.status,
            "created_at": report.created_at,
            "author_name": report.author.name
        })

    return {
        "counts": {
            "total": counts.total or 0,
            "pending": counts.pending or 0,
            "approved": counts.approved or 0,
            "rejected": counts.rejected or 0
        },
        "trends": trends,
        "categories": categories_data,
        "trend": {
            "labels": [item["label"] for item in trend_data],
            "total": [item["total"] for item in trend_data],
            "pending": [item["pending"] for item in trend_data],
            "approved": [item["approved"] for item in trend_data],
            "rejected": [item["rejected"] for item in trend_data]
        },
        "recentActivity": recent_activity,
        "recentReports": recent_reports_data,
        "data_scope": "organization" if current_user.role == "admin" else "personal" if current_user.role not in ["admin", "super_admin"] else "all"
    }
    
@app.post("/invitations/generate", response_model=dict)
async def generate_invitation_link(
    db: Session = Depends(get_db),
    current_user: UserInDB = Depends(is_org_admin_or_super_admin)
):
    # Generate a unique token
    token = str(uuid.uuid4())
    
    # Set expiration (7 days from now)
    expires_at = datetime.utcnow() + timedelta(days=7)
    
    # Create invitation link
    invitation = InvitationLink(
        token=token,
        created_by_id=current_user.id,
        organization_id=current_user.organization_id,
        expires_at=expires_at
    )
    
    db.add(invitation)
    db.commit()
    db.refresh(invitation)
    
    # Generate the full invitation URL
    invite_url = f"{token}"
    
    return {"invite_url": invite_url}

@app.get("/invitations/validate/{token}", response_model=dict)
async def validate_invitation_token(
    token: str,
    db: Session = Depends(get_db)
):
    # Find the invitation
    invitation = db.query(InvitationLink).filter(
        InvitationLink.token == token,
        InvitationLink.expires_at >= datetime.utcnow(),
        InvitationLink.is_used == False
    ).first()
    
    if not invitation:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired invitation link"
        )
    
    return {
        "valid": True,
        "organization_id": invitation.organization_id,
        "organization_name": invitation.organization.name
    }

@app.post("/invitations/accept", response_model=Token)
async def accept_invitation(
    name: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    token: str = Form(...),
    db: Session = Depends(get_db)
):
    # First validate the token
    invitation = db.query(InvitationLink).filter(
        InvitationLink.token == token,
        InvitationLink.expires_at >= datetime.utcnow(),
        InvitationLink.is_used == False
    ).first()
    
    if not invitation:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired invitation link"
        )
    
    # Check if email already exists
    existing_user = get_user(db, email)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Create user with the organization from the invitation
    hashed_password = get_password_hash(password)
    
    db_user = User(
        email=email,
        name=name,
        hashed_password=hashed_password,
        role="staff",  # Default role for invited users
        organization_id=invitation.organization_id
    )
    
    db.add(db_user)
    
    # Mark invitation as used
    invitation.is_used = True
    db.commit()
    db.refresh(db_user)
    
    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": db_user.email}, expires_delta=access_token_expires
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer"
    }
    
@app.get("/invitations", response_model=List[InvitationResponse])
async def get_invitations(
    db: Session = Depends(get_db),
    current_user: UserInDB = Depends(is_org_admin_or_super_admin)
):
    # Get all active invitations for the current user's organization
    invitations = db.query(InvitationLink).filter(
        InvitationLink.organization_id == current_user.organization_id
    ).order_by(InvitationLink.created_at.desc()).all()
    
    return invitations

@app.delete("/invitations/{invite_id}")
async def revoke_invitation(
    invite_id: int,
    db: Session = Depends(get_db),
    current_user: UserInDB = Depends(is_org_admin_or_super_admin)
):
    invitation = db.query(InvitationLink).filter(
        InvitationLink.id == invite_id,
        InvitationLink.organization_id == current_user.organization_id
    ).first()
    
    if not invitation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Invitation not found"
        )
    
    db.delete(invitation)
    db.commit()
    
    return {"message": "Invitation revoked successfully"}

@app.post("/auth/forgot-password")
async def forgot_password(
    request: ForgotPasswordRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Send a password reset email to the user"""
    user = get_user(db, request.email)
    if not user:
        # Don't reveal whether the email exists or not
        return {"message": "If the email exists, a password reset link has been sent"}
    
    # Generate a reset token (using JWT for simplicity)
    reset_token = create_access_token(
        data={"sub": user.email, "purpose": "password_reset"},
        expires_delta=timedelta(minutes=30)
    )
    
    # Send email in background
    background_tasks.add_task(send_password_reset_email, user.email, reset_token)
    
    return {"message": "If the email exists, a password reset link has been sent"}

@app.post("/auth/reset-password")
async def reset_password(
    request: ResetPasswordRequest,
    db: Session = Depends(get_db)
):
    """Reset the user's password using the provided token"""
    try:
        payload = jwt.decode(request.token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        purpose: str = payload.get("purpose")
        
        if email is None or purpose != "password_reset":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid token"
            )
            
        user = get_user(db, email)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid token"
            )
            
        # Update password
        user.hashed_password = get_password_hash(request.new_password)
        db.commit()
        
        return {"message": "Password updated successfully"}
        
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired token"
        )

# Add this utility function
async def send_password_reset_email(email: str, reset_token: str):
    """Send a password reset email with the token"""
    try:
        reset_link = f"https://dariusmumbere.github.io/reporting/reset-password.html?token={reset_token}"
        
        msg = MIMEMultipart()
        msg['From'] = EMAIL_FROM
        msg['To'] = email
        msg['Subject'] = "Password Reset Request"
        
        body = f"""
        <html>
            <body>
                <h2>Password Reset Request</h2>
                <p>We received a request to reset your password for ReportHub.</p>
                <p>Please click the link below to reset your password:</p>
                <p><a href="{reset_link}">Reset Password</a></p>
                <p>This link will expire in 30 minutes.</p>
                <p>If you didn't request this, please ignore this email.</p>
                <br>
                <p>Best regards,</p>
                <p>The ReportHub Team</p>
            </body>
        </html>
        """
        
        msg.attach(MIMEText(body, 'html'))
        
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.send_message(msg)
        
        return True
    except Exception as e:
        print(f"Error sending password reset email: {e}")
        return False
    
@app.get("/export/reports")
async def export_reports(
    date_range: str = Query("all", description="Date range filter"),
    start_date: str = Query(None, description="Start date for custom range"),
    end_date: str = Query(None, description="End date for custom range"),
    status: str = Query("all", description="Status filter"),
    format: str = Query("excel", description="Export format"),
    db: Session = Depends(get_db),
    current_user: UserInDB = Depends(get_current_active_user)
):
    try:
        # Base query
        query = db.query(Report)
        
        # Apply organization filter (unless super admin)
        if current_user.role != "super_admin":
            query = query.filter(Report.organization_id == current_user.organization_id)
        
        # Apply status filter
        if status != "all":
            query = query.filter(Report.status == status)
        
        # Apply date range filter
        if date_range != "all":
            now = datetime.utcnow()
            if date_range == "today":
                start = now.replace(hour=0, minute=0, second=0, microsecond=0)
                query = query.filter(Report.created_at >= start)
            elif date_range == "week":
                start = now - timedelta(days=now.weekday())
                start = start.replace(hour=0, minute=0, second=0, microsecond=0)
                query = query.filter(Report.created_at >= start)
            elif date_range == "month":
                start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
                query = query.filter(Report.created_at >= start)
            elif date_range == "quarter":
                quarter = (now.month - 1) // 3 + 1
                start_month = (quarter - 1) * 3 + 1
                start = now.replace(month=start_month, day=1, hour=0, minute=0, second=0, microsecond=0)
                query = query.filter(Report.created_at >= start)
            elif date_range == "year":
                start = now.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
                query = query.filter(Report.created_at >= start)
            elif date_range == "custom" and start_date and end_date:
                try:
                    start = datetime.strptime(start_date, "%Y-%m-%d")
                    end = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)
                    query = query.filter(Report.created_at >= start, Report.created_at <= end)
                except ValueError:
                    raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
        
        # Execute query
        reports = query.all()
        
        if not reports:
            raise HTTPException(status_code=404, detail="No reports found matching criteria")
        
        # Prepare data for export
        export_data = []
        for report in reports:
            report_data = {
                "id": report.id,
                "title": report.title,
                "description": BeautifulSoup(report.description or "", "html.parser").get_text(),
                "category": report.category,
                "status": report.status,
                "author": report.author.name,
                "created_at": report.created_at.isoformat(),
                "updated_at": report.updated_at.isoformat() if report.updated_at else None
            }
            
            # Include template fields if available
            if report.template_data:
                for field_name, field_value in report.template_data.items():
                    if isinstance(field_value, str):
                        field_value = BeautifulSoup(field_value, "html.parser").get_text()
                    report_data[field_name] = field_value
            
            export_data.append(report_data)
        
        # Generate export based on format
        if format == "json":
            return JSONResponse(content=export_data)
        
        elif format == "csv":
            import csv
            from io import StringIO
            
            # Create CSV string
            output = StringIO()
            writer = csv.DictWriter(output, fieldnames=export_data[0].keys())
            writer.writeheader()
            writer.writerows(export_data)
            
            return StreamingResponse(
                iter([output.getvalue()]),
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment;filename=reports_export_{datetime.now().date()}.csv"}
            )
        
        elif format == "excel":
            import pandas as pd
            from io import BytesIO
            
            # Create Excel file
            df = pd.DataFrame(export_data)
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False, sheet_name='Reports')
            
            return StreamingResponse(
                BytesIO(output.getvalue()),
                media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                headers={"Content-Disposition": f"attachment;filename=reports_export_{datetime.now().date()}.xlsx"}
            )
        
        elif format == "pdf":
            from reportlab.lib.pagesizes import letter
            from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.lib import colors
            from io import BytesIO
            
            # Create PDF
            buffer = BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=letter)
            styles = getSampleStyleSheet()
            elements = []
            
            # Add title
            elements.append(Paragraph("Reports Export", styles['Title']))
            
            # Prepare data for table
            if not export_data:
                elements.append(Paragraph("No reports found", styles['BodyText']))
            else:
                # Get all possible keys from the data
                all_keys = set()
                for report in export_data:
                    all_keys.update(report.keys())
                headers = sorted(all_keys)
                
                # Create table data
                table_data = [headers]
                for report in export_data:
                    row = [str(report.get(header, "")) for header in headers]
                    table_data.append(row)
                
                # Create table
                t = Table(table_data)
                t.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,0), colors.grey),
                    ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                    ('ALIGN', (0,0), (-1,-1), 'LEFT'),
                    ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0,0), (-1,0), 10),
                    ('BOTTOMPADDING', (0,0), (-1,0), 12),
                    ('BACKGROUND', (0,1), (-1,-1), colors.beige),
                    ('GRID', (0,0), (-1,-1), 1, colors.black)
                ]))
                elements.append(t)
            
            doc.build(elements)
            buffer.seek(0)
            
            return StreamingResponse(
                buffer,
                media_type="application/pdf",
                headers={"Content-Disposition": f"attachment;filename=reports_export_{datetime.now().date()}.pdf"}
            )
        
        else:
            raise HTTPException(status_code=400, detail="Invalid export format")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
@app.get("/admin/reports-raw")
async def get_raw_reports(db: Session = Depends(get_db)):
    """Development-only endpoint to get raw report data"""
    reports = db.query(Report).all()
    
    def serialize_report(report):
        result = {}
        for k, v in report.__dict__.items():
            if k.startswith('_'):
                continue
            # Handle datetime serialization
            if isinstance(v, datetime):
                result[k] = v.isoformat()
            else:
                result[k] = v
        return result
    
    return JSONResponse(content=[serialize_report(r) for r in reports])
    
@app.get("/notifications", response_model=List[NotificationInDB])
async def get_notifications(
    skip: int = 0,
    limit: int = 10,
    unread_only: bool = False,
    db: Session = Depends(get_db),
    current_user: UserInDB = Depends(get_current_active_user)
):
    query = db.query(Notification).filter(
        Notification.user_id == current_user.id
    ).order_by(
        Notification.created_at.desc()
    )
    
    if unread_only:
        query = query.filter(Notification.is_read == False)
    
    notifications = query.offset(skip).limit(limit).all()
    return notifications

@app.patch("/notifications/{notification_id}/read")
async def mark_notification_as_read(
    notification_id: int,
    db: Session = Depends(get_db),
    current_user: UserInDB = Depends(get_current_active_user)
):
    notification = db.query(Notification).filter(
        Notification.id == notification_id,
        Notification.user_id == current_user.id
    ).first()
    
    if not notification:
        raise HTTPException(status_code=404, detail="Notification not found")
    
    notification.is_read = True
    db.commit()
    
    return {"message": "Notification marked as read"}
async def create_report_notification(
    db: Session,
    report: Report,
    action: str,  # "created", "status_changed"
    current_user: UserInDB
):
    # Determine who should receive the notification
    recipients = []
    
    if action == "created":
        # Notify organization admins when a new report is created
        recipients = db.query(User).filter(
            User.organization_id == report.organization_id,
            User.role == "admin",
            User.id != current_user.id  # Don't notify the creator
        ).all()
        
        title = "New Report Submitted"
        message = f"A new report '{report.title}' has been submitted by {current_user.name}"
        link = f"/reports/{report.id}"
    elif action == "status_changed":
        # Notify the report author when status changes
        if report.author_id != current_user.id:  # Don't notify if the user changed their own report status
            recipients = [report.author]
            
            title = "Report Status Updated"
            message = f"Your report '{report.title}' status has been updated to {report.status}"
            link = f"/reports/{report.id}"
    
    # Create notifications for each recipient
    for recipient in recipients:
        notification = Notification(
            user_id=recipient.id,
            title=title,
            message=message,
            is_read=False,
            link=link
        )
        db.add(notification)
    
    db.commit()

@app.post("/notifications/mark-all-read")
async def mark_all_notifications_as_read(
    db: Session = Depends(get_db),
    current_user: UserInDB = Depends(get_current_active_user)
):
    db.query(Notification).filter(
        Notification.user_id == current_user.id,
        Notification.is_read == False
    ).update({"is_read": True})
    db.commit()
    
    return {"message": "All notifications marked as read"}
@app.post("/chatbot/ask", response_model=Dict[str, str])
async def ask_chatbot(
    message: ChatbotMessage,
    current_user: UserInDB = Depends(get_current_active_user)
):
    """
    Endpoint to interact with the Gemini chatbot
    """
    try:
        # Prepare the prompt with context if available
        prompt = message.message
        if message.context:
            prompt = f"Context: {message.context}\n\nQuestion: {message.message}"

        # Call Gemini API
        response = requests.post(
            f"{GEMINI_API_URL}?key={GEMINI_API_KEY}",
            json={
                "contents": [{
                    "parts": [{
                        "text": prompt
                    }]
                }]
            },
            timeout=30
        )
        response.raise_for_status()

        # Extract the response text
        result = response.json()
        if 'candidates' in result and result['candidates']:
            answer = result['candidates'][0]['content']['parts'][0]['text']
            return {"response": answer}
        else:
            return {"response": "I couldn't generate a response. Please try again."}

    except Exception as e:
        logger.error(f"Error calling Gemini API: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to get response from chatbot"
        )

from fastapi import FastAPI, HTTPException, Depends, status, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, EmailStr, validator
from typing import List, Optional, Dict
from datetime import datetime, timedelta
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
from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime, ForeignKey, func, inspect, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from sqlalchemy.sql import text
import boto3
from botocore.exceptions import ClientError

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
    logo_url = Column(String, nullable=True)
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
        
# Reset database and initialize data
init_default_admin()

# Pydantic models (remain the same as before)
class Token(BaseModel):
    access_token: str
    token_type: str
    requires_org_registration: bool = False

class TokenData(BaseModel):
    email: Optional[str] = None

class UserBase(BaseModel):
    email: EmailStr
    name: str

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
    organization_name: str
    created_at: datetime
    updated_at: Optional[datetime]
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
    
    # Create user without organization (will be set later)
    hashed_password = get_password_hash(password)
    
    # First user becomes admin, others are staff
    is_first_user = db.query(User).count() == 0
    role = "admin" if is_first_user else "staff"
    
    db_user = User(
        email=email,
        name=name,
        hashed_password=hashed_password,
        role=role,
        organization_id=None  # Organization will be set after registration
    )
    
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": db_user.email}, expires_delta=access_token_expires
    )
    
    # All new signups require organization registration
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "requires_org_registration": True
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
        # Upload the voice message to B2
        file_url = await upload_to_b2(voice_message, f"voice_messages/{current_user.id}")
        
        # Save message to database
        db_message = ChatMessage(
            sender_id=current_user.id,
            recipient_id=recipient_id,
            content=f"[VOICE_MESSAGE]{file_url}",
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
        logo_url=org.logo_url,
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
            # Upload new logo to B2
            file_url = await upload_to_b2(logo, "organization_logos")
            
            # Delete old logo if it exists
            if org.logo_url:
                try:
                    await delete_from_b2(org.logo_url)
                except Exception as e:
                    print(f"Error deleting old logo from B2: {e}")
            
            org.logo_url = file_url
            updated = True
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to upload logo: {str(e)}"
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
    
    return db_template

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
return chat messages with dates for older messages, just the way whatsapp does it, be brief;<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ReportHub | Professional Reporting System</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/animate.css/4.1.1/animate.min.css">
   
</head>
<body>
    <div id="app">
        <!-- Login View (initial state) -->
        <div class="login-container" id="login-view">
            <div class="login-left">
                <div class="login-left-content">
                    <h1>Welcome to ReportHub</h1>
                    <p>A modern, professional reporting system designed to streamline your workflow and enhance productivity with secure, role-based access and comprehensive reporting capabilities.</p>
                    <div class="login-features">
                        <div class="login-feature">
                            <i class="fas fa-shield-alt"></i>
                            <span>Secure role-based authentication</span>
                        </div>
                        <div class="login-feature">
                            <i class="fas fa-file-upload"></i>
                            <span>Document attachments for reports</span>
                        </div>
                        <div class="login-feature">
                            <i class="fas fa-chart-line"></i>
                            <span>Comprehensive reporting dashboard</span>
                        </div>
                        <div class="login-feature">
                            <i class="fas fa-comments"></i>
                            <span>Real-time team messaging</span>
                        </div>
                    </div>
                </div>
            </div>
            <div class="login-right">
                <div class="login-form">
                    <div class="login-logo">
                        <h2>ReportHub</h2>
                        <p>Sign in to your account</p>
                    </div>
                    <form id="login-form">
                        <div class="alert alert-danger hidden" id="login-error">
                            <i class="fas fa-exclamation-circle"></i>
                            <div class="alert-content">
                                <div class="alert-title">Login failed</div>
                                <p class="alert-message" id="login-error-message">Invalid email or password</p>
                            </div>
                        </div>
                        <div class="form-group">
                            <label for="email" class="form-label">Email Address</label>
                            <input type="email" id="email" class="form-control" placeholder="Enter your email" required>
                        </div>
                        <div class="form-group">
                            <label for="password" class="form-label">Password</label>
                            <input type="password" id="password" class="form-control" placeholder="Enter your password" required>
                        </div>
                        <div class="login-actions">
                            <div class="remember-me">
                                <input type="checkbox" id="remember-me" class="form-check-input">
                                <label for="remember-me" class="form-check-label">Remember me</label>
                            </div>
                            <a href="#" class="forgot-password">Forgot password?</a>
                        </div>
                        <button type="submit" class="btn btn-primary login-btn" id="login-btn">
                            <span id="login-btn-text">Sign In</span>
                            <span class="spinner hidden" id="login-spinner"></span>
                        </button>
                    </form>
                    <div class="login-footer">
                        Don't have an account? <a href="#" id="show-signup">Sign up</a>
                    </div>
                </div>
            </div>
            <form id="signup-form" class="hidden">
                <div class="alert alert-danger hidden" id="signup-error">
                    <i class="fas fa-exclamation-circle"></i>
                    <div class="alert-content">
                        <div class="alert-title">Signup failed</div>
                        <p class="alert-message" id="signup-error-message">Error during signup</p>
                    </div>
                </div>
                <div class="form-group">
                    <label for="signup-name" class="form-label">Full Name</label>
                    <input type="text" id="signup-name" class="form-control" placeholder="Enter your full name" required>
                </div>
                <div class="form-group">
                    <label for="signup-email" class="form-label">Email Address</label>
                    <input type="email" id="signup-email" class="form-control" placeholder="Enter your email" required>
                </div>
                <div class="form-group">
                    <label for="signup-password" class="form-label">Password</label>
                    <input type="password" id="signup-password" class="form-control" placeholder="Create a password" required minlength="8">
                </div>
                <div class="form-group">
                    <label for="signup-confirm-password" class="form-label">Confirm Password</label>
                    <input type="password" id="signup-confirm-password" class="form-control" placeholder="Confirm your password" required minlength="8">
                </div>
                <button type="submit" class="btn btn-primary login-btn" id="signup-btn">
                    <span id="signup-btn-text">Create Account</span>
                    <span class="spinner hidden" id="signup-spinner"></span>
                </button>
                <div class="login-footer">
                    Already have an account? <a href="#" id="show-login">Sign in</a>
                </div>
            </form>
        </div>

        <!-- OTP Verification Modal -->
        <div class="otp-modal" id="otp-modal">
            <div class="otp-modal-content">
                <h2>Verify Your Email</h2>
                <p>We've sent a 6-digit verification code to your email address. Please enter it below:</p>
                
                <div class="otp-input-container" id="otp-input-container">
                    <input type="text" class="otp-input" maxlength="1" data-index="0">
                    <input type="text" class="otp-input" maxlength="1" data-index="1">
                    <input type="text" class="otp-input" maxlength="1" data-index="2">
                    <input type="text" class="otp-input" maxlength="1" data-index="3">
                    <input type="text" class="otp-input" maxlength="1" data-index="4">
                    <input type="text" class="otp-input" maxlength="1" data-index="5">
                </div>
                
                <div class="alert alert-danger hidden" id="otp-error">
                    <i class="fas fa-exclamation-circle"></i>
                    <div class="alert-content">
                        <div class="alert-title">Verification failed</div>
                        <p class="alert-message" id="otp-error-message">Invalid or expired OTP</p>
                    </div>
                </div>
                
                <button class="btn btn-primary w-100" id="verify-otp-btn">
                    <span id="verify-otp-text">Verify</span>
                    <span class="spinner hidden" id="verify-otp-spinner"></span>
                </button>
                
                <div class="resend-otp">
                    Didn't receive a code? <a href="#" id="resend-otp">Resend code</a>
                </div>
            </div>
        </div>

        <!-- Main App View (hidden initially) -->
        <div class="hidden" id="app-view">
            <!-- Sidebar -->
            <div class="sidebar" id="sidebar">
                <div class="sidebar-header">
    <div class="d-flex align-items-center gap-2">
        <div id="sidebar-org-logo" style="width: 32px; height: 32px; border-radius: 4px; overflow: hidden; background-color: #f5f5f5; display: flex; align-items: center; justify-content: center;">
            <img id="sidebar-org-logo-img" src="" style="max-width: 100%; max-height: 100%; display: none;">
            <i class="fas fa-building" style="font-size: 1rem; color: #ccc;"></i>
        </div>
        <h2 id="sidebar-org-name">ReportHub</h2>
    </div>
    <button class="sidebar-toggle" id="collapse-sidebar">
        <i class="fas fa-chevron-left"></i>
    </button>
</div>
                <div class="sidebar-menu">
                    <div class="menu-category">Main</div>
                    <a href="#" class="menu-item active" data-view="dashboard">
                        <i class="fas fa-tachometer-alt"></i>
                        <span>Dashboard</span>
                    </a>
                    <a href="#" class="menu-item" data-view="reports">
                        <i class="fas fa-file-alt"></i>
                        <span>Reports</span>
                    </a>
                    <a href="#" class="menu-item" data-view="chat">
                        <i class="fas fa-comments"></i>
                        <span>Messages</span>
                        <span class="menu-item-badge hidden" id="unread-messages-count">0</span>
                    </a>
                    <a href="#" class="menu-item" id="invite-org-item">
                        <i class="fas fa-user-plus"></i>
                        <span>Invite Organization</span>
                    </a>
                    <div class="menu-category">Support</div>
                    <a href="#" class="menu-item" data-view="support">
                        <i class="fas fa-headset"></i>
                        <span>Contact Support</span>
                    </a>
                    <div id="admin-menu" class="hidden">
                        <div class="menu-category">Administration</div>
                        <a href="#" class="menu-item" data-view="users">
                            <i class="fas fa-users"></i>
                            <span>User Management</span>
                        </a>
                        <a href="#" class="menu-item" data-view="settings">
                            <i class="fas fa-cog"></i>
                            <span>Settings</span>
                        </a>
                    </div>
                </div>
                <div class="sidebar-footer">
                    <div class="user-profile">
                        <div class="user-avatar" id="user-avatar">A</div>
                        <div class="user-info">
                            <div class="user-name" id="user-name">Admin User</div>
                            <div class="user-role" id="user-role">Administrator</div>
                        </div>
                    </div>
                    <button class="logout-btn" id="logout-btn">
                        <i class="fas fa-sign-out-alt"></i>
                        <i class="fas fa-sign-out-alt"></i>
                    </button>
                </div>
            </div>

            <!-- Main Content -->
            <div class="main-content">
                <!-- Navbar -->
                <div class="navbar">
                    <div class="navbar-left">
                        <button class="sidebar-toggle" id="sidebar-toggle">
                            <i class="fas fa-bars"></i>
                        </button>
                        <div class="search-bar">
                            <i class="fas fa-search"></i>
                            <input type="text" placeholder="Search...">
                        </div>
                    </div>
                    <div class="navbar-right">
                        <button class="theme-toggle" id="theme-toggle">
                            <i class="fas fa-moon"></i>
                        </button>
                        <button class="notification-btn">
                            <i class="fas fa-bell"></i>
                            <span class="notification-badge">3</span>
                        </button>
                        <div class="user-dropdown">
                            <button class="user-dropdown-btn" id="user-dropdown-btn">
                                <div class="user-dropdown-avatar" id="user-dropdown-avatar">A</div>
                                <i class="fas fa-chevron-down"></i>
                            </button>
                            <div class="user-dropdown-menu" id="user-dropdown-menu">
                                <div class="user-dropdown-header">
                                    <div class="user-dropdown-name" id="user-dropdown-name">Admin User</div>
                                    <div class="user-dropdown-email" id="user-dropdown-email">admin@reporthub.com</div>
                                </div>
                                <a href="#" class="user-dropdown-item" data-view="profile">
                                    <i class="fas fa-user"></i>
                                    <span>Profile</span>
                                </a>
                                <a href="#" class="user-dropdown-item">
                                    <i class="fas fa-cog"></i>
                                    <span>Settings</span>
                                </a>
                                <div class="user-dropdown-footer">
                                    <button class="btn btn-outline btn-sm w-100" id="logout-dropdown-btn">
                                        <i class="fas fa-sign-out-alt"></i> Logout
                                    </button>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Content Area -->
                <div class="content">
                    <!-- Dashboard View -->
                    <div class="view" id="dashboard-view">
                        <div class="page-header">
                            <h1 class="page-title">Dashboard</h1>
                            <div class="page-actions">
                                <button class="btn btn-primary" id="new-report-btn">
                                    <i class="fas fa-plus"></i> New Report
                                </button>
                            </div>
                        </div>

                        <div class="grid" style="display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 1.5rem; margin-bottom: 1.5rem;">
                            <div class="card">
                                <div class="card-body">
                                    <div class="d-flex align-items-center justify-content-between">
                                        <div>
                                            <h3 style="font-size: 0.875rem; color: var(--text-secondary); margin-bottom: 0.5rem;">Total Reports</h3>
                                            <p style="font-size: 1.75rem; font-weight: 600; margin-bottom: 0;" id="total-reports-count">0</p>
                                        </div>
                                        <div style="width: 48px; height: 48px; border-radius: 50%; background-color: rgba(67, 97, 238, 0.1); display: flex; align-items: center; justify-content: center;">
                                            <i class="fas fa-file-alt" style="font-size: 1.25rem; color: var(--primary-color);"></i>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div class="card">
                                <div class="card-body">
                                    <div class="d-flex align-items-center justify-content-between">
                                        <div>
                                            <h3 style="font-size: 0.875rem; color: var(--text-secondary); margin-bottom: 0.5rem;">Pending</h3>
                                            <p style="font-size: 1.75rem; font-weight: 600; margin-bottom: 0;" id="pending-reports-count">0</p>
                                        </div>
                                        <div style="width: 48px; height: 48px; border-radius: 50%; background-color: rgba(248, 150, 30, 0.1); display: flex; align-items: center; justify-content: center;">
                                            <i class="fas fa-clock" style="font-size: 1.25rem; color: var(--warning-color);"></i>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div class="card">
                                <div class="card-body">
                                    <div class="d-flex align-items-center justify-content-between">
                                        <div>
                                            <h3 style="font-size: 0.875rem; color: var(--text-secondary); margin-bottom: 0.5rem;">Approved</h3>
                                            <p style="font-size: 1.75rem; font-weight: 600; margin-bottom: 0;" id="approved-reports-count">0</p>
                                        </div>
                                        <div style="width: 48px; height: 48px; border-radius: 50%; background-color: rgba(76, 201, 240, 0.1); display: flex; align-items: center; justify-content: center;">
                                            <i class="fas fa-check-circle" style="font-size: 1.25rem; color: var(--success-color);"></i>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div class="card">
                                <div class="card-body">
                                    <div class="d-flex align-items-center justify-content-between">
                                        <div>
                                            <h3 style="font-size: 0.875rem; color: var(--text-secondary); margin-bottom: 0.5rem;">Rejected</h3>
                                            <p style="font-size: 1.75rem; font-weight: 600; margin-bottom: 0;" id="rejected-reports-count">0</p>
                                        </div>
                                        <div style="width: 48px; height: 48px; border-radius: 50%; background-color: rgba(247, 37, 133, 0.1); display: flex; align-items: center; justify-content: center;">
                                            <i class="fas fa-times-circle" style="font-size: 1.25rem; color: var(--danger-color);"></i>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>

                        <div class="card">
                            <div class="card-header">
                                <h2 class="card-title">Recent Reports</h2>
                                <a href="#" class="btn btn-outline" data-view="reports">View All</a>
                            </div>
                            <div class="card-body">
                                <div class="table-responsive">
                                    <table class="table">
                                        <thead>
                                            <tr>
                                                <th>Report ID</th>
                                                <th>Title</th>
                                                <th>Status</th>
                                                <th>Date</th>
                                                <th>Actions</th>
                                            </tr>
                                        </thead>
                                        <tbody id="dashboard-reports-table-body">
                                            <!-- Reports will be loaded here -->
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- Reports View -->
                    <div class="view hidden" id="reports-view">
                        <div class="page-header">
                            <h1 class="page-title">Reports</h1>
                            <div class="page-actions">
                                <button class="btn btn-primary" id="create-report-btn">
                                    <i class="fas fa-plus"></i> Create Report
                                </button>
                            </div>
                        </div>

                        <div class="card">
                            <div class="card-header">
                                <h2 class="card-title">All Reports</h2>
                                <div class="d-flex align-items-center flex-wrap gap-2">
                                    <select class="form-control form-select" id="report-status-filter" style="width: 150px;">
                                        <option value="">All Status</option>
                                        <option value="pending">Pending</option>
                                        <option value="approved">Approved</option>
                                        <option value="rejected">Rejected</option>
                                    </select>
                                    <input type="text" class="form-control" id="report-search" placeholder="Search reports..." style="width: 200px;">
                                </div>
                            </div>
                            <div class="card-body">
                                <div class="table-responsive">
                                    <table class="table">
                                        <thead>
                                            <tr>
                                                <th>Report ID</th>
                                                <th>Title</th>
                                                <th>Author</th>
                                                <th>Status</th>
                                                <th>Date</th>
                                                <th>Attachments</th>
                                                <th>Actions</th>
                                            </tr>
                                        </thead>
                                        <tbody id="reports-table-body">
                                            <!-- Loading state -->
                                            <tr>
                                                <td colspan="7" class="text-center">
                                                    <div class="d-flex justify-content-center py-4">
                                                        <div class="spinner spinner-primary"></div>
                                                    </div>
                                                </td>
                                            </tr>
                                        </tbody>
                                    </table>
                                </div>
                                <div class="card-footer">
                                    <div class="text-muted">Showing <span id="reports-from">0</span> to <span id="reports-to">0</span> of <span id="reports-total">0</span> entries</div>
                                    <div class="d-flex gap-1">
                                        <button class="btn btn-outline btn-sm" disabled id="reports-prev-btn">
                                            <i class="fas fa-chevron-left"></i> Previous
                                        </button>
                                        <button class="btn btn-outline btn-sm" id="reports-next-btn">
                                            Next <i class="fas fa-chevron-right"></i>
                                        </button>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="view hidden" id="support-view">
    <div class="page-header">
        <h1 class="page-title">Contact Support</h1>
    </div>

    <div class="card">
        <div class="card-body">
            <div class="row">
                <!-- Contact Form Column -->
                <div class="col-md-6">
                    <div class="contact-form-container">
                        <h3 class="section-title">Send us a message</h3>
                        <p class="section-subtitle">Our team typically responds within 24 hours</p>
                        
                        <form id="support-form">
                            <div class="form-group">
                                <label for="support-subject" class="form-label">Subject</label>
                                <select id="support-subject" class="form-control" required>
                                    <option value="">Select a subject</option>
                                    <option value="General Inquiry">General Inquiry</option>
                                    <option value="Technical Issue">Technical Issue</option>
                                    <option value="Feature Request">Feature Request</option>
                                    <option value="Account Help">Account Help</option>
                                    <option value="Bug Report">Bug Report</option>
                                    <option value="Other">Other</option>
                                </select>
                            </div>
                            
                            <div class="form-group">
                                <label for="support-message" class="form-label">Message</label>
                                <textarea id="support-message" class="form-control" rows="5" placeholder="Describe your issue in detail..." required></textarea>
                            </div>
                            
                            <div class="form-group">
                                <label for="support-attachments" class="form-label">Attachments (optional)</label>
                                <div class="file-upload">
                                    <input type="file" id="support-attachments" class="file-upload-input" multiple>
                                    <label for="support-attachments" class="file-upload-label">
                                        <i class="fas fa-cloud-upload-alt"></i>
                                        <div class="file-upload-text">
                                            <strong>Click to upload</strong> or drag and drop
                                            <div style="font-size: 0.75rem; margin-top: 0.25rem;">PNG, JPG, PDF, DOCX (Max 5MB each)</div>
                                        </div>
                                    </label>
                                </div>
                                <div class="file-list" id="support-attachments-list"></div>
                            </div>
                            
                            <button type="submit" class="btn btn-primary" id="submit-support-btn">
                                <span id="submit-support-text">Send Message</span>
                                <span class="spinner hidden" id="submit-support-spinner"></span>
                            </button>
                        </form>
                    </div>
                </div>
                
                <!-- FAQ Column -->
                <div class="col-md-6">
                    <div class="faq-container">
                        <h3 class="section-title">Frequently Asked Questions</h3>
                        <p class="section-subtitle">Quick answers to common questions</p>
                        
                        <div class="accordion" id="faq-accordion">
                            <!-- FAQ Item 1 -->
                            <div class="accordion-item">
                                <h2 class="accordion-header" id="faq-heading-1">
                                    <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#faq-collapse-1" aria-expanded="false" aria-controls="faq-collapse-1">
                                        How do I create a new report?
                                    </button>
                                </h2>
                                <div id="faq-collapse-1" class="accordion-collapse collapse" aria-labelledby="faq-heading-1">
                                    <div class="accordion-body">
                                        To create a new report, click on the "New Report" button in the dashboard or navigate to the Reports section and click "Create Report". Fill in the required fields, add any attachments if needed, and submit the report.
                                    </div>
                                </div>
                            </div>
                            
                            <!-- FAQ Item 2 -->
                            <div class="accordion-item">
                                <h2 class="accordion-header" id="faq-heading-2">
                                    <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#faq-collapse-2" aria-expanded="false" aria-controls="faq-collapse-2">
                                        How can I track the status of my reports?
                                    </button>
                                </h2>
                                <div id="faq-collapse-2" class="accordion-collapse collapse" aria-labelledby="faq-heading-2">
                                    <div class="accordion-body">
                                        You can track report statuses in the Dashboard which shows counts for each status (Pending, Approved, Rejected). The Reports section provides a detailed table view where you can filter by status and view individual report details.
                                    </div>
                                </div>
                            </div>
                            
                            <!-- FAQ Item 3 -->
                            <div class="accordion-item">
                                <h2 class="accordion-header" id="faq-heading-3">
                                    <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#faq-collapse-3" aria-expanded="false" aria-controls="faq-collapse-3">
                                        What file types can I attach to reports?
                                    </button>
                                </h2>
                                <div id="faq-collapse-3" class="accordion-collapse collapse" aria-labelledby="faq-heading-3">
                                    <div class="accordion-body">
                                        You can attach PDF, DOCX, XLSX, JPG, and PNG files. Each file must be under 10MB in size. For support requests, you can also attach these file types with a maximum of 5MB each.
                                    </div>
                                </div>
                            </div>
                            
                            <!-- FAQ Item 4 -->
                            <div class="accordion-item">
                                <h2 class="accordion-header" id="faq-heading-4">
                                    <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#faq-collapse-4" aria-expanded="false" aria-controls="faq-collapse-4">
                                        How do I invite team members to the system?
                                    </button>
                                </h2>
                                <div id="faq-collapse-4" class="accordion-collapse collapse" aria-labelledby="faq-heading-4">
                                    <div class="accordion-body">
                                        Administrators can invite team members by navigating to User Management and clicking "Add User". You can also share an invite link via email or other platforms from the "Invite Organization" section.
                                    </div>
                                </div>
                            </div>
                            
                            <!-- FAQ Item 5 -->
                            <div class="accordion-item">
                                <h2 class="accordion-header" id="faq-heading-5">
                                    <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#faq-collapse-5" aria-expanded="false" aria-controls="faq-collapse-5">
                                        Can I download reports as PDF?
                                    </button>
                                </h2>
                                <div id="faq-collapse-5" class="accordion-collapse collapse" aria-labelledby="faq-heading-5">
                                    <div class="accordion-body">
                                        Yes, you can download any report as PDF by clicking the download icon (↓) next to the report in the Reports table or from the report details view.
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>

                    <!-- Chat View -->
                    <div class="view hidden" id="chat-view">
                        <div class="page-header">
                            <h1 class="page-title">Messages</h1>
                        </div>
                        
                        <div class="card" style="height: calc(100vh - 180px);">
                            <div class="card-body" style="padding: 0; display: flex;">
                                <!-- Users list -->
                                <div class="chat-users-container" style="width: 300px; border-right: 1px solid var(--border-color);">
                                    <div class="chat-users-list" id="chat-users-list">
                                        <!-- Users will be loaded here -->
                                    </div>
                                </div>
                                
                                <!-- Chat area -->
                                <div class="chat-container" style="flex: 1;">
                                    <!-- Empty state -->
                                    <div class="chat-empty-state" id="chat-empty-state">
                                        <div class="chat-empty-icon">
                                            <i class="fas fa-comments"></i>
                                        </div>
                                        <h3>Select a conversation</h3>
                                        <p>Choose a team member from the list to start chatting</p>
                                    </div>
                                    
                                    <!-- Active chat (hidden by default) -->
                                    <div class="hidden" id="active-chat">
                                        <div class="chat-header">
                                            <div class="chat-header-info">
                                                <div class="chat-header-avatar" id="chat-header-avatar">JD</div>
                                                <div>
                                                    <div class="chat-header-name" id="chat-header-name">John Doe</div>
                                                    <div class="chat-header-status">
                                                        <div class="chat-header-status-dot" id="chat-header-status-dot"></div>
                                                        <span id="chat-header-status-text">Online</span>
                                                    </div>
                                                </div>
                                            </div>
                                            <button class="btn btn-outline btn-sm" id="chat-back-btn">
                                                <i class="fas fa-chevron-left"></i>
                                            </button>
                                        </div>
                                        
                                        <div class="chat-messages" id="chat-messages">
                                            <!-- Messages will be loaded here -->
                                            <div class="no-messages-container hidden" id="no-messages-container">
                                                <img src="https://img.icons8.com/fluency/96/000000/nothing-found.png" class="no-messages-image" alt="No messages">
                                                <div class="no-messages-text">No messages yet</div>
                                                <div class="no-messages-subtext">Start the conversation by sending your first message!</div>
                                            </div>
                                        </div>
                                        
                                        <div class="chat-input-container">
    <form class="chat-input-form" id="chat-input-form">
        <div class="chat-input-wrapper">
            <textarea class="chat-input" id="chat-input" placeholder="Type a message..." rows="1"></textarea>
        </div>
        <button type="button" class="chat-action-btn" id="voice-record-btn" title="Record Voice Message">
            <i class="fas fa-microphone"></i>
        </button>
        <button type="submit" class="chat-send-btn" id="chat-send-btn" disabled>
            <i class="fas fa-paper-plane"></i>
        </button>
    </form>
    <div class="voice-recorder hidden" id="voice-recorder">
        <div class="recording-indicator">
            <div class="pulse-animation"></div>
            <span id="recording-time">00:00</span>
        </div>
        <button class="btn btn-danger" id="stop-recording-btn">
            <i class="fas fa-stop"></i> Stop
        </button>
        <div class="recording-controls hidden" id="recording-controls">
            <audio controls id="recorded-audio"></audio>
            <button class="btn btn-success" id="send-voice-message-btn">
                <i class="fas fa-paper-plane"></i> Send
            </button>
            <button class="btn btn-outline" id="cancel-voice-message-btn">
                <i class="fas fa-trash"></i> Cancel
            </button>
        </div>
    </div>
</div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- Users View (Admin only) -->
                    <div class="view hidden" id="users-view">
                        <div class="page-header">
                            <h1 class="page-title">User Management</h1>
                            <div class="page-actions">
                                <button class="btn btn-primary" id="add-user-btn">
                                    <i class="fas fa-plus"></i> Add User
                                </button>
                            </div>
                        </div>

                        <div class="card">
                            <div class="card-header">
                                <h2 class="card-title">All Users</h2>
                                <input type="text" class="form-control" id="user-search" placeholder="Search users..." style="width: 250px;">
                            </div>
                            <div class="card-body">
                                <div class="table-responsive">
                                    <table class="table">
                                        <thead>
                                            <tr>
                                                <th>Name</th>
                                                <th>Email</th>
                                                <th>Role</th>
                                                <th>Status</th>
                                                <th>Last Active</th>
                                                <th>Actions</th>
                                            </tr>
                                        </thead>
                                        <tbody id="users-table-body">
                                            <!-- Loading state -->
                                            <tr>
                                                <td colspan="6" class="text-center">
                                                    <div class="d-flex justify-content-center py-4">
                                                        <div class="spinner spinner-primary"></div>
                                                    </div>
                                                </td>
                                            </tr>
                                        </tbody>
                                    </table>
                                </div>
                                <div class="card-footer">
                                    <div class="text-muted">Showing <span id="users-from">0</span> to <span id="users-to">0</span> of <span id="users-total">0</span> entries</div>
                                    <div class="d-flex gap-1">
                                        <button class="btn btn-outline btn-sm" disabled id="users-prev-btn">
                                            <i class="fas fa-chevron-left"></i> Previous
                                        </button>
                                        <button class="btn btn-outline btn-sm" id="users-next-btn">
                                            Next <i class="fas fa-chevron-right"></i>
                                        </button>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- Settings View (Admin only) -->
                    <div class="view hidden" id="settings-view">
    <div class="page-header">
        <h1 class="page-title">System Settings</h1>
    </div>

    <div class="card">
        <div class="card-header">
            <h2 class="card-title">Organization Settings</h2>
        </div>
        <div class="card-body">
            <form id="org-settings-form">
                <div class="form-group">
                    <label for="org-name" class="form-label">Organization Name</label>
                    <input type="text" id="org-name" class="form-control" placeholder="Enter organization name">
                </div>
                <div class="form-group">
                    <label class="form-label">Organization Logo</label>
                    <div class="d-flex align-items-center gap-3">
                        <div id="org-logo-preview" style="width: 80px; height: 80px; border-radius: 4px; overflow: hidden; background-color: #f5f5f5; display: flex; align-items: center; justify-content: center;">
                            <img id="org-logo-img" src="" style="max-width: 100%; max-height: 100%; display: none;">
                            <i class="fas fa-building" style="font-size: 2rem; color: #ccc;" id="org-logo-placeholder"></i>
                        </div>
                        <div class="file-upload">
                            <input type="file" id="org-logo" class="file-upload-input" accept="image/*">
                            <label for="org-logo" class="file-upload-label">
                                <i class="fas fa-cloud-upload-alt"></i>
                                <div class="file-upload-text">
                                    <strong>Change Logo</strong>
                                    <div style="font-size: 0.75rem; margin-top: 0.25rem;">PNG, JPG (Max 2MB)</div>
                                </div>
                            </label>
                        </div>
                    </div>
                </div>
                <div class="form-group">
                    <div class="d-flex justify-content-end">
                        <button type="submit" class="btn btn-primary" id="save-org-settings-btn">
                            <i class="fas fa-save"></i> Save Changes
                        </button>
                    </div>
                </div>
            </form>
        </div>
    </div>

    <div class="card mt-4">
        <div class="card-header">
            <h2 class="card-title">General Settings</h2>
        </div>
        <div class="card-body">
            <form id="settings-form">
                <div class="form-group">
                    <label for="timezone" class="form-label">Timezone</label>
                    <select id="timezone" class="form-control form-select">
                        <option value="UTC">UTC</option>
                        <option value="EST" selected>Eastern Standard Time (EST)</option>
                        <option value="PST">Pacific Standard Time (PST)</option>
                        <option value="CST">Central Standard Time (CST)</option>
                    </select>
                </div>
                <div class="form-group">
                    <label for="date-format" class="form-label">Date Format</label>
                    <select id="date-format" class="form-control form-select">
                        <option value="YYYY-MM-DD">YYYY-MM-DD</option>
                        <option value="MM/DD/YYYY" selected>MM/DD/YYYY</option>
                        <option value="DD/MM/YYYY">DD/MM/YYYY</option>
                    </select>
                </div>
                <div class="form-group">
                    <div class="d-flex justify-content-end">
                        <button type="submit" class="btn btn-primary" id="save-settings-btn">
                            <i class="fas fa-save"></i> Save Changes
                        </button>
                    </div>
                </div>
            </form>
        </div>
    </div>
</div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Create Report Modal -->
        <div class="modal" id="create-report-modal">
            <div class="modal-content">
                <div class="modal-header">
                    <h3 class="modal-title">Create New Report</h3>
                    <button class="modal-close" id="close-create-report-modal">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
                <div class="modal-body">
                    <form id="create-report-form">
                        <div class="form-group">
                            <label for="report-title" class="form-label">Report Title</label>
                            <input type="text" id="report-title" class="form-control" placeholder="Enter report title" required>
                        </div>
                        <div class="form-group">
                            <label for="report-description" class="form-label">Description</label>
                            <div id="report-description-editor" style="height: 200px;"></div>
                            <input type="hidden" id="report-description" name="report-description">
                        </div>
                        <div class="form-group">
                            <label for="report-category" class="form-label">Category</label>
                            <select id="report-category" class="form-control form-select" required>
                                <option value="">Select a category</option>
                                <option value="financial">Financial</option>
                                <option value="marketing">Marketing</option>
                                <option value="operations">Operations</option>
                                <option value="hr">Human Resources</option>
                                <option value="it">Information Technology</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label class="form-label">Attachments</label>
                            <div class="file-upload">
                                <input type="file" id="report-attachments" class="file-upload-input" multiple>
                                <label for="report-attachments" class="file-upload-label">
                                    <i class="fas fa-cloud-upload-alt"></i>
                                    <div class="file-upload-text">
                                        <strong>Click to upload</strong> or drag and drop
                                        <div style="font-size: 0.75rem; margin-top: 0.25rem;">PDF, DOCX, XLSX, JPG, PNG (Max 10MB each)</div>
                                    </div>
                                </label>
                            </div>
                            <div class="file-list" id="report-attachments-list"></div>
                        </div>
                    </form>
                </div>
                <div class="modal-footer">
                    <button class="btn btn-outline" id="cancel-create-report">Cancel</button>
                    <button class="btn btn-primary" id="submit-create-report">
                        <span id="submit-report-text">Submit Report</span>
                        <span class="spinner hidden" id="submit-report-spinner"></span>
                    </button>
                </div>
            </div>
        </div>
    <!-- Template Management Modal -->
<div class="modal" id="template-management-modal">
    <div class="modal-content" style="max-width: 800px;">
        <div class="modal-header">
            <h3 class="modal-title">Report Templates</h3>
            <button class="modal-close" id="close-template-management-modal">
                <i class="fas fa-times"></i>
            </button>
        </div>
        <div class="modal-body">
            <div class="d-flex justify-content-between mb-3">
                <button class="btn btn-primary" id="create-template-btn">
                    <i class="fas fa-plus"></i> New Template
                </button>
                <div class="search-bar">
                    <i class="fas fa-search"></i>
                    <input type="text" id="template-search" placeholder="Search templates...">
                </div>
            </div>
            
            <div class="table-responsive">
                <table class="table">
                    <thead>
                        <tr>
                            <th>Name</th>
                            <th>Category</th>
                            <th>Fields</th>
                            <th>Last Updated</th>
                            <th>Actions</th>
                        </tr>
                    </thead>
                    <tbody id="templates-table-body">
                        <!-- Templates will be loaded here -->
                    </tbody>
                </table>
            </div>
        </div>
    </div>
</div>

<!-- Create/Edit Template Modal -->
<div class="modal" id="template-editor-modal">
    <div class="modal-content" style="max-width: 800px;">
        <div class="modal-header">
            <h3 class="modal-title" id="template-editor-title">Create New Template</h3>
            <button class="modal-close" id="close-template-editor-modal">
                <i class="fas fa-times"></i>
            </button>
        </div>
        <div class="modal-body">
            <form id="template-editor-form">
                <div class="form-group">
                    <label for="template-name" class="form-label">Template Name</label>
                    <input type="text" id="template-name" class="form-control" required>
                </div>
                <div class="form-group">
                    <label for="template-description" class="form-label">Description</label>
                    <textarea id="template-description" class="form-control" rows="2"></textarea>
                </div>
                <div class="form-group">
                    <label for="template-category" class="form-label">Category</label>
                    <input type="text" id="template-category" class="form-control">
                </div>
                
                <div class="card mt-3">
                    <div class="card-header d-flex justify-content-between align-items-center">
                        <h4 class="mb-0">Fields</h4>
                        <button type="button" class="btn btn-sm btn-primary" id="add-field-btn">
                            <i class="fas fa-plus"></i> Add Field
                        </button>
                    </div>
                    <div class="card-body">
                        <div id="template-fields-container">
                            <!-- Fields will be added here -->
                        </div>
                    </div>
                </div>
            </form>
        </div>
        <div class="modal-footer">
            <button class="btn btn-outline" id="cancel-template-editor">Cancel</button>
            <button class="btn btn-primary" id="save-template-btn">
                <span id="save-template-text">Save Template</span>
                <span class="spinner hidden" id="save-template-spinner"></span>
            </button>
        </div>
    </div>
</div>

<!-- Field Editor Modal (for editing individual fields) -->
<div class="modal" id="field-editor-modal">
    <div class="modal-content" style="max-width: 600px;">
        <div class="modal-header">
            <h3 class="modal-title">Edit Field</h3>
            <button class="modal-close" id="close-field-editor-modal">
                <i class="fas fa-times"></i>
            </button>
        </div>
        <div class="modal-body">
            <form id="field-editor-form">
                <div class="form-group">
                    <label for="field-name" class="form-label">Field Name (internal)</label>
                    <input type="text" id="field-name" class="form-control" required>
                    <small class="text-muted">No spaces or special characters (underscores allowed)</small>
                </div>
                <div class="form-group">
                    <label for="field-label" class="form-label">Display Label</label>
                    <input type="text" id="field-label" class="form-control" required>
                </div>
                <div class="form-group">
                    <label for="field-type" class="form-label">Field Type</label>
                    <select id="field-type" class="form-control" required>
                        <option value="text">Text</option>
                        <option value="number">Number</option>
                        <option value="textarea">Text Area</option>
                        <option value="date">Date</option>
                        <option value="datetime">Date & Time</option>
                        <option value="dropdown">Dropdown</option>
                        <option value="checkbox">Checkbox</option>
                        <option value="radio">Radio Buttons</option>
                        <option value="file">File Upload</option>
                    </select>
                </div>
                <div class="form-group">
                    <div class="form-check">
                        <input type="checkbox" id="field-required" class="form-check-input">
                        <label for="field-required" class="form-check-label">Required Field</label>
                    </div>
                </div>
                <div class="form-group">
                    <label for="field-placeholder" class="form-label">Placeholder Text</label>
                    <input type="text" id="field-placeholder" class="form-control">
                </div>
                <div class="form-group">
                    <label for="field-default-value" class="form-label">Default Value</label>
                    <input type="text" id="field-default-value" class="form-control">
                </div>
                <div class="form-group hidden" id="field-options-group">
                    <label for="field-options" class="form-label">Options (one per line)</label>
                    <textarea id="field-options" class="form-control" rows="4"></textarea>
                </div>
                <div class="form-group">
                    <label for="field-order" class="form-label">Display Order</label>
                    <input type="number" id="field-order" class="form-control" min="0" value="0">
                </div>
            </form>
        </div>
        <div class="modal-footer">
            <button class="btn btn-outline" id="cancel-field-editor">Cancel</button>
            <button class="btn btn-primary" id="save-field-btn">Save Field</button>
        </div>
    </div>
</div>

        <!-- Add User Modal (Admin only) -->
        <div class="modal" id="add-user-modal">
            <div class="modal-content">
                <div class="modal-header">
                    <h3 class="modal-title">Add New User</h3>
                    <button type="button" class="modal-close" id="close-add-user-modal">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
                <div class="modal-body">
                    <form id="add-user-form">
                        <div class="form-group">
                            <label for="user-fullname" class="form-label">Full Name</label>
                            <input type="text" id="user-fullname" name="user-fullname" class="form-control" placeholder="Enter full name" required>
                        </div>
                        <div class="form-group">
                            <label for="user-email" class="form-label">Email Address</label>
                            <input type="email" id="user-email" name="user-email" class="form-control" placeholder="Enter email address" required>
                        </div>
                        <div class="form-group">
                            <label for="user-role" class="form-label">Role</label>
                            <select id="user-role" name="user-role" class="form-control form-select" required>
                                <option value="">Select a role</option>
                                <option value="admin">Administrator</option>
                                <option value="staff">Staff Member</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label for="user-password" class="form-label">Password</label>
                            <input type="password" id="user-password" name="user-password" class="form-control" placeholder="Enter password" required minlength="8">
                        </div>
                        <div class="form-group">
                            <label for="user-confirm-password" class="form-label">Confirm Password</label>
                            <input type="password" id="user-confirm-password" name="user-confirm-password" class="form-control" placeholder="Confirm password" required minlength="8">
                        </div>
                    </form>
                </div>
                <div class="modal-footer">
                    <button type="button" class="btn btn-outline" id="cancel-add-user">Cancel</button>
                    <button type="submit" form="add-user-form" class="btn btn-primary" id="submit-add-user">
                        <span id="submit-user-text">Add User</span>
                        <span class="spinner hidden" id="submit-user-spinner"></span>
                    </button>
                </div>
            </div>
        </div>

        <!-- View Report Modal -->
        <div class="modal" id="view-report-modal">
            <div class="modal-content" style="max-width: 800px;">
                <div class="modal-header">
                    <h3 class="modal-title">Report Details</h3>
                    <button class="modal-close" id="close-view-report-modal">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
                <div class="modal-body">
                    <div class="card" style="margin-bottom: 1.5rem;">
                        <div class="card-body">
                            <h2 id="view-report-title" style="margin-bottom: 1rem;">Quarterly Financial Report</h2>
                            <div style="display: flex; flex-wrap: wrap; gap: 1.5rem; margin-bottom: 1.5rem;">
                                <div>
                                    <div style="font-size: 0.875rem; color: var(--text-secondary); margin-bottom: 0.25rem;">Author</div>
                                    <div id="view-report-author">John Doe</div>
                                </div>
                                <div>
                                    <div style="font-size: 0.875rem; color: var(--text-secondary); margin-bottom: 0.25rem;">Date</div>
                                    <div id="view-report-date">2023-10-15</div>
                                </div>
                                <div>
                                    <div style="font-size: 0.875rem; color: var(--text-secondary); margin-bottom: 0.25rem;">Status</div>
                                    <div id="view-report-status"><span class="badge badge-success">Approved</span></div>
                                </div>
                                <div>
                                    <div style="font-size: 0.875rem; color: var(--text-secondary); margin-bottom: 0.25rem;">Category</div>
                                    <div id="view-report-category">Financial</div>
                                </div>
                            </div>
                            <div>
                                <div style="font-size: 0.875rem; color: var(--text-secondary); margin-bottom: 0.25rem;">Description</div>
                                <div id="view-report-description" style="line-height: 1.7;">
                                    This report contains the financial performance of the company for the third quarter of 2023. It includes revenue, expenses, profit margins, and comparisons with previous quarters.
                                </div>
                            </div>
                            <div class="card hidden" id="template-fields-card" style="margin-top: 1.5rem;">
    <div class="card-header">
        <h2 class="card-title">Form Data</h2>
    </div>
    <div class="card-body" id="template-fields-container">
        <!-- Template fields will be rendered here -->
    </div>
</div>
                        </div>
                    </div>

                    <div class="card">
                        <div class="card-header">
                            <h2 class="card-title">Attachments</h2>
                        </div>
                        <div class="card-body">
                            <div class="file-list" id="view-report-attachments">
                                <div class="text-center text-muted">No attachments</div>
                            </div>
                        </div>
                    </div>

                    <div class="card hidden" id="admin-report-actions" style="margin-top: 1.5rem;">
                        <div class="card-header">
                            <h2 class="card-title">Admin Actions</h2>
                        </div>
                        <div class="card-body">
                            <div class="form-group">
                                <label for="report-status-change" class="form-label">Change Status</label>
                                <select id="report-status-change" class="form-control form-select">
                                    <option value="pending">Pending</option>
                                    <option value="approved">Approved</option>
                                    <option value="rejected">Rejected</option>
                                </select>
                            </div>
                            <div class="form-group">
                                <label for="admin-comments" class="form-label">Comments</label>
                                <textarea id="admin-comments" class="form-control form-textarea" placeholder="Enter your comments"></textarea>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="modal-footer">
                    <button class="btn btn-outline" id="close-report-modal">Close</button>
                    <button class="btn btn-primary hidden" id="save-report-changes">
                        <span id="save-changes-text">Save Changes</span>
                        <span class="spinner hidden" id="save-changes-spinner"></span>
                    </button>
                </div>
            </div>
        </div>
        <!-- Invite Organization Modal -->
<div class="modal" id="invite-org-modal">
    <div class="modal-content" style="max-width: 600px;">
        <div class="modal-header">
            <h3 class="modal-title">Invite to ReportHub</h3>
            <button class="modal-close" id="close-invite-org-modal">
                <i class="fas fa-times"></i>
            </button>
        </div>
        <div class="modal-body">
            <div class="invite-content">
                <div class="invite-hero">
                    <div class="invite-icon">
                        <i class="fas fa-rocket"></i>
                    </div>
                    <h3>Share ReportHub with your organization</h3>
                    <p>Collaborate with your team using our professional reporting platform</p>
                </div>
                
                <div class="invite-link-container">
                    <label class="form-label">Invitation Link</label>
                    <div class="input-group">
                        <input type="text" id="invite-link" class="form-control" readonly value="https://reporthub.com/signup?ref=org-invite">
                        <button class="btn btn-outline" id="copy-invite-link">
                            <i class="fas fa-copy"></i> Copy
                        </button>
                    </div>
                </div>
                
                <div class="invite-features">
                    <h4>Why your organization will love ReportHub:</h4>
                    <ul>
                        <li><i class="fas fa-check-circle"></i> Secure, role-based access control</li>
                        <li><i class="fas fa-check-circle"></i> Real-time collaboration and messaging</li>
                        <li><i class="fas fa-check-circle"></i> Comprehensive reporting dashboard</li>
                        <li><i class="fas fa-check-circle"></i> Document attachments and versioning</li>
                        <li><i class="fas fa-check-circle"></i> Automated report generation</li>
                    </ul>
                </div>
                
                <div class="social-share">
                    <h4>Share via:</h4>
                    <div class="social-buttons">
                        <button class="social-btn email" id="share-email">
                            <i class="fas fa-envelope"></i> Email
                        </button>
                        <button class="social-btn whatsapp" id="share-whatsapp">
                            <i class="fab fa-whatsapp"></i> WhatsApp
                        </button>
                        <button class="social-btn linkedin" id="share-linkedin">
                            <i class="fab fa-linkedin"></i> LinkedIn
                        </button>
                        <button class="social-btn twitter" id="share-twitter">
                            <i class="fab fa-twitter"></i> Twitter
                        </button>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>

        <!-- Organization Registration Modal -->
        <div class="modal" id="org-registration-modal">
            <div class="modal-content">
                <div class="modal-header">
                    <h3 class="modal-title">Register Your Organization</h3>
                </div>
                <div class="modal-body">
                    <form id="org-registration-form">
                        <div class="form-group">
                            <label for="organization-name" class="form-label">Organization Name</label>
                            <input type="text" id="organization-name" class="form-control" placeholder="Enter your organization name" required>
                            <small class="text-muted">This will be visible to all members of your organization</small>
                        </div>
                    </form>
                </div>
                <div class="modal-footer">
                    <button class="btn btn-primary" id="submit-org-registration">
                        <span id="submit-org-text">Register Organization</span>
                        <span class="spinner hidden" id="submit-org-spinner"></span>
                    </button>
                </div>
            </div>
        </div>
    </div>

    <script>
        // API Configuration
        const API_BASE_URL = 'https://reporting-api-uvze.onrender.com';
        
        // State Management
        let currentUser = null;
        let currentTheme = localStorage.getItem('theme') || 'light';
        let reports = [];
        let users = [];
        let selectedReportId = null;
        let isSidebarCollapsed = localStorage.getItem('sidebarCollapsed') === 'true';
        let currentReportsPage = 1;
        let currentUsersPage = 1;
        const reportsPerPage = 10;
        const usersPerPage = 10;
        let mediaRecorder;
        let audioChunks = [];
        let recordingStartTime;
        let recordingTimer;
        
        // Chat state
        let chatSocket = null;
        let activeChatUserId = null;
        let chatUsers = [];
        let unreadMessagesCount = 0;
        let currentTemplateId = null;
        let currentFieldIndex = null;
        let templates = [];
        let currentTemplateFields = [];


        // OTP verification state
        let otpVerificationEmail = null;

        // DOM Elements
        const loginView = document.getElementById('login-view');
        const appView = document.getElementById('app-view');
        const loginForm = document.getElementById('login-form');
        const signupForm = document.getElementById('signup-form');
        const showSignupLink = document.getElementById('show-signup');
        const showLoginLink = document.getElementById('show-login');
        const signupBtn = document.getElementById('signup-btn');
        const signupBtnText = document.getElementById('signup-btn-text');
        const signupSpinner = document.getElementById('signup-spinner');
        const signupError = document.getElementById('signup-error');
        const signupErrorMessage = document.getElementById('signup-error-message');
        const loginError = document.getElementById('login-error');
        const loginErrorMessage = document.getElementById('login-error-message');
        const emailInput = document.getElementById('email');
        const passwordInput = document.getElementById('password');
        const loginBtn = document.getElementById('login-btn');
        const loginBtnText = document.getElementById('login-btn-text');
        const loginSpinner = document.getElementById('login-spinner');
        const logoutBtn = document.getElementById('logout-btn');
        const logoutDropdownBtn = document.getElementById('logout-dropdown-btn');
        const sidebar = document.getElementById('sidebar');
        const sidebarToggle = document.getElementById('sidebar-toggle');
        const collapseSidebar = document.getElementById('collapse-sidebar');
        const themeToggle = document.getElementById('theme-toggle');
        const adminMenu = document.getElementById('admin-menu');
        const userAvatar = document.getElementById('user-avatar');
        const userName = document.getElementById('user-name');
        const userRole = document.getElementById('user-role');
        const userDropdownBtn = document.getElementById('user-dropdown-btn');
        const userDropdownMenu = document.getElementById('user-dropdown-menu');
        const userDropdownAvatar = document.getElementById('user-dropdown-avatar');
        const userDropdownName = document.getElementById('user-dropdown-name');
        const userDropdownEmail = document.getElementById('user-dropdown-email');
        const views = document.querySelectorAll('.view');
        const menuItems = document.querySelectorAll('.menu-item');
        const newReportBtn = document.getElementById('new-report-btn');
        const createReportBtn = document.getElementById('create-report-btn');
        const createReportModal = document.getElementById('create-report-modal');
        const closeCreateReportModal = document.getElementById('close-create-report-modal');
        const cancelCreateReport = document.getElementById('cancel-create-report');
        const submitCreateReport = document.getElementById('submit-create-report');
        const submitReportText = document.getElementById('submit-report-text');
        const submitReportSpinner = document.getElementById('submit-report-spinner');
        const createReportForm = document.getElementById('create-report-form');
        const reportAttachments = document.getElementById('report-attachments');
        const reportAttachmentsList = document.getElementById('report-attachments-list');
        const addUserBtn = document.getElementById('add-user-btn');
        const addUserModal = document.getElementById('add-user-modal');
        const closeAddUserModal = document.getElementById('close-add-user-modal');
        const cancelAddUser = document.getElementById('cancel-add-user');
        const submitAddUser = document.getElementById('submit-add-user');
        const submitUserText = document.getElementById('submit-user-text');
        const submitUserSpinner = document.getElementById('submit-user-spinner');
        const addUserForm = document.getElementById('add-user-form');
        const reportsTableBody = document.getElementById('reports-table-body');
        const usersTableBody = document.getElementById('users-table-body');
        const viewReportModal = document.getElementById('view-report-modal');
        const closeViewReportModal = document.getElementById('close-view-report-modal');
        const closeReportModal = document.getElementById('close-report-modal');
        const saveReportChanges = document.getElementById('save-report-changes');
        const saveChangesText = document.getElementById('save-changes-text');
        const saveChangesSpinner = document.getElementById('save-changes-spinner');
        const viewReportTitle = document.getElementById('view-report-title');
        const viewReportAuthor = document.getElementById('view-report-author');
        const viewReportDate = document.getElementById('view-report-date');
        const viewReportStatus = document.getElementById('view-report-status');
        const viewReportCategory = document.getElementById('view-report-category');
        const viewReportDescription = document.getElementById('view-report-description');
        const viewReportAttachments = document.getElementById('view-report-attachments');
        const adminReportActions = document.getElementById('admin-report-actions');
        const reportStatusChange = document.getElementById('report-status-change');
        const adminComments = document.getElementById('admin-comments');
        const settingsForm = document.getElementById('settings-form');
        const saveSettingsBtn = document.getElementById('save-settings-btn');
        const reportStatusFilter = document.getElementById('report-status-filter');
        const reportSearch = document.getElementById('report-search');
        const userSearch = document.getElementById('user-search');
        const reportsPrevBtn = document.getElementById('reports-prev-btn');
        const reportsNextBtn = document.getElementById('reports-next-btn');
        const usersPrevBtn = document.getElementById('users-prev-btn');
        const usersNextBtn = document.getElementById('users-next-btn');
        const dashboardReportsTableBody = document.getElementById('dashboard-reports-table-body');
        const totalReportsCount = document.getElementById('total-reports-count');
        const pendingReportsCount = document.getElementById('pending-reports-count');
        const approvedReportsCount = document.getElementById('approved-reports-count');
        const rejectedReportsCount = document.getElementById('rejected-reports-count');
        const orgRegistrationModal = document.getElementById('org-registration-modal');
        const orgRegistrationForm = document.getElementById('org-registration-form');
        const organizationNameInput = document.getElementById('organization-name');
        const submitOrgBtn = document.getElementById('submit-org-registration');
        const submitOrgText = document.getElementById('submit-org-text');
        const submitOrgSpinner = document.getElementById('submit-org-spinner');
        
        // Chat elements
        const chatUsersList = document.getElementById('chat-users-list');
        const chatEmptyState = document.getElementById('chat-empty-state');
        const activeChat = document.getElementById('active-chat');
        const chatMessages = document.getElementById('chat-messages');
        const chatInputForm = document.getElementById('chat-input-form');
        const chatInput = document.getElementById('chat-input');
        const chatSendBtn = document.getElementById('chat-send-btn');
        const chatBackBtn = document.getElementById('chat-back-btn');
        const chatHeaderName = document.getElementById('chat-header-name');
        const chatHeaderAvatar = document.getElementById('chat-header-avatar');
        const chatHeaderStatusDot = document.getElementById('chat-header-status-dot');
        const chatHeaderStatusText = document.getElementById('chat-header-status-text');
        const unreadMessagesBadge = document.getElementById('unread-messages-count');
        
        // OTP verification elements
        const otpModal = document.getElementById('otp-modal');
        const otpInputs = document.querySelectorAll('.otp-input');
        const verifyOtpBtn = document.getElementById('verify-otp-btn');
        const verifyOtpText = document.getElementById('verify-otp-text');
        const verifyOtpSpinner = document.getElementById('verify-otp-spinner');
        const otpError = document.getElementById('otp-error');
        const otpErrorMessage = document.getElementById('otp-error-message');
        const resendOtpLink = document.getElementById('resend-otp');

        //Invite DOM elements
        const inviteOrgItem = document.getElementById('invite-org-item');
        const inviteOrgModal = document.getElementById('invite-org-modal');
        const closeInviteOrgModal = document.getElementById('close-invite-org-modal');
        const inviteLinkInput = document.getElementById('invite-link');
        const copyInviteLinkBtn = document.getElementById('copy-invite-link');
        const shareEmailBtn = document.getElementById('share-email');
        const shareWhatsappBtn = document.getElementById('share-whatsapp');
        const shareLinkedinBtn = document.getElementById('share-linkedin');
        const shareTwitterBtn = document.getElementById('share-twitter');


        // Initialize the application
        function init() {
            applyTheme();
            setupSidebar();
            setupEventListeners();
            checkAuth();
            addTemplateManagementButton();
            initTemplateManagement();
        }

        // Apply the current theme
        function applyTheme() {
            document.documentElement.setAttribute('data-theme', currentTheme);
            localStorage.setItem('theme', currentTheme);
            
            if (currentTheme === 'dark') {
                themeToggle.innerHTML = '<i class="fas fa-sun"></i>';
            } else {
                themeToggle.innerHTML = '<i class="fas fa-moon"></i>';
            }
        }

        // Setup sidebar state
        function setupSidebar() {
            if (isSidebarCollapsed) {
                sidebar.classList.add('collapsed');
            }
        }

        // Toggle sidebar collapse
        function toggleSidebarCollapse() {
            isSidebarCollapsed = !isSidebarCollapsed;
            sidebar.classList.toggle('collapsed');
            localStorage.setItem('sidebarCollapsed', isSidebarCollapsed);
        }

        // Toggle between light and dark theme
        function toggleTheme() {
            currentTheme = currentTheme === 'light' ? 'dark' : 'light';
            applyTheme();
        }

        // Setup event listeners
        function setupEventListeners() {
            // Toggle between login and signup forms
            showSignupLink?.addEventListener('click', (e) => {
                e.preventDefault();
                loginForm.classList.add('hidden');
                signupForm.classList.remove('hidden');
                signupForm.reset();
                signupError.classList.add('hidden');
            });
            
            showLoginLink?.addEventListener('click', (e) => {
                e.preventDefault();
                signupForm.classList.add('hidden');
                loginForm.classList.remove('hidden');
            });
            
            // Login form submission
            loginForm.addEventListener('submit', handleLogin);
            
            // Signup form submission
            signupForm.addEventListener('submit', handleSignup);
            
            // Organization registration
            submitOrgBtn.addEventListener('click', handleOrgRegistration);
            
            // Logout buttons
            logoutBtn.addEventListener('click', handleLogout);
            logoutDropdownBtn.addEventListener('click', handleLogout);
            
            // Theme toggle
            themeToggle.addEventListener('click', toggleTheme);
            
            // Sidebar toggle
            sidebarToggle.addEventListener('click', () => {
                sidebar.classList.toggle('active');
            });
            
            // Collapse sidebar
            collapseSidebar.addEventListener('click', toggleSidebarCollapse);
            
            // User dropdown
            userDropdownBtn.addEventListener('click', () => {
                userDropdownMenu.classList.toggle('active');
            });
            
            // Close dropdown when clicking outside
            document.addEventListener('click', (e) => {
                if (!userDropdownBtn.contains(e.target) && !userDropdownMenu.contains(e.target)) {
                    userDropdownMenu.classList.remove('active');
                }
                if (e.target.closest('.download-report-btn')) {
                    const btn = e.target.closest('.download-report-btn');
                    const reportId = btn.getAttribute('data-id');
                    downloadReportAsPDF(reportId);
                }
            });
            
            // Menu items for view switching
            menuItems.forEach(item => {
                item.addEventListener('click', (e) => {
                    e.preventDefault();
                    const viewId = item.getAttribute('data-view');
                    switchView(viewId);
                    
                    // Update active state
                    menuItems.forEach(i => i.classList.remove('active'));
                    item.classList.add('active');
                });
            });
            
            // New report button
            newReportBtn.addEventListener('click', () => {
                createReportModal.classList.add('active');
            });
            
            createReportBtn.addEventListener('click', () => {
                createReportModal.classList.add('active');
            });
            
            // Close create report modal
            closeCreateReportModal.addEventListener('click', () => {
                createReportModal.classList.remove('active');
            });
            
            cancelCreateReport.addEventListener('click', () => {
                createReportModal.classList.remove('active');
            });
            
            // Submit create report form
            submitCreateReport.addEventListener('click', handleCreateReport);
            
            // File upload for report attachments
            reportAttachments.addEventListener('change', handleFileUpload);
            
            // Add user button (admin only)
            addUserBtn.addEventListener('click', () => {
                addUserModal.classList.add('active');
            });
            document.getElementById('support-form')?.addEventListener('submit', handleSupportFormSubmit);
            
            // Close add user modal
            closeAddUserModal.addEventListener('click', () => {
                addUserModal.classList.remove('active');
            });
            
            cancelAddUser.addEventListener('click', () => {
                addUserModal.classList.remove('active');
            });
            
            // Submit add user form
            addUserForm.addEventListener('submit', handleAddUser);
            
            // Close view report modal
            closeViewReportModal.addEventListener('click', () => {
                viewReportModal.classList.remove('active');
            });
            
            closeReportModal.addEventListener('click', () => {
                viewReportModal.classList.remove('active');
            });
            
            // Save report changes (admin only)
            saveReportChanges.addEventListener('click', handleSaveReportChanges);
            
            // Save settings
            settingsForm.addEventListener('submit', (e) => {
                e.preventDefault();
                handleSaveSettings();
            });
            
            // View report buttons (delegated event)
            document.addEventListener('click', (e) => {
                if (e.target.closest('.view-report-btn')) {
                    const btn = e.target.closest('.view-report-btn');
                    const reportId = btn.getAttribute('data-id');
                    viewReport(reportId);
                }
                
                // Edit report buttons
                if (e.target.closest('.edit-report-btn')) {
                    const btn = e.target.closest('.edit-report-btn');
                    const reportId = btn.getAttribute('data-id');
                    editReport(reportId);
                }
                
                // Delete report buttons
                if (e.target.closest('.delete-report-btn')) {
                    const btn = e.target.closest('.delete-report-btn');
                    const reportId = btn.getAttribute('data-id');
                    deleteReport(reportId);
                }
                
                // Edit user buttons
                if (e.target.closest('.edit-user-btn')) {
                    const btn = e.target.closest('.edit-user-btn');
                    const userId = btn.getAttribute('data-id');
                    editUser(userId);
                }
                
                // Delete user buttons
                if (e.target.closest('.delete-user-btn')) {
                    const btn = e.target.closest('.delete-user-btn');
                    const userId = btn.getAttribute('data-id');
                    deleteUser(userId);
                }
            });
            
            // Report status filter
            reportStatusFilter.addEventListener('change', () => {
                currentReportsPage = 1;
                loadReports();
            });
            
            // Report search
            reportSearch.addEventListener('input', debounce(() => {
                currentReportsPage = 1;
                loadReports();
            }, 500));
            
            // User search
            userSearch.addEventListener('input', debounce(() => {
                currentUsersPage = 1;
                loadUsers();
            }, 500));
            
            // Pagination buttons
            reportsPrevBtn.addEventListener('click', () => {
                if (currentReportsPage > 1) {
                    currentReportsPage--;
                    loadReports();
                }
            });
            
            reportsNextBtn.addEventListener('click', () => {
                currentReportsPage++;
                loadReports();
            });
            
            usersPrevBtn.addEventListener('click', () => {
                if (currentUsersPage > 1) {
                    currentUsersPage--;
                    loadUsers();
                }
            });
            
            usersNextBtn.addEventListener('click', () => {
                currentUsersPage++;
                loadUsers();
            });
            
            // Chat input
            chatInput.addEventListener('input', () => {
                chatSendBtn.disabled = chatInput.value.trim() === '';
            });
            
            // Chat form submission
            chatInputForm.addEventListener('submit', (e) => {
                e.preventDefault();
                sendMessage();
            });
            
            // Chat back button (for mobile)
            chatBackBtn.addEventListener('click', () => {
                activeChat.classList.add('hidden');
                chatEmptyState.classList.remove('hidden');
                activeChatUserId = null;
            });
            
            // OTP verification inputs
            otpInputs.forEach((input, index) => {
                input.addEventListener('input', (e) => {
                    if (e.target.value.length === 1) {
                        if (index < otpInputs.length - 1) {
                            otpInputs[index + 1].focus();
                        }
                    }
                });
                
                input.addEventListener('keydown', (e) => {
                    if (e.key === 'Backspace' && e.target.value.length === 0) {
                        if (index > 0) {
                            otpInputs[index - 1].focus();
                        }
                    }
                });
            });
            
            // Verify OTP button
            verifyOtpBtn.addEventListener('click', verifyOtp);
            
            // Resend OTP link
            resendOtpLink.addEventListener('click', (e) => {
                e.preventDefault();
                resendOtp();
            });
            
        }

        // Debounce function for search inputs
        function debounce(func, wait) {
            let timeout;
            return function() {
                const context = this;
                const args = arguments;
                clearTimeout(timeout);
                timeout = setTimeout(() => {
                    func.apply(context, args);
                }, wait);
            };
        }

        // Check authentication status
        function checkAuth() {
            const token = localStorage.getItem('token');
            if (token) {
                // Show loading state
                loginBtnText.textContent = 'Loading...';
                loginSpinner.classList.remove('hidden');
                
                // Validate token with backend
                fetch(`${API_BASE_URL}/auth/me`, {
                    headers: {
                        'Authorization': `Bearer ${token}`
                    }
                })
                .then(response => {
                    if (response.ok) {
                        return response.json();
                    } else {
                        throw new Error('Invalid token');
                    }
                })
                .then(data => {
                    currentUser = data;
                    setupUIForUser();
                    loadInitialData();
                    loginView.classList.add('hidden');
                    appView.classList.remove('hidden');
                    
                    // Initialize WebSocket connection for chat
                    initChatConnection();
                })
                .catch(error => {
                    console.error('Authentication check failed:', error);
                    localStorage.removeItem('token');
                    loginView.classList.remove('hidden');
                    appView.classList.add('hidden');
                })
                .finally(() => {
                    loginBtnText.textContent = 'Sign In';
                    loginSpinner.classList.add('hidden');
                });
            }
        }

        // Initialize WebSocket connection for chat
        function initChatConnection() {
            if (!currentUser) return;
            
            const token = localStorage.getItem('token');
            if (!token) return;
            
            // Create WebSocket connection
            const wsProtocol = window.location.protocol === 'https:' ? 'wss://' : 'ws://';
            const wsUrl = API_BASE_URL.replace(/^https?:\/\//, '');
            chatSocket = new WebSocket(`${wsProtocol}${wsUrl}/ws/chat?token=${token}`);
            
            chatSocket.onopen = () => {
                console.log('WebSocket connection established');
            };
            
            chatSocket.onmessage = (event) => {
                const message = JSON.parse(event.data);
                handleWebSocketMessage(message);
            };
            
            chatSocket.onclose = () => {
                console.log('WebSocket connection closed');
                // Attempt to reconnect after a delay
                setTimeout(initChatConnection, 5000);
            };
            
            chatSocket.onerror = (error) => {
                console.error('WebSocket error:', error);
            };
        }

        // Handle WebSocket messages
        function handleWebSocketMessage(message) {
            switch (message.type) {
                case 'message':
                    handleIncomingMessage(message.message);
                    break;
                case 'user_status':
                    updateUserStatus(message.user_id, message.status);
                    break;
                default:
                    console.log('Unknown message type:', message.type);
            }
        }

        // Handle incoming chat message
        function handleIncomingMessage(message) {
            // If the message is for the currently active chat
            if (activeChatUserId === message.sender_id) {
                // Add the message to the chat
                addMessageToChat(message, 'received');
                
                // Mark as read
                if (chatSocket && chatSocket.readyState === WebSocket.OPEN) {
                    chatSocket.send(JSON.stringify({
                        type: 'mark_read',
                        sender_id: message.sender_id
                    }));
                }
            } else {
                // Update unread count for the user
                updateUnreadCount(message.sender_id, 1);
                
                // Show notification
                const sender = chatUsers.find(u => u.id === message.sender_id);
                if (sender) {
                    showAlert(`New message from ${sender.name}`, 'info');
                }
            }
            
            // Scroll to bottom of messages
            scrollChatToBottom();
        }
        chatInput.addEventListener('input', function() {
  // Enable/disable send button
  chatSendBtn.disabled = chatInput.value.trim() === '';
  
  // Auto-resize textarea
  this.style.height = 'auto';
  this.style.height = (this.scrollHeight) + 'px';
  
  // Scroll to bottom of the wrapper
  const wrapper = this.closest('.chat-input-wrapper');
  wrapper.scrollTop = wrapper.scrollHeight;
});

        // Update user status (online/offline)
        function updateUserStatus(userId, status) {
            // Update in chat users list
            const userElement = document.querySelector(`.chat-user[data-id="${userId}"]`);
            if (userElement) {
                const statusDot = userElement.querySelector('.chat-user-status-dot');
                if (statusDot) {
                    statusDot.classList.remove('online', 'offline');
                    statusDot.classList.add(status);
                }
            }
            
            // Update in active chat header if this is the current chat
            if (activeChatUserId === userId) {
                chatHeaderStatusDot.className = 'chat-header-status-dot';
                chatHeaderStatusDot.classList.add(status);
                chatHeaderStatusText.textContent = status === 'online' ? 'Online' : 'Offline';
            }
        }

        // Update unread message count for a user
        function updateUnreadCount(userId, count) {
            // Update in chat users list
            const userElement = document.querySelector(`.chat-user[data-id="${userId}"]`);
            if (userElement) {
                const unreadElement = userElement.querySelector('.chat-user-unread');
                if (unreadElement) {
                    let currentCount = parseInt(unreadElement.textContent) || 0;
                    currentCount += count;
                    if (currentCount > 0) {
                        unreadElement.textContent = currentCount;
                        unreadElement.style.display = 'flex';
                    } else {
                        unreadElement.style.display = 'none';
                    }
                }
            }
            
            // Update total unread count in sidebar
            updateTotalUnreadCount();
        }

        // Update total unread count in sidebar
        function updateTotalUnreadCount() {
            let totalUnread = 0;
            document.querySelectorAll('.chat-user-unread').forEach(el => {
                totalUnread += parseInt(el.textContent) || 0;
            });
            
            unreadMessagesCount = totalUnread;
            const unreadBadge = document.getElementById('unread-messages-count');
            if (unreadBadge) {
                if (totalUnread > 0) {
                    unreadBadge.textContent = totalUnread;
                    unreadBadge.classList.remove('hidden');
                } else {
                    unreadBadge.classList.add('hidden');
                }
            }
        }

        // Send a chat message
        // Send a chat message
function sendMessage() {
    if (!activeChatUserId || !chatInput.value.trim()) return;
    
    const message = {
        type: 'message',
        recipient_id: activeChatUserId,
        content: chatInput.value.trim()
    };
    
    if (chatSocket && chatSocket.readyState === WebSocket.OPEN) {
        chatSocket.send(JSON.stringify(message));
        
        // Add the message to the chat immediately (optimistic UI)
        const tempMessage = {
            id: Date.now(), // Temporary ID
            sender_id: currentUser.id,
            recipient_id: activeChatUserId,
            content: message.content,
            timestamp: new Date().toISOString(),
            status: 'sending'
        };
        
        addMessageToChat(tempMessage, 'sent');
        chatInput.value = '';
        chatSendBtn.disabled = true;
        
        // Reset textarea height
        chatInput.style.height = 'auto';
        
        // Scroll to bottom
        scrollChatToBottom();
    }
    updateMessageVisibility();
}
        // Add a message to the chat UI
function addMessageToChat(message, type) {
    const messageElement = document.createElement('div');
    messageElement.className = `message message-${type} message-animate`;
    
    // Check if this is a voice message
    const isVoiceMessage = message.content && (message.content.startsWith('[VOICE_MESSAGE]') || message.content.endsWith('.wav'));
    
    if (isVoiceMessage) {
        // Extract audio URL (remove [VOICE_MESSAGE] prefix if present)
        let audioUrl = message.content;
        if (audioUrl.startsWith('[VOICE_MESSAGE]')) {
            audioUrl = audioUrl.replace('[VOICE_MESSAGE]', '');
        }
        
        // Create voice message element
        messageElement.innerHTML = `
  <div class="voice-message-container ${type}">
    <div class="voice-message-controls">
      <button class="play-pause-btn" onclick="togglePlayPause(this)">
        <i class="fas fa-play"></i>
      </button>
      <div class="voice-progress">
        <div class="voice-progress-bar"></div>
      </div>
      <div class="voice-duration">00:00</div>
      <audio src="${API_BASE_URL}${audioUrl}" preload="metadata"></audio>
    </div>
    <div class="message-time">
      ${formatTime(message.timestamp)}
      ${type === 'sent' ? `
      <span class="message-status ${message.status}">
        ${message.status === 'sending' ? '<i class="fas fa-clock"></i>' : 
          message.status === 'delivered' ? '<i class="fas fa-check"></i>' : 
          message.status === 'read' ? '<i class="fas fa-check-double"></i>' : ''}
      </span>
      ` : ''}
    </div>
  </div>
`;
        
        // Set up audio element
        // Set up audio element
const audioElement = messageElement.querySelector('audio');
const progressBar = messageElement.querySelector('.voice-progress-bar');
const durationDisplay = messageElement.querySelector('.voice-duration');

// Fallback duration in case metadata loading fails
let duration = 0;

// Try to get duration from metadata
audioElement.addEventListener('loadedmetadata', function() {
  if (!isNaN(audioElement.duration)) {
    duration = audioElement.duration;
    durationDisplay.textContent = formatDuration(duration);
  }
});

// Fallback if metadata doesn't load
setTimeout(() => {
  if (duration === 0) {
    durationDisplay.textContent = "00:00";
  }
}, 1000);

audioElement.addEventListener('timeupdate', function() {
  if (!isNaN(audioElement.duration)) {
    const progress = (audioElement.currentTime / audioElement.duration) * 100;
    progressBar.style.width = `${progress}%`;
    
    // Update current time display
    const currentTime = Math.floor(audioElement.currentTime);
    durationDisplay.textContent = formatDuration(currentTime);
  }
});

audioElement.addEventListener('ended', function() {
  const btn = messageElement.querySelector('.play-pause-btn');
  btn.innerHTML = '<i class="fas fa-play"></i>';
  progressBar.style.width = '0%';
  if (!isNaN(duration)) {
    durationDisplay.textContent = formatDuration(duration);
  } else {
    durationDisplay.textContent = "00:00";
  }
});
    } else {
        // Handle text message
        const sender = chatUsers.find(u => u.id === message.sender_id);
        const isAdmin = sender && sender.role === 'admin';
        
        messageElement.innerHTML = `
            <div>${message.content}</div>
            <div class="message-time">
                ${formatTime(message.timestamp)}
                ${type === 'sent' ? `
                <span class="message-status ${message.status}">
                    ${message.status === 'sending' ? '<i class="fas fa-clock"></i>' : 
                      message.status === 'delivered' ? '<i class="fas fa-check"></i>' : 
                      message.status === 'read' ? '<i class="fas fa-check-double"></i>' : ''}
                </span>
                ` : ''}
                ${isAdmin ? '<span class="admin-badge"><i class="fas fa-check-circle"></i></span>' : ''}
            </div>
        `;
    }
    
    chatMessages.appendChild(messageElement);
    return messageElement;
}

function formatDuration(seconds) {
  if (isNaN(seconds)) return "00:00";
  const minutes = Math.floor(seconds / 60).toString().padStart(2, '0');
  const secs = Math.floor(seconds % 60).toString().padStart(2, '0');
  return `${minutes}:${secs}`;
}
        
function formatTime(timestamp) {
    const date = new Date(timestamp);
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}
        // Scroll chat to bottom
        function scrollChatToBottom() {
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }

        // Load chat users
        function loadChatUsers() {
            if (!currentUser) return;
            
            fetch(`${API_BASE_URL}/chat/users`, {
                headers: {
                    'Authorization': `Bearer ${localStorage.getItem('token')}`
                }
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error('Failed to fetch chat users');
                }
                return response.json();
            })
            .then(users => {
                chatUsers = users;
                renderChatUsers(users);
            })
            .catch(error => {
                console.error('Error loading chat users:', error);
            });
        }

        // Render chat users list
        function renderChatUsers(users) {
            chatUsersList.innerHTML = '';
            
            if (users.length === 0) {
                chatUsersList.innerHTML = '<div class="text-center text-muted py-4">No users available</div>';
                return;
            }
            
            users.forEach(user => {
                const userElement = document.createElement('div');
                userElement.className = `chat-user ${activeChatUserId === user.id ? 'active' : ''}`;
                userElement.setAttribute('data-id', user.id);
                
                const initials = user.name.split(' ').map(n => n[0]).join('').toUpperCase();
                
                userElement.innerHTML = `
                    <div class="chat-user-avatar">${initials}</div>
                    <div class="chat-user-info">
                        <div class="chat-user-name">
                            ${user.name}
                            ${user.role === 'admin' ? '<span class="admin-badge"><i class="fas fa-check-circle"></i></span>' : ''}
                        </div>
                        <div class="chat-user-last-message">${user.last_message || ''}</div>
                    </div>
                    <div class="chat-user-status">
                        <div class="chat-user-time">${user.last_active_time || ''}</div>
                        ${user.unread_count > 0 ? `<div class="chat-user-unread">${user.unread_count}</div>` : '<div class="chat-user-status-dot ${user.status}"></div>'}
                    </div>
                `;
                
                userElement.addEventListener('click', () => {
                    openChat(user);
                });
                
                chatUsersList.appendChild(userElement);
            });
        }

        // Open chat with a user
        function openChat(user) {
            activeChatUserId = user.id;
            
            // Update UI
            chatEmptyState.classList.add('hidden');
            activeChat.classList.remove('hidden');
            
            // Set chat header
            const initials = user.name.split(' ').map(n => n[0]).join('').toUpperCase();
            chatHeaderName.innerHTML = `${user.name} ${user.role === 'admin' ? '<span class="admin-badge"><i class="fas fa-check-circle"></i></span>' : ''}`;
            chatHeaderAvatar.textContent = initials;
            chatHeaderStatusDot.className = 'chat-header-status-dot';
            chatHeaderStatusDot.classList.add(user.status);
            chatHeaderStatusText.textContent = user.status === 'online' ? 'Online' : 'Offline';
            
            // Load chat messages
            loadChatMessages(user.id);
            
            if (chatSocket && chatSocket.readyState === WebSocket.OPEN) {
                chatSocket.send(JSON.stringify({
                    type: 'mark_read',
                    sender_id: user.id
                }));
            }
            
            updateUnreadCount(user.id, -user.unread_count);
        }

        // Load chat messages
        function loadChatMessages(userId) {
    chatMessages.innerHTML = `
        <div class="no-messages-container" id="no-messages-container">
            <img src="https://img.icons8.com/fluency/96/000000/nothing-found.png" class="no-messages-image" alt="No messages">
            <div class="no-messages-text">No messages yet</div>
            <div class="no-messages-subtext">Start the conversation by sending your first message!</div>
        </div>
    `;
    
    fetch(`${API_BASE_URL}/chat/messages/${userId}`, {
        headers: {
            'Authorization': `Bearer ${localStorage.getItem('token')}`
        }
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Failed to fetch messages');
        }
        return response.json();
    })
    .then(messages => {
        if (messages.length === 0) {
            updateMessageVisibility();
            return;
        }
        
        chatMessages.innerHTML = ''; // Clear the no messages container
        messages.forEach(message => {
            const type = message.sender_id === currentUser.id ? 'sent' : 'received';
            addMessageToChat(message, type);
        });
        
        scrollChatToBottom();
    })
    .catch(error => {
        console.error('Error loading messages:', error);
        updateMessageVisibility();
    });
}


        // Handle login
        async function handleLogin(e) {
            e.preventDefault();
            
            const email = emailInput.value;
            const password = passwordInput.value;
            
            // Show loading state
            loginBtnText.textContent = 'Signing in...';
            loginSpinner.classList.remove('hidden');
            
            try {
                const response = await fetch(`${API_BASE_URL}/auth/login`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/x-www-form-urlencoded'
                    },
                    body: new URLSearchParams({
                        username: email,
                        password: password
                    })
                });
                
                if (!response.ok) {
                    const errorData = await response.json();
                    throw new Error(errorData.detail || 'Login failed');
                }
                
                const data = await response.json();
                localStorage.setItem('token', data.access_token);
                
                // Get user info
                const userResponse = await fetch(`${API_BASE_URL}/auth/me`, {
                    headers: {
                        'Authorization': `Bearer ${data.access_token}`
                    }
                });
                
                if (!userResponse.ok) {
                    throw new Error('Failed to get user info');
                }
                
                currentUser = await userResponse.json();
                
                // Check if organization registration is required
                if (data.requires_org_registration) {
                    orgRegistrationModal.classList.add('active');
                } else {
                    setupUIForUser();
                    loginView.classList.add('hidden');
                    appView.classList.remove('hidden');
                    loadInitialData();
                    
                    // Initialize chat connection
                    initChatConnection();
                }
                
            } catch (error) {
                loginErrorMessage.textContent = error.message;
                loginError.classList.remove('hidden');
                setTimeout(() => {
                    loginError.classList.add('hidden');
                }, 5000);
            } finally {
                loginBtnText.textContent = 'Sign In';
                loginSpinner.classList.add('hidden');
            }
        }

        // Handle signup
        async function handleSignup(e) {
            e.preventDefault();
            
            const name = document.getElementById('signup-name').value.trim();
            const email = document.getElementById('signup-email').value.trim();
            const password = document.getElementById('signup-password').value;
            const confirmPassword = document.getElementById('signup-confirm-password').value;
            
            // Validate inputs
            if (!name || !email || !password || !confirmPassword) {
                showSignupError('Please fill in all fields');
                return;
            }
            
            if (password !== confirmPassword) {
                showSignupError('Passwords do not match');
                return;
            }
            
            if (password.length < 8) {
                showSignupError('Password must be at least 8 characters');
                return;
            }
            
            // Show loading state
            signupBtnText.textContent = 'Creating account...';
            signupSpinner.classList.remove('hidden');
            signupError.classList.add('hidden');
            
            try {
                // First send verification email
                const otpResponse = await fetch(`${API_BASE_URL}/auth/send-verification-email`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        email: email
                    })
                });
                
                if (!otpResponse.ok) {
                    const errorData = await otpResponse.json();
                    throw new Error(errorData.detail || 'Failed to send verification email');
                }
                
                // Store the email for verification
                otpVerificationEmail = email;
                
                // Show OTP verification modal
                otpModal.classList.add('active');
                
                // Focus first OTP input
                otpInputs[0].focus();
                
            } catch (error) {
                console.error('Signup error:', error);
                showSignupError(error.message || 'An error occurred during signup');
            } finally {
                signupBtnText.textContent = 'Create Account';
                signupSpinner.classList.add('hidden');
            }
        }

        // Verify OTP
        async function verifyOtp() {
            // Get OTP from inputs
            let otp = '';
            otpInputs.forEach(input => {
                otp += input.value;
            });
            
            if (otp.length !== 6) {
                otpErrorMessage.textContent = 'Please enter a 6-digit code';
                otpError.classList.remove('hidden');
                return;
            }
            
            // Show loading state
            verifyOtpText.textContent = 'Verifying...';
            verifyOtpSpinner.classList.remove('hidden');
            otpError.classList.add('hidden');
            
            try {
                // Verify OTP with backend
                const response = await fetch(`${API_BASE_URL}/auth/verify-otp`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        email: otpVerificationEmail,
                        otp: otp
                    })
                });
                
                if (!response.ok) {
                    const errorData = await response.json();
                    throw new Error(errorData.detail || 'Invalid or expired OTP');
                }
                
                // OTP verified, proceed with signup
                const name = document.getElementById('signup-name').value.trim();
                const email = otpVerificationEmail;
                const password = document.getElementById('signup-password').value;
                
                const signupResponse = await fetch(`${API_BASE_URL}/auth/signup`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/x-www-form-urlencoded',
                    },
                    body: new URLSearchParams({
                        name: name,
                        email: email,
                        password: password
                    })
                });
                
                if (!signupResponse.ok) {
                    const errorData = await signupResponse.json();
                    throw new Error(errorData.detail || 'Signup failed');
                }
                
                const data = await signupResponse.json();
                
                // Store the token
                localStorage.setItem('token', data.access_token);
                
                // Close OTP modal
                otpModal.classList.remove('active');
                
                // Show organization registration modal
                orgRegistrationModal.classList.add('active');
                
            } catch (error) {
                console.error('OTP verification error:', error);
                otpErrorMessage.textContent = error.message || 'Invalid or expired OTP';
                otpError.classList.remove('hidden');
            } finally {
                verifyOtpText.textContent = 'Verify';
                verifyOtpSpinner.classList.add('hidden');
            }
        }

        // Resend OTP
        async function resendOtp() {
            if (!otpVerificationEmail) return;
            
            // Show loading state
            verifyOtpText.textContent = 'Sending...';
            verifyOtpSpinner.classList.remove('hidden');
            otpError.classList.add('hidden');
            
            try {
                const response = await fetch(`${API_BASE_URL}/auth/send-verification-email`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        email: otpVerificationEmail
                    })
                });
                
                if (!response.ok) {
                    const errorData = await response.json();
                    throw new Error(errorData.detail || 'Failed to resend OTP');
                }
                
                // Clear OTP inputs
                otpInputs.forEach(input => {
                    input.value = '';
                });
                
                // Focus first input
                otpInputs[0].focus();
                
                showAlert('New verification code sent to your email', 'success');
                
            } catch (error) {
                console.error('Resend OTP error:', error);
                otpErrorMessage.textContent = error.message || 'Failed to resend OTP';
                otpError.classList.remove('hidden');
            } finally {
                verifyOtpText.textContent = 'Verify';
                verifyOtpSpinner.classList.add('hidden');
            }
        }

        // Handle organization registration
async function handleOrgRegistration() {
    const orgName = organizationNameInput.value.trim();
    
    if (!orgName) {
        showAlert('Please enter an organization name', 'danger');
        return;
    }
    
    if (orgName.length < 3) {
        showAlert('Organization name must be at least 3 characters', 'danger');
        return;
    }
    
    // Show loading state
    submitOrgText.textContent = 'Registering...';
    submitOrgSpinner.classList.remove('hidden');
    
    try {
        const response = await fetch(`${API_BASE_URL}/auth/register-organization`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${localStorage.getItem('token')}`,
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                name: orgName
            })
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Failed to register organization');
        }
        
        // Close modal and proceed to app
        orgRegistrationModal.classList.remove('active');
        showAlert(`Welcome to ${orgName}!`, 'success');
        
        // Get updated user info
        const userResponse = await fetch(`${API_BASE_URL}/auth/me`, {
            headers: {
                'Authorization': `Bearer ${localStorage.getItem('token')}`
            }
        });
        
        if (!userResponse.ok) {
            throw new Error('Failed to get user info');
        }
        
        currentUser = await userResponse.json();
        setupUIForUser();
        
        // Update the sidebar header to show organization name
        const sidebarHeader = document.querySelector('.sidebar-header h2');
        if (sidebarHeader) {
            sidebarHeader.textContent = `${orgName} - ReportHub`;
        }
        
        loginView.classList.add('hidden');
        appView.classList.remove('hidden');
        loadInitialData();
        
        // Initialize chat connection
        initChatConnection();
        
    } catch (error) {
        console.error('Organization registration error:', error);
        showAlert(error.message || 'Failed to register organization', 'danger');
    } finally {
        submitOrgText.textContent = 'Register Organization';
        submitOrgSpinner.classList.add('hidden');
    }
}

        // Check if this is the first user in the system
        async function checkFirstUser() {
            try {
                const response = await fetch(`${API_BASE_URL}/auth/first-user`);
                if (!response.ok) {
                    throw new Error('Failed to check first user status');
                }
                const data = await response.json();
                return data.is_first_user;
            } catch (error) {
                console.error('Error checking first user:', error);
                return false;
            }
        }

        // Helper function to show signup errors
        function showSignupError(message) {
            signupErrorMessage.textContent = message;
            signupError.classList.remove('hidden');
            setTimeout(() => {
                signupError.classList.add('hidden');
            }, 5000);
        }

        // Handle logout
        function handleLogout() {
            // Close WebSocket connection
            if (chatSocket) {
                chatSocket.close();
            }
            
            localStorage.removeItem('token');
            currentUser = null;
            loginView.classList.remove('hidden');
            appView.classList.add('hidden');
            loginForm.reset();
        }

        // Setup UI based on user role
function setupUIForUser() {
    if (!currentUser) return;
    
    // Set user info
    const initials = currentUser.name ? currentUser.name.split(' ').map(n => n[0]).join('').toUpperCase() : 'U';
    userName.textContent = currentUser.name || 'User';
    userRole.textContent = currentUser.role === 'admin' ? 'Administrator' : 'Staff Member';
    userAvatar.textContent = initials;
    userDropdownAvatar.textContent = initials;
    userDropdownName.textContent = currentUser.name || 'User';
    userDropdownEmail.textContent = currentUser.email;
    
    // Update sidebar header with organization name
    const sidebarHeader = document.querySelector('.sidebar-header h2');
    if (sidebarHeader) {
        if (currentUser.organization_name) {
            sidebarHeader.textContent = `${currentUser.organization_name}`;
        } else {
            sidebarHeader.textContent = 'ReportHub';
        }
    }
    
    // Show/hide admin menu
    if (currentUser.role === 'admin') {
        adminMenu.classList.remove('hidden');
    } else {
        adminMenu.classList.add('hidden');
    }
    updateSidebarOrganizationInfo();
}

        // Switch between views
        function switchView(viewId) {
    views.forEach(view => {
        if (view.id === `${viewId}-view`) {
            view.classList.remove('hidden');
            
            // Load data for the view if needed
            if (viewId === 'reports') {
                loadReports();
            } else if (viewId === 'users' && currentUser.role === 'admin') {
                loadUsers();
            } else if (viewId === 'dashboard') {
                loadDashboardData();
            } else if (viewId === 'chat') {
                loadChatUsers();
            } else if (viewId === 'settings') {
                loadOrganizationSettings();
            }
        } else {
            view.classList.add('hidden');
        }
    });
    
    // Close sidebar on mobile
    if (window.innerWidth < 768) {
        sidebar.classList.remove('active');
    }
}

// Add logo preview functionality
document.getElementById('org-logo').addEventListener('change', function(e) {
    const file = e.target.files[0];
    if (!file) return;
    
    if (!file.type.startsWith('image/')) {
        showAlert('Please select an image file', 'danger');
        e.target.value = '';
        return;
    }
    
    const reader = new FileReader();
    reader.onload = function(event) {
        const logoPreview = document.getElementById('org-logo-img');
        const logoPlaceholder = document.getElementById('org-logo-placeholder');
        
        logoPreview.src = event.target.result;
        logoPreview.style.display = 'block';
        logoPlaceholder.style.display = 'none';
    };
    reader.readAsDataURL(file);
});

        // Load initial data for the dashboard
        function loadInitialData() {
            loadDashboardData();
            if (currentUser.role === 'admin') {
                loadUsers();
            }
            loadChatUsers();
        }

        // Load dashboard data
        function loadDashboardData() {
            // Show loading state
            dashboardReportsTableBody.innerHTML = `
                <tr>
                    <td colspan="5" class="text-center">
                        <div class="d-flex justify-content-center py-4">
                            <div class="spinner spinner-primary"></div>
                        </div>
                    </td>
                </tr>
            `;
            
            fetchReports()
                .then(reports => {
                    updateDashboardCounts(reports);
                    renderRecentReports(reports);
                })
                .catch(error => {
                    console.error('Error loading dashboard data:', error);
                    dashboardReportsTableBody.innerHTML = `
                        <tr>
                            <td colspan="5" class="text-center text-muted">Error loading reports</td>
                        </tr>
                    `;
                });
        }

        // Update dashboard counts
        function updateDashboardCounts(reports) {
            const total = reports.length;
            const pending = reports.filter(r => r.status === 'pending').length;
            const approved = reports.filter(r => r.status === 'approved').length;
            const rejected = reports.filter(r => r.status === 'rejected').length;
            
            totalReportsCount.textContent = total;
            pendingReportsCount.textContent = pending;
            approvedReportsCount.textContent = approved;
            rejectedReportsCount.textContent = rejected;
        }

        // Render recent reports for dashboard
        function renderRecentReports(reports) {
            dashboardReportsTableBody.innerHTML = '';
            
            // Sort by date (newest first) and take first 5
            const recentReports = [...reports]
                .sort((a, b) => new Date(b.created_at) - new Date(a.created_at))
                .slice(0, 5);
            
            if (recentReports.length === 0) {
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td colspan="5" class="text-center text-muted">No reports found</td>
                `;
                dashboardReportsTableBody.appendChild(row);
                return;
            }
            
            recentReports.forEach(report => {
                const row = document.createElement('tr');
                
                // Determine status badge
                let statusBadge;
                if (report.status === 'approved') {
                    statusBadge = '<span class="badge badge-success">Approved</span>';
                } else if (report.status === 'pending') {
                    statusBadge = '<span class="badge badge-warning">Pending</span>';
                } else {
                    statusBadge = '<span class="badge badge-danger">Rejected</span>';
                }
                
                // Format date
                const reportDate = new Date(report.created_at).toLocaleDateString();
                
                row.innerHTML = `
                    <td>#REP-${report.id.toString().padStart(3, '0')}</td>
                    <td>${report.title}</td>
                    <td>${statusBadge}</td>
                    <td>${reportDate}</td>
                    <td>
                        <button class="action-btn view-report-btn" data-id="${report.id}" title="View">
                            <i class="fas fa-eye"></i>
                        </button>
                        <button class="action-btn download-report-btn" data-id="${report.id}" title="Download PDF">
                            <i class="fas fa-download"></i>
                        </button>
                    </td>
                `;
                
                dashboardReportsTableBody.appendChild(row);
            });
        }

        // Fetch reports from API
        function fetchReports() {
            return fetch(`${API_BASE_URL}/reports`, {
                headers: {
                    'Authorization': `Bearer ${localStorage.getItem('token')}`
                }
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error('Failed to fetch reports');
                }
                return response.json();
            });
        }

        // Load reports from API
        function loadReports() {
            // Show loading state
            reportsTableBody.innerHTML = `
                <tr>
                    <td colspan="7" class="text-center">
                        <div class="d-flex justify-content-center py-4">
                            <div class="spinner spinner-primary"></div>
                        </div>
                    </td>
                </tr>
            `;
            
            const statusFilter = reportStatusFilter.value;
            const searchQuery = reportSearch.value.toLowerCase();
            
            fetchReports()
                .then(reports => {
                    // Filter reports based on status and search query
                    let filteredReports = reports;
                    
                    if (statusFilter) {
                        filteredReports = filteredReports.filter(r => r.status === statusFilter);
                    }
                    
                    if (searchQuery) {
                        filteredReports = filteredReports.filter(r => 
                            r.title.toLowerCase().includes(searchQuery) || 
                            r.description.toLowerCase().includes(searchQuery) ||
                            r.author_name.toLowerCase().includes(searchQuery));
                    }
                    
                    // Paginate results
                    const totalReports = filteredReports.length;
                    const totalPages = Math.ceil(totalReports / reportsPerPage);
                    const startIndex = (currentReportsPage - 1) * reportsPerPage;
                    const endIndex = Math.min(startIndex + reportsPerPage, totalReports);
                    const paginatedReports = filteredReports.slice(startIndex, endIndex);
                    
                    renderReportsTable(paginatedReports);
                    
                    // Update pagination info
                    document.getElementById('reports-from').textContent = startIndex + 1;
                    document.getElementById('reports-to').textContent = endIndex;
                    document.getElementById('reports-total').textContent = totalReports;
                    
                    // Update pagination buttons
                    reportsPrevBtn.disabled = currentReportsPage === 1;
                    reportsNextBtn.disabled = currentReportsPage >= totalPages;
                })
                .catch(error => {
                    console.error('Error loading reports:', error);
                    showAlert('Failed to load reports', 'danger');
                    
                    reportsTableBody.innerHTML = `
                        <tr>
                            <td colspan="7" class="text-center text-muted">Error loading reports</td>
                        </tr>
                    `;
                });
        }

        // Render reports table
        function renderReportsTable(reports) {
            reportsTableBody.innerHTML = '';
            
            if (reports.length === 0) {
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td colspan="7" class="text-center text-muted">No reports found</td>
                `;
                reportsTableBody.appendChild(row);
                return;
            }
            
            reports.forEach(report => {
                const row = document.createElement('tr');
                
                // Determine status badge
                let statusBadge;
                if (report.status === 'approved') {
                    statusBadge = '<span class="badge badge-success">Approved</span>';
                } else if (report.status === 'pending') {
                    statusBadge = '<span class="badge badge-warning">Pending</span>';
                } else {
                    statusBadge = '<span class="badge badge-danger">Rejected</span>';
                }
                
                // Format date
                const reportDate = new Date(report.created_at).toLocaleDateString();
                
                // Count attachments
                const attachmentCount = report.attachments ? report.attachments.length : 0;
                const attachmentIcon = attachmentCount > 0 ? 
                    `<i class="fas fa-paperclip"></i> ${attachmentCount}` : 
                    '<span class="text-muted">None</span>';
                
                row.innerHTML = `
                    <td>#REP-${report.id.toString().padStart(3, '0')}</td>
                    <td>${report.title}</td>
                    <td>${report.author_name}</td>
                    <td>${statusBadge}</td>
                    <td>${reportDate}</td>
                    <td>${attachmentIcon}</td>
                    <td>
                        <button class="action-btn view-report-btn" data-id="${report.id}" title="View">
                            <i class="fas fa-eye"></i>
                        </button>
                        <button class="action-btn download-report-btn" data-id="${report.id}" title="Download PDF">
                            <i class="fas fa-download"></i>
                        </button>
                        ${currentUser.role === 'admin' || currentUser.id === report.author_id ? `
                        <button class="action-btn edit-report-btn" data-id="${report.id}" title="Edit">
                            <i class="fas fa-edit"></i>
                        </button>
                        ` : ''}
                        ${currentUser.role === 'admin' ? `
                        <button class="action-btn delete-report-btn delete" data-id="${report.id}" title="Delete">
                            <i class="fas fa-trash"></i>
                        </button>
                        ` : ''}
                    </td>
                `;
                
                reportsTableBody.appendChild(row);
            });
        }

        // Load users from API (admin only)
        function loadUsers() {
            // Show loading state
            usersTableBody.innerHTML = `
                <tr>
                    <td colspan="6" class="text-center">
                        <div class="d-flex justify-content-center py-4">
                            <div class="spinner spinner-primary"></div>
                        </div>
                    </td>
                </tr>
            `;
            
            const searchQuery = userSearch.value.toLowerCase();
            
            fetch(`${API_BASE_URL}/users`, {
                headers: {
                    'Authorization': `Bearer ${localStorage.getItem('token')}`
                }
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error('Failed to fetch users');
                }
                return response.json();
            })
            .then(users => {
                // Filter users based on search query
                let filteredUsers = users;
                
                if (searchQuery) {
                    filteredUsers = filteredUsers.filter(u => 
                        u.name.toLowerCase().includes(searchQuery) || 
                        u.email.toLowerCase().includes(searchQuery));
                }
                
                // Paginate results
                const totalUsers = filteredUsers.length;
                const totalPages = Math.ceil(totalUsers / usersPerPage);
                const startIndex = (currentUsersPage - 1) * usersPerPage;
                const endIndex = Math.min(startIndex + usersPerPage, totalUsers);
                const paginatedUsers = filteredUsers.slice(startIndex, endIndex);
                
                renderUsersTable(paginatedUsers);
                
                // Update pagination info
                document.getElementById('users-from').textContent = startIndex + 1;
                document.getElementById('users-to').textContent = endIndex;
                document.getElementById('users-total').textContent = totalUsers;
                
                // Update pagination buttons
                usersPrevBtn.disabled = currentUsersPage === 1;
                usersNextBtn.disabled = currentUsersPage >= totalPages;
            })
            .catch(error => {
                console.error('Error loading users:', error);
                showAlert('Failed to load users', 'danger');
                
                usersTableBody.innerHTML = `
                    <tr>
                        <td colspan="6" class="text-center text-muted">Error loading users</td>
                    </tr>
                `;
            });
        }

        // Render users table
        function renderUsersTable(users) {
            usersTableBody.innerHTML = '';
            
            if (users.length === 0) {
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td colspan="6" class="text-center text-muted">No users found</td>
                `;
                usersTableBody.appendChild(row);
                return;
            }
            
            users.forEach(user => {
                const row = document.createElement('tr');
                
                // Format last active time
                const lastActive = user.last_active ? 
                    new Date(user.last_active).toLocaleString() : 
                    'Never';
                
                // Determine status
                const status = user.is_active ? 
                    '<span class="badge badge-success">Active</span>' : 
                    '<span class="badge badge-danger">Inactive</span>';
                
                row.innerHTML = `
                    <td>${user.name}</td>
                    <td>${user.email}</td>
                    <td>${user.role === 'admin' ? 'Administrator' : 'Staff Member'}</td>
                    <td>${status}</td>
                    <td>${lastActive}</td>
                    <td>
                        <button class="action-btn edit-user-btn" data-id="${user.id}" title="Edit">
                            <i class="fas fa-edit"></i>
                        </button>
                        ${user.id !== currentUser.id ? `
                        <button class="action-btn delete-user-btn delete" data-id="${user.id}" title="Delete">
                            <i class="fas fa-trash"></i>
                        </button>
                        ` : ''}
                    </td>
                `;
                
                usersTableBody.appendChild(row);
            });
        }

        // Handle file upload for report attachments
        function handleFileUpload(e) {
            const files = e.target.files;
            if (!files || files.length === 0) return;
            
            reportAttachmentsList.innerHTML = '';
            
            for (let i = 0; i < files.length; i++) {
                const file = files[i];
                const fileItem = document.createElement('div');
                fileItem.className = 'file-item';
                
                // Get file icon based on type
                let fileIcon, fileColor;
                if (file.type.includes('image')) {
                    fileIcon = 'fa-file-image';
                    fileColor = '#e74c3c';
                } else if (file.type.includes('pdf')) {
                    fileIcon = 'fa-file-pdf';
                    fileColor = '#e74c3c';
                } else if (file.type.includes('word') || file.type.includes('document')) {
                    fileIcon = 'fa-file-word';
                    fileColor = '#2c7be5';
                } else if (file.type.includes('excel') || file.type.includes('spreadsheet')) {
                    fileIcon = 'fa-file-excel';
                    fileColor = '#2ecc71';
                } else {
                    fileIcon = 'fa-file';
                    fileColor = '#95a5a6';
                }
                
                // Format file size
                const fileSize = formatFileSize(file.size);
                
                fileItem.innerHTML = `
                    <i class="fas ${fileIcon} file-icon" style="color: ${fileColor};"></i>
                    <div class="file-info">
                        <div class="file-name">${file.name}</div>
                        <div class="file-size">${fileSize}</div>
                    </div>
                    <i class="fas fa-times file-remove" data-index="${i}"></i>
                `;
                
                reportAttachmentsList.appendChild(fileItem);
            }
            
            // Add event listeners to remove buttons
            document.querySelectorAll('.file-remove').forEach(btn => {
                btn.addEventListener('click', (e) => {
                    const index = parseInt(e.currentTarget.getAttribute('data-index'));
                    removeFileFromList(index);
                });
            });
        }

        // Remove file from upload list
        function removeFileFromList(index) {
            // Create a new DataTransfer object to manipulate files
            const dataTransfer = new DataTransfer();
            const files = reportAttachments.files;
            
            // Add all files except the one to be removed
            for (let i = 0; i < files.length; i++) {
                if (i !== index) {
                    dataTransfer.items.add(files[i]);
                }
            }
            
            // Update the file input
            reportAttachments.files = dataTransfer.files;
            
            // Re-render the file list
            const event = new Event('change');
            reportAttachments.dispatchEvent(event);
        }

        // Format file size
        function formatFileSize(bytes) {
            if (bytes === 0) return '0 Bytes';
            
            const k = 1024;
            const sizes = ['Bytes', 'KB', 'MB', 'GB'];
            const i = Math.floor(Math.log(bytes) / Math.log(k));
            
            return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
        }

        // Handle create report submission
        function handleCreateReport() {
    // Ensure Quill content is captured
    if (quill) {
        document.getElementById('report-description').value = quill.root.innerHTML;
    }
    
    const title = document.getElementById('report-title').value;
    const description = document.getElementById('report-description').value;
    const category = document.getElementById('report-category').value;
    const files = reportAttachments.files;
    
    if (!title || !description || !category) {
        showAlert('Please fill in all required fields', 'danger');
        return;
    }
    
    const templateFieldsData = {};
    const dynamicFieldsContainer = document.getElementById('dynamic-fields-container');
    if (dynamicFieldsContainer) {
        dynamicFieldsContainer.querySelectorAll('.template-field').forEach(fieldGroup => {
            const fieldName = fieldGroup.dataset.fieldName;
            const inputElement = fieldGroup.querySelector('input, select, textarea');
            
            if (!inputElement) return;
            
            if (inputElement.type === 'checkbox' && !inputElement.name) {
                // Single checkbox
                templateFieldsData[fieldName] = inputElement.checked ? inputElement.value : false;
            } else if (inputElement.type === 'checkbox' || inputElement.type === 'radio') {
                // Multiple checkboxes or radios
                const inputs = fieldGroup.querySelectorAll(`input[name="${inputElement.name}"]`);
                if (inputs.length > 1) {
                    const values = [];
                    inputs.forEach(input => {
                        if (input.checked) {
                            values.push(input.value);
                        }
                    });
                    templateFieldsData[fieldName] = values.length > 0 ? values : null;
                } else {
                    templateFieldsData[fieldName] = inputs[0].checked ? inputs[0].value : null;
                }
            } else {
                templateFieldsData[fieldName] = inputElement.value;
            }
        });
    }
    
    // Create form data
    const formData = new FormData();
    formData.append('title', title);
    formData.append('description', description);
    formData.append('category', category);
    
    // Add template fields data if any
    if (Object.keys(templateFieldsData).length > 0) {
        formData.append('template_fields', JSON.stringify(templateFieldsData));
    }
    
    // Add attachments if any
    for (let i = 0; i < files.length; i++) {
        formData.append('attachments', files[i]);
    }
    
    fetch(`${API_BASE_URL}/reports`, {
        method: 'POST',
        headers: {
            'Authorization': `Bearer ${localStorage.getItem('token')}`
        },
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            return response.json().then(err => {
                throw new Error(err.detail || 'Failed to create report');
            });
        }
        return response.json();
    })
    .then(report => {
        showAlert('Report created successfully', 'success');
        createReportModal.classList.remove('active');
        createReportForm.reset();
        reportAttachmentsList.innerHTML = '';
        
        const dynamicFieldsContainer = document.getElementById('dynamic-fields-container');
        if (dynamicFieldsContainer) {
            dynamicFieldsContainer.remove();
        }
        
        loadReports();
        loadDashboardData();
    })
    .catch(error => {
        showAlert(error.message, 'danger');
    })
    .finally(() => {
        submitReportText.textContent = 'Submit Report';
        submitReportSpinner.classList.add('hidden');
    });
}

// Add template management button to the UI
function addTemplateManagementButton() {
    // Add to the page actions in reports view
    const pageActions = document.querySelector('#reports-view .page-actions');
    if (pageActions) {
        const templateBtn = document.createElement('button');
        templateBtn.className = 'btn btn-outline-primary';
        templateBtn.id = 'template-management-btn';
        templateBtn.innerHTML = '<i class="fas fa-file-alt"></i> Templates';
        pageActions.insertBefore(templateBtn, pageActions.firstChild);
    }
}

async function handleAddUser(e) {
    e.preventDefault(); // This prevents the form from submitting traditionally
    
    const form = document.getElementById('add-user-form');
    const fullName = form.elements['user-fullname'].value.trim();
    const email = form.elements['user-email'].value.trim();
    const role = form.elements['user-role'].value;
    const password = form.elements['user-password'].value;
    const confirmPassword = form.elements['user-confirm-password'].value;
    
    // Validation (same as before)
    if (!fullName || !email || !role || !password || !confirmPassword) {
        showAlert('Please fill in all required fields', 'danger');
        return;
    }
    
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(email)) {
        showAlert('Please enter a valid email address', 'danger');
        return;
    }
    
    if (password !== confirmPassword) {
        showAlert('Passwords do not match', 'danger');
        return;
    }
    
    if (password.length < 8 || 
        !/[A-Z]/.test(password) || 
        !/[a-z]/.test(password) || 
        !/[0-9]/.test(password) || 
        !/[!@#$%^&*()_+\-=\[\]{};':"\\|,.<>\/?]/.test(password)) {
        showAlert('Password must be at least 8 characters long and contain uppercase, lowercase, number, and special character', 'danger');
        return;
    }
    
    // Show loading state
    submitUserText.textContent = 'Adding...';
    submitUserSpinner.classList.remove('hidden');
    submitAddUser.disabled = true;
    
    try {
        const userData = {
            name: fullName,
            email: email,
            role: role,
            password: password
        };
        
        const response = await fetch(`${API_BASE_URL}/users`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${localStorage.getItem('token')}`,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(userData)
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Failed to add user');
        }
        
        showAlert('User added successfully', 'success');
        addUserModal.classList.remove('active');
        form.reset();
        loadUsers();
        
    } catch (error) {
        console.error('Error adding user:', error);
        showAlert(error.message || 'An error occurred while adding user', 'danger');
    } finally {
        submitUserText.textContent = 'Add User';
        submitUserSpinner.classList.add('hidden');
        submitAddUser.disabled = false;
    }
}
        // View report details
        // View report details
async function viewReport(reportId) {
    try {
        const response = await fetch(`${API_BASE_URL}/reports/${reportId}`, {
            headers: {
                'Authorization': `Bearer ${localStorage.getItem('token')}`
            }
        });
        
        if (!response.ok) {
            throw new Error('Failed to fetch report');
        }
        
        const report = await response.json();
        console.log('Full report data:', report);
        console.log('Template data:', report.template_data);
        selectedReportId = reportId;

        // Set report details
        viewReportTitle.textContent = report.title;
        viewReportAuthor.textContent = report.author_name;
        viewReportDate.textContent = new Date(report.created_at).toLocaleDateString();
        viewReportDescription.innerHTML = report.description;
        viewReportCategory.textContent = report.category;

        // Set status badge
        let statusBadge;
        if (report.status === 'approved') {
            statusBadge = '<span class="badge badge-success">Approved</span>';
        } else if (report.status === 'pending') {
            statusBadge = '<span class="badge badge-warning">Pending</span>';
        } else {
            statusBadge = '<span class="badge badge-danger">Rejected</span>';
        }
        viewReportStatus.innerHTML = statusBadge;

        // Set status dropdown and admin comments
        reportStatusChange.value = report.status;
        adminComments.value = report.admin_comments || '';

        // Display template fields if they exist
        const templateFieldsContainer = document.getElementById('template-fields-container');
        templateFieldsContainer.innerHTML = '';
        
        if (report.template_data) {  // Changed from template_fields to template_data
            const fieldsSection = document.createElement('div');
            fieldsSection.className = 'card mt-3';
            fieldsSection.innerHTML = `
                <div class="card-header">
                    <h3 class="card-title">Form Data</h3>
                </div>
                <div class="card-body">
                    <!-- Template fields will be rendered here -->
                </div>
            `;
            
            const fieldsView = fieldsSection.querySelector('.card-body');
            
            for (const [fieldName, fieldValue] of Object.entries(report.template_data)) {
                if (fieldValue === null || fieldValue === undefined) continue;
                
                const fieldElement = document.createElement('div');
                fieldElement.className = 'form-field-view mb-3';
                
                const label = document.createElement('div');
                label.className = 'form-field-label';
                label.textContent = fieldName.replace(/_/g, ' ');
                label.style.textTransform = 'capitalize';
                label.style.fontWeight = 'bold';
                
                const value = document.createElement('div');
                value.className = 'form-field-value';
                
                if (Array.isArray(fieldValue)) {
                    value.textContent = fieldValue.join(', ');
                } else if (typeof fieldValue === 'object') {
                    value.textContent = JSON.stringify(fieldValue, null, 2);
                } else {
                    value.textContent = fieldValue;
                }
                
                fieldElement.appendChild(label);
                fieldElement.appendChild(value);
                fieldsView.appendChild(fieldElement);
            }
            
            // Insert fields section after the description card
            const descriptionCard = document.querySelector('#view-report-modal .card');
            descriptionCard.parentNode.insertBefore(fieldsSection, descriptionCard.nextSibling);
        }
        
        // Set attachments
        viewReportAttachments.innerHTML = '';
        if (report.attachments && report.attachments.length > 0) {
            report.attachments.forEach(attachment => {
                const fileItem = document.createElement('div');
                fileItem.className = 'file-item';

                // Determine file icon based on type
                let fileIcon, fileColor;
                if (attachment.type.includes('image')) {
                    fileIcon = 'fa-file-image';
                    fileColor = '#e74c3c';
                } else if (attachment.type.includes('pdf')) {
                    fileIcon = 'fa-file-pdf';
                    fileColor = '#e74c3c';
                } else if (attachment.type.includes('word') || attachment.type.includes('document')) {
                    fileIcon = 'fa-file-word';
                    fileColor = '#2c7be5';
                } else if (attachment.type.includes('excel') || attachment.type.includes('spreadsheet')) {
                    fileIcon = 'fa-file-excel';
                    fileColor = '#2ecc71';
                } else {
                    fileIcon = 'fa-file';
                    fileColor = '#95a5a6';
                }

                const fileSize = formatFileSize(attachment.size);

                fileItem.innerHTML = `
                    <i class="fas ${fileIcon} file-icon" style="color: ${fileColor};"></i>
                    <div class="file-info">
                        <div class="file-name">${attachment.name}</div>
                        <div class="file-size">${fileSize}</div>
                    </div>
                    <button class="action-btn download-attachment-btn" 
                            data-url="${API_BASE_URL}/download/${attachment.url.split('/').pop()}" 
                            data-name="${attachment.name}"
                            title="Download">
                        <i class="fas fa-download"></i>
                    </button>
                `;

                viewReportAttachments.appendChild(fileItem);
            });

            // Add event listeners to download buttons
            document.querySelectorAll('.download-attachment-btn').forEach(btn => {
                btn.addEventListener('click', (e) => {
                    const url = e.currentTarget.getAttribute('data-url');
                    const fileName = e.currentTarget.getAttribute('data-name');
                    downloadFile(url, fileName);
                });
            });
        } else {
            const noFiles = document.createElement('div');
            noFiles.className = 'text-center text-muted';
            noFiles.textContent = 'No attachments';
            viewReportAttachments.appendChild(noFiles);
        }

        // Show admin actions based on role
        if (currentUser.role === 'admin') {
            adminReportActions.classList.remove('hidden');
            saveReportChanges.classList.remove('hidden');
        } else {
            adminReportActions.classList.add('hidden');
            saveReportChanges.classList.add('hidden');
        }

        // Show the modal
        viewReportModal.classList.add('active');
    } catch (error) {
        console.error('Error viewing report:', error);
        showAlert('Failed to load report details', 'danger');
    }
}
        // Edit report
        function editReport(reportId) {
            showAlert('Edit functionality would be implemented here', 'info');
        }

        // Delete report
        function deleteReport(reportId) {
            if (!confirm('Are you sure you want to delete this report?')) {
                return;
            }
            
            fetch(`${API_BASE_URL}/reports/${reportId}`, {
                method: 'DELETE',
                headers: {
                    'Authorization': `Bearer ${localStorage.getItem('token')}`
                }
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error('Failed to delete report');
                }
                return response.json();
            })
            .then(() => {
                // Show success message
                showAlert('Report deleted successfully', 'success');
                
                // Reload reports
                loadReports();
                loadDashboardData();
            })
            .catch(error => {
                console.error('Error deleting report:', error);
                showAlert('Failed to delete report', 'danger');
            });
        }

        // Edit user
        function editUser(userId) {
            showAlert('Edit user functionality would be implemented here', 'info');
        }

        // Delete user
        function deleteUser(userId) {
            if (!confirm('Are you sure you want to delete this user?')) {
                return;
            }
            
            fetch(`${API_BASE_URL}/users/${userId}`, {
                method: 'DELETE',
                headers: {
                    'Authorization': `Bearer ${localStorage.getItem('token')}`
                }
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error('Failed to delete user');
                }
                return response.json();
            })
            .then(() => {
                // Show success message
                showAlert('User deleted successfully', 'success');
                
                // Reload users
                loadUsers();
            })
            .catch(error => {
                console.error('Error deleting user:', error);
                showAlert('Failed to delete user', 'danger');
            });
        }

        // Handle saving report changes (admin only)
        async function handleSaveReportChanges() {
            const status = reportStatusChange.value;
            const comments = adminComments.value;
            
            // Show loading state
            saveChangesText.textContent = 'Saving...';
            saveChangesSpinner.classList.remove('hidden');
            
            try {
                const response = await fetch(`${API_BASE_URL}/reports/${selectedReportId}/status`, {
                    method: 'PATCH',
                    headers: {
                        'Authorization': `Bearer ${localStorage.getItem('token')}`,
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        status: status,
                        admin_comments: comments
                    })
                });

                if (!response.ok) {
                    const errorData = await response.json();
                    throw new Error(errorData.detail || 'Failed to update report status');
                }

                const updatedReport = await response.json();
                
                // Show success message
                showAlert('Report status updated successfully', 'success');
                
                // Close modal
                viewReportModal.classList.remove('active');
                
                // Update the UI with the new status
                updateReportStatusInUI(selectedReportId, status);
                
                // Reload reports and dashboard data
                loadReports();
                loadDashboardData();
            } catch (error) {
                console.error('Error updating report status:', error);
                showAlert(error.message || 'Failed to update report status', 'danger');
            } finally {
                saveChangesText.textContent = 'Save Changes';
                saveChangesSpinner.classList.add('hidden');
            }
        }

        function updateReportStatusInUI(reportId, newStatus) {
            // Update in the reports table
            const reportRow = document.querySelector(`.view-report-btn[data-id="${reportId}"]`)?.closest('tr');
            if (reportRow) {
                const statusCell = reportRow.querySelector('td:nth-child(4)');
                if (statusCell) {
                    let statusBadge;
                    if (newStatus === 'approved') {
                        statusBadge = '<span class="badge badge-success">Approved</span>';
                    } else if (newStatus === 'pending') {
                        statusBadge = '<span class="badge badge-warning">Pending</span>';
                    } else {
                        statusBadge = '<span class="badge badge-danger">Rejected</span>';
                    }
                    statusCell.innerHTML = statusBadge;
                }
            }

            // Update in the dashboard table if visible
            const dashboardRow = document.querySelector(`#dashboard-reports-table-body .view-report-btn[data-id="${reportId}"]`)?.closest('tr');
            if (dashboardRow) {
                const statusCell = dashboardRow.querySelector('td:nth-child(3)');
                if (statusCell) {
                    let statusBadge;
                    if (newStatus === 'approved') {
                        statusBadge = '<span class="badge badge-success">Approved</span>';
                    } else if (newStatus === 'pending') {
                        statusBadge = '<span class="badge badge-warning">Pending</span>';
                    } else {
                        statusBadge = '<span class="badge badge-danger">Rejected</span>';
                    }
                    statusCell.innerHTML = statusBadge;
                }
            }

            // Update the counts in the dashboard
            if (document.getElementById('dashboard-view') && !document.getElementById('dashboard-view').classList.contains('hidden')) {
                loadDashboardData();
            }
        }

        function downloadFile(url, fileName) {
    // Extract the file name from the URL
    const fileKey = url.split('/').pop();
    
    // Create a hidden link and trigger download
    const link = document.createElement('a');
    link.href = `${API_BASE_URL}/download/${fileKey}`;
    link.download = fileName;
    link.style.display = 'none';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}
        // Handle saving settings
        function handleSaveSettings() {
            // Show loading state
            saveSettingsBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Saving...';
            saveSettingsBtn.disabled = true;
            
            // Simulate API call with timeout
            setTimeout(() => {
                // Show success message
                showAlert('Settings saved successfully', 'success');
                
                // Reset button state
                saveSettingsBtn.innerHTML = '<i class="fas fa-save"></i> Save Changes';
                saveSettingsBtn.disabled = false;
            }, 1000); // Simulate network delay
        }

        // Show alert message
        function showAlert(message, type) {
            const alert = document.createElement('div');
            alert.className = `alert alert-${type} fade-in`;
            
            let icon;
            if (type === 'success') {
                icon = 'fa-check-circle';
            } else if (type === 'danger') {
                icon = 'fa-exclamation-circle';
            } else if (type === 'warning') {
                icon = 'fa-exclamation-triangle';
            } else {
                icon = 'fa-info-circle';
            }
            
            alert.innerHTML = `
                <i class="fas ${icon}"></i>
                <div class="alert-content">
                    <div class="alert-title">${type.charAt(0).toUpperCase() + type.slice(1)}</div>
                    <div class="alert-message">${message}</div>
                </div>
            `;
            
            // Insert at the top of the content area
            const content = document.querySelector('.content');
            if (content.firstChild) {
                content.insertBefore(alert, content.firstChild);
            } else {
                content.appendChild(alert);
            }
            
            // Remove after 5 seconds
            setTimeout(() => {
                alert.classList.add('hidden');
                setTimeout(() => {
                    alert.remove();
                }, 300);
            }, 5000);
        }

        // Initialize Quill editor when the create report modal is opened
        let quill;
        document.getElementById('create-report-btn').addEventListener('click', initQuillEditor);
        document.getElementById('new-report-btn').addEventListener('click', initQuillEditor);
        
        function initQuillEditor() {
    // Check if Quill is already initialized
    if (quill) return;
    
    // Initialize Quill editor
    const editorContainer = document.getElementById('report-description-editor');
    if (!editorContainer) return;
    
    // Clear any existing content
    editorContainer.innerHTML = '';
    
    quill = new Quill(editorContainer, {
        theme: 'snow',
        modules: {
            toolbar: [
                [{ 'header': [1, 2, 3, false] }],
                ['bold', 'italic', 'underline', 'strike'],
                [{ 'list': 'ordered'}, { 'list': 'bullet' }],
                ['link', 'image'],
                ['clean']
            ]
        },
        placeholder: 'Enter report description...'
    });
    
    // Update hidden input with HTML content when editor changes
    quill.on('text-change', function() {
        document.getElementById('report-description').value = quill.root.innerHTML;
    });
}
        
        // Reset Quill editor when modal is closed
        document.getElementById('close-create-report-modal').addEventListener('click', resetQuillEditor);
        document.getElementById('cancel-create-report').addEventListener('click', resetQuillEditor);
        
        function resetQuillEditor() {
            if (quill) {
                quill.root.innerHTML = '';
                document.getElementById('report-description').value = '';
            }
        }
        // Add event listeners for invite organization
inviteOrgItem?.addEventListener('click', (e) => {
    e.preventDefault();
    inviteOrgModal.classList.add('active');
    
    // Generate personalized invite link if user is logged in
    if (currentUser) {
        const inviteLink = `https://reporthub.com/signup?ref=${currentUser.id}`;
        inviteLinkInput.value = inviteLink;
    }
});

closeInviteOrgModal?.addEventListener('click', () => {
    inviteOrgModal.classList.remove('active');
});

// Copy invite link
copyInviteLinkBtn?.addEventListener('click', () => {
    inviteLinkInput.select();
    document.execCommand('copy');
    
    // Show tooltip
    const originalText = copyInviteLinkBtn.innerHTML;
    copyInviteLinkBtn.innerHTML = '<i class="fas fa-check"></i> Copied!';
    
    setTimeout(() => {
        copyInviteLinkBtn.innerHTML = originalText;
    }, 2000);
});

shareEmailBtn?.addEventListener('click', () => {
    const subject = 'Join me on ReportHub – Simplify Your Team’s Reporting';
    
    const body = `Hello,

I’d like to invite you to use **ReportHub**, a collaborative reporting platform that helps teams:

- Create and manage reports together
- Collaborate in real-time with team messaging
- Track progress, approvals, and deadlines
- Access insightful analytics

**Join us using this link:**  
${inviteLinkInput.value}

Let’s simplify reporting and boost our team’s efficiency.

Best regards,  
${currentUser?.name || 'Your Colleague'}`;

    const gmailUrl = `https://mail.google.com/mail/?view=cm&fs=1&su=${encodeURIComponent(subject)}&body=${encodeURIComponent(body)}`;

    window.open(gmailUrl, '_blank');
});

// Share via WhatsApp
shareWhatsappBtn?.addEventListener('click', () => {
    const text = `Join me on ReportHub - Professional Reporting Platform! Sign up here: ${inviteLinkInput.value}`;
    window.open(`https://wa.me/?text=${encodeURIComponent(text)}`);
});

// Share via LinkedIn
shareLinkedinBtn?.addEventListener('click', () => {
    const url = inviteLinkInput.value;
    const title = 'ReportHub - Professional Reporting Platform';
    const summary = 'Join me on ReportHub, the modern reporting platform for teams. Create, collaborate, and track reports in one place.';
    window.open(`https://www.linkedin.com/shareArticle?mini=true&url=${encodeURIComponent(url)}&title=${encodeURIComponent(title)}&summary=${encodeURIComponent(summary)}`);
});

// Share via Twitter
shareTwitterBtn?.addEventListener('click', () => {
    const text = `Join me on ReportHub - the professional reporting platform! ${inviteLinkInput.value}`;
    window.open(`https://twitter.com/intent/tweet?text=${encodeURIComponent(text)}`);
});
document.getElementById('voice-record-btn').addEventListener('click', startVoiceRecording);
document.getElementById('stop-recording-btn').addEventListener('click', stopVoiceRecording);
document.getElementById('send-voice-message-btn').addEventListener('click', sendVoiceMessage);
document.getElementById('cancel-voice-message-btn').addEventListener('click', cancelVoiceRecording);

// Add these functions to handle voice recording
async function startVoiceRecording() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                mediaRecorder = new MediaRecorder(stream);
                audioChunks = [];
                
                mediaRecorder.ondataavailable = event => {
                    audioChunks.push(event.data);
                };
                
                mediaRecorder.onstop = () => {
                    const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                    const audioUrl = URL.createObjectURL(audioBlob);
                    const audioElement = document.getElementById('recorded-audio');
                    audioElement.src = audioUrl;
                    
                    document.getElementById('recording-controls').classList.remove('hidden');
                    document.getElementById('recording-indicator').classList.add('hidden');
                    
                    // Stop all tracks in the stream
                    stream.getTracks().forEach(track => track.stop());
                };
                
                mediaRecorder.start();
                recordingStartTime = Date.now();
                updateRecordingTime();
                
                document.getElementById('voice-recorder').classList.remove('hidden');
                document.getElementById('voice-record-btn').classList.add('hidden');
                
            } catch (error) {
                console.error('Error accessing microphone:', error);
                showAlert('Microphone access denied. Please allow microphone access to record voice messages.', 'danger');
            }
        }

        function updateRecordingTime() {
            const elapsedTime = Math.floor((Date.now() - recordingStartTime) / 1000);
            const minutes = Math.floor(elapsedTime / 60).toString().padStart(2, '0');
            const seconds = (elapsedTime % 60).toString().padStart(2, '0');
            document.getElementById('recording-time').textContent = `${minutes}:${seconds}`;
            
            if (mediaRecorder && mediaRecorder.state === 'recording') {
                recordingTimer = setTimeout(updateRecordingTime, 1000);
            }
        }

        function stopVoiceRecording() {
            if (mediaRecorder && mediaRecorder.state === 'recording') {
                mediaRecorder.stop();
                clearTimeout(recordingTimer);
            }
        }

        function cancelVoiceRecording() {
            if (mediaRecorder && mediaRecorder.state === 'recording') {
                mediaRecorder.stop();
                clearTimeout(recordingTimer);
            }
            
            // Clean up
            audioChunks = [];
            document.getElementById('recorded-audio').src = '';
            document.getElementById('voice-recorder').classList.add('hidden');
            document.getElementById('voice-record-btn').classList.remove('hidden');
        }
        
async function sendVoiceMessage() {
            if (!activeChatUserId || audioChunks.length === 0) return;
            
            const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
            const formData = new FormData();
            formData.append('voice_message', audioBlob, `voice_message_${Date.now()}.wav`);
            formData.append('recipient_id', activeChatUserId);
            
            try {
                const response = await fetch(`${API_BASE_URL}/chat/voice-message`, {
                    method: 'POST',
                    headers: {
                        'Authorization': `Bearer ${localStorage.getItem('token')}`
                    },
                    body: formData
                });
                
                if (!response.ok) {
                    throw new Error('Failed to send voice message');
                }
                
                const message = await response.json();
                
                // Add the voice message to the chat
                const messageElement = addMessageToChat({
                    id: message.id,
                    sender_id: currentUser.id,
                    recipient_id: activeChatUserId,
                    content: `[VOICE_MESSAGE]${message.audio_url}`,
                    timestamp: new Date().toISOString(),
                    status: 'delivered'
                }, 'sent');
                
                // Reset the recording UI
                cancelVoiceRecording();
                
            } catch (error) {
                console.error('Error sending voice message:', error);
                showAlert('Failed to send voice message', 'danger');
            }
        }

function addVoiceMessageToChat(message, type) {
    const messageElement = document.createElement('div');
    messageElement.className = `voice-message ${type}`;
    
    // Make sure the audio URL is properly constructed
    const audioUrl = `${API_BASE_URL}${message.audio_url}`;
    
    messageElement.innerHTML = `
        <div class="voice-message-controls">
            <button class="play-pause-btn" onclick="togglePlayPause(this)">
                <i class="fas fa-play"></i>
            </button>
            <div class="voice-message-duration">00:00</div>
        </div>
        <audio src="${audioUrl}" preload="none"></audio>
        <div class="message-time">
            ${formatTime(message.timestamp)}
            ${type === 'sent' ? `
            <span class="message-status ${message.status}">
                ${message.status === 'sending' ? '<i class="fas fa-clock"></i>' : 
                  message.status === 'delivered' ? '<i class="fas fa-check"></i>' : 
                  message.status === 'read' ? '<i class="fas fa-check-double"></i>' : ''}
            </span>
            ` : ''}
        </div>
    `;
    
    // Set up duration when metadata loads
    const audioElement = messageElement.querySelector('audio');
    audioElement.addEventListener('loadedmetadata', function() {
        const duration = Math.floor(audioElement.duration);
        const minutes = Math.floor(duration / 60).toString().padStart(2, '0');
        const seconds = (duration % 60).toString().padStart(2, '0');
        messageElement.querySelector('.voice-message-duration').textContent = `${minutes}:${seconds}`;
    });
    
    chatMessages.appendChild(messageElement);
    scrollChatToBottom();
}

// Add this function to handle play/pause of voice messages
function togglePlayPause(button) {
  const messageElement = button.closest('.voice-message-container');
  if (!messageElement) {
    console.error('Message container not found');
    return;
  }

  const audioElement = messageElement.querySelector('audio');
  if (!audioElement) {
    console.error('Audio element not found');
    return;
  }

  // Pause all other audio elements
  document.querySelectorAll('.voice-message-container audio').forEach(audio => {
    if (audio !== audioElement && !audio.paused) {
      audio.pause();
      const otherButton = audio.closest('.voice-message-container')?.querySelector('.play-pause-btn');
      if (otherButton) {
        otherButton.innerHTML = '<i class="fas fa-play"></i>';
        const progressBar = audio.closest('.voice-message-container')?.querySelector('.voice-progress-bar');
        if (progressBar) {
          progressBar.style.width = '0%';
        }
      }
    }
  });

  if (audioElement.paused) {
    audioElement.play()
      .then(() => {
        button.innerHTML = '<i class="fas fa-pause"></i>';
      })
      .catch(error => {
        console.error('Error playing audio:', error);
        showAlert('Failed to play audio message', 'danger');
      });
  } else {
    audioElement.pause();
    button.innerHTML = '<i class="fas fa-play"></i>';
  }
}
async function loadOrganizationSettings() {
    try {
        const response = await fetch(`${API_BASE_URL}/organization`, {
            headers: {
                'Authorization': `Bearer ${localStorage.getItem('token')}`
            }
        });
        
        if (!response.ok) {
            throw new Error('Failed to load organization settings');
        }
        
        const orgData = await response.json();
        
        // Update form fields
        document.getElementById('org-name').value = orgData.name || '';
        
        // Update logo preview
        const logoPreview = document.getElementById('org-logo-img');
        const logoPlaceholder = document.getElementById('org-logo-placeholder');
        
        if (orgData.logo_url) {
            logoPreview.src = `${API_BASE_URL}${orgData.logo_url}`;
            logoPreview.style.display = 'block';
            logoPlaceholder.style.display = 'none';
        } else {
            logoPreview.style.display = 'none';
            logoPlaceholder.style.display = 'block';
        }
        
    } catch (error) {
        console.error('Error loading organization settings:', error);
        showAlert('Failed to load organization settings', 'danger');
    }
}

async function saveOrganizationSettings() {
    const name = document.getElementById('org-name').value.trim();
    const logoInput = document.getElementById('org-logo');
    const logoFile = logoInput.files[0];
    
    const formData = new FormData();
    if (name) formData.append('name', name);
    if (logoFile) formData.append('logo', logoFile);
    
    try {
        const response = await fetch(`${API_BASE_URL}/organization`, {
            method: 'PATCH',
            headers: {
                'Authorization': `Bearer ${localStorage.getItem('token')}`
            },
            body: formData
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Failed to update organization settings');
        }
        
        showAlert('Organization settings updated successfully', 'success');
        
        // Reload organization data
        await loadOrganizationSettings();
        
        // Update the sidebar header with new organization name and logo
        updateSidebarOrganizationInfo();
        
        // Clear file input
        logoInput.value = '';
        
    } catch (error) {
        console.error('Error saving organization settings:', error);
        showAlert(error.message || 'Failed to update organization settings', 'danger');
    }
}

function updateSidebarOrganizationInfo() {
    // This will be called after successful update to refresh the sidebar
    const sidebarHeader = document.querySelector('.sidebar-header h2');
    if (sidebarHeader) {
        const orgName = document.getElementById('org-name').value.trim();
        if (orgName) {
            sidebarHeader.textContent = orgName;
        }
    }
    
    // You can also update the logo in the sidebar if you have one
}

// Add event listeners
document.getElementById('org-settings-form').addEventListener('submit', function(e) {
    e.preventDefault();
    saveOrganizationSettings();
});
async function updateSidebarOrganizationInfo() {
    try {
        const response = await fetch(`${API_BASE_URL}/organization`, {
            headers: {
                'Authorization': `Bearer ${localStorage.getItem('token')}`
            }
        });
        
        if (!response.ok) return;
        
        const orgData = await response.json();
        
        // Update sidebar organization name
        const orgNameElement = document.getElementById('sidebar-org-name');
        if (orgNameElement) {
            orgNameElement.textContent = orgData.name || 'ReportHub';
        }
        
        // Update sidebar logo
        const logoImg = document.getElementById('sidebar-org-logo-img');
        const logoContainer = document.getElementById('sidebar-org-logo');
        
        if (orgData.logo_url) {
            logoImg.src = `${API_BASE_URL}${orgData.logo_url}`;
            logoImg.style.display = 'block';
            // Hide the building icon
            logoContainer.querySelector('i').style.display = 'none';
        } else {
            logoImg.style.display = 'none';
            // Show the building icon
            logoContainer.querySelector('i').style.display = 'block';
        }
        
    } catch (error) {
        console.error('Error updating sidebar organization info:', error);
    }
}
async function handleSupportFormSubmit(e) {
    e.preventDefault();
    
    const subject = document.getElementById('support-subject').value;
    const message = document.getElementById('support-message').value;
    const attachments = document.getElementById('support-attachments').files;
    
    if (!subject || !message) {
        showAlert('Please fill in all required fields', 'danger');
        return;
    }
    
    // Show loading state
    document.getElementById('submit-support-text').textContent = 'Sending...';
    document.getElementById('submit-support-spinner').classList.remove('hidden');
    document.getElementById('submit-support-btn').disabled = true;
    
    try {
        // Prepare email data
        const templateParams = {
            from_name: currentUser.name,
            from_email: currentUser.email,
            subject: `[ReportHub Support] ${subject}`,
            message: message,
            user_role: currentUser.role,
            organization: currentUser.organization_name || 'No organization'
        };
        
        // Initialize EmailJS with your credentials
        emailjs.init('-9DZ4wzmBt7s_maJN'); // Replace with your EmailJS user ID
        
        // Send email
        const response = await emailjs.send(
            'service_iv39jpy', // Replace with your service ID
            'template_l85d32m', // Replace with your template ID
            templateParams
        );
        
        // If there are attachments, send them separately (EmailJS has a 6MB limit per request)
        if (attachments.length > 0) {
            for (let i = 0; i < attachments.length; i++) {
                if (attachments[i].size > 5 * 1024 * 1024) {
                    showAlert(`Attachment ${attachments[i].name} is too large (max 5MB)`, 'warning');
                    continue;
                }
                
                const attachmentParams = {
                    ...templateParams,
                    attachment: attachments[i]
                };
                
                await emailjs.send(
                    'service_iv39jpy',
                    'template_l85d32m', // Different template for attachments if needed
                    attachmentParams
                );
            }
        }
        
        // Show success message
        showAlert('Your support request has been sent successfully!', 'success');
        
        // Reset form
        document.getElementById('support-form').reset();
        document.getElementById('support-attachments-list').innerHTML = '';
        
    } catch (error) {
        console.error('Error sending support request:', error);
        showAlert('Failed to send support request. Please try again later.', 'danger');
    } finally {
        // Reset button state
        document.getElementById('submit-support-text').textContent = 'Send Message';
        document.getElementById('submit-support-spinner').classList.add('hidden');
        document.getElementById('submit-support-btn').disabled = false;
    }
}

// Add this to handle file uploads for support attachments
document.getElementById('support-attachments')?.addEventListener('change', function(e) {
    const files = e.target.files;
    if (!files || files.length === 0) return;
    
    const attachmentsList = document.getElementById('support-attachments-list');
    attachmentsList.innerHTML = '';
    
    for (let i = 0; i < files.length; i++) {
        const file = files[i];
        const fileItem = document.createElement('div');
        fileItem.className = 'file-item';
        
        // Get file icon based on type
        let fileIcon, fileColor;
        if (file.type.includes('image')) {
            fileIcon = 'fa-file-image';
            fileColor = '#e74c3c';
        } else if (file.type.includes('pdf')) {
            fileIcon = 'fa-file-pdf';
            fileColor = '#e74c3c';
        } else if (file.type.includes('word') || file.type.includes('document')) {
            fileIcon = 'fa-file-word';
            fileColor = '#2c7be5';
        } else {
            fileIcon = 'fa-file';
            fileColor = '#95a5a6';
        }
        
        // Format file size
        const fileSize = formatFileSize(file.size);
        
        fileItem.innerHTML = `
            <i class="fas ${fileIcon} file-icon" style="color: ${fileColor};"></i>
            <div class="file-info">
                <div class="file-name">${file.name}</div>
                <div class="file-size">${fileSize}</div>
            </div>
            <i class="fas fa-times file-remove" data-index="${i}"></i>
        `;
        
        attachmentsList.appendChild(fileItem);
    }
    
    // Add event listeners to remove buttons
    document.querySelectorAll('#support-attachments-list .file-remove').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const index = parseInt(e.currentTarget.getAttribute('data-index'));
            removeFileFromList(index, 'support-attachments');
        });
    });
});

function removeFileFromList(index, inputId) {
    const input = document.getElementById(inputId);
    const dataTransfer = new DataTransfer();
    
    for (let i = 0; i < input.files.length; i++) {
        if (i !== index) {
            dataTransfer.items.add(input.files[i]);
        }
    }
    
    input.files = dataTransfer.files;
    const event = new Event('change');
    input.dispatchEvent(event);
}
function initTemplateManagement() {
    // Open template management modal
    document.getElementById('template-management-btn')?.addEventListener('click', () => {
        document.getElementById('template-management-modal').classList.add('active');
        loadTemplates();
    });
    
    // Close template management modal
    document.getElementById('close-template-management-modal')?.addEventListener('click', () => {
        document.getElementById('template-management-modal').classList.remove('active');
    });
    
    // Create new template button
    document.getElementById('create-template-btn')?.addEventListener('click', () => {
        openTemplateEditor();
    });
    
    // Close template editor modal
    document.getElementById('close-template-editor-modal')?.addEventListener('click', () => {
        document.getElementById('template-editor-modal').classList.remove('active');
    });
    
    // Cancel template editor
    document.getElementById('cancel-template-editor')?.addEventListener('click', () => {
        document.getElementById('template-editor-modal').classList.remove('active');
    });
    
    // Save template
    document.getElementById('save-template-btn')?.addEventListener('click', saveTemplate);
    
    // Add field button
    document.getElementById('add-field-btn')?.addEventListener('click', () => {
        openFieldEditor();
    });
    
    // Field type change handler
    document.getElementById('field-type')?.addEventListener('change', (e) => {
        const showOptions = ['dropdown', 'radio', 'checkbox'].includes(e.target.value);
        document.getElementById('field-options-group').classList.toggle('hidden', !showOptions);
    });
    
    // Save field button
    document.getElementById('save-field-btn')?.addEventListener('click', saveField);
    
    // Cancel field editor
    document.getElementById('cancel-field-editor')?.addEventListener('click', () => {
        document.getElementById('field-editor-modal').classList.remove('active');
    });
    
    // Template search
    document.getElementById('template-search')?.addEventListener('input', debounce(() => {
        loadTemplates();
    }, 300));
}

// Load templates from API
function loadTemplates() {
    const searchQuery = document.getElementById('template-search')?.value.toLowerCase() || '';
    
    fetch(`${API_BASE_URL}/templates`, {
        headers: {
            'Authorization': `Bearer ${localStorage.getItem('token')}`
        }
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Failed to fetch templates');
        }
        return response.json();
    })
    .then(data => {
        templates = data;
        renderTemplatesTable(data.filter(t => 
            t.name.toLowerCase().includes(searchQuery) || 
            t.category?.toLowerCase().includes(searchQuery)
        ));
    })
    .catch(error => {
        console.error('Error loading templates:', error);
        showAlert('Failed to load templates', 'danger');
    });
}

// Render templates table
function renderTemplatesTable(templates) {
    const tableBody = document.getElementById('templates-table-body');
    tableBody.innerHTML = '';
    
    if (templates.length === 0) {
        tableBody.innerHTML = `
            <tr>
                <td colspan="5" class="text-center text-muted">No templates found</td>
            </tr>
        `;
        return;
    }
    
    templates.forEach(template => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${template.name}</td>
            <td>${template.category || '-'}</td>
            <td>${template.fields?.length || 0}</td>
            <td>${new Date(template.updated_at || template.created_at).toLocaleDateString()}</td>
            <td>
                <button class="action-btn edit-template-btn" data-id="${template.id}" title="Edit">
                    <i class="fas fa-edit"></i>
                </button>
                <button class="action-btn delete-template-btn" data-id="${template.id}" title="Delete">
                    <i class="fas fa-trash"></i>
                </button>
                <button class="action-btn use-template-btn" data-id="${template.id}" title="Use Template">
                    <i class="fas fa-file-alt"></i>
                </button>
            </td>
        `;
        tableBody.appendChild(row);
    });
    
    // Add event listeners to action buttons
    document.querySelectorAll('.edit-template-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const templateId = e.currentTarget.getAttribute('data-id');
            editTemplate(templateId);
        });
    });
    
    document.querySelectorAll('.delete-template-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const templateId = e.currentTarget.getAttribute('data-id');
            deleteTemplate(templateId);
        });
    });
    
    document.querySelectorAll('.use-template-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const templateId = e.currentTarget.getAttribute('data-id');
            useTemplate(templateId);
        });
    });
}

// Open template editor (for new or existing template)
function openTemplateEditor(template = null) {
    currentTemplateId = template?.id || null;
    currentTemplateFields = template?.fields || [];
    
    // Set modal title
    document.getElementById('template-editor-title').textContent = 
        template ? `Edit Template: ${template.name}` : 'Create New Template';
    
    // Set form values
    if (template) {
        document.getElementById('template-name').value = template.name;
        document.getElementById('template-description').value = template.description || '';
        document.getElementById('template-category').value = template.category || '';
    } else {
        document.getElementById('template-editor-form').reset();
    }
    
    // Render fields
    renderTemplateFields();
    
    // Show modal
    document.getElementById('template-editor-modal').classList.add('active');
}

// Render template fields in the editor
function renderTemplateFields() {
    const container = document.getElementById('template-fields-container');
    container.innerHTML = '';
    
    if (currentTemplateFields.length === 0) {
        container.innerHTML = `
            <div class="text-center text-muted py-3">
                No fields added yet. Click "Add Field" to get started.
            </div>
        `;
        return;
    }
    
    // Sort fields by order
    currentTemplateFields.sort((a, b) => a.order - b.order);
    
    currentTemplateFields.forEach((field, index) => {
        const fieldElement = document.createElement('div');
        fieldElement.className = 'template-field-card';
        fieldElement.setAttribute('data-index', index);
        
        let fieldTypeBadge;
        switch(field.field_type) {
            case 'dropdown':
                fieldTypeBadge = '<span class="badge badge-info">Dropdown</span>';
                break;
            case 'checkbox':
                fieldTypeBadge = '<span class="badge badge-info">Checkbox</span>';
                break;
            case 'radio':
                fieldTypeBadge = '<span class="badge badge-info">Radio</span>';
                break;
            case 'date':
                fieldTypeBadge = '<span class="badge badge-info">Date</span>';
                break;
            case 'file':
                fieldTypeBadge = '<span class="badge badge-info">File</span>';
                break;
            default:
                fieldTypeBadge = '<span class="badge badge-secondary">Text</span>';
        }
        
        fieldElement.innerHTML = `
            <div class="d-flex justify-content-between align-items-center">
                <div>
                    <strong>${field.label}</strong>
                    ${field.required ? '<span class="badge badge-danger ml-2">Required</span>' : ''}
                    ${fieldTypeBadge}
                    <div class="text-muted small">${field.name}</div>
                </div>
                <div>
                    <button class="btn btn-sm btn-outline edit-field-btn" data-index="${index}">
                        <i class="fas fa-edit"></i>
                    </button>
                    <button class="btn btn-sm btn-outline-danger delete-field-btn" data-index="${index}">
                        <i class="fas fa-trash"></i>
                    </button>
                </div>
            </div>
        `;
        
        container.appendChild(fieldElement);
    });
    
    // Add event listeners to field buttons
    document.querySelectorAll('.edit-field-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const index = parseInt(e.currentTarget.getAttribute('data-index'));
            editField(index);
        });
    });
    
    document.querySelectorAll('.delete-field-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const index = parseInt(e.currentTarget.getAttribute('data-index'));
            deleteField(index);
        });
    });
}

// Open field editor (for new or existing field)
function openFieldEditor(field = null, index = null) {
    currentFieldIndex = index;
    
    // Reset form
    document.getElementById('field-editor-form').reset();
    document.getElementById('field-options-group').classList.add('hidden');
    
    // Set form values if editing existing field
    if (field) {
        document.getElementById('field-name').value = field.name;
        document.getElementById('field-label').value = field.label;
        document.getElementById('field-type').value = field.field_type;
        document.getElementById('field-required').checked = field.required;
        document.getElementById('field-placeholder').value = field.placeholder || '';
        document.getElementById('field-default-value').value = field.default_value || '';
        document.getElementById('field-order').value = field.order || 0;
        
        // Handle options for dropdown/radio/checkbox
        if (field.options) {
            const optionsText = Object.entries(field.options)
                .map(([value, label]) => `${value}|${label}`)
                .join('\n');
            document.getElementById('field-options').value = optionsText;
        }
        
        // Show options group if needed
        if (['dropdown', 'radio', 'checkbox'].includes(field.field_type)) {
            document.getElementById('field-options-group').classList.remove('hidden');
        }
    }
    
    // Show modal
    document.getElementById('field-editor-modal').classList.add('active');
}

// Save field to template
function saveField() {
    const name = document.getElementById('field-name').value.trim();
    const label = document.getElementById('field-label').value.trim();
    const fieldType = document.getElementById('field-type').value;
    const required = document.getElementById('field-required').checked;
    const placeholder = document.getElementById('field-placeholder').value.trim();
    const defaultValue = document.getElementById('field-default-value').value.trim();
    const order = parseInt(document.getElementById('field-order').value) || 0;
    
    // Validate
    if (!name || !label) {
        showAlert('Field name and label are required', 'danger');
        return;
    }
    
    // Validate field name format (no spaces, only letters, numbers, underscores)
    if (!/^[a-zA-Z0-9_]+$/.test(name)) {
        showAlert('Field name can only contain letters, numbers, and underscores', 'danger');
        return;
    }
    
    // Process options for dropdown/radio/checkbox
    let options = null;
    if (['dropdown', 'radio', 'checkbox'].includes(fieldType)) {
        const optionsText = document.getElementById('field-options').value.trim();
        if (optionsText) {
            options = {};
            optionsText.split('\n').forEach(line => {
                const parts = line.split('|');
                if (parts.length === 2) {
                    options[parts[0].trim()] = parts[1].trim();
                } else if (parts.length === 1) {
                    options[parts[0].trim()] = parts[0].trim();
                }
            });
        }
    }
    
    // Create field object
    const field = {
        name,
        label,
        field_type: fieldType,
        required,
        placeholder: placeholder || null,
        default_value: defaultValue || null,
        options,
        order
    };
    
    // Add or update field in template
    if (currentFieldIndex !== null) {
        currentTemplateFields[currentFieldIndex] = field;
    } else {
        currentTemplateFields.push(field);
    }
    
    // Close modal and refresh fields display
    document.getElementById('field-editor-modal').classList.remove('active');
    renderTemplateFields();
}

// Edit existing field
function editField(index) {
    if (index >= 0 && index < currentTemplateFields.length) {
        openFieldEditor(currentTemplateFields[index], index);
    }
}

// Delete field from template
function deleteField(index) {
    if (index >= 0 && index < currentTemplateFields.length) {
        if (confirm('Are you sure you want to delete this field?')) {
            currentTemplateFields.splice(index, 1);
            renderTemplateFields();
        }
    }
}

// Save template to API
function saveTemplate() {
    const name = document.getElementById('template-name').value.trim();
    const description = document.getElementById('template-description').value.trim();
    const category = document.getElementById('template-category').value.trim();
    
    if (!name) {
        showAlert('Template name is required', 'danger');
        return;
    }
    
    if (currentTemplateFields.length === 0) {
        showAlert('Please add at least one field to the template', 'danger');
        return;
    }
    
    // Show loading state
    document.getElementById('save-template-text').textContent = 'Saving...';
    document.getElementById('save-template-spinner').classList.remove('hidden');
    
    const templateData = {
        name,
        description: description || null,
        category: category || null,
        fields: currentTemplateFields
    };
    
    const method = currentTemplateId ? 'PUT' : 'POST';
    const url = currentTemplateId ? 
        `${API_BASE_URL}/templates/${currentTemplateId}` : 
        `${API_BASE_URL}/templates`;
    
    fetch(url, {
        method,
        headers: {
            'Authorization': `Bearer ${localStorage.getItem('token')}`,
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(templateData)
    })
    .then(response => {
        if (!response.ok) {
            return response.json().then(err => {
                throw new Error(err.detail || 'Failed to save template');
            });
        }
        return response.json();
    })
    .then(() => {
        showAlert('Template saved successfully', 'success');
        document.getElementById('template-editor-modal').classList.remove('active');
        loadTemplates();
    })
    .catch(error => {
        showAlert(error.message, 'danger');
    })
    .finally(() => {
        document.getElementById('save-template-text').textContent = 'Save Template';
        document.getElementById('save-template-spinner').classList.add('hidden');
    });
}

// Edit existing template
function editTemplate(templateId) {
    const template = templates.find(t => t.id == templateId);
    if (template) {
        openTemplateEditor(template);
    }
}

// Delete template
function deleteTemplate(templateId) {
    if (!confirm('Are you sure you want to delete this template? Reports created with this template will not be affected.')) {
        return;
    }
    
    fetch(`${API_BASE_URL}/templates/${templateId}`, {
        method: 'DELETE',
        headers: {
            'Authorization': `Bearer ${localStorage.getItem('token')}`
        }
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Failed to delete template');
        }
        return response.json();
    })
    .then(() => {
        showAlert('Template deleted successfully', 'success');
        loadTemplates();
    })
    .catch(error => {
        console.error('Error deleting template:', error);
        showAlert('Failed to delete template', 'danger');
    });
}

// Use template to create a report
function useTemplate(templateId) {
    const template = templates.find(t => t.id == templateId);
    if (!template) return;
    
    // Close template management modal
    document.getElementById('template-management-modal').classList.remove('active');
    
    // Open report creation modal with template
    createReportFromTemplate(template);
}

// Create report from template
function createReportFromTemplate(template) {
    // Reset create report modal
    document.getElementById('create-report-form').reset();
    document.getElementById('report-attachments-list').innerHTML = '';
    
    // Set template name as default title
    document.getElementById('report-title').value = template.name;
    
    // Set template category if available
    if (template.category) {
        document.getElementById('report-category').value = template.category;
    }
    
    // Initialize Quill editor if not already done
    if (!quill) {
        initQuillEditor();
    } else {
        // Clear existing content
        quill.root.innerHTML = '';
        document.getElementById('report-description').value = '';
    }
    
    // Create a container for dynamic fields
    const fieldsContainer = document.createElement('div');
    fieldsContainer.className = 'template-fields-container';
    fieldsContainer.id = 'dynamic-fields-container';
    
    // Add fields to container
    template.fields.sort((a, b) => a.order - b.order).forEach(field => {
        const fieldElement = createFieldElement(field);
        fieldsContainer.appendChild(fieldElement);
    });
    
    // Insert fields container after the description editor's parent
    const descriptionEditorParent = document.getElementById('report-description-editor').parentNode;
    descriptionEditorParent.parentNode.insertBefore(fieldsContainer, descriptionEditorParent.nextSibling);
    
    // Show create report modal
    document.getElementById('create-report-modal').classList.add('active');
}

// Create HTML element for a template field
function createFieldElement(field) {
    const fieldGroup = document.createElement('div');
    fieldGroup.className = 'form-group template-field';
    fieldGroup.dataset.fieldName = field.name;
    
    const label = document.createElement('label');
    label.className = 'form-label';
    label.textContent = field.label;
    if (field.required) {
        label.innerHTML += ' <span class="text-danger">*</span>';
    }
    
    fieldGroup.appendChild(label);
    
    let inputElement;
    
    switch(field.field_type) {
        case 'textarea':
            inputElement = document.createElement('textarea');
            inputElement.className = 'form-control';
            inputElement.rows = 3;
            break;
            
        case 'dropdown':
            inputElement = document.createElement('select');
            inputElement.className = 'form-control';
            
            if (field.placeholder) {
                const placeholderOption = document.createElement('option');
                placeholderOption.value = '';
                placeholderOption.textContent = field.placeholder;
                placeholderOption.selected = true;
                placeholderOption.disabled = true;
                inputElement.appendChild(placeholderOption);
            }
            
            if (field.options) {
                Object.entries(field.options).forEach(([value, label]) => {
                    const option = document.createElement('option');
                    option.value = value;
                    option.textContent = label;
                    if (field.default_value && value === field.default_value) {
                        option.selected = true;
                    }
                    inputElement.appendChild(option);
                });
            }
            break;
            
        case 'checkbox':
            inputElement = document.createElement('div');
            if (field.options) {
                Object.entries(field.options).forEach(([value, label]) => {
                    const checkboxGroup = document.createElement('div');
                    checkboxGroup.className = 'form-check';
                    
                    const checkbox = document.createElement('input');
                    checkbox.type = 'checkbox';
                    checkbox.className = 'form-check-input';
                    checkbox.id = `${field.name}_${value}`;
                    checkbox.value = value;
                    checkbox.name = field.name;
                    
                    const checkboxLabel = document.createElement('label');
                    checkboxLabel.className = 'form-check-label';
                    checkboxLabel.htmlFor = checkbox.id;
                    checkboxLabel.textContent = label;
                    
                    checkboxGroup.appendChild(checkbox);
                    checkboxGroup.appendChild(checkboxLabel);
                    inputElement.appendChild(checkboxGroup);
                });
            } else {
                // Single checkbox
                const checkboxGroup = document.createElement('div');
                checkboxGroup.className = 'form-check';
                
                const checkbox = document.createElement('input');
                checkbox.type = 'checkbox';
                checkbox.className = 'form-check-input';
                checkbox.id = field.name;
                checkbox.value = 'true';
                
                const checkboxLabel = document.createElement('label');
                checkboxLabel.className = 'form-check-label';
                checkboxLabel.htmlFor = checkbox.id;
                checkboxLabel.textContent = field.label;
                
                checkboxGroup.appendChild(checkbox);
                checkboxGroup.appendChild(checkboxLabel);
                inputElement.appendChild(checkboxGroup);
            }
            break;
            
        case 'radio':
            inputElement = document.createElement('div');
            if (field.options) {
                Object.entries(field.options).forEach(([value, label]) => {
                    const radioGroup = document.createElement('div');
                    radioGroup.className = 'form-check';
                    
                    const radio = document.createElement('input');
                    radio.type = 'radio';
                    radio.className = 'form-check-input';
                    radio.id = `${field.name}_${value}`;
                    radio.value = value;
                    radio.name = field.name;
                    if (field.default_value && value === field.default_value) {
                        radio.checked = true;
                    }
                    
                    const radioLabel = document.createElement('label');
                    radioLabel.className = 'form-check-label';
                    radioLabel.htmlFor = radio.id;
                    radioLabel.textContent = label;
                    
                    radioGroup.appendChild(radio);
                    radioGroup.appendChild(radioLabel);
                    inputElement.appendChild(radioGroup);
                });
            }
            break;
            
        case 'date':
            inputElement = document.createElement('input');
            inputElement.type = 'date';
            inputElement.className = 'form-control';
            break;
            
        case 'datetime':
            inputElement = document.createElement('input');
            inputElement.type = 'datetime-local';
            inputElement.className = 'form-control';
            break;
            
        case 'file':
            inputElement = document.createElement('input');
            inputElement.type = 'file';
            inputElement.className = 'form-control';
            break;
            
        case 'number':
            inputElement = document.createElement('input');
            inputElement.type = 'number';
            inputElement.className = 'form-control';
            break;
            
        default: // text
            inputElement = document.createElement('input');
            inputElement.type = 'text';
            inputElement.className = 'form-control';
    }
    
    // Set common attributes
    if (['input', 'textarea', 'select'].includes(inputElement.tagName.toLowerCase())) {
        inputElement.id = field.name;
        inputElement.name = field.name;
        inputElement.required = field.required;
        
        if (field.placeholder) {
            inputElement.placeholder = field.placeholder;
        }
        
        if (field.default_value && !['checkbox', 'radio'].includes(field.field_type)) {
            inputElement.value = field.default_value;
        }
    }
    
    fieldGroup.appendChild(inputElement);
    
    return fieldGroup;
}
function updateMessageVisibility() {
    const messagesContainer = document.getElementById('chat-messages');
    const noMessagesContainer = document.getElementById('no-messages-container');
    
    if (messagesContainer.children.length === 1) { // Only the no-messages container exists
        noMessagesContainer.classList.remove('hidden');
    } else {
        noMessagesContainer.classList.add('hidden');
    }
}

        // Initialize the app
        document.addEventListener('DOMContentLoaded', init);
    </script>
    
</body>
</html>backend;from fastapi import FastAPI, HTTPException, Depends, status, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, EmailStr, validator
from typing import List, Optional, Dict
from datetime import datetime, timedelta
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
from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime, ForeignKey, func, inspect, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from sqlalchemy.sql import text
import boto3
from botocore.exceptions import ClientError

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
    logo_url = Column(String, nullable=True)
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
        
# Reset database and initialize data
init_default_admin()

# Pydantic models (remain the same as before)
class Token(BaseModel):
    access_token: str
    token_type: str
    requires_org_registration: bool = False

class TokenData(BaseModel):
    email: Optional[str] = None

class UserBase(BaseModel):
    email: EmailStr
    name: str

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
    organization_name: str
    created_at: datetime
    updated_at: Optional[datetime]
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
    
    # Create user without organization (will be set later)
    hashed_password = get_password_hash(password)
    
    # First user becomes admin, others are staff
    is_first_user = db.query(User).count() == 0
    role = "admin" if is_first_user else "staff"
    
    db_user = User(
        email=email,
        name=name,
        hashed_password=hashed_password,
        role=role,
        organization_id=None  # Organization will be set after registration
    )
    
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": db_user.email}, expires_delta=access_token_expires
    )
    
    # All new signups require organization registration
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "requires_org_registration": True
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
        # Upload the voice message to B2
        file_url = await upload_to_b2(voice_message, f"voice_messages/{current_user.id}")
        
        # Save message to database
        db_message = ChatMessage(
            sender_id=current_user.id,
            recipient_id=recipient_id,
            content=f"[VOICE_MESSAGE]{file_url}",
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
        logo_url=org.logo_url,
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

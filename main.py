from fastapi import FastAPI, HTTPException, Depends, status, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
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
from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime, ForeignKey, func, inspect
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from sqlalchemy.sql import text
import boto3
from botocore.exceptions import ClientError

# Backblaze B2 Configuration
B2_BUCKET_NAME = "-"
B2_ENDPOINT_URL = "https://s3.us-east-005.backblazeb2.com"
B2_KEY_ID = "5ca7845641d3"
B2_APPLICATION_KEY = "0056c87f4b62022cc31fff83e42073b124680c4844"

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

def add_logo_url_column():
    db = SessionLocal()
    try:
        # Check if the column already exists
        inspector = inspect(db.get_bind())
        columns = inspector.get_columns('organizations')
        column_names = [column['name'] for column in columns]
        
        if 'logo_url' not in column_names:
            # Add the column if it doesn't exist
            db.execute(text("ALTER TABLE organizations ADD COLUMN logo_url VARCHAR"))
            db.commit()
            print("Successfully added logo_url column to organizations table")
        else:
            print("logo_url column already exists in organizations table")
    except Exception as e:
        db.rollback()
        print(f"Error adding logo_url column: {e}")
        raise e
    finally:
        db.close()

# Reset database and initialize data
init_default_admin()
add_logo_url_column()

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
@app.post("/reports", response_model=ReportInDB)
async def create_report(
    title: str = Form(...),
    description: str = Form(...),
    category: str = Form(...),
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
    
    # Create report
    db_report = Report(
        title=title,
        description=description,
        category=category,
        author_id=current_user.id,
        organization_id=current_user.organization_id if current_user.role != "super_admin" else None
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

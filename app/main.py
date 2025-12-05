"""
SummAI - Automated Business Text Summarization API (ML Version)

A FastAPI-based REST API for summarizing business documents using both:
1. Rule-based extractive summarization (NLTK)
2. ML-based abstractive summarization (Transformer models)

===============================================================================
📖 HOW TO RUN THIS PROJECT
===============================================================================

OPTION 1: Use Pre-trained NLTK Summarizer (No ML Training Required)
--------------------------------------------------------------------
1. Install dependencies:
   pip install -r requirements.txt

2. Run the server:
   uvicorn main:app --reload

3. Test at: http://127.0.0.1:8000/docs


OPTION 2: Train and Use ML Model (Recommended for Best Results)
----------------------------------------------------------------
1. Install dependencies:
   pip install -r requirements.txt

2. Train the ML model:
   python train.py
   (This will download CNN/DailyMail dataset and train a T5 model)
   (Training takes ~30-60 minutes on GPU, longer on CPU)

3. Evaluate the model:
   python evaluate.py

4. Run the server:
   uvicorn main:app --reload
   for mac -python3 -m uvicorn main:app --reload

5. Test at: http://127.0.0.1:8000/docs


QUICK TEST: Generate Prediction Without API
--------------------------------------------
python predict.py --text "Your business text here..."

OR interactive mode:
python predict.py

===============================================================================
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from app.core.summarizer import summarize_text as nltk_summarize, download_nltk_data
from typing import Optional, Dict
import os
import torch
import io
import json

from pypdf import PdfReader

# Try to import ML components
try:
    from app.ml.predict import Summarizer
    ML_AVAILABLE = True
except Exception as e:
    ML_AVAILABLE = False
    print(f"⚠️  ML components not available: {str(e)}")


# Initialize FastAPI app
app = FastAPI(
    title="SummAI - ML-Powered Business Text Summarization",
    description="Advanced REST API for summarizing business documents using transformer models",
    version="2.0.0"
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")


# Global ML model instance (lazy loaded)
ml_summarizer = None


# Request and Response Models
class SummarizeRequest(BaseModel):
    """Request model for text summarization"""
    text: str = Field(
        ..., 
        description="The business text to summarize",
        min_length=1,
        example="The global technology market is experiencing unprecedented growth..."
    )
    use_ml: bool = Field(
        default=True,
        description="Use ML model if available, otherwise fall back to NLTK"
    )


class SummarizeResponse(BaseModel):
    """Response model containing the summary and metadata"""
    summary: str = Field(..., description="The generated summary")
    original_length: int = Field(..., description="Number of characters in original text")
    summary_length: int = Field(..., description="Number of characters in summary")
    model_used: str = Field(..., description="Which model generated the summary")


class MetricsResponse(BaseModel):
    """Response model for evaluation metrics"""
    eval_loss: float
    perplexity: float
    token_accuracy: float
    token_precision: float
    token_recall: float
    token_f1: float
    rouge1: float
    rouge2: float
    rougeL: float


class TrainRequest(BaseModel):
    """Request model for training"""
    num_epochs: int = Field(default=3, ge=1, le=10, description="Number of training epochs")
    batch_size: int = Field(default=4, ge=1, le=32, description="Training batch size")
    learning_rate: float = Field(default=5e-5, gt=0, description="Learning rate")


class StatusResponse(BaseModel):
    """Response model for status checks"""
    status: str
    ml_available: bool
    ml_model_loaded: bool
    nltk_available: bool


class UserSignup(BaseModel):
    username: str
    email: str
    password: str


class UserLogin(BaseModel):
    email: str
    password: str


USERS_FILE = "users.json"


def load_users():
    if not os.path.exists(USERS_FILE):
        return []
    with open(USERS_FILE, "r") as f:
        return json.load(f)


def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=4)


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize components when the application starts"""
    print("🚀 Starting SummAI ML...")
    
    # Download NLTK data
    download_nltk_data()
    
    # Check ML model availability
    if ML_AVAILABLE:
        model_path = "./model/best_model"
        if os.path.exists(model_path):
            print("✅ ML model found! Loading...")
            try:
                global ml_summarizer
                ml_summarizer = Summarizer(model_path)
                print("✅ ML model loaded successfully!")
            except Exception as e:
                print(f"⚠️  Failed to load ML model: {str(e)}")
        else:
            print("ℹ️  No trained ML model found. Train one with: python train.py")
    
    print("✅ SummAI is ready!")


# API Endpoints

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the web UI"""
    try:
        with open("static/index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return HTMLResponse(
            content="<h1>SummAI</h1><p>UI not found. Please ensure static/index.html exists.</p>",
            status_code=404
        )


@app.get("/login", response_class=HTMLResponse)
async def login_page():
    """Serve the login page"""
    try:
        with open("static/login.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return HTMLResponse(content="Login page not found", status_code=404)


@app.get("/signup", response_class=HTMLResponse)
async def signup_page():
    """Serve the signup page"""
    try:
        with open("static/signup.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return HTMLResponse(content="Signup page not found", status_code=404)


@app.post("/api/signup")
async def register_user(user: UserSignup):
    users = load_users()
    
    # Check if email already exists
    if any(u['email'] == user.email for u in users):
        raise HTTPException(status_code=400, detail="Email already registered")
    
    new_user = {
        "username": user.username,
        "email": user.email,
        "password": user.password  # In a real app, hash this!
    }
    
    users.append(new_user)
    save_users(users)
    
    return {"message": "User registered successfully"}


@app.post("/api/login")
async def login_user(user: UserLogin):
    users = load_users()
    
    # Find user
    valid_user = next((u for u in users if u['email'] == user.email and u['password'] == user.password), None)
    
    if not valid_user:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    
    return {"message": "Login successful", "username": valid_user['username']}

@app.get("/api")
def api_info():
    """API information endpoint"""
    return {
        "message": "SummAI - ML-Powered Business Text Summarization API",
        "version": "2.0.0",
        "status": "active",
        "ml_available": ML_AVAILABLE and ml_summarizer is not None,
        "endpoints": {
            "ui": "/",
            "docs": "/docs",
            "summarize": "/summarize (POST)",
            "evaluate": "/evaluate (GET)",
            "train": "/train (POST)",
            "status": "/status (GET)"
        }
    }


@app.get("/status", response_model=StatusResponse)
def get_status():
    """Get API status and capabilities"""
    return StatusResponse(
        status="healthy",
        ml_available=ML_AVAILABLE,
        ml_model_loaded=ml_summarizer is not None,
        nltk_available=True
    )


@app.post("/summarize", response_model=SummarizeResponse)
def summarize(request: SummarizeRequest):
    """
    Summarize business text
    
    This endpoint can use either:
    - ML model (T5/BART transformer) for abstractive summarization
    - NLTK model for extractive summarization (fallback)
    """
    
    # Validate input
    if not request.text or request.text.strip() == "":
        raise HTTPException(
            status_code=400,
            detail="Text cannot be empty."
        )
    
    try:
        # Try ML model first if requested and available
        if request.use_ml and ml_summarizer is not None:
            summary = ml_summarizer.generate_summary(request.text)
            model_used = "ML (Transformer)"
        
        # Fall back to NLTK
        else:
            summary = nltk_summarize(request.text)
            model_used = "NLTK (Extractive)"
            
            # Check if NLTK failed
            if summary.startswith("No summary could be generated"):
                raise HTTPException(status_code=400, detail=summary)
        
        # Calculate lengths
        original_length = len(request.text)
        summary_length = len(summary)
        
        return SummarizeResponse(
            summary=summary,
            original_length=original_length,
            summary_length=summary_length,
            model_used=model_used
        )
    
    except HTTPException:
        raise
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Summarization failed: {str(e)}"
        )


@app.get("/evaluate", response_model=MetricsResponse)
def evaluate():
    """
    Evaluate the trained ML model
    
    Returns comprehensive metrics including:
    - Token-level accuracy, precision, recall, F1
    - ROUGE-1, ROUGE-2, ROUGE-L scores
    - Perplexity
    """
    
    if not ML_AVAILABLE or ml_summarizer is None:
        raise HTTPException(
            status_code=400,
            detail="ML model not available. Train a model first with: python train.py"
        )
    
    try:
        # Load evaluation results if they exist
        results_path = "./model/best_model/evaluation_results.json"
        
        if not os.path.exists(results_path):
            raise HTTPException(
                status_code=404,
                detail="No evaluation results found. Run: python evaluate.py"
            )
        
        import json
        with open(results_path, 'r') as f:
            metrics = json.load(f)
        
        return MetricsResponse(**metrics)
    
    except HTTPException:
        raise
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve metrics: {str(e)}"
        )


@app.post("/extract-text")
async def extract_text(file: UploadFile = File(...)):
    """
    Extract text from uploaded file (PDF or TXT)
    """
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    filename = file.filename.lower()
    content = await file.read()
    
    text = ""
    
    try:
        if filename.endswith('.pdf'):
            # Extract from PDF
            pdf_file = io.BytesIO(content)
            reader = PdfReader(pdf_file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        elif filename.endswith('.txt'):
            # Extract from TXT
            text = content.decode('utf-8')
        else:
            raise HTTPException(
                status_code=400, 
                detail="Unsupported file format. Please upload PDF or TXT."
            )
            
        return {"text": text.strip(), "filename": file.filename}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to extract text: {str(e)}")


@app.post("/train")
async def train(request: TrainRequest, background_tasks: BackgroundTasks):
    """
    Train a new ML model (Background task)
    
    Note: This starts training in the background.
    Training can take 30-60 minutes depending on hardware.
    """
    
    if not ML_AVAILABLE:
        raise HTTPException(
            status_code=400,
            detail="ML components not available. Check installation."
        )
    
    # Training is resource-intensive, recommend running separately
    return {
        "message": "Training via API is not recommended due to resource intensity.",
        "recommendation": "Please run training from command line:",
        "command": "python train.py",
        "parameters": {
            "epochs": request.num_epochs,
            "batch_size": request.batch_size,
            "learning_rate": request.learning_rate
        },
        "note": "Edit config.py to adjust these parameters before training"
    }


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "SummAI ML",
        "nltk_ready": True,
        "ml_ready": ml_summarizer is not None,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }


# Example test data
"""
EXAMPLE TEST DATA for /summarize endpoint:

{
  "text": "The global technology market is experiencing unprecedented growth in 2024. Artificial intelligence and machine learning continue to dominate industry discussions, with companies investing billions in AI infrastructure. Cloud computing has become essential for business operations, enabling remote work and digital transformation. Cybersecurity remains a top priority as organizations face increasingly sophisticated threats. The competitive landscape is rapidly evolving, with startups challenging established players through innovative solutions. Market analysts predict that AI adoption will increase productivity by 40% over the next five years. Companies that fail to embrace digital transformation risk falling behind their competitors. Customer experience has emerged as the key differentiator in saturated markets. Data analytics and business intelligence tools are empowering organizations to make data-driven decisions. The industry is witnessing a shift towards sustainable and ethical technology practices.",
  "use_ml": true
}
"""


# Run the application
if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting SummAI ML server...")
    print("📝 Visit http://127.0.0.1:8000/docs for API documentation")
    uvicorn.run(app, host="0.0.0.0", port=8000)

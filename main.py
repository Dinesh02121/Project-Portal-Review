from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import os
import json
import hashlib
import logging
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
import zipfile
import io
import requests

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Project Analysis API",
    description="Comprehensive code review and assessment using OpenAI/Groq",
    version="2.0.0"
)

# CORS configuration
allowed_origins_str = os.getenv("ALLOWED_ORIGINS", "")
env = os.getenv("ENV", "development")

if not allowed_origins_str:
    logger.warning("ALLOWED_ORIGINS not set")
    if env == "development":
        allowed_origins = ["http://localhost:3000", "http://localhost:8080", "*"]
    else:
        allowed_origins = []
else:
    allowed_origins = [origin.strip() for origin in allowed_origins_str.split(",") if origin.strip()]

logger.info(f"Environment: {env}")
logger.info(f"Allowed origins: {allowed_origins}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins if allowed_origins else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Supabase configuration (pure HTTP, no SDK)
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")  # Service role key for backend
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "projects")

if not SUPABASE_URL or not SUPABASE_KEY:
    logger.error("Supabase credentials not found")
    raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set")

logger.info(f"Supabase configured - URL: {SUPABASE_URL}, Bucket: {SUPABASE_BUCKET}")

# Initialize OpenAI/Groq client
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    logger.error("OPENAI_API_KEY not found")
    raise ValueError("OPENAI_API_KEY must be set")

# Auto-detect Groq or OpenAI
if api_key.startswith('gsk_'):
    logger.info("Using Groq API")
    client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
    MODEL = "llama-3.3-70b-versatile"
else:
    logger.info("Using OpenAI API")
    client = OpenAI(api_key=api_key)
    MODEL = "gpt-4o-mini"

# Simple in-memory cache
analysis_cache = {}

class AnalysisRequest(BaseModel):
    project_path: str  # Supabase storage path (e.g., "projects/123/file.zip")
    project_name: str
    student_description: str

class AnalysisResponse(BaseModel):
    project_name: str
    student_description: str
    detected_tech_stack: Dict[str, List[str]]
    code_quality_score: float
    overall_grade: str
    detailed_analysis: Dict[str, str]
    recommendations: List[str]
    strengths: List[str]
    weaknesses: List[str]
    analysis_timestamp: str

def download_from_supabase(storage_path: str) -> bytes:
    """Download ZIP from Supabase Storage using pure HTTP requests"""
    try:
        logger.info(f"Downloading from Supabase: {storage_path}")
        
        # Clean path
        clean_path = storage_path.lstrip('/')
        
        # Build storage URL
        # Format: {SUPABASE_URL}/storage/v1/object/{bucket}/{path}
        storage_url = f"{SUPABASE_URL}/storage/v1/object/{SUPABASE_BUCKET}/{clean_path}"
        
        logger.info(f"Storage URL: {storage_url}")
        
        # Make authenticated request
        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}"
        }
        
        response = requests.get(storage_url, headers=headers, timeout=60)
        
        if response.status_code == 200:
            logger.info(f"Successfully downloaded {len(response.content)} bytes")
            return response.content
        else:
            error_msg = f"HTTP {response.status_code}: {response.text[:200]}"
            logger.error(f"Download failed: {error_msg}")
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Failed to download from Supabase: {error_msg}"
            )
            
    except requests.exceptions.Timeout:
        logger.error("Timeout downloading from Supabase")
        raise HTTPException(status_code=504, detail="Download timeout")
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error: {e}")
        raise HTTPException(status_code=500, detail=f"Download error: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")

def get_cache_key(project_path: str) -> str:
    """Generate cache key"""
    try:
        cache_str = f"{project_path}:{datetime.now().strftime('%Y-%m-%d')}"
        return hashlib.md5(cache_str.encode()).hexdigest()
    except:
        return None

def detect_technology_stack(zip_bytes: bytes) -> Dict[str, List[str]]:
    """Detect technologies from ZIP"""
    tech_stack = {
        "languages": [],
        "frameworks": [],
        "databases": [],
        "tools": []
    }
    
    try:
        logger.info("Detecting tech stack")
        
        language_map = {
            '.py': 'Python', '.js': 'JavaScript', '.jsx': 'React/JSX',
            '.ts': 'TypeScript', '.tsx': 'React/TypeScript', '.java': 'Java',
            '.cpp': 'C++', '.c': 'C', '.cs': 'C#', '.go': 'Go',
            '.rs': 'Rust', '.php': 'PHP', '.rb': 'Ruby', '.swift': 'Swift',
            '.kt': 'Kotlin', '.html': 'HTML', '.css': 'CSS', '.sql': 'SQL'
        }
        
        framework_patterns = {
            'package.json': ['react', 'vue', 'angular', 'express', 'next'],
            'requirements.txt': ['django', 'flask', 'fastapi', 'pandas', 'numpy'],
            'pom.xml': ['spring', 'hibernate'],
            'build.gradle': ['spring', 'android'],
            'Gemfile': ['rails', 'sinatra'],
            'composer.json': ['laravel', 'symfony'],
            'go.mod': ['gin', 'echo'],
            'Cargo.toml': ['actix', 'rocket']
        }
        
        db_keywords = {
            'mysql': 'MySQL', 'postgresql': 'PostgreSQL', 'mongodb': 'MongoDB',
            'sqlite': 'SQLite', 'redis': 'Redis', 'supabase': 'Supabase',
            'firebase': 'Firebase'
        }
        
        tool_files = {
            'Dockerfile': 'Docker', 'docker-compose.yml': 'Docker Compose',
            '.gitignore': 'Git', 'package.json': 'npm'
        }
        
        languages_found = set()
        frameworks_found = set()
        databases_found = set()
        tools_found = set()
        
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            for file_info in zf.namelist():
                if '__MACOSX' in file_info or '.DS_Store' in file_info:
                    continue
                
                # Languages
                for ext, lang in language_map.items():
                    if file_info.lower().endswith(ext):
                        languages_found.add(lang)
                
                filename = os.path.basename(file_info)
                
                # Frameworks
                if filename in framework_patterns:
                    try:
                        content = zf.read(file_info).decode('utf-8', errors='ignore').lower()
                        for fw in framework_patterns[filename]:
                            if fw in content:
                                frameworks_found.add(fw.capitalize())
                    except:
                        pass
                
                # Databases
                if file_info.lower().endswith(('.py', '.js', '.java', '.env', '.yml')):
                    try:
                        content = zf.read(file_info).decode('utf-8', errors='ignore').lower()
                        for kw, db in db_keywords.items():
                            if kw in content:
                                databases_found.add(db)
                    except:
                        pass
                
                # Tools
                for tool_file, tool_name in tool_files.items():
                    if file_info.endswith(tool_file):
                        tools_found.add(tool_name)
        
        tech_stack["languages"] = sorted(list(languages_found))
        tech_stack["frameworks"] = sorted(list(frameworks_found))
        tech_stack["databases"] = sorted(list(databases_found))
        tech_stack["tools"] = sorted(list(tools_found))
        
        logger.info(f"Tech stack: {tech_stack}")
        
    except Exception as e:
        logger.error(f"Error detecting tech stack: {e}")
    
    return tech_stack

def read_project_files(zip_bytes: bytes, max_files: int = 20) -> Dict[str, str]:
    """Read project files from ZIP"""
    files_content = {}
    
    try:
        logger.info("Reading project files")
        
        skip_dirs = {
            'node_modules', 'venv', '__pycache__', 'build', 'dist',
            '.git', 'target', '__MACOSX'
        }
        
        priority_extensions = [
            '.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.cpp',
            '.go', '.rs', '.php', '.rb', '.kt'
        ]
        
        priority_files = [
            'README.md', 'package.json', 'requirements.txt',
            'pom.xml', 'build.gradle'
        ]
        
        files_read = 0
        
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            # Priority files first
            for pf in priority_files:
                if files_read >= max_files:
                    break
                for file_info in zf.namelist():
                    if file_info.endswith(pf) and not any(sd in file_info for sd in skip_dirs):
                        try:
                            content = zf.read(file_info).decode('utf-8', errors='ignore')
                            files_content[file_info] = content[:5000]
                            files_read += 1
                            break
                        except:
                            pass
            
            # Source files
            for file_info in zf.namelist():
                if files_read >= max_files:
                    break
                if file_info.endswith('/') or any(sd in file_info for sd in skip_dirs):
                    continue
                if any(file_info.lower().endswith(ext) for ext in priority_extensions):
                    try:
                        content = zf.read(file_info).decode('utf-8', errors='ignore')
                        files_content[file_info] = content[:3000]
                        files_read += 1
                    except:
                        pass
        
        logger.info(f"Read {files_read} files")
        
    except Exception as e:
        logger.error(f"Error reading files: {e}")
    
    return files_content

def analyze_with_ai(tech_stack: Dict, files_content: Dict, 
                    project_name: str, student_description: str) -> Dict:
    """Analyze with AI"""
    logger.info(f"Analyzing: {project_name}")
    
    files_summary = "\n\n".join([
        f"File: {fn}\n```\n{content[:1000]}\n```"
        for fn, content in list(files_content.items())[:10]
    ])
    
    prompt = f"""You are an expert code reviewer. Analyze this student project.

PROJECT: {project_name}
DESCRIPTION: {student_description}

TECH STACK:
{json.dumps(tech_stack, indent=2)}

CODE SAMPLES:
{files_summary}

Provide detailed analysis in JSON format:
{{
    "code_quality_score": <0-100>,
    "overall_grade": "<A+, A, A-, B+, B, B-, C+, C, C-, D, F>",
    "detailed_analysis": {{
        "code_structure": "<analysis>",
        "code_quality": "<analysis>",
        "functionality": "<analysis>",
        "documentation": "<analysis>",
        "testing": "<analysis>",
        "security": "<analysis>",
        "performance": "<analysis>"
    }},
    "strengths": ["<3-5 specific strengths>"],
    "weaknesses": ["<3-5 areas for improvement>"],
    "recommendations": ["<5-7 actionable recommendations>"]
}}

Be specific and constructive."""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are an expert code reviewer providing constructive feedback."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2500
        )
        
        content = response.choices[0].message.content
        
        # Extract JSON
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        analysis = json.loads(content)
        logger.info(f"Analysis complete: {analysis.get('overall_grade')}")
        return analysis
        
    except Exception as e:
        logger.error(f"AI error: {e}")
        return get_default_analysis()

def get_default_analysis() -> Dict:
    """Default analysis fallback"""
    return {
        "code_quality_score": 75.0,
        "overall_grade": "B",
        "detailed_analysis": {
            "code_structure": "Analysis unavailable. Manual review recommended.",
            "code_quality": "Manual review needed.",
            "functionality": "Manual inspection required.",
            "documentation": "Documentation review needed.",
            "testing": "Test coverage pending review.",
            "security": "Security review recommended.",
            "performance": "Performance analysis needed."
        },
        "strengths": [
            "Project structure appears organized",
            "Multiple technologies integrated"
        ],
        "weaknesses": [
            "Automated analysis unavailable"
        ],
        "recommendations": [
            "Add comprehensive documentation",
            "Implement unit tests",
            "Follow coding best practices",
            "Add error handling",
            "Review security practices"
        ]
    }

@app.get("/")
async def root():
    return {
        "message": "AI Project Analysis API",
        "status": "active",
        "version": "2.0.0",
        "storage": "Supabase",
        "endpoints": {
            "health": "/health",
            "analyze": "/analyze (POST)",
            "cache/clear": "/cache/clear (GET)",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    health = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "openai_configured": bool(api_key),
        "supabase_configured": bool(SUPABASE_URL and SUPABASE_KEY),
        "environment": env,
        "model": MODEL
    }
    
    # Test Supabase connection
    try:
        test_url = f"{SUPABASE_URL}/storage/v1/bucket"
        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}"
        }
        response = requests.get(test_url, headers=headers, timeout=5)
        health["supabase_connection"] = "ok" if response.status_code == 200 else f"error: {response.status_code}"
    except Exception as e:
        health["supabase_connection"] = f"error: {str(e)}"
        health["status"] = "degraded"
    
    return health

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_project(request: AnalysisRequest):
    logger.info(f"Analysis request: {request.project_name}")
    logger.info(f"Path: {request.project_path}")
    
    try:
        # Check cache
        cache_key = get_cache_key(request.project_path)
        if cache_key and cache_key in analysis_cache:
            logger.info("Returning cached result")
            return analysis_cache[cache_key]
        
        # Download from Supabase
        logger.info("Step 1: Downloading from Supabase")
        zip_bytes = download_from_supabase(request.project_path)
        
        # Detect tech stack
        logger.info("Step 2: Detecting tech stack")
        tech_stack = detect_technology_stack(zip_bytes)
        
        # Read files
        logger.info("Step 3: Reading files")
        files_content = read_project_files(zip_bytes)
        
        if not files_content:
            raise HTTPException(status_code=400, detail="No readable files found")
        
        # Analyze with AI
        logger.info("Step 4: AI analysis")
        ai_analysis = analyze_with_ai(
            tech_stack, files_content,
            request.project_name, request.student_description
        )
        
        # Build response
        response = AnalysisResponse(
            project_name=request.project_name,
            student_description=request.student_description,
            detected_tech_stack=tech_stack,
            code_quality_score=ai_analysis.get("code_quality_score", 0),
            overall_grade=ai_analysis.get("overall_grade", "N/A"),
            detailed_analysis=ai_analysis.get("detailed_analysis", {}),
            recommendations=ai_analysis.get("recommendations", []),
            strengths=ai_analysis.get("strengths", []),
            weaknesses=ai_analysis.get("weaknesses", []),
            analysis_timestamp=datetime.now().isoformat()
        )
        
        # Cache result
        if cache_key:
            analysis_cache[cache_key] = response
        
        logger.info("Analysis completed successfully")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/cache/clear")
async def clear_cache():
    size = len(analysis_cache)
    analysis_cache.clear()
    return {"message": f"Cache cleared ({size} entries)"}

@app.get("/cache/stats")
async def cache_stats():
    return {
        "cached_analyses": len(analysis_cache),
        "cache_keys": list(analysis_cache.keys())
    }

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)

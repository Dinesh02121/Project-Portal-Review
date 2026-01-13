from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import os
import json
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
import io
from supabase import create_client, Client
import zipfile

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Project Analysis API",
    description="Comprehensive code review and assessment using OpenAI",
    version="2.0.0"
)

# CORS configuration
allowed_origins_str = os.getenv("ALLOWED_ORIGINS", "")

if not allowed_origins_str:
    logger.warning("ALLOWED_ORIGINS not set, using localhost defaults")
    allowed_origins = ["http://localhost:3000", "http://localhost:8080"]
else:
    allowed_origins = [origin.strip() for origin in allowed_origins_str.split(",") if origin.strip()]

# Only add wildcard in development (NOT in production)
env = os.getenv("ENV", "development")
if env == "development":
    allowed_origins.append("*")

logger.info(f"Environment: {env}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Supabase client
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "project-files")

if not SUPABASE_URL or not SUPABASE_KEY:
    logger.error("Supabase credentials not found in environment variables")
    raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in environment variables")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
logger.info(f"Supabase client initialized for bucket: {SUPABASE_BUCKET}")

# Initialize OpenAI client
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    logger.error("OPENAI_API_KEY not found in environment variables")
    raise ValueError("OPENAI_API_KEY must be set in environment variables")

# Detect if using Groq or OpenAI
if api_key.startswith('gsk_'):
    logger.info("Using Groq API")
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1"
    )
    MODEL = "llama-3.3-70b-versatile"
else:
    logger.info("Using OpenAI API")
    client = OpenAI(api_key=api_key)
    MODEL = "gpt-4o-mini"

# Simple in-memory cache
analysis_cache = {}

class AnalysisRequest(BaseModel):
    project_path: str  # This will be the Supabase storage path (e.g., "projects/123/project.zip")
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

def get_cache_key(project_path: str) -> str:
    """Generate a cache key based on project path"""
    try:
        cache_str = f"{project_path}:{datetime.now().date()}"
        return hashlib.md5(cache_str.encode()).hexdigest()
    except Exception as e:
        logger.warning(f"Could not generate cache key: {e}")
        return None

def download_from_supabase(storage_path: str) -> bytes:
    """Download ZIP file from Supabase storage"""
    try:
        logger.info(f"Downloading from Supabase: {storage_path}")
        
        # Remove leading slash if present
        clean_path = storage_path.lstrip('/')
        
        # Download file from Supabase storage
        response = supabase.storage.from_(SUPABASE_BUCKET).download(clean_path)
        
        if response:
            logger.info(f"Successfully downloaded {len(response)} bytes from Supabase")
            return response
        else:
            raise Exception("Empty response from Supabase")
            
    except Exception as e:
        logger.error(f"Error downloading from Supabase: {e}")
        raise HTTPException(
            status_code=404,
            detail=f"Failed to download project from Supabase: {str(e)}"
        )

def detect_technology_stack(zip_bytes: bytes) -> Dict[str, List[str]]:
    """Detect technologies used in the project by analyzing ZIP contents"""
    tech_stack = {
        "languages": [],
        "frameworks": [],
        "databases": [],
        "tools": []
    }
    
    try:
        logger.info("Detecting tech stack from ZIP file")
        
        # File extension mapping
        language_map = {
            '.py': 'Python',
            '.js': 'JavaScript',
            '.jsx': 'React/JSX',
            '.ts': 'TypeScript',
            '.tsx': 'React/TypeScript',
            '.java': 'Java',
            '.cpp': 'C++',
            '.c': 'C',
            '.cs': 'C#',
            '.go': 'Go',
            '.rs': 'Rust',
            '.php': 'PHP',
            '.rb': 'Ruby',
            '.swift': 'Swift',
            '.kt': 'Kotlin',
            '.html': 'HTML',
            '.css': 'CSS',
            '.scss': 'SCSS',
            '.sql': 'SQL'
        }
        
        # Framework detection patterns
        framework_patterns = {
            'package.json': ['react', 'vue', 'angular', 'express', 'next', 'gatsby', 'svelte'],
            'requirements.txt': ['django', 'flask', 'fastapi', 'pandas', 'numpy', 'tensorflow', 'pytorch'],
            'pom.xml': ['spring', 'hibernate', 'maven'],
            'build.gradle': ['spring', 'android', 'gradle'],
            'Gemfile': ['rails', 'sinatra'],
            'composer.json': ['laravel', 'symfony', 'wordpress'],
            'go.mod': ['gin', 'echo', 'fiber'],
            'Cargo.toml': ['actix', 'rocket', 'tokio']
        }
        
        languages_found = set()
        frameworks_found = set()
        databases = set()
        tools = []
        
        # Database keywords
        db_keywords = {
            'mysql': 'MySQL',
            'postgresql': 'PostgreSQL',
            'postgres': 'PostgreSQL',
            'mongodb': 'MongoDB',
            'sqlite': 'SQLite',
            'redis': 'Redis',
            'cassandra': 'Cassandra',
            'oracle': 'Oracle',
            'mariadb': 'MariaDB',
            'dynamodb': 'DynamoDB',
            'firebase': 'Firebase',
            'supabase': 'Supabase'
        }
        
        # Tool files
        tool_files = {
            'Dockerfile': 'Docker',
            'docker-compose.yml': 'Docker Compose',
            '.gitignore': 'Git',
            'package.json': 'npm',
            'yarn.lock': 'Yarn',
            'Pipfile': 'Pipenv',
            'poetry.lock': 'Poetry',
            '.travis.yml': 'Travis CI',
            '.gitlab-ci.yml': 'GitLab CI',
            'Jenkinsfile': 'Jenkins'
        }
        
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            for file_info in zf.filelist:
                filename = file_info.filename
                
                # Skip macOS metadata and directories
                if '__MACOSX' in filename or filename.endswith('/'):
                    continue
                
                # Get file extension
                ext = os.path.splitext(filename)[1].lower()
                if ext in language_map:
                    languages_found.add(language_map[ext])
                
                # Check for framework files
                base_name = os.path.basename(filename)
                if base_name in framework_patterns:
                    try:
                        content = zf.read(file_info).decode('utf-8', errors='ignore').lower()
                        for framework in framework_patterns[base_name]:
                            if framework in content:
                                frameworks_found.add(framework.capitalize())
                    except Exception as e:
                        logger.warning(f"Could not read {base_name}: {e}")
                
                # Check for tool files
                if base_name in tool_files:
                    tools.append(tool_files[base_name])
                
                # Check for database usage in code files
                if ext in ['.py', '.js', '.java', '.properties', '.yml', '.yaml', '.env']:
                    try:
                        content = zf.read(file_info).decode('utf-8', errors='ignore').lower()
                        for keyword, db_name in db_keywords.items():
                            if keyword in content:
                                databases.add(db_name)
                    except Exception:
                        continue
        
        tech_stack["languages"] = sorted(list(languages_found))
        tech_stack["frameworks"] = sorted(list(frameworks_found))
        tech_stack["databases"] = sorted(list(databases))
        tech_stack["tools"] = sorted(list(set(tools)))
        
        logger.info(f"Detected tech stack: {tech_stack}")
        
    except Exception as e:
        logger.error(f"Error detecting tech stack: {e}")
    
    return tech_stack

def read_project_files_from_zip(zip_bytes: bytes, max_files: int = 20) -> Dict[str, str]:
    """Read important project files from ZIP for analysis"""
    files_content = {}
    
    try:
        logger.info("Reading project files from ZIP")
        
        # Skip directories
        skip_patterns = [
            'node_modules/', 'venv/', '__pycache__/', 'build/', 'dist/', 
            '.git/', 'target/', 'bin/', 'obj/', '.next/', '.nuxt/',
            '__MACOSX/'
        ]
        
        # Priority files to read
        priority_extensions = [
            '.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.cpp', '.c', 
            '.go', '.rs', '.php', '.rb', '.swift', '.kt'
        ]
        priority_files = [
            'README.md', 'README.txt', 'package.json', 'requirements.txt', 
            'pom.xml', 'build.gradle', 'Cargo.toml', 'go.mod'
        ]
        
        files_read = 0
        
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            file_list = zf.filelist
            
            # Read priority files first
            for file_info in file_list:
                if files_read >= max_files:
                    break
                    
                filename = file_info.filename
                base_name = os.path.basename(filename)
                
                # Skip unwanted patterns
                if any(pattern in filename for pattern in skip_patterns):
                    continue
                
                if base_name in priority_files:
                    try:
                        content = zf.read(file_info).decode('utf-8', errors='ignore')
                        files_content[filename] = content[:5000]
                        files_read += 1
                        logger.debug(f"Read priority file: {base_name}")
                    except Exception as e:
                        logger.warning(f"Could not read {filename}: {e}")
            
            # Read source code files
            for file_info in file_list:
                if files_read >= max_files:
                    break
                
                filename = file_info.filename
                ext = os.path.splitext(filename)[1].lower()
                
                # Skip unwanted patterns
                if any(pattern in filename for pattern in skip_patterns):
                    continue
                
                if ext in priority_extensions and filename not in files_content:
                    try:
                        content = zf.read(file_info).decode('utf-8', errors='ignore')
                        files_content[filename] = content[:3000]
                        files_read += 1
                        logger.debug(f"Read source file: {os.path.basename(filename)}")
                    except Exception as e:
                        logger.warning(f"Could not read {filename}: {e}")
        
        logger.info(f"Read {files_read} files from ZIP")
    
    except Exception as e:
        logger.error(f"Error reading project files: {e}")
    
    return files_content

def analyze_with_openai(tech_stack: Dict, files_content: Dict, project_name: str, student_description: str) -> Dict:
    """Use OpenAI to analyze the project and provide detailed feedback"""
    
    logger.info(f"Starting AI analysis for project: {project_name}")
    
    # Prepare code samples
    files_summary = "\n\n".join([
        f"File: {filename}\n```\n{content[:1000]}\n```" 
        for filename, content in list(files_content.items())[:10]
    ])
    
    prompt = f"""You are an expert code reviewer and educator. Analyze this student project comprehensively and provide constructive feedback.

PROJECT DETAILS:
Name: {project_name}
Student Description: {student_description}

DETECTED TECHNOLOGIES:
{json.dumps(tech_stack, indent=2)}

CODE SAMPLES:
{files_summary}

Please provide a detailed analysis in JSON format with the following structure:
{{
    "code_quality_score": <float between 0-100>,
    "overall_grade": "<A+, A, A-, B+, B, B-, C+, C, C-, D, F>",
    "detailed_analysis": {{
        "code_structure": "<detailed analysis of project architecture and organization>",
        "code_quality": "<analysis of code quality, readability, and adherence to best practices>",
        "functionality": "<analysis of features, completeness, and functionality>",
        "documentation": "<analysis of code comments, README, and documentation>",
        "testing": "<analysis of test coverage and quality>",
        "security": "<analysis of security considerations and vulnerabilities>",
        "performance": "<analysis of performance and efficiency>"
    }},
    "strengths": [
        "<3-5 specific strengths with examples>"
    ],
    "weaknesses": [
        "<3-5 specific areas for improvement with explanations>"
    ],
    "recommendations": [
        "<5-7 specific, actionable recommendations for improvement>"
    ]
}}

Be specific, constructive, and educational. Provide concrete examples and actionable advice."""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system", 
                    "content": "You are an expert code reviewer and educator who provides detailed, constructive feedback on student projects. Focus on helping students learn and improve."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            temperature=0.7,
            max_tokens=2500
        )
        
        content = response.choices[0].message.content
        logger.info("Received response from AI")
        
        # Extract JSON from response
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        analysis = json.loads(content)
        logger.info(f"Analysis completed with grade: {analysis.get('overall_grade')}")
        return analysis
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {e}")
        return get_default_analysis()
    except Exception as e:
        logger.error(f"AI API error: {e}")
        return get_default_analysis()

def get_default_analysis() -> Dict:
    """Return default analysis when AI analysis fails"""
    return {
        "code_quality_score": 75.0,
        "overall_grade": "B",
        "detailed_analysis": {
            "code_structure": "Automated analysis temporarily unavailable. Manual review recommended.",
            "code_quality": "Please review code manually for quality assessment.",
            "functionality": "Functionality assessment requires manual inspection.",
            "documentation": "Documentation review needed.",
            "testing": "Test coverage assessment pending manual review.",
            "security": "Security review recommended.",
            "performance": "Performance analysis requires manual testing."
        },
        "strengths": [
            "Project structure appears organized",
            "Multiple technologies integrated",
            "Clear project purpose from description"
        ],
        "weaknesses": [
            "Automated analysis unavailable",
            "Manual detailed review recommended"
        ],
        "recommendations": [
            "Add comprehensive documentation and README",
            "Implement unit and integration tests",
            "Follow coding best practices and style guides",
            "Add proper error handling throughout",
            "Review security best practices",
            "Optimize performance-critical sections",
            "Add code comments for complex logic"
        ]
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "AI Project Analysis API",
        "status": "active",
        "version": "2.0.0",
        "storage": "Supabase",
        "endpoints": {
            "health": "/health",
            "analyze": "/analyze (POST)",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Enhanced health check endpoint"""
    
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "openai_configured": bool(os.getenv("OPENAI_API_KEY")),
        "supabase_configured": bool(SUPABASE_URL and SUPABASE_KEY),
        "supabase_bucket": SUPABASE_BUCKET,
        "environment": os.getenv("ENV", "development"),
        "model": MODEL,
        "allowed_origins": os.getenv("ALLOWED_ORIGINS", "not set"),
    }
    
    # Test Supabase connection
    try:
        # Try to list buckets to verify connection
        buckets = supabase.storage.list_buckets()
        health_status["supabase_connection"] = "ok"
        health_status["available_buckets"] = len(buckets)
    except Exception as e:
        health_status["supabase_connection"] = "error"
        health_status["supabase_error"] = str(e)
    
    return health_status

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_project(request: AnalysisRequest):
    """Main endpoint to analyze a student project from Supabase storage"""
    
    logger.info(f"Received analysis request for project: {request.project_name}")
    logger.info(f"Supabase path: {request.project_path}")
    
    try:
        # Check cache
        cache_key = get_cache_key(request.project_path)
        if cache_key and cache_key in analysis_cache:
            logger.info(f"Returning cached analysis for: {request.project_name}")
            return analysis_cache[cache_key]
        
        # Step 1: Download ZIP from Supabase
        logger.info("Step 1: Downloading project from Supabase")
        zip_bytes = download_from_supabase(request.project_path)
        
        # Step 2: Detect technology stack
        logger.info("Step 2: Detecting technology stack")
        tech_stack = detect_technology_stack(zip_bytes)
        
        # Step 3: Read project files
        logger.info("Step 3: Reading project files")
        files_content = read_project_files_from_zip(zip_bytes)
        
        if not files_content:
            logger.error("No readable files found in project")
            raise HTTPException(status_code=400, detail="No readable files found in project")
        
        # Step 4: Analyze with OpenAI
        logger.info("Step 4: Analyzing with AI")
        ai_analysis = analyze_with_openai(
            tech_stack=tech_stack,
            files_content=files_content,
            project_name=request.project_name,
            student_description=request.student_description
        )
        
        # Construct response
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
        
        # Cache the result
        if cache_key:
            analysis_cache[cache_key] = response
            logger.info(f"Cached analysis result with key: {cache_key}")
        
        logger.info(f"Analysis completed successfully for: {request.project_name}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/cache/clear")
async def clear_cache():
    """Clear the analysis cache"""
    cache_size = len(analysis_cache)
    analysis_cache.clear()
    logger.info(f"Cache cleared. Removed {cache_size} entries")
    return {"message": f"Cache cleared. Removed {cache_size} entries"}

@app.get("/cache/stats")
async def cache_stats():
    """Get cache statistics"""
    return {
        "cached_analyses": len(analysis_cache),
        "cache_keys": list(analysis_cache.keys())
    }

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    
    logger.info(f"Starting AI Analysis API on {host}:{port}")
    logger.info(f"Using Supabase bucket: {SUPABASE_BUCKET}")
    uvicorn.run(app, host=host, port=port)

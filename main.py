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
import httpx

# Load environment variables
load_dotenv()

# Configure logging for production
env = os.getenv("ENV", "development")
log_handlers = [logging.StreamHandler()]

# Only create file handler in development
if env == "development":
    try:
        log_handlers.append(logging.FileHandler('analysis.log'))
    except:
        pass

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=log_handlers
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Project Analysis API",
    description="Comprehensive code review and assessment using OpenAI",
    version="1.0.0"
)

# CORS configuration
allowed_origins_str = os.getenv("ALLOWED_ORIGINS", "")

if not allowed_origins_str:
    logger.warning("ALLOWED_ORIGINS not set")
    if env == "development":
        allowed_origins = ["http://localhost:3000", "http://localhost:8080"]
    else:
        # In production, must specify origins
        logger.error("ALLOWED_ORIGINS must be set in production")
        allowed_origins = []
else:
    allowed_origins = [origin.strip() for origin in allowed_origins_str.split(",") if origin.strip()]

logger.info(f"Environment: {env}")
logger.info(f"Allowed origins: {allowed_origins}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple Supabase Storage client using httpx (no Rust dependencies)
class SupabaseStorage:
    """Lightweight Supabase Storage client"""
    def __init__(self, url: str, key: str):
        self.url = url
        self.key = key
        self.headers = {
            "apikey": key,
            "Authorization": f"Bearer {key}",
        }
        self.timeout = httpx.Timeout(30.0, connect=10.0)
    
    def download(self, bucket: str, path: str) -> bytes:
        """Download a file from Supabase storage"""
        url = f"{self.url}/storage/v1/object/{bucket}/{path}"
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(url, headers=self.headers)
                response.raise_for_status()
                return response.content
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP {e.response.status_code} error: {e}")
            raise
        except Exception as e:
            logger.error(f"Download error: {e}")
            raise
    
    def list_buckets(self):
        """List all buckets (for health check)"""
        url = f"{self.url}/storage/v1/bucket"
        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.get(url, headers=self.headers)
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"List buckets error: {e}")
            raise

# Initialize Supabase Storage client
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    logger.error("Supabase credentials not found in environment variables")
    raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set")

supabase_storage = SupabaseStorage(SUPABASE_URL, SUPABASE_KEY)
logger.info("Supabase storage client initialized")

# Initialize OpenAI client
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    logger.error("OPENAI_API_KEY not found in environment variables")
    raise ValueError("OPENAI_API_KEY must be set in environment variables")

if api_key.startswith('gsk_'):
    logger.info("Using Groq API")
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1"
    )
else:
    logger.info("Using OpenAI API")
    client = OpenAI(api_key=api_key)

# Simple in-memory cache
analysis_cache = {}

class AnalysisRequest(BaseModel):
    project_path: str  # This will be the Supabase storage path
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
    """Download file from Supabase storage"""
    try:
        logger.info(f"Downloading from Supabase: {storage_path}")
        
        # Remove leading slash if present
        if storage_path.startswith('/'):
            storage_path = storage_path[1:]
        
        # The storage_path should be like "projects/filename.zip"
        bucket_name = "projects"
        file_path = storage_path.replace(f"{bucket_name}/", "")
        
        response = supabase_storage.download(bucket_name, file_path)
        
        logger.info(f"Successfully downloaded {len(response)} bytes from Supabase")
        return response
            
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error downloading from Supabase: {e}")
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"Failed to download project from storage: HTTP {e.response.status_code}"
        )
    except Exception as e:
        logger.error(f"Error downloading from Supabase: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to download project from storage: {str(e)}"
        )

def get_cache_key(project_path: str) -> str:
    """Generate a cache key based on project path"""
    try:
        cache_str = f"{project_path}:{datetime.now().strftime('%Y-%m-%d')}"
        return hashlib.md5(cache_str.encode()).hexdigest()
    except Exception as e:
        logger.warning(f"Could not generate cache key: {e}")
        return None

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
            'firebase': 'Firebase'
        }
        
        tool_files = {
            'Dockerfile': 'Docker',
            'docker-compose.yml': 'Docker Compose',
            '.gitignore': 'Git',
            'package.json': 'npm',
            'yarn.lock': 'Yarn',
            'Pipfile': 'Pipenv',
            'poetry.lock': 'Poetry'
        }
        
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            for file_info in zf.namelist():
                # Skip macOS metadata
                if '__MACOSX' in file_info or '.DS_Store' in file_info:
                    continue
                
                # Detect languages by extension
                for ext, lang in language_map.items():
                    if file_info.lower().endswith(ext):
                        languages_found.add(lang)
                
                # Detect frameworks
                filename = file_info.split('/')[-1]
                if filename in framework_patterns:
                    try:
                        content = zf.read(file_info).decode('utf-8', errors='ignore').lower()
                        for framework in framework_patterns[filename]:
                            if framework in content:
                                frameworks_found.add(framework.capitalize())
                    except:
                        pass
                
                # Detect databases
                if file_info.lower().endswith(('.py', '.js', '.java', '.properties', '.yml', '.yaml', '.env')):
                    try:
                        content = zf.read(file_info).decode('utf-8', errors='ignore').lower()
                        for keyword, db_name in db_keywords.items():
                            if keyword in content:
                                databases.add(db_name)
                    except:
                        pass
                
                # Detect tools
                for tool_file, tool_name in tool_files.items():
                    if file_info.endswith(tool_file):
                        if tool_name not in tools:
                            tools.append(tool_name)
        
        tech_stack["languages"] = sorted(list(languages_found))
        tech_stack["frameworks"] = sorted(list(frameworks_found))
        tech_stack["databases"] = sorted(list(databases))
        tech_stack["tools"] = tools
        
        logger.info(f"Detected tech stack: {tech_stack}")
        
    except Exception as e:
        logger.error(f"Error detecting tech stack: {e}")
    
    return tech_stack

def read_project_files(zip_bytes: bytes, max_files: int = 20) -> Dict[str, str]:
    """Read important project files from ZIP for analysis"""
    files_content = {}
    
    try:
        logger.info("Reading project files from ZIP")
        
        skip_dirs = {
            'node_modules', 'venv', '__pycache__', 'build', 'dist', 
            '.git', 'target', 'bin', 'obj', '.next', '.nuxt', '__MACOSX'
        }
        
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
            # Read priority files first
            for priority_file in priority_files:
                if files_read >= max_files:
                    break
                
                for file_info in zf.namelist():
                    if file_info.endswith(priority_file):
                        # Skip if in unwanted directory
                        if any(skip_dir in file_info for skip_dir in skip_dirs):
                            continue
                        
                        try:
                            content = zf.read(file_info).decode('utf-8', errors='ignore')
                            files_content[file_info] = content[:5000]
                            files_read += 1
                            logger.debug(f"Read priority file: {file_info}")
                        except:
                            pass
                        break
            
            # Read source code files
            for file_info in zf.namelist():
                if files_read >= max_files:
                    break
                
                # Skip directories and unwanted paths
                if file_info.endswith('/'):
                    continue
                if any(skip_dir in file_info for skip_dir in skip_dirs):
                    continue
                if '__MACOSX' in file_info or '.DS_Store' in file_info:
                    continue
                
                # Check if it's a source file
                if any(file_info.lower().endswith(ext) for ext in priority_extensions):
                    try:
                        content = zf.read(file_info).decode('utf-8', errors='ignore')
                        files_content[file_info] = content[:3000]
                        files_read += 1
                        logger.debug(f"Read source file: {file_info}")
                    except:
                        pass
        
        logger.info(f"Read {files_read} files from project")
    
    except Exception as e:
        logger.error(f"Error reading project files: {e}")
    
    return files_content

def analyze_with_openai(tech_stack: Dict, files_content: Dict, project_name: str, student_description: str) -> Dict:
    """Use OpenAI to analyze the project and provide detailed feedback"""
    
    logger.info(f"Starting OpenAI analysis for project: {project_name}")
    
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
            model="llama-3.3-70b-versatile",
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
        logger.info("Received response from OpenAI")
        
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
        logger.error(f"OpenAI API error: {e}")
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
        "message": "AI Project Analysis API with Supabase",
        "status": "active",
        "version": "1.0.0",
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
        "environment": os.getenv("ENV", "development"),
    }
    
    try:
        # Test Supabase connection
        supabase_storage.list_buckets()
        health_status["supabase_connection"] = "connected"
    except Exception as e:
        logger.warning(f"Supabase connection check failed: {e}")
        health_status["supabase_connection"] = f"error: {str(e)}"
        health_status["status"] = "degraded"
    
    return health_status

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_project(request: AnalysisRequest):
    """Main endpoint to analyze a student project from Supabase storage"""
    
    logger.info(f"Received analysis request for project: {request.project_name}")
    logger.info(f"Supabase storage path: {request.project_path}")
    
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
        files_content = read_project_files(zip_bytes)
        
        if not files_content:
            logger.error("No readable files found in project ZIP")
            raise HTTPException(status_code=400, detail="No readable files found in project")
        
        # Step 4: Analyze with OpenAI
        logger.info("Step 4: Analyzing with OpenAI")
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
    logger.info("Using Supabase storage for project files")
    uvicorn.run(app, host=host, port=port)

from fastapi import FastAPI, HTTPException, File, UploadFile, Form
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
from zipfile import ZipFile
from io import BytesIO


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
    version="1.0.0"
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
logger.info(f"Allowed origins: {allowed_origins}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    MODEL_NAME = "llama-3.3-70b-versatile"
else:
    logger.info("Using OpenAI API")
    client = OpenAI(api_key=api_key)
    MODEL_NAME = "gpt-4"

# Simple in-memory cache
analysis_cache = {}

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

def detect_technology_stack_from_bytes(zip_bytes: bytes) -> Dict[str, List[str]]:
    """Detect technologies from ZIP bytes"""
    tech_stack = {
        "languages": [],
        "frameworks": [],
        "databases": [],
        "tools": []
    }
    
    try:
        logger.info("Detecting tech stack from ZIP bytes")
        
        language_map = {
            '.py': 'Python', '.js': 'JavaScript', '.jsx': 'React/JSX',
            '.ts': 'TypeScript', '.tsx': 'React/TypeScript', '.java': 'Java',
            '.cpp': 'C++', '.c': 'C', '.cs': 'C#', '.go': 'Go',
            '.rs': 'Rust', '.php': 'PHP', '.rb': 'Ruby', '.swift': 'Swift',
            '.kt': 'Kotlin', '.html': 'HTML', '.css': 'CSS', '.scss': 'SCSS', '.sql': 'SQL'
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
            'mysql': 'MySQL', 'postgresql': 'PostgreSQL', 'postgres': 'PostgreSQL',
            'mongodb': 'MongoDB', 'sqlite': 'SQLite', 'redis': 'Redis',
            'cassandra': 'Cassandra', 'oracle': 'Oracle', 'mariadb': 'MariaDB',
            'dynamodb': 'DynamoDB', 'firebase': 'Firebase'
        }
        
        tool_files = {
            'Dockerfile': 'Docker',
            'docker-compose.yml': 'Docker Compose',
            '.gitignore': 'Git',
            'package.json': 'npm',
            'yarn.lock': 'Yarn',
            'Pipfile': 'Pipenv',
            'poetry.lock': 'Poetry',
        }
        
        with ZipFile(BytesIO(zip_bytes)) as zf:
            for file_info in zf.filelist:
                filename = file_info.filename
                
                # Skip directories and unwanted files
                if filename.endswith('/') or '__MACOSX' in filename or '.DS_Store' in filename:
                    continue
                
                # Detect language by extension
                ext = Path(filename).suffix.lower()
                if ext in language_map:
                    languages_found.add(language_map[ext])
                
                # Check for framework files
                base_name = Path(filename).name
                if base_name in framework_patterns:
                    try:
                        content = zf.read(file_info).decode('utf-8', errors='ignore').lower()
                        for framework in framework_patterns[base_name]:
                            if framework in content:
                                frameworks_found.add(framework.capitalize())
                    except Exception as e:
                        logger.warning(f"Could not read {base_name}: {e}")
                
                # Detect databases
                if ext in ['.py', '.js', '.java', '.properties', '.yml', '.yaml', '.env']:
                    try:
                        content = zf.read(file_info).decode('utf-8', errors='ignore').lower()
                        for keyword, db_name in db_keywords.items():
                            if keyword in content:
                                databases.add(db_name)
                    except Exception:
                        pass
                
                # Detect tools
                if base_name in tool_files and base_name not in ['package.json']:
                    tools.append(tool_files[base_name])
        
        tech_stack["languages"] = sorted(list(languages_found))
        tech_stack["frameworks"] = sorted(list(frameworks_found))
        tech_stack["databases"] = sorted(list(databases))
        tech_stack["tools"] = sorted(list(set(tools)))
        
        logger.info(f"Detected tech stack: {tech_stack}")
        
    except Exception as e:
        logger.error(f"Error detecting tech stack: {e}")
    
    return tech_stack

def read_project_files_from_bytes(zip_bytes: bytes, max_files: int = 20) -> Dict[str, str]:
    """Read project files from ZIP bytes"""
    files_content = {}
    
    try:
        logger.info("Reading project files from ZIP bytes")
        
        skip_dirs = {
            'node_modules', 'venv', '__pycache__', 'build', 'dist', 
            '.git', 'target', 'bin', 'obj', '.next', '.nuxt', 'vendor'
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
        
        with ZipFile(BytesIO(zip_bytes)) as zf:
            # Read priority files first
            for priority_file in priority_files:
                if files_read >= max_files:
                    break
                
                for file_info in zf.filelist:
                    if file_info.filename.endswith(priority_file):
                        try:
                            content = zf.read(file_info).decode('utf-8', errors='ignore')
                            files_content[file_info.filename] = content[:5000]
                            files_read += 1
                            logger.debug(f"Read priority file: {file_info.filename}")
                            break
                        except Exception as e:
                            logger.warning(f"Could not read {file_info.filename}: {e}")
            
            # Read source code files
            for file_info in zf.filelist:
                if files_read >= max_files:
                    break
                
                filename = file_info.filename
                
                # Skip directories
                if filename.endswith('/'):
                    continue
                
                # Skip unwanted directories and files
                if any(skip_dir in filename for skip_dir in skip_dirs):
                    continue
                
                if '__MACOSX' in filename or '.DS_Store' in filename:
                    continue
                
                ext = Path(filename).suffix
                if ext in priority_extensions:
                    try:
                        content = zf.read(file_info).decode('utf-8', errors='ignore')
                        files_content[filename] = content[:3000]
                        files_read += 1
                        logger.debug(f"Read source file: {filename}")
                    except Exception as e:
                        logger.warning(f"Could not read {filename}: {e}")
        
        logger.info(f"Read {files_read} files from project")
    
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
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system", 
                    "content": "You are an expert code reviewer and educator who provides detailed, constructive feedback on student projects. Focus on helping students learn and improve. Always respond with valid JSON only."
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
        logger.error(f"Response content: {content}")
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
        "version": "1.0.0",
        "environment": env,
        "endpoints": {
            "health": "/health",
            "analyze": "/analyze (POST - multipart/form-data)",
            "cache_clear": "/cache/clear",
            "cache_stats": "/cache/stats",
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
        "environment": os.getenv("ENV", "development"),
        "model": MODEL_NAME,
        "cache_size": len(analysis_cache)
    }
    
    return health_status

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_project(
    project_name: str = Form(...),
    student_description: str = Form(...),
    project_zip: UploadFile = File(...)
):
    """Main endpoint to analyze a student project - receives ZIP file directly"""
    
    logger.info(f"Received analysis request for project: {project_name}")
    logger.info(f"Uploaded file: {project_zip.filename}, content_type: {project_zip.content_type}")
    
    try:
        # Read the uploaded ZIP file into memory
        zip_bytes = await project_zip.read()
        logger.info(f"Received ZIP file: {len(zip_bytes)} bytes")
        
        if len(zip_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty ZIP file received")
        
        # Generate cache key from zip content
        cache_key = hashlib.md5(zip_bytes + project_name.encode()).hexdigest()
        
        # Check cache
        if cache_key in analysis_cache:
            logger.info(f"Returning cached analysis for: {project_name}")
            return analysis_cache[cache_key]
        
        # Step 1: Detect technology stack from ZIP
        logger.info("Step 1: Detecting technology stack")
        tech_stack = detect_technology_stack_from_bytes(zip_bytes)
        
        # Step 2: Read project files from ZIP
        logger.info("Step 2: Reading project files")
        files_content = read_project_files_from_bytes(zip_bytes)
        
        if not files_content:
            logger.error("No readable files found in project")
            raise HTTPException(status_code=400, detail="No readable files found in project ZIP")
        
        # Step 3: Analyze with AI
        logger.info("Step 3: Analyzing with AI")
        ai_analysis = analyze_with_openai(
            tech_stack=tech_stack,
            files_content=files_content,
            project_name=project_name,
            student_description=student_description
        )
        
        # Construct response
        response = AnalysisResponse(
            project_name=project_name,
            student_description=student_description,
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
        analysis_cache[cache_key] = response
        logger.info(f"Cached analysis result with key: {cache_key}")
        
        logger.info(f"Analysis completed successfully for: {project_name}")
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
    logger.info(f"Environment: {env}")
    logger.info(f"Model: {MODEL_NAME}")
    uvicorn.run(app, host=host, port=port)

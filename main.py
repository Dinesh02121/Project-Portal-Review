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

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure base path based on environment
env = os.getenv("ENV", "development")
if env == "production":
    PROJECT_BASE_PATH = "/tmp/uploads/projects/extracted"
    os.makedirs(PROJECT_BASE_PATH, exist_ok=True)
else:
    PROJECT_BASE_PATH = os.getenv("PROJECT_BASE_PATH", os.getcwd())

logger.info(f"Environment: {env}")
logger.info(f"Project base path: {PROJECT_BASE_PATH}")

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
else:
    logger.info("Using OpenAI API")
    client = OpenAI(api_key=api_key)

# Simple in-memory cache
analysis_cache = {}

# ==================== MODELS ====================

class AnalysisRequest(BaseModel):
    project_path: str
    project_name: str
    student_description: str

class AnalysisRequestWithContent(BaseModel):
    project_name: str
    student_description: str
    files_content: Dict[str, str]  # filename -> content mapping

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

# ==================== HELPER FUNCTIONS ====================

def detect_tech_stack_from_files(files_content: Dict[str, str]) -> Dict[str, List[str]]:
    """Detect technologies from file names and contents (for content-based analysis)"""
    tech_stack = {
        "languages": [],
        "frameworks": [],
        "databases": [],
        "tools": []
    }
    
    logger.info(f"Detecting tech stack from {len(files_content)} files")
    
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
    
    languages_found = set()
    frameworks_found = set()
    databases = set()
    tools = set()
    
    for filename, content in files_content.items():
        # Detect language from extension
        for ext, lang in language_map.items():
            if filename.lower().endswith(ext):
                languages_found.add(lang)
        
        # Detect frameworks from filename
        if 'package.json' in filename.lower():
            tools.add('npm')
        if 'pom.xml' in filename.lower():
            tools.add('Maven')
        if 'build.gradle' in filename.lower():
            tools.add('Gradle')
        if 'requirements.txt' in filename.lower():
            tools.add('pip')
        
        # Detect frameworks and databases from content
        content_lower = content.lower()
        
        # Frameworks
        if 'spring' in content_lower or '@springboot' in content_lower:
            frameworks_found.add('Spring Boot')
        if 'react' in content_lower or 'import react' in content_lower:
            frameworks_found.add('React')
        if 'fastapi' in content_lower:
            frameworks_found.add('FastAPI')
        if 'django' in content_lower:
            frameworks_found.add('Django')
        if 'flask' in content_lower:
            frameworks_found.add('Flask')
        if 'express' in content_lower:
            frameworks_found.add('Express')
        if 'angular' in content_lower:
            frameworks_found.add('Angular')
        if 'vue' in content_lower:
            frameworks_found.add('Vue')
        
        # Databases
        db_keywords = {
            'mysql': 'MySQL',
            'postgresql': 'PostgreSQL',
            'postgres': 'PostgreSQL',
            'mongodb': 'MongoDB',
            'sqlite': 'SQLite',
            'redis': 'Redis',
            'h2database': 'H2',
            'oracle': 'Oracle',
            'mariadb': 'MariaDB'
        }
        
        for keyword, db_name in db_keywords.items():
            if keyword in content_lower:
                databases.add(db_name)
    
    tech_stack["languages"] = sorted(list(languages_found))
    tech_stack["frameworks"] = sorted(list(frameworks_found))
    tech_stack["databases"] = sorted(list(databases))
    tech_stack["tools"] = sorted(list(tools))
    
    logger.info(f"Detected tech stack: {tech_stack}")
    return tech_stack

def detect_technology_stack(project_path: Path) -> Dict[str, List[str]]:
    """Detect technologies used in the project by analyzing file extensions and content"""
    tech_stack = {
        "languages": [],
        "frameworks": [],
        "databases": [],
        "tools": []
    }
    
    try:
        logger.info(f"Detecting tech stack for: {project_path}")
        
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
        
        for file_path in project_path.rglob('*'):
            if file_path.is_file():
                ext = file_path.suffix.lower()
                if ext in language_map:
                    languages_found.add(language_map[ext])
                
                if file_path.name in framework_patterns:
                    try:
                        content = file_path.read_text(encoding='utf-8', errors='ignore').lower()
                        for framework in framework_patterns[file_path.name]:
                            if framework in content:
                                frameworks_found.add(framework.capitalize())
                    except Exception as e:
                        logger.warning(f"Could not read {file_path.name}: {e}")
        
        tech_stack["languages"] = sorted(list(languages_found))
        tech_stack["frameworks"] = sorted(list(frameworks_found))
        
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
        databases = set()
        
        for file_path in project_path.rglob('*'):
            if file_path.is_file() and file_path.suffix in ['.py', '.js', '.java', '.properties', '.yml', '.yaml', '.env']:
                try:
                    content = file_path.read_text(encoding='utf-8', errors='ignore').lower()
                    for keyword, db_name in db_keywords.items():
                        if keyword in content:
                            databases.add(db_name)
                except Exception as e:
                    continue
        
        tech_stack["databases"] = sorted(list(databases))
        
        tools = []
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
            'Jenkinsfile': 'Jenkins',
            'kubernetes': 'Kubernetes',
            'terraform': 'Terraform'
        }
        
        for file_name, tool in tool_files.items():
            if list(project_path.rglob(file_name)):
                tools.append(tool)
        
        tech_stack["tools"] = tools
        
        logger.info(f"Detected tech stack: {tech_stack}")
        
    except Exception as e:
        logger.error(f"Error detecting tech stack: {e}")
    
    return tech_stack

def read_project_files(project_path: Path, max_files: int = 20) -> Dict[str, str]:
    """Read important project files for analysis"""
    files_content = {}
    
    try:
        logger.info(f"Reading project files from: {project_path}")
        
        skip_dirs = {
            'node_modules', 'venv', '__pycache__', 'build', 'dist', 
            '.git', 'target', 'bin', 'obj', '.next', '.nuxt'
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
        
        for priority_file in priority_files:
            if files_read >= max_files:
                break
            file_matches = list(project_path.rglob(priority_file))
            for file_path in file_matches[:1]:
                try:
                    content = file_path.read_text(encoding='utf-8', errors='ignore')
                    files_content[str(file_path.relative_to(project_path))] = content[:5000]
                    files_read += 1
                    logger.debug(f"Read priority file: {file_path.name}")
                except Exception as e:
                    logger.warning(f"Could not read {file_path}: {e}")
        
        for file_path in project_path.rglob('*'):
            if files_read >= max_files:
                break
                
            if file_path.is_file() and file_path.suffix in priority_extensions:
                if any(skip_dir in file_path.parts for skip_dir in skip_dirs):
                    continue
                    
                try:
                    content = file_path.read_text(encoding='utf-8', errors='ignore')
                    files_content[str(file_path.relative_to(project_path))] = content[:3000]
                    files_read += 1
                    logger.debug(f"Read source file: {file_path.name}")
                except Exception as e:
                    logger.warning(f"Could not read {file_path}: {e}")
        
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

def resolve_project_path(path_str: str) -> Path:
    """Resolve project path to absolute path"""
    logger.info(f"Original path: {path_str}")
    
    path = Path(path_str)
    
    if os.getenv("ENV") == "production":
        possible_paths = [
            Path("/tmp") / path,
            Path("/tmp/uploads/projects/extracted") / path,
            Path("/tmp") / "uploads" / "projects" / "extracted" / path
        ]
        
        for resolved_path in possible_paths:
            logger.info(f"Checking: {resolved_path}")
            if resolved_path.exists():
                logger.info(f"Found at: {resolved_path}")
                return resolved_path
        
        raise FileNotFoundError(
            f"Project not found. Tried paths: {[str(p) for p in possible_paths]}"
        )
    
    if PROJECT_BASE_PATH:
        resolved_path = Path(PROJECT_BASE_PATH) / path
        logger.info(f"Development - checking: {resolved_path}")
        
        if resolved_path.exists():
            return resolved_path
    
    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        logger.info(f"Found in current directory: {cwd_path}")
        return cwd_path
    
    raise FileNotFoundError(
        f"Project path not found: {path_str}\n"
        f"Environment: {os.getenv('ENV', 'development')}\n"
        f"Base path: {PROJECT_BASE_PATH}"
    )

def get_cache_key(project_path: str) -> str:
    """Generate a cache key based on project path and modification time"""
    try:
        path = Path(project_path)
        if not path.exists():
            return None
        
        latest_mtime = 0
        for file_path in path.rglob('*'):
            if file_path.is_file():
                mtime = file_path.stat().st_mtime
                if mtime > latest_mtime:
                    latest_mtime = mtime
        
        cache_str = f"{project_path}:{latest_mtime}"
        return hashlib.md5(cache_str.encode()).hexdigest()
    except Exception as e:
        logger.warning(f"Could not generate cache key: {e}")
        return None

# ==================== ENDPOINTS ====================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "AI Project Analysis API",
        "status": "active",
        "version": "1.0.0",
        "base_path": PROJECT_BASE_PATH,
        "endpoints": {
            "health": "/health",
            "analyze": "/analyze (POST)",
            "analyze_content": "/analyze-content (POST)",
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
        "base_path": PROJECT_BASE_PATH,
        "base_path_exists": Path(PROJECT_BASE_PATH).exists() if PROJECT_BASE_PATH else False,
        "environment": os.getenv("ENV", "development"),
        "allowed_origins": os.getenv("ALLOWED_ORIGINS", "not set"),
    }
    
    try:
        if Path(PROJECT_BASE_PATH).exists():
            health_status["base_path_readable"] = True
            project_count = len(list(Path(PROJECT_BASE_PATH).iterdir()))
            health_status["projects_found"] = project_count
        else:
            health_status["base_path_readable"] = False
            health_status["warning"] = "Base path does not exist"
    except Exception as e:
        health_status["base_path_readable"] = False
        health_status["error"] = str(e)
    
    return health_status

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_project(request: AnalysisRequest):
    """Main endpoint to analyze a student project (path-based)"""
    
    logger.info(f"Received analysis request for project: {request.project_name}")
    logger.info(f"Original path: {request.project_path}")
    
    try:
        try:
            project_path = resolve_project_path(request.project_path)
        except FileNotFoundError as e:
            logger.error(str(e))
            raise HTTPException(
                status_code=404, 
                detail=f"Project path not found: {request.project_path}. Please check the path configuration."
            )
        
        logger.info(f"Resolved to absolute path: {project_path}")
        
        cache_key = get_cache_key(str(project_path))
        if cache_key and cache_key in analysis_cache:
            logger.info(f"Returning cached analysis for: {request.project_name}")
            return analysis_cache[cache_key]
        
        logger.info("Step 1: Detecting technology stack")
        tech_stack = detect_technology_stack(project_path)
        
        logger.info("Step 2: Reading project files")
        files_content = read_project_files(project_path)
        
        if not files_content:
            logger.error("No readable files found in project")
            raise HTTPException(status_code=400, detail="No readable files found in project")
        
        logger.info("Step 3: Analyzing with OpenAI")
        ai_analysis = analyze_with_openai(
            tech_stack=tech_stack,
            files_content=files_content,
            project_name=request.project_name,
            student_description=request.student_description
        )
        
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

@app.post("/analyze-content")
async def analyze_project_content(request: AnalysisRequestWithContent):
    """Analyze project using provided file contents (for separate server deployment)"""
    
    logger.info(f"Received content-based analysis request for project: {request.project_name}")
    logger.info(f"Files received: {len(request.files_content)}")
    
    try:
        if not request.files_content:
            raise HTTPException(status_code=400, detail="No file contents provided")
        
        # Detect tech stack from file contents
        logger.info("Step 1: Detecting technology stack from file contents")
        tech_stack = detect_tech_stack_from_files(request.files_content)
        
        # Analyze with OpenAI
        logger.info("Step 2: Analyzing with OpenAI")
        ai_analysis = analyze_with_openai(
            tech_stack=tech_stack,
            files_content=request.files_content,
            project_name=request.project_name,
            student_description=request.student_description
        )
        
        # Construct response
        response = {
            "project_name": request.project_name,
            "student_description": request.student_description,
            "detected_tech_stack": tech_stack,
            "code_quality_score": ai_analysis.get("code_quality_score", 0),
            "overall_grade": ai_analysis.get("overall_grade", "N/A"),
            "detailed_analysis": ai_analysis.get("detailed_analysis", {}),
            "recommendations": ai_analysis.get("recommendations", []),
            "strengths": ai_analysis.get("strengths", []),
            "weaknesses": ai_analysis.get("weaknesses", []),
            "analysis_timestamp": datetime.now().isoformat()
        }
        
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
    logger.info(f"Project base path: {PROJECT_BASE_PATH}")
    uvicorn.run(app, host=host, port=port)

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
import asyncio
from contextlib import asynccontextmanager

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

# Simple in-memory cache
analysis_cache = {}

# Environment configuration
env = os.getenv("ENV", "development")

# Lifespan context manager for startup/shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events"""
    # Startup
    logger.info("=" * 60)
    logger.info("AI Project Analysis API Starting...")
    logger.info(f"Environment: {env}")
    logger.info(f"Python Version: {os.sys.version}")
    logger.info(f"Port: {os.getenv('PORT', '8000')}")
    logger.info("=" * 60)
    
    yield
    
    # Shutdown
    logger.info("AI Project Analysis API Shutting Down...")

# Initialize FastAPI app
app = FastAPI(
    title="AI Project Analysis API",
    description="Comprehensive code review and assessment using OpenAI",
    version="1.0.0",
    lifespan=lifespan
)

# CORS configuration
allowed_origins_str = os.getenv("ALLOWED_ORIGINS", "")

if not allowed_origins_str:
    logger.warning("ALLOWED_ORIGINS not set, using localhost defaults")
    allowed_origins = ["http://localhost:3000", "http://localhost:8080"]
else:
    allowed_origins = [origin.strip() for origin in allowed_origins_str.split(",") if origin.strip()]

# Only add wildcard in development
if env == "development":
    allowed_origins.append("*")

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

logger.info(f"Model configured: {MODEL_NAME}")

# Pydantic models
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

# Helper functions
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

def read_project_files_from_bytes(zip_bytes: bytes, max_files: int = 15) -> Dict[str, str]:
    """Read project files from ZIP bytes - optimized for memory and speed"""
    files_content = {}
    
    try:
        logger.info("Reading project files from ZIP bytes")
        
        # Directories to skip
        skip_dirs = {
            'node_modules', 'venv', '__pycache__', 'build', 'dist', 
            '.git', 'target', 'bin', 'obj', '.next', '.nuxt', 'vendor',
            '.vscode', '.idea', 'coverage', '.pytest_cache', '__MACOSX'
        }
        
        # Priority file extensions (source code)
        priority_extensions = [
            '.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.cpp', '.c', 
            '.go', '.rs', '.php', '.rb', '.swift', '.kt', '.cs', '.html'
        ]
        
        # Priority configuration/documentation files
        priority_files = [
            'README.md', 'README.txt', 'README', 'package.json', 
            'requirements.txt', 'pom.xml', 'build.gradle', 'Cargo.toml', 
            'go.mod', 'Dockerfile', 'docker-compose.yml', '.env.example'
        ]
        
        files_read = 0
        
        with ZipFile(BytesIO(zip_bytes)) as zf:
            # First pass: Read priority configuration/documentation files
            for priority_file in priority_files:
                if files_read >= max_files:
                    break
                
                for file_info in zf.filelist:
                    filename = file_info.filename
                    
                    # Skip if already read
                    if filename in files_content:
                        continue
                    
                    # Check if this is a priority file
                    if filename.endswith(priority_file) or filename.endswith('/' + priority_file):
                        # Skip if in excluded directory
                        if any(skip_dir in filename for skip_dir in skip_dirs):
                            continue
                        
                        try:
                            # Read file content
                            content = zf.read(file_info).decode('utf-8', errors='ignore')
                            
                            # Limit content size (2KB for config files)
                            files_content[filename] = content[:2000]
                            files_read += 1
                            logger.debug(f"Read priority file: {filename} ({len(content)} chars)")
                            break
                        except Exception as e:
                            logger.warning(f"Could not read {filename}: {e}")
            
            # Second pass: Read source code files
            for file_info in zf.filelist:
                if files_read >= max_files:
                    break
                
                filename = file_info.filename
                
                # Skip directories
                if filename.endswith('/'):
                    continue
                
                # Skip if already read
                if filename in files_content:
                    continue
                
                # Skip unwanted directories
                if any(skip_dir in filename for skip_dir in skip_dirs):
                    continue
                
                # Skip hidden files and OS files
                if '/.DS_Store' in filename or '/__MACOSX' in filename:
                    continue
                
                # Check file extension
                ext = Path(filename).suffix.lower()
                if ext in priority_extensions:
                    try:
                        # Check file size before reading
                        if file_info.file_size > 100000:  # Skip files > 100KB
                            logger.debug(f"Skipping large file: {filename} ({file_info.file_size} bytes)")
                            continue
                        
                        # Read file content
                        content = zf.read(file_info).decode('utf-8', errors='ignore')
                        
                        # Limit content size (1.5KB for source files)
                        files_content[filename] = content[:1500]
                        files_read += 1
                        logger.debug(f"Read source file: {filename} ({len(content)} chars)")
                    except Exception as e:
                        logger.warning(f"Could not read {filename}: {e}")
        
        logger.info(f"Successfully read {files_read} files from project")
        
        if not files_content:
            logger.error("No files could be read from the ZIP")
    
    except Exception as e:
        logger.error(f"Error reading project files: {e}", exc_info=True)
    
    return files_content

def analyze_with_openai(tech_stack: Dict, files_content: Dict, project_name: str, student_description: str) -> Dict:
    """Use OpenAI/Groq to analyze the project and provide detailed feedback"""
    
    logger.info(f"Starting AI analysis for project: {project_name}")
    
    try:
        # Prepare code samples - limit to 5 most relevant files
        files_summary = "\n\n".join([
            f"File: {filename}\n```\n{content[:500]}\n```" 
            for filename, content in list(files_content.items())[:5]
        ])
        
        # Create analysis prompt
        prompt = f"""You are an expert code reviewer and educator. Analyze this student project and provide constructive feedback.

PROJECT DETAILS:
Name: {project_name}
Student Description: {student_description}

DETECTED TECHNOLOGIES:
{json.dumps(tech_stack, indent=2)}

CODE SAMPLES:
{files_summary}

Provide a detailed analysis in JSON format with this EXACT structure:
{{
    "code_quality_score": <float between 0-100>,
    "overall_grade": "<A+, A, A-, B+, B, B-, C+, C, C-, D, or F>",
    "detailed_analysis": {{
        "code_structure": "<2-3 sentences about project architecture and organization>",
        "code_quality": "<2-3 sentences about code quality, readability, and best practices>",
        "functionality": "<2-3 sentences about features and completeness>",
        "documentation": "<2-3 sentences about documentation quality>",
        "testing": "<1-2 sentences about testing>",
        "security": "<1-2 sentences about security considerations>",
        "performance": "<1-2 sentences about performance>"
    }},
    "strengths": [
        "<specific strength 1>",
        "<specific strength 2>",
        "<specific strength 3>"
    ],
    "weaknesses": [
        "<specific weakness 1>",
        "<specific weakness 2>",
        "<specific weakness 3>"
    ],
    "recommendations": [
        "<actionable recommendation 1>",
        "<actionable recommendation 2>",
        "<actionable recommendation 3>",
        "<actionable recommendation 4>",
        "<actionable recommendation 5>"
    ]
}}

IMPORTANT: 
- Respond with ONLY valid JSON, no markdown code blocks or extra text
- Be specific and constructive
- Focus on learning and improvement
- Provide concrete examples where possible"""

        # Make API call with timeout
        logger.info(f"Calling AI API with model: {MODEL_NAME}")
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system", 
                    "content": "You are an expert code reviewer and educator who provides detailed, constructive feedback on student projects. Always respond with valid JSON only, without markdown code blocks."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            temperature=0.7,
            max_tokens=2000,
            timeout=25  # 25 second timeout
        )
        
        content = response.choices[0].message.content
        logger.info("Received response from AI")
        logger.debug(f"Raw response length: {len(content)} chars")
        
        # Clean up response - remove markdown code blocks if present
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]  # Remove ```json
        if content.startswith("```"):
            content = content[3:]  # Remove ```
        if content.endswith("```"):
            content = content[:-3]  # Remove trailing ```
        content = content.strip()
        
        # Parse JSON
        try:
            analysis = json.loads(content)
            logger.info(f"Analysis completed successfully with grade: {analysis.get('overall_grade')}")
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            logger.error(f"Response content: {content[:500]}")
            return get_default_analysis()
        
        # Validate required fields
        required_fields = ['code_quality_score', 'overall_grade', 'detailed_analysis', 
                          'strengths', 'weaknesses', 'recommendations']
        for field in required_fields:
            if field not in analysis:
                logger.warning(f"Missing required field: {field}")
                return get_default_analysis()
        
        # Validate detailed_analysis sub-fields
        required_analysis_fields = ['code_structure', 'code_quality', 'functionality', 
                                    'documentation', 'testing', 'security', 'performance']
        for field in required_analysis_fields:
            if field not in analysis.get('detailed_analysis', {}):
                logger.warning(f"Missing detailed_analysis field: {field}")
                analysis['detailed_analysis'][field] = "Analysis not available"
        
        return analysis
        
    except Exception as e:
        logger.error(f"AI API error: {type(e).__name__}: {str(e)}", exc_info=True)
        return get_default_analysis()

def get_default_analysis() -> Dict:
    """Return default analysis when AI analysis fails"""
    logger.warning("Returning default analysis due to AI failure")
    return {
        "code_quality_score": 75.0,
        "overall_grade": "B",
        "detailed_analysis": {
            "code_structure": "The project appears to have a basic structure. A detailed review would provide more insights into the architectural decisions and organization patterns used.",
            "code_quality": "Code quality assessment requires manual inspection. Consider reviewing for adherence to language-specific best practices and coding standards.",
            "functionality": "The project demonstrates implementation of core features. Further testing would help evaluate completeness and edge case handling.",
            "documentation": "Documentation should be reviewed for completeness. Consider adding inline comments, README with setup instructions, and API documentation where applicable.",
            "testing": "Testing coverage should be evaluated. Consider implementing unit tests, integration tests, and end-to-end tests as appropriate.",
            "security": "Security considerations should be reviewed manually. Ensure proper input validation, authentication, authorization, and data protection measures.",
            "performance": "Performance optimization opportunities should be identified through profiling and testing under realistic load conditions."
        },
        "strengths": [
            "Project demonstrates integration of multiple technologies",
            "Code structure shows evidence of planning and organization",
            "Project scope aligns with stated objectives"
        ],
        "weaknesses": [
            "Automated analysis was unavailable for detailed assessment",
            "Manual code review recommended for comprehensive feedback",
            "Testing and documentation coverage needs evaluation"
        ],
        "recommendations": [
            "Add comprehensive README with project description, setup instructions, and usage examples",
            "Implement automated tests (unit, integration, and end-to-end)",
            "Add inline code comments for complex logic and architectural decisions",
            "Review and implement proper error handling throughout the codebase",
            "Consider adding logging for debugging and monitoring",
            "Review security best practices relevant to your technology stack",
            "Optimize performance-critical sections after profiling"
        ]
    }

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "AI Project Analysis API",
        "status": "active",
        "version": "1.0.0",
        "environment": env,
        "model": MODEL_NAME,
        "endpoints": {
            "health": "/health",
            "warmup": "/warmup",
            "analyze": "/analyze (POST - multipart/form-data)",
            "cache_clear": "/cache/clear",
            "cache_stats": "/cache/stats",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Lightweight health check for Render - responds quickly"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/warmup")
async def warmup():
    """Warmup endpoint to pre-load the service"""
    logger.info("Warmup request received")
    return {
        "status": "ready",
        "message": "Service is warmed up and ready",
        "timestamp": datetime.now().isoformat(),
        "model": MODEL_NAME,
        "cache_size": len(analysis_cache)
    }

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_project(
    project_name: str = Form(...),
    student_description: str = Form(...),
    project_zip: UploadFile = File(...)
):
    """Main endpoint to analyze a student project - receives ZIP file directly"""
    
    logger.info(f"=== ANALYSIS REQUEST START ===")
    logger.info(f"Project: {project_name}")
    logger.info(f"Description: {student_description[:100]}...")
    logger.info(f"Uploaded file: {project_zip.filename}, content_type: {project_zip.content_type}")
    
    try:
        # Step 0: Read and validate ZIP file
        logger.info("Step 0: Reading uploaded file...")
        zip_bytes = await project_zip.read()
        logger.info(f"✓ Read {len(zip_bytes)} bytes ({len(zip_bytes)/1024:.2f} KB)")
        
        # Validate file size
        MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
        if len(zip_bytes) == 0:
            logger.error("✗ Empty ZIP file received")
            raise HTTPException(status_code=400, detail="Empty ZIP file received")
        
        if len(zip_bytes) > MAX_FILE_SIZE:
            logger.error(f"✗ File too large: {len(zip_bytes)} bytes (max: {MAX_FILE_SIZE})")
            raise HTTPException(
                status_code=413, 
                detail=f"File too large. Maximum size: {MAX_FILE_SIZE/1024/1024:.0f}MB"
            )
        
        # Generate cache key from zip content
        cache_key = hashlib.md5(zip_bytes + project_name.encode()).hexdigest()
        logger.info(f"Cache key: {cache_key}")
        
        # Check cache
        if cache_key in analysis_cache:
            logger.info(f"✓ Returning cached analysis for: {project_name}")
            return analysis_cache[cache_key]
        
        # Step 1: Detect technology stack from ZIP
        logger.info("Step 1: Detecting technology stack...")
        try:
            tech_stack = detect_technology_stack_from_bytes(zip_bytes)
            logger.info(f"✓ Detected tech stack: Languages={len(tech_stack['languages'])}, Frameworks={len(tech_stack['frameworks'])}")
            logger.debug(f"Tech stack details: {tech_stack}")
        except Exception as e:
            logger.error(f"✗ Tech stack detection failed: {e}")
            tech_stack = {"languages": [], "frameworks": [], "databases": [], "tools": []}
        
        # Step 2: Read project files from ZIP
        logger.info("Step 2: Reading project files...")
        try:
            files_content = read_project_files_from_bytes(zip_bytes, max_files=15)
            logger.info(f"✓ Read {len(files_content)} files from project")
            
            if not files_content:
                logger.error("✗ No readable files found in project")
                raise HTTPException(
                    status_code=400, 
                    detail="No readable source files found in project ZIP. Please ensure your ZIP contains code files."
                )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"✗ File reading failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to read project files: {str(e)}")
        
        # Step 3: Analyze with AI
        logger.info("Step 3: Analyzing with AI...")
        try:
            ai_analysis = analyze_with_openai(
                tech_stack=tech_stack,
                files_content=files_content,
                project_name=project_name,
                student_description=student_description
            )
            logger.info(f"✓ AI analysis complete: Grade={ai_analysis.get('overall_grade')}, Score={ai_analysis.get('code_quality_score')}")
        except Exception as e:
            logger.error(f"✗ AI analysis failed: {e}", exc_info=True)
            ai_analysis = get_default_analysis()
            logger.warning("Using default analysis due to AI failure")
        
        # Step 4: Construct response
        logger.info("Step 4: Constructing response...")
        response = AnalysisResponse(
            project_name=project_name,
            student_description=student_description,
            detected_tech_stack=tech_stack,
            code_quality_score=ai_analysis.get("code_quality_score", 75.0),
            overall_grade=ai_analysis.get("overall_grade", "B"),
            detailed_analysis=ai_analysis.get("detailed_analysis", {}),
            recommendations=ai_analysis.get("recommendations", []),
            strengths=ai_analysis.get("strengths", []),
            weaknesses=ai_analysis.get("weaknesses", []),
            analysis_timestamp=datetime.now().isoformat()
        )
        
        # Cache the result
        analysis_cache[cache_key] = response
        logger.info(f"✓ Cached analysis result with key: {cache_key}")
        
        logger.info(f"=== ANALYSIS COMPLETED SUCCESSFULLY for: {project_name} ===")
        return response
        
    except HTTPException as he:
        # Re-raise HTTP exceptions as-is
        logger.error(f"HTTP Exception: {he.status_code} - {he.detail}")
        raise
    except Exception as e:
        # Catch-all for unexpected errors
        logger.error(f"✗ CRITICAL ERROR: {type(e).__name__}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error during analysis: {str(e)}"
        )

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

# Entry point for running locally
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    
    logger.info(f"Starting AI Analysis API on 0.0.0.0:{port}")
    logger.info(f"Environment: {env}")
    logger.info(f"Model: {MODEL_NAME}")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port,
        log_level="info",
        timeout_keep_alive=75
    )

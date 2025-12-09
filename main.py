"""
ClearCut - Background Removal API
Fast, free, no-signup background removal powered by U2Net
"""

import os
import io
import uuid
import time
import asyncio
from pathlib import Path
from typing import Optional
from datetime import datetime, timedelta

from fastapi import FastAPI, File, UploadFile, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from PIL import Image
import rembg

# Configuration
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".heic"}
TEMP_DIR = Path("/tmp/clearcut")
CLEANUP_AFTER_MINUTES = 30

# Create temp directory
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# Initialize FastAPI
app = FastAPI(
    title="ClearCut API",
    description="Free, fast background removal. No signup, no limits.",
    version="1.0.0"
)

# CORS - Allow all origins for now (configure for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store (use Redis in production)
sessions = {}


def cleanup_old_files():
    """Remove files older than CLEANUP_AFTER_MINUTES"""
    now = datetime.now()
    cutoff = now - timedelta(minutes=CLEANUP_AFTER_MINUTES)
    
    for file_path in TEMP_DIR.glob("*"):
        if file_path.is_file():
            file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
            if file_time < cutoff:
                try:
                    file_path.unlink()
                except Exception:
                    pass


def validate_image(file: UploadFile) -> None:
    """Validate uploaded image"""
    # Check extension
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    # Check content type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")


def process_image(
    image_bytes: bytes,
    output_format: str = "png",
    bg_color: Optional[str] = None,
    quality: int = 95
) -> bytes:
    """
    Remove background from image
    
    Args:
        image_bytes: Raw image bytes
        output_format: Output format (png, jpg, webp)
        bg_color: Optional background color (hex, e.g., "#ffffff")
        quality: Output quality for lossy formats
    
    Returns:
        Processed image bytes
    """
    # Remove background using rembg
    output_bytes = rembg.remove(image_bytes)
    
    # Open the result
    img = Image.open(io.BytesIO(output_bytes))
    
    # Apply background color if specified
    if bg_color and bg_color != "transparent":
        # Create background
        bg_color = bg_color.lstrip("#")
        rgb = tuple(int(bg_color[i:i+2], 16) for i in (0, 2, 4))
        
        # Create new image with background
        background = Image.new("RGBA", img.size, (*rgb, 255))
        background.paste(img, mask=img.split()[-1] if img.mode == "RGBA" else None)
        img = background
    
    # Convert to output format
    output_buffer = io.BytesIO()
    
    if output_format.lower() == "jpg" or output_format.lower() == "jpeg":
        # Convert to RGB for JPEG
        if img.mode == "RGBA":
            # Use white background for JPEG if transparent
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[-1])
            img = bg
        elif img.mode != "RGB":
            img = img.convert("RGB")
        img.save(output_buffer, format="JPEG", quality=quality)
    
    elif output_format.lower() == "webp":
        img.save(output_buffer, format="WEBP", quality=quality)
    
    else:  # PNG
        img.save(output_buffer, format="PNG", optimize=True)
    
    output_buffer.seek(0)
    return output_buffer.getvalue()


@app.get("/")
async def root():
    """Health check"""
    return {
        "status": "ok",
        "service": "ClearCut Background Remover",
        "version": "1.0.0"
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.post("/remove-background")
async def remove_background(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    output_format: str = Query("png", regex="^(png|jpg|jpeg|webp)$"),
    bg_color: Optional[str] = Query(None, regex="^(transparent|#[0-9A-Fa-f]{6})$"),
    quality: int = Query(95, ge=1, le=100)
):
    """
    Remove background from uploaded image
    
    - **file**: Image file (jpg, png, webp, heic)
    - **output_format**: Output format (png, jpg, webp)
    - **bg_color**: Background color (transparent or hex like #ffffff)
    - **quality**: Output quality 1-100 (for jpg/webp)
    """
    # Schedule cleanup
    background_tasks.add_task(cleanup_old_files)
    
    # Validate
    validate_image(file)
    
    # Read file
    contents = await file.read()
    
    # Check file size
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)}MB"
        )
    
    try:
        start_time = time.time()
        
        # Process image
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: process_image(contents, output_format, bg_color, quality)
        )
        
        processing_time = time.time() - start_time
        
        # Determine content type
        content_types = {
            "png": "image/png",
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "webp": "image/webp"
        }
        content_type = content_types.get(output_format.lower(), "image/png")
        
        # Generate filename
        original_name = Path(file.filename).stem
        output_filename = f"{original_name}_nobg.{output_format}"
        
        return StreamingResponse(
            io.BytesIO(result),
            media_type=content_type,
            headers={
                "Content-Disposition": f'attachment; filename="{output_filename}"',
                "X-Processing-Time": f"{processing_time:.2f}s",
                "X-Original-Size": str(len(contents)),
                "X-Output-Size": str(len(result))
            }
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.post("/remove-background/preview")
async def remove_background_preview(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Quick preview - returns lower resolution for faster processing
    Used for real-time preview in the UI
    """
    background_tasks.add_task(cleanup_old_files)
    validate_image(file)
    
    contents = await file.read()
    
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large")
    
    try:
        # Resize for faster preview
        img = Image.open(io.BytesIO(contents))
        
        # Calculate preview size (max 800px on longest side)
        max_size = 800
        ratio = min(max_size / img.width, max_size / img.height)
        
        if ratio < 1:
            new_size = (int(img.width * ratio), int(img.height * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        
        # Convert back to bytes
        preview_buffer = io.BytesIO()
        img.save(preview_buffer, format="PNG")
        preview_bytes = preview_buffer.getvalue()
        
        # Process preview
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: rembg.remove(preview_bytes)
        )
        
        return StreamingResponse(
            io.BytesIO(result),
            media_type="image/png"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preview failed: {str(e)}")


@app.post("/batch")
async def batch_remove_background(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
    output_format: str = Query("png", regex="^(png|jpg|jpeg|webp)$"),
    bg_color: Optional[str] = Query(None, regex="^(transparent|#[0-9A-Fa-f]{6})$")
):
    """
    Process multiple images at once
    Returns a session ID to retrieve results
    
    Max 10 images per batch
    """
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 images per batch")
    
    background_tasks.add_task(cleanup_old_files)
    
    session_id = str(uuid.uuid4())
    session_dir = TEMP_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    for file in files:
        try:
            validate_image(file)
            contents = await file.read()
            
            if len(contents) > MAX_FILE_SIZE:
                results.append({
                    "filename": file.filename,
                    "status": "error",
                    "error": "File too large"
                })
                continue
            
            # Process
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda c=contents: process_image(c, output_format, bg_color)
            )
            
            # Save result
            output_name = f"{Path(file.filename).stem}_nobg.{output_format}"
            output_path = session_dir / output_name
            
            with open(output_path, "wb") as f:
                f.write(result)
            
            results.append({
                "filename": file.filename,
                "status": "success",
                "output": output_name,
                "size": len(result)
            })
        
        except Exception as e:
            results.append({
                "filename": file.filename,
                "status": "error",
                "error": str(e)
            })
    
    # Store session info
    sessions[session_id] = {
        "created": datetime.now().isoformat(),
        "results": results,
        "dir": str(session_dir)
    }
    
    return {
        "session_id": session_id,
        "results": results,
        "download_url": f"/batch/{session_id}/download"
    }


@app.get("/batch/{session_id}/download")
async def download_batch(session_id: str):
    """Download all processed images from a batch as a zip file"""
    import zipfile
    
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    
    session = sessions[session_id]
    session_dir = Path(session["dir"])
    
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="Session files expired")
    
    # Create zip
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in session_dir.glob("*"):
            if file_path.is_file():
                zf.write(file_path, file_path.name)
    
    zip_buffer.seek(0)
    
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="clearcut_batch_{session_id[:8]}.zip"'
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

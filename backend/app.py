# backend/app.py
import subprocess
import threading
import pathlib
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.exceptions import HTTPException

# Import services modules
from services import idf_service, geometry_service

@asynccontextmanager
async def lifespan(app: FastAPI):
    frontend_thread = threading.Thread(target=start_frontend)
    frontend_thread.daemon = True
    frontend_thread.start()
    print("Backend Started, Starting Frontend...")
    yield

app = FastAPI(
    title="EnergyPlus API",
    description="Provide IDF file process and geometry data extraction service",
    version="0.1.0",
    lifespan=lifespan
)


def start_frontend():
    frontend_path = pathlib.Path(__file__).parent.parent / "frontend" / "idf-editor"
    print(f"Starting frontend at {frontend_path}")
    subprocess.Popen(["npm", "run", "dev"],shell=True, cwd=frontend_path)


# ---IDF File Upload---
@app.post("/api/idf/upload")
async def upload_idf(file: UploadFile):
    if not file.filename.endswith(".idf"):
        return JSONResponse(
            status_code=400,
            content={"message": "Invalid file format. Please upload a .idf file."}
        )
    
    try:
        idf_id = await idf_service.save_idf_file(file)
        return JSONResponse(
            status_code=200,
            content={"message": "IDF file uploaded successfully", "idf_id": idf_id}
        )
    except HTTPException as e:
        return JSONResponse(
            status_code=e.status_code,
            content={"message": e.detail}
        )
    except Exception as e:
        print(f"Error uploading IDF file: {e}")
        return JSONResponse(
            status_code=500,
            content={"message": "Failed to upload IDF file"}
        )
    
# ---Geometry Data Extraction---
@app.get("/api/geometry/{idf_id}")
async def get_geometry(idf_id: str):
    try:
        geometry_data = await geometry_service.get_geometry_data(idf_id)
        return JSONResponse(
            status_code=200,
            content={"message": "Geometry data extracted successfully", "geometry_data": geometry_data}
        )
    except HTTPException as e:
        return JSONResponse(
            status_code=e.status_code,
            content={"message": e.detail}
        )
    except Exception as e:
        print(f"Error extracting geometry data: {e}")
        return JSONResponse(
            status_code=500,
            content={"message": "Failed to extract geometry data"}
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)




from fastapi import APIRouter, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv
import supabase
import time
import os
import uvicorn

from backend.src.utils.upload_data import Database_Operation
from backend.src.utils.data_process import DataProcess
from backend.src.utils.ep_simulation import run_ep_simulation

load_dotenv()
router = APIRouter()

db_op = Database_Operation()
dp = DataProcess()

OUTPUT_PATH = os.getenv("OUTPUT_PATH")

# ------------------------------------------------------------
# get all filenames from the database
@router.get("/api/idf-files")
def get_all_idf_files():
    """
    Return all filenames from the 'IDF_DATA' table in the database
    """
    try:
        resp = db_op.client.table('IDF_DATA').select('filename').execute()
        filenames = [item['filename'] for item in resp.data]
        return {"filenames": filenames}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

# ------------------------------------------------------------
# acroding filename, get the idf data from the database
@router.get("/api/idf-files/{filename}")
def get_idf_data(filename: str):
    """
    Return the idf corresponding to the filename in the database

    Args:
        filename (str): the filename of the idf data
    """
    try:
        resp = db_op.client.table('IDF_DATA').select('*').eq("filename", filename).execute()
        if not resp.data:
            raise HTTPException(status_code=404, detail="File not found in DB")
        return {
            "filename": filename,
            "objects": resp.data[0]['objects']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ------------------------------------------------------------
# create a new idf record
class IDFCreate(BaseModel):
    filename: str
    objects: List[dict]

@router.post("/api/idf-files")
def create_idf_record(idf_data: IDFCreate):
    """
    Insert a new idf record into the database

    Args:
        idf_data (IDFCreate): the idf data to be inserted into the database
    """
    try:
        resp = db_op.client.table('IDF_DATA').insert({
            "filename": idf_data.filename,
            "objects": idf_data.objects,
        }).execute()
        return {"message": f"Created IDF with filename={idf_data.filename}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# ------------------------------------------------------------
# update an existing idf record
@router.put("/api/idf-files")
def update_idf_file(idf_data: IDFCreate):
    """
    Overwrite the data for the specified filename in the database

    Args:
        idf_data (IDFCreate): the idf data to be updated
    """
    filename = idf_data.filename
    objects = idf_data.objects
    try:
        resp = db_op.client.table("IDF_DATA").update({
            "objects": objects
        }).eq("filename", filename).execute()
        return {"message": f"Updated IDF with filename={filename}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# ------------------------------------------------------------
# run energy plus simulation(Example)
class IDFObject(BaseModel):
    name: str
    note: List[str] = []
    type: str
    units: List[str] = []
    value: Optional[str] = None
    programline: List[str] = []

class IDFData(BaseModel):
    filename: str
    objects: List[IDFObject]

class RunSimulationRequest(BaseModel):
    task_id: str | int
    idf_data: IDFData
    epw_name: str

@router.post("/api/run-simulation")
def run_simulation(request: RunSimulationRequest):
    task_id = str(request.task_id)
    idf_data = request.idf_data
    filename = idf_data.filename
    epw_name = request.epw_name

    result = dp._json2idf(filename, idf_data.model_dump())
    idf_content = result['objects']
    idf_filename = result['filename']

    out_dir = os.path.join(os.getenv("TEMP_PATH"), "output", task_id)
    os.makedirs(out_dir, exist_ok=True)

    idf_path = os.path.join(os.getenv("TEMP_PATH"), f"{idf_filename}.idf")
    with open(idf_path, "w") as f:
        f.write(idf_content)

    epw_path = os.path.join(os.getenv("TEMP_PATH"), f"{epw_name}.epw")
    print(idf_path, epw_path)

    def log_stream():
        for line in run_ep_simulation(idf_path, epw_path, out_dir):
            yield line

    return StreamingResponse(log_stream(), media_type="text/plain")
# ------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(router, host="0.0.0.0", port=8000)

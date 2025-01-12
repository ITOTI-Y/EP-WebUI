from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv
import supabase
import time
import os
import uvicorn

from backend.src.utils.upload_data import Database_Operation
from backend.src.utils.data_process import DataProcess

load_dotenv()
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

db_op = Database_Operation()
dp = DataProcess()

OUTPUT_PATH = os.getenv("OUTPUT_PATH")

# ------------------------------------------------------------
# get all filenames from the database
@app.get("/api/idf-files")
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
@app.get("/api/idf-files/{filename}")
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

@app.post("/api/idf-files")
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
@app.put("/api/idf-files/{filename}")
def update_idf_file(filename: str, objects: List[dict]):
    """
    Overwrite the data for the specified filename in the database

    Args:
        filename (str): the filename of the idf data
        objects (List[dict]): the new idf data
    """
    try:
        resp = db_op.client.table("IDF_DATA").update({
            "data": objects
        }).eq("filename", filename).execute()
        return {"message": f"Updated IDF with filename={filename}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# ------------------------------------------------------------
# run energy plus simulation(Example)
class IDFObject(BaseModel):
    type: str
    value: str = None
    name: str = ""
    note: List[str] = []
    programline: List[str] = []
    units: List[str] = []

@app.post("/api/run-simulation")
def run_simulation(idf_data: IDFObject):
    """
    Frontend will POST a IDFData with JSON format
    example:
    {
        "filename": "MySim",
        "objects": [
            {
                "type": "Version",
                "value": "24.2",
                "name": "",
                "note": [],
                "programline": [],
                "units": []
            },
        ]
    }

    1. Transform the json data into IDF file
    2. Call EnergyPlus to run the simulation
    3. Return the simulation result

    Args:
        idf_data (IDFObject): the idf data to be run
    """
    result = dp._json2idf(idf_data.filename, idf_data.objects)
    idf_content = result['content']
    idf_filename = result['filename']

    # 1. Writes the assembled idf file to a temporary backend file
    tmp_idf_name = idf_data.filename if idf_data.filename.endswith(".idf") else f"{idf_data.filename}"
    outdir = os.path.join(OUTPUT_PATH, f"{idf_data.filename}_run")
    os.makedirs(outdir, exist_ok=True)

    tmp_idf_path = os.path.join(outdir, tmp_idf_name)
    with open(tmp_idf_path, "w", encoding="utf-8") as f:
        f.write(idf_content)

# ------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

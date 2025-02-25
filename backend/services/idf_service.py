# backend/services/idf_service.py

import os
import uuid
import pathlib
from dotenv import load_dotenv
from fastapi import UploadFile, HTTPException
from eppy.modeleditor import IDF
from io import BytesIO

load_dotenv()
temp_dir = pathlib.Path(os.getenv("TEMP_DIR"))

IDD_FILE = pathlib.Path(__file__).parent.parent.parent / "data" / "Energy+.idd"
IDF.setiddname(IDD_FILE)

idf_storage = {}

async def save_idf_file(file: UploadFile):
    if not file.filename.endswith(".idf"):
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload a .idf file.")

    try:
        idf_id = str(uuid.uuid4())
        file_content = await file.read()

        tmp_path = temp_dir / f"{idf_id}.idf"
        with open(tmp_path, "wb") as f:
            f.write(file_content)

        idf_instance = IDF(tmp_path)

        idf_storage[idf_id] = {
            "path": tmp_path,
            "instance": idf_instance,
        }

        return idf_id

    except Exception as e:
        print(f"Error loading IDF file: {e}")
        raise HTTPException(status_code=500, detail="Failed to load IDF file.")

async def get_idf_object(idf_id: str) -> IDF:
    if idf_id not in idf_storage:
        raise HTTPException(status_code=404, detail="IDF object not found.")
    return idf_storage[idf_id]["instance"]

async def get_idf_path(idf_id: str) -> str:
    if idf_id not in idf_storage:
        raise HTTPException(status_code=404, detail="IDF object not found.")
    return idf_storage[idf_id]["path"]

class MockUploadFile:
    def __init__(self, filename, content: bytes):
        self.filename = filename
        self.content = BytesIO(content)

    async def read(self):
        return self.content.read()

if __name__ == "__main__":
    async def main_test():
        test_idf_path = pathlib.Path(__file__).parent.parent.parent / "data" / "test.idf"
        if not test_idf_path.exists():
            raise FileNotFoundError(f"Test IDF file not found at {test_idf_path}")

        with open(test_idf_path, "rb") as f:
            test_content = f.read()
        
        mock_file = MockUploadFile("test.idf", test_content)

        try:
            idf_id = await save_idf_file(mock_file)
            idf_object = await get_idf_object(idf_id)
            idf_path = await get_idf_path(idf_id)
        
        except Exception as e:
            print(f"Error: {e}")
    import asyncio
    asyncio.run(main_test())

import os
import asyncio
from io import BytesIO
from pathlib import Path
from dotenv import load_dotenv

from services.idf_service import *
from services.geometry_service import *
from services.shadow_service import *
from services.pv_service import *

load_dotenv()
test_idf_path = Path(os.getenv("TEST_IDF_PATH"))
test_epw_path = Path(os.getenv("TEST_EPW_PATH"))
idd_file = Path(os.getenv("IDD_FILE"))
temp_dir = Path(os.getenv("TEMP_DIR"))
pathlib.Path(temp_dir).mkdir(parents=True, exist_ok=True)

os.environ["ENERGYPLUS_DIR"] = os.getenv("ENERGY_PLUS_DIR")

class MockUploadFile:
    def __init__(self):
        with open(test_idf_path, "rb") as f:
            text_content = f.read()
        self.filename = test_idf_path.name
        self.content = BytesIO(text_content)

    async def read(self):
        return self.content.read()
    
if __name__ == "__main__":
    async def main_test():
        mock_file = MockUploadFile()
        try:
            idf_id = await save_idf_file(mock_file)
            idf_object = await get_idf_object(idf_id)
            idf_path = await get_idf_path(idf_id)
            shadow_analyzer = ShadowAnalyzer(idf_id, test_epw_path)
            suitable_surfaces = await shadow_analyzer.run()
            pv_analyzer = PVAnalyzer(idf_id, test_epw_path)
            pv_result = await pv_analyzer.run(suitable_surfaces)
        except Exception as e:
            print(f"Error: {e}")
        pass
    asyncio.run(main_test())

from backend.src.utils.upload_data import Database_Operation
from backend.src.routers.idf_routes import *
import uvicorn

if __name__ == "__main__":
    # db_op = Database_Operation()
    # db_op.upload_all()
    uvicorn.run("backend.src.routers.idf_routes:app", host="0.0.0.0", port=8000, reload=True)

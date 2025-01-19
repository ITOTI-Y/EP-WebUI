from fastapi import FastAPI
from backend.src.utils.upload_data import Database_Operation
from backend.src.routers.idf_routes import *
import uvicorn

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)

if __name__ == "__main__":
    # db_op = Database_Operation()
    # db_op.upload_all()
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
    # data = get_idf_data("1Zone")
    # run_simulation("1", data, "weather")

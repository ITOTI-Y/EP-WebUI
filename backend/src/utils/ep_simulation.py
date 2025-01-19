import os
import docker
import time
from typing import Generator
from dotenv import load_dotenv

load_dotenv()

def run_ep_simulation(idf_path: str, epw_path:str, out_dir:str) -> Generator[str, None, None]:
    """
    Run an EnergyPlus simulation and return the output directory

    Args:
        task_id (str): the id of the task
        idf_path (str): the path to the IDF file
        epw_path (str): the path to the EPW file
        out_dir (str): the path to the output directory

    Yields:
        Generator[str, None, None]: the path to the output directory
    """
    client = docker.from_env()

    # Map the path to the container
    container_idf_path = "/input/in.idf"
    container_epw_path = "/input/weather.epw"
    container_out_dir = "/output"

    volumes = {
        os.path.abspath(idf_path): {
            "bind": container_idf_path,
            "mode": "rw"
        },
        os.path.abspath(epw_path): {
            "bind": container_epw_path,
            "mode": "rw"
        },
        os.path.abspath(out_dir): {
            "bind": container_out_dir,
            "mode": "rw"
        }
    }

    command = f"energyplus -w {container_epw_path} -d {container_out_dir} {container_idf_path}"

    try:
        container = client.containers.run(
            image=os.getenv("IMAGE_NAME"),
            command=command,
            detach=True,
            stdout=True,
            stderr=True,
            volumes=volumes,
        )
    except docker.errors.ContainerError as e:
        yield f"[Error] Failed to start container: {str(e)}"
        return
    except docker.errors.ImageNotFound as e:
        yield f"[Error] Docker image not found: {str(e)}"
        return
    except Exception as e:
        yield f"[Error] Other error: {str(e)}"
        return
    
    try:
        for log_line in container.logs(stream=True,follow=True):
            yield log_line.decode("utf-8", errors="ignore")
    finally:
        container.wait()
        container.remove()
        yield f"[Info] Simulation Completed. Output directory: {out_dir}"

if __name__ == "__main__":
    TASK_ID = "test"
    IDF_PATH = os.path.join(os.getenv("TEMP_PATH"), "OfficeMedium.idf")
    EPW_PATH = os.path.join(os.getenv("TEMP_PATH"), "weather.epw")
    OUT_DIR = os.path.join(os.getenv("TEMP_PATH"), "output", TASK_ID)
    os.makedirs(OUT_DIR, exist_ok=True)
    container = run_ep_simulation(IDF_PATH, EPW_PATH, OUT_DIR)
    for line in container:
        print(line)


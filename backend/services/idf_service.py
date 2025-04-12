# backend/services/idf_service.py
"""Service module for handling EnergyPlus IDF file uploads, storage, and retrieval."""

import os
import uuid
import pathlib
from dotenv import load_dotenv
from fastapi import UploadFile, HTTPException
from eppy.modeleditor import IDF
from io import BytesIO

load_dotenv()
temp_dir = pathlib.Path(os.getenv("TEMP_DIR", "./temp")) # Use ./temp as default if not set
temp_dir.mkdir(parents=True, exist_ok=True) # Ensure temp directory exists

# Define the path to the EnergyPlus Data Dictionary file.
IDD_FILE = pathlib.Path(__file__).parent.parent.parent / "data" / "Energy+.idd"
# Set the IDD file path for the eppy library globally.
IDF.setiddname(str(IDD_FILE)) # Convert Path object to string

# In-memory storage for IDF file data (path and parsed object instance).
# Keys are UUIDs generated when files are saved.
idf_storage = {}

async def save_idf_file(file: UploadFile) -> str:
    """Saves an uploaded IDF file, validates it, stores it temporarily,
    parses it using eppy, and keeps track of it in memory.

    Args:
        file: The uploaded IDF file object from FastAPI.

    Returns:
        A unique identifier (UUID string) for the saved IDF file.

    Raises:
        HTTPException: If the file is not a .idf file (400).
        HTTPException: If there's an error reading or parsing the file (500).
    """
    if not file.filename or not file.filename.endswith(".idf"):
        raise HTTPException(
            status_code=400,
            detail="Invalid file format. Please upload a .idf file."
        )

    try:
        idf_id = str(uuid.uuid4())
        file_content = await file.read()

        # Ensure the temp directory exists
        temp_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = temp_dir / f"{idf_id}.idf"

        # Save the uploaded file content to a temporary file
        with open(tmp_path, "wb") as f:
            f.write(file_content)

        # Initialize the IDF object using the temporary file path
        # Convert tmp_path to string as IDF constructor expects str
        idf_instance = IDF(str(tmp_path))

        # Store the path and the parsed IDF instance in memory
        idf_storage[idf_id] = {
            "path": str(tmp_path), # Store path as string
            "instance": idf_instance,
        }

        return idf_id

    except Exception as e:
        # Log the detailed error for debugging purposes
        print(f"Error processing IDF file '{file.filename}': {e}")
        # Raise a generic server error to the client
        raise HTTPException(status_code=500, detail="Failed to process IDF file.")

async def get_idf_object(idf_id: str) -> IDF:
    """Retrieves the parsed eppy IDF object instance using its unique ID.

    Args:
        idf_id: The unique identifier (UUID string) of the IDF file.

    Returns:
        The eppy IDF object instance.

    Raises:
        HTTPException: If no IDF object is found for the given ID (404).
    """
    storage_entry = idf_storage.get(idf_id)
    if not storage_entry:
        raise HTTPException(status_code=404, detail="IDF object not found.")
    return storage_entry["instance"]

async def get_idf_path(idf_id: str) -> str:
    """Retrieves the temporary file path of the stored IDF file using its ID.

    Args:
        idf_id: The unique identifier (UUID string) of the IDF file.

    Returns:
        The absolute path (string) to the temporary IDF file.

    Raises:
        HTTPException: If no IDF object path is found for the given ID (404).
    """
    storage_entry = idf_storage.get(idf_id)
    if not storage_entry:
        raise HTTPException(status_code=404, detail="IDF object path not found.")
    return storage_entry["path"]

class MockUploadFile:
    """A mock object simulating FastAPI's UploadFile for testing purposes."""
    def __init__(self, filename: str, content: bytes):
        """Initializes the mock file with a filename and content.

        Args:
            filename: The simulated name of the uploaded file.
            content: The byte content of the simulated file.
        """
        self.filename = filename
        self._content = BytesIO(content) # Use a private attribute

    async def read(self) -> bytes:
        """Reads the entire content of the mock file."""
        # Reset position in case it was read before, though typically read once.
        self._content.seek(0)
        return self._content.read()

    # Add close method if needed by consumers, though not strictly required by UploadFile interface
    def close(self):
        """Closes the BytesIO stream."""
        self._content.close()

if __name__ == "__main__":
    async def main_test():
        """Runs a simple asynchronous test of the IDF service functions."""
        # Define the path to the test IDF file relative to this script's location
        test_idf_path = pathlib.Path(__file__).parent.parent.parent / "data" / "test.idf"

        if not test_idf_path.exists():
            print(f"Error: Test IDF file not found at {test_idf_path}")
            return # Exit test if file not found

        print(f"Attempting to load test file: {test_idf_path}")
        try:
            with open(test_idf_path, "rb") as f:
                test_content = f.read()

            # Create a mock file object
            mock_file = MockUploadFile(filename="test.idf", content=test_content)

            # Test saving the file
            print("Testing save_idf_file...")
            idf_id = await save_idf_file(mock_file)
            print(f"  Successfully saved. IDF ID: {idf_id}")

            # Test retrieving the IDF object
            print("Testing get_idf_object...")
            idf_object = await get_idf_object(idf_id)
            print(f"  Successfully retrieved IDF object. Type: {type(idf_object)}")
            # Optionally print some info from the object if needed
            # print(f"  IDF Version: {idf_object.idfobjects.get('VERSION', [{}])[0].get('Version_Identifier', 'N/A')}")


            # Test retrieving the IDF path
            print("Testing get_idf_path...")
            idf_path = await get_idf_path(idf_id)
            print(f"  Successfully retrieved IDF path: {idf_path}")
            print(f"  Does path exist? {pathlib.Path(idf_path).exists()}")

            print("All tests passed successfully!")

        except HTTPException as http_exc:
            print(f"HTTP Error during test: Status={http_exc.status_code}, Detail='{http_exc.detail}'")
        except FileNotFoundError as fnf_err:
            print(f"File Not Found Error during test: {fnf_err}")
        except Exception as e:
            # Catch any other unexpected errors during the test
            print(f"An unexpected error occurred during testing: {type(e).__name__}: {e}")
        finally:
            # Clean up: Close the mock file's stream if necessary
            if 'mock_file' in locals() and hasattr(mock_file, 'close'):
                 mock_file.close()
            # Optional: Clean up created temporary files if desired,
            # requires tracking created files or clearing the temp folder based on pattern.
            # For simplicity in this example, cleanup is omitted.
            pass

    import asyncio
    print("Running idf_service main test...")
    asyncio.run(main_test())
    print("idf_service main test finished.")

import os
import pandas as pd
from supabase import create_client, Client
from dotenv import load_dotenv
from pathlib import Path
import logging
import asyncio

# --- Logging Configuration (Global) ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# --- Load Environment Variables (Done before class instantiation) ---
load_dotenv()

class SensitivityDataUploader:
    """
    Handles connecting to Supabase, finding sensitivity CSV files,
    processing them, and uploading the data.
    """

    def __init__(self, url: str | None = None, key: str | None = None, table: str | None = None, root_dir: str | None = None, config: dict | None = None):
        """
        Initializes the uploader with necessary configurations.

        Args:
            url (str): Supabase project URL.
            key (str): Supabase service key (or anon key depending on RLS).
            table (str): Name of the Supabase table to upload data to.
            root_dir (str): The root directory to search for CSV files.
        """
        if config:
            self.url = config['supabase']['url']
            self.key = config['supabase']['key']
            self.table = config['supabase']['table']
            self.root_dir = config['paths']['epsim_dir']
        else:
            self.url = url
            self.key = key
            self.table = table
            self.root_dir = root_dir

        if not self.url or not self.key:
            raise ValueError("Supabase URL and Key must be provided.")
        if not self.table:
            raise ValueError("Supabase table name must be provided.")


        self.supabase: Client | None = None  # Initialize supabase client as None

        logging.info(
            f"Uploader initialized for table '{self.table}' in directory '{self.root_dir}'")

    async def connect(self):
        """Establishes connection to the Supabase database."""
        if self.supabase:
            logging.info("Already connected to Supabase.")
            return True
        try:
            self.supabase = create_client(self.url, self.key)
            # You might want a simple check here, like listing tables (if permissions allow)
            # or just assume connection is okay if no exception is raised.
            # Example check (requires admin rights or specific select grants):
            # await asyncio.to_thread(self.supabase.table(self.table).select('id', count='exact').limit(0).execute)
            logging.info("Successfully connected to Supabase.")
            return True
        except Exception as e:
            logging.error(f"Error connecting to Supabase: {e}")
            self.supabase = None  # Ensure client is None on failure
            return False

    async def extract_metadata(self) -> dict:
        """
        Extracts metadata from the file paths within the specified root directory.

        Returns:
            dict: A dictionary where keys are file paths (str) and values are
                  dictionaries containing extracted metadata ('method', 'city', etc.)
                  and the Path object ('file_path').
        """
        root_path = Path(self.root_dir)
        if not root_path.is_dir():
            logging.error(
                f"Error: Root directory '{self.root_dir}' not found or is not a directory.")
            return {}

        paths_dict = {}
        try:
            # Use the specific filename pattern
            paths = list(root_path.rglob(
                "**/sensitivity_discrete_results.csv"))
            logging.info(
                f"Found {len(paths)} sensitivity_discrete_results.csv files in '{self.root_dir}'.")
        except Exception as e:
            logging.error(f"Error searching for files in {root_path}: {e}")
            return {}

        for path in paths:
            try:
                # Assumes structure: root_dir / method / city_btype_ssp / sensitivity_discrete_results.csv
                city, btype, ssp = path.parent.name.split("_")
                paths_dict[str(path)] = {
                    "city": city,
                    "btype": btype,
                    "ssp_code": ssp,
                    "file_path": path  # Store the Path object for reading
                }
            except (IndexError, ValueError) as e:
                logging.warning(
                    f"Could not extract metadata from path {path}: {e}. Check directory structure. Skipping.")
            except Exception as e:
                logging.error(
                    f"Unexpected error processing path {path}: {e}. Skipping.")
        return paths_dict

    async def _process_and_upload_file(self, file_name: str, file_info: dict) -> int:
        """
        Processes a single CSV file and uploads its data to Supabase.
        Internal helper method.

        Args:
            file_name (str): The string representation of the file path (for logging).
            file_info (dict): Dictionary containing metadata and the 'file_path' Path object.

        Returns:
            int: The number of rows successfully processed and potentially uploaded.
                 Returns 0 if the file is skipped or an error occurs during upload.
        """
        if not self.supabase:
            logging.error("Supabase client not connected. Cannot upload.")
            return 0

        logging.info(f"Processing File: {file_name}")
        file_path = file_info["file_path"]
        try:
            df = pd.read_csv(file_path)
            if df.empty:
                logging.warning(f"File {file_name} is empty. Skipping...")
                return 0

            # Add metadata columns
            for key, value in file_info.items():
                if key != "file_path":  # Don't add the Path object itself as a column
                    df[key] = value.upper()

            # Convert specific columns to nullable integer type
            int_columns = ['insu', 'cool_air_temp', 'lighting']
            for col in int_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(
                        df[col], errors='coerce').astype(pd.Int64Dtype())
                else:
                    logging.warning(
                        f"Column '{col}' not found in {file_name}. Skipping conversion for this column.")

            # Prepare data and upsert
            data_to_insert = df.to_dict(orient='records')
            # Define conflict columns dynamically based on DataFrame columns excluding 'eui'
            conflict_columns = [col for col in df.columns if col != 'eui']
            if not conflict_columns:
                logging.error(
                    f"No conflict columns identified for upsert in file {file_name} (excluding 'eui'). Check columns. Skipping upload.")
                return 0

            logging.debug(
                f"Upserting {len(data_to_insert)} rows from {file_name} with conflict columns: {conflict_columns}")

            # Run synchronous Supabase call in a separate thread
            response = await asyncio.to_thread(
                self.supabase.table(self.table).upsert(
                    data_to_insert,
                    on_conflict=','.join(conflict_columns)
                ).execute
            )

            # Check response (Supabase-py v2+ returns APIResponse)
            if response.data:
                # Assuming response.data contains the upserted rows
                num_rows = len(response.data)
                logging.info(
                    f"Successfully uploaded/updated {num_rows} rows from {file_name} to table '{self.table}'.")
                return num_rows
            # Handle potential errors indicated in the response
            elif hasattr(response, 'error') and response.error:
                logging.error(
                    f"Error uploading data from {file_name} to Supabase: {response.error}")
                return 0
            elif response.status_code >= 400:  # Check HTTP status code for errors
                logging.error(
                    f"Error uploading data from {file_name}. Status: {response.status_code}, Response: {getattr(response, 'json', lambda: {})() or str(response)}")
                return 0
            else:
                # This case might occur if upsert affected 0 rows but wasn't an "error" per se
                logging.warning(
                    f"Upload from {file_name} completed but response indicates no data was returned or modified. Status: {response.status_code}")
                return 0  # Count as 0 rows uploaded in this ambiguous case

        except FileNotFoundError:
            logging.error(
                f"File not found during processing: {file_path}. Skipping.")
            return 0
        except pd.errors.EmptyDataError:
            logging.warning(
                f"File {file_name} is empty or unreadable (pandas EmptyDataError). Skipping.")
            return 0
        except Exception as e:
            logging.error(
                f"Error processing or uploading file {file_name}: {e}")
            return 0

    async def process_and_upload(self):
        """
        Finds CSV files, extracts metadata, and concurrently processes and uploads them.
        """
        if not self.supabase:
            logging.error(
                "Cannot process and upload without a Supabase connection.")
            return

        paths_dict = await self.extract_metadata()
        if not paths_dict:
            logging.warning("No files matching the pattern found to process.")
            return

        logging.info(
            f"Starting concurrent processing for {len(paths_dict)} files...")

        # Create tasks for concurrent execution
        tasks = [
            self._process_and_upload_file(file_name, file_info)
            for file_name, file_info in paths_dict.items()
        ]

        # Run tasks concurrently and gather results
        results = await asyncio.gather(*tasks, return_exceptions=True)

        total_rows_uploaded = 0
        successful_files = 0
        failed_files = 0

        # Process results
        for i, result in enumerate(results):
            # Get corresponding filename
            file_name = list(paths_dict.keys())[i]
            if isinstance(result, Exception):
                logging.error(
                    f"Task for file {file_name} failed with an unhandled exception: {result}")
                failed_files += 1
            elif isinstance(result, int):
                if result > 0:
                    total_rows_uploaded += result
                    successful_files += 1
                else:  # result is 0, indicating skipped file or upload error handled within the task
                    # The specific error/warning was already logged inside _process_and_upload_file
                    failed_files += 1
            else:
                logging.error(
                    f"Task for file {file_name} returned an unexpected result type: {type(result)}")
                failed_files += 1

        logging.info(f"Finished processing all files.")
        logging.info(
            f"Successfully processed and uploaded data from {successful_files} files.")
        if failed_files > 0:
            logging.warning(
                f"Failed to process or upload data from {failed_files} files (check logs for details).")
        logging.info(
            f"Total rows uploaded/updated across all files: {total_rows_uploaded}")

    async def run(self):
        """Connects to Supabase and runs the full processing and upload pipeline."""
        if await self.connect():
            await self.process_and_upload()
        else:
            logging.error("Aborting run due to failed Supabase connection.")


# --- Main Execution Block ---
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parents[2]))
    from config import CONFIG

    SUPABASE_URL = CONFIG['supabase']['url']
    SUPABASE_KEY = CONFIG['supabase']['key']
    SUPABASE_TABLE = CONFIG['supabase']['table']
    ROOT_DIR = CONFIG['paths']['epsim_dir']

    # 1. Perform Environment Variable Checks Early
    if not SUPABASE_URL or not SUPABASE_KEY:
        logging.error(
            "Error: SUPABASE_URL or SUPABASE_KEY not set in environment variables or .env file. Exiting.")
        exit(1)
    if not SUPABASE_TABLE:
        logging.error(
            "Error: SUPABASE_TABLE not set in environment variables or .env file. Exiting.")
        exit(1)

    # 2. Instantiate the Uploader
    try:
        uploader = SensitivityDataUploader(
            url=SUPABASE_URL,
            key=SUPABASE_KEY,
            table=SUPABASE_TABLE,
            root_dir=ROOT_DIR,
        )
    except ValueError as e:
        logging.error(f"Error initializing uploader: {e}")
        exit(1)

    # 3. Run the Uploader's Main Process
    asyncio.run(uploader.run())

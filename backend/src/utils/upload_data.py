import os
import json
from supabase import Client, create_client
from dotenv import load_dotenv
from .data_process import DataProcess


load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
DATA_PATH = os.getenv("DATA_PATH")


class Database_Operation:
    def __init__(self):
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise ValueError("Please set the SUPABASE_URL and SUPABASE_KEY in the .env file")
        self.client : Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        self.dp = DataProcess()

    def _upload_json_files(self):
        for json_file in self.dp.json_files:
            try:
                with open(os.path.join(DATA_PATH, json_file), 'r') as file:
                    data = json.load(file)
                response = self.client.table('IDF_DATA').insert({
                    'filename': json_file.strip('.json'),
                    'objects': data['objects']
                }).execute()
                print(f"Successfully uploaded {json_file} to Supabase")
            except Exception as e:
                print(f"Error uploading {json_file} to Supabase: {str(e)}")

    def _upload_idf_files(self):
        for idf_file in self.dp.idf_files:
            try:
                data_dict = self.dp._idf2dict(idf_file)
                response = self.client.table('IDF_DATA').insert({
                    'filename': data_dict['filename'].strip('.idf'),
                    'objects': data_dict['objects']
                }).execute()
                print(f"Successfully uploaded {idf_file} to Supabase")
            except Exception as e:
                print(f"Error uploading {idf_file} to Supabase: {str(e)}")

    def upload_all(self):
        self._upload_json_files()
        self._upload_idf_files()

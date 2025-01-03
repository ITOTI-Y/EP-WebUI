import re
from supabase import create_client, Client
import os
import json
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
DATA_PATH = os.getenv("DATA_PATH")
IDD_FILE = os.getenv("IDD_FILE")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Please set the SUPABASE_URL and SUPABASE_KEY in the .env file")

class DataProcess:
    def __init__(self):
        self.supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        self.files = self._get_all_filenames()
        self.data_dict = self._data_extract(self.files[0])

    def _get_all_filenames(self):
        """
        Get all filenames with .idf extension from the data path.
        
        Returns:
            list: A list of .idf filenames in the data path
            
        Raises:
            ValueError: If DATA_PATH environment variable is not set
            FileNotFoundError: If the data path directory does not exist
            PermissionError: If there is no permission to access the data path
        """
        if not DATA_PATH:
            raise ValueError("DATA_PATH is not set")
        try:
            all_files = os.listdir(DATA_PATH)
            files = [f for f in all_files if os.path.isfile(os.path.join(DATA_PATH, f)) and f.endswith(".idf")]
            return files
        except FileNotFoundError:
            raise FileNotFoundError(f"{DATA_PATH} not found")
        except PermissionError:
            raise PermissionError(f"No permission to access the {DATA_PATH}")
        
    def _data_extract(self, filename: str):
        """
        Extract data from the files
        """

        with open(os.path.join(DATA_PATH, filename), "r") as f:
            data = f.read()
        data_dict = {}
        pattern = re.compile(r"(?:^ {2})([^\s,;!]+)(?:,([^;]+);$)?|^ {4}(.*)[,;]\s*!- ([^{}\n]+)(?:{([^{}]+)})?", re.MULTILINE)
        lines = data.split("\n")
        key = None
        for line in lines:
            query = pattern.findall(line)
            if query:
                query = query[0]
                if query[1]:
                    key = query[0]
                    data_dict[key] = query[1]
                elif query[0]:
                    key = query[0]
                    if key not in data_dict:
                        data_dict[key] = {}
                elif query[3] and key:
                    data_dict[key][query[3]] = {}
                    if query[4]:
                        data_dict[key][query[3]]['unit'] = query[4]
                    else:
                        data_dict[key][query[3]]['unit'] = None
                    if query[2]:
                        data_dict[key][query[3]]['value'] = query[2]
                    else:
                        data_dict[key][query[3]]['value'] = None
        return data_dict


        sub_pattern = re.compile(r"(\S+)(?:,|;)$", re.MULTILINE)

        split_data = iter(pattern.split(data))

        # for item in split_data:
        #     if item in data_dict:
        #         content = next(split_data)
        #         sub_keys = set(sub_pattern.findall(content))
        #         for sub_key in sub_keys:
        #             data_dict[item][sub_key] = {}
        #         sub_item_list = [i for i in sub_pattern.split(content) if i.strip()]
        #         for sub_item in sub_item_list:
        #             if not sub_key_pattern.match(sub_item):
        #                 pass
        #         pass


if __name__ == "__main__":
    data_process = DataProcess()
    with open("data_dict.json", "w") as f:
        json.dump(data_process.data_dict, f)


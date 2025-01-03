import re
from supabase import create_client, Client
import os
import json
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
DATA_PATH = os.getenv("DATA_PATH")
OUTPUT_PATH = os.getenv("OUTPUT_PATH")

os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(DATA_PATH, exist_ok=True)

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Please set the SUPABASE_URL and SUPABASE_KEY in the .env file")

class DataProcess:
    def __init__(self):
        self.supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        self.idf_files = self._get_all_filenames(type="idf")
        self.json_files = self._get_all_filenames(type="json")

    def _get_all_filenames(self, type: str = "idf"):
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
            files = [f for f in all_files if os.path.isfile(os.path.join(DATA_PATH, f)) and f.endswith(f".{type}")]
            return files
        except FileNotFoundError:
            raise FileNotFoundError(f"{DATA_PATH} not found")
        except PermissionError:
            raise PermissionError(f"No permission to access the {DATA_PATH}")
        
    def _idf2dict(self, filename: str):
        """
        Extract data from the files
        """

        with open(os.path.join(DATA_PATH, filename), "r") as f:
            data = f.read()
        
        pattern = re.compile(r"(?:^ {2})([^\s,;!]+)(?:,([^;]+);$)?"
                             r"|^ {4}(.*)[,;]\s*!- ([^{}\n]+)(?:{([^{}]+)})?",
                             re.MULTILINE)
        
        data_dict = {'Root': {"type": "Root", "value": filename.strip(".idf")}}
        group = None

        for line in data.split("\n"):
            match_result = pattern.findall(line)
            if not match_result:
                continue

            groups = match_result[0]
            parent_key, parent_val, sub_val, sub_key, sub_unit = groups
            if parent_val:
                sub_group = None
                group = parent_key
                data_dict['Root'][group] = {"type": "Group", "value": parent_val}
            
            elif parent_key:
                sub_group = None
                group = parent_key
                if group not in data_dict['Root']:
                    data_dict['Root'][group] = {"type": "Group", "value": None}
            
            elif sub_key and group:
                if sub_key == "Name":
                    sub_group = sub_val
                    data_dict['Root'][group][sub_group] = {"type": "Sub_Group", "value": None}
                    continue

                if sub_group:
                    data_dict['Root'][group][sub_group][sub_key] = {
                        'type': 'Key',
                        'value': sub_val if sub_val else None,
                        'unit': sub_unit if sub_unit else None,
                    }

                else:
                    data_dict['Root'][group][sub_key] = {
                        'type': 'Key',
                        'value': sub_val if sub_val else None,
                        'unit': sub_unit if sub_unit else None,
                    }

        return data_dict
    
    def _json2dict(self, filename: str):
        with open(os.path.join(DATA_PATH, filename), "r") as f:
            data = json.load(f)
        return data
    
    def _dict2idf(self, data_dict: dict):
        idf_data = ""
        groups = [k for k in data_dict['Root'].keys() if k != "type" and k != "value"]
        for group in groups:
            idf_data += self._dict2str(data_dict['Root'][group], group)
            idf_data += "\n"

        return idf_data

    def _dict2str(self, data:dict, name:str):
        def _get_keys(keys:list):
            data_str = ""
            for i,key in enumerate(keys):
                sign = "," if i != len(keys) - 1 else ";"
                value = data[key]['value']
                unit = ' {' + data[key]['unit'] + '}' if data[key]['unit'] else ""
                data_str += f"    {value}{sign}    !- {key}{unit}\n"
            return data_str
        
        if data['type'] == "Group":
            keys = [k for k in data.keys() if k != "type" and k != "value"]
            value = data['value'] + ";" if data['value'] else ""
            
            if not keys:
                data_str = f"  {name},{value}\n"
                return data_str
            else:
                if self._dict_depth(data) == 2:
                    data_str = f"  {name},{value}\n"
                    keys = [k for k in data.keys() if k != "type" and k != "value"]
                    data_str += _get_keys(keys)
                    return data_str
                else:
                    data_str = ""
                    for key in keys:
                        data_str += f"  {name},{value}\n"
                        data_str += self._dict2str(data[key], key) + "\n"
                    return data_str
        
        elif data['type'] == "Sub_Group":
            keys = [k for k in data.keys() if k != "type" and k != "value"]
            data_str = f"    {name},    !- name\n"
            data_str += _get_keys(keys)
            return data_str
        
    def _dict_depth(self, data:dict):
        max_depth = 0
        if not isinstance(data, dict) or not data:
            return 0
        
        for value in data.values():
            if isinstance(value, dict):
                sub_depth = self._dict_depth(value)
                max_depth = max(max_depth, sub_depth)
        return max_depth + 1



    def to_idf(self):
        if not self.json_files:
            raise ValueError("No json files found")
        data_dict_list = [self._json2dict(f) for f in self.json_files]
        for data_dict in data_dict_list:
            idf_data = self._dict2idf(data_dict)
            with open(f"{OUTPUT_PATH}/{data_dict['Root']['value']}.idf", "w") as f:
                f.write(idf_data)
    
    def to_json(self):
        if not self.idf_files:
            raise ValueError("No idf files found")
        data_dict_list = [self._idf2dict(f) for f in self.idf_files]
        for data_dict in data_dict_list:
            with open(f"{OUTPUT_PATH}/{data_dict['Root']['value']}.json", "w") as f:
                json.dump(data_dict, f)


if __name__ == "__main__":
    data_process = DataProcess()
    data_process.to_idf()

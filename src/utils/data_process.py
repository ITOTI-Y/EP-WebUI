import re
import os
import json
from dotenv import load_dotenv

load_dotenv()

DATA_PATH = os.getenv("DATA_PATH")
OUTPUT_PATH = os.getenv("OUTPUT_PATH")

os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(DATA_PATH, exist_ok=True)


class DataProcess:
    def __init__(self):
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
        with open(os.path.join(DATA_PATH, filename), "r") as f:
            data = f.read()
        
        pattern = re.compile(r"(?:^ {2})([^\s,;!]+)(?:,([^;]+);$)?"
                             r"|^ {4}(.*)[,;]\s*!- ([^{}\n]+)(?:{([^{}]+)})?",
                             re.MULTILINE)
        
        data_dict = {'objects':[], 'filename': filename}

        for line in data.split("\n"):
            if not pattern.match(line):
                continue

            groups = pattern.match(line).groups()
            parent_key, parent_val, sub_val, sub_key, sub_unit = groups

            if parent_val:
                data_dict['objects'].append({
                    'type': parent_key,
                    'value': parent_val,
                    'name': '',
                    'note': [],
                    'programline': [],
                    'units': []
                })
            
            elif parent_key:
                data_dict['objects'].append({
                    'type': parent_key,
                    'value': '',
                    'name': '',
                    'note': [],
                    'programline': [],
                    'units': []
                })
            
            elif sub_key:
                if sub_key == "Name":
                    data_dict['objects'][-1]['name'] = sub_val
                data_dict['objects'][-1]['note'].append(sub_key)
                data_dict['objects'][-1]['programline'].append(sub_val if sub_val else '')
                data_dict['objects'][-1]['units'].append(sub_unit if sub_unit else '')
        
        return data_dict

    def _json2idf(self, filename: str) -> str:
        with open(os.path.join(DATA_PATH, filename), "r") as f:
            data = json.load(f)
        
        idf_data = ""

        for obj in data['objects']:
            sign = ";" if obj['value'] else ""
            idf_data += f"  {obj['type']},{obj['value']}{sign}\n"
            idf_data += self._combine_programline(obj)
            idf_data += "\n"
        
        return {'content': idf_data, 'filename': filename.strip(".json")}

    def _combine_programline(self, obj: dict) -> str:
        idf_data = ""
        if not obj['programline']:
            return idf_data
        
        else:
            programline = obj['programline']
            note = obj['note']
            units = obj['units']

            for i, value in enumerate(programline):
                sign = "," if i != len(programline) - 1 else ";"
                idf_data += f"    {value}{sign}    !- {note[i]}{units[i]}\n"
            return idf_data

    def to_idf(self):
        if not self.json_files:
            raise ValueError("No json files found")
        idf_data_list = [self._json2idf(f) for f in self.json_files]
        for idf_data in idf_data_list:
            with open(f"{OUTPUT_PATH}/{idf_data['filename']}.idf", "w") as f:
                f.write(idf_data['content'])
    
    def to_json(self):
        if not self.idf_files:
            raise ValueError("No idf files found")
        data_dict_list = [self._idf2dict(f) for f in self.idf_files]
        for data_dict in data_dict_list:
            with open(f"{OUTPUT_PATH}/{data_dict['filename']}.json", "w") as f:
                json.dump(data_dict, f)


if __name__ == "__main__":
    data_process = DataProcess()
    data_process.to_idf()
    data_process.to_json()

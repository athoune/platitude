import json
import sys
from tqdm import tqdm
from .cliche.detector import ClicheDetector

cliche = ClicheDetector()

print(sys.argv[1])
with open(sys.argv[1], 'r') as f:
    total = len(list(f.readlines()))
    f.seek(0)
    for line in tqdm(f.readlines(), total=total, desc="Reading JSONL"):
        data = json.loads(line)
        cliche.add_document(data["full_text"])

print(cliche.get_cliches())

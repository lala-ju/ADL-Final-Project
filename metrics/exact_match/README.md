usage: exact_match.py [-h] --pred PRED --ans ANS [--jsonl]

options:
  -h, --help   show this help message and exit
  --pred PRED  Prediction file in {"id": str, "output": str} format.
  --ans ANS    Answers file in {"id": id, "instruction": str, "output":
               List[str]} format
  --jsonl      Use if prediction file is in jsonline format

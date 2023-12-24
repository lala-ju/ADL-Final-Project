import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="process dataset for m2 metric")
    parser.add_argument("--predict", type=str, required=True)
    parser.add_argument("--valid", type=str, required=True)
    parser.add_argument("--outfile", type=str, required=True)
    args = parser.parse_args()
    return args

args = parse_args()
with open(args.predict, 'r') as f:
    result = json.load(f)
    
with open(args.valid, 'r') as f:
    valids = json.load(f)
    
for i, res in enumerate(result):
    res['instruction'] = valids[i]['instruction']
    res['output'] = [res['output']]
    
with open(args.outfile, 'w') as f:
    json.dump(result, f, indent=4, ensure_ascii=False)
    
    
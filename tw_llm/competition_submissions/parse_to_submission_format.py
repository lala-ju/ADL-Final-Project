import argparse
import json
import opencc

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', type=str, required=True)
    parser.add_argument('--test', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    return parser.parse_args()

def main():
    args = parse_args()
    cc = opencc.OpenCC('tw2s')
    # Parse input as jsonline format
    pred = []
    with open(args.pred) as f:
        for line in f:
            pred.append(json.loads(line))
    ans = json.load(open(args.test))
    
    output = {}
    for p, a in zip(pred, ans):
        # assert(p["id"] == a["id"])
        output[a["id"]] = {"error_flag": int(a["instruction"] != p["output"]), "error_type": "IWC", "correction": cc.convert(p["output"])}
        
    with open(args.output, "w") as f:
        json.dump(output, f, ensure_ascii=False, indent=4)
            
if __name__ == '__main__':
    main()
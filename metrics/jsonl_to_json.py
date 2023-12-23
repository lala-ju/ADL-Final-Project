import json
import argparse
import jsonlines

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", type=str, required=True)
    parser.add_argument("--ans", type=str, required=True)
    parser.add_argument("--outfile", type=str, required=True)
    return parser.parse_args()

def main():
    args = parse_args()
    pred = []
    with open(args.infile) as f:
        for line in f:
            pred.append(json.loads(line))

    ans = json.load(open(args.ans))
    for i, (p, a) in enumerate(zip(pred, ans)):
        pred[i]["instruction"] = a["instruction"]

    with open(args.outfile, 'w') as f:
        json.dump(pred, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()

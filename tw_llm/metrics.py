import argparse
import json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", type=str, required=True)
    parser.add_argument("--ans", type=str, required=True)
    return parser.parse_args()

def main():
    args = parse_args()
    pred = json.load(open(args.pred))
    ans = json.load(open(args.ans))
    
    exact_match = 0
    total = 0
    # Positive: Grammatically correct
    # Negative: Grammatically incorrect
    true_positive = 0
    true_negative = 0
    total_positive = 0
    total_negative = 0
    
    for p, a in zip(pred, ans):
        assert(p["id"] == a["id"])
        if a["instruction"] == a["output"]:
            total_positive += 1
            if p["output"] == a["output"]:
                true_positive += 1
                exact_match += 1
        else:
            total_negative += 1
            if p["output"] == a["output"]:
                true_negative += 1
                exact_match += 1
        total += 1
        
    print(f"Exact Match = {exact_match} / {total} = {exact_match / total}")
    print(f"Positive = {true_positive} / {total_positive} = {true_positive / total_positive}")
    print(f"Negative = {true_negative} / {total_negative} = {true_negative / total_negative}")
    
if __name__ == "__main__":
    main()
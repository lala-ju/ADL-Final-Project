# usage: python output_to_parallel.py --in <input file> --out <output file>
# expected input: [{"id": <str: id>, "instruction": <str: source>, "output": <list(str): answers>}]
# output format: id\tsource\tanswer1\tanswer2\t...\n

import argparse
import json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", type=str, required=True)
    parser.add_argument("--outfile", type=str, required=True)
    return parser.parse_args()

def main():
    args = parse_args()
    inputs = json.load(open(args.infile))
    outputs = []

    for obj in inputs:
        s = str(obj['id']) + '\t' + obj['instruction']
        if isinstance(obj["output"], list):
            for ans in obj["output"]:
                s = s + "\t" + ans
            s = s + "\n"
        else:
            s = s + "\t" + obj["output"] + "\n"
        outputs.append(s)

    with open(args.outfile, 'w') as f:
        f.writelines(outputs)

if __name__ == "__main__":
    main()


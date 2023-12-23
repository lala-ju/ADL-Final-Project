from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import argparse
import torch
import json

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--data")
    parser.add_argument("--output")
    parser.add_argument("--cuda")
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    with open(args.data, "r", encoding="utf8") as file:
        all_data = json.load(file)
    
    device = torch.device(f'cuda:{args.cuda}')

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(device)

    progress = tqdm(list(range(len(all_data))))

    result = []
    for data in all_data:
        temp = {"id": data["id"]}
        inputs = tokenizer(data["instruction"], return_tensors="pt").to(device).input_ids
        outputs = model.generate(inputs, max_new_tokens=100, num_beams=4, do_sample=False).to(device)
        temp["output"] = tokenizer.decode(outputs[0], skip_special_tokens=True)
        result.append(temp)
        progress.update(1)

    with open(args.output, 'w', encoding='utf8') as f:
        json.dump(result, f, ensure_ascii=False)
    

if __name__ == "__main__":
    main()

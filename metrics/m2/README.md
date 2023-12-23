# M2 Scorer
reference: https://github.com/HillZhang1999/MuCGEC/tree/main/scorers/ChERRANT

# TL;DR
To get the score: 
```
bash scorer.sh <prediction.json> <answer.json>
```
The format of the json object:
```
{"id": <str or int: id>, "instruction": <str: source>, "output": <list(str): answers>}
```

# Procedure
1. Convert the json files to the parallel format ```<id>\t<source>\t<answer1>\t<answer2>\t...\n``` with
```python output_to_parallel.py --infile <input.json> --outfile <output.para>```
2. Convert the parallel text file to the m2 format with
```python parallel_to_m2.py -f <input.para> -o <output.m2> -g char```
3. Compare the prediction and the answer with
```python compare_m2_for_evaluation.py -hyp <prediction.m2> -ref <answer.m2>```


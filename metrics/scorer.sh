PREDICTION=$1
ANSWER=$2

python output_to_parallel.py --infile $PREDICTION --outfile pred_para
python output_to_parallel.py --infile $ANSWER --outfile ans_para

python parallel_to_m2.py -f pred_para -o prediction.m2 -g char
python parallel_to_m2.py -f ans_para -o answer.m2 -g char

python compare_m2_for_evaluation.py -hyp prediction.m2 -ref answer.m2

rm -f prediction.m2 answer.m2 pred_para ans_para

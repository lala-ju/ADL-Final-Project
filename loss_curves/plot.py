import argparse
import pandas as pd
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--loss', type=str, required=True)
    parser.add_argument('--save_file', type=str, default='loss_curve.pdf')
    parser.add_argument('--title', type=str, default='Loss Curve')
    return parser.parse_args()

def main():
    args = parse_args()
    df = pd.read_csv(args.loss).dropna().to_numpy()
    # Columns: epoch,FCGEC,FCGEC_all,NLPCC,NLPCC_all
    # Plot the loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(df[:, 0], df[:, 1], label='FCGEC')
    plt.plot(df[:, 0], df[:, 2], label='FCGEC_all')
    plt.plot(df[:, 0], df[:, 3], label='NLPCC')
    plt.plot(df[:, 0], df[:, 4], label='NLPCC_all')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(args.title)
    plt.legend()

    plt.savefig(args.save_file)
    
    
    
    
if __name__ == '__main__':
    main() 
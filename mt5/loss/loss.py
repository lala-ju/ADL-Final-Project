import csv

epoch = ['epoch']
FCGEC = ['FCGEC']
FCGEC_all = ['FCGEC_all']
NLPCC = ['NLPCC']
NLPCC_all = ['NLPCC_all']

with open('FCGEC_10000.csv', 'r') as file:
    data = csv.DictReader(file)
    for col in data:
        epoch.append(col["epoch"])
        FCGEC.append(col["loss"])
    
with open('FCGEC_all.csv', 'r') as file:
    data = csv.DictReader(file)
    for col in data:
        FCGEC_all.append(col["loss"])
    
with open('NLPCC_10000.csv', 'r') as file:
    data = csv.DictReader(file)
    for col in data:
        NLPCC.append(col["loss"])
    
with open('NLPCC_all.csv', 'r') as file:
    data = csv.DictReader(file)
    for col in data:
        NLPCC_all.append(col["loss"])
    
datas = zip(epoch, FCGEC, FCGEC_all, NLPCC, NLPCC_all)
with open('trainloss.csv', 'w') as file:
    csvwriter = csv.writer(file)
    csvwriter.writerows(datas)

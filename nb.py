import numpy as np
import csv
import argparse
import pandas as pd
import math


def naive_bayes(class_labels,a,b,output_file):
    total = len(class_labels)

    countofA = countofB = attr_1a = attr_2a = attr_1b = attr_2b = 0
    error_sqr_1a = error_sqr_2a = error_sqr_1b = error_sqr_2b = 0

    misclasscount = 0

    for i in range(0,total):
        if class_labels[i] == 'A':
            countofA += 1
            attr_1a += a[i]
            attr_2a += b[i]

        else:
            countofB += 1
            attr_1b += a[i]
            attr_2b += b[i]

    mean_a1 = (1/countofA)*attr_1a
    mean_a2 = (1/countofA)*attr_2a
    mean_b1 = (1/countofB)*attr_1b
    mean_b2 = (1/countofB)*attr_2b
    
    for i in range(0,total):
        if class_labels[i] == 'A':
            error_sqr_1a += (a[i]-mean_a1)**2
            error_sqr_2a += (b[i]-mean_a2)**2
        else:
            error_sqr_1b += (a[i]-mean_b1)**2
            error_sqr_2b += (b[i]-mean_b2)**2
    variance_a1 = (1/(countofA-1))*error_sqr_1a
    variance_a2 = (1/(countofA-1))*error_sqr_2a
    variance_b1 = (1/(countofB-1))*error_sqr_1b
    variance_b2 = (1/(countofB-1))*error_sqr_2b

    prob_inst_a = countofA / (countofA+countofB)
    prob_inst_b = countofB / (countofA+countofB)

    denominator_a1 = math.sqrt(2*math.pi*variance_a1)
    denominator_a2 = math.sqrt(2*math.pi*variance_a2)
    denominator_b1 = math.sqrt(2*math.pi*variance_b1)
    denominator_b2 = math.sqrt(2*math.pi*variance_b2)

    for i in range(0,total):
        numerator_a1 = math.exp(-1*(((a[i]-mean_a1)**2)/(2*variance_a1)))
        numerator_a2 = math.exp(-1*(((b[i]-mean_a2)**2)/(2*variance_a2)))

        likelihood_a = (numerator_a1/denominator_a1) * (numerator_a2/denominator_a2) * prob_inst_a

        numerator_b1 = math.exp(-1*(((a[i]-mean_b1)**2)/(2*variance_b1)))
        numerator_b2 = math.exp(-1*(((b[i]-mean_b2)**2)/(2*variance_b2)))

        likelihood_b = (numerator_b1/denominator_b1) * (numerator_b2/denominator_b2) * prob_inst_b
        #print(likelihood_a,likelihood_b)
        if((likelihood_a > likelihood_b) and (class_labels[i] == 'B') or (likelihood_a < likelihood_b) and (class_labels[i] == 'A') ):
            misclasscount += 1
    
    op_list1 = [mean_a1,variance_a1,mean_a2,variance_a2,prob_inst_a]
    op_list2 = [mean_b1,variance_b1,mean_b2,variance_b2,prob_inst_b]
    op_list3 = [misclasscount]
    print(op_list1)
    print(op_list2)
    print(op_list3)
    with open(output_file, "w", newline="") as tsvfile:
        #create csv writer object having tab space as delimiter
        tsv_writer = csv.writer(tsvfile, delimiter = '\t')
        tsv_writer.writerow(op_list1)
        tsv_writer.writerow(op_list2)
        tsv_writer.writerow(op_list3)


#Reading data from command line arguments
parser = argparse.ArgumentParser(description = 'Please enter the location of the Data file. and the location where you need the results')
parser.add_argument("--data", help = 'Data file')
parser.add_argument("--output", help = 'Output file')

args = parser.parse_args()
file_name = args.data
output_tsv_file = args.output

#Reading data from TSV file
data_tsv = np.genfromtxt(file_name,delimiter='\t',dtype=None,encoding=None)

#Implemented using Panda dataframe
df = pd.DataFrame(data_tsv)


class_lab = df.iloc[:,0].values
attr_1 = df.iloc[:,1].values
attr_2 = df.iloc[:,2].values

#Calling the function using the available input
naive_bayes(class_lab,attr_1,attr_2,output_tsv_file)
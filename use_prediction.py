import argparse
import helper
from Bio import SeqIO
import os
import subprocess
import re
import prediction
import numpy as np
import pandas as pd
import copy
import shutil
from sequence_feature_extraction import AAindex_feature_extraction
from sklearn.manifold import TSNE
from sklearn import preprocessing
import matplotlib.pyplot as plt
import plotly

parser = argparse.ArgumentParser(description='python use_prediction.py')
parser.add_argument('-i', help='A fasta file which contains all STS sequences for prediction', required=True)
parser.add_argument('-o', help='result and intermediate file will store here', required=True)
parser.add_argument('--complete_sequence',  action='store_true', default=False, help='default:False, ML model will use complete sequence to predict. this will decrease the accuarcy of prediction.'
                                                                            'But some STSs which lack complete C domain will be included')
parser.add_argument('--including', action='store_true', default=False, help='default:False, the unsupervised clustering model will include more than 1000 STSs described in paper, '
                                                                            'we strongly recommend you to add this parameter when you only have few STSs to predict. This '
                                                                            'may take a long time to make the alignment')
parser.add_argument('--without_supervised_learning',  action='store_true', default=False, help='default:True, the supervised clustering model will predict the intermidiate of the input STSs, '
                                                                                     'the outfile will store in prediction_result.txt')
parser.add_argument('--without_unsupervised_learning', action='store_true', default=False, help='default:True, the unsupervised clustering model will cluster the input STSs with all characterized (and uncharacterized)'
                                                                                      ' STSs. The cluster map will store in culstering_map.jpg and culstering_map.html, The coordinate of each STS will store'
                                                                                      ' in STS_coordinate.txt')
args = parser.parse_args()

if not os.path.exists(args.o):
    os.mkdir(args.o)

if not os.path.exists(f'./{args.o}/sequence'):
    os.mkdir(f'./{args.o}/sequence')

if not os.path.exists(f'./{args.o}/hmm_result'):
    os.mkdir(f'./{args.o}/hmm_result')

if not os.path.exists(f'./{args.o}/sequence_C'):
    os.mkdir(f'./{args.o}/sequence_C')

if not os.path.exists(f'./{args.o}/STS_feature'):
    os.mkdir(f'./{args.o}/STS_feature')

if not os.path.exists(f'./{args.o}/ali_dir'):
    os.mkdir(f'./{args.o}/ali_dir')

for seq_record in SeqIO.parse(args.i, "fasta"):
    if len(seq_record.seq) > 250:
        f = open(f"./{args.o}/sequence/{seq_record.id}.fasta", 'w')
        f.write(f">{seq_record.id}\n")
        f.write(f"{seq_record.seq}\n")
        f.close()

env_ = os.environ.copy()
env_['PATH'] += f';{os.getcwd()}\software'

if args.complete_sequence == False:
    for fasta in os.listdir(f'./{args.o}/sequence'):
        command = f'hmmsearch ./training_data/fungi_Terpene_synth_C.hmm ./{args.o}/sequence/{fasta} > ./{args.o}/hmm_result/{os.path.splitext(fasta)[0]}.out'
        subprocess.check_call(command, shell=True, env=env_)
    for out in os.listdir(f'./{args.o}/hmm_result'):
        key = os.path.splitext(out)[0]
        f = open(f"./{args.o}/hmm_result/{out}", 'r')
        sequence = ''
        for line in f.readlines():
            if re.search(f' *?{key} *?[0-9]+? (.*?) [0-9]+?', line):
                sequence += re.search(f' *?{key} *?[0-9]+? (.*?) [0-9]+?', line).group(1)
        sequence = sequence.replace('-','')
        f.close()
        if 100 < len(sequence) < 200:
            with open(f'./{args.o}/prediction_result.txt' , 'a+') as f:
                f.write(f'The C domain of {key} STS sequence is shorter then 200 aa, which shows its huge sequence difference with characterized fungi STS sequences. This may lead to a unrelieble prediction result\n')
            f = open(f"./{args.o}/sequence_C/{key}.fasta", 'w')
            f.write(f'>{key}\n')
            f.write(f'{sequence}\n')
            f.close()
        elif len(sequence) > 200:
            f = open(f"./{args.o}/sequence_C/{key}.fasta", 'w')
            f.write(f'>{key}\n')
            f.write(f'{sequence}\n')
            f.close()
        elif len(sequence) < 100:
            continue
    for fasta in os.listdir(f'./{args.o}/sequence_C'):
        AAindex_feature_extraction(f'./{args.o}/sequence_C/{fasta}', f'./{args.o}/STS_feature')
    for pkl in os.listdir(f'./feature_dict'):
        shutil.copy(f"./feature_dict/{pkl}", f"./{args.o}/STS_feature/{pkl}")
else:
    for fasta in os.listdir(f'./{args.o}/sequence'):
        AAindex_feature_extraction(f'./{args.o}/sequence/{fasta}', f'./{args.o}/STS_feature')
    for pkl in os.listdir(f'./feature_dict'):
        shutil.copy(f"./feature_dict/{pkl}", f"./{args.o}/STS_feature/{pkl}")


def final_supervised_predict(train_labels, train_keys, test_key, train_sequence_file, test_sequence_file, feature_id, feature_dict, ali_dir):
    all_keys = copy.deepcopy(train_keys)
    all_keys.append(test_key)
    all_indices = list(range(len(all_keys)))
    train_indices = all_indices[:len(all_indices)-1]
    test_indices = all_indices[len(all_indices)-1:]
    train_labels.append(None)
    labels = np.array(train_labels)
    aln = helper.make_ali_file(test_key, test_sequence_file, train_sequence_file, ali_dir, env_)
    f_dict = prediction.get_features_dict(feature_dict, all_keys)
    aln_features_dict = prediction.get_aligned_features(all_keys, f_dict, f'{ali_dir}/{test_key}_aln.fasta')
    residue_indices = prediction.get_conserved_residue_indices(aln, all_keys, 0.5)
    num_residues = len(aln[train_keys[0]])
    Data = prediction.Data(train_indices, test_indices, residue_indices, feature_id)
    Terpene = prediction.TerpeneTrees(aln_features_dict, num_residues, all_keys, labels, [])
    Data.train_test_split = Terpene.split_train_test(Data.train_indices, Data.test_indices)
    Terpene.get_residue_scores_per_feature(Data)
    Terpene.fit_and_predict_from_res_and_features(feature_id, Data, Data.sorted_residues[:27])
    f = open(f'./{args.o}/prediction_result.txt' , 'a+')
    f.write(f'{test_key} : {Data.y_proba}\n')

print(1)
df = pd.read_excel('./training_data/characterized_fungi_STS.xlsx')
keys = []
labels = []
key_to_products = {}
key_to_intermidiate = {}
key_to_species = {}

for i in list(df['Name']):
    keys.append(str(i))

for i in range(len(list(df['Name']))):
    if ',' in list(df['Product'])[i]:
        num = list(df['Product'])[i].count(',')
        product = list(df['Product'])[i].split(',', num)
        key_to_products[str(list(df['Name'])[i])] = product
    else:
        key_to_products[str(list(df['Name'])[i])] = [list(df['Product'])[i]]

for i in range(len(list(df['Name']))):
    key_to_species[str(list(df['Name'])[i])] = str(list(df['Organism'])[i])
    if list(df['intermidiate'])[i] == 'germacrene':
        key_to_intermidiate[str(list(df['Name'])[i])] = 'germacrene'
    elif list(df['intermidiate'])[i] == 'humulene':
        key_to_intermidiate[str(list(df['Name'])[i])] = 'humulene'

for i in keys:
    if key_to_intermidiate[i] == 'germacrene':
        labels.append(0)
    elif key_to_intermidiate[i] == 'humulene':
        labels.append(1)

feature_id = ['CHOP780212', 'LEVM760103', 'ROBB760111', 'KOEP990101', 'GOLD730101', 'WERD780102', 'RICJ880115', 'KRIW710101', 'MAXF760106']

if args.without_supervised_learning == False:
    if args.complete_sequence == False:
        for fasta in os.listdir(f'./{args.o}/sequence_C'):
            test_key = os.path.splitext(fasta)[0]
            train_keys = copy.deepcopy(keys)
            final_supervised_predict(labels, train_keys, test_key, './training_data/characterized_STS_C.fasta',f'./{args.o}/sequence_C/{fasta}', feature_id, f"{args.o}/STS_feature", f'./{args.o}/ali_dir')
    else:
        for fasta in os.listdir(f'./{args.o}/sequence'):
            test_key = os.path.splitext(fasta)[0]
            train_keys = copy.deepcopy(keys)
            final_supervised_predict(labels, train_keys, test_key, './training_data/characterized_STS_C.fasta',f'./{args.o}/sequence/{fasta}', feature_id, f"{args.o}/STS_feature", f'./{args.o}/ali_dir')
else:
    print('skip supervised learning')


def final_unsupervised_predict(labels, train_keys, paper_keys, test_keys, key_sequence, key_to_intermediate,key_to_species, feature_id, feature_dict, ali_dir, output_dir, env):
    all_keys = copy.deepcopy(train_keys)
    for i in paper_keys:
        all_keys.append(i)
        labels.append(None)
    for j in test_keys:
        all_keys.append(j)
        labels.append(None)
    all_indices = list(range(len(all_keys)))
    train_indices = all_indices[:len(train_keys)]
    test_indices = all_indices[len(train_keys):]
    labels = np.array(labels)
    aln = helper.make_ali_file_2(key_sequence, ali_dir, env)
    f_dict = prediction.get_features_dict(feature_dict, all_keys)
    aln_features_dict = prediction.get_aligned_features(all_keys, f_dict, f'{ali_dir}/unsuperviesed_learning_aln.fasta')
    residue_indices = prediction.get_conserved_residue_indices(aln, all_keys, 0.5)
    num_residues = len(aln[train_keys[0]])
    Data = prediction.Data(train_indices, test_indices, residue_indices, feature_id)
    Terpene = prediction.TerpeneTrees(aln_features_dict, num_residues, all_keys, labels, [])
    Data.train_test_split = Terpene.split_train_test(Data.train_indices, Data.test_indices)
    Terpene.get_residue_scores_per_feature(Data)
    Data.train_test_split = Terpene.split_train_test(list(range(len(all_keys))), [])
    train_x, _, _, _ = Terpene.split_train_test_expanded(Data.sorted_residues[:27], feature_id, Data.train_test_split)
    scaler = preprocessing.MinMaxScaler()
    tx_str = np.nan_to_num(scaler.fit_transform(np.nan_to_num(train_x)))
    reducer = TSNE(n_components=2, random_state=42)
    red_x = reducer.fit_transform(np.nan_to_num(tx_str))
    prediction.plot_kernel_pca(red_x, all_keys, key_to_intermediate)
    plt.savefig(f'./{output_dir}/culstering_map.jpg')
    plt.show()
    fig = prediction.plot_kernel_pca_plotly(red_x, all_keys, key_to_intermediate, key_to_products, key_to_species, output_dir)
    plotly.offline.plot(fig, filename=f'./{{output_dir}}/culstering_map.html')

df2 = pd.read_excel('./training_data/fungi_STS.xlsx')
train_keys = copy.deepcopy(keys)
paper_keys = []
test_keys = []

if args.including == True:
    for i, j, m, n in zip(list(df2['Name']), list(df2['Product']), list(df2['intermidiate']), list(df2['Organism'])):
        paper_keys.append(str(i))
        key_to_products[i] = str(j)
        key_to_intermidiate[i] = str(m)
        key_to_species[i] = str(n)
else:
    for i, j, m, n in zip(list(df2['Name'])[0:3], list(df2['Product'])[0:3], list(df2['intermidiate'])[0:3], list(df2['Organism'])[0:3]):
        paper_keys.append(str(i))
        key_to_products[i] = str(j)
        key_to_intermidiate[i] = str(m)
        key_to_species[i] = str(n)

key_sequence = {}

for seq_record in SeqIO.parse('./training_data/characterized_STS_C.fasta', "fasta"):
    key_sequence[seq_record.id] = seq_record.seq

for seq_record in SeqIO.parse('./training_data/fungi_STS_C.fasta', "fasta"):
    if seq_record.id in paper_keys:
        key_sequence[seq_record.id] = seq_record.seq

if args.complete_sequence == False:
    for fasta in os.listdir(f'./{args.o}/sequence_C'):
        test_key = os.path.splitext(fasta)[0]
        test_keys.append(test_key)
        key_to_intermidiate[test_key] = 'this study'
        key_to_species[test_key] = 'unknown'
        key_to_products[test_key] = 'unknown'
        for seq_record in SeqIO.parse(f'./{args.o}/sequence_C/{fasta}', "fasta"):
            key_sequence[seq_record.id] = seq_record.seq
else:
    for fasta in os.listdir(f'./{args.o}/sequence'):
        test_key = os.path.splitext(fasta)[0]
        test_keys.append(test_key)
        key_to_intermidiate[test_key] = 'this study'
        key_to_species[test_key] = 'unknown'
        key_to_products[test_key] = 'unknown'
        for seq_record in SeqIO.parse(f'./{args.o}/sequence/{fasta}', "fasta"):
            key_sequence[seq_record.id] = seq_record.seq

if args.without_unsupervised_learning == False:
    final_unsupervised_predict(labels, train_keys, paper_keys, test_keys, key_sequence, key_to_intermidiate, key_to_species, feature_id, f"{args.o}/STS_feature", f'./{args.o}/ali_dir', args.o, env_)
else:
    print('skip unsupervised learning')


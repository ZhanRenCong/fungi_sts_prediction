import subprocess
import pickle
import os
import numpy as np
import helper
import re

SSPRO8_VALUES = "HGIEBTSC"
ACCPRO_VALUES = "be-"
SSPRO_VALUES = "HEC"
AA = 'ACDEFGHIKLMNPQRSTVWY-X'
STRING_MAPPINGS = {'sspro': dict(zip(SSPRO_VALUES, range(len(SSPRO_VALUES)))),
                   'sspro8': dict(zip(SSPRO8_VALUES, range(len(SSPRO8_VALUES)))),
                   'accpro': dict(zip(ACCPRO_VALUES, range(len(ACCPRO_VALUES)))),
                   'amino_acids': dict(zip(AA, range(len(AA))))}

def AAindex_feature_extraction(sequence_file, output_dir):
    """
    use iFeature to get features from amino acid sequences.
    """

    _, filename, _ = helper.get_file_parts(sequence_file)
    output_file = output_dir + '/' + filename + '_AAindex.txt'
    command = 'python ./iFeature-master/iFeature.py --file ' + sequence_file + ' --type AAINDEX --out ' + output_file
    subprocess.check_call(command, shell=True, env=None)
    f = open(output_file, 'r')
    feature_id = []
    feature = {}
    final_feature_id = ['CHOP780212', 'LEVM760103', 'ROBB760111', 'KOEP990101', 'GOLD730101', 'WERD780102', 'RICJ880115', 'KRIW710101', 'MAXF760106']
    final_feature = {}
    for line in f.readlines():
        data = line.split()
        if data[0] == '#':
            for i in data:
                a = re.sub('.*?\..*?\.', '', i)
                if a not in feature_id:
                    feature_id.append(a)
            feature_id.remove('#')
        else:
            pkl_file = data[0] + '_AAindex' + '.pkl'
            data.remove(data[0])
            for i in range(len(data)):
                if i < len(feature_id):
                    feature[feature_id[i]] = []
                    feature[feature_id[i]].append(data[i])
                else:
                    feature[feature_id[i % len(feature_id)]].append(data[i])
            for feature_name in feature_id:
                feature[feature_name] = np.array(feature[feature_name])
    for id in final_feature_id:
        final_feature[id] = feature[id]
    pickle.dump(final_feature, open(output_dir + '/' + pkl_file, "wb"))
    f.close()
    os.remove(output_file)

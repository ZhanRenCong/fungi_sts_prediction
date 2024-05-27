import helper
import re
import pickle
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from sequence_feature_extraction import STRING_MAPPINGS
import numba as nb
from sklearn import model_selection,svm,metrics, preprocessing
from sklearn.pipeline import Pipeline
from xgboost.sklearn import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import svm



SCORE_NAMES = ['accuracy',"f1", "balanced_accuracy", "roc_auc", "precision_recall_auc"]
PRODUCT_GROUPING = {"F": ([], {
        "10,1": (
            ['(+)-eremophilene', 'aristolene', 'guaiadiene', '(+)-eremophilene',
             'intermedeol', 'beta-elemene', 'alpha-guaiene',  'alpha-gurjunene', 'delta-cadinol', 'alloaromadendrene', 'germacrene a',
             'viridiflorol', 'ledene', 'virifloridol', 'viridiflorene', 'alpha-selinene', 'aristolochene','beta-gurjunene',  'germacrene d'
             ,'germacrene-d-4-ol','t-gurjunene']
            ,{}),
        "11,1": (['delta-6-protoilludene', 'presilphiperfolan-8beta-ol', 'hirsutene', 'sterprene', 'beta-caryophyllene', 'trichobrasileno', 'pentalenene',
                  'alpha-caryophyllene', 'caryophyllene', 'silphiene' , 'beta-caryophyllene', '6-protoilludene','koraiol', 'alpha-humulene'], {}),
        "acyclic": (["farnesene", "farnesol"], {})
    }),
              "N": ([], {
                  "acyclic": (["nerolidol"], {}),
                  '10,1' : (['alpha-cubebene', 'beta-cubebene', 'delta-cadinene', 'beta-copaene', 'gamma-cadinene', '(–)-gamma-cadinene','torreyol', 'alpha-cadinol',
                             '1-epi-cubenol', 'cubebol', 'alpha-cadinene', 'alpha-muurolene', 'gamma-muurolene'], {}),
                  "6,1": (
                      ['(+)-acoradiene','longiborneol','beta-trans-bergamotene','(-)-alpha-acorenol', 'beta-chamigrene','alpha-cuprenene','alpha-barbatene','beta-barbatene','trichodiene'], {}),
                  "7,1":(['daucadiene'], {}),
              }),
              'special mechanism' : (['alpha-ionylideneethane'], {}),
              'this study' : (['this study'], {}),
              'strains' : (['strains'], {}),
              'uncharacterized' : (['uncharacterized'], {}),
              'ascomycota' :(['ascomycota'], {}),
              'basdiomycota' :(['basdiomycota'], {}),
              'characterized' :(['characterized'], {})}

def get_product_group_graph(product: str) -> list:
    """
    Gets intermediate tree of a product from the (F)arneyl or (N)eroildyl cation

    Parameters
    ----------
    product: name of product

    Returns
    -------
    list of intermediates
    """
    product = product.lower()

    def is_product_in_group(p, group):
        for string in group:
            if string in p:
                return True
        return False

    def get_match(p, groups, graph):
        for parent in groups:
            if is_product_in_group(p, groups[parent][0]):
                return graph + [parent]
            else:
                result = get_match(p, groups[parent][1], graph + [parent])
                if result is not None:
                    return result

    return get_match(product, PRODUCT_GROUPING, [])


def get_onehots(num_columns):
    return {k: preprocessing.OneHotEncoder(categories=[list(range(len(STRING_MAPPINGS[k]) + 1))] * num_columns) for k in STRING_MAPPINGS}


def get_features_dict(features_dir, keys):
    features_dict = defaultdict(list)
    features_dir = Path(features_dir)
    for key in keys:
        AAindex_file = features_dir / f"{key}_AAindex.pkl"
        if AAindex_file.exists():
            with open(AAindex_file, "rb") as f:
                features_dict[key].append(pickle.load(f))
        else:
            print(f"AAindex file not found for key {key}")
    return {k: features_dict[k] for k in features_dict if len(features_dict[k]) == 1}

def get_conserved_residue_indices(aln_sequences, train_keys, discard_threshold):
    """
    Finds indices of positions in the an alignment (aln_sequences) which have more than discard_threshold percentage
    of train_keys with non-gap residues.
    """
    residue_indices = []
    for i in range(len(aln_sequences[train_keys[0]])):
        if sum(1 for key in train_keys if aln_sequences[key][i] == '-') / len(train_keys) <= discard_threshold:
            residue_indices.append(i)
    return np.array(residue_indices)

def _make_aln_features_x(key, features_x: np.ndarray, aligned_sequence: str):
    """
    Makes aligned matrix of features according to an aligned sequence

    Parameters
    ----------
    features_x
    aligned_sequence

    Returns

    -------
    matrix
    """
    mask = np.array([i for i in range(len(aligned_sequence)) if aligned_sequence[i] != '-'])
    if features_x.ndim == 2:
        assert len(aligned_sequence.replace('-', '')) == features_x.shape[1], f"{key} {features_x.shape[1]}"
        if features_x.shape[0] == features_x.shape[1]:
            aln_features = np.zeros((len(aligned_sequence), len(aligned_sequence)))
            aln_features[:] = np.nan
            aln_features[np.ix_(mask, mask)] = features_x.T
        else:
            aln_features = np.zeros((len(aligned_sequence), features_x.shape[0]))
            aln_features[:] = np.nan
            aln_features[mask, :] = features_x.T
        aln_features = aln_features.reshape((1, aln_features.shape[0] * aln_features.shape[1]))
    else:
        assert len(aligned_sequence.replace('-', '')) == features_x.shape[0], f"{key} {features_x.shape[0]}"
        aln_features = np.zeros((1, len(aligned_sequence)))
        aln_features[:] = np.nan
        aln_features[:, mask] = features_x
    return aln_features


def _get_feature_shapes(structure_feature_names: list, AAindex_feature_names: list, features_dict: dict, first_id: str, aln_length: int):
    feature_shapes = {}
    for feature in structure_feature_names:
        example_data = features_dict[first_id][2][feature]
        if isinstance(example_data, list) or isinstance(example_data, str) or example_data.ndim == 1:
            feature_shapes[feature] = 1
        else:
            if example_data.shape[0] == example_data.shape[1]:
                feature_shapes[feature] = aln_length
            else:
                feature_shapes[feature] = example_data.shape[0]
    for feature in AAindex_feature_names:
        example_data = features_dict[first_id][0][feature]
        if isinstance(example_data, list) or isinstance(example_data, str) or example_data.ndim == 1:
            feature_shapes[feature] = 1
        else:
            if example_data.shape[0] == example_data.shape[1]:
                feature_shapes[feature] = aln_length
            else:
                feature_shapes[feature] = example_data.shape[0]
    return feature_shapes


def get_aligned_features(identifiers, features_dict: dict, aln, num_models=0) -> dict:
    """
    Aligns features according to an alignment

    Parameters
    ----------
    identifiers
    features_dict
    aln_sequences
    num_models

    Returns
    -------
    dict of {feature: matrix of values with num rows = num identifiers}
    """

    def encode_string_features(features_x: str, mapping: dict):
        return np.array([mapping[x] if x in mapping else len(mapping) for x in features_x])
    AAindex_feature_names = list(features_dict[identifiers[0]][0].keys())
    if len(features_dict[identifiers[0]]) > 2:
        structure_feature_names = [x for x in features_dict[identifiers[0]][2].keys() if x != 'dssp_resnum']
    else:
        structure_feature_names = []
    aln_sequences = helper.get_sequences_from_fasta(aln)
    aln_length = len(aln_sequences[identifiers[0]])
    feature_shapes = _get_feature_shapes(structure_feature_names, AAindex_feature_names, features_dict, identifiers[0], aln_length)
    x_dict = {}
    for feature in structure_feature_names:
        for model_id in range(num_models):
            x_dict[f"{feature}:{model_id}"] = np.zeros((len(identifiers), aln_length * feature_shapes[feature]))
            for i in range(len(identifiers)):
                farray = features_dict[identifiers[i]][model_id + 2][feature]
                if "fluctuation" in feature:
                    farray = farray / np.nansum(farray ** 2) ** 0.5
                x_dict[f"{feature}:{model_id}"][i] = _make_aln_features_x(
                    identifiers[i],
                    farray,
                    aln_sequences[identifiers[i]])
    for feature in AAindex_feature_names:
        x_dict[feature] = np.zeros((len(identifiers), aln_length * feature_shapes[feature]))
        for i in range(len(identifiers)):
            feature_data = features_dict[identifiers[i]][0][feature]
            if feature in STRING_MAPPINGS:
                feature_data = encode_string_features(feature_data, STRING_MAPPINGS[feature])
            x_dict[feature][i] = _make_aln_features_x(
                identifiers[i],
                np.array(feature_data),
                aln_sequences[identifiers[i]])
    return x_dict


def get_metrics(test_labels, test_proba, threshold=0.5, pred=True):
    if not pred:
        test_pred = test_proba[:, 1] > threshold
        precision, recall, thresholds_pr = metrics.precision_recall_curve(test_labels, test_proba[:, 1])
        scores = {
            "f1": metrics.f1_score(test_labels, test_pred),
            "accuracy": metrics.accuracy_score(test_labels, test_pred),
            "balanced_accuracy": metrics.balanced_accuracy_score(test_labels, test_pred),
            "roc_auc": metrics.roc_auc_score(test_labels, test_proba[:, 1]),
            "precision_recall_auc": metrics.auc(recall, precision)
        }
    else:
        test_pred = test_proba
        precision, recall, thresholds_pr = metrics.precision_recall_curve(test_labels, test_pred)
        scores = {
            "f1": metrics.f1_score(test_labels, test_pred),
            "accuracy": metrics.accuracy_score(test_labels, test_pred),
            "balanced_accuracy": metrics.balanced_accuracy_score(test_labels, test_pred),
            "roc_auc": metrics.roc_auc_score(test_labels, test_pred),
            "precision_recall_auc": metrics.auc(recall, precision)
        }

    return [np.round(scores[s_name], 3) for s_name in SCORE_NAMES] + [metrics.confusion_matrix(test_labels, test_pred)]

def compute_Standard_Deviation(result_txt):
    condition_result = defaultdict(dict)
    f = open(result_txt, 'r')
    data = f.readlines()
    for i in list(range(len(data))):
        if 'Using STS full length' in data[i]:
            condition = re.search(r'\((.*)\)', data[i], re.M|re.I).group(1)
            condition_result[condition] = defaultdict(list)
        elif 'Using STS C domain' in data[i]:
            condition = str(re.search(r'\((.*)\)', data[i], re.M|re.I).group(1)) + '_C'
            condition_result[condition] = defaultdict(list)
    for i in list(range(len(data))):
        if 'Using STS full length' in data[i]:
            condition = re.search(r'\((.*)\)', data[i], re.M | re.I).group(1)
            accuracy = float(re.search(r':(.*)', data[i+1]).group(1))
            f1 = float(re.search(r':(.*)', data[i+2]).group(1))
            balanced_accuracy = float(re.search(r':(.*)', data[i+3]).group(1))
            roc_auc = float(re.search(r':(.*)', data[i+4]).group(1))
            precision_recall_auc = float(re.search(r':(.*)', data[i+4]).group(1))
            condition_result[condition]['accuracy'].append(accuracy)
            condition_result[condition]['f1'].append(f1)
            condition_result[condition]['balanced_accuracy'].append(balanced_accuracy)
            condition_result[condition]['roc_auc'].append(roc_auc)
            condition_result[condition]['precision_recall_auc'].append(precision_recall_auc)
        if 'Using STS C domain' in data[i]:
            condition = str(re.search(r'\((.*)\)', data[i], re.M|re.I).group(1)) + '_C'
            accuracy = float(re.search(r':(.*)', data[i+1]).group(1))
            f1 = float(re.search(r':(.*)', data[i+2]).group(1))
            balanced_accuracy = float(re.search(r':(.*)', data[i+3]).group(1))
            roc_auc = float(re.search(r':(.*)', data[i+4]).group(1))
            precision_recall_auc = float(re.search(r':(.*)', data[i+4]).group(1))
            condition_result[condition]['accuracy'].append(accuracy)
            condition_result[condition]['f1'].append(f1)
            condition_result[condition]['balanced_accuracy'].append(balanced_accuracy)
            condition_result[condition]['roc_auc'].append(roc_auc)
            condition_result[condition]['precision_recall_auc'].append(precision_recall_auc)
    for condition in condition_result.keys():
        for i in condition_result[condition].keys():
            mean = np.mean(condition_result[condition][i])
            mean = ("%.2f" % mean)
            std = np.std(condition_result[condition][i], ddof = 1)
            std = ("%.2f" %  std)
            print(condition + ' ' + i + ':' + str(mean) + '+' + str(std))

class Data:
    def __init__(self,
                 train_indices,
                 test_indices,
                 residue_indices,
                 feature_names,
                 ):
        self.train_indices = train_indices
        self.test_indices = test_indices
        self.train_test_split = None
        self.residue_indices = np.array(residue_indices)
        self.feature_names = feature_names
        self.feature_residue_scores = {}
        self.residue_scores_folds = None
        self.residue_scores = None
        self.sorted_residues = None
        self.feature_cv_indices = []
        self.feature_cv_predictions = defaultdict(list)
        self.feature_test_predictions = defaultdict(list)
        self.clf = None
        self.y_proba = None
        self.y_test = None
        self.test_x = None

    def get_residue_scores(self):
        assert len(self.feature_residue_scores)
        self.residue_scores_folds = normalize_matrix(np.array([np.sum(self.feature_residue_scores[f], axis=0) for f in self.feature_residue_scores]))
        self.residue_scores = np.sum(self.residue_scores_folds, axis=0)
        self.sorted_residues = self.residue_indices[np.argsort(-self.residue_scores)]


@nb.njit
def normalize_matrix(matrix):
    matrix_s = np.zeros(matrix.shape)
    for i in range(matrix.shape[0]):
        matrix_s[i] = normalize(matrix[i])
    return matrix_s


@nb.njit
def normalize(row):
    maxv, minv = np.max(row), np.min(row)
    return (row-maxv)/(maxv-minv)


def get_clf(num_cols,
            learning_rate=0.005,
            n_estimators=2000,
            max_depth=5,
            min_child_weight=2,
            gamma=0.01,
            max_delta_step=0,
            subsample=0.7,
            colsample_bytree=0.1,
            colsample_bylevel=0.1,
            colsample_bynode=1,
            verbosity: int = 0,
            base_score=0.5,
            scale_pos_weight=1,
            reg_lambda=1):
    if verbosity > 1:
        assert num_cols is not None
        print("Number of columns", num_cols,
              int(colsample_bytree * num_cols),
              int(colsample_bytree * colsample_bylevel * num_cols),
              int(colsample_bytree * colsample_bylevel * colsample_bynode * num_cols))
    return XGBClassifier(
        learning_rate=learning_rate,
        base_score=base_score,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        gamma=gamma,
        max_delta_step=max_delta_step,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        colsample_bylevel=colsample_bylevel,
        colsample_bynode=colsample_bynode,
        reg_alpha=0,
        reg_lambda=reg_lambda,
        missing=None,
        objective='binary:logistic',
        importance_type='weight',
        scale_pos_weight=scale_pos_weight,
        nthread=40,
        seed=42)


class TerpeneTrees:
    def __init__(self,
                 aln_features_dict,
                 num_residues,
                 keys,
                 labels,
                 model_feature_names,
                 num_models=3):
        self.aln_features_dict = aln_features_dict
        self.num_residues = num_residues
        self.onehots = get_onehots(num_residues)
        self.model_feature_names = set(model_feature_names)
        self.num_models = num_models
        self.aln_features_dict_expanded = self.expand_features()
        self.keys = keys
        self.labels = labels
        self.color_mapper = {
            'F-acyclic': '#fcbba1',
            'F-10,1': '#fb6a4a',
            'F-11,1': '#cb181d',
            'N-acyclic': '#c6dbef',
            'N-6,1': '#4292c6',
            'N-10,1': '#084594',
            'N-7,1': '#3eede7',
            #'strains' : '#FFFF00',
            'multiple': 'white',
            #'special mechanism': '#574266',
            'ascomycota': '#7CFC00',
            'basdiomycota': '#0000DB',
            'this study': '#FFCC00',
            'characterized' : '#FF00FF'
        }
        self.marker_mapper = {
            'F': 's',
            'N': 'D',
            'this study': '*',
            'characterized': '*',
            'uncharacterized' : 'o',
            # 'strains' : 'o',
            'ascomycota': 'o',
            'basdiomycota': 'o',
            'multiple': '^',
            #'special mechanism': '^'
        }

    def fit_and_predict(self,
            data: Data,
            num_folds_residue_scores: int = 3,
            test_size_residue_scores: float = 0.1,
            verbosity: int = 0,
            top_num_residues: int = 27):
        data.train_test_split = self.split_train_test(data.train_indices, data.test_indices)
        self.get_residue_scores_per_feature(data, num_folds=num_folds_residue_scores, test_size=test_size_residue_scores, verbosity=verbosity)
        top_residues = data.sorted_residues[:top_num_residues]
        train_x, y_train, test_x, y_test = self.split_train_test_expanded(top_residues, data.feature_names, data.train_test_split)
        data.clf = get_clf(num_cols=train_x.shape[1])
        data.clf.fit(train_x, y_train)
        data.y_proba = data.clf.predict_proba(test_x)
        data.y_test = y_test

    def fit_and_predict_from_res(self, data: Data, top_residues: list):
        data.train_test_split = self.split_train_test(data.train_indices, data.test_indices)
        train_x, y_train, test_x, y_test = self.split_train_test_expanded(top_residues, data.feature_names, data.train_test_split)
        data.clf = get_clf(num_cols=train_x.shape[1])
        data.clf.fit(train_x, y_train)
        data.y_proba = data.clf.predict_proba(test_x)
        data.y_test = y_test


    def fit_and_predict_from_res_and_features_SVM(self, feature_name, data: Data, top_residues: list):
        data.train_test_split = self.split_train_test(data.train_indices, data.test_indices)
        train_x, y_train, test_x, y_test = self.split_train_test_expanded(top_residues, feature_name, data.train_test_split)
        #regr = svm.SVR()
        #regr.fit(train_x, y_train.astype('int'))
        #data.y_proba = regr.predict(test_x)
        train_x[np.isnan(train_x)] = 0
        test_x[np.isnan(test_x)] = 0
        clf = Pipeline([("scaler", StandardScaler()),
                        ("svm_clf", svm.SVC(C=20, probability=True))
                        ])
        clf.fit(train_x, y_train.astype('int'))
        data.y_proba = clf.predict_proba(test_x)
        data.y_test = y_test



    def fit_and_predict_from_res_and_features(self, feature_name, data: Data, top_residues: list, learning_rate = 0.005, n_estimators = 2000,
                                              max_depth = 4, min_child_weight =3, gamma = 0.01, subsample = 0.7, colsample_bytree = 0.1, pos_weight=2, reg_lambda=1):
        data.train_test_split = self.split_train_test(data.train_indices, data.test_indices)
        train_x, y_train, test_x, y_test = self.split_train_test_expanded(top_residues, feature_name, data.train_test_split)
        data.clf = get_clf(num_cols=train_x.shape[1],
                           learning_rate=learning_rate,
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            min_child_weight=min_child_weight,
                            gamma=gamma,
                            max_delta_step=0,
                            subsample=subsample,
                            colsample_bytree=colsample_bytree,
                            colsample_bylevel=0.1,
                            colsample_bynode=1,
                            base_score=0.5,
                            scale_pos_weight=pos_weight,
                            reg_lambda=reg_lambda)
        data.clf.fit(train_x, y_train)
        data.y_proba = data.clf.predict_proba(test_x)
        data.y_test = y_test
        data.test_x = test_x

    def fit_and_predict_from_res_and_features_default(self, feature_name, data: Data, top_residues: list):
        data.train_test_split = self.split_train_test(data.train_indices, data.test_indices)
        train_x, y_train, test_x, y_test = self.split_train_test_expanded(top_residues, feature_name, data.train_test_split)
        data.clf = get_clf(num_cols=train_x.shape[1])
        data.clf.fit(train_x, y_train)
        data.y_proba = data.clf.predict_proba(test_x)
        data.y_test = y_test
        data.test_x = test_x


    def fit_and_predict_from_train_x(self, data: Data, train_x, y_train, test_x, y_test):
        data.clf = get_clf(num_cols=train_x.shape[1])
        data.clf.fit(train_x, y_train)
        data.y_proba = data.clf.predict_proba(test_x)
        data.y_test = y_test

    def get_residue_scores_per_feature(self,
                                       data: Data,
                                       num_folds=1,
                                       test_size=0.05,
                                       verbosity=0):
        assert data.train_test_split is not None
        if num_folds > 1:
            folder = model_selection.StratifiedShuffleSplit(n_splits=num_folds, random_state=42, test_size=test_size)
            data.feature_cv_indices = list(folder.split(X=np.zeros((len(data.train_indices), 2)),
                                                        y=np.zeros(len(data.train_indices))))
        else:
            data.feature_cv_indices = [(np.arange(len(data.train_indices)), np.arange(len(data.test_indices)))]
        group_feature_names = defaultdict(list)
        for feature_name in data.feature_names:
            if feature_name.split('_')[-1] in {'max', 'min', 'mean', 'ca', 'cb'}:
                group_feature_names['_'.join(feature_name.split('_')[:-1])].append(feature_name)
            else:
                group_feature_names[feature_name].append(feature_name)
        for feature_name_prefix in group_feature_names:
            data.feature_residue_scores[feature_name_prefix] = np.zeros((num_folds, len(data.residue_indices)))
            if verbosity > 0:
                print(feature_name_prefix)
            train_x, y_train, test_x, y_test = self.split_train_test_expanded(data.residue_indices, group_feature_names[feature_name_prefix],
                                                                              data.train_test_split)
            num_cols = sum(self.get_num_cols(feature_name) for feature_name in group_feature_names[feature_name_prefix])
            clf = get_clf(learning_rate=0.1,
                          n_estimators=100,
                          max_depth=3,
                          min_child_weight=1,
                          max_delta_step=0,
                          gamma=0,
                          subsample=1,
                          colsample_bytree=1,
                          colsample_bylevel=1,
                          verbosity=verbosity,
                          num_cols=num_cols * len(data.residue_indices))
            test_scores = np.zeros((num_folds, len(SCORE_NAMES)))
            for f, (fold_train_indices, fold_test_indices) in enumerate(data.feature_cv_indices):
                clf.fit(train_x[fold_train_indices], y_train[fold_train_indices])
                index = 0
                for fname in group_feature_names[feature_name_prefix]:
                    f_num_cols = self.get_num_cols(fname)
                    for i, r in enumerate(data.residue_indices):
                        data.feature_residue_scores[feature_name_prefix][f, i] += np.nansum(
                            clf.feature_importances_[index + i * f_num_cols: index + (i + 1) * f_num_cols])
                    index += f_num_cols * len(data.residue_indices)
                if verbosity > 1:
                    y_pred_test = clf.predict_proba(test_x)
                    data.feature_test_predictions[feature_name_prefix].append((y_test, y_pred_test))
                    indices = np.where(y_test != -1)
                    test_scores[f] = get_metrics(y_test[indices], y_pred_test[indices])[:-1]
            if verbosity > 1:
                print("Test")
                for s, score_name in enumerate(SCORE_NAMES):
                    print(f"{score_name}: {np.round(np.mean(test_scores[:, s]), 3)} +/- {np.round(np.std(test_scores[:, s]), 3)}")
                print()
        data.get_residue_scores()


    def get_inds(self, fname, residue_indices: list, expanded=True):
        if fname in self.onehots:
            if expanded:
                n_values = self.onehots[fname].categories_[0].shape[0]
            else:
                n_values = 1
        elif "pssm" in fname or "psfm" in fname:
            n_values = 20
        elif fname in self.model_feature_names:
            n_values = self.num_models
        else:
            n_values = 1
        inds = []
        for r in residue_indices:
            inds += list(range(r * n_values, (r + 1) * n_values))
        return inds

    def get_num_cols(self, fname):
        if fname in self.onehots:
            return self.onehots[fname].categories_[0].shape[0]
        elif "pssm" in fname or "psfm" in fname:
            return 20
        elif fname in self.model_feature_names:
            return self.num_models
        else:
            return 1

    def expand_features(self):
        expanded_dict = {}
        for feature_name in self.aln_features_dict:
            if feature_name in self.onehots:
                expanded_dict[feature_name] = np.array(self.aln_features_dict[feature_name])
                expanded_dict[feature_name][np.isnan(expanded_dict[feature_name])] = self.onehots[feature_name].categories[0][-1]
                expanded_dict[feature_name] = expanded_dict[feature_name].astype(np.int32)
                np.set_printoptions(threshold=np.inf)
                np.set_printoptions(threshold=np.inf)
                expanded_dict[feature_name] = self.onehots[feature_name].fit_transform(expanded_dict[feature_name]).todense()
            elif feature_name.split(':')[0] in self.model_feature_names:
                base_name = feature_name.split(':')[0]
                expanded_dict[base_name] = np.zeros(
                    (self.aln_features_dict[feature_name].shape[0], self.aln_features_dict[feature_name].shape[1] * self.num_models))
                index = 0
                for r in range(self.aln_features_dict[feature_name].shape[1]):
                    for m in range(self.num_models):
                        expanded_dict[base_name][:, index] = self.aln_features_dict[f"{base_name}:{m}"][:, r]
                        index += 1
            else:
                expanded_dict[feature_name] = self.aln_features_dict[feature_name]
        return expanded_dict

    def split_train_test(self, train_indices, test_indices):
        train_data = {}
        test_data = {}
        for feature in self.aln_features_dict_expanded:
            train_data[feature] = self.aln_features_dict_expanded[feature][train_indices]
            test_data[feature] = self.aln_features_dict_expanded[feature][test_indices]
        y_train = self.labels[train_indices]
        y_test = self.labels[test_indices]
        return train_data, y_train, test_data, y_test

    def split_train_test_expanded(self, residue_indices, feature_names, train_test_split):
        train_data, y_train, test_data, y_test = train_test_split
        num_cols = sum(self.get_num_cols(fname) * len(residue_indices) for fname in feature_names)
        train_x = np.zeros((len(y_train), num_cols))
        test_x = np.zeros((len(y_test), num_cols))
        index = 0
        for feature_name in feature_names:
            inds = self.get_inds(feature_name, residue_indices)
            num_cols_fname = len(inds)
            train_x[:, index: index + num_cols_fname] = train_data[feature_name][:, inds]
            test_x[:, index: index + num_cols_fname] = test_data[feature_name][:, inds]
            index += num_cols_fname
        return train_x, y_train, test_x, y_test


def plot_kernel_pca(red_x: np.ndarray, train_keys: list, key_to_intermediate: dict):
    colors = []
    markers = []
    color_mapper = {
        'acyclic': '#fcbba1',
        'germacrene': '#cb181d',
        'bisabolene': '#4292c6',
        'humulene': '#084594',
        'ascomycota': '#7CFC00',
        'basdiomycota': '#0000DB',
        'this study': '#FFFF00',
        #'uncharacterized' : '#808080'
    }
    marker_mapper = {
        'germacrene': 's',
        'acyclic': 's',
        'humulene': 'D',
        'bisabolene': 'D',
        'this study': '*',
        'ascomycota': 'o',
        'basdiomycota': 'o',
        #'uncharacterized' : 'o'
    }
    colors_ = {
        'acyclic': '#fcbba1',
        'germacrene': '#cb181d',
        'bisabolene': '#4292c6',
        'humulene': '#084594',
        'ascomycota': '#7CFC00',
        'basdiomycota': '#0000DB',
        'this study': '#FFFF00',
        #'uncharacterized' : '#808080'
    }
    marker_ = {
        'germacrene': 's',
        'acyclic': 's',
        'humulene': 'D',
        'bisabolene': 'D',
        'this study': '*',
        'ascomycota': 'o',
        'basdiomycota': 'o',
        #'uncharacterized' : 'o'
    }
    for key in train_keys:
        colors.append(color_mapper[key_to_intermediate[key]])
        markers.append(marker_mapper[key_to_intermediate[key]])
    plt.rc('font', size=15)
    plt.rc('text', usetex=False)
    plt.rc('xtick', labelsize=8)
    plt.rc('ytick', labelsize=8)
    plt.rc('axes', labelsize=8)
    width = 15
    height = 15
    plt.figure(figsize=(width, height))
    plt.axis('off')
    for marker in marker_mapper.values():
        if marker == 'o':
            indices = [i for i, m in enumerate(markers) if m == marker]
            plt.scatter(red_x[indices, 0],
                        red_x[indices, 1],
                        c=[colors[i] for i in indices],
                        marker=marker,
                        alpha=0.1,
                        s=20,
                        linewidths=0.5,
                        edgecolors='black',
                        )
        elif marker == '*':
            indices = [i for i, m in enumerate(markers) if m == marker]
            plt.scatter(red_x[indices, 0],
                        red_x[indices, 1],
                        c=[colors[i] for i in indices],
                        marker=marker,
                        alpha=0.9,
                        s=150,
                        linewidths=0.5,
                        edgecolors='black',
                        )
        else:
            indices = [i for i, m in enumerate(markers) if m == marker]
            plt.scatter(red_x[indices, 0],
                        red_x[indices, 1],
                        c=[colors[i] for i in indices],
                        marker=marker,
                        alpha=0.9,
                        s= 50,
                        linewidths=0.5,
                        edgecolors='black',
                        )
    legend_lines = []
    for color in colors_:
        if color == 'ascomycota' or color == 'basdiomycota':
            legend_lines.append(
                plt.scatter([], [], label=color, marker=marker_[color],
                            c=colors_[color], alpha=0.3,
                            s=50,
                            linewidth=0.5,
                            edgecolors='black'))
        else:
            legend_lines.append(
                plt.scatter([], [], label=color, marker=marker_[color],
                            c=colors_[color], alpha=0.9,
                            s=50,
                            linewidth=0.5,
                            edgecolors='black'))
    plt.legend(handles=legend_lines, fontsize=11)


def plot_kernel_pca_plotly(red_x: np.ndarray, train_keys: list, key_to_intermediate: dict, key_to_products:dict, key_to_species:dict, output_dir):
    data = []
    plotly_marker_mapping = {'s': 'square', 'D': 'diamond', '^': 'triangle-up', 'o': 'circle', '*': 'star'}
    colors = []
    markers = []
    color_mapper = {
        'acyclic': '#fcbba1',
        'germacrene': '#cb181d',
        'bisabolene': '#4292c6',
        'humulene': '#084594',
        'ascomycota': '#7CFC00',
        'basdiomycota': '#0000DB',
        'this study': '#FFFF00'
        #'uncharacterized' : '#808080'
    }
    marker_mapper = {
        'germacrene': 's',
        'acyclic': 's',
        'humulene': 'D',
        'bisabolene': 'D',
        'ascomycota': 'o',
        'basdiomycota': 'o',
        'this study': '*'
        #'uncharacterized' : 'o'
    }
    f = open(f'./{output_dir}/STS_coordinate.txt', 'w')
    for key in train_keys:
        colors.append(color_mapper[key_to_intermediate[key]])
        markers.append(marker_mapper[key_to_intermediate[key]])
    for marker in marker_mapper.values():
        if marker == 'o':
            indices = [i for i, m in enumerate(markers) if m == marker]
            for a in indices:
                f.write(train_keys[a] + ' ' + '(' + str(red_x[a, 0]) + ',' + str(red_x[a, 1]) + ')' + '\n')
            data.append(go.Scatter(x=red_x[indices, 0],
                                   y=red_x[indices, 1],
                                   mode='markers',
                                   marker=dict(
                                       size=5,
                                       symbol=plotly_marker_mapping[marker],
                                       color=[colors[x] for x in indices],
                                       line=dict(
                                           width=1,
                                       ),
                                       opacity=0.05
                                   ),
                                   text=[train_keys[x] + "\n" + key_to_species[train_keys[x]] + "\n" +
                                         key_to_products[train_keys[x]] for x in indices],
                                   ))
        else:
            indices = [i for i, m in enumerate(markers) if m == marker]
            for a in indices:
                f.write(train_keys[a] + ' ' + '(' + str(red_x[a, 0]) + ',' + str(red_x[a, 1]) + ')'+ '\n')
            data.append(go.Scatter(x=red_x[indices, 0],
                                   y=red_x[indices, 1],
                                   mode='markers',
                                   marker=dict(
                                       size=10,
                                       symbol=plotly_marker_mapping[marker],
                                       color=[colors[x] for x in indices],
                                       line=dict(
                                           width=1,
                                       ),
                                       opacity=0.9
                                   ),
                                   text=[train_keys[x] + "\n" + key_to_species[train_keys[x]] + "\n" +
                                         str(key_to_products[train_keys[x]]) for x in indices],
                                   ))

    layout = go.Layout(
        plot_bgcolor='rgba(0,0,0,0)',
        width=800,
        height=800,
        xaxis={"ticks": "", "showticklabels": False, 'showline': False, 'zeroline': False, 'showgrid': False},
        yaxis={"ticks": "", "showticklabels": False, 'showline': False, 'zeroline': False, 'showgrid': False},
    )
    f.close()
    return go.Figure(data, layout=layout)
import json
import argparse
import joblib
import os
import numpy as np
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sentence_transformers import SentenceTransformer
import sys


def log_cross_val_auc(clf, X, y, cv_splits, log_prefix):
    cv_auc = cross_val_score(clf, X, y, cv=cv_splits, scoring='roc_auc')
    cv_auc_mean = cv_auc.mean()
    cv_auc_error = cv_auc.std() * 2
    log = "{}_auc_cv: {:.5f} (+/- {:.5f})"
    print(log.format(log_prefix, cv_auc_mean, cv_auc_error))


def log_auc(clf, X, y, log_prefix):
    y_pred_proba = clf.predict_proba(X)
    auc = roc_auc_score(y, y_pred_proba[:, 1])
    log = '{}_auc: {:.5f}'
    print(log.format(log_prefix, auc))


def parse_args(sys_args):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=os.environ.get("SM_MODEL_DIR")
    )
    parser.add_argument(
        "--data-train",
        type=str,
        default=os.environ.get("SM_CHANNEL_TRAIN"),
    )
    parser.add_argument(
        "--data-test",
        type=str,
        default=os.environ.get("SM_CHANNEL_TEST")
    )
    parser.add_argument(
        "--numerical-feature-names",
        type=str
    )
    parser.add_argument(
        "--categorical-feature-names",
        type=str
    )
    parser.add_argument(
        "--textual-feature-names",
        type=str
    )
    parser.add_argument(
        "--label-name",
        type=str
    )

    args, _ = parser.parse_known_args(sys_args)
    print(args)
    return args


def load_data(data_dir, filename):
    with open(Path(data_dir, filename), 'r') as f:
        lines = f.readlines()
    data = [json.loads(line) for line in lines]
    return data


def extract_labels(
    data,
    label_name
):
    labels = []
    for sample in data:
        value = sample[label_name]
        labels.append(value)
    labels = np.array(labels).astype('int')
    return labels


def extract_numerical_features(
    sample,
    numerical_feature_names
):
    output = []
    for feature_name in numerical_feature_names:
        value = sample[feature_name]
        if value is None:
            value = np.nan
        output.append(value)
    return output


def extract_categorical_features(
    sample,
    categorical_feature_names
):
    output = []
    for feature_name in categorical_feature_names:
        value = sample[feature_name]
        if value is None:
            value = ""
        output.append(value)
    return output


def extract_textual_features(
    sample,
    textual_feature_names
):
    output = []
    for feature_name in textual_feature_names:
        value = sample[feature_name]
        if value is None:
            value = ""
        output.append(value)
    return output


def extract_features(
    data,
    numerical_feature_names,
    categorical_feature_names,
    textual_feature_names
):
    numerical_features = []
    categorical_features = []
    textual_features = []
    for sample in data:
        num_feat = extract_numerical_features(sample, numerical_feature_names)
        numerical_features.append(num_feat)
        cat_feat = extract_categorical_features(sample, categorical_feature_names)
        categorical_features.append(cat_feat)
        text_feat = extract_textual_features(sample, textual_feature_names)
        textual_features.append(text_feat)
    return numerical_features, categorical_features, textual_features


class BertEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, model_name='bert-base-nli-cls-token'):
        self.model = SentenceTransformer(model_name)
        self.model.parallel_tokenization = False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        output = []
        for sample in X:
            encodings = self.model.encode(sample)
            output.append(encodings)
        return output


def save_feature_names(
    numerical_feature_names,
    categorical_feature_names,
    textual_feature_names,
    filepath
):
    feature_names = {
        'numerical': numerical_feature_names,
        'categorical': categorical_feature_names,
        'textual': textual_feature_names
    }
    with open(filepath, 'w') as f:
        json.dump(feature_names, f)
        
        
def load_feature_names(filepath):
    with open(filepath, 'r') as f:
        feature_names = json.load(f)
    numerical_feature_names = feature_names['numerical']
    categorical_feature_names = feature_names['categorical']
    textual_feature_names = feature_names['textual']
    return numerical_feature_names, categorical_feature_names, textual_feature_names
    

def train_fn(args):
    # load data
    print('loading data')
    data = load_data(args.data_train, 'train.json')

    # parse feature names
    print('parsing feature names')
    numerical_feature_names = args.numerical_feature_names.split(',')
    categorical_feature_names = args.categorical_feature_names.split(',')
    textual_feature_names = args.textual_feature_names.split(',')
    print('saving feature names')
    save_feature_names(
        numerical_feature_names,
        categorical_feature_names,
        textual_feature_names,
        Path(args.model_dir, "feature_names.json")
    )
    
    # extract label
    print('extracting label')
    labels = extract_labels(
        data,
        args.label_name
    )

    # extract features
    print('extracting features')
    numerical_features, categorical_features, textual_features = extract_features(
        data,
        numerical_feature_names,
        categorical_feature_names,
        textual_feature_names
    )

    # define preprocessors
    print('defining preprocessors')
    numerical_transformer = SimpleImputer(missing_values=np.nan, strategy='mean', add_indicator=True)
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")
    textual_transformer = BertEncoder()

    # fit and save preprocessors
    print('fitting numerical_transformer')
    numerical_transformer.fit(numerical_features)
    print('saving categorical_transformer')
    joblib.dump(numerical_transformer, Path(args.model_dir, "numerical_transformer.joblib"))
    print('fitting categorical_transformer')
    categorical_transformer.fit(categorical_features)
    print('saving categorical_transformer')
    joblib.dump(categorical_transformer, Path(args.model_dir, "categorical_transformer.joblib"))

    # transform features
    print('transforming numerical_features')
    numerical_features = numerical_transformer.transform(numerical_features)
    print('transforming categorical_features')
    categorical_features = categorical_transformer.transform(categorical_features)
    print('transforming textual_features')
    textual_features = textual_transformer.transform(textual_features)

    # concat features
    print('concatenating features')
    categorical_features = categorical_features.toarray()
    textual_features = np.array(textual_features)
    textual_features = textual_features.reshape(textual_features.shape[0], -1)
    features = np.concatenate([
        numerical_features,
        categorical_features,
        textual_features
    ], axis=1)

    # define model
    print('instantiating model')
    classifier = RandomForestClassifier(
        n_estimators=args.n_estimators
    )

    # fit and save model
    print('fitting model')
    classifier = classifier.fit(features, labels)
    print('saving model')
    joblib.dump(classifier, Path(args.model_dir, "classifier.joblib"))


### DEPLOYMENT FUNCTIONS

def model_fn(model_dir):
    print('loading feature_names')
    numerical_feature_names, categorical_feature_names, textual_feature_names = load_feature_names(Path(model_dir, "feature_names.json"))
    print('loading numerical_transformer')
    numerical_transformer = joblib.load(Path(model_dir, "numerical_transformer.joblib"))
    print('loading categorical_transformer')
    categorical_transformer = joblib.load(Path(model_dir, "categorical_transformer.joblib"))
    print('loading textual_transformer')
    textual_transformer = BertEncoder()
    classifier = joblib.load(Path(model_dir, "classifier.joblib"))
    model_assets = {
        'numerical_feature_names': numerical_feature_names,
        'numerical_transformer': numerical_transformer,
        'categorical_feature_names': categorical_feature_names,
        'categorical_transformer': categorical_transformer,
        'textual_feature_names': textual_feature_names,
        'textual_transformer': textual_transformer,
        'classifier': classifier
    }
    return model_assets


def input_fn(request_body_str, request_content_type):
    assert (
        request_content_type == "application/json"
    ), "content_type must be 'application/json'"
    request_body = json.loads(request_body_str)
    return request_body


def predict_fn(request, model_assets):
    print('making batch')
    request = [request]
    print('extracting features')
    numerical_features, categorical_features, textual_features = extract_features(
        request,
        model_assets['numerical_feature_names'],
        model_assets['categorical_feature_names'],
        model_assets['textual_feature_names']
    )
    
    print('transforming numerical_features')
    numerical_features = model_assets['numerical_transformer'].transform(numerical_features)
    print('transforming categorical_features')
    categorical_features = model_assets['categorical_transformer'].transform(categorical_features)
    print('transforming textual_features')
    textual_features = model_assets['textual_transformer'].transform(textual_features)
    
    # concat features
    print('concatenating features')
    categorical_features = categorical_features.toarray()
    textual_features = np.array(textual_features)
    textual_features = textual_features.reshape(textual_features.shape[0], -1)
    features = np.concatenate([
        numerical_features,
        categorical_features,
        textual_features
    ], axis=1)
    
    print('predicting using model')
    prediction = model_assets['classifier'].predict_proba(features)
    probability = prediction[0][1].tolist()
    output = {
        'probability': probability
    }
    return output


def output_fn(prediction, response_content_type):
    assert (
        response_content_type == "application/json"
    ), "accept must be 'application/json'"
    response_body_str = json.dumps(prediction)
    return response_body_str


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    train_fn(args)

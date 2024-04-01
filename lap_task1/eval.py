from .files import files
from hartufo import CustomSphericalDataset
from hartufo import HrirSpec, SubjectSpec, CollectionSpec
from hartufo.transforms.hrir import Hrir3dTransform
from hartufo.sklearn import Flatten, DomainTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from pathlib import Path
import numpy as np


def classification_accuracy(base_dir, common_positions, side, sample_rate, hrir_length):
    datasets = []
    azimuths, elevations = zip(*common_positions)
    for collection_id, paths in files.items():
        print(f'Reading {collection_id}')
        datasets.append(CustomSphericalDataset(collection_id,
            [base_dir / collection_id / p for p in paths],
            features_spec=HrirSpec('time', side, azimuths, elevations, distance='farthest', samplerate=sample_rate, length=hrir_length, transform=Hrir3dTransform()),
            group_spec=SubjectSpec(), target_spec=CollectionSpec(),
        ))

    all_features = np.array([ex for d in datasets for ex in d.features])
    all_targets = np.array([ex for d in datasets for ex in d.target])
    all_groups = np.array([ex for d in datasets for ex in d.group])


    estimators = [
        ('domain', 'passthrough'),
        ('flatten', Flatten()),
        ('clf', 'passthrough'),
    ]
    pipe = Pipeline(estimators)

    param_grid = {
        'domain': [DomainTransformer('time'), DomainTransformer('magnitude'), DomainTransformer('magnitude_db')],
        'clf': [LinearSVC(dual='auto', max_iter=20000), SVC(), LogisticRegression(max_iter=500), DecisionTreeClassifier(max_depth=8)],
    }
    exp = GridSearchCV(pipe, param_grid, scoring='accuracy', cv=GroupKFold(n_splits=5), return_train_score=True, error_score='raise', n_jobs=-2).fit(all_features, all_targets, groups=all_groups)
    return exp.best_score_


def cli():
    import argparse
    from itertools import chain, product
    parser = argparse.ArgumentParser(description='Task 1 Evaluator')
    parser.add_argument('processed_dir', type=str, help='Path to the directory containing the processed SOFA files.')
    args = parser.parse_args()

    ## Config
    common_positions = chain(product(range(-180, 180, 10), [-30, 0, 30]), product(range(-180, 180, 20), [60]))
    side = 'any-left'
    sample_rate = 44100
    hrir_length = 235

    acc = classification_accuracy(Path(args.processed_dir), common_positions, side, sample_rate, hrir_length)
    print(acc)


if __name__ == '__main__':
    cli()

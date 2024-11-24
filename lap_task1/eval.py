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
from sklearn.metrics import ConfusionMatrixDisplay, make_scorer, confusion_matrix
from pathlib import Path
import numpy as np
import csv

N_SPLITS = 5


def classification_accuracy(base_dir, seed, common_positions, side, sample_rate, hrir_length, confmat_path, all_results_path):
    datasets = []
    azimuths, elevations = zip(*common_positions)
    for collection_id, paths in files.items():
        print(f'Reading {collection_id}')
        datasets.append(CustomSphericalDataset(collection_id,
            [base_dir / collection_id / p for p in paths],
            features_spec=HrirSpec('time', side, flat_positions=False, fundamental_angles=azimuths, orthogonal_angles=elevations, distance='farthest', samplerate=sample_rate, length=hrir_length, transform=Hrir3dTransform()),
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
        'clf': [LinearSVC(dual='auto', max_iter=20000, random_state=seed), SVC(random_state=seed), LogisticRegression(max_iter=500, random_state=seed), DecisionTreeClassifier(max_depth=8, random_state=seed)],
    }
    score_metrics = {
        'accuracy': 'accuracy',
    }
    if confmat_path:
        score_metrics['confusion_matrix'] = make_scorer(confusion_matrix, normalize=None)
    exp = GridSearchCV(pipe, param_grid, scoring=score_metrics, refit='accuracy', cv=GroupKFold(n_splits=N_SPLITS), return_train_score=True, error_score='raise', n_jobs=-2).fit(all_features, all_targets, groups=all_groups)
    if confmat_path:
        confmat = N_SPLITS*exp.cv_results_['mean_test_confusion_matrix'][exp.best_index_]
        confmat_display = ConfusionMatrixDisplay(confmat, display_labels=files.keys()).plot(xticks_rotation='vertical', cmap='Blues')
        confmat_display.figure_.savefig(confmat_path, bbox_inches='tight', pad_inches=0.05)
    if all_results_path:
        result_keys = [k for k in exp.cv_results_.keys() if not k.endswith('_confusion_matrix') and k != 'params']
        with open(all_results_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(result_keys)
            for row in zip(*[v for k, v in exp.cv_results_.items() if not k.endswith('_confusion_matrix')]):
                writer.writerow(row)
    mean_acc = exp.cv_results_['mean_test_accuracy'][exp.best_index_]
    std_acc = exp.cv_results_['std_test_accuracy'][exp.best_index_]
    return mean_acc, std_acc


def cli():
    import argparse
    from itertools import chain, product
    parser = argparse.ArgumentParser(description='Task 1 Evaluator')
    parser.add_argument('processed_dir', type=str, help='Path to the directory containing the processed SOFA files.')
    parser.add_argument('--seed', default=0, help='Fix the random seed that initialises the classifiers.')
    parser.add_argument('--confmat', nargs='?', type=str, default='', const='confmat.png', help='Save a PNG confusion matrix file to the given path.')
    parser.add_argument('--all-results', type=str, default='', help='Path to the CSV file containing results for all folds and classifiers.')
    args = parser.parse_args()

    ## Config
    common_positions = chain(product(range(-180, 180, 10), [-30, 0, 30]), product(range(-180, 180, 20), [60]))
    side = 'any-left'
    sample_rate = 44100
    hrir_length = 235

    mean_acc, std_acc = classification_accuracy(Path(args.processed_dir), args.seed, common_positions, side, sample_rate, hrir_length, args.confmat, args.all_results)
    print(f'{100*mean_acc:.3f} +/- {100*std_acc:.3f}')


if __name__ == '__main__':
    cli()

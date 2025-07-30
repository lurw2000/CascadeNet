"""A demo for detecting novelty with a model learned from given data.

"""
import pathlib

from sklearn.model_selection import train_test_split

from netml.ndm.model import MODEL
from netml.utils.tool import dump_data, load_data


RANDOM_STATE = 42

EXAMPLES_PATH = pathlib.Path(__file__).parent.parent

DATA_FILE = EXAMPLES_PATH / 'out' / 'data' / 'demo_IAT.dat'


def generate_model(model_name='GMM'):
    """Generate a model according to the given name.
    current implemented models are "OCSVM", "GMM",  "IF", "KDE", "PCA" and "AE"

    Parameters
    ----------
    model_name: string (default is 'OCSVM')
        the name of the model wants to be generated.

    Returns
    -------
        a MODEL instance

    """

    if model_name == 'OCSVM':
        from netml.ndm.ocsvm import OCSVM
        model = OCSVM(kernel='rbf', nu=0.5, random_state=RANDOM_STATE)
    elif model_name == 'GMM':
        from netml.ndm.gmm import GMM
        model = GMM(n_components=2, covariance_type='full', random_state=RANDOM_STATE)
    elif model_name == 'IF':
        from netml.ndm.iforest import IF
        model = IF(n_estimators=100, random_state=RANDOM_STATE)
    elif model_name == 'PCA':
        from netml.ndm.pca import PCA
        model = PCA(n_components=1, random_state=RANDOM_STATE)
    elif model_name == 'KDE':
        from netml.ndm.kde import KDE
        model = KDE(kernel='gaussian', bandwidth=1.0, random_state=RANDOM_STATE)
    elif model_name == 'AE':
        from netml.ndm.ae import AE
        model = AE(epochs=100, batch_size=32, random_state=RANDOM_STATE)
    else:
        msg = f'{model_name} is not implemented yet!'
        raise NotImplementedError(msg)

    model.name = model_name

    return model


def main(data_file=DATA_FILE):
    # load data
    X, y = load_data(data_file)
    # split train and test test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=RANDOM_STATE)
    print(f'X_train.shape: {X_train.shape}, X_test.shape: {X_test.shape}, y_train.shape: {y_train.shape}, '
          f'y_test.shape: {y_test.shape}')

    # model_name in ['OCSVM', 'KDE','IF', 'AE', 'GMM', 'PCA']
    model_name = 'OCSVM'
    print(f'model_name: {model_name}')
    # create detection model
    model = generate_model(model_name)

    ndm = MODEL(model, score_metric='auc', verbose=10, random_state=RANDOM_STATE)

    # learned the model from the train set
    ndm.train(X_train)

    # evaluate the learned model
    ndm.test(X_test, y_test)

    # dump data to disk
    out_file = data_file.parent / f'{ndm.model_name}-results.dat'
    dump_data((model, ndm.history), out_file=out_file)

    print(ndm.train.tot_time, ndm.test.tot_time, ndm.score)


if __name__ == '__main__':
    main()

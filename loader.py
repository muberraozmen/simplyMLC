""" DATA READERS
    Description: Functions to download and/or read data files
    Required output format:
    data = {'train': {'features': List[List[int]], 'labels': List[List[int]]},
            'test' : {'features': List[List[int]], 'labels': List[List[int]]},
            'vocab': {'id2feature': Dict[int, str], 'id2label': Dict[int, str]}
            }
    for the input features padding index is set to 0 by default
"""

__all__ = ['load']


def load(opt):
    import skmultilearn.dataset

    def _format(x, y):
        src = []
        tgt = []
        for i in range(x.shape[0]):
            src.append(list(x[i].nonzero()[1] + 1))
            tgt.append(list(y[i].nonzero()[1]))
        return src, tgt

    if opt['source'] == 'skmultilearn':
        x_train, y_train, feature_names, label_names = skmultilearn.dataset.load_dataset(opt['dataset'], 'train')
        x_test, y_test, feature_names_, label_names_ = skmultilearn.dataset.load_dataset(opt['dataset'], 'test')
    elif opt['source'] == 'local_mulan':
        x_train, y_train, feature_names, label_names = \
            skmultilearn.dataset.load_from_arff(opt['path_train'], label_count=opt['label_count'],
                                                label_location=opt['label_count'], load_sparse=opt['load_sparse'],
                                                return_attribute_definitions=True)
        x_test, y_test, feature_names, label_names = \
            skmultilearn.dataset.load_from_arff(opt['path_test'], label_count=opt['label_count'],
                                                label_location=opt['label_count'], load_sparse=opt['load_sparse'],
                                                return_attribute_definitions=True)
    else:
        NotImplementedError

    if feature_names != feature_names_ or label_names != label_names_:
        assert ValueError

    vocab_src = dict()
    vocab_src[0] = '<pad>'
    for idx in range(len(feature_names)):
        vocab_src[idx + 1] = feature_names[idx][0]

    vocab_tgt = dict()
    for idx in range(len(label_names)):
        vocab_tgt[idx] = label_names[idx][0]

    # list of dictionaries format
    train_src, train_tgt = _format(x_train, y_train)
    test_src, test_tgt = _format(x_test, y_test)

    data = {'train': {'features': train_src, 'labels': train_tgt},
            'test':  {'features': test_src, 'labels': test_tgt},
            'vocab': {'id2feature': vocab_src, 'id2label': vocab_tgt}}

    return data

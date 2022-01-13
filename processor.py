import torch
from torch.utils.data import Dataset
from typing import Dict, List
from torch.utils.data import DataLoader

__all__ = ['process']


def process(data, batch_size=32, shuffle=False, drop_last=True, device='cpu', padding_index=0, ** kwargs):

    train_split = MultiLabelDataset(
        data['train']['features'], data['train']['labels'], data['vocab']['id2feature'], data['vocab']['id2label'])
    test_split = MultiLabelDataset(
        data['test']['features'], data['test']['labels'], data['vocab']['id2feature'], data['vocab']['id2label'])

    num_features = train_split.num_features
    num_labels = train_split.num_labels

    train_loader = DataLoader(
        train_split, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
        collate_fn=BatchProcFn(num_labels=num_labels, device=device, padding_index=padding_index))

    test_loader = DataLoader(
        test_split, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
        collate_fn=BatchProcFn(num_labels=num_labels, device=device, padding_index=padding_index))

    return train_loader, test_loader, num_features, num_labels


class MultiLabelDataset(Dataset):
    def __init__(self, features: List[List[int]], labels: List[List[int]],
                 id2feature: Dict[int, str], id2label: Dict[int, str]):
        self.features = features
        self.labels = labels
        if len(self.features) != len(self.labels):
            assert ValueError
        self.id2feature = id2feature
        self.id2label = id2label
        self.num_labels = len(id2label)
        self.num_features = len(id2feature)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        features = self.features[index]
        labels = self.labels[index]
        return features, labels

    def id_to_label(self, ids):
        labels = []
        for id in ids:
            labels.append(self.id2label[id])
        return labels

    def id_to_feature(self, ids):
        inputs = []
        for id in ids:
            inputs.append(self.id2feature[id])
        return inputs


class BatchProcFn:
    def __init__(self, num_labels, device='cpu', padding_index=0):
        self.num_labels = num_labels
        self.padding_index = padding_index
        self.device = device

    def __call__(self, batch):
        features = []
        labels = []
        for x, y in batch:
            seq = self.pad_to_longest(x, max(len(inst[0]) for inst in batch))
            features.append(seq)
            labels.append(y)
        features = torch.as_tensor(features, dtype=torch.long, device=self.device)
        labels = torch.as_tensor(self.seq2bin(labels), dtype=torch.float, device=self.device)
        return features, labels

    def pad_to_longest(self, seq, max_len):
        seq = seq + [self.padding_index] * (max_len - len(seq))
        return seq

    def seq2bin(self, seq):
        y = torch.zeros((len(seq), self.num_labels))
        for i in range(len(seq)):
            y[i, seq[i]] = 1
        return y


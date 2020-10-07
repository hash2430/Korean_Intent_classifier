from torch.utils.data import Dataset
import json
from torch.nn.utils.rnn import pad_sequence
import torch
import glob
# Read txt from Main and return input_ids without label
class TestDataset(Dataset):
    def __init__(self, tokenizer, path='/mnt/sdd1/selvas_txt/Main/script/*.txt'):
        paths = glob.glob(path)
        self.tokenizer = tokenizer
        self.lines = []
        for path in paths:
            with open(path, 'r', encoding='utf-8-sig') as f:
                line = f.readline()
            self.lines.append(line)

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        text = self.lines[idx]
        input_ids = self.tokenizer.encode(text)
        return text, input_ids

def collate_test_data(samples):
    texts, input_ids = zip(*samples)
    attention_mask = [[1] * len(input_id) for input_id in input_ids]

    input_ids = pad_sequence([torch.Tensor(input_id).to(torch.long) for input_id in input_ids],
                             padding_value=0, batch_first=True)
    attention_mask = pad_sequence([torch.Tensor(mask).to(torch.long) for mask in attention_mask],
                                  padding_value=0, batch_first=True)

    return texts, input_ids, attention_mask

class IntentDataset(Dataset):
    def __init__(self, txt_file='/mnt/sdd1/text/sae4k/sae4k_v2.txt'):
        self.lines = []
        with open(txt_file, 'r', encoding='utf-8-sig') as f:
            self.lines = f.readlines()

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        items = self.lines[idx].rstrip().split('\t')
        label = int(items[0])
        text = items[1]
        sample = {'text':text, 'label':label}
        return sample

class EmotionDataset(Dataset):
    def __init__(self, json_file='data/train.json'):
        with open(json_file, 'r', encoding='utf-8-sig') as f:
            json_data = json.load(f)
        self.rows = json_data
        self.texts = []
        self.labels = []

        for row in self.rows:
            label = row['label']
            text = row['text']

            self.texts.append(text)
            self.labels.append(label)
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        sample = {'text':text, 'label':label}
        return sample

class IntentFeatureDataset(Dataset):
    def __init__(self, dataset, tokenizer, return_sample=False, eval=False):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.return_sample = return_sample
        self.eval = eval

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        text = sample['text']
        label = sample['label']

        input_ids = self.tokenizer.encode(text)
        return input_ids, label

class EmotionFeatureDataset(Dataset):
    def __init__(self, dataset, tokenizer, return_sample=False, eval=False):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.return_sample = return_sample
        self.eval = eval
        self.label_map={'neutral':0, 'happy':1, 'sad':2, 'angry':3}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        text = sample['text']
        label = sample['label']

        input_ids = self.tokenizer.encode(text)
        label = self.label_map[label]
        return input_ids, label
# List of tuple of list and int
def collate_fn(samples):
    input_ids, labels = zip(*samples)
    attention_mask = [[1] * len(input_id) for input_id in input_ids]

    input_ids = pad_sequence([torch.Tensor(input_id).to(torch.long) for input_id in input_ids],
                             padding_value=0, batch_first=True)
    attention_mask = pad_sequence([torch.Tensor(mask).to(torch.long) for mask in attention_mask],
                                  padding_value=0, batch_first=True)

    labels = torch.LongTensor([labels]).squeeze(0)
    return input_ids, attention_mask, labels
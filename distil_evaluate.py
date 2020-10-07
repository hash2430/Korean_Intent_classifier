from typing import List, Tuple
import random
import torch
from sklearn.metrics import confusion_matrix
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Sampler
from model import BertForSquad
from distil_dataset import EmotionDataset, EmotionFeatureDataset, collate_fn, IntentDataset, IntentFeatureDataset
from distil_model import EmotionDistilBert, EmotionBert, IntentionDistilBert
from tokenization_kobert import KoBertTokenizer
from transformers import DistilBertModel, DistilBertConfig
from tqdm import tqdm, trange
tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert')
device = torch.device("cpu")
def inference(model_type, checkpoint_path):
    batch_size=8

    # dataset = EmotionFeatureDataset(EmotionDataset(json_file='/mnt/sdd1/text/emotion_data_400/test.json'), tokenizer)
    dataset = IntentFeatureDataset(IntentDataset('/home/admin/projects/graduate/DistilBERT_intent_classifier_3i4k/3i4k/fci_test.txt'), tokenizer)

    if model_type == 'BERT':
        raise NotImplementedError
    else:
        model = IntentionDistilBert.from_pretrained(checkpoint_path).to(device)

    train_loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=batch_size, collate_fn=collate_fn, drop_last=False)
    print(len(dataset))
    train_iterator = tqdm(train_loader, leave=False)
    score = 0
    cnt = 0
    logits = torch.FloatTensor(6121,7)
    gt_labels = torch.LongTensor(6121)
    pointer = 0
    for data in train_iterator:
        input_ids, attention_mask, labels = data
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        intention_predict, _ = model(input_ids, attention_mask=attention_mask)
        logits[pointer:pointer+len(intention_predict)] = intention_predict
        gt_labels[pointer:pointer+len(intention_predict)] = labels
        pointer += len(intention_predict)

        intention_predict = torch.argmax(intention_predict, dim=1)
        bools =  (intention_predict - labels)
        bools = (bools == 0).sum()
        score += bools
        cnt += len(intention_predict)
    score = float(score)/float(cnt)
    m = {'predict': logits.cpu(), 'labels': gt_labels.cpu()}
    torch.save(m, 'analysis/intention_embedding_predict_and_labels.pt')
    print(cnt)
    print(score)

    predicted = torch.argmax(logits,dim=1)
    c_mat = confusion_matrix(predicted, gt_labels, normalize='all')
    torch.save(c_mat, 'analysis/confusion_matrix_normalize_all.pt')
    print()
if __name__ == '__main__':
    checkpoint_path = 'checkpoint_intent_3'
    model_type = 'DistilBERT'
    inference(model_type, checkpoint_path)
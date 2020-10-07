from typing import List, Tuple
import random
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Sampler
from model import BertForSquad
from distil_dataset import EmotionDataset, EmotionFeatureDataset, collate_fn, IntentDataset, IntentFeatureDataset
from distil_model import EmotionDistilBert, EmotionBert, IntentionDistilBert
from tokenization_kobert import KoBertTokenizer
from transformers import DistilBertModel, DistilBertConfig
from tqdm import tqdm, trange
tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert')
tmp = tokenizer.tokenize("[CLS] 곧바로 다운 받아서 사용할 수 있습니다. [SEP]")
print(tmp)
device = torch.device("cuda:3")
def train():
    epochs = 3
    learning_rate = 5e-5
    batch_size=64

    dataset = IntentFeatureDataset(IntentDataset('3i4k/fci_train.txt'), tokenizer)
    model = IntentionDistilBert.from_pretrained('monologg/distilkobert').to(device)
    # model = EmotionBert.from_pretrained('monologg/kobert').to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=batch_size, collate_fn=collate_fn)
    train_iterator = tqdm(train_loader, leave=False)
    loss_fct = torch.nn.CrossEntropyLoss()
    model.save_pretrained('./checkpoint_intent')
    for epoch in range(epochs):
        print('epoch:{}'.format(epoch))
        for data in train_iterator:
            optimizer.zero_grad()
            input_ids, attention_mask, labels = data
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            emotion_predict, _ = model(input_ids, attention_mask=attention_mask)
            loss = loss_fct(emotion_predict, labels)
            loss.backward()
            optimizer.step()
            print(loss.item())

    model.save_pretrained('./checkpoint_intent_3')
if __name__ == '__main__':
    train()
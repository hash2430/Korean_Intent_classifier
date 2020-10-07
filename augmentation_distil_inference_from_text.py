from typing import List, Tuple
import random
import torch
import os
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Sampler
from model import BertForSquad
from distil_dataset import EmotionDataset, EmotionFeatureDataset, collate_fn, TestDataset, collate_test_data
from distil_model import EmotionDistilBert
from tokenization_kobert import KoBertTokenizer
from transformers import DistilBertModel, DistilBertConfig
from tqdm import tqdm, trange
tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert')
device = torch.device("cuda:2")
save_dir = '/mnt/sdd1/text/aug_selvas_label'
label_map = ['neutral.txt','happy.txt','sad.txt','angry.txt']
save_paths = []
for lab in label_map:
    path = os.path.join(save_dir, lab)
    save_paths.append(path)
# Save according to the label
def inference(checkpoint_path):
    batch_size=8

    # dataset = EmotionFeatureDataset(EmotionDataset(json_file='/mnt/sdd1/text/emotion_data_400/test.json'), tokenizer)
    dataset = TestDataset(tokenizer)
    model = EmotionDistilBert.from_pretrained(checkpoint_path).to(device)


    train_loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=batch_size, collate_fn=collate_test_data, drop_last=False)
    print(len(dataset))
    train_iterator = tqdm(train_loader, leave=False)
    cnt = 0
    for data in train_iterator:
        texts, input_ids, attention_mask = data
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        emotion_predict,_ = model(input_ids, attention_mask=attention_mask)
        emotion_predict = torch.argmax(emotion_predict, dim=1)

        for i in range(len(texts)):
            save_path = save_paths[emotion_predict[i]]
            if os.path.exists(save_path):
                with open(save_path, 'a', encoding='utf-8') as f:
                    f.write(texts[i])
            else:
                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write(texts[i])


    print(cnt)
if __name__ == '__main__':
    checkpoint_path = 'Acc3/checkpoint_30_epochs'
    inference(checkpoint_path)
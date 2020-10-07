from transformers import DistilBertModel, DistilBertPreTrainedModel, DistilBertConfig, BertModel, BertPreTrainedModel
import torch
class IntentionDistilBert(DistilBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = DistilBertModel.from_pretrained('monologg/distilkobert', output_hidden_states=True, output_attentions=True)
        self.seq_len = config.max_length
        self.linear = torch.nn.Linear(config.hidden_size, 7)

    def forward(self, input_ids, attention_mask):
        # output: [last_hidden_states, (word_embedding, layer1_output, layer2_output, layer_3output)]
        output = self.bert(input_ids, attention_mask=attention_mask)
        last_hidden_layer = output[0]
        all_hidden_layers = output[1]
        attentions=output[2]
        # I wish this returned attention and other hidden layers for further analysis,
        # but this pretrained model does not provide that.
        interested_probing_layer = 3
        pooled = all_hidden_layers[interested_probing_layer][:,0,:] # batch, length, hidden_dim
        linear_output = self.linear(pooled)
        return linear_output, all_hidden_layers


class EmotionDistilBert(DistilBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = DistilBertModel.from_pretrained('monologg/distilkobert', output_hidden_states=True, output_attentions=True)
        self.seq_len = config.max_length
        self.linear = torch.nn.Linear(config.hidden_size, 4)

        # Erasing below line gives 7 to 9 %p performance gain
        # self.init_weights()

    def forward(self, input_ids, attention_mask):
        # output: [last_hidden_states, (word_embedding, layer1_output, layer2_output, layer_3output)]
        output = self.bert(input_ids, attention_mask=attention_mask)
        last_hidden_layer = output[0]
        all_hidden_layers = output[1]
        attentions=output[2]
        # I wish this returned attention and other hidden layers for further analysis,
        # but this pretrained model does not provide that.
        pooled = last_hidden_layer[:,0,:] # batch, length, hidden_dim
        linear_output = self.linear(pooled)
        return linear_output, all_hidden_layers

class EmotionBert(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel.from_pretrained('monologg/kobert', output_hidden_states=True, output_attentions=True)
        self.seq_len = config.max_length
        self.linear = torch.nn.Linear(config.hidden_size, 4)

        # Erasing below line gives 7 to 9 %p performance gain
        # self.init_weights()

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids, attention_mask=attention_mask)
        last_hidden_layer = output
        # Bert returning pooled_last_hidden_states is not just last layer output at [CLS].
        # It went through additional dense layer
        last_hideen_states, pooled_last_hidden_states, all_hidden_states, attention_weights = output[0], output[1], output[2], output[3]
        own_pooling = last_hideen_states[:,0,:]
        linear_output = self.linear(own_pooling)
        return linear_output, all_hidden_states
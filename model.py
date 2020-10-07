import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertPreTrainedModel, DistilBertModel, BertModel, BertConfig

class BertForSquad(BertPreTrainedModel):
    """ BERT model for Squad dataset
    Implement proper a question and answering model based on BERT.
    We are not going to check whether your model is properly implemented.
    If the model shows proper performance, it doesn't matter how it works.

    BertPretrinedModel Examples:
    https://huggingface.co/transformers/_modules/transformers/modeling_bert.html#BertForQuestionAnswering
    """
    def __init__(self, config: BertConfig):
        """ Model Initializer
        You can declare and initialize any layer if you want.
        """
        super().__init__(config)
        # self.bert = BertModel(config)
        self.bert = DistilBertModel.from_pretrained('monologg/distilkobert')
        self.seq_len = config.max_length
        self.qa_outputs = nn.Linear(config.hidden_size, 2)

        # Don't forget initializing the weights
        self.init_weights()

    def forward(
        self,
        input_ids,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor
    ):
        """ Model Forward Function
        There is no format for the return values.
        However, the input must be in the prescribed form.

        Arguments:
        input_ids -- input_ids is a tensor
                    in shape (batch_size, sequence_length)
        attention_mask -- attention_mask is a tensor
                    in shape (batch_size, sequence_length)
        token_type_ids -- token_type ids is a tensor
                    in shape (batch_size, sequence_length)

        Returns:
        FREE-FORMAT
        """
        outputs = self.bert(input_ids, attention_mask=attention_mask,
                                                        token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        start_logits = torch.nn.functional.softmax(start_logits, -1)
        end_logits = torch.nn.functional.softmax(end_logits, -1)
        print (max(start_logits[0]))
        print(max(end_logits[0]))


        return start_logits, end_logits
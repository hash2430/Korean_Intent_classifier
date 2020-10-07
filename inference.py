from typing import List
import json

import torch
import torch.nn.functional as F

### YOUR LIBRARIES HERE
import re
from train import squad_feature_collate_fn
### END YOUR LIBRARIES

from model import BertForSquad
from dataset import squad_features, SquadDataset, SquadFeatureDataset
from evaluation import evaluate

from transformers import BertTokenizerFast
from tqdm import trange

##### CHANGE BELOW OPTIONS IF IT IS DIFFENT FROM YOUR OPTIONS #####
device = torch.device('cuda:2') if torch.cuda.is_available() else torch.device('cpu')
bert_type = 'bert-base-uncased' 

def inference_start_end(
    start_probs: torch.Tensor,
    end_probs: torch.Tensor,
    context_start_pos: int,
    context_end_pos: int
):
    """ Inference fucntion for the start and end token position.

    Find the start and end positions of the answer which maximize
    p(start, end | context_start_pos <= start <= end <= context_end_pos)

    Note: assume that p(start) and p(end) are independent.

    Hint: torch.tril or torch.triu function would be helpful.

    Arguments:
    start_probs -- Probability tensor for the start position
                    in shape (sequence_length, )
    end_probs -- Probatility tensor for the end position
                    in shape (sequence_length, )
    context_start_pos -- Start index of the context
    context_end_pos -- End index of the context
    """
    tmp = start_probs.sum()
    assert start_probs.sum().allclose(torch.scalar_tensor(1.).to(tmp.device))
    assert end_probs.sum().allclose(torch.scalar_tensor(1.).to(tmp.device))

    start_pos: int = None
    end_pos: int = None
    p_start = start_probs[context_start_pos:context_end_pos+1]
    p_end = end_probs[context_start_pos:context_end_pos+1]
    outter_prod = torch.ger(p_start, p_end)
    p_upper = torch.triu(outter_prod)
    p_upper = torch.flatten(p_upper)
    idx = p_upper.argmax()
    start_pos = context_start_pos + idx // len(p_start)
    end_pos = context_start_pos + idx % len(p_end)

    return start_pos, end_pos

def inference_answer(
    question: str,
    context: str,

    input_ids: List[int],
    token_type_ids: List[int],
    start_pos: int,
    end_pos: int,

    tokenizer: BertTokenizerFast
) -> str:
    """ Inference fucntion for the answer.

    Because the tokenizer lowers the capital letters and splits punctuation marks,
    you may get wrong answer words if you detokenize it directly.
    For example, if you encode "$5.000 Dollars" and decode it, you get different words from the orignal.

    "$5.00 USD" --(Tokenize)--> ["$", "5", ".", "00", "usd"] --(Detokenize)--> "$ 5. 00 usd"

    Thus, you should find the original words in the context by the start and end token positions of the answer.
    Implement the function inferencing the answer from the context and the answer token postion.

    Note 1: We have already implmented direct decoding so you can skip this problem if you want.

    Note 2: When we implement squad_feature, we have arbitrarily split tokens if the answer is a subword,
            so it is very tricky to extract the original word by start_pos and end_pos.`
            However, as None is entered into the answer when evaluating,
            you can assume the word tokens follow general tokenizing rule in this problem.
            In fact, the most appropriate solution is storing the character index when tokenizing them.

    Hint: You can find a simple solution if you carefully search the documentation of the transformers library.
    Library Link: https://huggingface.co/transformers/index.html

    Arguments:
    question -- Question string
    context -- Context string

    input_ids -- Input ids
    token_type_ids -- Token type ids
    start_pos -- Predicted start token position of the answer
    end_pos -- Predicted end token position of the answer

    tokenizer -- Tokenizer to encode and decode the string

    Return:
    answer -- Answer string
    """
    tokens = tokenizer.tokenize(context)
    context_as_is = context
    context = context.lower()
    token2char_map = {}
    start = 0
    for j in range(len(tokens)):
        for i in range(len(tokens[j])):
            if tokens[j][i] == '#':
                continue
            else:
                break
        token = tokens[j][i:]
        start = context.find(token,start)
        end = start + len(token)
        token2char_map[j] = [start, end-1]
        start = end
    question_tokens = ['[CLS]']+tokenizer.tokenize(question)+['[SEP]']
    start = token2char_map[int(start_pos)-len(question_tokens)][0]
    end = token2char_map[int(end_pos)-len(question_tokens)][1]
    answer = context_as_is[start:end+1]

    return answer

def inference_model(
    model: BertForSquad,
    tokenizer: BertTokenizerFast,

    context: str,
    question: str,
    input_ids: List[int],
    token_type_ids: List[int]
) -> str:
    """ Inferene function with the model 
    Because we don't know how your model works, we can't not infer the answer from your model.
    Implement inference process for you model.
    Please use inference_start_end and inference_answer functions you have implemented
    
    Argumentes:
    model -- Model you have trained.
    tokenizer -- Tokenizer to encode and decode the string
    context -- Context string
    question -- Question string
    input_ids -- Input ids
    token_type_dis -- Token type ids

    Return:
    answer -- Answer string
    """
    answer: str = None
    tuple = input_ids, token_type_ids, -1, -1
    tmp = [tuple]
    input_ids, attention_mask, token_type_ids, _, _ = squad_feature_collate_fn(tmp)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    token_type_ids = token_type_ids.to(device)
    start, end = model(input_ids, attention_mask, token_type_ids)
    context_start = len(tokenizer.tokenize(question)) + 2
    context_end = len(input_ids[0])-2
    start, end = inference_start_end(start[0], end[0],context_start, context_end)
    answer = inference_answer(question, context, input_ids, token_type_ids, start.to('cpu'), end.to('cpu'), tokenizer)

    return answer 

#############################################
# Testing functions below.                  #
#############################################

def test_inference_start_end(tokenizer):
    print("======Start End Inference Test Case======")
    
    # First test
    input_tokens = ['[CLS]', 'this', 'is', 'a', 'question', '.', '[SEP]', 'this', 'is', 'an', 'answer', '.', '[SEP]']
    start_probs  = [    0.0,    0.0,  0.0, 0.0,        0.0, 0.0,     0.0,    0.0,  0.1,  0.8,      0.1, 0.0,     0.0]
    end_probs    = [    0.0,    0.0,  0.0, 0.0,        0.0, 0.0,     0.0,    0.0,  0.0,  0.1,      0.8, 0.1,     0.0]
    context_start_pos = input_tokens.index('[SEP]') + 1
    context_end_pos = len(input_tokens) - 2

    start_probs = torch.Tensor(start_probs)
    end_probs = torch.Tensor(end_probs)
    start_pos, end_pos = inference_start_end(start_probs, end_probs, context_start_pos, context_end_pos)
    answer = input_tokens[start_pos: end_pos+1] 

    assert answer == ['an', 'answer'], \
        "Your infered position is different from the expected position."
    
    print("The first test passed!")

    # Second test
    input_tokens = ['[CLS]', 'this', 'is', 'a', 'question', '.', '[SEP]', 'this', 'is', 'an', 'answer', '.', '[SEP]']
    start_probs  = [    0.0,    0.0,  0.0, 0.0,        0.0, 0.0,     0.0,    0.0,  0.1,  0.8,      0.1, 0.0,     0.0]
    end_probs    = [    0.0,    0.0,  0.0, 0.0,        0.0, 0.0,     0.0,    0.0,  0.6,  0.1,      0.3, 0.0,     0.0]
    context_start_pos = input_tokens.index('[SEP]') + 1
    context_end_pos = len(input_tokens) - 2

    start_probs = torch.Tensor(start_probs)
    end_probs = torch.Tensor(end_probs)
    start_pos, end_pos = inference_start_end(start_probs, end_probs, context_start_pos, context_end_pos)
    answer = input_tokens[start_pos: end_pos+1] 

    assert answer == ['an', 'answer'], \
        "Your infered position is different from the expected position."
    print("The second test passed!")

    # third test
    input_tokens = ['[CLS]', 'this', 'is', 'a', 'question', '.', '[SEP]', 'this', 'is', 'an', 'answer', '.', '[SEP]']
    start_probs  = [    0.0,    0.0,  0.0, 0.0,        0.0, 0.0,     0.0,    0.1,  0.2,  0.3,      0.1, 0.3,     0.0]
    end_probs    = [    0.0,    0.0,  0.0, 0.0,        0.0, 0.0,     0.0,    0.4,  0.2,  0.1,      0.2, 0.1,     0.0]
    context_start_pos = input_tokens.index('[SEP]') + 1
    context_end_pos = len(input_tokens) - 2

    start_probs = torch.Tensor(start_probs)
    end_probs = torch.Tensor(end_probs)
    start_pos, end_pos = inference_start_end(start_probs, end_probs, context_start_pos, context_end_pos)
    answer = input_tokens[start_pos: end_pos+1] 

    assert answer == ['an', 'answer'], \
        "Your infered position is different from the expected position."
    print("The third test passed!")

    # forth test
    input_tokens = ['[CLS]', 'this', 'is', 'a', 'question', '.', '[SEP]', 'this', 'is', 'an', 'answer', '.', '[SEP]']
    start_probs  = [    0.0,    0.0,  0.0, 0.0,        0.0, 0.3,     0.3,    0.0,  0.1,  0.2,      0.1, 0.0,     0.0]
    end_probs    = [    0.0,    0.0,  0.0, 0.0,        0.0, 0.0,     0.0,    0.0,  0.2,  0.0,      0.2, 0.0,     0.6]
    context_start_pos = input_tokens.index('[SEP]') + 1
    context_end_pos = len(input_tokens) - 2

    start_probs = torch.Tensor(start_probs)
    end_probs = torch.Tensor(end_probs)
    start_pos, end_pos = inference_start_end(start_probs, end_probs, context_start_pos, context_end_pos)
    answer = input_tokens[start_pos: end_pos+1] 

    assert answer == ['an', 'answer'], \
        "Your infered position is different from the expected position."
    print("The forth test passed!")

    print("All 4 test passed!")

def test_inference_answer(tokenizer):
    print("======Answer Inference Test Case======")

    # First test
    context = "The example answer was $5.00 USD."
    question = "What was the answer?"
    answer = "$5.00 USD"
    start_pos = context.find(answer)

    input_ids, token_type_ids, start_pos, end_pos = squad_features(context, question, answer, start_pos, tokenizer)
    prediction = inference_answer(question, context, input_ids, token_type_ids, start_pos, end_pos, tokenizer)
    
    if prediction == "$ 5. 00 usd":
        print("Skip the test. You get no score.")
        return

    assert prediction == answer, \
        "Your answer is different from the expected answer."

    print("The first test passed!")

    # Second test
    context = "The speed of the light is 299,794,458 m/s."
    question = "What is the speed of the light?"
    answer = "299,794,458 m/s"
    start_pos = context.find(answer)

    input_ids, token_type_ids, start_pos, end_pos = squad_features(context, question, answer, start_pos, tokenizer)
    prediction = inference_answer(question, context, input_ids, token_type_ids, start_pos, end_pos, tokenizer)

    assert prediction == answer, \
        "Your answer is different from the expected answer."

    print("The second test passed!")

    print("All 2 test passed!")

#############################################
# Analysis functions below.                  #
#############################################

def qualitative_analysis(tokenizer, model):
    print("======Qualitative Analysis======")

    question = 'Which NFL team represented the AFC at Super Bowl 50?'
    context = 'Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 2410 to earn their third Super Bowl title. The game was played on February 7, 2016, at Levi\'s Stadium in the San Francisco Bay Area at Santa Clara, California. As this was the 50th Super Bowl, the league emphasized the "golden anniversary" with various gold-themed initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals (under which the game would have been known as "Super Bowl L"), so that the logo could prominently feature the Arabic numerals 50.'
    plausible_answers = ['Denver Broncos']

    # question = 'Which Carolina Panthers player was named Most Valuable Player?'
    # context = 'The Panthers finished the regular season with a 151 record, and quarterback Cam Newton was named the NFL Most Valuable Player (MVP). They defeated the Arizona Cardinals 4915 in the NFC Championship Game and advanced to their second Super Bowl appearance since the franchise was founded in 1995. The Broncos finished the regular season with a 124 record, and denied the New England Patriots a chance to defend their title from Super Bowl XLIX by defeating them 2018 in the AFC Championship Game. They joined the Patriots, Dallas Cowboys, and Pittsburgh Steelers as one of four teams that have made eight appearances in the Super Bowl.'
    # plausible_answers = ['Cam Newton']

    # question = 'When is the final exam?'
    # context = 'At first, we thought there was going to be a final exam with no doubt. However, due to the recent outbreak of Corona virus, we considered maybe canceling the final exam. Some people also suggested postponing the exam from November 11th to December 3rd. After careful consideration we decided to have final exam on next tuesday, December 3rd 5:00PM at the great auditorium even though most of the students voted for taking final exam at November 11th as notified at the beginning of the semester.'
    # plausible_answers = ['December 3rd 5:00PM', 'next tuesday, December 3rd 5:00PM', 'next tuesday, 5:00PM']
    #
    # context = 'Although the model finds the correct start and end positions of the answer phase, it does not mean that the model successfully finds the exactly matching words in the context. The tokenizer converts all characters to lowercase and splits all special characters. For this reason, it is impossible to restore the original sentence from the tokens. We need a function that finds the exact matched words in the context when the start and end token positions are given. Implement inference_answer in inference.py.'
    # question = 'Where do we implement the inference_answer?'
    # plausible_answers = ['inference.py']
    # question = 'How many fumbles did Von Miller force in Super Bowl 50?'
    # context = "The Broncos took an early lead in Super Bowl 50 and never trailed. Newton was limited by Denver's defense, which sacked him seven times and forced him into three turnovers, including a fumble which they recovered for a touchdown. Denver linebacker Von Miller was named Super Bowl MVP, recording five solo tackles, 2½ sacks, and two forced fumbles."
    # plausible_answers = ['2']
    start_pos = context.find(plausible_answers[0])
    input_ids, token_type_ids, _, _ = squad_features(context, question, plausible_answers[0], start_pos, tokenizer)

    prediction = inference_model(model, tokenizer, context, question, input_ids, token_type_ids)

    print(f"Context: {context}")
    print(f"Question: {question}")
    print(f"Plausible Answers: {plausible_answers}")
    print(f"Prediction: {prediction}")

def quantative_analysis(tokenizer, model):
    print("======Quantitative Analysis======")
    dataset = SquadDataset('data/dev-v1.1.json')
    dataset = SquadFeatureDataset(dataset, bert_type=bert_type, lazy=True, return_sample=True, eval=True)

    answers = dict()

    for index in trange(len(dataset), desc="Answering"):
        (input_ids, token_type_ids, _, _), sample = dataset[index]
        answers[sample['id']] = \
                inference_model(model, tokenizer, sample['context'], sample['question'], input_ids, token_type_ids)

    with open('dev-v1.1-answers.json', mode='w') as f:
        json.dump(answers, f)

    with open('data/dev-v1.1.json', mode='r') as f:
        dataset = json.load(f)['data']

    results = evaluate(dataset, answers)
    print(f"Exact Match: {results['exact_match']}.")
    print(f"F1 score: {results['f1']}.")

if __name__ == "__main__":
    tokenizer = BertTokenizerFast.from_pretrained(bert_type)

    test_inference_start_end(tokenizer)
    test_inference_answer(tokenizer)

    model = BertForSquad.from_pretrained('./checkpoint')
    model.to(device)
    model.eval()

    qualitative_analysis(tokenizer, model)
    # quantative_analysis(tokenizer, model)

from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorForSeq2Seq
import evaluate
import numpy as np

#load dataset
books = load_dataset("opus_books", "en-fr")

#seperate into training and testing data
books = books["train"].train_test_split(test_size=0.2)

#visualize
print(books["train"][0])

#load tokenizer to process language pairs
checkpoint = "google-t5/t5-small"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

#set source and target languages
source_lang = "en"
target_lang = "fr"
prefix = "translate English to French: "

def preprocess_function(examples):

    inputs = ["translate English to French: " + example[source_lang] for example in examples["translation"]]
    targets = [example[target_lang] for example in examples["translation"]]
    model_inputs = tokenizer(inputs, text_target=targets, max_length=128, truncation=True)
    return model_inputs

#apply preprocessing to entire dataset
tokenized_books = books.map(preprocess_function, batched=True)

#dynamically pads sentences to longest length of batch (rather than longest length of dataset)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)

#set evaluation metric
metric = evaluate.load("sacrebleu")


#??? not actually sure what this does
def postprocess_text(preds, labels):

    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels

#compute translation metrics
def compute_metrics(eval_preds):

    preds, labels = eval_preds

    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}

    return result


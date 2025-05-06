from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorForSeq2Seq
import evaluate
import numpy as np

from torch import Tensor
from sentence_transformers import SentenceTransformer, util #Semantic similarity
from thefuzz import fuzz #Character level similarity 

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




#Character level differences of best matching substring, so 'hello world' vs 'hello world!' would return 1
#A very low ratio means low similarity
def compareChars(string1, string2):
    """
    Returns the Character level differences of best matching substring, as a ratio between 0 and 1
    (ex: compareChars('hello world', 'hello world!') = 1)
    """
    ratio = fuzz.partial_ratio(string1, string2) / 100

    return ratio

def compareSentences(string1, string2):
    """
    Returns the semantic similarity of two strings as a ratio between 0 and 1
    """
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embedding_1: Tensor = model.encode(string1, convert_to_tensor=True)
    embedding_2: Tensor = model.encode(string2, convert_to_tensor=True)

    ratio = util.pytorch_cos_sim(embedding_1, embedding_2)[0][0].item()

    return ratio

string1 = "Yes, yes."
string2 = "--Oui … oui …"
charSimilarity = compareChars(string1, string2)
sentenceSimilarity = compareSentences(string1, string2)

print("charSimilarity", charSimilarity)
print("sentenceSimilarity", sentenceSimilarity)


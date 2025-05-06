from transformers import AutoTokenizer, M2M100ForConditionalGeneration
from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np
import pandas as pd
from torch import Tensor
from sentence_transformers import SentenceTransformer, util #Semantic similarity
from thefuzz import fuzz #Character level similarity 

model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
tokenizer = AutoTokenizer.from_pretrained("facebook/m2m100_418M")

def translate(inputString, outputLang):
    """NOTE: ADD DOCUMENTATION"""
    model_inputs = tokenizer(inputString, return_tensors="pt")
    translation = model.generate(**model_inputs, forced_bos_token_id=tokenizer.get_lang_id(outputLang))
    return tokenizer.batch_decode(translation, skip_special_tokens=True)

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

def compareLangs(lang1, lang2, inputString):
    """NOTE: ADD DOCUMENTATION"""

    charAccs = []
    sentenceAccs = []
    translation1 = translate(inputString, lang2)
    translations = [inputString, translation1]

    count = 1

    while len(translations) < 11:
        #to language 2
        translation = translate(translations[len(translations) - 1], lang1)
        translations.append(translation)

        charAcc = compareChars(translations[len(translations) - 3], translation)
        sentenceAcc = compareSentences(translations[len(translations) - 3], translation)
        charAccs.append(charAcc)
        sentenceAccs.append(sentenceAcc)

        #to language 1
        translation = translate(translations[len(translations) - 1], lang2)
        translations.append(translation)

        charAcc = compareChars(translations[len(translations) - 3], translation)
        sentenceAcc = compareSentences(translations[len(translations) - 3], translation)
        charAccs.append(charAcc)
        sentenceAccs.append(sentenceAcc)


    return translations, charAccs, sentenceAccs

def testLatinLanguages(inputString):
    """Tests latin languages against eachother"""
    avgCharAccs = []
    avgSentenceAccs = []
  
    #between spanish and french
    print("Testing Spanish and French...")
    start = translate(inputString, 'es') #translate to spanish to start

    translations, charAccs, sentenceAccs = compareLangs('es', 'fr', inputString)
    avgCharAcc = np.mean(charAccs)
    avgSentenceAcc = np.mean(sentenceAccs)
    avgCharAccs.append(avgCharAcc)
    avgSentenceAccs.append(avgSentenceAcc)

    #between italian and spanish
    print("Testing Italian and Spanish...")
    start = translate(inputString, 'es') #translate to italian to start

    translations, charAccs, sentenceAccs = compareLangs('it', 'es', inputString)
    avgCharAcc = np.mean(charAccs)
    avgSentenceAcc = np.mean(sentenceAccs)
    avgCharAccs.append(avgCharAcc)
    avgSentenceAccs.append(avgSentenceAcc)

    #between french and italian
    print("Testing French and Italian...")
    start = translate(inputString, 'fr') #translate to french to start

    translations, charAccs, sentenceAccs = compareLangs('fr', 'it', inputString)
    avgCharAcc = np.mean(charAccs)
    avgSentenceAcc = np.mean(sentenceAccs)
    avgCharAccs.append(avgCharAcc)
    avgSentenceAccs.append(avgSentenceAcc)

    # Display Results
    results = {'Between languages': ['Spanish and French',  'Italian and Spanish', 'French and Italian'],
               'Char Accuracy': avgCharAccs, 'Semantic Accuracy': avgSentenceAccs}
    df = pd.DataFrame(data=results)
    print(df)

    return df

testLatinLanguages("Life is like a box of chocolates")
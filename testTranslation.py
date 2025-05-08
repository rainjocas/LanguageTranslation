from transformers import AutoTokenizer, M2M100ForConditionalGeneration
from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np
import pandas as pd
from torch import Tensor
from sentence_transformers import SentenceTransformer, util #Semantic similarity
from thefuzz import fuzz #Character level similarity 

model = M2M100ForConditionalGeneration.from_pretrained("facebook/nllb-200-distilled-600M")
tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")

def translate(inputString, outputLang):
    """NOTE: ADD DOCUMENTATION"""
    # model_inputs = tokenizer(inputString, return_tensors="pt")
    # translation = model.generate(**model_inputs, forced_bos_token_id=tokenizer.get_lang_id(outputLang))
    # return tokenizer.batch_decode(translation, skip_special_tokens=True)
    # outputLang = str(outputLang)
    # inputString = str(inputString)
    # print("inputString:", type(inputString), inputString)
    # print(type(outputLang))
    # inputString = str(outputLang) + " " + inputString
    encoded_hi = tokenizer(inputString, return_tensors="pt")
    generated_tokens = model.generate(**encoded_hi, forced_bos_token_id=tokenizer.convert_tokens_to_ids(outputLang))
    print(tokenizer.batch_decode(generated_tokens, skip_special_tokens=True), outputLang)
    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

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
        translation = translate(str(translations[len(translations) - 1]), lang2)
        translations.append(translation)

        charAcc = compareChars(translations[len(translations) - 3], translation)
        sentenceAcc = compareSentences(translations[len(translations) - 3], translation)
        charAccs.append(charAcc)
        sentenceAccs.append(sentenceAcc)

        #to language 1
        translation = translate(translations[len(translations) - 1], lang1)
        translations.append(translation)

        charAcc = compareChars(translations[len(translations) - 3], translation)
        sentenceAcc = compareSentences(translations[len(translations) - 3 ], translation)
        charAccs.append(charAcc)
        sentenceAccs.append(sentenceAcc)


    return translations, charAccs, sentenceAccs

def testIndoEuropeanLanguages(inputString):
    """
    Tests top 3 most spoken Indo-European languages against eachother, by number of native speakers
    English, Hindi, and Spanish
    """
    avgCharAccs = []
    avgSentenceAccs = []
  
    #between English and Hindi
    print("Testing English and Hindi...")
    start = translate(inputString, 'eng_Latn') #translate to spanish to start

    translations, charAccs, sentenceAccs = compareLangs('eng_Latn', 'hin_Deva', start)
    avgCharAcc = np.mean(charAccs)
    avgSentenceAcc = np.mean(sentenceAccs)
    avgCharAccs.append(avgCharAcc)
    avgSentenceAccs.append(avgSentenceAcc)

    print(translations[9])
    print(translations[10])

    #between Hindi and Spanish
    print("Testing Hindi and Spanish...")
    start = translate(inputString, 'hin_Deva') #translate to spanish to start

    translations, charAccs, sentenceAccs = compareLangs('hin_Deva', 'spa_Latn', start)
    avgCharAcc = np.mean(charAccs)
    avgSentenceAcc = np.mean(sentenceAccs)
    avgCharAccs.append(avgCharAcc)
    avgSentenceAccs.append(avgSentenceAcc)

    print(translations[9])
    print(translations[10])

    #between Spanish and English
    print("Testing Spanish and English...")
    start = translate(inputString, 'spa_Latn') #translate to french to start

    translations, charAccs, sentenceAccs = compareLangs('spa_Latn', 'eng_Latn', start)
    avgCharAcc = np.mean(charAccs)
    avgSentenceAcc = np.mean(sentenceAccs)
    avgCharAccs.append(avgCharAcc)
    avgSentenceAccs.append(avgSentenceAcc)

    print(translations[9])
    print(translations[10])

    # Display Results
    results = {'Between languages': ['English and Hindi',  'Hindi and Spanish', 'Spanish and English'],
               'Char Accuracy': avgCharAccs, 'Semantic Accuracy': avgSentenceAccs}
    df = pd.DataFrame(data=results)
    print(df)

    return df

def testSinoTibetanLanguages(inputString):
    """
    Tests top 3 most spoken Sino-Tibetan languages available in this model against eachother, by number
    of native speakers: Simplified Chinese, Burmese, and Tibetan

    *NOTE: Mandarin Chinese is most commonly spoken but simplified chinese is the most common writing system
    for chinese speakers and this is a text based expiriment
    *NOTE: Several more commonly spoken Sino-Tibetan languages needed to be skipped because they were not
    available in this model, including Wu Chinese, Jinyu Chinese, and Min Nan Chinese. Yue Chinese is
    supposedly available in this model, but the translation key did not work, so it was also skipped. This
    has been flagged as a bug.
    """
    avgCharAccs = []
    avgSentenceAccs = []
  
    #between Simplified Chinese and Burmese
    print("Testing Simplified Chinese and Burmese...")
    start = translate(inputString, 'zho_Hans') #translate to Simplified Chinese to start

    translations, charAccs, sentenceAccs = compareLangs('zho_Hans', 'mya_Mymr', start)
    avgCharAcc = np.mean(charAccs)
    avgSentenceAcc = np.mean(sentenceAccs)
    avgCharAccs.append(avgCharAcc)
    avgSentenceAccs.append(avgSentenceAcc)

    print(translations[9])
    print(translations[10])

    #between Burmese and Tibetan
    print("Testing Burmese and Tibetan...")
    start = translate(inputString, 'mya_Mymr') #translate to Burmese to start

    translations, charAccs, sentenceAccs = compareLangs('mya_Mymr', 'bod_Tibt', start)
    avgCharAcc = np.mean(charAccs)
    avgSentenceAcc = np.mean(sentenceAccs)
    avgCharAccs.append(avgCharAcc)
    avgSentenceAccs.append(avgSentenceAcc)

    print(translations[9])
    print(translations[10])

    #between Tibetan and Simplified Chinese
    print("Testing Tibetan and Simplified Chinese...")
    start = translate(inputString, 'bod_Tibt') #translate to Tibetan to start

    translations, charAccs, sentenceAccs = compareLangs('bod_Tibt', 'zho_Hans', start)
    avgCharAcc = np.mean(charAccs)
    avgSentenceAcc = np.mean(sentenceAccs)
    avgCharAccs.append(avgCharAcc)
    avgSentenceAccs.append(avgSentenceAcc)

    print(translations[9])
    print(translations[10])

    # Display Results
    results = {'Between languages': ['Simplified and Yue Chinese',  'Yue Chinese and Burmese', 'Burmese and Simplified Chinese'],
               'Char Accuracy': avgCharAccs, 'Semantic Accuracy': avgSentenceAccs}
    df = pd.DataFrame(data=results)
    print(df)

    return df

def testAfroAsiaticLanguages(inputString):
    """
    Tests top 3 most spoken Afro-Asiatic languages available in this model against eachother, by number
    of native speakers: Arabic, and Egyptian Arabic, and Hausa

    *NOTE: Standard Arabic (MSA), the most common form of written arabic is not available in this dataset.
    Arabic is the most commonly spoken, but it is a macrolanguage which encompasses all arabic languages,
    including Egyptian Arabic
    """
    avgCharAccs = []
    avgSentenceAccs = []
  
    #between Arabic and Egyptian Arabic
    print("Testing Arabic and Egyptian Arabic...")
    start = translate(inputString, 'arb_Arab') #translate to Arabic to start

    translations, charAccs, sentenceAccs = compareLangs('arb_Arab', 'arz_Arab', start)
    avgCharAcc = np.mean(charAccs)
    avgSentenceAcc = np.mean(sentenceAccs)
    avgCharAccs.append(avgCharAcc)
    avgSentenceAccs.append(avgSentenceAcc)

    print(translations[9])
    print(translations[10])

    #between Egyptian Arabic and Hausa
    print("Testing Egyptian Arabic and Hausa...")
    start = translate(inputString, 'arz_Arab') #translate to Egyptian Arabic to start

    translations, charAccs, sentenceAccs = compareLangs('arz_Arab', 'hat_Latn', start)
    avgCharAcc = np.mean(charAccs)
    avgSentenceAcc = np.mean(sentenceAccs)
    avgCharAccs.append(avgCharAcc)
    avgSentenceAccs.append(avgSentenceAcc)

    print(translations[9])
    print(translations[10])

    #between Hausa and Arabic
    print("Testing Hausa and Arabic...")
    start = translate(inputString, 'hat_Latn') #translate to Hausa to start

    translations, charAccs, sentenceAccs = compareLangs('hat_Latn', 'arb_Arab', start)
    avgCharAcc = np.mean(charAccs)
    avgSentenceAcc = np.mean(sentenceAccs)
    avgCharAccs.append(avgCharAcc)
    avgSentenceAccs.append(avgSentenceAcc)

    print(translations[9])
    print(translations[10])

    # Display Results
    results = {'Between languages': ['Arabic and Egyptian Arabic',  'Egyptian Arabic and Hausa', 'Hausa and Arabic'],
               'Char Accuracy': avgCharAccs, 'Semantic Accuracy': avgSentenceAccs}
    df = pd.DataFrame(data=results)
    print(df)
    print(translations)

    return df

def testLanguageGroups(inputString):
    """
    Tests the most commonly spoken/written language of the Indo-European, Sino-Tibetan, and Afro-Asiatic
    language groups: English, Simplified Chinese, Arabic
    """
    avgCharAccs = []
    avgSentenceAccs = []

    #between English and Simplified Chinese
    print("Testing English and Simplified Chinese...")
    start = translate(inputString, 'en') #translate to English to start

    translations, charAccs, sentenceAccs = compareLangs('en', 'zh', start)
    avgCharAcc = np.mean(charAccs)
    avgSentenceAcc = np.mean(sentenceAccs)
    avgCharAccs.append(avgCharAcc)
    avgSentenceAccs.append(avgSentenceAcc)

    #between Simplified Chinese and Simplified Chinese
    print("Testing Simplified Chinese and Arabic...")
    start = translate(inputString, 'zh') #translate to Simplified Chinese to start

    translations, charAccs, sentenceAccs = compareLangs('zh', 'ar', start)
    avgCharAcc = np.mean(charAccs)
    avgSentenceAcc = np.mean(sentenceAccs)
    avgCharAccs.append(avgCharAcc)
    avgSentenceAccs.append(avgSentenceAcc)

    #between Arabic and English
    print("Testing Arabic and English...")
    start = translate(inputString, 'ar') #translate to Arabic to start

    translations, charAccs, sentenceAccs = compareLangs('ar', 'en', start)
    avgCharAcc = np.mean(charAccs)
    avgSentenceAcc = np.mean(sentenceAccs)
    avgCharAccs.append(avgCharAcc)
    avgSentenceAccs.append(avgSentenceAcc)


testIndoEuropeanLanguages("Life is like a box of chocolates")
testSinoTibetanLanguages("Life is like a box of chocolates")
testAfroAsiaticLanguages("Life is like a box of chocolates")
# testLanguageGroups("Life is like a box of chocolates")

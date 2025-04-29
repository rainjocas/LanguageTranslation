# Use a pipeline as a high-level helper
from transformers import pipeline
# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

pipe = pipeline("translation", model="facebook/nllb-200-distilled-600M")

text = "translate English to French: Legumes share resources with nitrogen-fixing bacteria."

translator = pipeline("translation_xx_to_yy", model="username/my_awesome_opus_books_model")
print(translator(text))

tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
inputs = tokenizer(text, return_tensors="pt").input_ids

model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")

#generate translation
outputs = model.generate(inputs, max_new_tokens=40, do_sample=True, top_k=30, top_p=0.95)
tokenizer.decode(outputs[0], skip_special_tokens=True) #change ids back to text
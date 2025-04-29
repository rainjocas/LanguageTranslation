from transformers import pipeline
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM


text = "translate English to French: Legumes share resources with nitrogen-fixing bacteria."

translator = pipeline("translation_xx_to_yy", model="username/my_awesome_opus_books_model")
print(translator(text))

tokenizer = AutoTokenizer.from_pretrained("username/my_awesome_opus_books_model")
inputs = tokenizer(text, return_tensors="pt").input_ids

#grab model
model = AutoModelForSeq2SeqLM.from_pretrained("username/my_awesome_opus_books_model")

#generate translation
outputs = model.generate(inputs, max_new_tokens=40, do_sample=True, top_k=30, top_p=0.95)
tokenizer.decode(outputs[0], skip_special_tokens=True) #change ids back to text



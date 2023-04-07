from transformers import MarianMTModel, MarianTokenizer
from typing import Sequence
from flask import Flask

class Translator:
    def __init__(self, source_lang: str, dest_lang: str) -> None:
        self.model_name = f'Helsinki-NLP/opus-mt-{source_lang}-{dest_lang}'
        self.model = MarianMTModel.from_pretrained(self.model_name)
        self.tokenizer = MarianTokenizer.from_pretrained(self.model_name)
        
    def translate(self, texts: Sequence[str]) -> Sequence[str]:
        tokens = self.tokenizer(list(texts), return_tensors="pt", padding=True)
        translate_tokens = self.model.generate(**tokens)
        return [self.tokenizer.decode(t, skip_special_tokens=True) for t in translate_tokens]
        
        
        
app = Flask('')

marian_ru_en = Translator('ru', 'en')



@app.route('/')
def index():
  return marian_ru_en.translate(['что слишком сознавать — это болезнь, настоящая, полная болезнь.'])
  
  
app.run('0.0.0.0')

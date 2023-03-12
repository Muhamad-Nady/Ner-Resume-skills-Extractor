# import libraries
import sys
import re
import PyPDF2
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import pandas as pd
from seqeval.metrics import classification_report
from transformers import pipeline
import numpy as np
from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertConfig, BertForTokenClassification, BertForSequenceClassification
from torch import cuda
from transformers import AutoConfig
device = 'cuda' if cuda.is_available() else 'cpu'
print(device)

from transformers import AutoConfig


## import the model from it's path
def import_model():
    #model = AutoModelForSequenceClassification.from_pretrained(model)
    id2label = {0:'O', 1:'B-SKILL', 2:'I-SKILL', 3:'O-SKILL'}
    label2id = {'O': 0, 'B-SKILL': 1, 'I-SKILL': 2, 'O-SKILL':3}
    config = r"C:\Users\trainee\Desktop\hits_project\solution\model\config.json"
    model_path = r"C:\Users\trainee\Desktop\hits_project\solution\model\pytorch_model.bin"
    tokenizer = BertTokenizer.from_pretrained(r"C:\Users\trainee\Desktop\hits_project\solution\model\vocab.txt", local_files_only=True)
    model = BertForTokenClassification.from_pretrained(model_path,config=config,num_labels=len(label2id), local_files_only=True)
    model.eval()
    return model, tokenizer

def read_pdf(pdf_path):
    text = ""
# creating a pdf file object
    pdfFileObj = open(pdf_path, 'rb')
# creating a pdf reader object
    pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
# printing number of pages in pdf file
## print(pdfReader.numPages)
# creating a page object
    for i in range(len(pdfReader.pages)):
        pageObj = pdfReader.getPage(i)
# extracting text from page
        text += pageObj.extractText()  
# closing the pdf file object
    pdfFileObj.close()
    return text

def clean_txt(text):
    ## cleaning corpus with cleantetx fun
    from cleantext import clean
    new_txt = clean(text, lower=True,
        fix_unicode=True,
        to_ascii=True,
        normalize_whitespace=False,
        no_line_breaks=True,
        strip_lines=True,
        keep_two_line_breaks=False,
        no_urls=True,
        no_emails=False,
        no_phone_numbers=True,
        no_numbers=False,
        no_digits=False,
        no_currency_symbols=True,
        no_punct=False,
        no_emoji=True,
        replace_with_url="<URL>",
        replace_with_email="<EMAIL>",
        replace_with_phone_number="<PHONE>",
        #replace_with_number="",
        #replace_with_digit="",
        replace_with_currency_symbol="",
        replace_with_punct=" ",
        lang="en",)
    # as per recommendation from @freylis, compile once only
    CLEANR = re.compile('<.*?>') 
    new_txt = re.sub(CLEANR, '', new_txt)
    return new_txt
def inferance(new_txt):
    token = []
    for sub_tex in text.split("."):
        pipe = pipeline(task="ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
        #r"I usualy used bootstrap for web tasks and jupyter as ide for python, server sql"
        token.extend(pipe(sub_tex))
#print(token)
    token_md = token
    entities = []
    for row, entity in enumerate(token_md):
        if re.search("#", entity["word"]) != None:
            word = re.sub("#", "", entity["word"])
            #word = entity["word"]
            entities[-1]["word"] = entities[-1]["word"] + word
            #token_md.remove(entity)
        else:
            entities.append(entity)
    return entities
def plotted(entities):
    skills_extracted = []
    for entity in entities:
        skills_extracted.append(entity["word"])
        print(f"[Skill: {entity['word'].capitalize ()}, 'Score': {entity['score']:.2f}]", end="\n")
    figure(figsize=(8, 6), dpi=100, facecolor='k')
    from wordcloud import WordCloud 
    # Creating word_cloud with text as argument in .generate() method
    word_cloud = WordCloud(width=1600, height=800).generate(" ".join(set(skills_extracted)))
    # Display the generated Word Cloud
    plt.imshow(word_cloud)#, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout(pad=0)
    #plt.savefig("twitter.png", format="png")
    plt.show()

if __name__ == '__main__':
    resume_path = sys.argv[1]
    model, tokenizer = import_model()
    text = read_pdf(rf"{resume_path}")
    new_txt = clean_txt(text)
    entities = inferance(new_txt)
    plotted(entities)
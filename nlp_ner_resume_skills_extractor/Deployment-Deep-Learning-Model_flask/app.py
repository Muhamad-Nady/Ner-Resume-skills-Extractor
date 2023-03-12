from __future__ import division, print_function
# coding=utf-8
from itertools import groupby
import sys
import os
import glob
import re
import numpy as np
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

# Flask utils
from flask import Flask, redirect, url_for, request, render_template, flash
from flask import Flask, request, send_from_directory
from werkzeug.utils import secure_filename
from flask import Flask, session

#from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
# Load your trained model
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
# Load your trained model
# Necessary
# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#model.save('')

model, tokenizer = import_model()
print('Model loaded. Check http://127.0.0.1:5000/')

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


def inferance(text):

    sentence = text        #"i usualy used bootstrap for web tasks and jupyter as ide for python, server sql database"
    ## prediction using pytorch
    MAX_LEN = 512
    id2label = {0:'O', 1:'B-SKILL', 2:'I-SKILL', 3:"O-SKILL"}

    inputs = tokenizer(sentence, padding='max_length', truncation=True, max_length=MAX_LEN, return_tensors="pt")

    # move to gpu
    ids = inputs["input_ids"]
    mask = inputs["attention_mask"]
    # forward pass
    outputs = model(ids, mask)
    logits = outputs[0]
    active_logits = logits.view(-1, model.num_labels) # shape (batch_size * seq_len, num_labels)
    score_softmax = torch.softmax(active_logits, axis=1)  ## calculate softmax for predection value
    score = torch.max(score_softmax, axis=1)              ## get max output
    flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size*seq_len,) - predictions at the token level
    #score = torch.max(active_logits, axis=1)
    tokens = tokenizer.convert_ids_to_tokens(ids.squeeze().tolist())
    token_predictions = [id2label[i] for i in flattened_predictions.cpu().numpy()]
    wp_preds = list(zip(tokens, token_predictions, score[0])) # list of tuples. Each tuple = (wordpiece, prediction)
    #yhat_classes = np.where(flattened_predictions.cpu().numpy() > 0.5, 1, 0).squeeze().item()

    word_level_predictions = []
    for pair in wp_preds:
        if (pair[0].startswith("##")) or (pair[0] in ['[CLS]', '[SEP]', '[PAD]']):
        # skip prediction
            continue
        else:
            word_level_predictions.append([pair[1], pair[2].item()])

    # we join tokens, if they are not special ones
    str_rep = " ".join([t[0] for t in wp_preds if t[0] not in ['[CLS]', '[SEP]', '[PAD]']]).replace(" ##", "")
    #print(f"sentnce length:- {len(str_rep.split())}", f"[{str_rep}]", sep="\n")
    #print(f"predicted_label len:- {len(word_level_predictions)}", word_level_predictions, sep="\n")
    f_ls = list(zip(str_rep.split(), word_level_predictions))
    skills = [( x[0], x[1][0], x[1][1]) for x in f_ls if x[1][0] == "B-SKILL" or x[1][0] == "I-SKILL" or x[1][0] == "O-SKILL"]
    #print(yhat_classes)
    skills = [next(g) for _, g in groupby(skills, key=lambda x:x[0])]

    #print(skills, end="\n")
    return skills, tokens

# split resume in sentences
def max_inf(text):
    skills = []
    tokens = []
    for sub_tex in text.split("."):
        skill_line, token_line = inferance(text=sub_tex)
        skills.extend(skill_line)
        tokens.extend(token_line)
    return skills, tokens




def join_tokens(tokens):
    res = ''
    if tokens:
        res = tokens[0]
        for token in tokens[1:]:
            if not (token.isalpha() and res[-1].isalpha()):
                res += token  # punctuation
            else:
                res += ' ' + token  # regular word
    return res

def collapse(skills):
    # List with the result
    ner_result = skills
    collapsed_result = []


    current_entity_tokens = []
    current_entity = None

    # Iterate over the tagged tokens
    for token, tag, score in ner_result:

        if tag.startswith("B-"):
            # ... if we have a previous entity in the buffer, store it in the result list
            if current_entity is not None:
                collapsed_result.append([join_tokens(current_entity_tokens), current_entity])

            current_entity = tag[2:]
            # The new entity has so far only one token
            current_entity_tokens = [token]

        # If the entity continues ...
        elif current_entity_tokens!= None and tag == "I-" + str(current_entity):
            # Just add the token buffer
            current_entity_tokens.append(token)
        elif current_entity_tokens!= None and tag == "O-" + str(current_entity):
            # Just add the token buffer
            current_entity_tokens.append(token)
        else:
            collapsed_result.append([join_tokens(current_entity_tokens), current_entity])
            collapsed_result.append([token,tag[2:]])

            current_entity_tokens = []
            current_entity = None

            pass

    # The last entity is still in the buffer, so add it to the result
    # ... but only if there were some entity at all
    if current_entity is not None:
        collapsed_result.append([join_tokens(current_entity_tokens), current_entity])
        collapsed_result = sorted(collapsed_result)
        collapsed_result = list(k for k, _ in groupby(collapsed_result))

    return collapsed_result



@app.route('/', methods=['GET'])
def home():
    # Main page
    return render_template('home.html')




def uploader_file():
   if request.method == 'POST':
      f = request.files['resume']
      f.save(secure_filename(f.filename))
      return f


@app.route('/uploader', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = uploader_file()
        text = read_pdf(rf"C:\Users\trainee\Desktop\hits_project\solution\Deployment-Deep-Learning-Model-master\{file.filename}")
        # clean pdf text output
        #new_txt = clean_txt(text)
        # create enitity predicion dict
        entities, tokens = max_inf(text=text)
        # this function used for collape to concat all BIO
        res = join_tokens(tokens)
        # Concat all BIO entity of inferance function
        collapsed_result = collapse(entities)
        ## combine score of entity with the entity
        collapsed_result =  dict(collapsed_result)
        for entity in collapsed_result.keys():
            for skill in entities:
                if skill[0] in entity.split()[-1]:
                    collapsed_result[entity] =  "SKILL" + ", Score: " + "{:.2f}".format(skill[2])
                    # Convert to string
        prediction  = collapsed_result
        return render_template('home.html', prediction = prediction)

if __name__ == '__main__':
    app.debug = True
    app.run()
    

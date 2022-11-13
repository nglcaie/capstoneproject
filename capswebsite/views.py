import email
from multiprocessing import context
from django.shortcuts import render, redirect,get_object_or_404
from django.urls import reverse
from django.http import HttpResponse,HttpResponseRedirect, JsonResponse
from .models import *
from .forms import *
from datetime import datetime
from datetime import date, timedelta
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout, update_session_auth_hash
from django.contrib.auth.forms import PasswordChangeForm
from django.db.models import Q
import os
from django.conf import settings
# Create your views here.

#csv bago toh
import csv

#INTEGRATE TOPIC MODELING
#Step 3
import re
import numpy as np
import pandas as pd
from pprint import pprint
import string
from io import StringIO 

#NLTK
import nltk
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet 
import contractions

#Gensim
#!pip install gensim
import gensim
from gensim.parsing.preprocessing import STOPWORDS
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.models.phrases import Phrases
from gensim.test.utils import datapath
from gensim import  models

#Spacy for lemmatization
#!pip install spacy
import spacy

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim_models
#!pip install matplotlib
import matplotlib.pyplot as plt

import os

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

def index(request):
    return render(request, 'landing.html')
 
def loginPage(request):
    if request.user.is_authenticated and request.user.is_admin:
        return redirect('student_list')
    else:
        if request.method == 'POST':
            email = request.POST.get('username')
            password = request.POST.get('password')
            user = authenticate(request, email=email, password=password)
            if user is not None:
                login(request, user)
                users = request.user
                if user.is_authenticated and users.is_admin:
                    users.last_login = datetime.today()
                    users.save()
                    return redirect('student_list')
                else:
                    messages.error (request,'You have entered an invalid email or password.')
                    return render(request, 'login.html')
            else:
                    messages.error (request,'You have entered an invalid email or password.')
                    return render(request, 'login.html')
    return render(request, 'login.html')
 
def logout_view(request):
    logout(request)
    return redirect('login')
 
def sign_upPage(request):
    return render(request, 'sign_up.html')
 
def student_navbar(request):
    return render(request, 'student/student_navbar.html')
 
def answer_summary(request):
    return render(request, 'student/answer_summary.html')
 
def start_survey(request):
        return render(request, 'student/start_survey.html')
 
def survey_question(request):
    context ={}
    if request.POST:
        form = AnswerForm(request.POST)
        change = request.POST.get('change')
        change2 = request.POST.get('change2')
        change3 = request.POST.get('change3')
        change4 = request.POST.get('change4')
        change5 = request.POST.get('change5')
        param = 15
        if form.is_valid():
            if (int(change) < param) or (int(change2) < param) or (int(change3) < param) or (int(change4) < param) or (int(change5) < param):
                messages.error(request, "The survey will only accept a minimum of 15 words per answer")
                context['register'] = form
            else:
                form.save()
                return redirect('thankyou')
        else:
            context['register'] = form
    else:
        form = AnswerForm()
        context['register'] = form
    return render(request, 'student/survey_question.html', context)
 
def thankyou(request):
    return render(request, 'student/thankyou.html')

def admin_navbar(request):
    return render(request, 'admin/admin_navbar.html')

def evaluation(request):
    context={}
    if (request.method == 'POST'):
        csvFile = request.FILES.get('file')
        if not csvFile.name.endswith('.csv'):
            messages.error(request, 'Please only upload csv file')
        else:
            #temp_file = datapath('D:\capstone\capstone\capswebsite\model/lda_model1')
            temp_file = datapath(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'model\lda_model'))
            lda = models.ldamodel.LdaModel.load(temp_file)
            x1= lda.print_topics()
            #dicts = corpora.Dictionary.load('D:\capstone\capstone\capswebsite\model/lda_model1.id2word')
            id2word = corpora.Dictionary.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'model\lda_model.id2word'))
            #print(lda.print_topics())
            #pprint(dicts)

            df_test = pd.read_csv(StringIO(csvFile.read().decode('utf-8')), delimiter=',')
            df_test['q1']=df_test['question1'].astype(str) #convert type to string
            df_test['q1']=df_test['q1'].apply(lambda x: x.lower()) #all lowercase

            raw_data_test = df_test.q1.values.tolist() #<--covert to list

            #!!CONTRACTION DICTIONARY CAN REMOVE OR ADD!!
            #Step 2 - contractions
            contractions.add('profs', 'professors')
            contractions.add('prof', 'professor')
            contractions.add('f2f','face to face')
            contractions.add('ok','okay')
            contractions.add('papa','father')
            contractions.add('mom','mother')
            def expand_contractions(inputs):
                expanded = []
                for sent in inputs:
                    text_out = []
                    for word in sent.split():
                        text_out.append(contractions.fix(word))  
                        expanded_text = ' '.join(text_out)
                    expanded.append(text_out)
                return expanded

            data_test = expand_contractions(raw_data_test)

            #Step 3 -Tokenization
            def sent_to_words(sentences):
                for sentence in sentences:
                    yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

            data_words_test = list(sent_to_words(data_test)) #apply tokenization


            #print(data_words[:1])

            #Step 4 - Lemmarization
            lemmatizer = WordNetLemmatizer()

            def lemmatization(inputs):
                text_out = []
                for sent in inputs:
                    lem = [lemmatizer.lemmatize(word=x, pos='v') for x in sent]
                    text_out.append(lem)
                return text_out
            # Step 5 - Stopwords Removal
            #!!STOPWORDS DICTIONARY CAN REMOVE OR ADD!!
            #GENSIM STOPWORDS
            stop_words = STOPWORDS
            #for addition of stopwords
            stop_words = STOPWORDS.union(set(['yes','tree','know','way','cause','specially','especially','create','come','ung','make','become','like','also','able',
                                            'currently','really','have','lot','sa','mag']))
            sw_list = {'cannot','not','do','can','should','would','very','much','too','lot','alot','really','to','sometimes','of','does','no'}
            #for removing the stopwords from the list
            stop_words = stop_words.difference(sw_list)

            def remove_stopwords(texts):
                return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

            #Step6 - Bigrams
            # Build the bigram and trigram models
            bigram = gensim.models.Phrases(data_words_test, min_count=2, threshold=3) # higher threshold fewer phrases.
            trigram = gensim.models.Phrases(bigram[data_words_test], threshold=3)  
            quadgram = gensim.models.Phrases(trigram[data_words_test],threshold=3)  

            # Faster way to get a sentence clubbed as a trigram/bigram
            bigram_mod = gensim.models.phrases.Phraser(bigram)
            trigram_mod = gensim.models.phrases.Phraser(trigram)
            quadram_mod = gensim.models.phrases.Phraser(quadgram)

            # See trigram example
            #print(trigram_mod[bigram_mod[data_words[6]]])
            def make_bigrams(texts):
                return [bigram_mod[doc] for doc in texts]

            def make_trigrams(texts):
                return [trigram_mod[bigram_mod[doc]] for doc in texts]

            def make_quadrams(texts):
                return [quadram_mod[trigram_mod[bigram_mod[doc]]] for doc in texts]

            #print(trigram_mod[bigram_mod[data_words[0]]])


            #Step 7 - Additonal Stopwords
            stop_words_improve = STOPWORDS
            stop_words_improve = STOPWORDS.union(set(['need_to','used_to','of_time','alot_of','have_choice','do_not','do_not_have',
                                                    'to_do','use_to','need_to_do','nott','really_can_not','have_no','have_to_do','use_to','ampact','can_not','lack_of',
                                                    'do_not_feel','have_to','tend_to','of_course',
                                                    'to_focus','term_of','ability_to','care_of']))
            #sw_list = {}
            #for removing the stopwords from the list
            #stop_words_improve = stop_words_improve.difference(sw_list)

            def remove_stopwords_improve(texts):
                return [[word for word in simple_preprocess(str(doc)) if word not in stop_words_improve] for doc in texts]


            #APPLICATION

            data_lemmatized_test = lemmatization(data_words_test)
            # Remove Stop Words
            data_words_nostops_test = remove_stopwords(data_lemmatized_test)

            # Form Bigrams
            data_words_bigrams_test = make_bigrams(data_words_nostops_test)

            # Form Trigrams
            data_words_trigrams_test = make_trigrams(data_words_bigrams_test)

            data_words_quadrams_test = make_quadrams(data_words_trigrams_test)

            #improve_stop_words_test = data_words_quadrams_test

            improve_stop_words_test = remove_stopwords_improve(data_words_quadrams_test)

            # Create Dictionary
            #id2word_test = corpora.Dictionary(improve_stop_words_test)
            #print(id2word[10])
            # Create Corpus
            texts_test = improve_stop_words_test

            # Term Document Frequency
            corpus_test = [id2word.doc2bow(text) for text in texts_test]

            unseen_doc = corpus_test
            vector = lda[unseen_doc]
            lda.update(corpus_test)
            vector = lda[unseen_doc]

            #pprint(lda.print_topics())
            x2= lda.print_topics()
            context['x1'] = x1
            context['x2'] = x2
    return render(request, 'admin/evaluation.html',context)
 
def student_list(request):
    answers = Answers.objects.all
    if (request.method == 'POST'):
        csvFile = request.FILES.get('file')
        if not csvFile.name.endswith('.csv'):
            messages.error(request, 'Please only upload csv file')
        else:
            dataset = StringIO(csvFile.read().decode('utf-8'))
            next(dataset)
            for row in csv.reader(dataset, delimiter=','):
                _, create = Answers.objects.update_or_create(
                    pk = row[0],
                    email=row[1],
                    firstName = row[2],
                    lastName = row[3],
                    numberID = row[4],
                    college_id = row[5],
                    course_id = row[6],
                    year = row[7],
                    block = row[8],
                    question1 = row[9],
                    question2 = row[10],
                    question3 = row[11],
                    question4 = row[12],
                    question5 = row[13],
                )
    return render(request,'admin/student_list.html',{'answers':answers})
    
def student_answer(request, pk):
    answers = Answers.objects.filter(id=pk)
    context = {'answers':answers}
    return render(request, 'admin/student_answer.html',context)
 
def load_slot(request):
    collegeId = request.GET.get('college_Id')
    course = Course.objects.filter(college=collegeId)
    return render(request, 'student/dropdown_option.html', {'course': course})

def answers_csv(request): #csv bago toh
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename=MH_answers.csv'

    #Create a csv writer
    writer = csv.writer(response)

    #Designate the model
    answers = Answers.objects.all()

    #Add column headings to the csv file
    writer.writerow(['id','email', 'firstName', 'lastName', 'numberID', 'college', 'course', 'year', 'block', 'question1', 'question2', 'question3', 'question4', 'question5'])

    #Loop thru and output
    for answer in answers:
        writer.writerow([answer.pk,answer.email, answer.firstName, answer.lastName, answer.numberID, answer.college, answer.course, answer.year, answer.block, answer.question1, answer.question2, answer.question3, answer.question4, answer.question5])

    return response
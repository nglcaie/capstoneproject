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
#INTEGRATE TOPIC MODELING
#Step 3
import re
import numpy as np
import pandas as pd
from pprint import pprint
import string

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
        if form.is_valid():
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
    #temp_file = datapath('D:\capstone\capstone\capswebsite\model/lda_model1')
    temp_file = datapath(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'model\lda_model1'))
    lda = models.ldamodel.LdaModel.load(temp_file)
    x1= lda.print_topics()
    #dicts = corpora.Dictionary.load('D:\capstone\capstone\capswebsite\model/lda_model1.id2word')
    dicts = corpora.Dictionary.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'model\lda_model1.id2word'))
    pprint(lda.print_topics())
    pprint(dicts)
    df2 = pd.read_csv('C:/Users/Nella/Downloads/testSet.csv')
    df2['q1']=df2['question1'].astype(str) #convert type to string
    df2['q1']=df2['q1'].apply(lambda x: x.lower()) #all lowercase

    #!!CONTRACTION DICTIONARY CAN REMOVE OR ADD!!
    #!!CONTRACTION DICTIONARY CAN REMOVE OR ADD!!
    contractions_dict = {"its":"it is","it's":"it is","im":"i am","i'm":"i am","can't":"cannot","sometimes":"sometimes","don't":"do not","dont":"do not",
                        "hardtime":"hard time","time":"time",'overstimulate':"overstimulate","stimulating":"stimulating","i've":"i have","doesnt":"does not",
                        "doesn't":"does not","distracted":"distracted","limited":"limited","minimal":"minimal","a lot":"alot","set up":"setup",
                        "set-up":"setup","couldn":"could not","couldnt":"could not","couldn't":"could not","a bit":"little bit","wont":"would not",
                        "there's":"there is","won't":"would not"}

    # Regular expression for finding contractions
    contractions_re=re.compile('(%s)' % '|'.join(contractions_dict.keys()))

    def expand_contractions(text,contractions_dict=contractions_dict):
        def replace(match):
            return contractions_dict[match.group(0)]
        return contractions_re.sub(replace, text)

    # Expanding Contractions in the reviews
    df2['q1']=df2['q1'].apply(lambda x:expand_contractions(x))

    data2 = df2.q1.values.tolist()
    def remove_punctuation(text):
        no_punct=[words for words in text if words not in string.punctuation]
        words_wo_punct=''.join(no_punct)
        return words_wo_punct

    def tokenization(inputs):
        return word_tokenize(inputs)

    df2['q1']=df2['q1'].apply(lambda x:expand_contractions(x))
    df2['q1']=df2['q1'].apply(lambda x:remove_punctuation(x))
    df2['q1']=df2['q1'].apply(lambda x:tokenization(x))


    lemmatizer = WordNetLemmatizer()

    def lemmatization(inputs):
        return [lemmatizer.lemmatize(word=x, pos='v') for x in inputs]

    df2['q1']=df2['q1'].apply(lambda x:lemmatization(x))

    data_words2 = df2.q1.values.tolist()
    #print(data_words2[:2]) 

    #GENSIM STOPWORDS
    stop_words2 = STOPWORDS
    #for addition of stopwords
    stop_words2 = STOPWORDS.union(set(['yes','tree','know','way','cause','specially','especially','create','keep','come','ung','make','plenty','schedule','become','like','also','able','currently','really','have','lot','nan','pass','go','sa', 'na', 'ko', 'yung', 'hindi', 'ng', 'kasi', 'ako', 'pa','gusto', 
            'una', 'tungkol', 'ibig', 'kahit', 'nabanggit', 'huwag', 'nasaan', 'tayo', 'napaka', 'iyo', 'nakita', 'pataas', 'may', 'pagkatapos', 
            'anumang', 'lima', 'ibabaw', 'habang', 'at', 'tulad', 'nilang', 'pa', 'doon', 'ay', 'ngayon', 'akin', 'masyado', 'dito', 'din', 
            'likod', 'pangalawa', 'katiyakan', 'maaari', 'pero', 'bakit', 'pagitan', 'niya', 'kaya', 'makita', 'hanggang', 'paraan', 'siya',
            'para', 'kapag', 'ang', 'kapwa', 'kong', 'panahon', 'kanya', 'mula', 'kanila', 'bababa', 'kailangan', 'dahil', 'iyong', 'marapat',
            'sila', 'ginawang', 'ni', 'ako', 'kumuha', 'karamihan', 'gumawa', 'noon', 'muli', 'ating', 'mismo', 'ng', 'lahat', 'palabas',
            'hindi', 'niyang', 'kanilang', 'pumupunta', 'ito', 'lamang', 'apat', 'marami', 'iyon', 'sa', 'o', 'sabi', 'kanino', 'ginawa',
            'narito', 'bilang', 'saan', 'alin', 'gagawin', 'mahusay', 'namin', 'ilagay', 'nagkaroon', 'isa', 'ibaba', 'ilalim', 'ko', 'na',
            'naging', 'minsan', 'iba', 'dalawa', 'paano', 'pagkakaroon', 'aming', 'maging', 'atin', 'sabihin', 'nais', 'pamamagitan', 'ilan',
            'pumunta', 'kung', 'paggawa', 'amin', 'am', 'sarili', 'nito', 'dapat', 'sino', 'walang', 'bago', 'ano', 'nila', 'tatlo', 'kanyang',
            'itaas', 'kaysa', 'gayunman', 'laban', 'isang', 'pababa', 'mayroon', 'kulang', 'aking', 'ka', 'maaaring', 'pareho', 'kami',
            'kailanman', 'ginagawa', 'mga', 'katulad', 'ikaw', 'inyong', 'bawat', 'kay', 'lang', 'yung', 'yan', 'iyan', 'di', 'niya',
            'nya', 'ba', 'mong', 'mo', 'naman', 'kayo', 'di', 'ur', 'ano', 'anu','po','sakin','nag','pag']))
    sw_list2 = {'cannot','not','do','can','should','would','very','much','too','lot','alot','really','to','sometimes','of','does'}
    #for removing the stopwords from the list
    stop_words2 = stop_words2.difference(sw_list2)

    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(data_words2, min_count=2, threshold=2) # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[data_words2], threshold=2)  
    quadgram = gensim.models.Phrases(trigram[data_words2], threshold=2)  

    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    quadram_mod = gensim.models.phrases.Phraser(quadgram)

    def make_bigrams(texts):
        return [bigram_mod[doc] for doc in texts]

    def make_trigrams(texts):
        return [trigram_mod[bigram_mod[doc]] for doc in texts]

    def make_quadrams(texts):
        return [quadram_mod[trigram_mod[bigram_mod[doc]]] for doc in texts]


    def stopwords_remove(texts):
        return [[word for word in simple_preprocess(str(doc)) if word not in stop_words2] for doc in texts]

    data_words_stopwords2 = stopwords_remove(data_words2)


    data_words_bigrams2 = make_bigrams(data_words_stopwords2)

    # Form Trigrams
    data_words_trigrams2 = make_trigrams(data_words_bigrams2)

    data_words_quadrams2 = make_quadrams(data_words_trigrams2)

    stop_words_improve2 = STOPWORDS
    #ADDING STOPWORDS
    #sw_list_imp = ['need_to','used_to']
    #ADDING STOPWORDS
    stop_words_improve2 = STOPWORDS.union(set(['need_to','used_to','theres_alot_of','hard_to','of_time','alot_of','have_choice','do_not','do_not_have',
                                            'to_do','use_to','need_to_do','nott','really_can_not','have_no','have_to_do','use_to','ampact','can_not','lack_of',
                                            'do_not_feel','of_distractions','motivation_to','of_distraction','have_to','tend_to','of_course',
                                            'to_focus','term_of','ability_to']))

    def remove_stopwords(texts):
        return [[word for word in simple_preprocess(str(doc)) if word not in stop_words_improve2] for doc in texts]

    test_improve_stop_words = remove_stopwords(data_words_quadrams2)

    test_texts = test_improve_stop_words

    test_corpus = [dicts.doc2bow(text) for text in test_texts]

    unseen_doc = test_corpus
    vector = lda[unseen_doc]
    lda.update(test_corpus)
    vector = lda[unseen_doc]

    #pprint(lda.print_topics())
    x2= lda.print_topics()
    context = {'x1':x1,'x2':x2}
    return render(request, 'admin/evaluation.html',context)
 
def student_list(request):
    answers = Answers.objects.all
    return render(request,'admin/student_list.html',{'answers':answers})
    
def student_answer(request, pk):
    answers = Answers.objects.filter(id=pk)
    context = {'answers':answers}
    return render(request, 'admin/student_answer.html',context)
 
def load_slot(request):
    collegeId = request.GET.get('college_Id')
    course = Course.objects.filter(college=collegeId)
    return render(request, 'student/dropdown_option.html', {'course': course})
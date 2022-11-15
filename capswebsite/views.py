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
#!pip install contractions
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
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import io, base64
from matplotlib.ticker import FuncFormatter

#word cloud
#!pip install word_cloud
from wordcloud import WordCloud


import os

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)



#FUNCTIONS FOR DATA CLEANING
#Step 3 -Tokenization
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

#print(data_words[:1])

#Step 4 - Lemmarization
lemmatizer = WordNetLemmatizer()
def lemmatization(inputs):
    text_out = []
    for sent in inputs:
        lem = [lemmatizer.lemmatize(word=x, pos='v') for x in sent]
        text_out.append(lem)
    return text_out

def format_topics_sentences(ldamodel, corpus, texts):
    sent_topics_df = pd.DataFrame()
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row[0], key=lambda x: x[1], reverse=True) 
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)

def frequency_plot(topic):
    doc_lens = [len(d) for d in topic.Text]
    # Plot
    fig = plt.figure(figsize=(16,7), dpi=300)
    plt.hist(doc_lens, bins = 600, color='navy')
    plt.gca().set(xlim=(0, 600), ylabel='Number of Documents', xlabel="Document Word Count \n\n" + "Mean:" + str(round(np.mean(doc_lens))) + "   Median:" + str(round(np.median(doc_lens))) + "   Stdev:" + str(round(np.std(doc_lens))) + "   1%ile:" + str(round(np.quantile(doc_lens, q=0.01))) + "   99%ile:" + str(round(np.quantile(doc_lens, q=0.99))))
    plt.tick_params(size=20)
    plt.xticks(np.linspace(0,600,9))
    plt.title('Distribution of Document Word Counts', fontdict=dict(size=18))
    fig.tight_layout()
    flike = io.BytesIO()
    fig.savefig(flike)
    b64 = base64.b64encode(flike.getvalue()).decode()
    return b64
# Sentence Coloring of N Sentences
def topics_per_document(model, corpus, start=0, end=1):
    corpus_sel = corpus[start:end]
    dominant_topics = []
    topic_percentages = []
    for i, corp in enumerate(corpus_sel):
        topic_percs, wordid_topics, wordid_phivalues = model[corp]
        dominant_topic = sorted(topic_percs, key = lambda x: x[1], reverse=True)[0][0]
        dominant_topics.append((i, dominant_topic))
        topic_percentages.append(topic_percs)
    return(dominant_topics, topic_percentages)

def distrib_dominant(dominant_topics,lda_model,num_topics):
    # Distribution of Dominant Topics in Each Document
    df = pd.DataFrame(dominant_topics, columns=['Document_Id', 'Dominant_Topic'])
    dominant_topic_in_each_doc = df.groupby('Dominant_Topic').size()
    df_dominant_topic_in_each_doc = dominant_topic_in_each_doc.to_frame(name='count').reset_index()
    # Top 3 Keywords for each Topic
    topic_top3words = [(i, topic) for i, topics in lda_model.show_topics(num_topics=num_topics,formatted=False) 
                                    for j, (topic, wt) in enumerate(topics) if j < 10]
    df_top3words_stacked = pd.DataFrame(topic_top3words, columns=['topic_id', 'words'])
    df_top3words = df_top3words_stacked.groupby('topic_id').agg(', \n'.join)
    df_top3words.reset_index(level=0,inplace=True)
    # Topic Distribution by Dominant Topics
    dominant_fig, ax1= plt.subplots(1, figsize=(20, 10),dpi=300, sharey=True)
    ax1.bar(x='Dominant_Topic', height='count', data=df_dominant_topic_in_each_doc, width=.5, color='firebrick')
    ax1.set_xticks(range(df_dominant_topic_in_each_doc.Dominant_Topic.unique().__len__()))
    tick_formatter = FuncFormatter(lambda x, pos: 'Topic ' + str(x+1)+ '\n' + df_top3words.loc[df_top3words.topic_id==x, 'words'].values[0])
    ax1.xaxis.set_major_formatter(tick_formatter)
    ax1.set_title('Number of Documents by Dominant Topic', fontdict=dict(size=30))
    ax1.set_ylabel('Number of Documents',fontdict=dict(size=30))
    ax1.set_ylim(0, 700)
    flike = io.BytesIO()
    dominant_fig.tight_layout()
    dominant_fig.savefig(flike)
    dom = base64.b64encode(flike.getvalue()).decode()
    return dom

def weightage_topic(topic_percentages,lda_model,num_topics):
    # Distribution of Dominant Topics in Each Document
    topic_weightage_by_doc = pd.DataFrame([dict(t) for t in topic_percentages])
    df_topic_weightage_by_doc = topic_weightage_by_doc.sum().to_frame(name='count').reset_index()
    # Top 3 Keywords for each Topic
    topic_top3words = [(i, topic) for i, topics in lda_model.show_topics(num_topics=num_topics,formatted=False) 
                                    for j, (topic, wt) in enumerate(topics) if j < 10]
    df_top3words_stacked = pd.DataFrame(topic_top3words, columns=['topic_id', 'words'])
    df_top3words = df_top3words_stacked.groupby('topic_id').agg(', \n'.join)
    df_top3words.reset_index(level=0,inplace=True)
    # Topic Distribution by Topic Weights
    weightage_fig,ax2 = plt.subplots(1, figsize=(20, 10),dpi=300, sharey=True)
    ax2.bar(x='index', height='count', data=df_topic_weightage_by_doc, width=.5, color='steelblue')
    ax2.set_xticks(range(df_topic_weightage_by_doc.index.unique().__len__()))
    tick_formatter = FuncFormatter(lambda x, pos: 'Topic ' + str(x+1)+ '\n' + df_top3words.loc[df_top3words.topic_id==x, 'words'].values[0])
    ax2.xaxis.set_major_formatter(tick_formatter)
    ax2.set_title('Number of Documents by Topic Weightage', fontdict=dict(size=30))
    ax2.set_ylabel('Number of Documents', fontdict=dict(size=30))
    ax2.set_ylim(0, 700)
    weightage_fig.tight_layout()
    flike = io.BytesIO()
    weightage_fig.savefig(flike)
    weight = base64.b64encode(flike.getvalue()).decode()
    return weight

def word_cloud_gen(raw_data):
    stop_words = STOPWORDS
    sw_list = {'cannot','not','do','can','should','would','very','much','too','lot','alot','really','to','sometimes','of','does','no','will','just'}
    stop_words = stop_words.difference(sw_list)
    wordcloud = WordCloud(stopwords = stop_words, width=1600,height=800,background_color='white').generate((str(raw_data)))
    # create a figure
    word_fig, ax = plt.subplots(1,1, figsize = (5,5), dpi=300)
    # add interpolation = bilinear to smooth things out
    plt.imshow(wordcloud, interpolation='bilinear')
    # and remove the axis
    plt.axis("off")
    plt.tight_layout()
    flike = io.BytesIO()
    word_fig.savefig(flike)
    word = base64.b64encode(flike.getvalue()).decode()
    return word

#DJANGO WEBSITE

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
        param = 10
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
    param = 'Empty'
    if (request.method == 'POST'):
        csvFile = request.FILES.get('file')
        if not csvFile.name.endswith('.csv'):
            messages.error(request, 'Please only upload csv file')
        else:
            param = 'CSV'
            df_test = pd.read_csv(StringIO(csvFile.read().decode('utf-8')), delimiter=',')
            df_test['q1']=df_test['question1'].astype(str) #convert type to string
            df_test['q1']=df_test['q1'].apply(lambda x: x.lower()) #all lowercase
            df_test['q2']=df_test['question2'].astype(str) #convert type to string
            df_test['q2']=df_test['q2'].apply(lambda x: x.lower()) #all lowercase

            raw_data_test1 = df_test.q1.values.tolist() #<--covert to list
            raw_data_test2 = df_test.q2.values.tolist() #<--covert to list

        def cleaning1(raw_data_test):
            #temp_file = datapath('D:\capstone\capstone\capswebsite\model\lda_model1')
            temp_file = datapath(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'static\model\lda_model'))
            lda = models.ldamodel.LdaModel.load(temp_file)
            x1= lda.print_topics()
            #dicts = corpora.Dictionary.load('D:\capstone\capstone\capswebsite\model\lda_model1.id2word')
            id2word = corpora.Dictionary.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'static\model\lda_model.id2word'))
            def expand_contractions(inputs):
                contractions.add('profs', 'professors')
                contractions.add('prof', 'professor')
                contractions.add('f2f','face to face')
                contractions.add('ok','okay')
                contractions.add('papa','father')
                contractions.add('mom','mother')
                expanded = []
                for sent in inputs:
                    text_out = []
                    for word in sent.split():
                        text_out.append(contractions.fix(word))  
                        expanded_text = ' '.join(text_out)
                    expanded.append(text_out)
                return expanded
            stop_words = STOPWORDS
            stop_words = STOPWORDS.union(set(['yes','tree','know','way','cause','specially','especially','create','come','ung','make','become','like','also','able',
                                            'currently','really','have','lot','sa','mag']))
            sw_list = {'cannot','not','do','can','should','would','very','much','too','lot','alot','really','to','sometimes','of','does','no'}
            stop_words = stop_words.difference(sw_list)
            def remove_stopwords(texts):
                return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]
            stop_words_improve = STOPWORDS
            stop_words_improve = STOPWORDS.union(set(['need_to','used_to','of_time','alot_of','have_choice','do_not','do_not_have',
                                                    'to_do','use_to','need_to_do','nott','really_can_not','have_no','have_to_do','use_to','ampact','can_not','lack_of',
                                                    'do_not_feel','have_to','tend_to','of_course',
                                                    'to_focus','term_of','ability_to','care_of']))
            def remove_stopwords_improve(texts):
                return [[word for word in simple_preprocess(str(doc)) if word not in stop_words_improve] for doc in texts]

            data_test = expand_contractions(raw_data_test)
            data_words_test = list(sent_to_words(data_test)) #apply tokenization
            data_lemmatized_test = lemmatization(data_words_test)
            data_words_nostops_test = remove_stopwords(data_lemmatized_test)
            bigram = gensim.models.Phrases(data_words_test, min_count=2, threshold=3) # higher threshold fewer phrases.
            trigram = gensim.models.Phrases(bigram[data_words_test], threshold=3)  
            quadgram = gensim.models.Phrases(trigram[data_words_test],threshold=3)  
            bigram_mod = gensim.models.phrases.Phraser(bigram)
            trigram_mod = gensim.models.phrases.Phraser(trigram)
            quadram_mod = gensim.models.phrases.Phraser(quadgram)
            def make_bigrams(texts):
                return [bigram_mod[doc] for doc in texts]
            def make_trigrams(texts):
                return [trigram_mod[bigram_mod[doc]] for doc in texts]
            def make_quadrams(texts):
                return [quadram_mod[trigram_mod[bigram_mod[doc]]] for doc in texts]
            data_words_bigrams_test = make_bigrams(data_words_nostops_test)
            data_words_trigrams_test = make_trigrams(data_words_bigrams_test)
            data_words_quadrams_test = make_quadrams(data_words_trigrams_test)
            improve_stop_words_test = remove_stopwords_improve(data_words_quadrams_test)
            texts_test = improve_stop_words_test
            corpus_test = [id2word.doc2bow(text) for text in texts_test]
            unseen_doc = corpus_test
            vector = lda[unseen_doc]
            lda.update(corpus_test)
            vector = lda[unseen_doc]
            return improve_stop_words_test,texts_test,corpus_test,lda
    
        def cleaning2(raw_data_test):
            #temp_file = datapath('D:\capstone\capstone\capswebsite\model\lda_model1')
            temp_file = datapath(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'static\model\lda_model2'))
            lda = models.ldamodel.LdaModel.load(temp_file)
            x1= lda.print_topics()
            #dicts = corpora.Dictionary.load('D:\capstone\capstone\capswebsite\model\lda_model1.id2word')
            id2word = corpora.Dictionary.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'static\model\lda_model2.id2word'))
            def expand_contractions(inputs):
                contractions.add('profs', 'professors')
                contractions.add('prof', 'professor')
                contractions.add('f2f','face to face')
                contractions.add('ok','okay')
                contractions.add('exams','examinations')
                contractions.add('exam','examination')
                contractions.add('assign','assignment')
                contractions.add('sem','semester')
                contractions.add('professoressors','professors')
                contractions.add('final','finals')
                expanded = []
                for sent in inputs:
                    text_out = []
                    for word in sent.split():
                        text_out.append(contractions.fix(word))  
                        expanded_text = ' '.join(text_out)
                    expanded.append(text_out)
                return expanded
            stop_words = STOPWORDS
            stop_words = STOPWORDS.union(set(['react','date','specially','think','far','honestly','foo','come','ask','look','sadyang','past','end','nott',
                                            'pretty','gon','si','thing','slightly','lately','anymore','especially','haha','cause','guess','usually','like',
                                            'know','currently','feel','actually','let','ok','okay','felt','past','use']))
            sw_list = {'cannot','not','do','can','should','would','very','much','too','lot','alot','really','to','sometimes','of','does','no','will','just'}
            stop_words = stop_words.difference(sw_list)
            def remove_stopwords(texts):
                return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]
            stop_words_improve = STOPWORDS
            stop_words_improve = STOPWORDS.union(set(['able_to','lot_of','can_not','time_to','to_do','hard_to','compare_to',
                                                    'lack_of','do_not','want_to','feel_like','of_time','try_to','need_to','come_to','kind_of','number_of','use_to',
                                                    'instead_of','harder_to','of_work','best_to','to_start','professor_do','to_catch','can_do','cannot_do']))
            def remove_stopwords_improve(texts):
                return [[word for word in simple_preprocess(str(doc)) if word not in stop_words_improve] for doc in texts]

            data_test = expand_contractions(raw_data_test)
            data_words_test = list(sent_to_words(data_test)) #apply tokenization
            data_lemmatized_test = lemmatization(data_words_test)
            data_words_nostops_test = remove_stopwords(data_lemmatized_test)
            bigram = gensim.models.Phrases(data_words_test, min_count=2, threshold=1) # higher threshold fewer phrases.
            trigram = gensim.models.Phrases(bigram[data_words_test], threshold=1)  
            quadgram = gensim.models.Phrases(trigram[data_words_test],threshold=1)   
            bigram_mod = gensim.models.phrases.Phraser(bigram)
            trigram_mod = gensim.models.phrases.Phraser(trigram)
            quadram_mod = gensim.models.phrases.Phraser(quadgram)
            def make_bigrams(texts):
                return [bigram_mod[doc] for doc in texts]
            def make_trigrams(texts):
                return [trigram_mod[bigram_mod[doc]] for doc in texts]
            def make_quadrams(texts):
                return [quadram_mod[trigram_mod[bigram_mod[doc]]] for doc in texts]
            data_words_bigrams_test = make_bigrams(data_words_nostops_test)
            data_words_trigrams_test = make_trigrams(data_words_bigrams_test)
            data_words_quadrams_test = make_quadrams(data_words_trigrams_test)
            improve_stop_words_test = remove_stopwords_improve(data_words_quadrams_test)
            texts_test = improve_stop_words_test
            corpus_test = [id2word.doc2bow(text) for text in texts_test]
            unseen_doc = corpus_test
            vector = lda[unseen_doc]
            lda.update(corpus_test)
            vector = lda[unseen_doc]
            return improve_stop_words_test,texts_test,corpus_test,lda
            
        #question1   
        improve_stop_words_test1,texts_test1,corpus_test1,lda_model1 = cleaning1(raw_data_test1)
        #dominant and weightage
        dominant_topics1, topic_percentages1 = topics_per_document(model=lda_model1, corpus=corpus_test1, end=-1)
        dom_plot1 = distrib_dominant(dominant_topics1,lda_model1,20)
        weightage_plot1 = weightage_topic(topic_percentages1,lda_model1,20)

        #question2
        improve_stop_words_test2,texts_test2,corpus_test2,lda_model2= cleaning2(raw_data_test2)
        #dominant and weightage
        dominant_topics2, topic_percentages2 = topics_per_document(model=lda_model2, corpus=corpus_test2, end=-1)
        dom_plot2 = distrib_dominant(dominant_topics2,lda_model2,20)
        weightage_plot2 = weightage_topic(topic_percentages2,lda_model2,20)



        context['dominant1'] = dom_plot1
        context['weightage1'] = weightage_plot1
        context['dominant2'] = dom_plot2
        context['weightage2'] = weightage_plot2
        context['param'] = param
        #pprint(lda.print_topics())
        #x1= lda_model1.print_topics()
        #context['x1'] = x1
        #x2= lda_model2.print_topics()
        #context['x2'] = x2
    context['param'] = param
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


def word_cloud_page(request):
    context={}
    answers = Answers.objects.all().values()
    ans_count = Answers.objects.all().count()
    print(ans_count)
    if ans_count == 0:
        param = "Empty"
        context['param'] = param
    else:
        param = "Not Empty"
        df = pd.DataFrame(answers)
        #print(df.head(10))
        df['q1']=df['question1'].astype(str) #convert type to string
        df['q1']=df['q1'].apply(lambda x: x.lower()) #all lowercase
        df['q2']=df['question2'].astype(str) #convert type to string
        df['q2']=df['q2'].apply(lambda x: x.lower()) #all lowercase
        df['q3']=df['question3'].astype(str) #convert type to string
        df['q3']=df['q3'].apply(lambda x: x.lower()) #all lowercase
        df['q4']=df['question4'].astype(str) #convert type to string
        df['q4']=df['q4'].apply(lambda x: x.lower()) #all lowercase
        df['q5']=df['question5'].astype(str) #convert type to string
        df['q5']=df['q5'].apply(lambda x: x.lower()) #all lowercase
        raw_data_test1 = df.q1.values.tolist() #<--covert to list
        raw_data_test2 = df.q2.values.tolist() #<--covert to list
        raw_data_test3 = df.q3.values.tolist() #<--covert to list
        raw_data_test4 = df.q4.values.tolist() #<--covert to list
        raw_data_test5 = df.q5.values.tolist() #<--covert to list
        word_cloud1 = word_cloud_gen(raw_data_test1)
        word_cloud2 = word_cloud_gen(raw_data_test2)
        word_cloud3 = word_cloud_gen(raw_data_test3)
        word_cloud4 = word_cloud_gen(raw_data_test4)
        word_cloud5 = word_cloud_gen(raw_data_test5)
        context['word_cloud1'] = word_cloud1
        context['word_cloud2'] = word_cloud2
        context['word_cloud3'] = word_cloud3
        context['word_cloud4'] = word_cloud4
        context['word_cloud5'] = word_cloud5
    context['param'] = param
    return render(request, 'word_cloud.html', context)
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
    return render(request, 'admin/evaluation.html')
 
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
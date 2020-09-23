from django.shortcuts import render

import random
import pandas as pd
import re
from django.core import serializers
import time
import pickle
from django.http import HttpResponse,JsonResponse
import nltk
import random as conf
##from pywsd.allwords_wsd import disambiguate
##from pywsd.lesk import simple_lesk
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import VotingClassifier
from sklearn import decomposition
from sklearn.feature_extraction.text  import CountVectorizer
from sklearn.feature_extraction.text  import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import  SelectKBest, chi2
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
import numpy as np
import sklearn_crfsuite
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import warnings

from .models import *
import os
from django.contrib.sessions.models import Session
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage
from django.conf import settings
# from django.contrib.auth.models import User
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import nltk
import pandas as pd
import numpy as np
import time
import os,cv2
import pickle
import re
from sklearn.svm import LinearSVC
#protocol;src_bytes;ds_bytes;same_srv;diff_srv_rate;dst_host_srv_count;dst_host_same_srv_rate;dst_host_diff_srv_rate;dst_host_same_src_port_rate;dst_host_rerror_rate;dst_host;count;dst_host_count;dst_host_srv_count;dst_host_srv_serror_rate;cl;label
import dropbox
import os
class TransferData:
    def __init__(self, access_token):
        self.access_token = access_token

    def upload_file(self, file_from, file_to):
        """upload a file to Dropbox using API v2
        """
        dbx = dropbox.Dropbox(self.access_token)

        with open(file_from, 'rb') as f:
            dbx.files_upload(f.read(), file_to)
# get the word lists of sentences
def get_words_in_sentences(sentences):
            all_words = []
            for (words, sentiment) in sentences:
                    all_words.extend(words)
            return all_words

def close(wndw):
        wndw.destroy()

def result(result):
        rs=result

# get the unique word from the word list
def get_word_features(wordlist):
            wordlist = nltk.FreqDist(wordlist)
            word_features = wordlist.keys()
            return word_features

def startpage(request):
    return render(request,"login.html",{})

def transfer(request):
    return render(request,"transfer.html",{})

def login(request):
    return render(request,"login.html",{})
def index(request):
    return render(request,"index.html",{})

def transferingfile(request):
    if request.FILES["files"]:
        s = request.FILES["files"]
        fs = FileSystemStorage("App\\static\\files")
##        try:
##            os.remove("App\\static\\files\\"+s.name)
##        except Exception as ex:
##            print ("Exception: "+str(ex)+"!@!@")
##            pass
        fs.save(s.name, s)
        import pyAesCrypt
        # encryption/decryption buffer size - 64K
        bufferSize = 64 * 1024
        password = "foopassword"
        # encrypt
        pyAesCrypt.encryptFile("App\\static\\files\\"+s.name, "App\\static\\files\\"+s.name+".aes", password, bufferSize)

        depname=request.POST.get("dep")
        access_token = 'ttVqa5_XPaAAAAAAAAAAylEHHmcckHbfcG041udS0MqkXhFnA06ME-RbmjjmO_ul'
        transferData = TransferData(access_token)

        file_from = s.name
        file_to = '/test_dropbox/'+s.name+".aes"  # The full path to upload the file to, including the file name

        # API v2
        transferData.upload_file(file_from, file_to)
        try:
            o = FileRegister.objects.get(dep=depname)
            o.filepath="App\\static\\files\\"+s.name
            o.filename= s.name
            o.save()
        except:
            ob=FileRegister(filepath="App\\static\\files\\"+s.name,dep=depname,filename=s.name).save()
        return HttpResponse("<script>alert('File Transfered to Admin');window.location.href='/transfer/'</script>")
    return HttpResponse("<script>alert('Some Error Occured');window.location.href='/transfer/'</script>")

def adminhome(request):
    return render(request,"admin_new_home.html",{})

def about(request):
    return render(request,'about.html',{})

def contact(request):
    return render(request,'contact.html',{})

def newregister(request):
    return render(request,'register.html',{})

def contactmess(request):
    name=request.POST.get("name")
    email=request.POST.get("email")
    phone=request.POST.get("phone")
    mess=request.POST.get("mess")
    print("Name: ",name)
    print("Email: ",email)
    print("Phone: ",phone)
    print("Message: ",mess)
    try:
        ob=contactdet(name=name,email=email,phone=phone,mess=mess).save()
        return HttpResponse("<script>alert('Your Enquiry is send');window.location.href='/contact/'</script>")
    except:
        return HttpResponse("<script>alert('Failed to send your message');window.location.href='/contact/'</script>")

att=""
def attackch(request):
    ab=0
    data={}
    data["dt1"]=ab
    return JsonResponse(data,safe=False)

def newatt(request):
    return render(request,'listattcheck.html',{"att":att})

def testit(summ):
    resp=1
    sente_tests= summ[-1]
    word_features = get_word_features(sente_tests)
    def extract_features(document):
        document_words = set(document)
        features = {}
        for word in word_features:
          features['contains(%s)' % word] = (word in document_words)
        return features
    a=(int(summ[1])+int(summ[2])+int(summ[3]))/3
    if a>60:
        resp=0
    if "https" in sente_tests:
        return 0
    else:
        f=open("myclass.pickle",'rb')
        classi=pickle.load(f)
        res=classi.classify(extract_features(sente_tests.split()))
        print(resp)
        return resp

def singletest(sente_tests):
    f=open("finalized_model.sav",'rb')
    classi=pickle.load(f)
    emot=classi.classify(extract_features(sente_tests.split()))
    print(emot)
    return emot

def cValues(request):
    fn=request.POST.get("filename")
    print(fn)
    try:
        access_token = 'ttVqa5_XPaAAAAAAAAAAylEHHmcckHbfcG041udS0MqkXhFnA06ME-RbmjjmO_ul'
        dbx = dropbox.Dropbox(access_token)
        metadata, f = dbx.files_download('/test_dropbox/'+fn)
        con=f.content
        con=str(con).replace("b'","")
        con=str(con).replace("'","")
        print("con value: ",con)
    except:
        pass
    ob1=FileRegister.objects.get(filename=fn)
    ff = "App\\static\\files\\"+fn
    dat = pd.read_csv(ff)
    studs = []
    
    for i in range(len(dat)):
        ss = str(dat["Name"][i])+","+str(dat["Mark1"][i])+","+str(dat["Mark2"][i])+","+str(dat["Mark3"][i])+","+str(dat["Summary"][i])
        studs.append(ss)
    print("Studs: ",studs)
    l=[]
    for i in range(1,len(studs)-1):
        summ = studs[i].split(",")
        print("--> ",summ)
        resp = testit(summ)
        l.append(resp)
    print("List: ",l)
    try:
        obn = Results.objects.get(dep = str(ob1.dep).upper())
        obn.res= str(l)
        obn.filename= str(fn)
        obn.save()
    except:
        obb = Results(res=str(l),filename= str(fn),dep = str(ob1.dep)).save()
    p = l.count(0)
    q = l.count(1)
    import matplotlib.pyplot as plt

    labels = 'Pass', 'Fail'
    sizes = [p, q]
    explode = (0.1, 0)  # only "explode" the 2nd slice

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    plt.savefig(str(ob1.dep)+".png")
    img = cv2.imread(str(ob1.dep)+".png")
    cv2.imshow("Graph of Department", img)
    cv2.waitKey(0)
        
    return HttpResponse("<script>alert('Predicted');window.location.href='/studepred/'</script>")
##    except Exception as ex:
##        print("Exception: ", ex)
##        return HttpResponse("<script>alert('Request Failed');window.location.href='/inbpage/'</script>")
import time

def userdeppred(request):
    ob = Results.objects.all()
    d = {}
    rr = []
    langs = []
    students = []
    for i in ob:
        a = i.res
        b = a.replace("[","")
        b = b.replace("]","")
        l = b.split(",")
        print("l",l)
        r = [int(i) for i in l]
        p = r.count(0)
        q = r.count(1)
        langs.append(i.dep)
        students.append(p)
        langs.append(i.dep)
        students.append(q)
        rr.append([i.dep,p,q])
        pass
    print(rr)
    print(len(langs))
    c=[]
    for i in range(len(langs)):
        if i%2==0:
            c.append('g')
        else:
            c.append('r')
    print(len(students))
    print(langs)
    print(students)
    # libraries
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
     
    y_pos = np.arange(len(langs))
    plt.bar(y_pos, students, color=c)
    plt.xticks(y_pos, langs)
    legend_elements = [Patch(facecolor='g', edgecolor='g',
                         label='Pass'),
                   Patch(facecolor='r', edgecolor='r',
                         label='Fail')]

    # Create the figure
    plt.legend(handles=legend_elements, loc='upper right')
    plt.savefig('userdeppred.png')
    img = cv2.imread('userdeppred.png')
    cv2.imshow("Department Predictions",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return HttpResponse("<script>alert('Predicted');window.location.href='/index/'</script>")

def deppred(request):
    ob = Results.objects.all()
    d = {}
    rr = []
    langs = []
    students = []
    for i in ob:
        a = i.res
        b = a.replace("[","")
        b = b.replace("]","")
        l = b.split(",")
        print("l",l)
        r = [int(i) for i in l]
        p = r.count(0)
        q = r.count(1)
        langs.append(i.dep)
        students.append(p)
        langs.append(i.dep)
        students.append(q)
        rr.append([i.dep,p,q])
        pass
    print(rr)
    print(len(langs))
    c=[]
    for i in range(len(langs)):
        if i%2==0:
            c.append('g')
        else:
            c.append('r')
    print(len(students))
    print(langs)
    print(students)
    # libraries
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    
     
    y_pos = np.arange(len(langs))
    plt.bar(y_pos, students, color=c)
    plt.xticks(y_pos, langs)
    legend_elements = [Patch(facecolor='g', edgecolor='g',
                         label='Pass'),
                   Patch(facecolor='r', edgecolor='r',
                         label='Fail')]

    # Create the figure
    plt.legend(handles=legend_elements, loc='upper right')
    plt.savefig('deppred.png')
    img = cv2.imread('deppred.png')
    cv2.imshow("Department Predictions",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    
    return HttpResponse("<script>alert('Predicted');window.location.href='/studepred/'</script>")
    

def studepred(request):
    ob = FileRegister.objects.all()
    cn = ob.count()
    return render(request,'admin_student_prediction.html',{"data":ob,"cn":cn})
def registeration(request):
    name=request.POST.get("name")
    email=request.POST.get("email")
    uid=request.POST.get("uid")
    utype=request.POST.get("utype")
    dep=request.POST.get("dep")
    mob=request.POST.get("mobile")
    username=request.POST.get("username")
    password=request.POST.get("password")
    print("Name: ",name)
    n1=str(name)
    print("Email: ",email)
    print("uid: ",uid)
    print("utype: ",utype)
    print("dep: ",dep)
    print("Username: ",username)
    print("Password: ",password)
    
    
    try:
        if username=="admin":
            return HttpResponse("<script>alert('You cannot use admin as username.');window.location.href='/login/'</script>")
        ob=register(name=name,email=email,mob=mob,uid=uid,utype=utype,dep=dep,username=username,password=password).save()
        return HttpResponse("<script>alert('SUCCESSFULLY REGISTERED');window.location.href='/login/'</script>")
    except Exception as ex:
        print("Exception",str(ex))
        return HttpResponse("<script>alert('FAILED TO REGISTER. USERNAME ALREADY TAKEN');window.location.href='/login/'</script>")
import random
def genatt(request):
    global l
    att=random.choice(l)
    return render(request,'attcheck.html',{"att":att})

def userApprove(request):
    ob=register.objects.raw("select * from App_register where status=0 order by id desc limit 10;")
    return render(request,'admin_user_approval.html',{"data":ob})

def regUsers(request):
    ob=register.objects.raw("select * from App_register where status=1 order by id desc limit 10;")
    return render(request,'admin_registered.html',{"data":ob})


def approve(request):
    name=request.POST.get("name")
    email=request.POST.get("email")
    mob=request.POST.get("utype")
    username=request.POST.get("username")
    obj=register.objects.get(username=username)
    obj.status=1
    obj.save()

    return HttpResponse("<script>alert('User Approved');window.location.href='/userApprove/'</script>")


def check(request):
    email=request.POST.get("email")
    password=request.POST.get("password")
    print ("Email: ",email)
    print("password",password)
    if email=="admin" and password=="admin":
        return HttpResponse("<script>alert('Success. Welcome Admin');window.location.href='/adminhome/'</script>")
    else:
        try:
            obj=register.objects.get(username=email,password=password)
            if obj.status==1:
                if obj.utype=="HOD":
                    request.session["email"]=email
                    return HttpResponse("<script>alert('Success. Welcome HOD');window.location.href='/index/'</script>")
                else:
                    request.session["email"]=email
                    return HttpResponse("<script>alert('Success. Welcome Teacher');window.location.href='/index/'</script>")
            else:
                return HttpResponse("<script>alert('Please be Patient. Admin is Verifying');window.location.href='/login/'</script>")
            
                 
        except Exception as ex:
            print(ex)
            return HttpResponse("<script>alert('Try Again');window.location.href='/login/'</script>")

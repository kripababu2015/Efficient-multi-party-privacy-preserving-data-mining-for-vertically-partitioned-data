"""intrusion_det URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/1.10/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  url(r'^$', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  url(r'^$', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.conf.urls import url, include
    2. Add a URL to urlpatterns:  url(r'^blog/', include('blog.urls'))
"""
from django.conf.urls import url
from django.contrib import admin
from App.views import *
urlpatterns = [
    url(r'^admin/', admin.site.urls),
    url(r'^$',startpage),
    url(r'^index',index,name="index"),
    url(r'^about',about,name="about"),
    url(r'^contact',contact,name="contact"),
    url(r'^check',check,name="check"),
    url(r'^cValues',cValues,name="cValues"),
    url(r'^regUsers',regUsers,name="regUsers"),
    url(r'^approve',approve,name="approve"),
    url(r'^userApprove',userApprove,name="userApprove"),
    url(r'^deppred',deppred,name="deppred"),
    url(r'^userdeppred/',userdeppred,name="userdeppred"),
    url(r'^contactmess/',contactmess,name="contactmess"),
    url(r'^registeration/',registeration,name="registeration"),
    url(r'^newregister/',newregister,name="newregister"),
    url(r'^login/',login,name="login"),
    url(r'^genatt/',genatt,name="genatt"),
    url(r'^studepred/',studepred,name="studepred"),
    url(r'^transferingfile/',transferingfile,name="transferingfile"),
    url(r'^adminhome/',adminhome,name="adminhome"),
    url(r'^transfer/',transfer,name="transfer"),
    url(r'^newatt/',newatt,name="newatt"),
    url(r'^attackch/',attackch,name="attackch"),
    
]

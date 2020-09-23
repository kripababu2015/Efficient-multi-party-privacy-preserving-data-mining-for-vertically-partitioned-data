from django.db import models

# Create your models here.
class register(models.Model):
     name=models.CharField(max_length=200,default="vis")
     utype=models.CharField(max_length=200,default="HOD")
     email=models.CharField(max_length=200,default="vis@gmail.com")
     uid=models.CharField(max_length=200,default="12345")
     dep=models.CharField(max_length=200,default="MCA")
     mob=models.CharField(max_length=200,default="9874563210")
     username=models.CharField(max_length=200,unique=True,default="vis")
     password=models.CharField(max_length=200,default="vis")
     status=models.IntegerField(default=0)
class contactdet(models.Model):
     name=models.CharField(max_length=200)
     email=models.CharField(max_length=200)
     phone=models.CharField(max_length=200)
     mess=models.CharField(max_length=1000)

class FileRegister(models.Model):
     filepath=models.CharField(max_length=200)
     dep=models.CharField(max_length=200)
     filename=models.CharField(max_length=200,default="MCAData.csv")

class Results(models.Model):
     res=models.CharField(max_length=200,default="[1,1,1,1,1,1]")
     dep=models.CharField(max_length=200,default="MCA")
     filename=models.CharField(max_length=200,default="MCAData.csv")
     
class newregister(models.Model):
     name=models.CharField(max_length=200)
     age=models.CharField(max_length=20)
     email=models.CharField(max_length=200,unique=True)
     phone=models.CharField(max_length=200)
     address=models.CharField(max_length=200)
     password=models.CharField(max_length=200)
     pub=models.CharField(max_length=200)
     pri=models.CharField(max_length=200)

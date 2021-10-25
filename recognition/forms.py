from django.forms import ModelForm
from django.contrib.auth.models import User
from django import forms
from .models import Camera
from django.contrib.auth.forms import UserCreationForm  
from django.core.exceptions import ValidationError  
#from django.contrib.admin.widgets import AdminDateWidget

class usernameForm(forms.Form):
	username=forms.CharField(max_length=30)

class DateForm(forms.Form):
	date=forms.DateField(widget = forms.SelectDateWidget(empty_label=("Choose Year", "Choose Month", "Choose Day")))

class UsernameAndDateForm(forms.Form):
	username=forms.CharField(max_length=30)
	date_from=forms.DateField(widget = forms.SelectDateWidget(empty_label=("Choose Year", "Choose Month", "Choose Day")))
	date_to=forms.DateField(widget = forms.SelectDateWidget(empty_label=("Choose Year", "Choose Month", "Choose Day")))

class DateForm_2(forms.Form):
	date_from=forms.DateField(widget = forms.SelectDateWidget(empty_label=("Choose Year", "Choose Month", "Choose Day")))
	date_to=forms.DateField(widget = forms.SelectDateWidget(empty_label=("Choose Year", "Choose Month", "Choose Day")))

class CameraForm(forms.Form):
	ip=forms.CharField(label='Ip', min_length=5, max_length=40,widget=forms.TextInput(attrs={'class':'form-control validate effect-17'}))
	username=forms.CharField(label='Username', min_length=5, max_length=40,widget=forms.TextInput(attrs={'class':'form-control validate effect-17'}))
	password=forms.CharField(label='Password', min_length=5, max_length=40,widget=forms.TextInput(attrs={'class':'form-control validate effect-17'}))
	channel=forms.CharField(label='Channel',  max_length=20,widget=forms.TextInput(attrs={'class':'form-control validate effect-17'}))
	location=forms.CharField(label='Location', min_length=5, max_length=100,widget=forms.TextInput(attrs={'class':'form-control validate effect-17'}))
	cameratype=forms.CharField(label='Cameratype', max_length=10,widget=forms.TextInput(attrs={'class':'form-control validate effect-17'}))
	def save(self, commit = True):  
		camera=Camera()
		camera.ip=self.cleaned_data['ip']
		camera.username=self.cleaned_data['username']
		camera.password=self.cleaned_data['password']
		camera.channel=self.cleaned_data['channel']
		camera.location=self.cleaned_data['location']
		camera.cameratype=self.cleaned_data['cameratype']	
		camera.save()
		return camera
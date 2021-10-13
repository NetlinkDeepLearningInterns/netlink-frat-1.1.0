from django.forms import ModelForm
from django.contrib.auth.models import User
from django import forms
from django.contrib.auth.forms import UserCreationForm  
from django.core.exceptions import ValidationError  
#from django.contrib.admin.widgets import AdminDateWidget


class CustomUserCreationForm(UserCreationForm):  
    username = forms.CharField(label='Employee Id', min_length=5, max_length=150,widget=forms.TextInput(attrs={'class':'form-control validate effect-17'}))  
    email = forms.EmailField(label='email',widget=forms.EmailInput(attrs={'class':'form-control validate effect-17'}))
    first_name = forms.CharField(label='First Name',min_length=5, max_length=150,widget=forms.TextInput(attrs={'class':'form-control validate effect-17'}))  
    last_name = forms.CharField(label='Last Name',min_length=5, max_length=150,widget=forms.TextInput(attrs={'class':'form-control validate effect-17'}))  
    password1 = forms.CharField(label='password', widget=forms.PasswordInput(attrs={'class':'form-control validate effect-17'}))  
    password2 = forms.CharField(label='Confirm password', widget=forms.PasswordInput(attrs={'class':'form-control validate effect-17'}))  
    class Meta:
        model = User
        fields = ('username', 'first_name', 'last_name',
                'email', 'password1', 'password2')
        fields_order = ['username', 'first_name', 'last_name',
                'email', 'password1', 'password2']
    def username_clean(self):  
        username = self.cleaned_data['username'].lower()  
        new = User.objects.filter(username = username)  
        if new.count():  
            raise ValidationError("User Already Exist")  
        return username  
  
    def email_clean(self):  
        email = self.cleaned_data['email'].lower()  
        new = User.objects.filter(email=email)  
        if new.count():  
            raise ValidationError(" Email Already Exist")  
        return email  
  
    def clean_password2(self):  
        password1 = self.cleaned_data['password1']  
        password2 = self.cleaned_data['password2']  
  
        if password1 and password2 and password1 != password2:  
            raise ValidationError("Password don't match")  
        return password2  
  
    def save(self, commit = True):  
        user = User.objects.create_user(  
            self.cleaned_data['username'],  
            self.cleaned_data['email'],
            self.cleaned_data['password1']
        )  
        user.first_name=self.cleaned_data['first_name']
        user.last_name=self.cleaned_data['last_name']
        user.save()

        return user  
"""attendance_system_facial_recognition URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from django.contrib.auth import views as auth_views
from recognition import views as recog_views
from users import views as users_views

urlpatterns = [
    path('admin/', admin.site.urls),
     path('', recog_views.home, name='home'),
    
      path('dashboard/', recog_views.dashboard, name='dashboard'),
       path('registeremp/', recog_views.registeremp, name='registeremp'),
       path('videofeed/', recog_views.livefeed, name='videofeed'),
       path('viewemp/', recog_views.viewemp, name='viewemp'),
       path('delete_employee/<user>', recog_views.delete_employee, name='delete-employee'),
      path('control/', recog_views.control, name='control'),
      path('train/', recog_views.train, name='train'),
      path('train_modal/', recog_views.train_modal, name='train_modal'),
    path('markin/', recog_views.markin, name='markin'),
    path('markout/', recog_views.markout, name='markout'),
    path('add_photos/', recog_views.add_photos, name='add-photos'),
    path('add_camera/', recog_views.add_camera, name='add-camera'),
     path('login/',auth_views.LoginView.as_view(template_name='users/login.html'),name='login'),
    path('logout/',auth_views.LogoutView.as_view(template_name='recognition/home.html'),name='logout'),
     path('register/', users_views.register, name='register'),
     path('mark_your_attendance/<cam>', recog_views.test_mark_your_attendance ,name='mark-your-attendance'),
      path('mark_your_attendance_out/<cam>', recog_views.test_mark_your_attendance_out ,name='mark-your-attendance-out'),
      path('view_attendance_home', recog_views.view_attendance_home ,name='view-attendance-home'),
          path('test_close', recog_views.test_close ,name='test-close'),
        path('view_attendance_date', recog_views.view_attendance_date ,name='view-attendance-date'),
        path('view_attendance_employee', recog_views.view_attendance_employee ,name='view-attendance-employee'),
         path('view_my_attendance', recog_views.view_my_attendance_employee_login ,name='view-my-attendance-employee-login'),
       path('not_authorised', recog_views.not_authorised, name='not-authorised')
     

]

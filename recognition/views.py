from django.shortcuts import render,redirect
from .forms import usernameForm,DateForm,UsernameAndDateForm, DateForm_2,CameraForm
from django.contrib import messages
from django.views.decorators import gzip
from django.http import StreamingHttpResponse,HttpResponse
from django.contrib.auth.models import User
from .models import Camera
import cv2
import dlib
import imutils
from imutils import face_utils
from imutils.video import VideoStream
from imutils.face_utils import rect_to_bb
from imutils.face_utils import FaceAligner
import time
from attendance_system_facial_recognition.settings import BASE_DIR
import os
import json
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import numpy as np
from django.contrib.auth.decorators import login_required
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import datetime
from django_pandas.io import read_frame
from users.models import Present, Time
import seaborn as sns
import pandas as pd
from django.db.models import Count
#import mpld3
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
from matplotlib import rcParams
import math
import sqlite3
from sqlite3 import Error
import threading
import shutil
mpl.use('Agg')

#utility functions:

class VideoCamera(object):
	def __init__(self,cam):
		# cam="rtsp://mdpadmin:admin@10.95.9.27:554/Streaming/Channels/"+cam+"/"

		self.video = cv2.VideoCapture(cam)
		(self.grabbed, self.frame) = self.video.read()
		threading.Thread(target=self.update, args=()).start()
	def __del__(self):
		self.video.release()
	def get_frame(self):
		image=self.frame
		_,jpeg=cv2.imencode('.jpg',image)
		return jpeg.tobytes()
		
	def update(self):
		while True:
			(self.grabbed, self.frame) = self.video.read()


def gen(camera):
    while True:
        frame = camera.get_frame()
        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@gzip.gzip_page
def livefeed(request):
	cam=request.GET.get('cam')
	camobject=Camera.objects.get(id=cam)
	cam=camobject.getUrl()
	try:
		camx = VideoCamera(cam)
		return StreamingHttpResponse(gen(camx), content_type="multipart/x-mixed-replace;boundary=frame")
	except Exception:
		pass

def create_userlog():
	global userlog
	userlog={}
	us=User.objects.all()
	for u in us:
		if u.username!='admin':
			try:
				qs=Time.objects.filter(user=u).latest('time')		
				userlog[u.username]=not qs.out
			except Exception:
				userlog[u.username]=False
	print(userlog)

def create_connection(db_file):
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except Error as e:
        print(e)

    return conn

def username_present(username):
	if User.objects.filter(username=username).exists():
		return True
	
	return False

def create_dataset(username):
	eid = username
	if(os.path.exists('face_recognition_data/training_dataset/{}/'.format(eid))==False):
		os.makedirs('face_recognition_data/training_dataset/{}/'.format(eid))
	directory='face_recognition_data/training_dataset/{}/'.format(eid)

	# Detect face
	#Loading the HOG face detector and the shape predictpr for allignment

	# print("[INFO] Loading the facial detector")
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor('face_recognition_data/shape_predictor_68_face_landmarks.dat')   #Add path to the shape predictor ######CHANGE TO RELATIVE PATH LATER
	fa = FaceAligner(predictor , desiredFaceWidth = 96)
	#capture images from the webcam and process and detect the face
	# Initialize the video stream
	# print("[INFO] Initializing Video stream")
	# 'rtsp://mdpadmin:admin@10.95.9.27:554/Streaming/Channels/101/'
	vs = VideoStream(src=0).start()
	#time.sleep(2.0) ####CHECK######

	# Our identifier
	# We will put the id here and we will store the id with a face, so that later we can identify whose face it is
	
	# Our dataset naming counter
	sampleNum = 0
	# Capturing the faces one by one and detect the faces and showing it on the window
	while(True):
		# Capturing the image
		#vs.read each frame
		frame = vs.read()
		#Resize each image
		frame = imutils.resize(frame ,width = 800)
		#the returned img is a colored image but for the classifier to work we need a greyscale image
		#to convert
		gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		#To store the faces
		#This will detect all the images in the current frame, and it will return the coordinates of the faces
		#Takes in image and some other parameter for accurate result
		faces = detector(gray_frame,0)
		#In above 'faces' variable there can be multiple faces so we have to get each and every face and draw a rectangle around it.

		for face in faces:
			# print("inside for loop")
			(x,y,w,h) = face_utils.rect_to_bb(face)

			face_aligned = fa.align(frame,gray_frame,face)
			# Whenever the program captures the face, we will write that is a folder
			# Before capturing the face, we need to tell the script whose face it is
			# For that we will need an identifier, here we call it id
			# So now we captured a face, we need to write it in a file
			sampleNum = sampleNum+1
			# Saving the image dataset, but only the face part, cropping the rest
			
			if face is None:
				print("face is none")
				continue

			cv2.imwrite(directory+'/'+str(sampleNum)+'.jpg'	, face_aligned)
			face_aligned = imutils.resize(face_aligned ,width = 400)
			#cv2.imshow("Image Captured",face_aligned)
			# @params the initial point of the rectangle will be x,y and
			# @params end point will be x+width and y+height
			# @params along with color of the rectangle
			# @params thickness of the rectangle
			cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1)
			# Before continuing to the next loop, I want to give it a little pause
			# waitKey of 100 millisecond
			cv2.waitKey(50)

		#Showing the image in another window
		#Creates a window with window name "Face" and with the image img
		cv2.imshow("Add Images",frame)
		#Before closing it we need to give a wait command, otherwise the open cv wont work
		# @params with the millisecond of delay 1
		cv2.waitKey(1)
		#To get out of the loop
		if(sampleNum>300):
			break
	
	#Stoping the videostream
	vs.stop()
	# destroying all the windows
	cv2.destroyAllWindows()

def predict(face_aligned,svc,threshold=0.85):
	face_encodings=np.zeros((1,128))
	try:
		x_face_locations=face_recognition.face_locations(face_aligned)
		faces_encodings=face_recognition.face_encodings(face_aligned,known_face_locations=x_face_locations)
		if(len(faces_encodings)==0):
			return ([-1],[0])
	except Exception:
		return ([-1],[0])

	prob=svc.predict_proba(faces_encodings)
	result=np.where(prob[0]==np.amax(prob[0]))
	if(prob[0][result[0]]<=threshold):
		return ([-1],prob[0][result[0]])

	return (result[0],prob[0][result[0]])

def vizualize_Data(embedded, targets,):
	
	X_embedded = TSNE(n_components=2).fit_transform(embedded)

	for i, t in enumerate(set(targets)):
		idx = targets == t
		plt.scatter(X_embedded[idx, 0], X_embedded[idx, 1], label=t)

	plt.legend(bbox_to_anchor=(1, 1))
	rcParams.update({'figure.autolayout': True})
	plt.tight_layout()	
	plt.savefig('./recognition/static/recognition/img/training_visualisation.png')
	plt.close()
	return "/static/recognition/img/training_visualisation.png"

def update_attendance_in_db_in(present):
	today=datetime.date.today()
	time=datetime.datetime.now()

	for person in present:
		user=User.objects.get(username=person)
		try:
		   qs=Present.objects.get(user=user,date=today)
		except Exception:
			qs= None
		
		if qs is None:
			if present[person]==True:
						a=Present(user=user,date=today,present=True)
						a.save()
			else:
				a=Present(user=user,date=today,present=False)
		else:
			if present[person]==True:
				qs.present=True
				qs.save(update_fields=['present'])
		if present[person]==True:
			a=Time(user=user,date=today,time=time, out=False)
			a.save()

def update_attendance_in_db_out(present):
	
	today=datetime.date.today()
	time=datetime.datetime.now()
	
	for person in present:
		user=User.objects.get(username=person)
		print("sas",user)
		qs=Time.objects.filter(date=today).filter(user=user).filter(out=False)
		if present[person]==True and len(qs)>0:
			a=Time(user=user,date=today,time=time, out=True)
			a.save()
		
def check_validity_times(times_all):

	if(len(times_all)>0):
		sign=times_all.first().out
	else:
		sign=True

	times_in=times_all.filter(out=False)
	times_out=times_all.filter(out=True)

	if(len(times_in)!=len(times_out)):
		sign=True
	
	break_hourss=0
	
	if(sign==True):
			check=False
			break_hourss=0
			return (check,break_hourss)
	prev=True
	prev_time=times_all.first().time

	for obj in times_all:
		curr=obj.out
		if(curr==prev):
			check=False
			break_hourss=0
			return (check,break_hourss)
		if(curr==False):
			curr_time=obj.time
			to=curr_time
			ti=prev_time
			break_time=((to-ti).total_seconds())/3600
			break_hourss+=break_time

		else:
			prev_time=obj.time

		prev=curr

	return (True,break_hourss)

def convert_hours_to_hours_mins(hours):
	
	h=int(hours)
	hours-=h
	m=hours*60
	m=math.ceil(m)

	return str(str(h)+ " hrs " + str(m) + "  mins")

#used
def hours_vs_date_given_employee(present_qs,time_qs,admin=True):

	register_matplotlib_converters()
	df_hours=[]
	df_break_hours=[]
	qs=present_qs

	for obj in qs:
		date=obj.date
		times_in=time_qs.filter(date=date).filter(out=False).order_by('time')
		times_out=time_qs.filter(date=date).filter(out=True).order_by('time')
		times_all=time_qs.filter(date=date).order_by('time')
		obj.time_in=None
		obj.time_out=None
		obj.hours=0
		obj.break_hours=0
		if (len(times_in)>0):			
			obj.time_in=times_in.first().time
			
		if (len(times_out)>0):
			obj.time_out=times_out.last().time

		if(obj.time_in is not None and obj.time_out is not None):
			ti=obj.time_in
			to=obj.time_out
			hours=((to-ti).total_seconds())/3600
			obj.hours=hours
		else:
			obj.hours=0

		(check,break_hourss)= check_validity_times(times_all)
		if check:
			obj.break_hours=break_hourss

		else:
			obj.break_hours=0
		
		df_hours.append(obj.hours)
		df_break_hours.append(obj.break_hours)
		obj.hours=convert_hours_to_hours_mins(obj.hours)
		obj.break_hours=convert_hours_to_hours_mins(obj.break_hours)
				
	df = read_frame(qs)	
	
	df["hours"]=df_hours
	df["break_hours"]=df_break_hours

	# print(df)
	
	sns.barplot(data=df,x='date',y='hours')
	plt.xticks(rotation='vertical')
	rcParams.update({'figure.autolayout': True})
	plt.tight_layout()
	if(admin):
		plt.savefig('./recognition/static/recognition/img/attendance_graphs/hours_vs_date/1.png')
		plt.close()
	else:
		plt.savefig('./recognition/static/recognition/img/attendance_graphs/employee_login/1.png')
		plt.close()
	return qs
	
#used
def hours_vs_employee_given_date(present_qs,time_qs):
	
	register_matplotlib_converters()
	df_hours=[]
	df_break_hours=[]
	df_username=[]
	qs=present_qs

	for obj in qs:
		user=obj.user
		times_in=time_qs.filter(user=user).filter(out=False)
		times_out=time_qs.filter(user=user).filter(out=True)
		times_all=time_qs.filter(user=user)
		obj.time_in=None
		obj.time_out=None
		obj.hours=0
		obj.hours=0
		if (len(times_in)>0):			
			obj.time_in=times_in.first().time
		if (len(times_out)>0):
			obj.time_out=times_out.last().time
		if(obj.time_in is not None and obj.time_out is not None):
			ti=obj.time_in
			to=obj.time_out
			hours=((to-ti).total_seconds())/3600
			obj.hours=hours
		else:
			obj.hours=0
		(check,break_hourss)= check_validity_times(times_all)
		if check:
			obj.break_hours=break_hourss

		else:
			obj.break_hours=0
		
		df_hours.append(obj.hours)
		df_username.append(user.username)
		df_break_hours.append(obj.break_hours)
		obj.hours=convert_hours_to_hours_mins(obj.hours)
		obj.break_hours=convert_hours_to_hours_mins(obj.break_hours)

	df = read_frame(qs)	
	df['hours']=df_hours
	df['username']=df_username
	df["break_hours"]=df_break_hours

	sns.barplot(data=df,x='username',y='hours')
	plt.xticks(rotation='vertical')
	rcParams.update({'figure.autolayout': True})
	plt.tight_layout()
	plt.savefig('./recognition/static/recognition/img/attendance_graphs/hours_vs_employee/1.png')
	plt.close()
	return qs

def total_number_employees():
	qs=User.objects.all()
	
	return (len(qs) -1) # -1 to account for admin
	 
def attendance_log():
	qs=Present.objects.filter(present=True)
	data={}
	for q in qs:
		if str(q.date) not in data.keys():
			data[str(q.date)]=1
		else:
			data[str(q.date)]+=1
	data=sorted(data.items(),key=lambda x:x[0],reverse=True)
	return data

def employees_present_today():
	today=datetime.date.today()
	qs=Present.objects.filter(date=today).filter(present=True)
	return len(qs)

#used	
def this_week_emp_count_vs_date():
	today=datetime.date.today()
	some_day_last_week=today-datetime.timedelta(days=7)
	monday_of_last_week=some_day_last_week-  datetime.timedelta(days=(some_day_last_week.isocalendar()[2] - 1))
	monday_of_this_week = monday_of_last_week + datetime.timedelta(days=7)
	qs=Present.objects.filter(date__gte=monday_of_this_week).filter(date__lte=today)
	str_dates=[]
	emp_count=[]
	str_dates_all=[]
	emp_cnt_all=[]
	cnt=0
	
	for obj in qs:
		date=obj.date
		str_dates.append(str(date))
		qs=Present.objects.filter(date=date).filter(present=True)
		emp_count.append(len(qs))

	while(cnt<5):

		date=str(monday_of_this_week+datetime.timedelta(days=cnt))
		cnt+=1
		str_dates_all.append(date)
		if(str_dates.count(date))>0:
			idx=str_dates.index(date)
			emp_cnt_all.append(emp_count[idx])
		else:
			emp_cnt_all.append(0)

	df=pd.DataFrame()
	df["date"]=str_dates_all
	df["Number of employees"]=emp_cnt_all
		
	sns.lineplot(data=df,x='date',y='Number of employees')
	plt.savefig('./recognition/static/recognition/img/attendance_graphs/this_week/1.png')
	plt.close()

#used
def last_week_emp_count_vs_date():

	today=datetime.date.today()
	some_day_last_week=today-datetime.timedelta(days=7)
	monday_of_last_week=some_day_last_week-  datetime.timedelta(days=(some_day_last_week.isocalendar()[2] - 1))
	monday_of_this_week = monday_of_last_week + datetime.timedelta(days=7)
	qs=Present.objects.filter(date__gte=monday_of_last_week).filter(date__lt=monday_of_this_week)
	str_dates=[]
	emp_count=[]
	str_dates_all=[]
	emp_cnt_all=[]
	cnt=0

	for obj in qs:
		date=obj.date
		str_dates.append(str(date))
		qs=Present.objects.filter(date=date).filter(present=True)
		emp_count.append(len(qs))

	while(cnt<5):

		date=str(monday_of_last_week+datetime.timedelta(days=cnt))
		cnt+=1
		str_dates_all.append(date)
		if(str_dates.count(date))>0:
			idx=str_dates.index(date)
			emp_cnt_all.append(emp_count[idx])	
		else:
			emp_cnt_all.append(0)

	df=pd.DataFrame()
	df["date"]=str_dates_all
	df["emp_count"]=emp_cnt_all

	sns.lineplot(data=df,x='date',y='emp_count')
	plt.savefig('./recognition/static/recognition/img/attendance_graphs/last_week/1.png')
	plt.close()

def mark_your_attendance(request):
	global userlog
	detector = dlib.get_frontal_face_detector()	
	predictor = dlib.shape_predictor('face_recognition_data/shape_predictor_68_face_landmarks.dat')   #Add path to the shape predictor ######CHANGE TO RELATIVE PATH LATER
	svc_save_path="face_recognition_data/svc.sav"	

	with open(svc_save_path, 'rb') as f:
			svc = pickle.load(f)
	fa = FaceAligner(predictor , desiredFaceWidth = 96)
	encoder=LabelEncoder()
	encoder.classes_ = np.load('face_recognition_data/classes.npy')
	faces_encodings = np.zeros((1,128))
	no_of_faces = len(svc.predict_proba(faces_encodings)[0])
	count = dict()
	present = dict()
	log_time = dict()
	start = dict()
	for i in range(no_of_faces):
		count[encoder.inverse_transform([i])[0]] = 0
		present[encoder.inverse_transform([i])[0]] = False

# 'rtsp://mdpadmin:admin@10.95.9.27:554/Streaming/Channels/101/'
	vs = VideoStream(src=0).start()
	sampleNum = 0

	while(True):
		frame = vs.read()
		frame = imutils.resize(frame ,width = 800)	
		gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		faces = detector(gray_frame,0)
		for face in faces:
			print("INFO : inside for loop")
			(x,y,w,h) = face_utils.rect_to_bb(face)
			face_aligned = fa.align(frame,gray_frame,face)
			cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1)
			(pred,prob)=predict(face_aligned,svc)
			if(pred!=[-1]):
				person_name=encoder.inverse_transform(np.ravel([pred]))[0]
				pred=person_name
				if count[pred] == 0:
					start[pred] = time.time()
					count[pred] = count.get(pred,0) + 1

				if count[pred] == 4 and (time.time()-start[pred]) > 1.2:
					 count[pred] = 0
				else:
				#if count[pred] == 4 and (time.time()-start) <= 1.5:
					present[pred] = True
					log_time[pred] = datetime.datetime.now()
					count[pred] = count.get(pred,0) + 1
					print(pred, present[pred], count[pred])
				try:
					update_attendance_in_db_in(present)
				except Exception as e:
					print(e)
				cv2.putText(frame, str(person_name)+ str(prob), (x+6,y+h-6), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)

			else:
				person_name="unknown"
				cv2.putText(frame, str(person_name), (x+6,y+h-6), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
				
			#cv2.putText()
			# Before continuing to the next loop, I want to give it a little pause
			# waitKey of 100 millisecond
			#cv2.waitKey(50)

		#Showing the image in another window
		#Creates a window with window name "Face" and with the image img
		cv2.imshow("Mark Attendance - In - Press q to exit",frame)
		#Before closing it we need to give a wait command, otherwise the open cv wont work
		# @params with the millisecond of delay 1
		#cv2.waitKey(1)
		#To get out of the loop
		key=cv2.waitKey(50) & 0xFF
		if(key==ord("q")):
			break
	
	#Stoping the videostream
	vs.stop()
	# destroying all the windows
	cv2.destroyAllWindows()
	return render(request,'recognition/markin.html')

def test_mark_your_attendance(request,cam):
	global userlog
	detector = dlib.get_frontal_face_detector()	
	predictor = dlib.shape_predictor('face_recognition_data/shape_predictor_68_face_landmarks.dat')   #Add path to the shape predictor ######CHANGE TO RELATIVE PATH LATER
	svc_save_path="face_recognition_data/svc.sav"	

	with open(svc_save_path, 'rb') as f:
			svc = pickle.load(f)
	fa = FaceAligner(predictor , desiredFaceWidth = 96)
	encoder=LabelEncoder()
	encoder.classes_ = np.load('face_recognition_data/classes.npy')
	faces_encodings = np.zeros((1,128))
	no_of_faces = len(svc.predict_proba(faces_encodings)[0])
	count = dict()
	present = dict()
	log_time = dict()
	start = dict()
	for i in range(no_of_faces):
		count[encoder.inverse_transform([i])[0]] = 0
		present[encoder.inverse_transform([i])[0]] = False

# 'rtsp://mdpadmin:admin@10.95.9.27:554/Streaming/Channels/101/'
	camobject=Camera.objects.get(id=cam)
	cam_src=camobject.getUrl()
	# 'rtsp://admin:12345@103.46.196.100:554'
	vs = VideoStream(src=cam_src).start()
	sampleNum = 0

	while(True):
		frame = vs.read()
		frame = imutils.resize(frame ,width = 800)	
		gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		faces = detector(gray_frame,0)
		for face in faces:
			print("INFO : inside for loop")
			(x,y,w,h) = face_utils.rect_to_bb(face)
			face_aligned = fa.align(frame,gray_frame,face)
			cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1)
			(pred,prob)=predict(face_aligned,svc)
			if(pred!=[-1]):
				print("face detected")
				person_name=encoder.inverse_transform(np.ravel([pred]))[0]
				pred=person_name
				if count[pred] == 0:
					start[pred] = time.time()
					count[pred] = count.get(pred,0) + 1
					try:
						present[pred] = True
						update_attendance_in_db_in(present)
					except Exception as e:
						print(e)

				if count[pred] == 10 and (time.time()-start[pred]) > 1.2:
					count[pred] = 0
					try:
						update_attendance_in_db_in(present)
					except Exception as e:
						print(e)
				else:
				#if count[pred] == 4 and (time.time()-start) <= 1.5:
					present[pred] = True
					log_time[pred] = datetime.datetime.now()
					count[pred] = count.get(pred,0) + 1
					print(pred, present[pred], count[pred])
				cv2.putText(frame, str(person_name)+ str(prob), (x+6,y+h-6), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)

			else:
				person_name="unknown"
				cv2.putText(frame, str(person_name), (x+6,y+h-6), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
				
			#cv2.putText()
			# Before continuing to the next loop, I want to give it a little pause
			# waitKey of 100 millisecond
			#cv2.waitKey(50)

		#Showing the image in another window
		#Creates a window with window name "Face" and with the image img
		cv2.imshow(cam,frame)
		#Before closing it we need to give a wait command, otherwise the open cv wont work
		# @params with the millisecond of delay 1
		#cv2.waitKey(1)
		#To get out of the loop
		key=cv2.waitKey(50) & 0xFF
		if(key==ord("q")):
			break
	
	#Stoping the videostream
	vs.stop()
	# destroying all the windows
	cv2.destroyAllWindows()
	return render(request,'recognition/markin.html')

def mark_your_attendance_out(request):

	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor('face_recognition_data/shape_predictor_68_face_landmarks.dat')   #Add path to the shape predictor ######CHANGE TO RELATIVE PATH LATER
	svc_save_path="face_recognition_data/svc.sav"			
	with open(svc_save_path, 'rb') as f:
			svc = pickle.load(f)
	fa = FaceAligner(predictor , desiredFaceWidth = 96)
	encoder=LabelEncoder()
	encoder.classes_ = np.load('face_recognition_data/classes.npy')
	faces_encodings = np.zeros((1,128))
	no_of_faces = len(svc.predict_proba(faces_encodings)[0])
	count = dict()
	present = dict()
	log_time = dict()
	start = dict()
	for i in range(no_of_faces):
		count[encoder.inverse_transform([i])[0]] = 0
		present[encoder.inverse_transform([i])[0]] = False

# 'rtsp://mdpadmin:admin@10.95.9.27:554/Streaming/Channels/201/'
	
	vs = VideoStream(src=0).start()
	sampleNum = 0
	
	while(True):
		frame = vs.read()
		frame = imutils.resize(frame ,width = 800)
		gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		faces = detector(gray_frame,0)

		for face in faces:
			# print("INFO : inside for loop")
			(x,y,w,h) = face_utils.rect_to_bb(face)
			face_aligned = fa.align(frame,gray_frame,face)
			cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1)
			(pred,prob)=predict(face_aligned,svc)
			if(pred!=[-1]):
				person_name=encoder.inverse_transform(np.ravel([pred]))[0]
				pred=person_name
				if count[pred] == 0:
					start[pred] = time.time()
					count[pred] = count.get(pred,0) + 1

				if count[pred] == 4 and (time.time()-start[pred]) > 1.5:
					 count[pred] = 0
				else:
				#if count[pred] == 4 and (time.time()-start) <= 1.5:
					present[pred] = True
					log_time[pred] = datetime.datetime.now()
					count[pred] = count.get(pred,0) + 1
					print(pred, present[pred], count[pred])
				cv2.putText(frame, str(person_name)+ str(prob), (x+6,y+h-6), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
				try:
					update_attendance_in_db_out(present)
				except Exception as e:
					print(e)

			else:
				person_name="unknown"
				cv2.putText(frame, str(person_name), (x+6,y+h-6), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)

			#cv2.putText()
			# Before continuing to the next loop, I want to give it a little pause
			# waitKey of 100 millisecond
			#cv2.waitKey(50)

		#Showing the image in another window
		#Creates a window with window name "Face" and with the image img
		cv2.imshow("Mark Attendance- Out - Press q to exit",frame)
		#Before closing it we need to give a wait command, otherwise the open cv wont work
		# @params with the millisecond of delay 1
		#cv2.waitKey(1)
		#To get out of the loop
		key=cv2.waitKey(50) & 0xFF
		if(key==ord("q")):
			break
	
	#Stoping the videostream
	vs.stop()

	# destroying all the windows
	cv2.destroyAllWindows()
	# update_attendance_in_db_out(present)
	# return redirect('home')
	return render(request,'recognition/markout.html')

def test_mark_your_attendance_out(request,cam):

	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor('face_recognition_data/shape_predictor_68_face_landmarks.dat')   #Add path to the shape predictor ######CHANGE TO RELATIVE PATH LATER
	svc_save_path="face_recognition_data/svc.sav"			
	with open(svc_save_path, 'rb') as f:
			svc = pickle.load(f)
	fa = FaceAligner(predictor , desiredFaceWidth = 96)
	encoder=LabelEncoder()
	encoder.classes_ = np.load('face_recognition_data/classes.npy')
	faces_encodings = np.zeros((1,128))
	no_of_faces = len(svc.predict_proba(faces_encodings)[0])
	count = dict()
	present = dict()
	log_time = dict()
	start = dict()
	for i in range(no_of_faces):
		count[encoder.inverse_transform([i])[0]] = 0
		present[encoder.inverse_transform([i])[0]] = False

# 'rtsp://mdpadmin:admin@10.95.9.27:554/Streaming/Channels/201/'
	camobject=Camera.objects.get(id=cam)
	cam_src=camobject.getUrl()
	vs = VideoStream(src=cam_src).start()
	sampleNum = 0
	
	while(True):
		frame = vs.read()
		frame = imutils.resize(frame ,width = 800)
		gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		faces = detector(gray_frame,0)
		for face in faces:
			# print("INFO : inside for loop")
			(x,y,w,h) = face_utils.rect_to_bb(face)
			face_aligned = fa.align(frame,gray_frame,face)
			cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1)
			(pred,prob)=predict(face_aligned,svc)
			if(pred!=[-1]):
				person_name=encoder.inverse_transform(np.ravel([pred]))[0]
				pred=person_name
				if count[pred] == 0:
					start[pred] = time.time()
					count[pred] = count.get(pred,0) + 1
					try:
						present[pred] = True
						update_attendance_in_db_out(present)
					except Exception as e:
						print(e)

				if count[pred] == 10 and (time.time()-start[pred]) > 1.5:
					count[pred] = 0
					try:
						update_attendance_in_db_out(present)
					except Exception as e:
						print(e)

				else:
				#if count[pred] == 4 and (time.time()-start) <= 1.5:
					present[pred] = True
					log_time[pred] = datetime.datetime.now()
					count[pred] = count.get(pred,0) + 1
					print(pred, present[pred], count[pred])
				cv2.putText(frame, str(person_name)+ str(prob), (x+6,y+h-6), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)

			else:
				person_name="unknown"
				cv2.putText(frame, str(person_name), (x+6,y+h-6), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)

			#cv2.putText()
			# Before continuing to the next loop, I want to give it a little pause
			# waitKey of 100 millisecond
			#cv2.waitKey(50)

		#Showing the image in another window
		#Creates a window with window name "Face" and with the image img
		cv2.imshow(cam,frame)
		#Before closing it we need to give a wait command, otherwise the open cv wont work
		# @params with the millisecond of delay 1
		#cv2.waitKey(1)
		#To get out of the loop
		key=cv2.waitKey(50) & 0xFF
		if(key==ord("q")):
			break
	
	#Stoping the videostream
	vs.stop()

	# destroying all the windows
	cv2.destroyAllWindows()
	# update_attendance_in_db_out(present)
	# return redirect('home')
	return render(request,'recognition/markout.html')

# Create your views here.
def attendance_check(request,area):
	return render(request,'recognition/attendance_check.html')

def update_attendance(request):
	data_in={}
	data_out={}
	data=[]
	today=datetime.date.today()
	now = datetime.datetime.now()
	current_time = now.strftime("%Y-%m-%d %H:%M:00")
	time_qs=Time.objects.filter(date=today).filter(time__gte=current_time)
	out_data=time_qs.filter(out=True)
	in_data=time_qs.filter(out=False)
	for t in in_data:
		data_in[t.user.username]='<div class="col-6"><div class="card shadow-sm text-center"><h3 class=" text-center poppins  text-orange ">'+t.user.username+'</h3><h4 class=" text-center poppins "> '+t.user.get_full_name()+' </h4><p class="text-muted">'+str(t.time.strftime("%d-%m-%Y %H:%M:%S"))+'</p></div></div>'
		# data_in+='<div class="col-6"><div class="card shadow-sm text-center"><h3 class=" text-center poppins  text-orange ">'+t.user.username+'</h3><h4 class=" text-center poppins "> '+t.user.get_full_name()+' </h4><p class="text-muted">'+str(t.time.strftime("%d-%m-%Y %H:%M:%S"))+'</p></div></div>'
	data.append(data_in)
	for t in out_data:
		data_out[t.user.username]='<div class="col-6"><div class="card shadow-sm text-center"><h3 class=" text-center poppins  text-orange ">'+t.user.username+'</h3><h4 class=" text-center poppins "> '+t.user.get_full_name()+' </h4><p class="text-muted">'+str(t.time.strftime("%d-%m-%Y %H:%M:%S"))+'</p></div></div>'
	data.append(data_out)
	return HttpResponse(json.dumps(data))
def home(request):
	return redirect('login')
	# return render(request, 'recognition/home.html')

@login_required
def view_my_attendance_employee_login(request):
	if request.user.username=='admin':
		return redirect('not-authorised')
	qs=None
	time_qs=None
	present_qs=None
	u=request.user
	in_time=None
	out_time=None
	presenttoday=None
	time_qs=Time.objects.filter(user=u)
	present_qs=Present.objects.filter(user=u)
	in_time=time_qs.filter(date=datetime.date.today()).filter(out=False).order_by('time')
	out_time=time_qs.filter(date=datetime.date.today()).filter(out=True).order_by('time')
	in_time=in_time.first()
	out_time=out_time.last()
	presenttoday=present_qs.filter(date=datetime.date.today())
	
	if request.method=='POST':
		form=DateForm_2(request.POST)
		if form.is_valid():
			
			date_from=form.cleaned_data.get('date_from')
			date_to=form.cleaned_data.get('date_to')
			if date_to < date_from:
				messages.warning(request, f'Invalid date selection.')
				return redirect('view-my-attendance-employee-login')
			
			time_qs=time_qs.filter(date__gte=date_from).filter(date__lte=date_to).order_by('-date')
			present_qs=present_qs.filter(date__gte=date_from).filter(date__lte=date_to).order_by('-date')
			if (len(time_qs)>0 or len(present_qs)>0):
				qs=hours_vs_date_given_employee(present_qs,time_qs,admin=False)
				return render(request,'recognition/employee_dashboard.html', {'form' : form, 'qs' :qs,'intime' :in_time,'outtime':out_time,'presenttoday':presenttoday})
			messages.warning(request, f'No records for selected duration.')
			return redirect('view-my-attendance-employee-login')
	form=DateForm_2()
	return render(request,'recognition/employee_dashboard.html', {'form' : form, 'intime' :in_time,'outtime':out_time,'presenttoday':presenttoday})

@login_required
def dashboard(request):
	if(request.user.username=='admin'):
		print("admin")
		data=attendance_log()
		total_num_of_emp=total_number_employees()
		emp_present_today=employees_present_today()
		this_week_emp_count_vs_date()
		last_week_emp_count_vs_date()
		return render(request,"recognition/test_dashboard.html", {'total_num_of_emp' : total_num_of_emp, 'emp_present_today': emp_present_today,'alog':data})
	# return render(request, 'recognition/admin_dashboard.html')
	print("not admin")
	return redirect('view-my-attendance-employee-login')

@login_required
def registeremp(request):
	if(request.user.username=='admin'):
		print("admin")
		return render(request, 'recognition/admin_dashboard.html')
	print("not admin")
	return render(request,'recognition/employee_dashboard.html')
@login_required
def control(request):
	if(request.user.username=='admin'):
		print("admin")
		emp_present_today=employees_present_today()
		in_cameras=Camera.objects.filter(cameratype="In")
		out_cameras=Camera.objects.filter(cameratype="Out")
		if in_cameras:
			in_cameras=[ic.id for ic in in_cameras]
		if out_cameras:
			out_cameras=[oc.id for oc in out_cameras]
		return render(request, 'recognition/control_panel.html',{'emp_present_today': emp_present_today,'in_cameras':in_cameras,'out_cameras':out_cameras})
	print("not admin")
	return redirect('/')

@login_required
def markin(request):
	in_cameras=Camera.objects.filter(cameratype="In")
	if in_cameras:
		in_cameras=[ic.id for ic in in_cameras]
	return render(request,"recognition/markin.html", {'in_cameras' : in_cameras})

@login_required
def markout(request):
	out_cameras=Camera.objects.filter(cameratype="Out")
	if out_cameras:
		out_cameras=[oc.id for oc in out_cameras]
	return render(request,"recognition/markout.html", {'out_cameras' : out_cameras})
@login_required
def start_cameras(request):
	in_cameras=Camera.objects.filter(cameratype="In")
	if in_cameras:
		in_cameras=[ic.id for ic in in_cameras]
	out_cameras=Camera.objects.filter(cameratype="Out")
	if out_cameras:
		out_cameras=[oc.id for oc in out_cameras]
	return render(request,"recognition/start.html", {'in_cameras' : in_cameras,'out_cameras' : out_cameras})
@login_required
def viewemp(request):
	records=None
	if(request.user.username=='admin'):
		print("admin")
		rec=User.objects.all()
		data=[]
		for u in rec:
			udata={'user':u,'in':None,'out':None}
			today=datetime.date.today()
			in_time=None
			out_time=None
			time_qs=Time.objects.filter(user=u)
			in_time=time_qs.filter(date=today).filter(out=False).order_by('time')
			out_time=time_qs.filter(date=today).filter(out=True).order_by('time')
			in_time=in_time.last()
			
			out_time=out_time.last()
			if in_time:
				udata['in']=in_time.time
			if out_time:
				udata['out']=out_time.time
			data.append(udata)
		return render(request, 'recognition/view_employee.html',{'qs' : rec,'udata':data})

	print("not admin")
	return render(request,'/')
@login_required
def delete_employee(request,user):
	if(request.user.username=='admin'):
		print("admin")
		rec=None
		folder="face_recognition_data/training_dataset/"
		try:
			rec=User.objects.get(username=user)
			rec.delete()
			if user in os.listdir(folder):
				shutil.rmtree(folder+user)

		except Exception as e:
			print(e)
			print("Employee Not found")
		return render(request, 'recognition/view_employee.html')

	print("not admin")
	return render(request,'/')
@login_required
def add_photos(request):
	if request.user.username!='admin':
		return redirect('not-authorised')
	if request.method=='POST':
		form=usernameForm(request.POST)
		data = request.POST.copy()
		username=data.get('username')
		if username_present(username):
			create_dataset(username)
			messages.success(request, f'Dataset Created')
			return redirect('add-photos')

		messages.warning(request, f'No such username found. Please register employee first.')
		return redirect('add-photos')

	form=usernameForm()
	return render(request,'recognition/add_photos.html', {'form' : form})


@login_required
def add_camera(request):
	if request.user.username!='admin':
		return redirect('not-authorised')
	if request.method=='POST':
		form=CameraForm(request.POST)
		if form.is_valid():
			form.save() ###add camera to database
			messages.success(request, f'Camera Added successfully!')
			return redirect('control')
	else:
		form=CameraForm()
	return render(request,'recognition/add_camera.html', {'form' : form})

@login_required
def train(request):
	if request.user.username!='admin':
		return redirect('not-authorised')
	src=""
	training_dir='face_recognition_data/training_dataset'

	count=0
	for person_name in os.listdir(training_dir):
		curr_directory=os.path.join(training_dir,person_name)
		if not os.path.isdir(curr_directory):
			continue
		for imagefile in image_files_in_folder(curr_directory):
			count+=1

	X=[]
	y=[]
	i=0
	for person_name in os.listdir(training_dir):
		print(str(person_name))
		curr_directory=os.path.join(training_dir,person_name)
		if not os.path.isdir(curr_directory):
			continue
		for imagefile in image_files_in_folder(curr_directory):
			print(str(imagefile))
			image=cv2.imread(imagefile)
			try:
				X.append((face_recognition.face_encodings(image)[0]).tolist())		
				y.append(person_name)
				i+=1
			except Exception :
				print("removed")
				os.remove(imagefile)

	targets=np.array(y)
	encoder = LabelEncoder()
	encoder.fit(y)
	y=encoder.transform(y)
	X1=np.array(X)
	print("shape: "+ str(X1.shape))
	np.save('face_recognition_data/classes.npy', encoder.classes_)
	svc = SVC(kernel='linear',probability=True)
	svc.fit(X1,y)
	svc_save_path="face_recognition_data/svc.sav"
	with open(svc_save_path, 'wb') as f:
		pickle.dump(svc,f)
	src=vizualize_Data(X1,targets)
	
	return HttpResponse(src)
	# return render(request,"recognition/train.html")

@login_required
def train_modal(request):
	if request.user.username!='admin':
		return redirect('not-authorised')
	return render(request,"recognition/train1.html")
	

@login_required
def not_authorised(request):
	return render(request,'recognition/not_authorised.html')

@login_required
def view_attendance_home(request):
	total_num_of_emp=total_number_employees()
	emp_present_today=employees_present_today()
	this_week_emp_count_vs_date()
	last_week_emp_count_vs_date()
	return render(request,"recognition/view_attendance_home.html", {'total_num_of_emp' : total_num_of_emp, 'emp_present_today': emp_present_today})

@login_required
def view_attendance_date(request):
	if request.user.username!='admin':
		return redirect('not-authorised')
	qs=None
	time_qs=None
	present_qs=None

	if request.method=='POST':
		form=DateForm(request.POST)
		if form.is_valid():
			date=form.cleaned_data.get('date')
			print("date:"+ str(date))
			time_qs=Time.objects.filter(date=date)
			present_qs=Present.objects.filter(date=date)
			if(len(time_qs)>0 or len(present_qs)>0):
				qs=hours_vs_employee_given_date(present_qs,time_qs)
				return render(request,'recognition/view_attendance_date.html', {'form' : form,'qs' : qs })
			
			messages.warning(request, f'No records for selected date.')
			return redirect('view-attendance-date')

	form=DateForm()
	return render(request,'recognition/view_attendance_date.html', {'form' : form, 'qs' : qs})

@login_required
def view_attendance_employee(request):
	if request.user.username!='admin':
		return redirect('not-authorised')
	time_qs=None
	present_qs=None
	qs=None

	if request.method=='POST':
		form=UsernameAndDateForm(request.POST)
		if form.is_valid():
			username=form.cleaned_data.get('username')
			if username_present(username):
				
				u=User.objects.get(username=username)
				
				time_qs=Time.objects.filter(user=u)
				present_qs=Present.objects.filter(user=u)
				date_from=form.cleaned_data.get('date_from')
				date_to=form.cleaned_data.get('date_to')
				
				if date_to < date_from:
					messages.warning(request, f'Invalid date selection.')
					return redirect('view-attendance-employee')
				
				time_qs=time_qs.filter(date__gte=date_from).filter(date__lte=date_to).order_by('-date')
				present_qs=present_qs.filter(date__gte=date_from).filter(date__lte=date_to).order_by('-date')
					
				if (len(time_qs)>0 or len(present_qs)>0):
					qs=hours_vs_date_given_employee(present_qs,time_qs,admin=True)
					return render(request,'recognition/view_attendance_employee.html', {'form' : form, 'qs' :qs})
			
				messages.warning(request, f'No records for selected duration.')
				return redirect('view-attendance-employee')
		
			print("invalid username")
			messages.warning(request, f'No such username found.')
			return redirect('view-attendance-employee')
	
	form=UsernameAndDateForm()
	return render(request,'recognition/view_attendance_employee.html', {'form' : form, 'qs' :qs})


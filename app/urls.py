from django.contrib import admin
from django.urls import path
from app import views

urlpatterns = [
    path('',views.index,name='index'),
    path('log_in',views.log_in,name='log_in'),
    path('teacher_log_in',views.teacher_log_in,name='teacher_log_in'),
    path('register',views.register,name='register'),
    path('teacher_register',views.teacher_register,name='teacher_register'),
    path('student_dashboard',views.student_dashboard,name='student_dashboard'),
    path('teacher_dashboard',views.teacher_dashboard,name='teacher_dashboard'),
    path('create_exam',views.create_exam,name='create_exam'),
    path('upload',views.upload,name='upload'),
    path('submit_exam', views.submit_exam, name='submit_exam'),
    path('response', views.response, name='response'),
    path('view_blockchain', views.view_blockchain, name='view_blockchain'),
    path('open_camera', views.open_camera, name='open_camera'),
    path('send-results/<int:student_id>/',views.send_results, name='send_results'),
    path('view_result/', views.view_result, name='view_result'),
    #path('detect_emotion', views.detect_emotion, name='detect_emotion'),
    path('video_feed/', views.VideoStreamView.as_view(), name='video_feed'),
    path('submit_exam/', views.submit_exam, name='submit_exam'),


    path('logout/', views.log_out, name='log_out'),
    path('hash-lookup/', views.hash_lookup, name='hash_lookup'),
    path('verify/', views.verify_data, name='verify_data'),

]

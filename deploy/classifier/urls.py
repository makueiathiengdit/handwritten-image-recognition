from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('upload/', views.upload, name='upload'),
    path('images/', views.images, name='images'),
    path('success/', views.success, name='success'),
    path('m2/', views.m2, name="m2")
]
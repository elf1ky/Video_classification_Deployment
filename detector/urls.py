from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
]
# this code is from the website https://docs.djangoproject.com/en/3.0/intro/tutorial01/   that is where i learned everything about all this django stuff
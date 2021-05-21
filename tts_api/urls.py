from django.urls import path 
from . import views

app_name = 'tts_api' 
urlpatterns = [ 
    path('tts', views.tts_transcription),
]

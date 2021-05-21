import os
import sys
sys.path.append('/home/web/tacotron_models/')

from django.shortcuts import render
from tacotron_models.handler import TacotronHandler
from rest_framework.response import Response
from rest_framework.decorators import api_view
from rest_framework import status
from django.http import HttpResponse

tacotron_handler = TacotronHandler()
tacotron_handler.initialize()

@api_view(['GET'])
def tts_transcription(request):
    text = request.query_params.get('text')
    input_sequence = tacotron_handler.preprocess(text)

    output_audio = tacotron_handler.inference(input_sequence)
    file_path = tacotron_handler.postprocess(output_audio)

    if os.path.exists(file_path):
        with open(file_path, 'rb') as fh:
            response = HttpResponse(fh.read(), content_type="audio/wav")
            response['Content-Length'] = os.path.getsize(file_path)
            response['Content-Disposition'] = 'attachment; filename="sample.wav"'
            return response

    return Response(status=status.HTTP_200_OK)

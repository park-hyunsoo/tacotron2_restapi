import sys
sys.path.append('/home/web/tacotron_models/')

from django.shortcuts import render
from tacotron_models.handler import TacotronHandler
from rest_framework.response import Response
from rest_framework.decorators import api_view
from rest_framework import status

tacotron_handler = TacotronHandler()
tacotron_handler.initialize()

@api_view(['GET'])
def tts_transcription(request):
    text = request.query_params.get('text')
    input_sequence = tacotron_handler.preprocess(text)

    output_audio = tacotron_handler.inference(input_sequence)
    store_path = tacotron_handler.postprocess(output_audio)

    return Response(status=status.HTTP_200_OK)

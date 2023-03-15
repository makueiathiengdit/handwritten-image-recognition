import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image as PIL_Image

# django imports
from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.views import generic
from .forms import ImageIndexForm
from .models import Image
from .forms import *


# rest things
# from rest_framework.decorators import api_view
# from rest_framework.response import Response
# from rest_framework import status
# from .models import NoteSerializer

def to_np_array(arr):
    return np.array(arr)
def reshape_arr(arr):
    return  arr.reshape(len(arr), 28,28)   

model = None
model_status = 'unavailable'
submission = {}

class ModelUnavailableError(Exception):
    pass

try:
    test_data_path = "classifier/compiled_model/test.csv"
    model_path = "classifier/compiled_model/model.sav"
    
    #load model
    model = pickle.load(open(model_path, 'rb'))

    if model:
        print("model loaded successfully")
            
        df_test = pd.read_csv(test_data_path, nrows=80, skiprows=1)  
       
        df_test = np.asarray(df_test).astype(np.float32)
        df_test = df_test.reshape(len(df_test),28,28,1)
        first_img = PIL_Image.fromarray(df_test[2], 'RGB')
        print(first_img.size)
        first_img.save('first_img.png')
        first_img.show()
        results = model.predict(df_test)


        predicted_val=[]
        for i in range(len(results)):
            predicted_val.append(np.argmax(results[i]))
        
        img_ids = []
        for i in range(1, len(predicted_val)):
            img_ids.append(i)
            
        submission["image_id"] = img_ids
        submission["label"] = predicted_val

        # print(submission)
        model_status = 'available'
    else:
        raise  ModelUnavailableError("Model unavailable")
    
except Exception as e:
    print("\n====Print exception occured here======\n")
    print(e)
    print("\n")


class IndexView(generic.ListView):
    template_name = "classifier/index.html"
    context_object_name = "model_status"

    def get_queryset(self):
        pass
        # return super().get_queryset()
    


class ResultView(generic.DetailView):
    template_name = "classifier/result.html"
    context_object_name = 'result'


def index(request):
    img_id = None
    predicted_label = None
    prediction = {}
    if request.method == 'POST':
        form = ImageIndexForm(request.POST)
        # print(request.POST)
        if form.is_valid():
            print(f"got form data with {request.POST['image_index']}")
            img_id = int(request.POST['image_index'])
            if img_id <= len(submission['image_id']):
                predicted_label = submission['label'][img_id]
                prediction['image_id'] = img_id
                prediction['predicted_label'] = predicted_label
                prediction['error'] = None
            else:
                prediction['image_id'] = img_id
                prediction['predicted_label'] = predicted_label
                prediction['error'] = "no image with such label"

    context = {'model_status': model_status,\
                'results': submission,\
               'request':request.method,\
                'img_id':img_id, \
                'predicted_label': predicted_label,\
                'prediction': prediction}
    return render(request, 'classifier/index.html', context)
   



# def get_prediction(input):
#     if hasattr(model, 'predict'):
#         prediction = model.predict(input)

def upload(request):
    print(f"{request.method} request from : {request.get_host()}")
    form  = ImageForm()
    
    if request.method == 'POST':
        form = ImageForm(request.POST)
        if form.is_valid():
            form.save()
            print("accepted data")
            return redirect('success')
        else:
            form = ImageForm()
    return render(request, 'classifier/upload.html', {'form': form})    


def success(request):
    return render(request, 'classifier/success.html')

def images(request):
    _images = Image.objects.all()
    # if _images:
    #     print(len(_images))
    #     for img in _images:
    #         print(img.img)
    return render(request, 'classifier/images.html', {'images': _images})


def predict_digit(img, model):
    
    #resize image to 28x28 pixels
    img = img.resize((28,28))
    #convert rgb to grayscale
    
   
    img = img.convert('L')
    img = np.array(img)
    # plt.imshow(img)
    # save_image(img)
    #reshaping to support our model input and normalizing
    img = img.reshape(1,28,28,1)
    img = img/255.0
    
    #predicting the class
    res = model.predict([img])[0]
    
    return np.argmax(res), max(res)


def m2(request):
    from keras.models import load_model
    from PIL import Image
   
    
    model_path = "/home/surbhi/awet/web/deploy/classifier/compiled_model/mnist.h5"
    try:
        model = load_model(model_path)
        IMG_BASE_PATH = "/home/surbhi/awet/web/deploy/classifier/static/images/"

        if model:
            img_name = "six.png"
            img_path  = IMG_BASE_PATH + img_name
            img = Image.open(img_path)
            pred, acc = predict_digit(img, model)


            prediction = {"image":img_name,"predicted_class": pred, 'accurracy': acc}
            print(prediction)
            return render(request, 'classifier/results.html',prediction )
    except:
        print("something went wrong")
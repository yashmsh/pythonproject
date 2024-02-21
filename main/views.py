from django.shortcuts import render
from numpy import percentile
from .models import dataMain as Medicine
import urllib

from medDetectFold import mlModel

import matplotlib.pyplot as plt
plt.switch_backend('Agg')


from django.core.files import File
from django.core.files.temp import NamedTemporaryFile

# Create your views here.

def home(request):
    medicines = Medicine.objects
    flag = False
    title = "Remdesivir"
    if (Medicine.objects.count() > 0):
        flag = True

        if (medicines.first().title == "Remdesivir"):
            title = "Remdesivir"
    
        elif (medicines.first().title == "Coartem"):
            title = "Coartem"


    return render(request, 'main/home.html', {'medicines': medicines, 'flag': flag, 'title': title})

def check(request):
    if request.method == 'POST':
        if request.POST['title'] and request.FILES['image']:
            Medicine.objects.all().delete() #delete all previous data cause only 1 to be displayed
            medicine = Medicine()
            medicine.title = request.POST['title']
            medicine.image = request.FILES['image']
            medicine.Rx = "Rx Symbol Found"
            medicine.Spelling = "No Spelling Errors"
            medicine.Colour = "Colour nearly accurate"
            medicine.Alignment = "Correct Alignment"
            medicine.percentage = 90
            
            print(medicine.image.url)
            medicine.save()
            print(medicine.image.path)
            mlModel.implementModel(medicine.image.path)

            print(mlModel.warning)

            medicine.Rx = mlModel.warning[0]
            medicine.Colour = mlModel.warning[1]
            medicine.Spelling = mlModel.warning[2]
            medicine.percentage = mlModel.percentage_certainty[0]


            medicine.save()

            flag = False
            
            return render(request, 'main/home.html')
        else:
            return render(request, 'main/home.html',{'error':'All fields are required.'})
    else:
        return render(request, 'main/home.html')


# percentage_certainty = mlModel.percentage_certainty

# RxSymbolStatus = mlModel.warning[0]
# SpellingErrorStatus = mlModel.warning[1]
# fakeNotFakeStatus = mlModel.warning[2]


# print(percentage_certainty)

from django.db import models

class dataMain(models.Model):
    title = models.CharField(max_length = 255)
    image = models.ImageField(upload_to = 'images/')
    checkedImage = models.CharField(max_length = 255)
    Alignment = models.CharField(max_length = 255)
    Colour = models.CharField(max_length = 255)
    Rx = models.CharField(max_length = 255)
    Spelling = models.CharField(max_length = 255)
    percentage = models.IntegerField()

# Create your models here.

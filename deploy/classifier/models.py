from django.db import models
# from rest_framework import serializers
# Create your models here.
import random

def get_random_name():
    prefix = 'img_'
    nums =  ['0','1','2','3','4','5','6','7','8','9']
    suffix = ""
    for _ in range(len(nums)):
        suffix += random.choice(nums)
    return prefix + suffix    



class Image(models.Model):
    name = models.CharField(max_length=30, default=get_random_name)
    img = models.ImageField(upload_to='images/', default='cat.jpg')

    def __str__(self) -> str:
        return self.name


#
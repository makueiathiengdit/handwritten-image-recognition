from django import forms
from .models import  Image

class ImageIndexForm(forms.Form):
	pass
    # image_index = forms.IntegerField(label="Enter image index")

    # def __str__(self) -> str:
    #     return ImageIndexForm.image_index


class ImageForm(forms.ModelForm):
	class Meta:
		model = Image
		fields = ['name', 'img']

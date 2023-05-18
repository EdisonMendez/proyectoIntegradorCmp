from django import forms

class buscadorCodigo(forms.Form):
    codigoProductoPG = forms.CharField()
    nombreProductoPG = forms.CharField()
    stockActualPG = forms.CharField()
    codigoNPartePG = forms.CharField
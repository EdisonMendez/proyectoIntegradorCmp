from django.contrib.auth.forms import UserCreationForm
from django.http import HttpResponse as response
from django.template import Template
from django.template import Context
from django.shortcuts import render
from django.db.models import Q
from django.views import View
from gestionDB.models import infoGeneralProducto,infoProductoBodega #importacion del modelo para cargar datos a BDSQL3


from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.views.generic import TemplateView


def index(request):
    openHtml = open("templates/index.html", "r")
    plt = Template(openHtml.read())
    openHtml.close()
    context = Context()
    readIndex = plt.render(context)

    return response(readIndex)

def buscador(request):
    return render(request,"buscador.html")

def buscadorTmp(request):

    infoGneral = infoGeneralProducto.objects.all()
    search = request.GET.get("buscadorTmp")

    if search:
       infoGneral =  infoGeneralProducto.objects.filter(
           Q(codigoProducto__icontains=search) | #revisa cada campo del modelo
           Q(nombreProducto__icontains=search)

       ).distinct()

    else:
        mensaje = "No se encontraron resultados."
        return response(mensaje)

    return render(request,"buscadorTmp.html", {"search" : infoGneral, "query" : search})


def buscar(request):
    if(request.GET["buscadorTf"]):

        producto = request.GET["buscadorTf"]
        articulos = infoGeneralProducto.objects.filter(codigoProducto__icontains=producto)
        return render(request,"resultados_busqueda.html",{"articulos" :articulos, "query": producto})
    else:
        mensaje ="Por favor, ingrese datos."

    return render(mensaje)




def busquedaFinalCodigo(request):
    busquedaS = request.GET.get("buscar")
    gettingInfo = infoGeneralProducto.objects.all()

    if busquedaS:
        gettingInfo = infoGeneralProducto.objects.filter(
            Q(codigoProducto__icontains=busquedaS) |  # revisa cada campo del modelo
            Q(nombreProducto__icontains=busquedaS) |
            Q(codigoNParte__icontains=busquedaS)).distinct()
        return render(request, "resultados2.html", {"clave" : gettingInfo, "valor" : busquedaS})


    return render(request, "buscarInfo.html")


def login(request):
    openHtml = open("templates/login.html", "r")
    plt = Template(openHtml.read())
    openHtml.close()
    context = Context()
    reader = plt.render(context)

    return response(reader)

def registro2(request):
    openHtml = open("templates/registro.html", "r")
    plt = Template(openHtml.read())
    openHtml.close()
    context = Context()
    reader = plt.render(context)

    return response(reader)


def simple_upload(request):
    if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        uploaded_file_url = fs.url(filename)
        return render(request, 'cargarDocs.html', {
            'uploaded_file_url': uploaded_file_url
        })
    return render(request, 'cargarDocs.html')


def middleR(request):
    openHtml = open("templates/middle.html", "r")
    plt = Template(openHtml.read())
    openHtml.close()
    context = Context()
    reader = plt.render(context)

    return response(reader)



#class PostTemplateView(TemplateView):
#    template_name = 'index.html'






class registroUser(View):
    def get(self,request):
        form = UserCreationForm()
        return render(request,"registro.html")

    def __pos__(self, request):
        pass

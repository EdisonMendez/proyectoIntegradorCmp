import export as export
import pyodbc
import os

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "FinalProject.settings")
from django.core.wsgi import get_wsgi_application

application = get_wsgi_application()

import sqlite3
from sqlite3 import Error
from openpyxl import load_workbook
from django import setup
from os import path, environ
from sys import path as sys_path
from django.conf import settings
from gestionDB.models import infoGeneralProducto, infoProductoBodega
import FinalProject

conn = pyodbc.connect('Driver={4D v19 ODBC Driver 64-bit} ;Server=181.198.83.12 ;UID=API ;PWD=API')
curs = conn.cursor()
curs.execute(
    "select PRODUCT_NAME, PRODUCT_ID, AVAILABLE from INVT_Ficha_PrincipaL where AVAILABLE >= 0 order by AVAILABLE desc ")
if (curs):
    print("success")

productName = []
productId = []
available = []
nParteProducto = []

for row in curs:
    print(row)
    productName.append(row[0])
    productId.append(row[1])
    available.append(row[2])


def getNparteCod():
    print("PRODUCTOS BD: ")
    prodIdBD = productId.copy()
    prodNameBD = productName.copy()
    prodAvailableBD = available.copy()

    # print(prodIdBD)
    # print(prodNameBD)
    # print(prodAvailableBD)

    wb = load_workbook("numsParte.xlsx")
    sh = wb["Hoja1"]

    cod = []
    nombre = []
    nParteArr = []

    for index, item in enumerate(sh["A1":"A18881"]):
        if index > 0:
            for cell in item:
                nombre.append(str(cell.value))

    for index, item in enumerate(sh["B1":"B18881"]):
        if index > 0:
            for cell in item:
                cod.append(str(cell.value))

    for index, item in enumerate(sh["C1":"C18881"]):
        if index > 0:
            for cell in item:
                nParteArr.append(str(cell.value))

    #diccionario = dict(zip(cod, [prodNameBD,prodAvailableBD,nParteArr]))
    diccionario1 = dict(zip(cod, prodIdBD))
    diccionario2 = dict(zip(cod, prodNameBD))
    diccionario3 = dict(zip(cod, prodAvailableBD))
    diccionario4 = dict(zip(cod, nParteArr))

    dictTmp = dict(zip(cod,[prodNameBD,prodAvailableBD,nParteArr]))
    print(diccionario4.get('12050'))


    # buscar el valor esperado en el diccionario
    # de momento, esto me va a servir para aqui insertar datos en la tabla sql3lite
    # print(productId)

    # INSERCION DE DATOS EN LA BASE DE DATOS POSTGRE SQL
    for i in diccionario4:

         if (i in prodIdBD) and (i in diccionario4.keys()):

             sqlInstruction = infoGeneralProducto.objects.create(codigoProducto=diccionario1.get(i,i), nombreProducto= diccionario2.get(i,i),
                                                                 stock_actual=diccionario3.get(i,i), codigoNParte=diccionario4.get(i, i))


getNparteCod()

# forecasting o prediccion para el segundo avance

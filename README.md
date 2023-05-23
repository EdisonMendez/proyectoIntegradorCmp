# proyectoIntegradorCmp : Edison Méndez 214464
Proyecto integrador

El proyecto integrador tuvo 2 partes: 
# 1. Extracción, conexión y visualización de códigos y números de parte de productos SKU:
En esta parte se utiliza el framework django de Python para poder crear una aplicación web, en esta configuración también se enlazó una conexión a base de datos postgreSQL. La creación de las tablas de la base de datos, 
se hacen en el archivo gestionDB/models.py. 

Esta parte del proyecto integrador tiene como objetivo hacer un datawharehouse de datos que provienen de una base de datos SQL, y de una base de datos seteada en Excel. 

Sobre el código, dbConnect.py, se establece una conexión ODBC mediante una IP pública, para extraer la información de la base de datos productiva. De esta base, se extrae:
- Nombre del producto.
- ID del producto (conocido también como SKU).
- cantidad disponible (actualizada en tiempo real).

Adicional a lo antes mencionado, se utiliza también una base de datos en Excel. Dicho archivo fue provisionado por el área de calidad de la empresa en la que trabajo. De este archivo se extrae: 
- ID del producto (conocido también como SKU).
- Número de parte del producto. 

Con toda esa información a disposición, se hizo un diccionario para que con cada ejecución, cargue esta información a la base de datos configurada con PostgreSQL. 

Una vez realizada la carga exitosa a la base de datos, esta información es la que mostrará en el front-end de la aplicación de django.

En la carpeta templates, se encuentran los archivos HTML que contienen el proyecto web de django. Sobre el archivo resultados_busqueda.html, se realiza la consulta que se hará a la base de datos para traerla al sitio. 
Cuando la búsqueda sea exitosa, se arrojarán -dependiendo el caso-, el producto buscado y sus productos relacionados, es decir, no se arroja una búsqueda precisa debido a que, muchos productos pueden tener el mismo 
nombre, pero no el código. Por tal, si el usuario busca dentro de las coincidencias, también podrá ver el código exacto que buscó. 



# 2. Forecasting de ventas usando datos históricos de la compañía
Para esta parte, se usaron 6 archivos con la información de ventas de la compañía servicat, entre los años 2017-2022. Estos datos al ser muy pesados, no fueron cargados en este repositorio,
ya que GitHub no permite subir archivos con un peso superior a 20Mb. Estos archivos fueron extraídos directamente del sistema ERP MBA3, ERP de uso producto de la empresa. Por lo que los datos usados 
en este proyecto integrador fueron reales. 

Con todo lo antes dicho, se resume lo que hace cada código. 

En la carpeta coding, se usaron los siguientes archivos: 

## cleanData.py: 
- Los 6 archivos usados 2017.xlsx; 2018.xlsx; 2019.xlsx; 2020.xlsx; 2021.xlsx, 2022.xlsx tienen un formato de excel, XLSX y tiene 19 columnas. Por tal razón, primero se convirtieron estos datos a formato CSV, 
en este paso se eliminaron outiers para cada archivo, así como también se eliminaron columnas que resultaron innecesarias para el análisis. Las columnas que únicamente se tomaron en cuenta son:

Codigo,Nombre Codigo,Fecha Venta,bodega,Nombre cliente,Unidades_vendidas,precio,Venta (dolares)

Una vez procesados los archivos de forma independiente, se consolidó toda la información en un solo archivo, es decir, se hizo un merge de estos archivos. El archivo final se llama: historicsSalesSV.csv. 

## final_forecasting.py: 
Aquí se usa el archivo consolidado, historicsSalesSV.csv. Para reducir su numerosidad, se agrupan las ventas totales mes a mes, para cada año. Este nuevo dataframe es exportado como: ventasMensualesTotales.csv.
Con el dataframe de las ventas mensuales, se realiza el análisis exploratorio de datos. Se observa que en el año 2020 es en donde las ventas son las más bajas, esto se debe en gran parte por la crisis sanitaria declarada
en ese año por causa de la pandemia ocasionada por el COVID-19.

Luego sobre este conjunto de datos se aplica la técnica Dicker-Fuller test para determinar estacionalidad. A simple vista, los datos indican no ser estacionarios, por que se aplica la técnica de diferenciación para penalizar
los valores más altos y más bajos para que el conjunto de datos, tenga una media constante, y por tanto sea estacionaria. 

Después se aplica la técnica de distribuited-lags para crear un dataframe supervisado, esta técnica es muy utilizada para series de tiempo, que requieren convertir sus datos en un problema de aprendizaje supervisado. 
Básicamente lo que se hace es crear copias de las copias desplazándolas en t+1, aquí también se eliminan valores nulos. Este nuevo dataframe se exporta a: dfSupervisado.csv. 

Este nuevo dataframe es el que se usa para en el modelo. 

Para la división del dataset, en train y test se usó la técnica Time-Series-Split que básicamente hace es dividir de forma secuencial el dataset, para luego asignarlos como test o train. 

Se crea también una función que hará que una vez que la red se haya entrenado, arroje los resultados a un CSV, en el que se muestra la fecha, las ventas originales, la venta predicha, la diferencia entre uno y otro, y 
el porcentaje de error en la predicción (usando la fórmula de error porcentual).

Luego se crea una función que medirá el rendimiento que tuvo la red neuronal en el entrenamiento, estos datos servirán para ver qué tan precisos fueron los resultados de la predicción. 
Las métricas usadas son: 
- RMSE: error cuadrático medio, mide la cantidad de error entre dos conjuntos de datos.
- MAE: Es la diferencia entre el valor pronosticado y el valor real en cada punto pronosticado.
- R^2: Es el coeficiente de determinación que mide la capacidad de un modelo para predecir futuros resultados, los valores de la métrica varían entre 1-0, siendo 1 el mejor resultado posible medido. Esto ocurre cuando la predicción coincide con los valores de la variable objetivo. 
- MAPE: Error porcentual medio absoluto se utiliza para medir la precisión de un modelo, cuanto menor sea el valor de MAPE, mejor será el modelo para predecir resultados

También se creó 2 funciones para mostrar el comportamiento que tuvo el modelo en el entrenamiento, la curva Loss VS Eppoch. Una sirve para plotear la gráfica, y otra para manejar un historial en base a los valores de 
loss y val loss que aparecen cuando la red se está entrenando. 

La red neuronal LSTM se creó con la siguiente arquitectura:

- 1 capa de entrada LSTM con 4 neuronas, 1 capa oculta dense con función de activación tanh, con 3 neuronas, y una capa dense de salida con una neurona. 
- optimizador:SGD 
- loss: MSE
- épocas/eppochs: 200. 

Y finalmente, se hizo un plot para mostrar gráficamente una curva de color azul con las ventas originales, y otra curva de color rojo indicando las ventas que se predijeron con la red neuronal. Esta es una forma muy intui-
tiva de ver qué tan precisa fue la red neuronal comparándola con los resultados que se obtuvieron en la medición de errores descritas anteriormente. 


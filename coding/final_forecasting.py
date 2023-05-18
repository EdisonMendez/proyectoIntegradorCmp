import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow
from keras import preprocessing
from statsmodels.tsa.stattools import adfuller
import seaborn as sns

from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score,mean_absolute_percentage_error

from keras.layers import Dense
from keras.models import Sequential
from keras.layers import LSTM
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def sumByMonth():
    df = pd.read_csv('converted/historicsSalesSV.csv')
    df_s = df.copy()

    f0 = df.pop('Fecha Venta')
    df_s.insert(0, 'date', f0)
    df_s['date'] = pd.to_datetime(df_s['date'])
    print(df_s.info)
    print(df_s)

    # dfVentasMes = df_s.resample('M'),sum()
    # df_s['date'] = pd.to_datetime(df_s['date'])
    df_s = df_s.set_index('date')
    dfVentasMes = df_s['Unidades vendidas'].resample('M').sum()
    dfVentasMes.to_csv('ventasMensualesTotales.csv')
    print('done')



#análisis exploratorio de datos
def dExplorer(df):
    # ANÁLISIS EXPLORATORIO DE DATOS
    df_2017 = df.loc[df['date'].dt.year == 2017]
    df_2018 = df.loc[df['date'].dt.year == 2018]
    df_2019 = df.loc[df['date'].dt.year == 2019]
    df_2020 = df.loc[df['date'].dt.year == 2020]
    df_2021 = df.loc[df['date'].dt.year == 2021]
    df_2022 = df.loc[df['date'].dt.year == 2022]

    fig, (ax1) = plt.subplots(figsize=(8, 6))
    ax1.plot(df_2017['Unidades vendidas'], "o-", label='2017')
    ax1.plot(df_2018['Unidades vendidas'], "o-", label='2018')
    ax1.plot(df_2019['Unidades vendidas'], "o-", label='2019')
    ax1.plot(df_2020['Unidades vendidas'], "o-", label='2020')
    ax1.plot(df_2021['Unidades vendidas'], "o-", label='2021')
    ax1.plot(df_2022['Unidades vendidas'], "o-", label='2022')
    plt.title('ventas del 2017 al 2022')
    plt.legend()
    plt.show()



def diff(df_diff):
    df_diff['unidadesVendidas_diff'] = df_diff['Unidades vendidas'].diff()
    df_diff = df_diff.dropna()

    df_diff.to_csv('diff_unidadesVendidas.csv', index=False)
    print('DIFF SALES: ')
    print(df_diff)

    toSupervised(df_diff)


#funcion para determinar si la serie es estacionaria o no
def testDickerFuller():

    df_m_diff = df_s['Unidades vendidas'].diff()
    nDf = df_m_diff.dropna().reset_index(drop=True)

    rolmean = nDf.rolling(12).mean()
    rolstd = nDf.rolling(12).std()

    fig,ax = plt.subplots()
    orig = ax.plot(nDf, color='blue', label='Original')
    mean = ax.plot(rolmean, color='red', label='Rolling Mean')
    plt.legend(loc='best')
    plt.title('Rolling Mean de ventas')
    plt.show(block=False)
    print('rolling mean: ' + str(rolmean.mean()))
    # Perform Dickey-Fuller test:
    print('Resultados del Dickey-Fuller Test:')
    dftest = adfuller(nDf, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Valor de test estadístico: ', 'valor-p:', '#Lags usados: ',
                                             'Número de observaciones utilizadas: '])

    for key, value in dftest[4].items():
        dfoutput['Valores críticos (%s)' % key] = value
    print(dfoutput)


def toSupervised(df):
    dfSupervised = df.copy()
    for i in range(0, 13):
        nombreCol = 't+' + str(i)
        dfSupervised[nombreCol] = df['Unidades vendidas'].shift(i)
        # print(df)

    # quitar nulos
    dfSupervised = dfSupervised.dropna().reset_index(drop=True)
    dfSupervised.to_csv('dfSupervisado.csv', index=False)
    print('done df supervisado')
    print()
    print(dfSupervised.info())

    return dfSupervised



#################CREACION DEL MODELO#####################
def preprocessingDf(train,test):

    # normalizacion
    minMax = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    minMax = minMax.fit(train,test)

    # matriz np(mxn).Toma las 47 filas shape[0], y las 14 columnas(unidades vendidas_diff+ lag_1:12 shape[1])
    train = train.reshape(train.shape[0],train.shape[1])

    trainScaled = minMax.transform(train)#escala/normaliza la matriz de train con MinMaxScaler

    #reshape test
    test = test.reshape(test.shape[0],test.shape[1])
    testScaled = minMax.transform(test)

    X_train = trainScaled[:,1:] #selecciona todas las columnas a excepcion de la primera. Esto crea una nueva matriz numpy X_train que contiene los valores de características de los datos de entrenamiento.
    Y_train = trainScaled[:,0:1].ravel() #selecciona solo la columna de unidades vendidas_diff a excepción del 2022.

    X_test = testScaled[:,1:]
    Y_test = testScaled[:,0:1].ravel()


    return X_train,Y_train,X_test,Y_test,minMax


def reverseProcessing(y_pred, x_test,scaler):

    y_pred = y_pred.reshape(y_pred.shape[0],1,1)

    pred_test = []

    #Reconstruir conjunto de prueba para transformada inversa:
    for i in range(0,len(y_pred)):
        pred_test.append(np.concatenate([y_pred[i],x_test[i]],axis=1))

    #reshape
    pred_test = np.array(pred_test)
    pred_test = pred_test.reshape(pred_test.shape[0],pred_test.shape[2])

    #transformacion inversa
    predInverseTransform = scaler.inverse_transform(pred_test)

    return predInverseTransform

#se muestran las ventas predichas para cada mes
def predictDf(unscaledPred, df_mes):

    # desde el año 2017 al año 2021 se usaron para el test, y el año 2022 para el train:
    resultados = []
    fecha_venta = list(df_mes[-13:].date)
    venta_real = list(df_mes['Unidades vendidas'][-13:])


    for i in range(0,len(unscaledPred)):
        dicResults = {}
        dicResults['date'] = fecha_venta[i + 1]
        dicResults['org_value'] = venta_real[i]
        dicResults['pred_value'] = int(unscaledPred[i][0] + venta_real[i])
        dicResults['diff[org-pred]'] = dicResults['org_value'] - dicResults['pred_value']
        dicResults['%error'] = round((abs((dicResults['org_value'] - dicResults['pred_value'])/dicResults['org_value'])*100),3)

        resultados.append(dicResults)

    dfResult = pd.DataFrame(resultados)
    dfResult.to_csv('resultados_prediccionDf.csv',index=False)

    return dfResult

def scoreModel(unscaledData,dfMesoriginal):

    rmse = np.sqrt(mean_squared_error(dfMesoriginal['Unidades vendidas'][-12:],unscaledData.pred_value[-12:]))
    mae = mean_absolute_error(dfMesoriginal['Unidades vendidas'][-12:],unscaledData.pred_value[-12:])
    r2 = r2_score(dfMesoriginal['Unidades vendidas'][-12:],unscaledData.pred_value[-12:])
    mape = mean_absolute_percentage_error(dfMesoriginal['Unidades vendidas'][-12:],unscaledData.pred_value[-12:])

    scores = {}
    scores['lstm:'] = [rmse,mae,r2,mape]

    print("RMSE: " + str(rmse))
    print("MAE: " + str(mae))
    print("R2: " + str(r2))
    print("MAPE: " + str(mape))


def plotResultados(results,origin_df):

    fig, ax = plt.subplots(figsize=(15, 5))
    sns.lineplot(data=origin_df, x=origin_df.date, y=origin_df['Unidades vendidas'], ax=ax,
                 label='Original', color='blue')
    sns.lineplot(data=results, x=results.date, y=results.pred_value, ax=ax,
                 label='Predicted', color='red')

    ax.set(xlabel="Año",
           ylabel="Ventas",
           title=f" LSTM Ventas vs Año")

    ax.legend(loc='best')

    filepath = Path('./model_output/LSTM_forecasting_SV.svg')
    filepath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(f'./model_output/LSTM_forecasting_SV.svg')
    plt.show()


def loss_epoch_plot(loss, val_loss):

    fig,ax = plt.subplots()
    ax.plot(loss, label='Training')
    ax.plot(val_loss, label='Validation')
    plt.xlabel("Epoch")
    plt.ylabel('Mean of Loss')
    plt.legend()
    plt.savefig('loss_epoch_SV' + '.png')
    plt.show()
    #plt.close()

def manage_history(data):
    aux_loss = []
    aux_val_loss = []
    for h in data:
        temp_loss = h.history['loss']
        aux_loss.append(temp_loss)

        temp_loss = h.history['val_loss']
        aux_val_loss.append(temp_loss)

    aux_loss = np.array(aux_loss)
    loss_mean = np.mean(aux_loss, axis=0)

    aux_val_loss = np.array(aux_val_loss)
    val_loss_mean = np.mean(aux_val_loss, axis=0)

    loss_epoch_plot(loss_mean, val_loss_mean)



def redNeuronalLstm(toTrain, toTest):

    X_train, y_train, X_test, y_test, pivot = preprocessingDf(toTrain, toTest)


    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1]) #se convierte la matriz 2D a 3D
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

    model = Sequential()
    model.add(LSTM(4, batch_input_shape=(1, X_train.shape[1], X_train.shape[2]),
                   stateful=True))
    model.add(Dense(1, activation='tanh'))
    model.add(Dense(1))
    model.summary()
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    history = model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=200, batch_size=1, verbose=1,shuffle=False)

    aux_history = []
    aux_history.append(history)
    manage_history(aux_history)


    predictions = model.predict(X_test, batch_size=1)
    print('predictions LSTM: ')
    print(predictions)

    origin_df = df_s
    unscaled = reverseProcessing(predictions, X_test, pivot)

    unscaled_df = predictDf(unscaled, origin_df)

    print('unscaled df:')
    print(unscaled_df)

    scoreModel(unscaled_df, origin_df)
    plotResultados(unscaled_df, origin_df)



if __name__ == '__main__':

    #sumByMonth()
    dfTmpMonth = pd.read_csv('ventasMensualesTotales.csv', usecols=['date', 'Unidades vendidas']) #archivo que se genera en el método sumByMoth, aquí empieza el análisis
    df_s = dfTmpMonth.copy()
    df_s['date'] = pd.to_datetime(df_s['date'], format='%Y-%m-%d')

    dExplorer(df_s)     #data exploration
    diff(df_s)          #se obtiene la diferencia de los datos para convertirlos en estacionarios
    testDickerFuller()  #se verifica si es o no estacionario

    df_supervisado = pd.read_csv('dfSupervisado.csv') #archivo que se genera en el método diff con la llamada al método toSupervised, aquí ya se maneja con datos estacionarios

    df_supervisadoData = df_supervisado.drop(['Unidades vendidas','date'],axis=1)
    train = df_supervisadoData[:-12].values
    test = df_supervisadoData[-12:].values

    redNeuronalLstm(train, test)


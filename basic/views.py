from django.shortcuts import render, redirect
from .forms import NewUserForm
from django.contrib.auth import login
from django.contrib.auth import login, authenticate, logout  # add this
from django.contrib import messages
from django.contrib.auth.forms import AuthenticationForm  # add this
from django.contrib.auth.decorators import login_required
from keras.models import Sequential
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
import pandas as pd
import numpy as np
from matplotlib.pylab import rcParams
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from keras.models import load_model
from pandas_datareader.data import DataReader
import csv
from .models import *
from plotly.offline import plot
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error
import math

# Create your views here.


# @login_required
def makemodel(s_type):
    rcParams["figure.figsize"] = 20, 10

    scaler = MinMaxScaler(feature_range=(0, 1))

    df = pd.read_csv("data.csv")
    df.head()

    df["Date"] = pd.to_datetime(df.Date, format="%Y-%m-%d")
    df.index = df["Date"]

    # plt.figure(figsize=(16,8))
    # plt.plot(df["Close"],label='Close Price history')

    data = df.sort_index(ascending=True, axis=0)
    new_dataset = pd.DataFrame(index=range(0, len(df)), columns=["Date", s_type])
    for i in range(0, len(data)):
        new_dataset["Date"][i] = data["Date"][i]
        new_dataset[s_type][i] = data[s_type][i]

    new_dataset.index = new_dataset.Date
    new_dataset.drop("Date", axis=1, inplace=True)

    final_dataset = new_dataset.values

    t = int(0.7 * len(final_dataset))
    train_data = final_dataset[0:t, :]
    valid_data = final_dataset[t:, :]

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(final_dataset)

    x_train_data, y_train_data = [], []

    for i in range(60, len(train_data)):
        x_train_data.append(scaled_data[i - 60 : i, 0])
        y_train_data.append(scaled_data[i, 0])

    x_train_data, y_train_data = np.array(x_train_data), np.array(y_train_data)

    x_train_data = np.reshape(
        x_train_data, (x_train_data.shape[0], x_train_data.shape[1], 1)
    )

    lstm_model = Sequential()
    lstm_model.add(
        LSTM(units=50, return_sequences=True, input_shape=(x_train_data.shape[1], 1))
    )
    lstm_model.add(LSTM(units=50))
    lstm_model.add(Dense(1))

    print(x_train_data, y_train_data)
    lstm_model.compile(loss="mean_squared_error", optimizer="adam")
    lstm_model.fit(x_train_data, y_train_data, epochs=1, batch_size=1, verbose=2)

    inputs_data = new_dataset[len(new_dataset) - len(valid_data) - 60 :].values
    inputs_data = inputs_data.reshape(-1, 1)
    inputs_data = scaler.transform(inputs_data)
    X_test = []
    for i in range(60, inputs_data.shape[0]):
        X_test.append(inputs_data[i - 60 : i, 0])
    X_test = np.array(X_test)

    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    closing_price = lstm_model.predict(X_test)
    closing_price = scaler.inverse_transform(closing_price)

    lstm_model.save("saved_model.h5")

    train_data = new_dataset[:t]
    valid_data = new_dataset[t:]
    valid_data["Predictions"] = closing_price
    # plt.plot(train_data["Close"])
    # plt.plot(valid_data[['Close',"Predictions"]])
    # plt.show()
    # print(closing_price)


def getdata(symbol, startdate, enddate):
    df = DataReader(
        symbol, "av-daily", start=startdate, end=enddate, api_key="ALPHAVANTAGE_API_KEY"
    )
    rows = df.to_string(header=False).split("\n")

    # field names
    fields = ["Date", "Open", "High", "Low", "Close", "Volume"]

    # data rows of csv file
    with open("data.csv", "w") as f:
        write = csv.writer(f)
        write.writerow(fields)
        for i in range(len(rows)):
            df2 = rows[i].split(" ")
            df2 = " ".join(df2).split()
            df2 = list(df2)
            write.writerow(df2)


def drawgraph(s_type):
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_nse = pd.read_csv("data.csv")

    df_nse["Date"] = pd.to_datetime(df_nse.Date, format="%Y-%m-%d")
    df_nse.index = df_nse["Date"]

    data = df_nse.sort_index(ascending=True, axis=0)
    new_data = pd.DataFrame(index=range(0, len(df_nse)), columns=["Date", s_type])

    t = int(0.7 * len(df_nse))

    for i in range(0, len(data)):
        new_data["Date"][i] = data["Date"][i]
        new_data[s_type][i] = data[s_type][i]

    new_data.index = new_data.Date
    new_data.drop("Date", axis=1, inplace=True)

    dataset = new_data.values

    train = dataset[0:t, :]
    valid = dataset[t:, :]

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    x_train, y_train = [], []

    for i in range(60, len(train)):
        x_train.append(scaled_data[i - 60 : i, 0])
        y_train.append(scaled_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    model = load_model("saved_model.h5")

    inputs = new_data[len(new_data) - len(valid) - 60 :].values
    inputs = inputs.reshape(-1, 1)
    inputs = scaler.transform(inputs)

    X_test = []
    for i in range(60, inputs.shape[0]):
        X_test.append(inputs[i - 60 : i, 0])
    X_test = np.array(X_test)

    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    closing_price = model.predict(X_test)
    closing_price = scaler.inverse_transform(closing_price)

    train = new_data[:t]
    valid = new_data[t:]
    valid["Predictions"] = closing_price
    # plt.clf()
    # plt.title('Model')
    # plt.xlabel('Date', fontsize=18)
    # plt.ylabel('Close Price USD ($)', fontsize=18)

    # plt.plot(valid[['Close', 'Predictions']])
    # plt.legend([ 'Close', 'Predictions'], loc='lower right')
    # plt.savefig('plot')

    return [valid, train]


def addstocks():
    df = open("data.csv", "r")

    with open("nasdaq.csv", newline="") as f:
        reader = csv.reader(f)
        data = list(reader)

    for i in range(1, len(data)):
        MyStock.objects.create(
            symbol=data[i][0],
            name=data[i][1],
            last_sale=data[i][2],
            net_change=data[i][3],
            p_change=data[i][4],
            market_cap=data[i][5],
            country=data[i][6],
            ipo_year=data[i][7],
            volume=data[i][8],
            sector=data[i][9],
            industry=data[i][10],
        )


@login_required
def home(request):
    stocks = MyStock.objects.all()
    return render(request, "home.html", {"stocks": stocks})


@login_required
def pickdate(request, pk, s_type):
    stock = MyStock.objects.get(id=pk)
    return render(request, "stock.html", {"stock": stock, "s_type": s_type})


@login_required
def graph(request, pk, s_type):
    data = request.GET
    stock = MyStock.objects.get(id=pk)
    startdate = data.get("start_date")
    enddate = data.get("end_date")
    getdata(stock.symbol, startdate, enddate)
    makemodel(s_type)
    valid, train = drawgraph(s_type)
    graphs = []

    graphs = []
    print(valid)
    # Adding linear plot of y1 vs. x.

    graphs.append(
        go.Line(
            x=valid.index, y=valid[s_type], mode="lines", name=s_type + " (Test data)"
        )
    )
    graphs.append(
        go.Line(
            x=train.index, y=train[s_type], mode="lines", name=s_type + " (Train data)"
        )
    )
    graphs.append(
        go.Line(
            x=valid.index,
            y=valid["Predictions"],
            mode="lines",
            name=s_type + " (Predictions)",
        )
    )

    realVals = valid[s_type]
    predictedVals = valid["Predictions"]
    mse = mean_squared_error(realVals, predictedVals)
    rmse = round(math.sqrt(mse), 2)

    # Setting layout of the figure.
    layout = {
        "title": {
            "text": "<a href='/' class='badge badge-success' target='_self'>Goto Home</a>"
            + "<br><b>"
            + stock.name
            + "</b><br>"
            + "Root-mean square error: <i>"
            + str(rmse)
            + "</i>",
            "y": 0.95,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        },
        "xaxis_title": "Date",
        "yaxis_title": "Stock Price in USD",
        "hovermode": "x",
    }

    # Getting HTML needed to render the plot.
    plot_div = plot({"data": graphs, "layout": layout}, output_type="div")

    return render(request, "graph.html", context={"plot_div": plot_div})


def register_request(request):
    if request.method == "POST":
        form = NewUserForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, "Registration successful.")
            return redirect("home")
        messages.error(request, "Unsuccessful registration. Invalid information.")
    form = NewUserForm()
    return render(
        request=request, template_name="register.html", context={"register_form": form}
    )


def login_request(request):
    if request.method == "POST":
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get("username")
            password = form.cleaned_data.get("password")
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                messages.info(request, f"You are now logged in as {username}.")
                return redirect("home")
            else:
                messages.error(request, "Invalid username or password.")
        else:
            messages.error(request, "Invalid username or password.")
    form = AuthenticationForm()
    return render(
        request=request, template_name="login.html", context={"login_form": form}
    )


def logout_request(request):
    logout(request)
    messages.info(request, "You have successfully logged out.")
    return redirect("login")

from flask import Flask, render_template, jsonify, request
import csv
import json
import pandas as pd
from utils import read_file
from testt import predict_all
import os

# (A2) FLASK SETTINGS + INIT
HOST_NAME = "localhost"
HOST_PORT = 8000
app = Flask(__name__)
predict_open, predict_high, predict_low, predict_close, predict_volume = 0, 0, 0, 0, 0
num_days = 0


@app.route("/")
def index():
    return render_template("main.html")


@app.route("/read_csv", methods=['GET', 'POST'])
def index2():
    stock_name, date_data, openn, high, low, close, volume = get_table_data(r"E:\GR1\dataset\excel_fpt.csv")
    return render_template("candlestick.html", csv={'name': stock_name, 'date': date_data, 'open': openn, 'high': high,
                                                    'low': low, 'close': close, 'volume': volume})


def get_table_data(file_path):
    df = pd.read_csv(file_path, dtype=str)
    stock_name = df.filter(['<Ticker>']).values.flatten()
    date = df.filter(['<DTYYYYMMDD>']).values.flatten()
    openn = df.filter(['<OpenFixed>']).values.flatten()
    high = df.filter(['<HighFixed>']).values.flatten()
    low = df.filter(['<LowFixed>']).values.flatten()
    close = df.filter(['<CloseFixed>']).values.flatten()
    volume = df.filter(['<Volume>']).values.flatten()
    return list(stock_name), list(date), list(openn), list(high), list(low), list(close), list(volume)


@app.route('/test', methods=['GET', "POST"])
def test():
    global predict_open, predict_high, predict_low, predict_close, predict_volume, num_days
    output = request.get_json()  # This is the output that was stored in the JSON within the browser
    result = json.loads(output)  # this converts the json output to a python dictionary
    num_days = int(result["number_of_days"])
    url_path = result["url"]
    filename = os.path.basename(url_path)
    name_without_extension = os.path.splitext(filename)[0]
    checkpoint_path = r"E:/GR1/checkpoint/" + name_without_extension
    print(checkpoint_path, num_days,url_path)
    predict_open, predict_high, predict_low, predict_close, predict_volume = predict_all(num_days,
                                                                                         url_path, checkpoint_path)
    pred_data = {"pred_open": predict_open, "pred_high": predict_high,
                 "pred_low": predict_low, "pred_close": predict_close,
                 "pred_volume": predict_volume}

    j = json.dumps(pred_data)
    print(j)
    return j


@app.route('/change_stock', methods=['POST'])
def change_stock():
    output2 = request.get_json()  # This is the output that was stored in the JSON within the browser
    result2 = json.loads(output2)  # this converts the json output to a python dictionary
    stock_path = result2["url"]
    print(stock_path)
    df2 = pd.read_csv(stock_path, dtype=str)
    stock_name = df2.filter(['<Ticker>']).values.flatten()
    date = df2.filter(['<DTYYYYMMDD>']).values.flatten()
    openn = df2.filter(['<OpenFixed>']).values.flatten()
    high = df2.filter(['<HighFixed>']).values.flatten()
    low = df2.filter(['<LowFixed>']).values.flatten()
    close = df2.filter(['<CloseFixed>']).values.flatten()
    volume = df2.filter(['<Volume>']).values.flatten()
    data = {'name': list(stock_name), 'date': list(date), 'open': list(openn), 'high': list(high),
            'low': list(low), 'close': list(close), 'volume': list(volume)}
    return data


if __name__ == "__main__":
    app.run(HOST_NAME, HOST_PORT, debug=True)

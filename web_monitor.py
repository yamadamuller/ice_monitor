from flask import Flask, render_template, jsonify
from flask_cors import CORS, cross_origin
from flask_caching import Cache
import json
import os
from multiprocessing import Process
from multiprocessing import synchronize
import multiprocessing
import threading
import sys
import signal
import time
import os
import datetime
import numpy as np
import pandas as pd
import csv
from dotenv import load_dotenv
load_dotenv() #load environmental variables

class IPCRunner:
    def __init__(self):
        self.producer_process = None #process that consumes from the CSV file and pushes to the buffer
        self.consumer_process = None #process that consumes from the buffer and sends to the plotter
        self.api_thread = None #process that forwards data by frequency in the api

def autoconstrast(x:np.ndarray, l=0, u=1):
    ratio = (x-np.min(x))/(np.max(x)-np.min(x))
    return l + ratio*(u-l)

def load_acquisition_configs():
    with open("./configs.json") as f:
        confs = json.load(f)
    return confs

#load the visualization configurations
configs = load_acquisition_configs() #load the configs .json
n_freqs = int(configs["visualization"]["n_freqs"]) #number of acquired frequencies
freqs = np.logspace(configs["visualization"]["freq_0"], configs["visualization"]["freq_n"], n_freqs) #list of acquired frequencies
idx_caps = np.arange(2, int(2*n_freqs+2), 2) #indexes of the capacitance within the csv file
idx_res = np.arange(3, int(2*n_freqs+2), 2) #indexes of the resistance within the csv file
csv_file = os.path.join(configs["csv_file"]["path"], configs["csv_file"]["filename"]) #file that stores readings
n_modes = configs["visualization"]["n_modes"] #number of acquisition pairs
n_mean = configs["visualization"]["n_samples"] #every n samples compute the mean
last_n_hours = configs["visualization"]["time_window"] #threshold to mask timestamps on plot

def writter_main(rel_acq_file:str):
    print(f'[WritterMain] Launched writting process')
    columns = ["timestamp", "mode", "1000 Z", "1000 TD", "10000 Z", "10000 TD", "100000 Z", "100000 TD", "1000000 Z", "1000000 TD"]

    #synthetic data
    synth_filename = "./testICE_24_11_25/c_test.csv"
    data = pd.read_csv(synth_filename).to_numpy()
    data_idx = 0
    datestamp = datetime.datetime.now()  # timestamp of the acquisition
    os.makedirs(os.path.dirname(rel_acq_file), exist_ok=True) if os.path.dirname(rel_acq_file) else None #create file if it doesn't exist

    with open(rel_acq_file, 'a', buffering=1) as file:
        writer = csv.writer(file)
        writer.writerow(columns)
        while True:
            row = (
                data[data_idx, 0], data[data_idx, 1], data[data_idx, 2], data[data_idx, 3],
                data[data_idx, 4], data[data_idx, 5], data[data_idx, 6], data[data_idx, 7],
                data[data_idx, 8], data[data_idx, 9]
            )

            writer.writerow(row)
            file.flush()
            os.fsync(file.fileno())
            data_idx += 1  # increase the times
            if data_idx == len(data):
                break

#constantly read from the buffer CSV and push to the sample buffer in memory
def producer_main(csv_path:str, buffer:multiprocessing.Queue, buffer_event:multiprocessing.synchronize.Event):
    print(f'[ProducerMain] Launched producer process')
    with open(csv_path, 'r') as file:
        file.readline() #skip header
        while True:
            if buffer_event.is_set():
                break

            where = file.tell() #monitor the file pointer
            line = file.readline() #read a line

            if not line:
                file.seek(where)
                time.sleep(1e-3)
                continue

            #process the line
            line = line.strip().replace(' ', '') #strip spaces in the string
            arr = np.array(line.split(',')) #split on commas (reading value)
            if not buffer.full():
                buffer.put(arr) #push to the buffer

        file.close() #close file to exit

#constantly consume samples from the buffer in memory
def consumer_main(plot_buffer:multiprocessing.Queue, buffer:multiprocessing.Queue, buffer_event:multiprocessing.synchronize.Event):
    print(f'[ConsumerMain] Launched consumer process')
    batch = {}  #dictionary to handle acquisition modes
    while True:
        if buffer_event.is_set():
            break

        if not buffer.empty():
            try:
                sample = buffer.get() #acquire a sample from the buffer
                timestamp = int(sample[0]) #unix timestamp
                electrode_mode = str(sample[1].item()) #electrode mode
                cap_readings = sample[idx_caps].astype(float) #capacitance reading
                res_readings = sample[idx_res].astype(float) #resistance reading

                #handle if some samples have already been acquired
                if electrode_mode in list(batch.keys()):
                    if len(np.atleast_2d(batch[electrode_mode]["cap"])) < n_mean:
                        batch[electrode_mode]["timestamp"].append(timestamp) #append the timestamp to the mode dictionary
                        batch[electrode_mode]["cap"] = np.vstack([batch[electrode_mode]["cap"], cap_readings]) #append the capacitance to the mode dictionary
                        batch[electrode_mode]["res"] = np.vstack([batch[electrode_mode]["res"], res_readings]) #append the resistance to the mode dictionary
                    else:
                        if n_mean > 1:
                            plot_buffer.put(
                                    {
                                        "mode": electrode_mode, #which mode
                                        "avg_timestamp": np.mean(batch[electrode_mode]["timestamp"], axis=0), #mean of the timestamps
                                        "avg_cap": np.mean(batch[electrode_mode]["cap"], axis=0), #mean of the capacitance
                                        "avg_res": np.mean(batch[electrode_mode]["res"], axis=0) #mean of the resistance
                                    }
                            ) #if n_mean samples have been appended -> push to the plot buffer

                        else:
                            plot_buffer.put(
                                {
                                    "mode": electrode_mode,  # which mode
                                    "avg_timestamp": np.mean(batch[electrode_mode]["timestamp"], axis=0),
                                    "avg_cap": batch[electrode_mode]["cap"],  # mean of the capacitance
                                    "avg_res": batch[electrode_mode]["res"]  # mean of the resistance
                                }
                            ) #if n_mean samples have been appended -> push to the plot buffer

                        del batch[electrode_mode] #reset the batch after mean
                else:
                    batch[electrode_mode] = {
                            "timestamp": [timestamp],
                            "cap": cap_readings,
                            "res": res_readings,
                    } #if the mode hasn't been registered yet
            except:
                time.sleep(1e-3)
                continue #if something fails during the registering
        else:
            time.sleep(1e-3)
            continue

#global variable for server launch
app = Flask(__name__) #flask app
app.secret_key = configs["flask_server"]["key"]
app.config['CACHE_TYPE'] = configs["flask_server"]["cache_type"]
cache = Cache(app)
CORS(app) #enable CORS for all routes
ipc_runner = IPCRunner()  # class to manage inter-process communication
last_proc_image = {} #dictionary to store the last processed frame per frequency

@app.route('/')
@cross_origin()
def index():
    return render_template('index.html',
                           filename=csv_file,
                           n_freqs=n_freqs,
                           freqs=freqs.tolist())

def frame_processing(plot_buffer:multiprocessing.Queue, buffer_event:multiprocessing.synchronize.Event):
    #plotting helpers
    capacitance_full = []
    resistance_full = []
    time_history = []
    mode_order = [
        "d:10-1", "d:9-10", "d:8-9", "d:7-8", "d:6-7",
        "d:5-6", "d:4-5", "d:3-4", "d:2-3", "d:1-2"]  # plot order in image
    mode_index = {p: i for i, p in enumerate(mode_order)}
    img_counter = 0  # counter to monitor mode batches
    cap = np.zeros((n_modes, n_freqs))  # batch-specific capacitance matrix
    res = np.zeros((n_modes, n_freqs))  # batch-specific resistance matrix

    while True:
        if buffer_event.is_set():
            break

        if not plot_buffer.empty():
            try:
                img_counter += 1  #increase the counter as a mode was processed
                raw_frame = plot_buffer.get()  #acquire a raw frame from the buffer
                idx_frame = mode_index[raw_frame["mode"]]  #mode-specific index
                cap_frame = raw_frame["avg_cap"]  #avg. capacitance of the last "n_mean" readings
                res_frame = raw_frame["avg_res"]  #avg. resistance of the last "n_mean" readings
                cap[idx_frame, :] = cap_frame  #store the mode-specific readings at the capacitance image matrix
                res[idx_frame, :] = res_frame  #store the mode-specific readings at the resistance image matrix

                #once all samples in a batch for all modes are processed
                if img_counter == n_modes:
                    capacitance_full.append(cap) #append the capacitance
                    resistance_full.append(res) #append the resistance
                    time_frame = raw_frame["avg_timestamp"] #avg. timestamp of the last "n_mean" readings
                    time_history.append(time_frame) #append only the average timestamp of the last mode (last to process)

                    #mask the arrays with the visualization time window
                    human_timestamp = [datetime.datetime.fromtimestamp(ts / 1e3) for ts in time_history] #list of the timestamps in human timestamp
                    human_timestamp = np.array(human_timestamp) #convert to array for better masking
                    time_mask = human_timestamp >= human_timestamp[-1] - datetime.timedelta(hours=last_n_hours) #timestamp mask (last "n" hours)
                    time_history = np.array(time_history)[time_mask] #mask the timestamps
                    time_history = list(time_history) #convert back to list
                    capacitance_full = np.array(capacitance_full)[time_mask, :, :] #mask the total capacitance matrix
                    capacitance_full = list(capacitance_full) #convert back to list
                    resistance_full = np.array(resistance_full)[time_mask, :, :] #mask to the total resistance matrix
                    resistance_full = list(resistance_full) #convert back to list
                    cap_matrix = np.stack(capacitance_full, axis=0) #stack the values
                    res_matrix = np.stack(resistance_full, axis=0) #stack the values

                    #update the API data based on sweeped frequecies
                    for sweep_freq in freqs:
                        freq_indx = freqs ==sweep_freq
                        last_proc_image[str(int(sweep_freq))] = {
                            "cap_mtx": cap_matrix[:,:,freq_indx].tolist(),
                            "res_mtx": res_matrix[:,:,freq_indx].tolist(),
                            "timestamp": human_timestamp.tolist()
                        } #update frequency specific image api

                    img_counter = 0 #reset the counter
                    cap = np.zeros((n_modes, n_freqs)) #reset the batch-specific capacitance matrix
                    res = np.zeros((n_modes, n_freqs)) #reset the batch-specific resistance matrix
            except:
                continue  #in case something goes wrong processing a frame/batch, continue to not freeze the image
        else:
            time.sleep(1e-3)  #empty buffer

@app.route('/monitor_start')
@cross_origin()
def start_data_acquisition():
    #launch the producer/consumer processes
    ipc_runner.producer_process = Process(target=producer_main, args=(csv_file, buffer, visu_event,))
    ipc_runner.producer_process.start()
    ipc_runner.consumer_process = Process(target=consumer_main, args=(plots, buffer, visu_event,))
    ipc_runner.consumer_process.start()
    ipc_runner.api_thread = threading.Thread(target=frame_processing, args=(plots, visu_event,))
    ipc_runner.api_thread.start()
    return "Monitoring CSV file for plotting"

@app.route('/monitor_stop')
@cross_origin()
def stop_data_acquisition():
    global visu_event, buffer, plots, last_proc_image
    visu_event.set() #set the visualization event to end

    #if exited the recording loop, terminate all processes
    if write_process and write_process.is_alive():
        print(f'[StartMonitoring] Terminating producer process with PID = {write_process.pid}', flush=True)
        write_process.terminate()
        write_process.join(timeout=1) #timeout in case the process freezes on shutdown
        if not write_process.is_alive():
            print(f'[StartMonitoring] Process {write_process.pid} terminated!', flush=True)
        else:
            write_process.kill()
            print(f'[StartMonitoring] Process {write_process.pid} killed!', flush=True)

    if ipc_runner.producer_process and ipc_runner.producer_process.is_alive():
        print(f'[StartMonitoring] Terminating producer process with PID = {ipc_runner.producer_process.pid}', flush=True)
        ipc_runner.producer_process.terminate()
        ipc_runner.producer_process.join(timeout=1) #timeout in case the process freezes on shutdown
        if not ipc_runner.producer_process.is_alive():
            print(f'[StartMonitoring] Process {ipc_runner.producer_process.pid} terminated!', flush=True)
        else:
            ipc_runner.producer_process.kill()
            print(f'[StartMonitoring] Process {ipc_runner.producer_process.pid} killed!', flush=True)

    if ipc_runner.consumer_process and ipc_runner.consumer_process.is_alive():
        print(f'[StartMonitoring] Terminating consumer process with PID = {ipc_runner.consumer_process.pid}', flush=True)
        ipc_runner.consumer_process.terminate()
        ipc_runner.consumer_process.join(timeout=1) #timeout in case the process freezes on shutdown
        if not ipc_runner.consumer_process.is_alive():
            print(f'[StartMonitoring] Process {ipc_runner.consumer_process.pid} terminated!', flush=True)
        else:
            ipc_runner.consumer_process.kill()
            print(f'[StartMonitoring] Process {ipc_runner.consumer_process.pid} killed!', flush=True)

    if ipc_runner.api_thread:
        ipc_runner.api_thread.join()
        print(f'[StartMonitoring] API thread joined!', flush=True)

    last_proc_image.clear() #clear the last processed dictionary
    buffer = multiprocessing.Queue() #reset buffer
    plots = multiprocessing.Queue() #reset plot buffer
    visu_event = multiprocessing.Manager().Event() #reset event

    return "Stopped monitoring CSV file for plotting"

@app.route("/monitor_status")
@cross_origin()
def monitor_status():
    if not visu_event.is_set():
        time.sleep(0.1)
        return jsonify({"status": True})
    else:
        time.sleep(0.1)
        return jsonify({"status": False})

@app.route("/api/<sweeped_freq>Hz")
@cross_origin()
def frequency_data_api(sweeped_freq):
    if last_proc_image[sweeped_freq]:
        return jsonify(last_proc_image[sweeped_freq])

if __name__ == '__main__':
    # buffers in memory
    buffer = multiprocessing.Queue()
    plots = multiprocessing.Queue()
    visu_event = multiprocessing.Manager().Event()  # event to signal both threads to stop
    write_process = Process(target=writter_main, args=(csv_file,))
    write_process.start()

    #CSV file verifications
    while True:
        if os.path.isfile(csv_file):
            break
    time.sleep(0.1) #busy wait until accessible

    print(f'[StartMonitoring] Monitoring CSV file @ {csv_file}')
   
    #run the web server on port "UI_SERVER_PORT"
    app.run(host=configs["flask_server"]["host"], port=configs["flask_server"]["port"], debug=True)

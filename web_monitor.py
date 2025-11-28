from flask import Flask, render_template, session, Response
from flask_cors import CORS, cross_origin
from flask_caching import Cache
import json
import os
from multiprocessing import Process
import multiprocessing
import sys
import signal
import time
import os
import datetime
import numpy as np
from dotenv import load_dotenv
load_dotenv() #load environmental variables

class IPCRunner:
    def __init__(self):
        self.producer_process = None #process that consumes from the CSV file and pushes to the buffer
        self.consumer_process = None #process that consumes from the buffer and sends to the plotter
        self.visu_event = multiprocessing.Event() #event to signal both threads to stop

def autoconstrast(x:np.ndarray, l=0, u=1):
    ratio = (x-np.min(x))/(np.max(x)-np.min(x))
    return l + ratio*(u-l)

def signal_handler(sig, frame):
    global stop_recording
    print("[SignalHandler] Interrupt received. Stopping recording...", flush=True)
    stop_recording = True
    sys.exit()

def load_acquisition_configs():
    with open("./configs.json") as f:
        confs = json.load(f)
    return confs

#buffers in memory
stop_recording = False
buffer = multiprocessing.Queue()
plots = multiprocessing.Queue()

#load the visualization configurations
configs = load_acquisition_configs() #load the configs .json
freqs = configs["visualization"]["freqs"] #list of acquired frequencies
n_freqs = len(freqs) #number of acquired frequencies
idx_caps = np.arange(2, int(2*n_freqs+2), 2) #indexes of the capacitance within the csv file
idx_res = np.arange(3, int(2*n_freqs+2), 2) #indexes of the resistance within the csv file
csv_file = os.path.join(configs["csv_file"]["path"], configs["csv_file"]["filename"]) #file that stores readings
n_modes = configs["visualization"]["n_modes"] #number of acquisition pairs
n_mean = configs["visualization"]["n_samples"] #every n samples compute the mean
last_n_hours = configs["visualization"]["time_window"] #threshold to mask timestamps on plot

#constantly read from the buffer CSV and push to the sample buffer in memory
def producer_main(csv_path:str, buffer:multiprocessing.Queue):
    print(f'[ProducerMain] Launched producer process')
    with open(csv_path, 'r') as file:
        file.readline() #skip header
        while True:
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

#constantly consume samples from the buffer in memory
def consumer_main(plot_buffer:multiprocessing.Queue, buffer:multiprocessing.Queue):
    print(f'[ConsumerMain] Launched consumer process')
    batch = {}  #dictionary to handle acquisition modes
    while True:
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
                            print('sample')
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
                continue #if something fails during the registering
        else:
            time.sleep(1e-3)

#global variable for server launch
app = Flask(__name__) #flask app
app.secret_key = os.getenv("UI_KEY")
app.config['CACHE_TYPE'] = 'simple'
cache = Cache(app)
CORS(app) #enable CORS for all routes
signal.signal(signal.SIGINT, signal_handler) #ctrl-c
ipc_runner = IPCRunner() #class to manage inter-process communication

@app.route('/')
@cross_origin()
def index():
    return render_template('index.html')

@app.route('/monitor_start')
@cross_origin()
def start_data_acquisition():
    #plotting helpers
    capacitance_full = []
    resistance_full = []
    time_history = []
    mode_order = [
        "d:10-1", "d:9-10", "d:8-9", "d:7-8", "d:6-7",
        "d:5-6", "d:4-5", "d:3-4", "d:2-3", "d:1-2"] #plot order in image
    mode_index = {p: i for i, p in enumerate(mode_order)}
    img_counter = 0 #counter to monitor mode batches
    cap = np.zeros((n_modes, n_freqs)) #batch-specific capacitance matrix
    res = np.zeros((n_modes, n_freqs)) #batch-specific resistance matrix

    while not stop_recording:
        if ipc_runner.visu_event.is_set():
            break

        if not plots.empty():
            try:
                img_counter += 1 #increase the counter as a mode was processed
                raw_frame = plots.get() #acquire a raw frame from the buffer
                idx_frame = mode_index[raw_frame["mode"]] #mode-specific index
                cap_frame = raw_frame["avg_cap"] #avg. capacitance of the last "n_mean" readings
                res_frame = raw_frame["avg_res"] #avg. resistance of the last "n_mean" readings
                cap[idx_frame,:] = cap_frame #store the mode-specific readings at the capacitance image matrix
                res[idx_frame,:] = res_frame #store the mode-specific readings at the resistance image matrix

                #once all samples in a batch for all modes are processed
                print('chama')
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
                    capacitance_full = np.array(capacitance_full)[time_mask, :, :] #mask the total capacitance matrix
                    resistance_full = np.array(resistance_full)[time_mask, :, :] #mask to the total resistance matrix
                    cap_matrix = np.stack(capacitance_full, axis=0) #stack the values
                    res_matrix = np.stack(resistance_full, axis=0) #stack the values

                    #update the plot
                    img_counter = 0 #reset the counter
                    cap = np.zeros((len(mode_index), len(freqs))) #reset the batch-specific capacitance matrix
                    res = np.zeros((len(mode_index), len(freqs))) #reset the batch-specific resistance matrix
            except:
                continue #in case something goes wrong processing a frame/batch, continue to not freeze the image
        else:
            continue #empty buffer

    return "Monitoring CSV file for plotting"

@app.route('/monitor_stop')
@cross_origin()
def stop_data_acquisition():
    #if exited the recording loop, terminate all processes
    if ipc_runner.producer_proc and ipc_runner.producer_proc.is_alive():
        print(f'[StartMonitoring] Terminating producer process with PID = {ipc_runner.producer_proc.pid}')
        ipc_runner.producer_proc.terminate()
        ipc_runner.producer_proc.join()
        if not ipc_runner.producer_proc.is_alive():
            print(f'[StartMonitoring] Process {ipc_runner.producer_proc.pid} terminated!')

    if ipc_runner.consumer_proc and ipc_runner.consumer_proc.is_alive():
        print(f'[StartMonitoring] Terminating consumer process with PID = {ipc_runner.consumer_proc.pid}')
        ipc_runner.consumer_proc.terminate()
        ipc_runner.consumer_proc.join()
        if not ipc_runner.consumer_proc.is_alive():
            print(f'[StartMonitoring] Process {ipc_runner.consumer_proc.pid} terminated!')

    ipc_runner.visu_event.clear() #reset the event

    return "Stopped monitoring CSV file for plotting"

if __name__ == '__main__':
    #CSV file verifications
    while True:
        if os.path.isfile(csv_file):
            break
    time.sleep(0.1) #busy wait until accessible

    print(f'[StartMonitoring] Monitoring CSV file @ {csv_file}')

    # launch the producer/consumer processes
    ipc_runner.producer_proc = Process(target=producer_main, args=(csv_file, buffer,))
    ipc_runner.producer_proc.start()
    ipc_runner.consumer_proc = Process(target=consumer_main, args=(plots, buffer,))
    ipc_runner.consumer_proc.start()

    #run the web server on port "UI_SERVER_PORT"
    app.run(host=os.getenv("UI_SERVER_HOST"), port=os.getenv("UI_SERVER_PORT"), debug=True)
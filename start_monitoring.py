from multiprocessing import Process
import multiprocessing
import sys
import signal
import time
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import datetime
import pandas as pd
import csv
import numpy as np
from matplotlib.ticker import MaxNLocator

#--- ACQUISITION CONFIGS ------------------------------------------------------
#path to where the real acquisitions are stored                               |
output_path = 'C:/Users/Everton/Desktop/PHOBOS_sw_test_Data/testtest'#        |
filename = f'test_realtime.csv' #file that stores the readings                |
final_path = os.path.join(output_path, filename) #file that stores readings   |
n_modes = 10  #number of acquisition pairs                                    |
n_mean = 1  #every 3 samples compute the mean                                 |
cap_freqs = ["1kHz", "10kHz", "100kHz", "1MHz"] #frequencies                  |
res_freqs = ["1kHz", "10kHz", "100kHz", "1MHz"] #frequencies                  |
#------------------------------------------------------------------------------

# buffers in memory
stop_recording = False
buffer = multiprocessing.Queue()
plots = multiprocessing.Queue()

#array indexes
n_freqs = len(cap_freqs)
idx_caps = np.arange(2, int(2*n_freqs+2), 2) #indexes of the capacitance
idx_res = np.arange(3, int(2*n_freqs+2), 2) #indexes of the resistance

# plotting helpers
capacitance_full = []
resistance_full = []
time_history = []
mode_order = [
    "d:10-1", "d:9-10", "d:8-9", "d:7-8", "d:6-7",
    "d:5-6", "d:4-5", "d:3-4", "d:2-3", "d:1-2"
]
mode_index = {p: i for i, p in enumerate(mode_order)}

def autoconstrast(x:np.ndarray, l=0, u=1):
    ratio = (x-np.min(x))/(np.max(x)-np.min(x))
    return l + ratio*(u-l)

def signal_handler(sig, frame):
    global stop_recording
    print("[SignalHandler] Interrupt received. Stopping recording...", flush=True)
    stop_recording = True
    plt.close('all') #close all images
    sys.exit()

signal.signal(signal.SIGINT, signal_handler)

#constantly read from the buffer CSV and push to the sample buffer in memory
def producer_main(buffer:multiprocessing.Queue):
    global final_path
    print(f'[ProducerMain] Launched producer process')
    with open(final_path, 'r') as file:
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
            arr = np.array(line.split(',')) #split on commas
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
                        plot_buffer.put(
                                {
                                    "mode": electrode_mode, #which mode
                                    "avg_timestamp": np.mean(batch[electrode_mode]["timestamp"], axis=0), #mean of the timestamps
                                    "avg_cap": np.mean(batch[electrode_mode]["cap"], axis=0), #mean of the capacitance
                                    "avg_res": np.mean(batch[electrode_mode]["res"], axis=0) #mean of the resistance
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

def init_plot(mode_order:list[str]):
    global cap_freqs, res_freqs

    #monitored frequencies
    plt.ion()
    fig, axes = plt.subplots(2, len(cap_freqs), figsize=(10,5))
    print(axes)
    cap_ims = []
    res_ims = []

    for f in range(len(cap_freqs)):
        if len(cap_freqs) == 1:
            ax = axes[0]
        else:
            ax = axes[0, f]

        im = ax.imshow(np.zeros((len(mode_order), 1)),
                       aspect='auto', cmap='jet')
        ax.set_title(f"Cap. norm. @ {cap_freqs[f]}")
        ax.set_xlabel("Time")

        #set y label only for the first image
        if f == 0:
            ax.set_ylabel("Mode")
            ax.set_yticks(range(len(mode_order)))
            ax.set_yticklabels(mode_order)
        else:
            ax.set_yticks([]) #hide the y label in other plots

        fig.colorbar(im, ax=ax)
        cap_ims.append(im)

    for f in range(len(cap_freqs)):
        if len(cap_freqs) == 1:
            ax = axes[1]
        else:
            ax = axes[1, f]

        im = ax.imshow(np.zeros((len(mode_order), 1)),
                       aspect='auto', cmap='jet')
        ax.set_title(f"Res. norm. @ {res_freqs[f]}")
        ax.set_xlabel("Time")

        #set y labels only for the first image
        if f == 0:
            ax.set_ylabel("Mode")
            ax.set_yticks(range(len(mode_order)))
            ax.set_yticklabels(mode_order)
        else:
            ax.set_yticks([]) #hide the y label in other plots

        fig.colorbar(im, ax=ax)
        res_ims.append(im)

    plt.tight_layout()
    plt.show()
    return fig, axes, cap_ims, res_ims

def update_plot(cap_matrix:np.ndarray, res_matrix:np.ndarray, cap_ims:list, res_ims:list, timehistory:list):
    x = [datetime.datetime.fromtimestamp(ts/1e3) for ts in timehistory] #list of the timestamps in human timestamp

    #generate the images for capacitance
    for f, im in enumerate(cap_ims):
        data = cap_matrix[:, :, f].T #transpose the array to have modes on the Y-axis
        data = autoconstrast(data) #normalize the image
        im.set_data(data) #update the capacitance data
        im.set_extent([x[0], x[-1], 0, data.shape[0]]) #map x-axis to timestamps
        im.axes.set_xlim(x[0], x[-1])
        im.set_clim(0, 1)
        ax = im.axes
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        #fig = cap_ims[f].axes.figure
        #fig.canvas.draw()
        #fig.canvas.flush_events()

    for f, im in enumerate(res_ims):
        data = res_matrix[:, :, f].T #transpose the array to have modes on the Y-axis
        data = autoconstrast(data) #normalize the image
        im.set_data(data+1e-8) #update the resistance data
        im.set_extent([x[0], x[-1], 0, data.shape[0]])
        im.axes.set_xlim(x[0], x[-1])
        im.set_clim(0, 1)
        ax = im.axes
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        #fig = res_ims[f].axes.figure
        #fig.canvas.draw()
        #fig.canvas.flush_events()

    fig = cap_ims[0].axes.figure
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.01) #plotly refresh

#launch the three processes
if __name__ == '__main__':
    #find the most recent "c_test.csv" file
    while True:
        if os.path.isfile(final_path):
            break
        time.sleep(1e-3) #busy wait

    print(f'[StartMonitoring] Monitoring CSV file @ {final_path}')

    #busy wait while target filepath doesn't exist
    while not os.path.isfile(final_path):
        time.sleep(1e-3)

    producer_proc = Process(target=producer_main, args=(buffer,))
    producer_proc.start()
    consumer_proc = Process(target=consumer_main, args=(plots, buffer,))
    consumer_proc.start()

    fig, axes, cap_ims, res_ims = init_plot(mode_order) #create the image object that will be overwritten every new sample
    img_counter = 0 #counter to monitor batches
    cap = np.zeros((len(mode_index), 4)) #batch-specific capacitance array
    res = np.zeros((len(mode_index), 4)) #batch-specific resistance array
    t_last_frame = time.time() #variable to monitor the time it takes to generate a new frame
    while not stop_recording:
        if not plots.empty():
            try:
                img_counter += 1
                frame = plots.get()
                idx_frame = mode_index[frame["mode"]] #mode-specific index
                cap_frame = frame["avg_cap"] #avg. capacitance of the last 3 readings
                res_frame = frame["avg_res"] #avg. resistance of the last 3 readings
                cap[idx_frame,:] = cap_frame #register the mode-specific readings at the cap array
                res[idx_frame,:] = res_frame #register the mode-specific readings at the res array

                #once all samples in a batch for all modes are processed
                if img_counter == len(mode_index):
                    capacitance_full.append(cap) #append the capacitance
                    resistance_full.append(res) #append the resistances
                    time_frame = frame["avg_timestamp"] #avg. timestamp of the last 3 readings
                    time_history.append(time_frame) #append only the timestamp of the last mode
                    cap_matrix = np.stack(capacitance_full, axis=0) #stack the values
                    res_matrix = np.stack(resistance_full, axis=0) #stack the values
                    update_plot(cap_matrix, res_matrix, cap_ims, res_ims, time_history) #update images with the newly stacked matrices
                    img_counter = 0 #reset the counter
                    cap = np.zeros((len(mode_index), 4)) #reset the batch-specific capacitance array
                    res = np.zeros((len(mode_index), 4)) #reset the batch-specific resistance array
                    t_last_frame = time.time()
            except:
                continue #in case something goes wrong in the processing
        else:
            continue #empty buffer

    #if exited the recording loop, terminate all processes
    if producer_proc and producer_proc.is_alive():
        print(f'[StartMonitoring] Terminating producer process with PID = {producer_proc.pid}')
        producer_proc.terminate()
        producer_proc.join()
        if not producer_proc.is_alive():
            print(f'[StartMonitoring] Process {producer_proc.pid} terminated!')

    if consumer_proc and consumer_proc.is_alive():
        print(f'[StartMonitoring] Terminating consumer process with PID = {consumer_proc.pid}')
        consumer_proc.terminate()
        consumer_proc.join()
        if not consumer_proc.is_alive():
            print(f'[StartMonitoring] Process {consumer_proc.pid} terminated!')

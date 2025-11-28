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
from scipy.ndimage import median_filter, gaussian_filter

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

#--- ACQUISITION CONFIGS ------------------------------------------------------
output_path = './acquisition' #path to where the real acquisitions are stored |
directory_split = '/' #might be '\' in windows                                |
filename = 'c_test.csv' #file that stores the readings                        |
n_modes = 10  #number of acquisition pairs                                    |
n_mean = 3  #every 3 samples compute the mean                                 |
cap_freqs = ["1kHz", "10kHz", "100kHz", "1MHz"] #frequencies                  |
res_freqs = ["1kHz", "10kHz", "100kHz", "1MHz"] #frequencies                  |
last_n_hours = 36
filter_kernel = 5
#------------------------------------------------------------------------------

#buffers in memory
stop_recording = False
buffer = multiprocessing.Queue()
plots = multiprocessing.Queue()

#array indexes
n_freqs = len(cap_freqs)
idx_caps = np.arange(2, int(2*n_freqs+2), 2) #indexes of the capacitance
idx_res = np.arange(3, int(2*n_freqs+2), 2) #indexes of the resistance

#plotting helpers
capacitance_full = []
resistance_full = []
time_history = []
mode_order = [
    "d:10-1", "d:9-10", "d:8-9", "d:7-8", "d:6-7",
    "d:5-6", "d:4-5", "d:3-4", "d:2-3", "d:1-2"
]
mode_index = {p: i for i, p in enumerate(mode_order)}

def writter_main():
    print(f'[WritterMain] Launched writting process')
    columns = ["timestamp", "mode", "1000 Z", "1000 TD", "10000 Z", "10000 TD", "100000 Z", "100000 TD", "1000000 Z", "1000000 TD"]

    #synthetic data
    synth_filename = "./testICE_25_11_25/c_test.csv"
    data = pd.read_csv(synth_filename).to_numpy()
    data_idx = 0
    datestamp = datetime.datetime.now()  # timestamp of the acquisition
    acq_subdir = f'test_{datestamp.day}_{datestamp.month}_{datestamp.year}' #subdirectory based on timestamp
    acq_path = os.path.join(output_path, acq_subdir) #relative path to the subdirectory
    rel_acq_file = os.path.join(acq_path, filename) #relative path to the readings file
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
def producer_main(final_path:str, buffer:multiprocessing.Queue):
    #global final_path
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
                        if n_mean > 1:
                            plot_buffer.put(
                                {
                                    "mode": electrode_mode,  # which mode
                                    "avg_timestamp": np.mean(batch[electrode_mode]["timestamp"], axis=0),
                                    "avg_cap": np.mean(batch[electrode_mode]["cap"], axis=0),  # mean of the capacitance
                                    "avg_res": np.mean(batch[electrode_mode]["res"], axis=0)  # mean of the resistance
                                }

                            )  # if n_mean samples have been appended -> push to the plot buffer
                        else:
                            plot_buffer.put(
                                {
                                    "mode": electrode_mode,  # which mode
                                    "avg_timestamp": np.mean(batch[electrode_mode]["timestamp"]),
                                    "avg_cap": batch[electrode_mode]["cap"],  # mean of the capacitance
                                    "avg_res": batch[electrode_mode]["res"]  # mean of the resistance
                                }
                            )  # if n_mean samples have been appended -> push to the plot buffer

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
    fig, axes = plt.subplots(2, 4, figsize=(9,7))

    cap_ims = []
    res_ims = []

    for f in range(len(cap_freqs)):
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

    for f in range(len(res_freqs)):
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

def update_plot(cap_matrix:np.ndarray, res_matrix:np.ndarray, cap_ims:list, res_ims:list, time_array:np.ndarray, fig:plt.Figure):
    #time_array = [datetime.datetime.fromtimestamp(ts/1e3) for ts in time_array] #list of the timestamps in human timestamp

    #generate the images for capacitance
    for f, im in enumerate(cap_ims):
        data = cap_matrix[:, :, f].T #transpose the array to have modes on the Y-axis
        data = autoconstrast(data) #normalize the image
        im.set_data(data) #update the capacitance data
        im.set_extent([time_array[0], time_array[-1], 0, data.shape[0]]) #map x-axis to timestamps
        im.axes.set_xlim(time_array[0], time_array[-1])
        im.set_clim(0, 1)
        ax = im.axes
        ax.xaxis.set_major_locator(MaxNLocator(nbins=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    for f, im in enumerate(res_ims):
        data = res_matrix[:, :, f].T #transpose the array to have modes on the Y-axis
        data = autoconstrast(data) #normalize the image
        im.set_data(data) #update the resistance data
        im.set_extent([time_array[0], time_array[-1], 0, data.shape[0]])
        im.axes.set_xlim(time_array[0], time_array[-1])
        im.set_clim(0, 1)
        ax = im.axes
        ax.xaxis.set_major_locator(MaxNLocator(nbins=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    fig.canvas.draw_idle()
    fig.canvas.flush_events()

#launch the three processes
if __name__ == '__main__':
    write_proc = Process(target=writter_main)
    write_proc.start()
    time.sleep(0.5) #wait 0.5 seconds to ensure the file exists

    #find the most recent "c_test.csv" file
    curr_date = datetime.datetime.now()  # current timestamp

    while True:
        all_output_subdirs = [os.path.join(output_path, d) for d in os.listdir(output_path) if os.path.isdir(os.path.join(output_path, d))] #list existing sub-directories
        if all_output_subdirs:
            specific_path = max(all_output_subdirs, key=os.path.getmtime) #find the most recent directory
            specific_path_split = specific_path.split(f'{directory_split}')[-1] #return only the subdirectory name

            #check if the found subdirectory matches the date
            if specific_path_split == f'test_{curr_date.day}_{curr_date.month}_{curr_date.year}':
                break

        time.sleep(1e-3) #busy wait

    final_path = os.path.join(specific_path, filename) #relative path to the readings file
    print(f'[StartMonitoring] Monitoring CSV file @ {final_path}')

    #busy wait while target filepath doesn't exist
    while not os.path.isfile(final_path):
        time.sleep(1e-3)

    producer_proc = Process(target=producer_main, args=(final_path, buffer,))
    producer_proc.start()
    consumer_proc = Process(target=consumer_main, args=(plots, buffer,))
    consumer_proc.start()

    fig, axes, cap_ims, res_ims = init_plot(mode_order) #create the image object that will be overwritten every new sample
    img_counter = 0 #counter to monitor batches
    cap = np.zeros((len(mode_index), len(cap_freqs))) #batch-specific capacitance array
    res = np.zeros((len(mode_index), len(res_freqs))) #batch-specific resistance array
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
                if img_counter == n_modes:
                    capacitance_full.append(cap) #append the capacitance
                    resistance_full.append(res) #append the resistances
                    time_frame = frame["avg_timestamp"] #avg. timestamp of the last 3 readings
                    time_history.append(time_frame) #append only the timestamp of the last mode

                    #mask the arrays to window the signals on time
                    human_timestamp = [datetime.datetime.fromtimestamp(ts / 1e3) for ts in time_history] #list of the timestamps in human timestamp
                    human_timestamp = np.array(human_timestamp) #convert to array for better masking
                    time_mask = human_timestamp >= human_timestamp[-1] - datetime.timedelta(hours=last_n_hours) #timestamp mask (last "n" hours)

                    #TODO: optimize the masking
                    time_history = np.array(time_history)[time_mask] #mask the timestamps
                    time_history = list(time_history)
                    capacitance_full = np.array(capacitance_full)[time_mask, :, :] #mask the total capacitance matrix
                    capacitance_full = list(capacitance_full) #convert to list
                    resistance_full = np.array(resistance_full)[time_mask, :, :] #mask to the total resistance matrix
                    resistance_full = list(resistance_full) #convert to list
                    cap_matrix = np.stack(capacitance_full, axis=0) #stack the values
                    res_matrix = np.stack(resistance_full, axis=0) #stack the values
                    #print(f'[{time.time() - t_last_frame} s] Updated frame = {np.shape(cap_matrix)}')
                    update_plot(cap_matrix, res_matrix, cap_ims, res_ims, human_timestamp, fig) #update images with the newly stacked matrices
                    img_counter = 0 #reset the counter
                    cap = np.zeros((len(mode_index), len(cap_freqs))) #reset the batch-specific capacitance array
                    res = np.zeros((len(mode_index), len(res_freqs))) #reset the batch-specific resistance array
                    t_last_frame = time.time()
            except:
                continue #in case something goes wrong in the processing
        else:
            continue #empty buffer

    #if exited the recording loop, terminate all processes
    if write_proc and write_proc.is_alive():
        print(f'[StartMonitoring] Terminating CSV writing process with PID = {write_proc.pid}')
        write_proc.terminate()
        write_proc.join()
        if not write_proc.is_alive():
            print(f'[StartMonitoring] Process {write_proc.pid} terminated!')

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

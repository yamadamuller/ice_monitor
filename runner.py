from multiprocessing import Process
import multiprocessing
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

stop_recording = False
buffer = multiprocessing.Queue()
plots = multiprocessing.Queue()
cp_history = []
rp_history = []
time_history = []
pair_order = [
    "d:1-2", "d:2-3", "d:3-4", "d:4-5", "d:5-6",
    "d:6-7", "d:7-8", "d:8-9", "d:9-10", "d:10-1"
]
pair_index = {p: i for i, p in enumerate(pair_order)}


def rescale(x:np.ndarray, l=0, u=1):
    ratio = (x-np.min(x))/(np.max(x)-np.min(x))
    return l + ratio*(u-l)

def signal_handler(sig, frame):
    global stop_recording
    print("Interrupt received. Stopping recording...", flush=True)
    stop_recording = True
    sys.exit()

signal.signal(signal.SIGINT, signal_handler)

#output path configuration
output_path = './acquisition'
datestamp = datetime.datetime.now()
specific_path = f'test_{datestamp.day}_{datestamp.month}_{datestamp.year}'
acquisition_path = os.path.join(output_path, specific_path)
filename = 'c_test.csv'
final_path = os.path.join(acquisition_path, filename)
os.makedirs(os.path.dirname(final_path), exist_ok=True) if os.path.dirname(final_path) else None

#array indexes
idx_caps = np.arange(2, 10, 2)  # indexes of the capacitance
idx_res = np.arange(3, 10, 2)  # indexes of the resistance

#file writer
def writter_main():
    global final_path
    print(f'[WritterMain] Launched writting process')
    columns = ["timestamp", "mode", "1000 Z", "1000 TD", "10000 Z", "10000 TD", "100000 Z", "100000 TD", "1000000 Z",
               "1000000 TD"]

    #synthetic data
    filename = "./testICE_24_11_25/c_test.csv"
    data = pd.read_csv(filename).to_numpy()
    data_idx = 0

    with open(final_path, 'a', buffering=1) as file:
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
            #time.sleep(0.1)
            data_idx += 1  # increase the times

            if data_idx == len(data):
                break

def producer_main():
    global final_path
    print(f'[ProducerMain] Launched producer process')
    lines = 0
    with open(final_path, 'r') as file:
        file.readline() #skip header line
        while True:
            where = file.tell()
            line = file.readline()

            if not line:
                file.seek(where)
                time.sleep(1e-3)
                continue

            # process line
            line = line.strip().replace(' ', '')
            arr = np.array(line.split(','))
            if not buffer.full():
                buffer.put(arr)
                lines += 1

def consumer_main(plot_buffer:multiprocessing.Queue):
    global idx_caps, idx_res
    print(f'[ConsumerMain] Launched consumer process')
    batch = {}  #dictionary to handle data grouping
    while True:
        if not buffer.empty():
            try:
                sample = buffer.get()
                timestamp = int(sample[0])  # Unix timestamp
                electrode_pair = str(sample[1].item())  # which pair
                cap_readings = sample[idx_caps].astype(float)  # capacitance readings
                res_readings = sample[idx_res].astype(float)  # resistance readings

                # handle if some samples have already been acquired
                if electrode_pair in list(batch.keys()):
                    if len(np.atleast_2d(batch[electrode_pair]["cp"])) < 3:
                        batch[electrode_pair]["timestamp"].append(timestamp)
                        batch[electrode_pair]["cp"] = np.vstack([batch[electrode_pair]["cp"], cap_readings])
                        batch[electrode_pair]["rp"] = np.vstack([batch[electrode_pair]["rp"], res_readings])
                    else:
                        plot_buffer.put(
                                {
                                    "pair": electrode_pair,
                                    "avg_timestamp": np.mean(batch[electrode_pair]["timestamp"], axis=0),
                                    "avg_cp": np.mean(batch[electrode_pair]["cp"], axis=0),
                                    "avg_rp": np.mean(batch[electrode_pair]["rp"], axis=0)
                                }
                        )

                        del batch[electrode_pair] #reset the batch
                else:
                    batch[electrode_pair] = {
                            "timestamp": [timestamp],
                            "cp": cap_readings,
                            "rp": res_readings,
                    }
            except:
                continue
        else:
            time.sleep(1e-3)

def init_plot(pair_order):
    cp_freqs = ["1k", "10k", "100k", "1M"]
    rp_freqs = ["1k", "10k", "100k", "1M"]

    plt.ion()
    fig, axes = plt.subplots(2, 4, figsize=(20,10))

    cp_ims = []
    rp_ims = []

    for f in range(4):
        ax = axes[0, f]
        im = ax.imshow(np.zeros((len(pair_order), 1)),
                       aspect='auto', cmap='jet')
        ax.set_title(f"CP @ {cp_freqs[f]}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Electrode pair")
        ax.set_yticks(range(len(pair_order)))
        ax.set_yticklabels(pair_order)
        fig.colorbar(im, ax=ax)
        cp_ims.append(im)

    for f in range(4):
        ax = axes[1, f]
        im = ax.imshow(np.zeros((len(pair_order), 1)),
                       aspect='auto', cmap='jet')
        ax.set_title(f"RP @ {rp_freqs[f]}")
        ax.set_xlabel("Time (frames)")
        ax.set_ylabel("Electrode pair")
        ax.set_yticks(range(len(pair_order)))
        ax.set_yticklabels(pair_order)
        fig.colorbar(im, ax=ax)
        rp_ims.append(im)

    plt.tight_layout()
    plt.show()
    return fig, axes, cp_ims, rp_ims

def update_plot(cp_matrix, rp_matrix, cp_ims, rp_ims, timehistory):
    x = [datetime.datetime.fromtimestamp(ts/1e3) for ts in timehistory]
    for f, im in enumerate(cp_ims):
        data = cp_matrix[:, :, f].T
        data = rescale(data)
        im.set_data(data)
        im.set_extent([x[0], x[-1], 0, data.shape[0]])  # map x-axis to timestamps
        im.axes.set_xlim(x[0], x[-1])
        im.set_clim(0, 1)
        ax = im.axes
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%m %H:%M:%S"))

    for f, im in enumerate(rp_ims):
        data = rp_matrix[:, :, f].T
        data = rescale(data)
        im.set_data(data+1e-8)
        im.set_extent([x[0], x[-1], 0, data.shape[0]])
        im.axes.set_xlim(x[0], x[-1])
        im.set_clim(0, 1)
        ax = im.axes
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%m %H:%M:%S"))

    plt.pause(0.01)   # important for GUI refresh

write_proc = Process(target=writter_main)
write_proc.start()
time.sleep(0.5)
producer_proc = Process(target=producer_main)
producer_proc.start()
consumer_proc = Process(target=consumer_main, args=(plots,))
consumer_proc.start()

def frame_from_dict(msg, pair_index):
    idx = pair_index[msg["pair"]]
    return msg["avg_cp"], msg["avg_rp"], msg["avg_timestamp"], idx

fig, axes, cp_ims, rp_ims = init_plot(pair_order)
img_counter = 0
cp = np.zeros((len(pair_index), 4))
rp = np.zeros((len(pair_index), 4))
while not stop_recording:
    if not plots.empty():
        img_counter += 1
        frame = plots.get()
        cp_frame, rp_frame, time_frame, idx_frame = frame_from_dict(frame, pair_index)
        cp[idx_frame,:] = cp_frame
        rp[idx_frame,:] = rp_frame

        if img_counter == len(pair_index):
            cp_history.append(cp)
            rp_history.append(rp)
            time_history.append(time_frame)
            cp_matrix = np.stack(cp_history, axis=0)
            rp_matrix = np.stack(rp_history, axis=0)
            update_plot(cp_matrix, rp_matrix, cp_ims, rp_ims, time_history)
            img_counter = 0  # reset the counter
            cp = np.zeros((len(pair_index), 4))
            rp = np.zeros((len(pair_index), 4))

write_proc.terminate()
producer_proc.terminate()
consumer_proc.terminate()
write_proc.join()
producer_proc.join()
consumer_proc.join()
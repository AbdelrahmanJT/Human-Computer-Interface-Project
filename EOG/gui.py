import random
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import os
import re
import joblib
import numpy as np
import pickle
import numpy as np
import pandas as pd
from scipy.signal import periodogram
from scipy.signal import butter, filtfilt, resample, find_peaks
import scipy.integrate as integrate
from statsmodels.tsa.ar_model import AutoReg
from scipy.stats import kurtosis
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from collections import defaultdict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import pywt

def butter_bandpass_filter(Input_Signal, Low_CutOff, High_CutOff, SamplingRate, order=2):
    nyquist_freq = 0.5 * SamplingRate
    low = Low_CutOff / nyquist_freq
    high = High_CutOff / nyquist_freq
    Numerator, Denominator = butter(order, [low, high], btype='band',output='ba',analog=False,fs=None)
    Filtered = filtfilt(Numerator, Denominator, Input_Signal)
    return Filtered

def get_max_peak(signal):
    peaks, _ = find_peaks(signal)
    max_peak = max(signal[peaks])
    return max_peak

def get_mean_and_std(signal):
    mean = np.mean(signal)
    std = np.std(signal)
    return mean, std

def wavlet_extract(signal):
    wavelet_name = 'db4'
    n_samples = len(signal)
    max_level = pywt.dwt_max_level(n_samples, pywt.Wavelet(wavelet_name).dec_len)
    wavelet_level = min(5, max_level)  # use the smaller of 5 and max_level
    n_samples = len(signal)
    wavelet_scale = pywt.dwt_coeff_len(n_samples, pywt.Wavelet(wavelet_name).dec_len, mode='constant')
    padded_signal_length = int(np.ceil(n_samples / wavelet_scale) * wavelet_scale)
    padded_signal = np.pad(signal, (0, padded_signal_length - n_samples), mode='constant')
    wavelet_coeffs = pywt.wavedec(padded_signal, wavelet_name, level=wavelet_level)
    feature_vector = []
    for coeffs in wavelet_coeffs:
        mean = np.mean(coeffs)
        std = np.std(coeffs)
        var = np.var(coeffs)
        energy = np.sum(np.square(coeffs))
        coeffs_sq = np.square(coeffs)
        coeffs_sq[coeffs_sq == 0] = 1e-10  # set 0 values to a small non-zero value
        entropy = -np.sum(coeffs_sq * np.log2(coeffs_sq))
        kurtosis = np.mean(np.power(coeffs - mean, 4)) / np.power(std, 4)
        feature_vector.extend([mean, std, var, energy, entropy, kurtosis])
    return feature_vector

def extract_psd_features(signal, fs):
    # Compute the PSD using the periodogram method
    f, Pxx = periodogram(signal, fs)

    # Compute the PSD features
    feature1 = np.mean(Pxx[(f >= 0.5) & (f <= 4)])
    feature2 = np.mean(Pxx[(f >= 4) & (f <= 8)])
    feature3 = np.mean(Pxx[(f >= 8) & (f <= 13)])
    feature4 = np.mean(Pxx[(f >= 13) & (f <= 30)])
    feature5 = np.mean(Pxx[(f >= 30) & (f <= 60)])
    feature6 = np.mean(Pxx[(f >= 60) & (f <= 100)])
    
    # Return the feature values
    return [feature1, feature2, feature3, feature4, feature5, feature6]

def khaled_fun(resampled_Signal):
    feature_vector = []
    max_peak = get_max_peak(resampled_Signal)
    area = integrate.simpson(resampled_Signal)
    model = AutoReg(resampled_Signal, lags=4)
    model_fit = model.fit()
    mean, std = get_mean_and_std(resampled_Signal)
    kt = kurtosis(resampled_Signal)
    ar_pred = model_fit.predict(start=len(resampled_Signal), end=len(resampled_Signal))
    feature_vector.extend([max_peak, area, mean, std, kt, ar_pred[0]]) 

    return feature_vector

def preprocessing_signal(signal_v, signal_h):
    feature_extraction1 = []
    feature_extraction2 = []
    feature_extraction3 = []


    #////SIGNAL V////////////////

    # Invoking butter_bandpass_filter and retrieving the EOG Filtered Signal
    Filtered_Signal_1 = butter_bandpass_filter(signal_v, Low_CutOff=0.5, High_CutOff=20.0, SamplingRate=172, order=2)
    # Resampling the Filtered Signal
    Filtered_Signal_1 = resample(Filtered_Signal_1, 50)

    sampling_rates = [172]*30
    avg_sampling_rate = np.mean(sampling_rates)
    wvlt1 = wavlet_extract(Filtered_Signal_1)

    psd1 = extract_psd_features(Filtered_Signal_1,avg_sampling_rate)
    kh1 = khaled_fun(Filtered_Signal_1)
    #////SIGNAL H////////////////
    # Invoking butter_bandpass_filter and retrieving the EOG Filtered Signal
    Filtered_Signal_2 = butter_bandpass_filter(signal_h, Low_CutOff=0.5, High_CutOff=20.0, SamplingRate=172, order=2)
    # Resampling the Filtered Signal
    Filtered_Signal_2 = resample(Filtered_Signal_2, 50)
    wvlt2 = wavlet_extract(Filtered_Signal_2)

    psd2 = extract_psd_features(Filtered_Signal_2,avg_sampling_rate)
    kh2 = khaled_fun(Filtered_Signal_2)
    # Appending the kurtosis to the Yukari_feature array
    
    #Concatenate H V
    concWvlt = np.concatenate([wvlt1, wvlt2])
    concPsd = np.concatenate([psd1, psd2])
    conckh = np.concatenate([kh1, kh2])


    feature_extraction1.append(concWvlt)
    feature_extraction2.append(concPsd)
    feature_extraction3.append(conckh)

    return feature_extraction1,feature_extraction2,feature_extraction3


labels_dict = {'down': 0, 'blink': 1, 'right': 2, 'left': 3, 'up': 4}

# Load the model
loaded_model = joblib.load('random_forest_model.pkl')

# Function to process the signal files and display the result
def process_signal_files():
    # Get the selected files
    file_paths = [file1_path.get(), file2_path.get()]

    # Check if both files are selected
    if '' in file_paths:
        messagebox.showerror("Error", "Please select two signal files.")
        return

    # Check if the selected files have the same name with different last letters
    file_names = [os.path.basename(path) for path in file_paths]
    if not re.match(r'^.*[hHvV]\.txt$', file_names[0]) or not re.match(r'^.*[hHvV]\.txt$', file_names[1]):
        messagebox.showerror("Error", "File names should end with 'h' and 'v' respectively.")
        return
    if file_names[0][:-5] != file_names[1][:-5]:
        messagebox.showerror("Error", "File Names Should Be Identical The Same Signal.")
        return
    
    if file_names[0][-5] != "h" and file_names[1][-5] != "v":
        messagebox.showerror("Error", "The First File Should End With (H) And The Other With (V).")
        return

    Signal_h = np.loadtxt(file_paths[0])
    Signal_v = np.loadtxt(file_paths[1])

    # Process the signal files using machine learning (replace this with your ML code)
    # Extract the features for the test data
    features_dict_test =  preprocessing_signal(Signal_v, Signal_h)

    # Concatenate the extracted features for the test data
    data_test_x = np.concatenate([features_dict_test[2],],axis=0)

    # Get the predictions from the loaded model
    result = loaded_model.predict(data_test_x)
    # Show the corresponding message based on the result
    if result == 0:
        result_text.set("Result: Down (Drink)")
    elif result == 1:
        result_text.set("Result: Blink (Sleepy)")
    elif result == 2:
        result_text.set("Result: Right (Talk)")
    elif result == 3:
        result_text.set("Result: Left (Bathroom)")
    elif result == 4:
        result_text.set("Result: Up (Eat)")

# Create the main window
window = tk.Tk()
window.geometry("400x200")
window.title("EOG Signal Processing")

# Create variables to store the file paths
file1_path = tk.StringVar()
file2_path = tk.StringVar()

# Create a label and entry for the first file
tk.Label(window, text="Signal File 1:").pack()
file1_entry = tk.Entry(window, textvariable=file1_path, state='readonly', width=50)
file1_entry.pack()

# Create a button to browse and select the first file
def browse_file1():
    file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
    if file_path:
        file1_path.set(file_path)
browse_button1 = tk.Button(window, text="Browse", command=browse_file1)
browse_button1.pack()

# Create a label and entry for the second file
tk.Label(window, text="Signal File 2:").pack()
file2_entry = tk.Entry(window, textvariable=file2_path, state='readonly', width=50)
file2_entry.pack()

# Create a button to browse and select the second file
def browse_file2():
    file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
    if file_path:
        file2_path.set(file_path)
browse_button2 = tk.Button(window, text="Browse", command=browse_file2)
browse_button2.pack()

# Create a button to process the files
process_button = tk.Button(window, text="Process", command=process_signal_files)
process_button.pack()

# Create a label to display the result
result_text = tk.StringVar()
result_label = tk.Label(window, textvariable=result_text)
result_label.pack()

# Run the main event loop
window.mainloop()


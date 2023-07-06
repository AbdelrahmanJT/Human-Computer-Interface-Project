import pickle
import numpy as np
import pandas as pd
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

from scipy.signal import periodogram

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

def preprocessing_signal(data):
    feature_extraction1 = defaultdict(list)
    feature_extraction2 = defaultdict(list)
    feature_extraction3 = defaultdict(list)


    # Loading the file
    for key , signals in data.items():
        for s in range(0,len(data[key]),2):

                
            np_array1 = np.array(signals[s]).astype(np.float32)
            np_array2 = np.array(signals[s+1]).astype(np.float32)
            #////SIGNAL V////////////////

            # Invoking butter_bandpass_filter and retrieving the EOG Filtered Signal
            Filtered_Signal_1 = butter_bandpass_filter(np_array1, Low_CutOff=0.5, High_CutOff=20.0, SamplingRate=172, order=2)
            # Resampling the Filtered Signal
            Filtered_Signal_1 = resample(Filtered_Signal_1, 50)

            sampling_rates = [172]*30
            avg_sampling_rate = np.mean(sampling_rates)
            wvlt1 = wavlet_extract(Filtered_Signal_1)

            psd1 = extract_psd_features(Filtered_Signal_1,avg_sampling_rate)
            kh1 = khaled_fun(Filtered_Signal_1)
            #////SIGNAL H////////////////
            # Invoking butter_bandpass_filter and retrieving the EOG Filtered Signal
            Filtered_Signal_2 = butter_bandpass_filter(np_array2, Low_CutOff=0.5, High_CutOff=20.0, SamplingRate=172, order=2)
            # Resampling the Filtered Signal
            Filtered_Signal_2 = resample(Filtered_Signal_2, 50)
            wvlt2 = wavlet_extract(Filtered_Signal_2)

            psd2 = extract_psd_features(Filtered_Signal_2,avg_sampling_rate)
            kh2 = khaled_fun(Filtered_Signal_2)
            # Appending the kurtosis to the Yukari_feature array
            
            #Concatenate H V
            concWvlt = wvlt1 + wvlt2
            concPsd = psd1 + psd2
            conckh = kh1 + kh2


            feature_extraction1[key].append(concWvlt)
            feature_extraction2[key].append(concPsd)
            feature_extraction3[key].append(conckh)

    return feature_extraction1,feature_extraction2,feature_extraction3

df = pd.read_csv("EOG_dataset.csv")


df_down = df[df['label']=='Down'].sort_values('id')
df_Blink = df[df['label']=='Blink'].sort_values('id')
df_Right = df[df['label']=='Right'].sort_values('id')
df_Left = df[df['label']=='Left'].sort_values('id')
df_Up = df[df['label']=='Up'].sort_values('id')
#spliting Data

df_down_train, df_down_test = train_test_split(df_down, test_size=0.25, shuffle=False, random_state=1)

df_Blink_train, df_Blink_test = train_test_split(df_Blink, test_size=0.25, shuffle=False, random_state=1)

df_Right_train, df_Right_test = train_test_split(df_Right, test_size=0.25, shuffle=False, random_state=1)

df_Left_train, df_Left_test = train_test_split(df_Left, test_size=0.25, shuffle=False, random_state=1)

df_Up_train, df_Up_test = train_test_split(df_Up, test_size=0.25, shuffle=False, random_state=1)

#print train
print (f"Down train: {len(df_down_train)} , Blink train: {len(df_Blink_train)} , Right train: {len(df_Right_train)} , Left train: {len(df_Left_train)} , Up train: {len(df_Up_train)}")

#print test
print (f"Down train: {len(df_down_test)} , Blink train: {len(df_Blink_test)} , Right train: {len(df_Right_test)} , Left train: {len(df_Left_test)} , Up train: {len(df_Up_test)}")

#////////////////preprocessing training////////////////////
extarcted_signals_train = {
                    'down' : df_down_train.loc[: , '0':'250'].values.tolist(),
                    'blink' : df_Blink_train.loc[: , '0':'250'].values.tolist(),
                    'right' : df_Right_train.loc[: , '0':'250'].values.tolist(), 
                    'left' : df_Left_train.loc[: , '0':'250'].values.tolist(),
                    'up' : df_Up_train.loc[: , '0':'250'].values.tolist()
                    }

print(f" ana hena : {len(extarcted_signals_train['down'])}")

features_dict_train =  preprocessing_signal(extarcted_signals_train)
print(len(features_dict_train[0]['down']))
print(features_dict_train[0]['down'])
#print(len(features_dict[0]['right']))
data_train_x = np.concatenate([features_dict_train[2]['down'], 
                             features_dict_train[2]['blink'],
                             features_dict_train[2]['right'],
                             features_dict_train[2]['left'],
                             features_dict_train[2]['up']
                            ],axis=0)

labels_dict = {'down': 0, 'blink': 1, 'right': 2, 'left': 3, 'up': 4}

data_train_y = np.concatenate([
                             [labels_dict['down']] * len(features_dict_train[2]['down']), 
                             [labels_dict['blink']] * len(features_dict_train[2]['blink']), 
                             [labels_dict['right']] * len(features_dict_train[2]['right']), 
                             [labels_dict['left']] * len(features_dict_train[2]['left']), 
                             [labels_dict['up']] * len(features_dict_train[2]['up']) 
                            ],axis=0)

print(f"train features: {data_train_x.shape}")
print(f"labels : {data_train_y.shape}")

#labels
df_labels_train = pd.DataFrame({'labels': data_train_y})

#////////////preprocessing testing//////////////

extarcted_signals_test = {
                    'down' : df_down_test.loc[: , '0':'250'].values.tolist(),
                    'blink' : df_Blink_test.loc[: , '0':'250'].values.tolist(),
                    'right' : df_Right_test.loc[: , '0':'250'].values.tolist(), 
                    'left' : df_Left_test.loc[: , '0':'250'].values.tolist(),
                    'up' : df_Up_test.loc[: , '0':'250'].values.tolist()
                    }
                    
 
 
features_dict_test =  preprocessing_signal(extarcted_signals_test)

data_test_x = np.concatenate([features_dict_test[2]['down'], 
                             features_dict_test[2]['blink'],
                             features_dict_test[2]['right'],
                             features_dict_test[2]['left'],
                             features_dict_test[2]['up']
                            ],axis=0)

data_test_y = np.concatenate([
                             [labels_dict['down']] * len(features_dict_test[2]['down']), 
                             [labels_dict['blink']] * len(features_dict_test[2]['blink']), 
                             [labels_dict['right']] * len(features_dict_test[2]['right']), 
                             [labels_dict['left']] * len(features_dict_test[2]['left']), 
                             [labels_dict['up']] * len(features_dict_test[2]['up']) 
                            ],axis=0)


rf = RandomForestClassifier(n_estimators=200, random_state=42)

forest = rf.fit(data_train_x , data_train_y)

acc_rf = forest.score(data_test_x,data_test_y)

print(f"accuracy rf :{acc_rf}")

y_pred = forest.predict(data_test_x)
print(y_pred)
print(classification_report(data_test_y, y_pred))

from sklearn.ensemble import RandomForestClassifier
import joblib

# Assuming you have trained and obtained the 'forest' model

# Save the model
joblib.dump(forest, 'random_forest_model.pkl')


#KNN
k = 5
# Train the model
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(data_train_x, data_train_y)
# Test the model
y_pred = knn.predict(data_test_x)
acc_knn = accuracy_score(data_test_y, y_pred)
print("Accuracy KNN :", acc_knn)
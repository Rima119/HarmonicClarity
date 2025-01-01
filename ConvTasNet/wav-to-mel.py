from hmac import digest_size

import librosa
import matplotlib.pyplot as plt
import numpy as np

wave_path = r"E:\pycharm_projects\PythonProject\mel\Ludlows.mp3"

waveform,sample_rate = librosa.load(wave_path,sr=None)

frame_size,hop_size = 1024,512
n_fft = 1024

if len(waveform) % hop_size != 0:
    frame_num = int((len(waveform) - frame_size) / hop_size) + 1
    pad_num = frame_num * hop_size + frame_size - len(waveform)
    waveform =np.pad(waveform,pad_width=(0,pad_num),mode='wrap')
frame_num = int((len(waveform)-frame_size)/hop_size)+1

row = np.tile(np.arange(0,frame_size),(frame_num,1))
column = np.tile(np.arange(0,frame_num*hop_size,hop_size),(frame_size,1)).T
index = row + column

waveform_frame = waveform[index]
waveform_frame = waveform_frame*np.hanning(frame_size)

waveform_stft = np.fft.rfft(waveform_frame,n_fft)

waveform_pow = np.abs(waveform_stft)**2/n_fft
waveform_db = 20*np.log10(waveform_pow)

plt.figure(figsize=(10,10))
plt.imshow(waveform_db)
y_ticks = np.arange(0,int(n_fft/2),100)
plt.yticks(ticks=y_ticks,labels=y_ticks*sample_rate/n_fft)
plt.show()
print("hello")
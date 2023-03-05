---
title: Time Domain Audio 특징 이해와 구현 with librosa
date: 2022-08-07T09:23:28.272Z
categories: [Audio Signal Processing]
tags: [audio signal]		# TAG는 반드시 소문자
---


# Time-domain 특징
* Amplitude envelope (AE)
* Root-mean-square energy (RMS)
* Zero-crossing rate (ZCR)

## Amplitude envelope (AE)
* 프레임 내 모든 샘플의 최대 진폭 값
* 음량에 대한 대략적인 정보 제공
* Outliers에 민감함
* Onset detection, 음악 장르 분류

## Root-mean-square energy (RMS)
* 프레임에 있는 모든 표본의 RMS
* 음량 표시기
* AE보다 Outliers에 덜 민감함
* 오디오 세분화, 음악 장르 분류

## Zero-crossing rate (ZCR)
* 타악기 소리와 음조 인식
* Monophonic pitch 단음 음조 추정
* 음성 신호에 대한 Voice/unvoiced 결정

# 1. Basic informations of audio files

## Loading the libraries
```python
import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display
import IPython.display as ipd
```

## Loading audio files
```python
debussy_file = "debussy.wav"
redhot_file = "redhot.wav"
duke_file = "duke.wav"
```
```python
ipd.Audio(debussy_file)
```
<audio src="https://docs.google.com/uc?export=open&id=1t07O0ay5oIqtJzbPJbiYgAEp80S-Do2H" controls preload>
<p>Your browser does not support the audio element.</p>
</audio>

```python
ipd.Audio(redhot_file)
```
<audio src="https://docs.google.com/uc?export=open&id=1oOqKvNO3v51P6b1mFB7_sIkMSNfYQ37l" controls preload>
<p>Your browser does not support the audio element.</p>
</audio>

```python
ipd.Audio(duke_file)
```
<audio src="https://docs.google.com/uc?export=open&id=1Mo10gbImObYRdbsr80y4DeGce3Se4k-Y" controls preload>
<p>Your browser does not support the audio element.</p>
</audio>

```python
# load audio files with librosa
debussy, sr = librosa.load(debussy_file)
redhot, _ = librosa.load(redhot_file)
duke, _ = librosa.load(duke_file)
```
```python
sr
```
    22050

## Basic information regarding audio files
```python
debussy.shape
```
    (661500,)

```python
# duration in seconds of 1 sample
sample_duration = 1 / sr
print(f"One sample lasts for {sample_duration:6f} seconds")
```
    One sample lasts for 0.000045 seconds


```python
# total number of samples in audio file
tot_samples = len(debussy)
tot_samples
```




    661500




```python
# duration of debussy audio in seconds
duration = 1 / sr * tot_samples
print(f"The audio lasts for {duration} seconds")
```

    The audio lasts for 30.0 seconds


## Visualising audio signal in the time domain


```python
plt.figure(figsize=(15, 17))

plt.subplot(3, 1, 1)
librosa.display.waveplot(debussy, alpha=0.5)
plt.ylim((-1, 1))
plt.title("Debusy")

plt.subplot(3, 1, 2)
librosa.display.waveplot(redhot, alpha=0.5)
plt.ylim((-1, 1))
plt.title("RHCP")

plt.subplot(3, 1, 3)
librosa.display.waveplot(duke, alpha=0.5)
plt.ylim((-1, 1))
plt.title("Duke Ellington")

plt.show()
```  

![TimeDomain0](/assets/img/to/TimeDomain0.png)

# 1. Extracting the amplitude envelope feature from scratch
## Calculating amplitude envelope


```python
FRAME_SIZE = 1024
HOP_LENGTH = 512

def amplitude_envelope(signal, frame_size, hop_length):
    """Calculate the amplitude envelope of a signal with a given frame size nad hop length."""
    amplitude_envelope = []
    
    # calculate amplitude envelope for each frame
    for i in range(0, len(signal), hop_length): 
        amplitude_envelope_current_frame = max(signal[i:i+frame_size]) 
        amplitude_envelope.append(amplitude_envelope_current_frame)
    
    return np.array(amplitude_envelope)
```


```python
def fancy_amplitude_envelope(signal, frame_size, hop_length):
    """Fancier Python code to calculate the amplitude envelope of a signal with a given frame size."""
    return np.array([max(signal[i:i+frame_size]) for i in range(0, len(signal), hop_length)])
```


```python
# number of frames in amplitude envelope
ae_debussy = amplitude_envelope(debussy, FRAME_SIZE, HOP_LENGTH)
len(ae_debussy)
```




    1292




```python
# calculate amplitude envelope for RHCP and Duke Ellington
ae_redhot = amplitude_envelope(redhot, FRAME_SIZE, HOP_LENGTH)
ae_duke = amplitude_envelope(duke, FRAME_SIZE, HOP_LENGTH)
```

## Visualising amplitude envelope


```python
frames = range(len(ae_debussy))
t = librosa.frames_to_time(frames, hop_length=HOP_LENGTH)
```


```python
# amplitude envelope is graphed in red

plt.figure(figsize=(15, 17))

ax = plt.subplot(3, 1, 1)
librosa.display.waveplot(debussy, alpha=0.5)
plt.plot(t, ae_debussy, color="r")
plt.ylim((-1, 1))
plt.title("Debusy")

plt.subplot(3, 1, 2)
librosa.display.waveplot(redhot, alpha=0.5)
plt.plot(t, ae_redhot, color="r")
plt.ylim((-1, 1))
plt.title("RHCP")

plt.subplot(3, 1, 3)
librosa.display.waveplot(duke, alpha=0.5)
plt.plot(t, ae_duke, color="r")
plt.ylim((-1, 1))
plt.title("Duke Ellington")

plt.show()
```
![TimeDomain1](/assets/img/to/TimeDomain1.png)


# 2. Extracting Root-Mean Square Energy
## Root-mean-squared energy with Librosa


```python
FRAME_SIZE = 1024
HOP_LENGTH = 512
```


```python
rms_debussy = librosa.feature.rms(debussy, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
rms_redhot = librosa.feature.rms(redhot, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
rms_duke = librosa.feature.rms(duke, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
```

## Visualise RMSE + waveform


```python
frames = range(len(rms_debussy))
t = librosa.frames_to_time(frames, hop_length=HOP_LENGTH)
```


```python
# rms energy is graphed in red

plt.figure(figsize=(15, 17))

ax = plt.subplot(3, 1, 1)
librosa.display.waveplot(debussy, alpha=0.5)
plt.plot(t, rms_debussy, color="r")
plt.ylim((-1, 1))
plt.title("Debusy")

plt.subplot(3, 1, 2)
librosa.display.waveplot(redhot, alpha=0.5)
plt.plot(t, rms_redhot, color="r")
plt.ylim((-1, 1))
plt.title("RHCP")

plt.subplot(3, 1, 3)
librosa.display.waveplot(duke, alpha=0.5)
plt.plot(t, rms_duke, color="r")
plt.ylim((-1, 1))
plt.title("Duke Ellington")

plt.show()
```
![TimeDomain2](/assets/img/to/TimeDomain2.png)

## RMSE from scratch


```python
def rmse(signal, frame_size, hop_length):
    rmse = []
    
    # calculate rmse for each frame
    for i in range(0, len(signal), hop_length): 
        rmse_current_frame = np.sqrt(sum(signal[i:i+frame_size]**2) / frame_size)
        rmse.append(rmse_current_frame)
    return np.array(rmse)  
```


```python
rms_debussy1 = rmse(debussy, FRAME_SIZE, HOP_LENGTH)
rms_redhot1 = rmse(redhot, FRAME_SIZE, HOP_LENGTH)
rms_duke1 = rmse(duke, FRAME_SIZE, HOP_LENGTH)
```


```python
plt.figure(figsize=(15, 17))

ax = plt.subplot(3, 1, 1)
librosa.display.waveplot(debussy, alpha=0.5)
plt.plot(t, rms_debussy, color="r")
plt.plot(t, rms_debussy1, color="y")
plt.ylim((-1, 1))
plt.title("Debusy")

plt.subplot(3, 1, 2)
librosa.display.waveplot(redhot, alpha=0.5)
plt.plot(t, rms_redhot, color="r")
plt.plot(t, rms_redhot1, color="y")
plt.ylim((-1, 1))
plt.title("RHCP")

plt.subplot(3, 1, 3)
librosa.display.waveplot(duke, alpha=0.5)
plt.plot(t, rms_duke, color="r")
plt.plot(t, rms_duke1, color="y")
plt.ylim((-1, 1))
plt.title("Duke Ellington")

plt.show()
```
![TimeDomain3](/assets/img/to/TimeDomain3.png)

# 3. Extracting Zero-Crossing Rate from Audio

## Zero-crossing rate with Librosa


```python
zcr_debussy = librosa.feature.zero_crossing_rate(debussy, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
zcr_redhot = librosa.feature.zero_crossing_rate(redhot, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
zcr_duke = librosa.feature.zero_crossing_rate(duke, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
```


```python
zcr_debussy.size
```


## Visualise zero-crossing rate with Librosa


```python
plt.figure(figsize=(15, 10))

plt.plot(t, zcr_debussy, color="y")
plt.plot(t, zcr_redhot, color="r")
plt.plot(t, zcr_duke, color="b")
plt.ylim(0, 1)
plt.show()
```
![TimeDomain4](/assets/img/to/TimeDomain4.png)

## ZCR: Voice vs Noise


```python
voice_file = "voice.wav"
noise_file = "noise.wav"
```


```python
ipd.Audio(voice_file)
```
<audio src="https://docs.google.com/uc?export=open&id=1BjYJWW0F7ZCj8SPfV3mf2nL951faVBs3" controls preload>
<p>Your browser does not support the audio element.</p>
</audio>

```python
ipd.Audio(noise_file)
```
<audio src="https://docs.google.com/uc?export=open&id=17xNPjXX-HSpNJUzCejE_SHkr8bRi7x28" controls preload>
<p>Your browser does not support the audio element.</p>
</audio>

```python
# load audio files
voice, _ = librosa.load(voice_file, duration=15)
noise, _ = librosa.load(noise_file, duration=15)
```


```python
# get ZCR
zcr_voice = librosa.feature.zero_crossing_rate(voice, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
zcr_noise = librosa.feature.zero_crossing_rate(noise, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
```


```python
frames = range(len(zcr_voice))
t = librosa.frames_to_time(frames, hop_length=HOP_LENGTH)
```


```python
plt.figure(figsize=(15, 10))

plt.plot(t, zcr_voice, color="y")
plt.plot(t, zcr_noise, color="r")
plt.ylim(0, 1)
plt.show()
```
![TimeDomain5](/assets/img/to/TimeDomain5.png)


# Reference
* [musikalkemist/AudioSignalProcessingForML](https://github.com/musikalkemist/AudioSignalProcessingForML)

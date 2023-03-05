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

# Extracting the amplitude envelope feature from scratch in Python

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
<audio controls="controls">
  <source type="audio/wav" src="_site/assets/audio/debussy.wav"></source>
  <p>Your browser does not support the audio element.</p>
</audio>




# How to Extract Root-Mean Square Energy and Zero-Crossing Rate from Audio


























# Reference
* [musikalkemist/AudioSignalProcessingForML](https://github.com/musikalkemist/AudioSignalProcessingForML)

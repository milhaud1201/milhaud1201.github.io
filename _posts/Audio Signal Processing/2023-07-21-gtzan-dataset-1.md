---
title: [ML] GTZAN dataset으로 음악 장르 분류 - 1
date: 2023-07-21T09:23:28.272Z
categories: [Flutter, Dart]
tags: [flutter, dart]		# TAG는 반드시 소문자
---

# GTZAN dataset 알아보자
>  💡 Note:
* 연구 목적 무료 사용 가능
* 10 장르 x 30초 길이 x 100개 오디오 트랙
* 각 트랙은 22,050Hz Mono 16-bit audio files, WAV format
* 비교적 고품질, 잡음이나 왜곡이 없음


GTZAN 데이터셋은 음악 장르 인식(`MGR`)에서 Machine listening research에서 평가를 위해 가장 많이 사용되는 테이터셋 중 하나이다. 다양한 녹음 조건을 나타내기 위해 개인 CDs, 라디오, 마이크 녹음을 포함한 다양한 소스에서 2001-2001년에 수집되었다. 실제로는 MGR용으로 명시되지 않았지만 `availablilty`로 인해 MIR 시스템을 비교하기 위한 측정 지표로써 benchmark 데이터셋이 되었다, e.g., [[1]](https://ieeexplore.ieee.org/document/5664796). 모든 트랙이 서구권에서 온 것으로 전세계를 대표하지 않을 수 있으며 모두 같은 key로 구성되어 분류가 더 쉽고 고품질로 인해 real world를 대표하지 않을 수 있다는 점[[2]](https://arxiv.org/abs/1306.1461)에서 한계가 있다. 그럼에도 GTZAN 데이터셋은 2001년 이후로 현재까지도 매년 100번 이상의 인용되고[[3] ](https://scholar.google.co.kr/citations?view_op=view_citation&hl=en&user=yPgxxpwAAAAJ&citation_for_view=yPgxxpwAAAAJ:u5HHmVD_uO8C)이는 많은 연구에서 사용되었으며, MGR에서 새로운 method 발전을 위해 사용되었던 가치있는 리소스라는 것에는 충분하다.
## 데이터셋 어디서 받을 수 있나
전에는 George Tzanetakis의 웹사이트에서 GTZAN 데이터셋을 다운로드받을 수 있었으나 더 이상 유지 관리되지 않으며 TensorFlow 데이터셋 저장소[[4]](https://www.tensorflow.org/datasets/catalog/gtzan)로 이동되었다. kaggle에서는 오디오 트랙을 시각적으로 표현한 Mel Spectrogram으로 변환한 이미지 원본 파일[[5]](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification?resource=download&select=Data)을 포함하여 받을 수 있다. Pytorch를 쓰는 분이라면 torchaudio dataset도 있다.
![](https://velog.velcdn.com/images/milhaud/post/8e386877-2251-4f4c-bfc4-b36b8a54e165/image.png)

## MGR을 위한 오디오 데이터 전처리
데이터 전처리를 하기 전에 가장 중요한 것은 데이터인지 탐색하고 관찰하는 과정이다. data augmentation할 때에도 적용되어 반드시 이 과정이 필요하다.
### 데이터셋 준비
kaggle에서 2개의 디렉토리와 2개의 파일이 있는 GTZAN 데이터셋을 다운받고 파일의 압축을 풀어준다.
```python
kaggle datasets download -d andradaolteanu/gtzan-dataset-music-genre-classification
!unzip -qq "/content/data.zip"
```
<details style="text-align: right">
<summary>Click to show</summary>
<div style="text-align: left">

```python
from torchaudio.datasets import GTZAN

train_loader = get_dataloader(
    data_path="../../codes/split", split="train", is_augmentation=False
)
dataset = train_loader.dataset
idx = 5
print(f"Number of datapoints in the GTZAN dataset: f{len(dataset)}\n")
print(f"Selected track no.: {idx}")
audio, sr, genre = dataset[idx]
print(
    f"Genre: {genre}\nSample rate: {sr}\nChannels: {audio.shape[0]}\nSamples: {audio.shape[1]}"
)
display(Audio(audio, rate=sr))
```

</div>
</details>
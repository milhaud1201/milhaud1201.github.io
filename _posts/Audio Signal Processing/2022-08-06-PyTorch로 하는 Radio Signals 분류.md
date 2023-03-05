---
title: PyTorch로 하는 Radio Signals 분류
date: 2022-08-06T09:23:28.272Z
categories: [Audio Signal Processing]
tags: [audio signal, classification]		# TAG는 반드시 소문자
---

# 개요
* 이번 테스크에서는 Sepectrogram이 무엇인지 알 수 있다.  
* Sepectrogram의 이미지가 입력값으로 주어진다. 오디오가 있을 때마다 파일 또는 오디오 신호를 Sepectrogram을 사용하여 나타낸다.  
* 이론적인 부분에서, Covolutional Neural Network의 작동과 Covolutional Neural Network가 무엇인지 분명히 한다.  
* Optimization이 어떻게 작동하는지, Gradient decent나 adam과 같은 최적화 유형을 알아야 한다.  
* Python 프로그래밍과 PyTorch의 기본기가 명확해야 한다.  

# Imports 


```python
!pip install timm
```

Installing collected packages: huggingface-hub, timm  
Successfully installed huggingface-hub-0.12.1 timm-0.6.12



```python
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

import torch 
from torch import nn, optim 
from torch.utils.data import Dataset, DataLoader 

from torchvision import transforms as T

import timm
```

# Configurations

```python
 TRAIN_CSV = 'train.csv'
 VALID_CSV = 'valid.csv'

 BATCH_SIZE = 128
 DEVICE = 'cuda'

 MODEL_NAME = 'efficientnet_b0'

 LR = 0.001
 EPOCHS = 15
```

* 각 컬럼은 스펙트로그램 이미지의 각 픽셀 값을 나타낸다.
* 스펙트로그램 이미지를 1차원 벡터로 convert 할 것이다.


```python
df_train = pd.read_csv(TRAIN_CSV)
df_valid = pd.read_csv(VALID_CSV)

df_train.head()
```

  <div id="df-c9c81121-7a31-4321-8605-3e17caf03290">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>8183</th>
      <th>8184</th>
      <th>8185</th>
      <th>8186</th>
      <th>8187</th>
      <th>8188</th>
      <th>8189</th>
      <th>8190</th>
      <th>8191</th>
      <th>labels</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.631373</td>
      <td>0.623529</td>
      <td>0.713726</td>
      <td>0.705882</td>
      <td>0.658824</td>
      <td>0.666667</td>
      <td>0.654902</td>
      <td>0.635294</td>
      <td>0.647059</td>
      <td>0.705882</td>
      <td>...</td>
      <td>0.611765</td>
      <td>0.650980</td>
      <td>0.658824</td>
      <td>0.600000</td>
      <td>0.603922</td>
      <td>0.654902</td>
      <td>0.694118</td>
      <td>0.658824</td>
      <td>0.666667</td>
      <td>Squiggle</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.725490</td>
      <td>0.752941</td>
      <td>0.749020</td>
      <td>0.701961</td>
      <td>0.690196</td>
      <td>0.721569</td>
      <td>0.709804</td>
      <td>0.745098</td>
      <td>0.654902</td>
      <td>0.721569</td>
      <td>...</td>
      <td>0.698039</td>
      <td>0.721569</td>
      <td>0.686275</td>
      <td>0.713726</td>
      <td>0.682353</td>
      <td>0.690196</td>
      <td>0.698039</td>
      <td>0.701961</td>
      <td>0.725490</td>
      <td>Squiggle</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.717647</td>
      <td>0.701961</td>
      <td>0.713726</td>
      <td>0.733333</td>
      <td>0.705882</td>
      <td>0.717647</td>
      <td>0.725490</td>
      <td>0.682353</td>
      <td>0.717647</td>
      <td>0.674510</td>
      <td>...</td>
      <td>0.694118</td>
      <td>0.705882</td>
      <td>0.682353</td>
      <td>0.639216</td>
      <td>0.713726</td>
      <td>0.670588</td>
      <td>0.678431</td>
      <td>0.737255</td>
      <td>0.674510</td>
      <td>Squiggle</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.705882</td>
      <td>0.674510</td>
      <td>0.654902</td>
      <td>0.678431</td>
      <td>0.666667</td>
      <td>0.662745</td>
      <td>0.678431</td>
      <td>0.662745</td>
      <td>0.686275</td>
      <td>0.686275</td>
      <td>...</td>
      <td>0.662745</td>
      <td>0.631373</td>
      <td>0.643137</td>
      <td>0.705882</td>
      <td>0.662745</td>
      <td>0.705882</td>
      <td>0.666667</td>
      <td>0.654902</td>
      <td>0.631373</td>
      <td>Squiggle</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.647059</td>
      <td>0.729412</td>
      <td>0.701961</td>
      <td>0.674510</td>
      <td>0.611765</td>
      <td>0.698039</td>
      <td>0.713726</td>
      <td>0.662745</td>
      <td>0.701961</td>
      <td>0.674510</td>
      <td>...</td>
      <td>0.670588</td>
      <td>0.705882</td>
      <td>0.674510</td>
      <td>0.721569</td>
      <td>0.694118</td>
      <td>0.674510</td>
      <td>0.705882</td>
      <td>0.749020</td>
      <td>0.729412</td>
      <td>Squiggle</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 8193 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-c9c81121-7a31-4321-8605-3e17caf03290')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none">
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z">
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-c9c81121-7a31-4321-8605-3e17caf03290 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-c9c81121-7a31-4321-8605-3e17caf03290');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
</div>


```python
print(f"No. of examples present in df_train : {len(df_train)}")
print(f"No. of examples present in df_valid : {len(df_valid)}")
print(f"Labels are : {df_train['labels'].unique()}")
```

    No. of examples present in df_train : 3200
    No. of examples present in df_valid : 800
    Labels are : ['Squiggle' 'Narrowbanddrd' 'Noises' 'Narrowband']


### 이미지 시각화
지정된 인덱스로 픽셀 값을 이미지로 보여준다.


```python
idx = 80

row = df_train.iloc[idx]

image_pixels = np.array(row[0:8192], dtype = np.float64)
label = row.labels

image = np.resize(image_pixels, (64, 128))  #64*128 = 8192

plt.imshow(image)
plt.title(label);
```  
  
![classify_radio_signals_0](/_site/assets/img/to/RadioSignals0.png)


# Declare Spec Augmentations 

개와 고양이 이미지 분류에서의 전통적인 이미지 증강 기법은 스펙트로그램 이미지에서는 사용할 수 없다. 오디오 신호의 상태를 바꾸기 때문에 spec augmnet로써 spectrogram augmentation을 사용할 것이다. 

### spectrogram augmentation 유형  
1. Time masks (vertical strips)
2. Frequency mark masks (horizontal strips)

사용할 모듈에는 두 개의 매개 변수를 선언할 수 있다.
* Mask의 width
* Mask의 number


```python
from spec_augment import TimeMask, FreqMask
```


```python
def get_train_transform():
    return T.Compose([
        TimeMask(T = 15, num_masks = 4),
        FreqMask(F = 15, num_masks = 3)
    ])
```

# Create Custom Dataset 

사용자 지정 데이터셋은 이미지와 레이블 쌍을 반환하는 것만 한다.
* get item에서 모델에 이미지를 전달할 때 채널 차수가 필요하기 때문에 general dimension을 추가했다.


```python
class SpecDataset(Dataset):

    def __init__(self, df, augmentations = None):
        self.df = df
        self.augmentations = augmentations

        label_mapper = {
            'Squiggle' : 0,
            'Narrowband' : 1,
            'Narrowbanddrd' : 2,
            'Noises' : 3
        }

        self.df.loc[:, 'labels'] = self.df.labels.map(label_mapper)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]
        image_pixels = np.array(row[0: 8192], dtype = np.float64)

        image = np.resize(image_pixels, (64, 128, 1))  #(h, w, c)
        label = np.array(row.labels, dtype = np.int64)

        image = torch.Tensor(image).permute(2, 0, 1)  #(c, h, w)

        if self.augmentations != None:
            image = self.augmentations(image)

        return image.float(), label
```


```python
trainset = SpecDataset(df_train, get_train_transform())
validset = SpecDataset(df_valid)
```


```python
image, label = trainset[591]

plt.imshow(image.permute(0, 1, 2).squeeze())
print(label)
```

    0



    
![classify_radio_signals_1](/_site/assets/img/to/RadioSignals1.png)
    


# Load dataset into Batches

앞서 batch size를 128로 선언했다.


```python
trainloader = DataLoader(trainset, batch_size = BATCH_SIZE, shuffle = True)
validloader = DataLoader(validset, batch_size = BATCH_SIZE)
```


```python
print(f"Total no. of batches in trainloader : {len(trainloader)}")
print(f"Total no. of batches in validloader : {len(validloader)}")
```

    Total no. of batches in trainloader : 25
    Total no. of batches in validloader : 7



```python
for images, labels in trainloader:
    break;

print(f"One image batch shape : {images.shape}")
print(f"One label batch shape : {labels.shape}")
```

    One image batch shape : torch.Size([128, 1, 64, 128])
    One label batch shape : torch.Size([128])


# Load Model


```python
class SpecModel(nn.Module):

    def __init__(self):
        super(SpecModel, self).__init__()

        self.net = timm.create_model(MODEL_NAME, num_classes = 4, pretrained = True, in_chans = 1)

    def forward(self, images, labels = None):

        logits = self.net(images)

        if label != None:
            loss = nn.CrossEntropyLoss()
            return logits, loss(logits, labels)
        
        return logits
```


```python
model = SpecModel().to(DEVICE)
model;
```

# Create Train and Eval Function


```python
from tqdm.notebook import tqdm 
from utils import multiclass_accuracy
```


```python
def train_fn(model, dataloader, optimizer, current_epoch):

    model.train()
    total_loss = 0.0
    total_acc = 0.0
    progress_bar = tqdm(dataloader, desc = "EPOCH" + "[TRAIN]" + str(current_epoch + 1) + '/' + str(EPOCHS))

    for t, data in enumerate(progress_bar):
        images, labels = data
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        logits, loss = model(images, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += multiclass_accuracy(logits, labels)

        temp = {'loss': '%6f' %float(total_loss/ (t+1)), 'acc': '%6f' %float(total_acc/ (t+1))}

        progress_bar.set_postfix(temp)

    return total_loss / len(dataloader), total_acc / len(dataloader)
```


```python
def eval_fn(model, dataloader, current_epoch):

    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    progress_bar = tqdm(dataloader, desc = "EPOCH" + "[VALID]" + str(current_epoch + 1) + '/' + str(EPOCHS))

    with torch.no_grad():
        for t, data in enumerate(progress_bar):
            images, labels = data
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            logits, loss = model(images, labels)

            total_loss += loss.item()
            total_acc += multiclass_accuracy(logits, labels)

            temp = {'loss': '%6f' %float(total_loss/ (t+1)), 'acc': '%6f' %float(total_acc/ (t+1))}

            progress_bar.set_postfix(temp)

        return total_loss / len(dataloader), total_acc / len(dataloader)
```

# Training Loop 


```python
def fit(model, trainloader, validloader, optimizer):

    best_valid_loss = np.Inf

    for i in range(EPOCHS):
        train_loss, train_acc = train_fn(model, trainloader, optimizer, i)
        valid_loss, valid_acc = eval_fn(model, validloader, i)

        if valid_loss < best_valid_loss:
            torch.save(model.state_dict(), MODEL_NAME + '-best_weights.pt')
            print('SAVED-BEST-WEIGHTS')
            best_valid_loss = valid_loss
```


```python
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
fit(model, trainloader, validloader, optimizer)
```

```
    EPOCH[TRAIN]1/15:   0%|          | 0/25 [00:00<?, ?it/s]



    EPOCH[VALID]1/15:   0%|          | 0/7 [00:00<?, ?it/s]


    SAVED-BEST-WEIGHTS



    EPOCH[TRAIN]2/15:   0%|          | 0/25 [00:00<?, ?it/s]



    EPOCH[VALID]2/15:   0%|          | 0/7 [00:00<?, ?it/s]


    SAVED-BEST-WEIGHTS



    EPOCH[TRAIN]3/15:   0%|          | 0/25 [00:00<?, ?it/s]



    EPOCH[VALID]3/15:   0%|          | 0/7 [00:00<?, ?it/s]


    SAVED-BEST-WEIGHTS



    EPOCH[TRAIN]4/15:   0%|          | 0/25 [00:00<?, ?it/s]



    EPOCH[VALID]4/15:   0%|          | 0/7 [00:00<?, ?it/s]


    SAVED-BEST-WEIGHTS



    EPOCH[TRAIN]5/15:   0%|          | 0/25 [00:00<?, ?it/s]



    EPOCH[VALID]5/15:   0%|          | 0/7 [00:00<?, ?it/s]



    EPOCH[TRAIN]6/15:   0%|          | 0/25 [00:00<?, ?it/s]



    EPOCH[VALID]6/15:   0%|          | 0/7 [00:00<?, ?it/s]



    EPOCH[TRAIN]7/15:   0%|          | 0/25 [00:00<?, ?it/s]



    EPOCH[VALID]7/15:   0%|          | 0/7 [00:00<?, ?it/s]



    EPOCH[TRAIN]8/15:   0%|          | 0/25 [00:00<?, ?it/s]



    EPOCH[VALID]8/15:   0%|          | 0/7 [00:00<?, ?it/s]



    EPOCH[TRAIN]9/15:   0%|          | 0/25 [00:00<?, ?it/s]



    EPOCH[VALID]9/15:   0%|          | 0/7 [00:00<?, ?it/s]



    EPOCH[TRAIN]10/15:   0%|          | 0/25 [00:00<?, ?it/s]



    EPOCH[VALID]10/15:   0%|          | 0/7 [00:00<?, ?it/s]



    EPOCH[TRAIN]11/15:   0%|          | 0/25 [00:00<?, ?it/s]



    EPOCH[VALID]11/15:   0%|          | 0/7 [00:00<?, ?it/s]



    EPOCH[TRAIN]12/15:   0%|          | 0/25 [00:00<?, ?it/s]



    EPOCH[VALID]12/15:   0%|          | 0/7 [00:00<?, ?it/s]



    EPOCH[TRAIN]13/15:   0%|          | 0/25 [00:00<?, ?it/s]



    EPOCH[VALID]13/15:   0%|          | 0/7 [00:00<?, ?it/s]



    EPOCH[TRAIN]14/15:   0%|          | 0/25 [00:00<?, ?it/s]



    EPOCH[VALID]14/15:   0%|          | 0/7 [00:00<?, ?it/s]



    EPOCH[TRAIN]15/15:   0%|          | 0/25 [00:00<?, ?it/s]



    EPOCH[VALID]15/15:   0%|          | 0/7 [00:00<?, ?it/s]
```

# Inference 


```python
from utils import view_classify
```
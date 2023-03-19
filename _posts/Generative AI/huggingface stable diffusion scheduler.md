---
title: [Stable Diffusion] Hugging Face에서 다른 Scheduler 사용하기
date: 2022-10-01T09:23:28.272Z
categories: [Generative AI]
tags: [generative ai, stable diffusion]		# TAG는 반드시 소문자
---
<a target="_blank" href="https://colab.research.google.com/drive/1fCSlwJO6dsoXvzfqSxdqdcQtBPb9R0hF?usp=share_link">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
# Schedulers

**Stable diffusion**은 일반적으로 노이즈에서 덜 노이즈가 많은 샘플로의 순방향 패스를 정의하는 반면,  
**Scheduler**는 전체 노이즈 제거 프로세스를 정의한다.

* 노이즈 제거 단계 Denoising steps가 몇 개인지?
* 확률론적 Stochastic 또는 결정론적인지 Deterministic?
* 노이즈 제거된 샘플을 찾는 데 사용할 알고리즘은 무엇인지?

**노이즈 제거 속도 Denoising speed**와 **노이즈 제거 품질 Denoising quality** 사이의 trade-off를 정의하는 경우가 많다.  
다만, 복잡하고 어떤 스케줄러가 가장 잘 작동하는지 정량적 측정하는 것이 어렵기 때문에, 다양한 시도를 해보는 것이 좋다.

# Load pipeline


```python
# !pip install huggingface_hub
!pip install diffusers==0.11.1
# !pip install transformers
```


```python
from huggingface_hub import login
from diffusers import DiffusionPipeline
import torch

# first we need to login with our access token
# login()

# Now we can download the pipeline
pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
```

GPU 활용:


```python
pipeline.to("cuda")
```




    StableDiffusionPipeline {
      "_class_name": "StableDiffusionPipeline",
      "_diffusers_version": "0.11.1",
      "feature_extractor": [
        "transformers",
        "CLIPFeatureExtractor"
      ],
      "requires_safety_checker": true,
      "safety_checker": [
        "stable_diffusion",
        "StableDiffusionSafetyChecker"
      ],
      "scheduler": [
        "diffusers",
        "PNDMScheduler"
      ],
      "text_encoder": [
        "transformers",
        "CLIPTextModel"
      ],
      "tokenizer": [
        "transformers",
        "CLIPTokenizer"
      ],
      "unet": [
        "diffusers",
        "UNet2DConditionModel"
      ],
      "vae": [
        "diffusers",
        "AutoencoderKL"
      ]
    }



# Access the scheduler

Scheduler는 파이프라인의 구성요소 중 하나이고, "scheduler" 속성을 통해 엑세스할 수 있다.


```python
pipeline.scheduler
```




    LMSDiscreteScheduler {
      "_class_name": "LMSDiscreteScheduler",
      "_diffusers_version": "0.11.1",
      "beta_end": 0.012,
      "beta_schedule": "scaled_linear",
      "beta_start": 0.00085,
      "clip_sample": false,
      "num_train_timesteps": 1000,
      "prediction_type": "epsilon",
      "set_alpha_to_one": false,
      "skip_prk_steps": true,
      "steps_offset": 1,
      "trained_betas": null
    }



# Scheduler 성능 비교
위의 Scheduler는 PNDMScheduler 타입을 보여준다. 다른 Scheduler와 성능을 비교해볼 수 있다.

1. 다른 스케줄러와 테스트하기 위한 Prompt를 정의한다.
2. 유사한 이미지를 생성하여 비교하는데, 이때 파이프라인을 실행할 수 있도록 Random seed로 `torch.Generator`를 생성한다.

## Scheduler로 이미지 생성하기

### PNDMScheduler


```python
prompt = "A photograph of an astronaut riding a horse on Mars, high resolution, high definition."
```


```python
generator = torch.Generator(device="cuda").manual_seed(8)
image = pipeline(prompt, generator=generator).images[0]
image
```


      0%|          | 0/50 [00:00<?, ?it/s]





    
![png](Hugging%20Face%20Using%20Stable%20Diffusion%20Scheduler_files/Hugging%20Face%20Using%20Stable%20Diffusion%20Scheduler_13_1.png)
    



## 비교할 Scheduler로 바꾸기

모든 Scheduler에는 호환 가능한 스케줄러를 모두 정의하는 `SchedulerMixin.compatibles` 속성을 가지고 있다.   
아래와 같이, Stable Diffusion 파이프라인에서 호환 가능한 스케줄러를 보여준다.


```python
pipeline.scheduler.compatibles
```




    [diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler,
     diffusers.schedulers.scheduling_pndm.PNDMScheduler,
     diffusers.schedulers.scheduling_heun_discrete.HeunDiscreteScheduler,
     diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler,
     diffusers.schedulers.scheduling_lms_discrete.LMSDiscreteScheduler,
     diffusers.schedulers.scheduling_ddim.DDIMScheduler,
     diffusers.schedulers.scheduling_euler_ancestral_discrete.EulerAncestralDiscreteScheduler,
     diffusers.schedulers.scheduling_dpmsolver_singlestep.DPMSolverSinglestepScheduler,
     diffusers.schedulers.scheduling_ddpm.DDPMScheduler]



Pipeline의 Scheduler를 변경하려면 `ConfigMixin.from_config()`함수로 `ConfigMixin.config` 속성을 간단하게 사용할 수 있다.


```python
pipeline.scheduler.config
```




    FrozenDict([('num_train_timesteps', 1000),
                ('beta_start', 0.00085),
                ('beta_end', 0.012),
                ('beta_schedule', 'scaled_linear'),
                ('trained_betas', None),
                ('skip_prk_steps', True),
                ('set_alpha_to_one', False),
                ('prediction_type', 'epsilon'),
                ('steps_offset', 1),
                ('_class_name', 'PNDMScheduler'),
                ('_diffusers_version', '0.11.1'),
                ('clip_sample', False)])



### DDIMScheduler



```python
from diffusers import DDIMScheduler

pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
```


```python
generator = torch.Generator(device="cuda").manual_seed(8)
image = pipeline(prompt, generator=generator).images[0]
image
```


      0%|          | 0/50 [00:00<?, ?it/s]





    
![png](Hugging%20Face%20Using%20Stable%20Diffusion%20Scheduler_files/Hugging%20Face%20Using%20Stable%20Diffusion%20Scheduler_21_1.png)
    



### LMSDiscreteScheduler


```python
from diffusers import LMSDiscreteScheduler

pipeline.scheduler = LMSDiscreteScheduler.from_config(pipeline.scheduler.config)

generator = torch.Generator(device="cuda").manual_seed(8)
image = pipeline(prompt, generator=generator).images[0]
image
```


      0%|          | 0/50 [00:00<?, ?it/s]





    
![png](Hugging%20Face%20Using%20Stable%20Diffusion%20Scheduler_files/Hugging%20Face%20Using%20Stable%20Diffusion%20Scheduler_23_1.png)
    



# JAX/FLAX로 Scheduler 변경

JAX는 Google Research에서 개발한 고성능 수치 컴퓨팅 및 기계 학습 연구에 사용되는 프레임워크이고, Flax는 JAX 기반 신경망 라이브러리로 초기에 Google Research의 Brain Team에서 개발했지만(JAX 팀과 긴밀히 협력하여) 현재는 오픈 소스이다. 대규모 언어 모델을 사용하는 프로젝트에 적합하다.

JAX/FLAX getting started: https://www.kaggle.com/getting-started/315696


```python
!pip install flax
```


```python
import flax
import jax
import numpy as np
from flax.jax_utils import replicate
from flax.training.common_utils import shard

from diffusers import FlaxStableDiffusionPipeline, FlaxDPMSolverMultistepScheduler

model_id = "runwayml/stable-diffusion-v1-5"
scheduler, scheduler_state = FlaxDPMSolverMultistepScheduler.from_pretrained(
    model_id,
    subfolder="scheduler"
)
pipeline, params = FlaxStableDiffusionPipeline.from_pretrained(
    model_id,
    scheduler=scheduler,
    revision="bf16",
    dtype=jax.numpy.bfloat16,
)
params["scheduler"] = scheduler_state

# Generate 1 image per parallel device (8 on TPUv2-8 or TPUv3-8)
prompt = "a photo of an astronaut riding a horse on mars"
num_samples = jax.device_count()
prompt_ids = pipeline.prepare_inputs([prompt] * num_samples)

prng_seed = jax.random.PRNGKey(0)
num_inference_steps = 25

# shard inputs and rng
params = replicate(params)
prng_seed = jax.random.split(prng_seed, jax.device_count())
prompt_ids = shard(prompt_ids)

images = pipeline(prompt_ids, params, prng_seed, num_inference_steps, jit=True).images
images = pipeline.numpy_to_pil(np.asarray(images.reshape((num_samples,) + images.shape[-3:])))
```

# Reference
https://huggingface.co/docs/diffusers/using-diffusers/schedulers#access-the-scheduler

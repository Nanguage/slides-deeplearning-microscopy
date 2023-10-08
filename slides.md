---
theme: seriph
#background: https://source.unsplash.com/collection/94734566/1920x1080
background: https://raw.githubusercontent.com/Nanguage/slides-deeplearning-microscopy/main/statics/img/ai_and_microscopy.jpg
class: text-center
highlighter: shiki
lineNumbers: false
info: |
  ## Slidev Starter Template
  Presentation slides for developers.

  Learn more at [Sli.dev](https://sli.dev)
drawings:
  persist: false
title: Deep learning in microscopy
mdc: true
hideInToc: true
---

# Deep learning in microscopy

Deep Learning (DL) methods are powerful analytical tools for microscopy.

Weize Xu, HZAU · 2023.10

<div class="pt-12">
  <span @click="$slidev.nav.next" class="px-2 py-1 rounded cursor-pointer" hover="bg-white bg-opacity-10">
    Press Space for next page <carbon:arrow-right class="inline"/>
  </span>
</div>

<div class="abs-br m-6 flex gap-2">
  <button @click="$slidev.nav.openInEditor()" title="Open in Editor" class="text-xl slidev-icon-btn opacity-50 !border-none !hover:text-white">
    <carbon:edit />
  </button>
  <a href="https://github.com/Nanguage/slide-deeplearning-microscopy" target="_blank" alt="GitHub"
    class="text-xl slidev-icon-btn opacity-50 !border-none !hover:text-white">
    <carbon-logo-github />
  </a>
</div>

<!--
left bottom corner show the image source:
Cover image made with DALLE·3
-->

<!--
The last comment block of each slide will be treated as slide notes. It will be visible and editable in Presenter Mode along with the slide. [Read more in the docs](https://sli.dev/guide/syntax.html#notes)
-->

---
layout: default
hideInToc: true
---

# Table of contents

<Toc maxDepth="2" columns="2"></Toc>

---
layout: center
class: text-center
---

# Introduction

---
level: 2
layout: default
---

# What and Why

<div style="display:flex;gap:40px">
<img src="/statics/img/dl_img_proc.webp" style="width: 300px"/>
<img src="/statics/img/dl_img_proc2.webp" style="width: 460px"/>
</div>

+ Performance: Deep learning outperforms traditional methods in many tasks.
+ Learn from data: no need to design features, get rid of the human bias.
+ Generalization: DL models can be applied to different datasets without parameter tuning.

<img src="/statics/img/ml_vs_prog.png" style="width: 600px"/>

---
level: 2
layout: image-right
image: https://raw.githubusercontent.com/Nanguage/slides-deeplearning-microscopy/main/statics/img/DNN.png
---

# Deep learning techniques for Computer Vision(CV)

+ Network architectures
  - Fully connected neural networks (FCNN)
  - Convolutional neural networks (CNN)
  - Vision Transformer (ViT)
+ Generative models
  - Generative adversarial networks (GAN)
  - Variational autoencoders (VAE)
  - Diffusion models
+ Self-supervised learning
  - Masked autoencoder (MAE)

---
level: 2
layout: image-right
image: https://raw.githubusercontent.com/Nanguage/slides-deeplearning-microscopy/main/statics/img/Deep_learning_in_microscopy.jpg
---

# Applications of DL in microscopy

+ Image reconstruction
  - ANNA-PALM
  - CARE (Content-aware image restoration)
  - Self-Net
  - ...
+ Higher level tasks
  - Classification
  - Cell segmentation
  - Cell tracking
  - FISH spot detection
  - ...

---
layout: center
class: text-center
---

# Background about the DL for CV

---
level: 2
layout: two-cols-header
---

# Convolutional neural networks (CNN)

<img src="/statics/img/Typical_cnn.png" style="width: 600px"/>

+ Local receptive fields
+ Parameter sharing

::left::

<img src="/statics/img/imgnet.jpg"/>

::right::

<img src="/statics/img/imgnet-top5.png"/>

---
level: 2
layout: default
---

# Image-to-image model

U-Net architecture

<img src="/statics/img/unet1.png" style="width: 600px"/>

---
level: 2
layout: default
---

# Generative models

<div style="display: flex; gap: 20px">
  <div>
    <img src="/statics/img/gen_models.png" style="height: 300px"/>
    <p>Generative models</p>
  </div>
  <div>
    <div>
      <img src="/statics/img/pix2pix-1.png" style="width: 400px"/>
      <p>Pix2pix based on GAN[1]</p>
    </div>
    <div>
      <img src="/statics/img/diffusion.png" style="width: 400px"/>
      <p>Text to image generation</p>
    </div>
  </div>
</div>

<p style="height: 20px"></p>

\[1\]: Isola, Phillip, et al. "Image-to-image translation with conditional adversarial networks. arXiv e-prints." arXiv preprint arXiv:1611.07004 (2016).

---
level: 2
layout: default
---

# CycleGAN

Unpaired image-to-image translation[^1].

<img src="/statics/img/cyclegan-1.png" style="height: 160px"/>
<img src="/statics/img/cyclegan-2.png" style="height: 150px"/>

[^1]: Zhu, Jun-Yan, et al. "Unpaired image-to-image translation using cycle-consistent adversarial networks." Proceedings of the IEEE international conference on computer vision. 2017.

---
level: 2
layout: default
---

# Vision Transformer (ViT)

<style>
  .footnotes p {
    margin-top: 0;
    margin-bottom: 0;
  }
</style>

Transformers were introduced in 2017,[^1] and have found widespread use in Natural Language Processing. In 2020, they were adapted for computer vision, yielding ViT.[^2]

<div style="display: flex; gap: 20px">
  <div>
    <img src="/statics/img/transformers.png" style="width: 500px"/>
    <p>Transformer in NLP</p>
  </div>
  <div>
    <img src="/statics/img/Vision_Transformer.gif" style="width: 500px"/>
    <p>Vision Transformer</p>
  </div>
</div>

[^1]: Vaswani, Ashish, et al. "Attention is all you need." Advances in neural information processing systems 30 (2017).
[^2]: Dosovitskiy, Alexey, et al. "An image is worth 16x16 words: Transformers for image recognition at scale." arXiv preprint arXiv:2010.11929 (2020).
---
layout: center
class: text-center
---

# Microscopy image reconstruction

---
level: 2
layout: default
---

# ANNA-PALM

The drastic reduction in acquisition time and sample irradiation afforded by ANNA-PALM enables faster and gentler high-throughput and live-cell super-resolution imaging.[^1]

<div style="display: flex; gap: 10px">
<img src="/statics/img/anna-palm-1.webp" style="height: 300px"/>
<img src="/statics/img/anna-palm-2.webp" style="width: 500px"/>
</div>

[^1]: Ouyang, Wei, et al. "Deep learning massively accelerates super-resolution localization microscopy." Nature biotechnology 36.5 (2018): 460-468.

---
level: 2
layout: default
---

# Content-aware image restoration (CARE)

Content-aware image restoration(CARE) based on deep learning extends the range of biological phenomena observable by microscopy.[^1] Software: 
https://github.com/CSBDeep/CSBDeep

<div style="display: flex; gap: 50px; justify-content: center">
  <img src="/statics/img/care-1.webp" style="height: 320px"/>
  <img src="/statics/img/care-2.webp" style="height: 320px"/>
</div>

[^1]: Weigert, Martin, et al. "Content-aware image restoration: pushing the limits of fluorescence microscopy." Nature methods 15.12 (2018): 1090-1097.

---
level: 3
---

# CARE application: Thick tissue MERFISH

Enable three-dimensional (3D) single-cell transcriptome imaging of thick tissue 
specimens by integrating MERFISH with confocal microscopy for optical sectioning and 
deep learning for increasing imaging speed and quality.[^1]

<div style="display: flex; gap: 50px; justify-content: center">
  <img src="/statics/img/care-merfish-1.png" style="height: 300px"/>
  <img src="/statics/img/care-merfish-2.png" style="height: 300px"/>
</div>

[^1]: Fang, Rongxin, et al. "Three-dimensional single-cell transcriptome imaging of thick tissues." bioRxiv (2023): 2023-07.

---
level: 2
layout: default
---


# Self-Net

Self-Net that significantly improves the resolution of axial images by using the lateral images from the same raw dataset as rational targets. By incorporating unsupervised learning for realistic anisotropic degradation and supervised learning for high-fidelity isotropic recovery.[^1]

<div style="display: flex; gap: 50px; justify-content: center">
  <img src="/statics/img/self-net-1.png" style="height: 300px"/>
  <img src="/statics/img/self-net-2.webp" style="height: 300px"/>
</div>

[^1]: Ning, Kefu, et al. "Deep self-learning enables fast, high-fidelity isotropic resolution restoration for volumetric fluorescence microscopy." Light: Science & Applications 12.1 (2023): 204.

---
layout: center
class: text-center
---

# Higher level tasks

---
level: 2
---

# Classification

Human Protein Atlas Image Classification Challenge[^1]


<div style="display: flex; gap: 50px; justify-content: center">
  <img src="/statics/img/hpa-1.webp" style="height: 300px"/>
  <img src="/statics/img/hpa-2.webp" style="height: 300px"/>
</div>

[^1]: Ouyang, Wei, et al. "Analysis of the human protein atlas image classification competition." Nature methods 16.12 (2019): 1254-1261.

---
level: 2
---

# Cell segmentation

<style>
  .footnotes p {
    margin-top: 0;
    margin-bottom: 0;
  }
</style>

Cellpose is a generalist, deep learning-based segmentation method, which can precisely segment cells from a wide range of image types and does not require model retraining or parameter adjustments. [^1] The Multi-modality Cell Segmentation Challenge: Comprising over 1500 labeled images derived from more than 50 diverse biological experiments. [^2]

<div style="display: flex; gap: 50px; justify-content: center">
  <img src="/statics/img/cellpose-1.webp" style="height: 240px"/>
  <img src="/statics/img/cell_seg_comp1.png" style="height: 240px"/>
</div>


[^1]: Stringer, Carsen, et al. "Cellpose: a generalist algorithm for cellular segmentation." Nature methods 18.1 (2021): 100-106.
[^2]: Ma, Jun, et al. "The Multi-modality Cell Segmentation Challenge: Towards Universal Solutions." arXiv preprint arXiv:2308.05864 (2023).

---
level: 2
---

# Cell tracking

LIM Tracker[^1]

<div style="display: flex; gap: 50px; justify-content: center">
  <img src="/statics/img/LIM_tracker-1.webp" style="height: 300px"/>
  <img src="/statics/img/LIM_tracker-2.webp" style="height: 300px"/>
</div>

[^1]: Aragaki, Hideya, et al. "LIM Tracker: a software package for cell tracking and analysis with advanced interactivity." Scientific Reports 12.1 (2022): 2702.

---
level: 2
layout: two-cols-header
---

# FISH spot detection

U-FISH is an advanced FISH spot calling algorithm based on deep learning(Unpublished).
Software: https://github.com/UFISH-Team/U-FISH

1. Diverse dataset: 4000+ images with approximately 1.6 million targets from seven sources.
2. Small network: archiving state-of-the-art performance with only 160k parameters, 680kB ONNX file.
3. 3D support: Support FISH spot detection on 3D images.
4. Support for large-scale data storage formats such as OME-Zarr and N5.
5. User-friendly interface: API, CLI, Napari plugin, ImJoy plugin.

::left::
<img src="/statics/img/ufish.png" style="height: 220px"/>
::right::
<img src="/statics/img/ufish-bench.png" style="height: 240px"/>


---
layout: center
class: text-center
---

# Plantforms for DL application in microscopy

---
level: 2
layout: iframe-right
url: https://imjoy.io/#/app
---

# ImJoy

ImJoy is a plugin powered hybrid computing platform for deploying deep learning applications such as advanced image analysis tools. [^1]
Website: https://imjoy.io/#/

<img src="/statics/img/ImJoy.webp" style="height: 220px"/>

[^1]: Ouyang, Wei, et al. "ImJoy: an open-source computational platform for the deep learning era." Nature methods 16.12 (2019): 1199-1200.

---
level: 2
layout: iframe-right
url: https://bioimage.io/#/
---

# BioImage.IO

BioImage.IO: building AI-powered bioimage analysis model zoo. [^1] Website: https://bioimage.io/#/

<img src="/statics/img/model-zoo.png" style="height: 220px"/>

[^1]: Ouyang, Wei, et al. "Bioimage model zoo: a community-driven resource for accessible deep learning in bioimage analysis." bioRxiv (2022): 2022-06.

---
level: 2
layout: default
---

# ZeroCostDL4Mic

ZeroCostDL4Mic is a collection of self-explanatory Jupyter Notebooks for Google Colab that features an easy-to-use graphical user interface. [^1]  Software: https://github.com/HenriquesLab/ZeroCostDL4Mic

<div style="display: flex; gap: 50px; justify-content: center">
  <img src="/statics/img/ZeroCostDL4Mic-1.webp" style="height: 300px"/>
  <img src="/statics/img/ZeroCostDL4Mic-2.webp" style="height: 300px"/>
</div>

[^1]: von Chamier, Lucas, et al. "Democratising deep learning for microscopy with ZeroCostDL4Mic." Nature communications 12.1 (2021): 2276.

---
layout: center
class: text-center
---

# Perspectives

---
level: 2
layout: default
---

# Large self-supervised learning models

---
level: 2
layout: default
---

# Multi-modality

---
level: 2
layout: default
class: text-center
---

# Launguage UI based on Large Language Models(LLM)

Complex GUI -> A simple dialog

<div style="display: flex; justify-content: center">
  <img src="/statics/img/Software-UI-and-Codex.png" style="width: 650px"/>
</div>

---
level: 3
layout: two-cols
---


# Codex notebook

<iframe width="400" height="315" src="https://www.youtube.com/embed/pkOp_oUybsc?si=PL9BSPu1aDvIpzMB" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

https://aicell.io/project/codex-chat-notebook/

::right::

# Napari-ChatGPT

<video src="https://user-images.githubusercontent.com/1870994/235768559-ca8bfa84-21f5-47b6-b2bd-7fcc07cedd92.mp4" data-canonical-src="https://user-images.githubusercontent.com/1870994/235768559-ca8bfa84-21f5-47b6-b2bd-7fcc07cedd92.mp4" controls="controls" muted="muted" class="d-block rounded-bottom-2 border-top width-fit" style="width:400; width: 315">
</video>

https://github.com/royerlab/napari-chatgpt

---
layout: cover
class: text-center
hideInToc: true
background: https://raw.githubusercontent.com/Nanguage/slides-deeplearning-microscopy/main/statics/img/ai_and_microscopy5.jpg
---

# Thank you for your attention!

Please feel free to contact me if you have any questions.


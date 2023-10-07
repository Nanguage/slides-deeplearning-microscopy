---
theme: seriph
#background: https://source.unsplash.com/collection/94734566/1920x1080
background: statics/img/ai_and_microscopy.jpg
class: text-center
highlighter: shiki
lineNumbers: false
info: |
  ## Slidev Starter Template
  Presentation slides for developers.

  Learn more at [Sli.dev](https://sli.dev)
drawings:
  persist: false
transition: slide-left
title: Deep learning in microscopy
mdc: true
---

# Deep learning in microscopy

Deep Learning (DL) methods are powerful analytical tools for microscopy.

<div class="pt-12">
  <span @click="$slidev.nav.next" class="px-2 py-1 rounded cursor-pointer" hover="bg-white bg-opacity-10">
    Press Space for next page <carbon:arrow-right class="inline"/>
  </span>
</div>

<div class="abs-br m-6 flex gap-2">
  <button @click="$slidev.nav.openInEditor()" title="Open in Editor" class="text-xl slidev-icon-btn opacity-50 !border-none !hover:text-white">
    <carbon:edit />
  </button>
  <a href="https://github.com/slidevjs/slidev" target="_blank" alt="GitHub"
    class="text-xl slidev-icon-btn opacity-50 !border-none !hover:text-white">
    <carbon-logo-github />
  </a>
</div>

<!--
The last comment block of each slide will be treated as slide notes. It will be visible and editable in Presenter Mode along with the slide. [Read more in the docs](https://sli.dev/guide/syntax.html#notes)
-->

---
layout: default
transition: fade-out
---

# Table of contents

<Toc maxDepth="3"></Toc>

---
layout: center
class: text-center
---

# Introduction

---
level: 2
layout: image-right
image: statics/img/DNN.png
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
image: statics/img/Deep_learning_in_microscopy.jpg
---

# Applications of DL in microscopy

+ Image restoration
  - CARE (Content-aware image restoration)
  - Self-Net
  - ...
+ Object detection
  - Cell segmentation
    * Cellpose
    * StarDist
    * ...
  - FISH spot detection
    * DeepBlink
    * U-FISH

---
layout: default
---

# Convolutional neural networks (CNN)

---
level: 2
layout: default
transition: fade-out
---

# Image-to-image models

U-Net architecture

---
layout: center
class: text-center
---

# Plantforms for DL application in microscopy


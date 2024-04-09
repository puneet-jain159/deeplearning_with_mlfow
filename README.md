# 🔥 Build Your Custom AI/LLM With PyTorch Lightning

## Introduction

Processes and information are at the heart of every business. The vulnerability and
opportunity of this moment is the question of whether your business can automate your
processes using AI, and reap the rewards of doing so. ChatGPT, a general purpose AI, has
opened our eyes to what AI can do. What matters now is directing the power of AI to *your*
business problems and unlocking the value of *your* proprietary data. In this document, I 
will
show you how.

## Table of Contents

1. [Overview of LLM Fine Tuning](#1-overview-of-llm-fine-tuning)
2. [Problem Background](#2-problem-background)
3. [Architecture of the AI System](#3-architecture-of-the-ai-system)
4. [Code Deep Dive](#4-code-deep-dive)
5. [Model Results](#5-model-results)
6. [Installation and Quick Start](#6-installation-and-quick-start)
7. [Contribute](#7-contribute)
8. [References](#8-references)

## 1. Overview of LLM Fine Tuning

You don't want an AI that just can chat; what you really want are automations that perform
the work that keeps your business running--powering through business processes at
great accuracy and scale. The proven way to customize AI to your business
processes is to fine tune an LLM on your data and on the action you want AI to perform.

Let's talk specifically about the fine tuning that we are going to do in this document
and the technology behind it. Listed below are the five tools we will use extensively:

- **PyTorch Lightning** (gives us just enough control without excessive boilerplate)
- **Hugging Face** (access to thousands of community-maintained models)
- **Polars** (this data manipulation library is orders of magnitude faster than pandas
  and is
  really trending now)
- **MLflow** (for tracking results during the process of fine tuning)

At the end of the day, you should take away two things from this document:

- How to build deep learning models that attain state-of-the-art results
- How the experience of building such models can be pretty easy when your mentality is
  to build small, modular, and reusable components (something that PyTorch Lightning
  facilitates)


## 1. Problem Background

A good process for finding suitable problems for machine learning and for quality datasets
is to start by browsing sites with benchmarks. Benchmarks provide a frame of reference
for the level of difficulty of the problem for machine learning, which we use to measure
our progress during model development. One particular dataset with well established
benchmarks is
the [Unfair Terms of Service dataset (UNFAIR-ToS)](https://huggingface.co/datasets/lex_glue/viewer/unfair_tos);
here's an intriguing
problem statement for it: Use AI to find all unfair clauses in Terms of Service
contracts. The context is that the European consumer law on unfair contracts establishes
what unfair clauses are and the different types of unfair clauses. What makes UNFAIR-ToS
perfect for text classification is that it has been manually labeled in accordance with
that which was set down in the European law.

Chalkidis et al. (2021) applied eight different machine learning methods to UNFAIR-ToS
and obtained macro F1 ranging from 75 to 83, and in Figure 1 just below, we excerpt from
their published results.

#### Figure 1

| Method       | UNFAIR-ToS |          |
|--------------|------------|----------|
|              | micro F1   | macro F1 |
| TFIDF-SVM    | 94.7       | 75.0     |
| BERT         | 95.6       | 81.3     |
| RoBERTa      | 95.2       | 79.2     |
| DeBERTa      | 95.5       | 80.3     |
| Longformer   | 95.5       | 80.9     |
| BigBird      | 95.7       | 81.3     |
| Legal-BERT   | 96.0       | 83.0     |
| CaseLaw-BERT | 96.0       | 82.3     |

Interesting things we can infer from this table are:

1. F1, not accuracy, is the authors' preferred metric
2. macro F1 is a better differentiator than micro F1
3. all the deep learning models exhibit similar performance while SVM is materially worse

Looking at the data, class imbalance is certainly present, which is a good reason for the
first and second point above.

There are eight different types of unfair clauses. The authors of that paper developed
multi-label classification models for the eight types, but we are simply going to
build a binary classification model that classifies a clause as fair or unfair.

Let's design how we're going to do it.

## 3. Architecture of the AI System
#Todo

#### Figure 2

#Todo

### 🚶 A Tour of the Project

#Todo

## 5. Model Results

#Todo

#### Figure 10

#Todo

## 6. Installation and Quick Start

#Todo



Ilias Chalkidis, Abhik Jana, Dirk Hartung, Michael Bommarito, Ion Androutsopoulos,
Daniel Martin Katz, Nikolaos Aletras. (2021). *LexGLUE: A Benchmark Dataset for Legal
Language Understanding in English*. Retrieved from arXiv: https://arxiv.org/abs/2110.00976

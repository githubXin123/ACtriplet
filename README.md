# ACtriplet
ACtriplet: an improved deep learning model for activity cliffs prediction by integrating triplet loss and pre-training

## Requirements
  * python = 3.8
  * pytorch = 2.0
  * torch ge ometric
  * rdkit
  * numpy
  * pandas
  * prefetch-generator
  * pyyaml
  * moleculeACE

## Pre-train
### Step 1: Prepare dataset
Extract Ac and non-Ac from the 30 macromolecular targets dataset  
`data pre/get collection-1.py`
### Step 2: Pre-train
`python pretrain/main pretrain.py`

## Finetune
`python finetune/main finetune.py`

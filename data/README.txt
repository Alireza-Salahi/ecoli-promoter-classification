Bioinformatics Assignment 1 - Promoter Sequence Classification
Dataset README
==============================================================

FILES
-----
task1_promoter.csv   Task 1 and Task 3 (promoter vs. non-promoter)
task2_sigma.csv      Task 2 (sigma-factor classification)
README.txt           this file

Both files are plain CSV with a header row.

task1_promoter.csv columns:
  id        unique sample id (T1_00001, ...)
  sequence  DNA sequence, 81 characters, uppercase A/C/G/T only
  label     1 = promoter, 0 = non-promoter
  split     train / test

task2_sigma.csv columns:
  id        unique sample id (T2_00001, ...)
  sequence  promoter sequence, 81 characters, uppercase A/C/G/T only
  sigma     sigma24, sigma28, sigma32, sigma38, sigma54 or sigma70
  split     train / test


SEQUENCE COORDINATES AND THE TSS
--------------------------------
Every sequence is 81 bp long and aligned on the Transcription Start
Site (TSS). For non-promoters, the same coordinates are used but
there is no real TSS.

  Index (1-based)   Index (0-based, Python)   Position relative to TSS
  1 ... 60          0 ... 59                  -60 ... -1  (upstream)
  61                60                        +1          (the TSS)
  62 ... 81         61 ... 80                 +2 ... +20  (downstream)

There is no position 0 in this numbering. To convert a 0-based Python
index i to the position relative to the TSS:

  pos = i - 60   if i < 60
  pos = i - 59   if i >= 60

Expected locations of the sigma70 motifs:
  -35 box (TTGACA): about indices 25-30 (0-based)
  -10 box (TATAAT): about indices 48-53 (0-based)
These are approximate; real promoters vary by a few bases.


CLASS COUNTS
------------
Task 1                 train    test   total
  promoter (1)          2287     572    2859
  non-promoter (0)      2288     572    2860
  total                 4575    1144    5719

Task 2                 train    test   total
  sigma70               1354     339    1693
  sigma24                387      97     484
  sigma32                233      58     291
  sigma38                130      33     163
  sigma28                107      27     134
  sigma54                 75      18      93
  total                 2286     572    2858


SPLIT RULES
-----------
- The split is a fixed 80% / 20% split, stratified by class
  (random seed 42). Use it exactly as provided.
- Do all model selection (k, features, hyperparameters, epochs)
  on the training rows only. Use the test rows once, for the final
  results.
- The two files share their promoters: every promoter in
  task2_sigma.csv also appears in task1_promoter.csv, with the same
  split. A promoter that is in the Task 2 test set is also in the
  Task 1 test set.


SOURCE AND PREPROCESSING
------------------------
Source: the iPromoter-2L benchmark dataset of Escherichia coli K-12
promoters, built from RegulonDB.
  Liu B, Yang F, Huang DS, Chou KC. iPromoter-2L: a two-layer
  predictor for identifying promoters and their types by
  multi-window-based PseKNC. Bioinformatics. 2018;34(1):33-40.
Files obtained from: https://github.com/Shujaatmalik/pcPromoter-CNN

Preprocessing done by the TAs:
1. All sequences were converted to uppercase. (In the original
   files, promoters are lowercase with only the TSS in uppercase,
   while non-promoters are fully uppercase. Leaving this would let
   a model separate the classes by letter case alone.)
2. One promoter appeared twice in the positive set and was kept
   once in Task 1.
3. The same promoter is annotated with two sigma factors (sigma54
   and sigma70), so it was removed from Task 2.
4. Rows were shuffled and the stratified train/test split was added.


LOADING EXAMPLE (Python)
------------------------
import pandas as pd

df = pd.read_csv("task1_promoter.csv")
train = df[df.split == "train"]
test  = df[df.split == "test"]
X_train, y_train = train.sequence.tolist(), train.label.values
X_test,  y_test  = test.sequence.tolist(),  test.label.values

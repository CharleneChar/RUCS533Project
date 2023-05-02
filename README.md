# CS533 NLP Final Project: Reimplementation - Inducing Positive Perspectives with Text Reframing

We have reimplemented the paper [***Inducing Positive Perspectives with Text Reframing***](https://aclanthology.org/2022.acl-long.257/) which introduces the task of *positive reframing*, in which we aim to transform a negative sentence such that we introduce a positive perspective without transforming the underlying meaning. The paper also introduced a large-scale benchmark [Positive Psychology Frames](https://github.com/SALT-NLP/positive-frames), which contains 8,349 sentence pairs as a parallel corpus, which will be the basis to test the performance of the current state-of-the-art text style transfer models

## Steps to Run

### 1. Install the requirements for the project

```
pip install -r requirements.txt
```

### 2. Run the models
|   Flags  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | Definition | Values Accepted| Default Values|
|---------------|----------|----------------|-----------------|
| --train | Path to the training set | Any valid path | (For GPT): data/wholetrain_gpt.txt <br/>(Everything Else): data/wholetrain.csv|
| --dev | Path to the development set | Any Valid Path | data/wholedev.csv
| --test | Path to the test set | Any Valid Path | data/wholetest.csv|
| --output_dir | Path to the output directory| Any Valid Path | output/|
| -s, <br/> --setting | Define the setting for training (Only used in BART and T5)| 'unconstrained', 'controlled', 'predict' | unconstrained|

```
python3 <model_name>.py --arguments
```
The following are the list of files to run for a given model

    1. Random Retrieval - random.py
    2. SBERT Retrieval - sbert.py
    3. GPT - gpt.py
    4. BART - bart.py
    5. T5 - t5.py



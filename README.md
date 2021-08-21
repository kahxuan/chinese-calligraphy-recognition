# Chinese Calligraphy Character Recognition with Intra-class Variant Clustering

The goal of this project is to recognise the 100 most commonly used chinese characters, written in calligraphy in the semi-cursive script. As compared to handwritten chinese characters, chinese calligraphy characters are harder to recognise, even for trained experts because they can be written in a less restrictive way to express the author’s personality and emotions.

Read more [here](https://kahxuan.github.io/html/projects/cccr.html).

Sample data from character "总":

<img alt="sample data" src="https://github.com/kahxuan/chinese-calligraphy-recognition/blob/main/images/sample_data.png" height="150" />

## Requirements

[lear-gist-python](https://github.com/whitphx/lear-gist-python) is used to extract GIST features. The installation instruction can be found in the repo.

The other depedencies can be installed by running:
```
$ pip install -r requirements.txt
```

## Notes

* `notebooks/*` - ipynb files to collect and preprocess the data.
* `train.ipynb` - Demo to train the model. The data directories and model hyperparameters have to be specified in config.yaml.
* `data/cccr/*` - Train, validation and test set.
* `data/common.txt` - The most common chinese characters arranged in descending order, ie. the first 100 lines contain the 100 most commonly used characters.

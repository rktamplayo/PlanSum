# PlanSum
[AAAI2021] Unsupervised Opinion Summarization with Content Planning

This PyTorch code was used in the experiments of the research paper

Reinald Kim Amplayo, Stefanos Angelidis, and Mirella Lapata.
[**Unsupervised Opinion Summarization with Content Planning**](https://rktamplayo.github.io/publications/aaai21.pdf). _AAAI_, 2021.

The code is cleaned post-acceptance and may run into some errors. Although I did some quick check and saw the code ran fine, please create an issue if you encounter errors and I will try to fix them as soon as possible.

## Data

We used three different datasets from three different papers: Amazon (Brazinskas et al., 2020), Rotten Tomatoes (Wang and Ling, 2016), and Yelp (Chu and Liu, 2019). For convenience, we provide the train/dev/test datasets [here](https://drive.google.com/drive/folders/1pLBmzFIUquBQ0lU_0U0swOn-OStbY4Jr?usp=sharing) which are preprocessed accordingly and saved in three separate json files. A file contains a list of instances, where one instance is formatted as follows:

```json
{
    "reviews": [
       ["this is the first review.", 5],
       ["this is the second review.", 3],
       "..."
    ],
    "summary": "this is the first summary.",
    "..."
}
```

In the example above, `reviews` is a list of review-rating tuples. For Amazon dev/test files, the `summary` is instead a list of reference summaries. There can be other information included in the files but are not used in the code (e.g., `category` and `prod_id` in the Amazon datasets). When using the datasets, please also cite the corresponding papers (listed below).

## Running the code

PlanSum follows a [Condense-Abstract Framework](https://arxiv.org/pdf/1909.02322.pdf) (Amplayo and Lapata, 2019) where we first condense the reviews into encodings and then use the encodings as input to a summarization model. In PlanSum, the content plan induction model is the Condense model, while the opinion summarization model is the Abstract model. Below, we show a step-by-step procedure on how to run and generate summaries using PlanSum on the Yelp dataset.

### Step 0: Download the datasets

Download the preprocessed datasets [here](https://drive.google.com/drive/folders/1pLBmzFIUquBQ0lU_0U0swOn-OStbY4Jr?usp=sharing). You can also skip steps by downloading the model files in the `model/` directory and the `train.plan.json` files in the `data/` directory.

### Step 1: Train the content plan induction model

This can be done by simply running `src/train_condense.py` in the form:

`python src/train_condense.py -mode=train -data_type=yelp`

This will create a `model/` directory and a model file named `condense.model`.
There are multiple arguments that need to be set, but the default setting is fine for Yelp. The settings used for Amazon and Rotten Tomatoes are commented in the code.

### Step 2: Create the synthetic training dataset

PlanSum uses a synthetic data creation method where we sample reviews from the corpus and transform them into review-summary pairs. To do this, we use the same code `-mode=create`, i.e.

`python src/train_condense.py -mode=create -data_type=yelp`

This will create a new json file named `train.plan.json` in the `data/yelp/` directory. This is the synthetic training dataset used to train the summarization model.

### Step 3: Train the summarization model

This is done by simply running `src/train_abstract.py`:

`python src/train_abstract.py -mode=train -data_type=yelp`

This will create a model file named `abstract.model` in the `model/` directory.
There are also arguments here that need to be set, but the default setting is fine for Yelp. Settings used for other datasets are commented in the code.

### Step 4: Generate the summaries

Generating the summaries can be done by running:

`python src/train_abstract.py -mode=eval -data_type=yelp`

This will create an `output/` directory and a file containing the summaries named `predictions.txt`.

## I just want your summaries!

This repo also include an `output/` directory which includes the generated summaries from five different systems:

- `gold.sol` contains the gold-standard summaries
- `plansum.sol` contains summaries produced by PlanSum (this paper)
- `denoisesum.sol` contains summaries produced by [DenoiseSum](https://www.aclweb.org/anthology/2020.acl-main.175.pdf) (Amplayo and Lapata, 2020)
- `copycat.sol` contains summaries produced by [CopyCat](https://www.aclweb.org/anthology/2020.acl-main.461.pdf) (Brazinskas et al., 2020)
- `bertcent.sol` contains summaries produced by BertCent (this paper)

Please do rightfully cite the corresponding papers when using these outputs (e.g., by comparing them with your model's).

## Cite the necessary papers

To cite the paper/code/data splits, please use this BibTeX:

```
@inproceedings{amplayo2021unsupervised,
	Author = {Amplayo, Reinald Kim and Angelidis, Stefanos and Lapata, Mirella},
	Booktitle = {AAAI},
	Year = {2021},
	Title = {Unsupervised Opinion Summarization with Content Planning},
}
```

If using the datasets, please also cite the original authors of the datasets:

```
@inproceedings{bravzinskas2020unsupervised,
	Author = {Bra{\v{z}}inskas, Arthur and Lapata, Mirella and Titov, Ivan},
	Booktitle = {ACL},
	Year = {2020},
	Title = {Unsupervised Multi-Document Opinion Summarization as Copycat-Review Generation},
}
```

```
@inproceedings{wang2016neural,
	Author = {Wang, Lu and Ling, Wang},
	Booktitle = {NAACL},
	Year = {2016},
	Title = {Neural Network-Based Abstract Generation for Opinions and Arguments},
}
```

```
@inproceedings{chu2019meansum,
	Author = {Chu, Eric and Liu, Peter},
	Booktitle = {ICML},
	Year = {2019},
	Title = {{M}ean{S}um: A Neural Model for Unsupervised Multi-Document Abstractive Summarization},
}
```

If there are any questions, please send me an email: reinald.kim at ed dot ac dot uk

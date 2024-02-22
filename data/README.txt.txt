This folder contains data files for assessment item 1 of CMP9137M Advanced Machine Learning 2023-24.

The following references are pointers to the proposed sources of the data:
M. Hodosh, et.al., Framing Image Description as a Ranking Task: Data, Models and Evaluation Metrics. Journal of Artificial Intelligence Research 47, 2013.
P. Young, et. al., From Image Descriptions to Visual Denotations: New Similarity Metrics for Semantic Inference Over Event Descriptions. Transactions of the Association for Computational Linguistics, 2, 2014.

The former reference (Hodosh et al., 2013) was chosen for this assignment due to a more managable size of data containing 40460 image-caption pairs in total. The latter reference contains 150K image-caption pairs, which require much more intensive compute requirements than its counterpart.

Subsets were sampled from the chosen dataset resulting in the following sizes of image-sentence-label triples:
train=19386
dev=1164
test=1161

Whilst the original data only has sensical image-caption pairs, the newly prepared dataset generated image-caption-labels including both sensical and random captions and their corresponding labels. In this way, the goal of a classifier using the proposed data is to detect whether a given image-caption pair is a true match or a no-match. The latter would mean that the classifier would have learnt to identify non-sense captions, i.e., to distinguish those that make sense versus those that do not.

The original image data was resised to a resolution of 224x224x3 due to disk space and compute conveniences, which can be found in the following folder: 
flickr8k.dataset-cmp9137-item1\flickr8k-resised\*.jpg

The generated triples can be found in the following files:
flickr8k.dataset-cmp9137-item1\flickr8k.TrainImages.txt
flickr8k.dataset-cmp9137-item1\flickr8k.DevImages.txt
flickr8k.dataset-cmp9137-item1\flickr8k.TestImages.txt

The last file in this folder (flickr8k.cmp9137.sentence_transformers.pkl) contains sentence features that represent all sentences in the dataset in a numerical representation, which can be fed as input to neural classifiers. The use of this file is optional, only to be used instead (or as an alternative) of the original word-based representations. The contents of this file can be displayed with the following command:
python -m pickle flickr8k.cmp9137.sentence_transformers.pkl

Last update 2 February 2024

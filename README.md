# Visual Language Processing
Trying to figure out if AI can understand natural language from raw pixels.

For a detailed report on the project, please refer to this [blog](https://medium.com/@basarafilip/visual-language-processing-e2496ce67f94)! 

Note:
* Relevant file for MLM training dataset - [text_recog_masked.py](https://github.com/filipbasara0/visual-language-processing/blob/main/dataset/text_recog_masked.py)
* Relevant file for MLM training loop - [train_text_rec_masked.py](https://github.com/filipbasara0/visual-language-processing/blob/main/training/train_text_rec_masked.py)

## Intro
The goal of this project is to train a model to recognize and understand tokens from an image.

The VLP model is trained to output text from an image token by token, with some tokens being masked and the model having to predict them.

A hybrid CNN-transformer model (similar to DETR) is used for this task:

![Untitled Diagram drawio (17)](https://github.com/filipbasara0/visual-language-processing/assets/29043871/e65f1b86-3696-4a2d-8e6f-65ee3d23cbf6)

For larger models, the transformer encoder-decoder is asymetric, meaning that the decoder has a lot less parameters compared to the encoder. This is done in purpose, to make the encoder store as much knowledge about the language as possible. The asymetric structure reflected well on MNLI results.

Inference is done in autoregressive fashion, using greedy decoding.

The model failed to converge without the CNN layer, but increasing the size of the CNN didn't yield improvements. There could still be more improvements in the CNN part, since I conducted this experiment a long time ago. More investigation is required to determine exactly why the model fails without the CNN.

The repository is still work in progress, more info, improvements and revisited code structure to come.

## Results
The `encoder_decoder_lg` version of the model achieves 0.73 F1 on MNLI.

```
Accuracy Score = 0.728533550887212
F1 Score (Micro) = 0.728533550887212
F1 Score (Macro) = 0.7283811357776412
              precision    recall  f1-score   support

           0       0.80      0.73      0.76      3477
           1       0.66      0.72      0.69      3119
           2       0.73      0.73      0.73      3210

    accuracy                           0.73      9806
   macro avg       0.73      0.73      0.73      9806
weighted avg       0.73      0.73      0.73      9806
```

The model has a decent understanding of language, but still falls short on more complex examples.

To test the embedding quality, I did linear probing on `imdb` sentiment classification dataset and achieved `70` F1 score, which is 10 points less compared to using standard BERT embeddings.

## Training

Coming soon

## Data

Training data was sampled from the Wikipedia and Bookcorpus datasets.

Texts were filtered to have less than 144 tokens.


## WIP

The text recognition part works very well.

The model knows when to use verbs, punctuation, co-reference, etc.

Mistakes often occur in examples that require memorization, such as `The capital of France is [MASK].`

Currently training with a larger model to see how it would affect the memorization and the performance on MNLI datasets.

The VLP model was also trained on SQUAD and Ontonotes and showed promising results. More experiments will be conducted in the future.


## Setup

1. `git clone git@github.com:filipbasara0/simple-object-detection.git`
2. `create virtual environment: virtualenv -p python3.8 env`
3. `activate virtual environment: source env/bin/activate`
4. `install requirements: pip install -r requirements.txt`

## Usage

To train the model for text recognition, use the following command:

```
python train.py --model_name=encoder_decoder_lg --dataset_name=wiki_text_recognition --dataset_path="data/images_masked_combined/" --out_model_path="best_model_wiki_enc.pth" --learning_rate=1e-4 --num_epochs=1 --task_name=text_recognition_masked --batch_size=16 --image_size=512 --max_text_len=144
```

## License

[MIT](https://choosealicense.com/licenses/mit/)

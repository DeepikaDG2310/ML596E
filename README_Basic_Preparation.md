### Chile Rescue - Review dangerousness: Classify whether the review conveys that an app is dangerous to use

To determine whether an app is safe or not using binary classification based on app reviews, you can use pretrained models from the Hugging Face Transformers library. As the pretrained models like BERT, RoBERTa, or DistilBERT are well-suited for text classification tasks. These models can be fine-tuned on labeled app review datasets, where the reviews are categorized as indicating whether the app is safe or unsafe. The input dataset would contains reviews about the App (either through google app store or ios app store)

The process involves loading a pretrained model (e.g., BERT) and its tokenizer, tokenizing the reviews, and then fine-tuning the model on your dataset using libraries like PyTorch. After fine-tuning, you can use the model to classify new reviews. Libraries like pandas for data handling would be essential. After going through few possible topics, this seems to be straightforward and most effective as it uses pretrained model.

### Steps:

Install Pandas, Numpy, Sklearn, argparse, Torch, Flask, transformers 

`pip install pandas numpy scikit-learn argparse torch flask transformers`

Save the parameters in `config.py`

`# Set up parameters`

`bert_model_name = 'bert-base-uncased'`

`num_classes = 2`

`max_length = 128`

`batch_size = 16`

`num_epochs = 4`

`learning_rate = 2e-5`

`tokenizer = BertTokenizer.from_pretrained(bert_model_name)`

`model_path = "bert_classifier.pth"`

Save and Run the `util.py' to prepare for helper functions

`python util.py`

Save and Run the `App_dangerrousness.py` to prepare the model and fine-tune the model

`python App_dangerrousness.py`

Save and Run the `CLI_Train_Run.py`. This will run the code to train and fine - tuned the pre trained BERT model. Save the model

`python CLI_Train_Run.py --data_file  "App_review_Class.csv" --eval_flag "Y"'

#### Evaluation using Validate Dataset

![image](https://github.com/user-attachments/assets/1c9bcb85-60da-4c14-94cf-d3f9a2be6e74)


###This will prepare the model and save it. So that it can be loaded into different module to classify the Review




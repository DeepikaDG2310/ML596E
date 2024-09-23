## Review dangerousness: Classify whether the review conveys that an app is dangerous to use

To determine whether an app is safe or not using binary classification based on app reviews, you can use pretrained models from the Hugging Face Transformers library. As the pretrained models like BERT, RoBERTa, or DistilBERT are well-suited for text classification tasks. These models can be fine-tuned on labeled app review datasets, where the reviews are categorized as indicating whether the app is safe or unsafe. The input dataset would contains reviews about the App (either through google app store or ios app store)

Please refer the Basic_Preparation, CLI and Flask Readme files

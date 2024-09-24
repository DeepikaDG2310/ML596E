## Review dangerousness: Classify whether the review conveys that an app is dangerous to use

To prepare the configuration for classification model is given in the README_Basic_Prepare.md
Once you have prepared the dataset and set up the environment, you can train the classification model using the Command Line Interface (CLI). This process involves feeding the app review data into the model, which will be fine-tuned to classify whether a review indicates the app is safe or dangerous.

## To Run the model using CLI:

Save and run the model using  CLI with input file 

`python Review_class_prediction_CLI.py --data_file "input_file_path" --output_file "output_path_file" `

Arguments:

1. --data_file:

This specifies the path to the CSV file containing your training data. The CSV should contain a list of app reviews and corresponding labels indicating whether the app is dangerous

2. --output_file:

This specifiess the path for output csv file

### Model Output
Once the model is trained, it will save the classifier for later use in classifying app reviews as safe or dangerous. The output file will contain the original reviews along with a classification label Safe and Not-Safe

![image](https://github.com/user-attachments/assets/75ea193a-eb8e-4369-a058-0ac2e97beca3)


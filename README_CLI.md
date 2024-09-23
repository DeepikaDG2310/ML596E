## Review dangerousness: Classify whether the review conveys that an app is dangerous to use

To prepare the configuration for classification model is given in the README_Basic_Prepare.md


## To Run the model using CLI:

Save and run the model using  CLI with input file 

`python Review_class_prediction.py --data_file "input_file_path" --output_file "output_path_file" `

### Model Output
Once the model is trained, it will save the classifier for later use in classifying app reviews as safe or dangerous. The output file will contain the original reviews along with a classification label Safe and Not-Safe


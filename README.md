# Churn Prediction With Text

Customer churn is a problem faced by a wide range of companies, from telecommunications to banking, where customers are typically lost to competitors. It's in a company's best interest to retain existing customer instead of acquiring new customers because it usually costs significantly more to attract new customers. When trying to retain customers, companies often focus their efforts on customers who are more likely to leave. User behaviour and customer support chat logs can contain value indicators on the likelihood of a customer leaving the service.

In this solution, we train and deploy a churn prediction model on Amazon SageMaker that uses state-of-the-art natural language processing model to find useful signals in text. In addition to textual inputs, this model uses traditional structured data inputs such as numerical and categorical fields.

## Getting Started

To get started quickly, use the following quick-launch link to launch a CloudFormation Stack create form and follow the instructions below to deploy the resources in this project.

| Region | Stack |
| ---- | ---- |
|US East (N. Virginia) | [<img src="https://s3.amazonaws.com/cloudformation-examples/cloudformation-launch-stack.png">](https://us-east-1.console.aws.amazon.com/cloudformation/home?region=us-east-1#/stacks/create/review?templateURL=https://sagemaker-solutions-prod-us-east-1.s3.us-east-1.amazonaws.com/Churn-prediction-with-text/cloudformation/template.yaml&stackName=sagemaker-soln-churn) |
|US East (Ohio) | [<img src="https://s3.amazonaws.com/cloudformation-examples/cloudformation-launch-stack.png">](https://us-east-2.console.aws.amazon.com/cloudformation/home?region=us-east-2#/stacks/create/review?templateURL=https://sagemaker-solutions-prod-us-east-2.s3.us-east-2.amazonaws.com/Churn-prediction-with-text/cloudformation/template.yaml&stackName=sagemaker-soln-churn) |
|US West (Oregon) | [<img src="https://s3.amazonaws.com/cloudformation-examples/cloudformation-launch-stack.png">](https://us-west-2.console.aws.amazon.com/cloudformation/home?region=us-west-2#/stacks/create/review?templateURL=https://sagemaker-solutions-prod-us-west-2.s3.us-west-2.amazonaws.com/Churn-prediction-with-text/cloudformation/template.yaml&stackName=sagemaker-soln-churn) |


### Additional Instructions

* On the stack creation page, check the box to acknowledge creation of IAM resources, and click **Create Stack**. This should trigger the creation of the CloudFormation stack.

* Once the stack is created, go to the Outputs tab and click on the *SageMakerNotebook* link. This will open up a Jupyter notebook in a SageMaker Notebook instance where you can run the code. Follow the instructions in the notebook to run the solution. You can use `Cells->Run All` from the Jupyter menu to run all cells, and return to the notebook later after all cells have executed. The total time to run all cells should be around 20 minutes.

## Architecture

We focus on Amazon SageMaker components in this solution. Amazon SageMaker Training Jobs are used to train the churn prediction model and an Amazon SageMaker Endpoint is used to deploy the model. We use Amazon S3 alongside Amazon SageMaker to store the training data and model artifacts, and Amazon CloudWatch to log training and endpoint outputs.

![architecture-diagram](https://sagemaker-solutions-prod-us-west-2.s3.us-west-2.amazonaws.com/Churn-prediction-with-text/docs/architecture_diagram_light.png)

## Credits

* Packages
  * [Scikit-learn](https://scikit-learn.org/stable/)
  * [sentence-transformers](https://github.com/UKPLab/sentence-transformers)
  * [Hugging Face Transformers](https://huggingface.co/)
* Datasets
  * [KDD Cup 2009](https://www.kdd.org/kdd-cup/view/kdd-cup-2009) (use to create synthetic dataset)
* Models
  * GPT2
    * Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever
  * BERT
    * Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova
  * Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks
    * Reimers, Nils and Gurevych, Iryna

## License

This project is licensed under the Apache-2.0 License.

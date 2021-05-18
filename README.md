# Churn Prediction With Text

Customer churn is a problem faced by a wide range of companies, from telecommunications to banking, where customers are typically lost to competitors. It's in a company's best interest to retain existing customer instead of acquiring new customers because it usually costs significantly more to attract new customers. When trying to retain customers, companies often focus their efforts on customers who are more likely to leave. User behaviour and customer support chat logs can contain value indicators on the likelihood of a customer leaving the service.

In this solution, we train and deploy a churn prediction model on Amazon SageMaker that uses state-of-the-art natural language processing model to find useful signals in text. In addition to textual inputs, this model uses traditional structured data inputs such as numerical and categorical fields.

## Getting Started

To run this JumpStart 1P Solution and have the infrastructure deploy to your AWS account you will need to create an active SageMaker Studio instance (see Onboard to Amazon SageMaker Studio). When your Studio instance is Ready, use the instructions in SageMaker JumpStart to 1-Click Launch the solution.

The solution artifacts are included in this GitHub repository for reference.

*Note*: Solutions are available in most regions including us-west-2, and us-east-1.

**Caution**: Cloning this GitHub repository and running the code manually could lead to unexpected issues! Use the AWS CloudFormation template. You'll get an Amazon SageMaker Notebook instance that's been correctly setup and configured to access the other resources in the solution.

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

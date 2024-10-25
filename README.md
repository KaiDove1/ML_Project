# ML_Project

Workflow
1. Data Collection
Gather regional and satellite data on agricultural practices and runoff impact from the data sources listed above.
Organize data in the data/raw/ directory, ensuring consistency in formats and resolutions.
2. Data Processing
Image Processing: Pre-process satellite or field images to highlight plant health indicators. Consider using vegetation indices or color segmentation.
Feature Engineering: Extract features relevant to plant health and create labels based on the presence of healthy vs. unhealthy vegetation.
Use src/data_preprocessing.py to convert raw data into a model-friendly format and save processed data in data/processed/.
3. Model Training
Implement and train a simple convolutional neural network (CNN) to detect plant health based on processed image data.
Define a binary classification task: “Needs Treatment” vs. “No Treatment Needed.”
Use src/model_training.py to build, train, and save the model.
4. Model Evaluation
Evaluate model performance using metrics like accuracy, precision, and recall to assess its effectiveness in detecting plant health needs.
Run src/model_evaluation.py to generate and store evaluation results.
Future Directions
Integrate with IoT sensors on agricultural sprayers for real-time detection and automated pesticide/fertilizer application control.
Experiment with more complex models or custom datasets for specific crop types or regions.

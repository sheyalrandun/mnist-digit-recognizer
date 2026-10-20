# MNIST Hand Drawing Recognizer

***A custom CNN built from scratch in PyTorch to classify handwritten digits, paired with an interactive desktop drawing app.***

## KEY METRICS AND RESULTS

- Being trained for 10 epochs while having a **97.08%** training accuracy, a **97.23%** validation accuracy, and a **97.44%** test accuracy on unseen data, this model has achieved strong generalization performance.
- The model uses Stochastic Gradient Descent (**SGD**) as the optimization method with a **0.01** learning rate to train on the MNIST image dataset.

## TOOLS

- **Language:** Python
- **Libraries & Framework:** PyTorch (`torch`, `torchvision`), Tkinter (GUI), Pillow (`PIL` image processing)

## STRUCTURE

```text
mnist-digit-recognizer/
│
├── Models/
│   └── mnist_model_10epoch.pth        # Trained model state dictionary
├── Project1-(MNIST)-CODE.py          # CNN architecture definition
├── PROJECT1-(MNIST)-Evaluate.py      # Script to evaluate model on MNIST test split
├── PROJECT1-(MNIST)-GUI.py           # Interactive Tkinter drawing & inference app
├── Requirements.txt                  # Required Python packages
└── README.md                         # Project documentation

SETUP & INSTRUCTIONS

*Clone or download this repository to your local machine.

**Install the required dependencies by running the following command in your terminal:
	pip install -r Requirements.txt

**Evaluate the model on the unseen MNIST test dataset:
	python "PROJECT1-(MNIST)-Evaluate.py"

**Launch the interactive drawing application:
	python "PROJECT1-(MNIST)-GUI.py"

(Draw any digit from 0–9 clearly on the white canvas with your mouse, click "Predict", and test the live inference!)

ENGINEERING & OBSERVATION

* At first i used learning rate as 0.001 and 5 epochs to train the model and calculated the models' training accuracy, validation accuracy, training loss and
validation loss.
* I got somewhere around 85%for both training and validation accuracy in the last epoch.
* Then i changed the learning rate in to 0.01 and trained the model for 10 epochs to get the best out of it and i ended up with 97% for both training and
validation accuracies.

GRAPHICAL USER INTERFACE

* While using the GUI platform, we need to draw the relevant numbers using a clear set of lines. If not the models' prediction might not be well accurate.

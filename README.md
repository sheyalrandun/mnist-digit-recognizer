# MNIST Hand Drawing Recognizer

**A custom CNN built from scratch in PyTorch to classify handwritten digits, paired with an interactive desktop drawing app.**

## KEY METRICS AND RESULTS

- Trained for 10 epochs, achieving **97.08% training accuracy**, **97.23% validation accuracy**, and **97.44% test accuracy** on unseen data, demonstrating strong generalization performance.
- Uses Stochastic Gradient Descent (**SGD**) with a learning rate of **0.01** to train on the MNIST image dataset.

## TOOLS

- **Language:** Python
- **Libraries & Frameworks:** PyTorch (`torch`, `torchvision`), Tkinter (GUI), Pillow (`PIL` image processing)

## STRUCTURE

```text
mnist-digit-recognizer/
│
├── Models/
│   └── mnist_model_10epoch.pth    # Trained model state dictionary
├── Project1-(MNIST)-CODE.py       # CNN architecture definition
├── PROJECT1-(MNIST)-Evaluate.py   # Script to evaluate model on MNIST test split
├── PROJECT1-(MNIST)-GUI.py        # Interactive Tkinter drawing and inference app
├── Requirements.txt               # Required Python packages
└── README.md                      # Project documentation
```

## SETUP & INSTRUCTIONS

### 1. Clone or Download the Repository

Clone this repository or download it to your local machine.

### 2. Install Dependencies

Open a terminal in the project directory and run:

```bash
pip install -r Requirements.txt
```

### 3. Evaluate the Model

Evaluate the model on the unseen MNIST test dataset:

```bash
python "PROJECT1-(MNIST)-Evaluate.py"
```

### 4. Launch the Drawing Application

Run the following command to launch the interactive GUI:

```bash
python "PROJECT1-(MNIST)-GUI.py"
```

Draw any digit from **0–9** clearly on the white canvas using your mouse, click **Predict**, and test the model's live inference!

## ENGINEERING & OBSERVATIONS

Initially, I used a learning rate of `0.001` and trained the model for 5 epochs while tracking training accuracy, validation accuracy, training loss, and validation loss.

I achieved approximately **85% training and validation accuracy** by the final epoch.

I then increased the learning rate to `0.01` and trained the model for 10 epochs. This improved performance to over **97% across the training, validation, and test splits**, without severe overfitting.

## GRAPHICAL USER INTERFACE

When using the GUI, draw each digit with clear, continuous strokes on the white canvas. Unclear or unusually drawn digits may lead to incorrect predictions.


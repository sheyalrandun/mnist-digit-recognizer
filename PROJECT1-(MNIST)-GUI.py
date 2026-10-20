import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import torch
import torch.nn as nn
from torchvision import transforms

# 1. Define the CNN architecture (must match your trained model)
class Project(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.classifier = nn.Linear(784, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

class DigitApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MNIST Handwritten Digit Recognizer")
        
        # Load the trained model weights
        self.model = Project()
        self.model.load_state_dict(torch.load("mnist_model_10epoch.pth", map_location=torch.device('cpu')))
        self.model.eval() # Set to evaluation mode
        
        # Canvas dimensions
        self.canvas_width = 280
        self.canvas_height = 280
        
        # Create drawing canvas
        self.canvas = tk.Canvas(root, width=self.canvas_width, height=self.canvas_height, bg="white")
        self.canvas.pack(pady=10)
        self.canvas.bind("<B1-Motion>", self.draw_line)
        
        # Track drawing in memory (White background = 255, Black ink = 0)
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), color=255)
        self.draw = ImageDraw.Draw(self.image)
        
        # Buttons & Result Display
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=5)
        
        self.predict_btn = tk.Button(btn_frame, text="Predict Digit", font=("Arial", 12, "bold"), bg="green", fg="white", command=self.predict_digit)
        self.predict_btn.pack(side=tk.LEFT, padx=5)
        
        self.clear_btn = tk.Button(btn_frame, text="Clear Canvas", font=("Arial", 12), command=self.clear_canvas)
        self.clear_btn.pack(side=tk.LEFT, padx=5)
        
        self.result_label = tk.Label(root, text="Draw a digit and click Predict!", font=("Arial", 16))
        self.result_label.pack(pady=10)

    def draw_line(self, event):
        r = 12  # Brush size
        x1, y1 = event.x - r, event.y - r
        x2, y2 = event.x + r, event.y + r
        self.canvas.create_oval(x1, y1, x2, y2, fill="black", outline="black")
        self.draw.ellipse([x1, y1, x2, y2], fill=0)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, self.canvas_width, self.canvas_height], fill=255)
        self.result_label.config(text="Draw a digit and click Predict!")

    def predict_digit(self):
        # 1. Resize image from 280x280 down to 28x28 (MNIST size)
        img_resized = self.image.resize((28, 28), Image.Resampling.LANCZOS)
        
        # 2. Invert colors: Canvas has white background (255) & black ink (0). 
        # MNIST has black background (0) & white ink (255).
        img_inverted = ImageOps.invert(img_resized)
        
        # 3. Transform to PyTorch Tensor (scales pixels from [0, 255] to [0.0, 1.0])
        transform = transforms.ToTensor()
        img_tensor = transform(img_inverted).unsqueeze(0) # Adds batch dimension: shape (1, 1, 28, 28)
        
        # 4. Run model inference
        with torch.no_grad():
            output = self.model(img_tensor)
            prediction = output.argmax(dim=1).item()
            confidence = torch.softmax(output, dim=1)[0][prediction].item() * 100

        # 5. Display result
        self.result_label.config(text=f"Prediction: {prediction} ({confidence:.1f}% confidence)")

if __name__ == "__main__":
    root = tk.Tk()
    app = DigitApp(root)
    root.mainloop()
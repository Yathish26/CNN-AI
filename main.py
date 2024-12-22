import numpy as np
from tkinter import *
from PIL import Image, ImageDraw
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("saved_model/model.h5")

class DrawingApp:
    def __init__(self, master):
        self.canvas = Canvas(master, width=200, height=200, bg="white")
        self.canvas.pack()
        self.image = Image.new("L", (200, 200), color="white")
        self.draw = ImageDraw.Draw(self.image)
        self.canvas.bind("<B1-Motion>", self.paint)

        Button(master, text="Predict", command=self.predict).pack()
        Button(master, text="Clear", command=self.clear).pack()

    def paint(self, event):
        x, y = event.x, event.y
        self.canvas.create_oval(x-5, y-5, x+5, y+5, fill="black")
        self.draw.ellipse([x-5, y-5, x+5, y+5], fill="black")

    def clear(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (200, 200), color="white")
        self.draw = ImageDraw.Draw(self.image)

    def predict(self):
        img = self.image.resize((28, 28)).convert("L")
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)

        prediction = model.predict(img_array)
        print(f"Prediction: {np.argmax(prediction)}")

root = Tk()
app = DrawingApp(root)
root.mainloop()

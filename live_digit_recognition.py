import cv2
import torch
import torch.nn as nn               #   using nn and optim for Neural network (optim = Optimizer)
import torch.optim as optim
from torchvision import datasets, transforms    #   Dataset (MNIST) နဲ့ Image Preprocessing ပြင်ဖို့ပါ။
from torch.utils.data import DataLoader         #   Data များကို batch-by-batch ဖတ်ဖို့။

# Define CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)     #   input 1 channel --> (black & white) output 32 filters (3x3 kernel, stride 1)
                                                #   nput size = 28 (MNIST image size = 28×28)
                                                #   Output Size = (Input Size−Kernel Size + 2×Padding) / Stride + 1
                                                #   Shape = [batch, 32, 26, 26]        2 နဲ့စားဖို့တော့လို

        self.conv2 = nn.Conv2d(32, 64, 3, 1)    #   input 32 → output 64 filters (3x3 kernel)။
        self.fc1 = nn.Linear(64 * 5 * 5, 128)   #   image ကို 1D tensor (vector) ပြောင်း     #   128 = neurons
        self.fc2 = nn.Linear(128, 10)           #   digit (0-9)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))  # [batch, 32, 26, 26]    Conv1 လုပ်ပြီး ReLU activation ထည့်တယ်။
        x = nn.functional.max_pool2d(x, 2)     # [batch, 32, 13, 13]    image size ကို 2x2 Max Pool လုပ်ပြီးလျှော့တယ်။
        x = nn.functional.relu(self.conv2(x))  # [batch, 64, 11, 11]    Conv2 အပြီး ReLU activation ထပ်တင်တယ်။
        x = nn.functional.max_pool2d(x, 2)     # [batch, 64, 5, 5]      Image size ကို ထပ်မံ Pooling လုပ်ပြီး 5x5 ဖြစ်အောင်လျှော့တယ်။
        x = x.view(-1, 64 * 5 * 5)             # flatten dynamically matching size above        -1 ဆိုတာ → batch size ကို PyTorch ကအလိုအလျောက်တွက်ပေးမှာဖြစ်တယ်။
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)                        # Layer output: digit class 10
        return x


# Training function
def train_model():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))    # pre-process for MNIST Image (Image ကို tensor ပြောင်းပြီး [-1, +1] range ပြင်တယ်။)
                                                # normalize
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    model = CNN()
    criterion = nn.CrossEntropyLoss()           # use CrossEntropy Loss to predict true or false
    optimizer = optim.Adam(model.parameters(), lr=0.001)    #lr is learning rate | 

    model.train()                               # model ကို tain mode ထဲထည့်
    for epoch in range(1):                      # try 10 or more epochs
        print(f"Epoch {epoch + 1}")
        for images, labels in train_loader:     #  Batch-by-batch Data ကို Read လုပ်တယ်။
            optimizer.zero_grad()               #  Gradient တွေ reset ပြန်လုပ်
            outputs = model(images)             #  Model ထဲကို images ထည့်ပြီး prediction lote p
            loss = criterion(outputs, labels)   #  Prediction နဲ့ Ground Truth ကို loss တွက်
            loss.backward()                     #  Gradient တွေတွက်ဖို့။ Do Backpropagation
            optimizer.step()                    #  Gradient တွေကိုအသုံးပြုပြီး Weight တွေ update လုပ်


    torch.save(model.state_dict(), 'mnist_cnn.pth')     #  သင်ပြီးတဲ့ model ကို file (mnist_cnn.pth) သိမ်းတယ်။
    print("Training complete and model saved as mnist_cnn.pth")

# Live webcam recognition
def live_recognition():
    # Load model
    model = CNN()
    model.load_state_dict(torch.load('mnist_cnn.pth'))
    model.eval()

    # Preprocessing transform (must match training)
    preprocess = transforms.Compose([           # WebCam image တွေကို 28x28 grayscale ပြင်ရန်။
        transforms.ToPILImage(),
        transforms.Resize((28, 28)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    cap = cv2.VideoCapture(0)               

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # to enable mirror mode

        h, w = frame.shape[:2]

        # Define ROI box (center square)
        box_size = 200
        x1, y1 = w // 2 - box_size // 2, h // 2 - box_size // 2
        x2, y2 = x1 + box_size, y1 + box_size

        # Draw rectangle ROI on frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Crop ROI and preprocess
        roi = frame[y1:y2, x1:x2]
        img = preprocess(roi)
        img = img.unsqueeze(0)  # add batch dimension

        # Predict digit
        with torch.no_grad():                  # Evaluation mode မှာ Gradient မတွက်ပါ။
            outputs = model(img)               # Prediction result မှာ max score ရတဲ့ digit ကို ယူတယ်။
            _, pred = torch.max(outputs, 1)
            predicted_digit = pred.item()    

        # Display prediction on frame
        cv2.putText(frame, f'Prediction: {predicted_digit}', (10, 30),           # Predict ဖြစ်တဲ့ နံပါတ်ကို screen ပေါ်မှာ ပြတယ်
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Live Digit Recognition - Press 'q' to quit", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Training model...")
    train_model()
    print("Starting live digit recognition...")
    live_recognition()

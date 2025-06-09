import cv2
import torch
import torch.nn as nn               #   using nn and optim for Neural network (optim = Optimizer)
import torch.optim as optim
from torchvision import datasets, transforms    #   change image to tensor
from torch.utils.data import DataLoader         #   Data များကို batch-by-batch ဖတ်ဖို့။

# Define CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)  # <- update this after checking output size
        self.fc2 = nn.Linear(128, 10)   # digit (0-9)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))  # [batch, 32, 26, 26]
        x = nn.functional.max_pool2d(x, 2)     # [batch, 32, 13, 13]
        x = nn.functional.relu(self.conv2(x))  # [batch, 64, 11, 11]
        x = nn.functional.max_pool2d(x, 2)     # [batch, 64, 5, 5]
        x = x.view(-1, 64 * 5 * 5)              # flatten dynamically matching size above
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Training function
def train_model():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    model = CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(1):  # try 10 or more epochs
        print(f"Epoch {epoch + 1}")
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()


    torch.save(model.state_dict(), 'mnist_cnn.pth')
    print("Training complete and model saved as mnist_cnn.pth")

# Live webcam recognition
def live_recognition():
    # Load model
    model = CNN()
    model.load_state_dict(torch.load('mnist_cnn.pth'))
    model.eval()

    # Preprocessing transform (must match training)
    preprocess = transforms.Compose([
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
        with torch.no_grad():
            outputs = model(img)
            _, pred = torch.max(outputs, 1)
            predicted_digit = pred.item()

        # Display prediction on frame
        cv2.putText(frame, f'Prediction: {predicted_digit}', (10, 30),
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

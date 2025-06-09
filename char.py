from torchvision.datasets import EMNIST
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define CNN model for 26 characters
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 26)  # 26 characters A-Z

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))     # [batch, 32, 26, 26]
        x = nn.functional.max_pool2d(x, 2)        # [batch, 32, 13, 13]
        x = nn.functional.relu(self.conv2(x))     # [batch, 64, 11, 11]
        x = nn.functional.max_pool2d(x, 2)        # [batch, 64, 5, 5]
        x = x.view(-1, 64 * 5 * 5)                # flatten
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train_model():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.EMNIST(root='./data', split='letters', train=True,transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    model = CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(1):  # Increase to 10+ for better performance
        print(f"Epoch {epoch + 1}")
        for images, labels in train_loader:
            labels = labels - 1  # EMNIST labels: 1-26 → 0-25
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), 'char_emnist_cnn.pth')
    print("Training complete and model saved as char_emnist_cnn.pth")


def live_recognition():
    model = CNN()
    model.load_state_dict(torch.load('char_emnist_cnn.pth'))
    model.eval()

    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((28, 28)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Label mapping: 0 → A, ..., 25 → Z
    label_map = [chr(i) for i in range(ord('A'), ord('Z') + 1)]

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        if not ret:
            break

        h, w = frame.shape[:2]
        box_size = 200
        x1, y1 = w // 2 - box_size // 2, h // 2 - box_size // 2
        x2, y2 = x1 + box_size, y1 + box_size

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        roi = frame[y1:y2, x1:x2]
        img = preprocess(roi)
        img = img.unsqueeze(0)

        with torch.no_grad():
            outputs = model(img)
            _, pred = torch.max(outputs, 1)
            predicted_char = label_map[pred.item()]

        cv2.putText(frame, f'Prediction: {predicted_char}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Live Character Recognition - Press 'q' to quit", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("Training model...")
    train_model()
    print("Starting live character recognition...")
    live_recognition()
"""Train the digit classifier and save it to models/."""
from model import MODEL_PATH, train

if __name__ == "__main__":
    _, accuracy = train()
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Model saved to {MODEL_PATH}")

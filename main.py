import torch
import torchvision
import cv2
from deep_learning_model import MakeBlue
import time
import pygame

def distraction_sound():
    pygame.init()
    pygame.mixer.init()
    pygame.mixer.Sound("beep.wav").play()

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = torch.load("resnet152.pth")

    for parameter in model.parameters():
        parameter.requires_grad = False

    model.fc = torch.nn.Linear(2048, 10)
    model.eval()

    transformation = torchvision.transforms.Compose((MakeBlue(), torchvision.transforms.ToTensor()))

    camera = cv2.VideoCapture(0)

    distracted_start_time = None

    while True:
        _, frame = camera.read()
        frame = cv2.resize(frame, (240, 240), interpolation=cv2.INTER_AREA)

        frame_transformed = transformation(frame)
        frame_transformed.to(device)

        output = model(torch.stack([frame_transformed]))
        _, prediction = torch.max(output, 1)
        distracted = prediction.item() > 0
        print(distracted)

        if distracted:
            current_time = time.perf_counter_ns()

            if distracted_start_time is None:
                distracted_start_time = current_time

            if current_time - distracted_start_time >= 2 * 10 ** 9:
                distraction_sound()

        else:
            distracted_start_time = None

        cv2.imshow("Camera", frame)

        if cv2.waitKey(1) == ord("q"):
            break

    cv2.destroyAllWindows()
    camera.release()

if __name__ == "__main__":
    main()

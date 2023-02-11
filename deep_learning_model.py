from copy import deepcopy
import time
from PIL import Image
import torch
import torchvision

class MakeBlue:
    def __init__(self):
        self.dark_blue = Image.new("RGB", (224, 224), "#0000ff")

    def __call__(self, image):
        try:
            image = torchvision.transforms.ToPILImage()(image)

        except TypeError:
            try:
                image = torchvision.transforms.ToPILImage()(image.byte())

            except Exception:
                pass

        image = image.resize((224, 224))
        image = image.convert("RGB")
        image = Image.blend(image, self.dark_blue, 0.6)
        return image

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    transformation = torchvision.transforms.Compose((MakeBlue(), torchvision.transforms.ToTensor()))
    distracted_driver_data = torchvision.datasets.ImageFolder("data", transform=transformation)

    data_size = len(distracted_driver_data)

    training_data_size, test_data_size = int(0.9 * data_size), data_size - int(0.9 * data_size)
    training_data, test_data = torch.utils.data.random_split(distracted_driver_data, (training_data_size, test_data_size))
    training_data_loader, test_data_loader = torch.utils.data.DataLoader(training_data, batch_size=224, shuffle=True, num_workers=8, pin_memory=True), \
                                             torch.utils.data.DataLoader(training_data, batch_size=224, shuffle=True, num_workers=8, pin_memory=True)

    loss_criteria = torch.nn.CrossEntropyLoss()

    def train(model: torchvision.models.ResNet, epochs: int):
        optimizer = torch.optim.SGD([parameter for _, parameter in model.named_parameters() if parameter.requires_grad is True], lr=0.01, momentum=0.9)

        best_model, best_accuracy = deepcopy(model.state_dict()), 0

        for epoch in range(1, epochs + 1):
            print(f"Epoch {epoch}/{epochs}")
            print("-" * 24)

            for mode in ("train", "test"):
                model.train(mode == "train")
                accuracy, loss = 0, 0

                data_loader = training_data_loader if mode == "train" else test_data_loader

                for images, correct_classes in data_loader:
                    images.to(device)
                    correct_classes.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(mode == "train"):
                        output = model(images)
                        image_loss = loss_criteria(output, correct_classes)
                        _, predictions = torch.max(output, 1)

                        if mode == "train":
                            image_loss.backward()
                            optimizer.step()

                    loss += image_loss.item() * images.size(0) / len(data_loader.dataset)
                    accuracy += torch.sum(predictions == correct_classes.data) / len(data_loader.dataset)

                print(f"{mode}: {loss = }, {accuracy = }")

                if mode == "test" and accuracy > best_accuracy:
                    best_accuracy, best_model = accuracy, deepcopy(model.state_dict())

                torch.save(model, "resnet152.pth")
                torch.save(best_model, "best_resnet152.pth")

        model.load_state_dict(best_model)
        return model

    start_time = time.perf_counter()
    model = torch.load("resnet152.pth")

    for parameter in model.parameters():
        parameter.requires_grad = False

    model.fc = torch.nn.Linear(2048, 10)

    model = train(model, 6)
    torch.save(model, "resnet152.pth")
    end_time = time.perf_counter()

    print(f"Resnet 152: {end_time - start_time} secs")

if __name__ == "__main__":
    main()

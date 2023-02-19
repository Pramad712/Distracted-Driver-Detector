import numpy
import torch
import torchvision
from matplotlib import pyplot
from deep_learning_model import MakeBlue

class Resize:
    def __init__(self):
        pass

    def __call__(self, image):
        try:
            image = torchvision.transforms.ToPILImage()(image)

        except TypeError:
            try:
                image = torchvision.transforms.ToPILImage()(image.byte())

            except Exception:
                pass

        image = image.resize((224, 224))
        return image

def confusion_matrix(model, device, data):
    model.train(False)

    array = [[0 for _ in range(3)] for _ in range(3)]
    array_2 = [[0 for _ in range(10)] for _ in range(10)]

    for index, (image, correct_class) in enumerate(data):
        image.to(device)

        with torch.set_grad_enabled(False):
            output = model(torch.stack([image]))
            _, prediction = torch.max(output, 1)

        array[int(correct_class == 0)][int(prediction == 0)] += 1
        array_2[correct_class][prediction] += 1

        print(index)

    array[0][2] = array[0][0] / (array[0][0] + array[0][1])
    array[1][2] = array[1][1] / (array[1][1] + array[1][0])
    array[2][0] = array[0][0] / (array[0][0] + array[1][0])
    array[2][1] = array[1][1] / (array[1][1] + array[0][1])
    array[2][2] = (array[0][0] + array[1][1]) / (array[0][0] + array[0][1] + array[1][0] + array[1][1])

    for row_index in range(len(array)):
        for column_index in range(len(array)):
            array[row_index][column_index] = str(array[row_index][column_index])

    for row_index in range(len(array_2)):
        for column_index in range(len(array_2)):
            array_2[row_index][column_index] = str(array_2[row_index][column_index])

    return array, array_2

def display_table(array, row_labels, column_labels):
    pyplot.rcParams["figure.figsize"] = [7, 7]
    pyplot.rcParams["figure.autolayout"] = True

    figure, axes = pyplot.subplots()
    axes.axis("tight")
    axes.axis("off")

    axes.table(cellText=array, rowLabels=row_labels, colLabels=column_labels)
    pyplot.show()

def show_image(image):
    try:
        image = image.numpy()

    except Exception:
        image = numpy.array(image)

    image = image.transpose((1, 2, 0))
    pyplot.imshow(image)

def sample_predictions(model, data, transformation):
    model.eval()

    for index, (images, correct_class) in enumerate(data):
        if index >= 7:
            break

        with torch.no_grad():
            for index, image in enumerate(images):
                pyplot.rcParams["figure.figsize"] = [7, 7]
                pyplot.rcParams["figure.autolayout"] = True

                figure, axes = pyplot.subplots()
                axes.axis("off")

                show_image(image)
                image = transformation(image)

                output = model(torch.stack([image]))
                _, prediction = torch.max(output, 1)

                prediction = (prediction > 0)
                correct_class = (correct_class > 0)

                axes.set_title(f"Prediction: {prediction.item()}, Answer: {correct_class[index]}")
                pyplot.show()

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transformation = torchvision.transforms.Compose((MakeBlue(), torchvision.transforms.ToTensor()))
    resize = torchvision.transforms.Compose((Resize(), torchvision.transforms.ToTensor()))

    model = torch.load("resnet152.pth")

    # WARNING: May take hours to run!
    distracted_driver_data = torchvision.datasets.ImageFolder("data", transform=transformation)
    array, array_2 = confusion_matrix(model, device, distracted_driver_data)
    display_table(array, ["Positive", "Negative", ""], ["Positive", "Negative", ""])
    display_table(array_2, [f"c{distraction_type}" for distraction_type in range(10)], [f"c{distraction_type}" for distraction_type in range(10)])

    distracted_driver_data = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder("data", transform=resize), shuffle=True, batch_size=10)
    sample_predictions(model, distracted_driver_data, transformation)

if __name__ == "__main__":
    main()

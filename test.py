import torch
from torchvision import ops
from torchvision.transforms import transforms

from load_LIDC_crops import LIDC_CROPS


def test_ops():
    # Bounding box coordinates.
    ground_truth_bbox = torch.tensor([[1202, 123, 1650, 868]], dtype=torch.float)
    prediction_bbox = torch.tensor([[1162.0001, 92.0021, 1619.9832, 694.0033]], dtype=torch.float)

    print(ground_truth_bbox.shape)
    print(prediction_bbox.shape)

    # Get iou.
    iou = ops.box_iou(ground_truth_bbox, prediction_bbox)
    print('IOU : ', iou.numpy()[0][0])


def main():
    data_path_train = "data/train/"
    data_path_test = "data/test/"
    img_size = 128

    train_data = LIDC_CROPS(dataset_location=data_path_train, img_size=128)
    test_data = LIDC_CROPS(dataset_location=data_path_test, img_size=128)
    torch.save(test_data.images, "data/pt_files/train/images.pt")
    torch.save(test_data.labels, "data/pt_files/train/labels.pt")
    torch.save(test_data.images, "data/pt_files/test/images.pt")
    torch.save(test_data.labels, "data/pt_files/test/labels.pt")
    print(test_data[0][0].shape, test_data[0][1].shape)


def main2():
    data_path_train = "data/pt_files/train/"
    data_path_test = "data/pt_files/test/"
    img_size = 128

    train_data = LIDC_CROPS(dataset_location=data_path_train, folder_type=False)
    test_data = LIDC_CROPS(dataset_location=data_path_test, folder_type=False)
    print(test_data[0][0].shape, test_data[0][1].shape)


if __name__ == "__main__":
    main2()

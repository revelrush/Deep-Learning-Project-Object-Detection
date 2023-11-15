import os
import numpy as np
import torch
import torch.utils.data
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import cv2


def get_instance_segmentation_model(num_classes):
    # Load a model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # Replace the classifier with a new one, that has num_classes which is user-defined
    # Get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def main(args):
    path = os.getcwd()
    if args.image != None:
        image = path = os.path.join(path, args.image)
    else:
        image = os.path.join(path, 'Drinks', '0000044.jpg')

    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_classes = 4  # 3 class (number of classname) + 1 class (background)
    model = get_instance_segmentation_model(num_classes)
    path = os.getcwd()
    model.load_state_dict(torch.load(os.path.join(path, 'Drinks', 'model_weightsv2.pth')))
    model.to(device)

    # open file and create copy
    image = cv2.imread('test.jpg')
    original = image.copy()
    # convert the image from BGR to RGB channel
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # change the image from channels last to channels first ordering
    image = image.transpose((2, 0, 1))
    # add the batch dimension, normalize image and convert to tensor
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    image = torch.FloatTensor(image)
    # load to device and predict
    image = image.to(device)
    detections = model(image)[0]
    classes = ['bg', 'Summit Drinking Water 500ml', 'Coca-Cola 330ml', 'Del Monte 100% Pineapple Juice 240ml']
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    for i in range(0, len(detections["boxes"])):
        # extract probability associated with the prediction
        confidence = detections["scores"][i]
        # filter out weak detections
        if confidence > 0.8:
            # extract the index of the class label from the detections and find bbox coordinates
            idx = int(detections["labels"][i])
            box = detections["boxes"][i].detach().cpu().numpy()
            (xmin, ymin, xmax, ymax) = box.astype("int")
            # display the prediction to our terminal
            label = "{}: {:.2f}%".format(classes[idx], confidence * 100)
            print("[INFO] {}".format(label))
            # draw the bounding box and label on the image
            cv2.rectangle(original, (xmin, xmax), (ymin, ymax), colors[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(original, label, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[idx], 2)
    # show the output image
    cv2.imshow(orig)

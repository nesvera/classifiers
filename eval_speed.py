import torch
from models import MobileNet
from utils import utils

from torchsummary import summary

import time

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #model = MobileNet.MobileNetV1Conv224(num_classes=30)
    #model = MobileNet.MobileNetV1Dw224(num_classes=30)
    model = torch.hub.load('pytorch/vision', 'MobileNetV2')
    
    
    model = model.to(device)
    model.eval()

    summary(model, (3, 224, 224))

    input()

    prediction_time = utils.Average()    
    start_time = 0

    with torch.no_grad():
        for i in range(10000):

            image = torch.randn(1, 3, 224, 224).to(device)

            start_time = time.time()

            prediction = model(image)

            prediction_time.add_value(time.time()-start_time)

            print("[{0}/{1}]".format(i,1000))

    print("Average prediction time: {0:.4f}".format(prediction_time.get_average()))
    print("Min prediction time: {0:.4f}".format(prediction_time.min))
    print("Max prediction time: {0:.4f}".format(prediction_time.max))
    
    


import torch

import os
import argparse
import matplotlib.pyplot as plt

from torchsummary import summary

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",
                        dest="model_path",
                        help="Path to the model",
                        required=True)
    
    args = parser.parse_args()
    model_path = args.model_path

    if os.path.exists(model_path) == False:
        print("ERROR: Model file was not found!")
        exit(1)
       
    # load model
    checkpoint = torch.load(model_path, map_location=device)

    #model_name =    checkpoint['model_name']
    cur_epoch =     checkpoint['epoch']
    model =         checkpoint['model']
    optimizer =     checkpoint['optimizer']
    train_loss_log =checkpoint['train_loss_log']
    val_loss_log =  checkpoint['val_loss_log']
    top_5_log =     checkpoint['top_5_log']
    top_1_log =     checkpoint['top_1_log']
    lr_log =        checkpoint['lr_log']

    # summarize the model
    print()
    print("----------------------------------------------------------------")
    print("--------------------- Model summary ----------------------------")
    print("----------------------------------------------------------------")
    summary(model, (3, 224, 224))
    
    # plot stats
    fig, sub_plot = plt.subplots(2, 1)

    # train and validation loss
    train_loss = sorted(train_loss_log.items())
    train_loss_x, train_loss_y = zip(*train_loss) # unzip a list

    val_loss = sorted(val_loss_log.items())
    val_loss_x, val_loss_y = zip(*val_loss)

    tl_graph, = sub_plot[0].plot(train_loss_x, train_loss_y, 'b')
    vl_graph, = sub_plot[0].plot(val_loss_x, val_loss_y, 'r')
    sub_plot[0].set_xlabel('Epoch')
    sub_plot[0].set_ylabel('Loss')
    sub_plot[0].set_xlim(0, max(train_loss_x[-1], val_loss_x[-1]))

    # learning rate
    lr = sorted(lr_log.items())
    lr_x, lr_y = zip(*lr)

    aux_sub_plot = sub_plot[0].twinx()
    lr_graph, = aux_sub_plot.plot(lr_x, lr_y, 'g--')
    aux_sub_plot.set_ylabel('Learning rate')


    # top-5 and top-1
    top_5 = sorted(top_5_log.items())
    top_1 = sorted(top_1_log.items())
    
    top_5_x, top_5_y = zip(*top_5)
    top_1_x, top_1_y = zip(*top_1)

    t5, = sub_plot[1].plot(top_5_x, top_5_y, 'k')
    t1, = sub_plot[1].plot(top_1_x, top_1_y, 'y')
    sub_plot[1].set_xlabel('Epoch')
    sub_plot[1].set_ylabel('Accuracy')
    sub_plot[1].set_xlim(0, max(top_5_x[-1], top_1_x[-1]))

    fig.legend((tl_graph, vl_graph, lr_graph, t5, t1), 
               ('Train loss', 'Val loss', 'Learning rate', 'Top-5', 'Top-1'), 
               'upper right')

    print("Best")
    print("Train loss: {0:.3f} - Val loss: {1:.3f}"
          .format(min(train_loss_y), min(val_loss_y)))
    print("Accuracy Top-5 {0:.3f} - Top-1 {1:.3f}"
          .format(max(top_5_y), max(top_1_y)))

    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

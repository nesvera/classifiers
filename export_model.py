import torch

import os
import argparse
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",
                        dest="model_path",
                        help="Path to the model",
                        required=True)

    parser.add_argument("--output",
                        dest="output_path",
                        help="Path and filename to export the model",
                        required=True)
    
    args = parser.parse_args()
    model_path = args.model_path
    output_path = args.output_path

    if os.path.exists(model_path) == False:
        print("ERROR: Model file was not found!")
        exit(1)
       
    # load model
    checkpoint = torch.load(model_path, map_location=device)
    model =         checkpoint['model']

    torch.save(model.state_dict(), output_path)

if __name__ == "__main__":
    main()

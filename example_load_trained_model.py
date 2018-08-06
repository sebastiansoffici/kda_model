import sys
sys.path.append("..")
from kda_model import KDAModel
import pandas
import argparse

def main(args):
  predict = KDAModel()
  model = predict.load_model(args.loaded_model)


  data_frame = predict.show_predictions(model, args.digit_path, args.slash_path, args.num_predictions)

  data_frame.to_csv(args.predictions_path)

if __name__=='__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--digit_path', type=str, help='path to digit samples')
  parser.add_argument('--slash_path', type=str, help='path to slash samples')
  parser.add_argument('--loaded_model', type=str, help='path to loaded model')
  parser.add_argument('--num_predictions', type=int, help='number of predictions to be executed')
  parser.add_argument('--predictions_path', type=str, help='path to store CSV of predictions')
  main(parser.parse_args())

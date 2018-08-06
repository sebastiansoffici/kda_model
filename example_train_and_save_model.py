import sys
import argparse
sys.path.append("..")
from kda_data import DigitSample
from kda_model import KDAModel
import pandas
import tensorflow

def main(args):
  sample = DigitSample(args.digit_path, args.slash_path)
  image = sample.get_kda(1, 2, 3)
  model = KDAModel(args.num_samples, image)
  for i in range(args.num_samples):
   k,d,a = sample.generate_random_kda()
   model.add_observation(i, sample.get_kda(k,d,a), k, d, a)

  with tensorflow.device('/gpu:0'):
    comp_model = model.get_model()
    model.evaluate_model(comp_model, args.num_epochs)

  model.save_model(comp_model, args.model_path)

  data_frame = model.show_predictions(comp_model, args.digit_path, args.slash_path, args.num_predictions)
  data_frame.to_csv(args.predictions_path)

if __name__=='__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--digit_path', type=str, help='path to digit samples')
  parser.add_argument('--slash_path', type=str, help='path to slash samples')
  parser.add_argument('--model_path', type=str, help='path to where you want to save model')
  parser.add_argument('--num_samples', type=int, help='number of observations to be made')
  parser.add_argument('--num_epochs', type=int, help='number of epochs')
  parser.add_argument('--num_predictions', type=int, help='number of predictions to be executed')
  parser.add_argument('--predictions_path', type=str, help='path to store CSV of predictions')
  main(parser.parse_args())

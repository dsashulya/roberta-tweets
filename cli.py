#!/Users/dsashulya/PycharmProjects/all_projects/venv/bin/python3
import argparse
from model import RobertaClassifier, tokenize
import torch
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("text", help="return the probability of the text being sexist/racist")
    args = parser.parse_args()

    model = RobertaClassifier(hidden_size=768, n_classes=2, dropout=0.3)
    model.load_state_dict(torch.load('roberta.pt', map_location=torch.device('cpu')))

    start_time = time.time()
    input_ids, attention_mask = tokenize(args.text)
    model.eval()
    with torch.no_grad():
        output = model.compute_probabilities(input_ids, attention_mask)
    label = torch.argmax(output).item()
    end_time = time.time()
    print(f"Probability of the tweet being sexist/racist: {(output[0][1] * 100):.1f}%")
    print(f"Time taken: {(end_time - start_time):.1f} sec")


if __name__ == "__main__":
    main()


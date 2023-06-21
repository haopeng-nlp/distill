import matplotlib.pyplot as plt
import re

ROOT = "/gscratch/ark/ivyg/fasttext-debias/logs/"
INPUT_FILES = [
    # "tdrop0.1.dist.out",
    # "tdrop0.2.dist.out",
    "tdrop0.0-seed42-tdrop0.3-seed42.dist.out",
    "tdrop0.0-seed42.dist.out",
    "tdrop0.3-seed42.dist.out",
]

def parse_file(filename: str):
    """
    input: filename
    """
    bleu_scores = []
    with open(filename, "r") as f:
        data = f.readlines()
    for line in data:
        if "| bleu " not in line:
            continue
        try:
            regex = re.compile("| bleu (\d+\.\d+)")
            bleu = next(s for s in regex.findall(line) if s)
        except Exception as e:
            print(e)
        bleu_scores.append(float(bleu))
    return bleu_scores

if __name__ == "__main__":
    for f in INPUT_FILES:
        bleu_scores = parse_file(ROOT + f)
        epochs = range(len(bleu_scores))
        plt.plot(epochs, bleu_scores, label=f)
    plt.legend(loc="lower right")
    plt.savefig(ROOT + "training_curves.png")

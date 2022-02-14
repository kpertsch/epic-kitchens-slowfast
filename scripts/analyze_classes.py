import pickle
from collections import defaultdict

NOUN_CLASSES = ['microwave', 'hob', 'cupboard', 'button', 'kettle']

VERBS = defaultdict(lambda: defaultdict(lambda: 0))


with open("data/EPIC_100_train.pkl", "rb") as F:
    data = pickle.load(F)

for noun, verb in zip(data['noun'], data['verb']):
    if noun in NOUN_CLASSES:
        VERBS[noun][verb] += 1

print(VERBS)
x = 0

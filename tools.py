# -*- encoding: utf-8 -*-
import bz2
import csv
import json
import logging
import os
import pickle
import re
import requests
import time
import tqdm
from collections import Counter, defaultdict as deft
from copy import deepcopy as cp  # noqa

from difflib import SequenceMatcher

from nltk import wordpunct_tokenize as tokenizer, ngrams  # noqa

from tqdm import tqdm

logger = logging.getLogger(__name__)


VOWELS = set(list('aeiou'))
USERS = re.compile('(@[^ ]+|[^ ]+@)')
HASHTAGS = re.compile('(#[^ ]+|[^ ]+#)')


def remove_twitter_metadata(doc):
    doc = USERS.sub('', doc)
    doc = HASHTAGS.sub('', doc)
    return doc

  
def extract_words(string, lowercase=True, rm_num=True):
    return [
        w.lower() if lowercase else w for w in tokenizer(string)
        if not rm_num or w.isalpha()
    ]


def print_time(t=None):
    if not t:
        t = time.time()
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t))


def fetch(url):
    response = requests.get(url)
    return response.text


def from_tsv(path):
    with open(path, encoding='utf-8') as rd:
        for line in rd:
            yield line.split('\t')


def from_csv(path, delimiter=','):
    with open(path, 'r') as rd:
        rdr = csv.reader(rd, delimiter=delimiter)
        return [row for row in rdr]


def to_csv(rows, path):
    with open(path, 'w') as wrt:
        wrtr = csv.writer(wrt, quoting=csv.QUOTE_MINIMAL)
        for row in rows:
            wrtr.writerow(row)


def from_json(path):
    with open(path, 'r') as rd:
        data = json.load(rd)

    return data


def from_pickle(path):
    with open(path, 'rb') as rd:
        return pickle.load(rd)


def has_vowels(w):
    if VOWELS.intersection(set(list(w.lower()))):
        return True
    return False


def to_pickle(data, path):
    with open(path, 'wb') as wrt:
        pickle.dump(data, wrt)


def string_matcher(string, regex):
    space = ''.join(list(string))
    matches = []
    while space:
        match = regex.search(space)
        if not match:
            break
        elif set(match.group()).intersection(set(['#'])):
            break
        start = match.start()
        end = match.end()
        _space = ('#' * end) + space[end:]
        if space == _space:
            break
        space = _space
        matches.append((start, end, match))
    return matches


def strip(string):
    chars = list(string)
    left, right = False, False
    while chars:
        if not chars[0].isalpha():
            chars.pop(0)
        else:
            left = True
        if chars and not chars[-1].isalpha():
            chars.pop()
        else:
            right = True
        if left and right:
            break
    return ''.join(chars)


def product(probs):
    if len(probs) < 2:
        return probs[0]
    prob = probs.pop(0)
    while probs:
        n = probs.pop(0)
        prob *= n
    return prob


def to_json(data, path, indent=None):
    with open(path, 'w') as wrt:
        if indent:
            json.dump(data, wrt, indent=indent)
        else:
            json.dump(data, wrt)


def word_dist(text):
    return Counter(
        [w for w in tokenizer(text.lower())
         if w.isalpha() and len(w) > 3]
    )


def mean(a):
    return sum(a) / len(a)


def variance(a):
    avg = mean(a)
    diffs = [
        (x - avg) ** 2 for x in a
    ]
    return sum(diffs) / len(diffs)


def stddev(a):
    var = variance(a)
    return var ** (1 / 2)


def rescale(values, ignore_top=None, ignore_bottom=None):
    space = cp(values)
#     print('-')
#     print('values', values)
    if len(values) == 1:
        return [1.0]

    if ignore_top:
        n = ignore_top if isinstance(ignore_top, int) \
            else int(len(space) * ignore_top)
        _space = sorted(space)[:-n if n else -1]
        maxim = max(_space)
    else:
        maxim = max(space)

    if ignore_bottom:
        n = ignore_bottom if isinstance(ignore_bottom, int) \
            else int(len(space) * ignore_bottom)
        _space = sorted(space)[:n if n else 1]
        minim = min(_space)
    else:
        minim = min(space)
    
    if maxim and maxim - minim == 0:
        minim = maxim * 0.5
        rock_bottom = (minim * 0.5) / (maxim * 2)
    elif maxim:
        rock_bottom = (minim * 0.5) / (maxim * 2)
    else:
        rock_bottom = 0.0
    
    out_vals = []
    for val in values:
        if maxim and minim:
            _val = (val - minim) / float(maxim - minim)
        else:
            _val = 0.0
        if _val > 1.0:
            out_val = 1.0
        else:
            out_val = _val if val > minim else rock_bottom
        out_vals.append(out_val)

#     print('out_vals', out_vals)
    return out_vals




if __name__ == '__main__':

    print(rescale([0.004270082525315654, 0.004270082525315654, 0.004127272744232898, 0.004127272744232898], ignore_top=0.1, ignore_bottom=0.1))
    #   out_vals [1.0, 1.0, 14.450245329629054, 14.450245329629054]

    print(rescale([0.03718901778794769, 0.03718901778794769, 0.07624929556809702, 0.03831373660488426], ignore_top=0.1, ignore_bottom=0.1))
    #out_vals [16.532584512652015, 16.532584512652015, 1.0, 1.0]
    #exit()

    print([0.05, 0.25, 0.1, 0.5, 0.1])
    print(rescale([0.05, 0.25, 0.1, 0.5, 0.1]))
    
    print([0.05, 0.25, 0.1, 0.5, 0.1])
    print(rescale([0.05, 0.25, 0.1, 0.5, 0.1], ignore_top=1, ignore_bottom=0.1))
    exit()

    print([0.1, 0.2, 0.6, 0.1])
    print(product([0.1, 0.2, 0.6, 0.1]))
    exit()
    
    for start, end, match in string_matcher(
        'alfa',
        re.compile('a')
    ):
        print(start, end, match.group())
    exit()

    print(string_matcher(
        'alfa beta gamma eta',
        re.compile('a')
    ))

    exit()
    logger.debug(strip('Ves per on!'))
    logger.debug(strip('!,Castelló?'))
    logger.debug(strip('autobús comarcal..443?'))

from __init__ import Cluster, from_pickle

import re

PARENTHESES = re.compile('[\(\)]')

class DocumentGetter:

    def __init__(self, path):
        self._path = path
        self._clusters = from_pickle(self._path)
    
    def __getitem__(self, tree):
        if not isinstance(tree, str):
            return
        tree = PARENTHESES.sub('', tree)
        features = set(tree.split('/'))
        num = 1
        for cluster in self._clusters:
            _features = set(cluster._features)
            #print(features, _features)
            intersect = features.intersection(_features)
            diff = features.difference(_features)
            if features <= _features \
            and (
                not diff
                or diff == set(['*'])
            ):
                for text in list(sorted(cluster)):
                    print('[%d]  %s' % (num, text))
                    print()
                    num += 1



if __name__ == '__main__':
    
    dg = DocumentGetter('clusters.p')
    #print(dg["(122/124/123/)403"])
    print(dg['(122/124/)123'])
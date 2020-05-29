import random


from tools import (
    Counter,
    cp,
    deft,
    from_csv,
    from_json,
    from_pickle,
    ngrams,
    remove_twitter_metadata,
    to_csv,
    to_json,
    tokenizer,
    to_pickle
)









class Cluster:

    def __init__(self, freqdist, tree, texts, stopwords=set([])):
        self._freqdist = freqdist
        self._tree = tree
        self._texts = cp(texts)
        self._type = 'core'
        self.active = True
        self._consumed = set(stopwords)
        self._features = []


    def __iter__(self):
        for text in self._texts:
            yield text


    def disable(self):
        self.active = False


    def __len__(self):
        return len(self._texts)
    

    def __str__(self):
        return '%s (%d)' % (
            '/'.join([str(clid) for clid in self._tree]),
            len(self)
        )


    def __clusterize(self, bows):
        freqdist = Counter()
        bowids_by_w = deft(set)
        for bowid, bow in enumerate(bows):
            for w in bow:
                bowids_by_w[w].add(bowid)
                freqdist[w] += 1
        covered = set([])
        rankdist = freqdist.most_common()
        clusters = [-1 for _ in bows]
        features = [None for _ in bows]
        clusterid = 0

        if self._features:
            lower_boundary = min(
                [self._freqdist[_feature] for _feature in self._features]
            )
        else:
            lower_boundary = None
        while rankdist and len(covered) < len(bows):
            feature, frequency = rankdist.pop(0)
            if lower_boundary and self._freqdist[feature] > lower_boundary:
                continue
            elif frequency / float(len(self._texts)) >= 0.66:
                continue
            bowids = bowids_by_w[feature]
            unique_bowids = bowids - covered
            if len(unique_bowids) / float(len(bowids)) < 0.5:
                continue
            for bowid in unique_bowids:
                clusters[bowid] = clusterid
                features[bowid] = feature
            covered.update(unique_bowids)
            clusterid += 1
            if clusterid == MAXIMUM_DEGREE - 1:
                break

        for i, cl in enumerate(clusters):
            if cl == -1:
                clusters[i] = MAXIMUM_DEGREE - 1

        return clusters, features

    def __set_max_freq(self, l):
        if self._tree:
            return l * 0.33
        else:
            return l * 0.05

    def split(self):
        
        self.active = False
        if len(self._texts) < MINIMUM_DOCUMENTS_IN_CLUSTER + 1:
            return [self]

        l = len(self)

        # Extract features
        max_freq = self.__set_max_freq(l)
        
        freqdist = Counter()
        _bows = []
        for text in self._texts:
            tokens = set([
                w.lower() for w in tokenizer(text)
#                 if w.isalpha() or [
#                     char for char in w if not char.isalpha()
#                 ] == ['_']
            ])
            tokens -= self._consumed
            _bows.append(tokens)
            for w in tokens:
                freqdist[w] += 1 * (1 / float(len(w)))
        vocab = set([
            w for w, freq in freqdist.items()
            if freq < max_freq
#             and freq > round(math.log(l, 2.5))
        ])

        bows = []
        for _bow in _bows:
            bow = [w for w in _bow if w in vocab]
            bows.append(bow)

        clusters, features = self.__clusterize(bows)
        if not self._tree:
            self._consumed.update(
                [w for w, freq in freqdist.items() if freq > max_freq]
            )
        self._consumed.update(features)

        clustered = list(zip(clusters, features, self._texts))        
        clusters = deft(list)
        cluster_feature = dict([])
        for clid, feature, text in clustered:
            clusters[clid].append(text)
            cluster_feature[clid] = feature
        
        outgoing_clusters, catchall_cluster = [], []
        available_clids = set([])
        for clid, texts in clusters.items():
            if len(texts) < MINIMUM_DOCUMENTS_IN_CLUSTER \
            or not cluster_feature[clid]:
                catchall_cluster += texts
                available_clids.add(clid)
                continue
            outgoing_cluster = Cluster(
                self._freqdist,
                cp(self._tree) + [clid],
                texts
            )
            outgoing_cluster._consumed = cp(self._consumed)
            outgoing_cluster._features = \
                cp(self._features) + [cluster_feature[clid]]
            outgoing_clusters.append(outgoing_cluster)
        
        if catchall_cluster:
            fallback_clid = min(available_clids)
            __tree = cp(self._tree) + [fallback_clid]

            outgoing_cluster = Cluster(
                self._freqdist, __tree, catchall_cluster
            )
            outgoing_cluster._consumed = cp(self._consumed)
            outgoing_cluster._features = cp(self._features) + ['*']
            outgoing_cluster._type = 'catch_all'
            outgoing_clusters.append(outgoing_cluster)

        if len(outgoing_clusters) == 1:
            outgoing_clusters[0].disable()

        return outgoing_clusters

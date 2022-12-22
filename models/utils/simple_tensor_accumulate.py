import itertools
import operator


def accumulate_predictions(l):
    it = itertools.groupby(l, operator.itemgetter(0))
    for key, subiter in it:
        group_list = list(subiter)
        total = sum(tensors for tensor_id, tensors in group_list)
        yield key, total / len(group_list)
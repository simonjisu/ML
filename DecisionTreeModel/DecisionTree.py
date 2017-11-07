# -*- coding: utf-8 -*-
import numpy as np

class Decision_Tree(object):
    def __init__(self):
        self.bestfeat_list = []
        self.bestfeat_label_list = []
        pass

    def example_dataset(self):
        data = np.array([[0, 1, 1, 'yes'],
                         [0, 1, 0, 'no'],
                         [1, 0, 1, 'no'],
                         [1, 1, 1, 'no'],
                         [0, 1, 0, 'no'],
                         [0, 0, 1, 'no'],
                         [1, 0, 1, 'no'],
                         [1, 1, 0, 'no']])
        labels = ['cartoon','winter', 'more than 1 person']
        self.labels = labels.copy()
        return data, labels


    def encode_dtype(self, x):
        x_ = np.zeros_like(x, dtype=np.int)
        for i, k in enumerate(np.unique(x)):
            x_[x == k] = i
        return x_


    def frequency_matrix(self, x, y=None, total=True):
        """x type must be numbers, if total True return total counts at last cols or row"""

        if y is not None:
            mat_list = []
            labels = {'rows': np.unique(x), 'cols': np.unique(y)}
            # encode to numer
            if x.dtype not in [int, float]:
                x_ = self.encode_dtype(x)

            for k in labels['cols']:
                mask = (y == k)
                mat_list.append(np.bincount(x_[mask], minlength=len(labels['rows'])))

            if total:
                nr, nc = len(labels['rows']), len(labels['cols'])
                mat = np.zeros((nr + 1, nc + 1))
                mat[:-1, :-1] = np.vstack(mat_list).T
                mat.sum(axis=1, out=mat[:, -1])
                mat.sum(axis=0, out=mat[-1, :])
                labels['rows'] = np.append(labels['rows'], 'total')
                labels['cols'] = np.append(labels['cols'], 'total')
            else:
                mat = np.vstack(mat_list)

            return mat, labels

        else:
            labels = np.unique(x)
            if x.dtype not in [int, float]:
                x_ = self.encode_dtype(x)

            if total:
                mat = np.zeros(len(np.unique(x)) + 1)
                mat[:-1] = np.bincount(x_, minlength=len(np.unique(x)))
                mat[-1] = mat.sum()
                labels = np.append(labels, 'total')
            return mat, labels


    def cal_entropy(self, data, i=0):
        if data.ndim == 1:
            x = data
            e = 0.
            mat, labels = self.frequency_matrix(x)
            # np.log2(0) can get -inf, can't calculate, has to be np.log2(1)
            for k in range(len(labels)):  # labels = np.unique(x)
                prob = mat[k] / mat[-1]
                e -= prob * np.log2(1) if prob == 0. else prob * np.log2(prob)
            return e

        else:
            y = data[:, -1]
            x = data[:, i]
            e = 0.
            mat, labels = self.frequency_matrix(x, y)
            for i in range(mat.shape[0] - 1):
                for j in range(mat.shape[1] - 1):
                    prob = mat[i, j] / mat[-1, -1]
                    cond_prob = mat[i, j] / mat[i, -1]
                    e -= prob * np.log2(1) if cond_prob == 0. else prob * np.log2(cond_prob)
            return e


    def choosebestFeature(self, data):
        n_features = data.shape[1] - 1
        baseE = self.cal_entropy(data[:, -1])  # calculate the base entropy from y = data[:, -1]
        bestIG = 0.
        best_feat = -1

        for i in range(n_features):
            newE = self.cal_entropy(data, i=i)
            newIG = baseE - newE
            if (newIG > bestIG):
                bestIG = newIG
                best_feat = i
        return best_feat


    def build_Tree(self, data, feat_labels):
        y_array = data[:, -1]
        if self.stop_condition(y_array, feat_labels):
            return y_array[0]
        best_feat = self.choosebestFeature(data)
        best_feat_label = feat_labels[best_feat]
        self.bestfeat_list.append(self.labels.index(best_feat_label))
        self.bestfeat_label_list.append(best_feat_label)
        myTree = {best_feat_label:{}}
        del(feat_labels[best_feat])
        unique_x = np.unique(data[:, best_feat])
        for x in unique_x:
            sublabels = feat_labels[:]
            myTree[best_feat_label][x] = self.build_Tree(self.delete_used_data(data, best_feat, x), sublabels)

        return myTree

    def stop_condition(self, y_array, feat_labels):
        if (len(np.unique(y_array)) == 1) | (feat_labels == []):
            return True

    def delete_used_data(self, data, i, value):
        data = data[data[:, i] == value, :]
        mask = np.arange(data.shape[1])
        mask = np.delete(mask, i)
        data = data[:, mask]

        return data

    def predict(self, tree, test_data, test_label=None):
        path = self.make_path(test_data)
        path_items = path.split('/')
        for path_item in path_items:
            value = tree[path_item]
            if isinstance(value, dict):
                tree = value
            else:
                if test_label:
                    return value, (test_label == value)
                else:
                    return value

    def make_path(self, data):
        path = ''
        data = data[self.bestfeat_list]
        for val, label in zip(data, self.bestfeat_label_list):
            path += str(label) + '/' + str(val) + '/'
        path = path.strip('/')
        return path


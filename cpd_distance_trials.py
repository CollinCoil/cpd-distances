import distance
import math
import numpy as np
import time
from abc import ABC
from dataclasses import dataclass
from typing import Iterable, List
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

import numpy as np

@dataclass
class Task:
    name: str
    data: np.array
    labels: np.array


class ClusteredDataReader(ABC):
    def __init__(self, data_path):
        data = np.load(data_path, allow_pickle=True)
        self.train_tasks = []
        self.test_tasks = []

        for c in data:
            self.train_tasks.append(Task(name=c['name'], data=c['train_data'], labels=None))
            if 'test_data' in c and len(c['test_data']) > 0:
                test_data = c['test_data']
                test_labels = c['test_labels']
                self.test_tasks.append(Task(name=c['name'], data=test_data, labels=test_labels))

    def load_test_tasks(self) -> List[Task]:
        return self.test_tasks

    def iterate_tasks(self) -> Iterable[Task]:
        return self.train_tasks



def distance_function(sample, dist, dist_metric):
    try:
        dist_mean = np.mean(dist, axis = 0)
        dist_mean_list = [dist_mean]*sample.shape[0]
        return dist_metric(sample, dist_mean_list)
    except:
        print("Failure")
        exit()


def iterate_batches(points, batch_size):
    samples_number = math.ceil(points.shape[0] / batch_size)
    for sample_id in range(0, samples_number):
        sample = points[sample_id * batch_size: (sample_id + 1) * batch_size]
        yield sample


class modular_detector:
    def __init__(self, threshold=3, new_dist_buffer_size=3, batch_size=3, max_dist_size=0, dist_metric = distance.euclidean, DEBUG=False):
        self.threshold_ratio = threshold
        self.max_dist_size = max_dist_size
        self.new_dist_buffer_size = new_dist_buffer_size
        self.batch_size = batch_size
        self.is_creating_new_dist = True
        self.dist = []
        self.dist_values = []
        self.locations = []
        self.dist_metric = dist_metric
        self.DEBUG=DEBUG
        if self.DEBUG:
            self.values = []

    def detect(self, data):
        for batch_id, batch in enumerate(iterate_batches(data, batch_size=self.batch_size)):
            if self.is_creating_new_dist:
                self.dist.extend(batch)

                if len(self.dist) >= self.new_dist_buffer_size:
                    self.is_creating_new_dist = False
                    values = [distance_function(np.array(s), np.array(self.dist), self.dist_metric) for s in
                                        iterate_batches(np.array(self.dist), self.batch_size)]
                    self.threshold = np.max(values) * self.threshold_ratio

            else:
                value = distance_function(np.array(batch), np.array(self.dist), self.dist_metric)

                if value > self.threshold:
                    self.locations.append(batch_id * self.batch_size)
                    self.dist = []
                    self.is_creating_new_dist = True

                if self.max_dist_size == 0 or len(self.dist) < self.max_dist_size:
                    self.dist.extend(batch)
                    values = [distance_function(np.array(s), np.array(self.dist), self.dist_metric) for s in
                                        iterate_batches(np.array(self.dist), self.batch_size)]
                    self.threshold = np.max(values) * self.threshold_ratio

        return self.locations



def load_data(data_path):

    cdr = ClusteredDataReader(data_path)

    train = cdr.train_tasks

    unique_task = []
    for train_task in train:
        # print(train_task.data.shape)
        unique_task.append(train_task.data)
    data = np.vstack(unique_task)

    true_changes = [9000, 15487, 24487, 33487]
    return true_changes, data

# Most metrics are borrowed from Turing Change Points Detection Benchmark

def true_positives(T, X, margin=5):
    # make a copy so we don't affect the caller
    X = set(list(X))
    TP = set()
    for tau in T:
        close = [(abs(tau - x), x) for x in X if abs(tau - x) <= margin]
        close.sort()
        if not close:
            continue
        dist, xstar = close[0]
        TP.add(tau)
        X.remove(xstar)
    return TP

def accuracy_estimation(actual, predicted):
    delays = []

    for prediction in predicted:
        closest_true_change = actual[min(range(len(actual)), key = lambda i: abs(actual[i]-prediction))]
        difference = prediction - closest_true_change # This should be positive for delayed predictions
        delays.append(difference)

    return np.mean(delays)



def partition_from_cps(locations, n_obs):
    """Return a list of sets that give a partition of the set [0, T-1], as
    defined by the change point locations.
    """
    T = n_obs
    partition = []
    current = set()

    all_cps = iter(sorted(set(locations)))
    cp = next(all_cps, None)
    for i in range(T):
        if i == cp:
            if current:
                partition.append(current)
            current = set()
            cp = next(all_cps, None)
        current.add(i)
    partition.append(current)
    return partition


def overlap(A, B):
    """
    Return the overlap (i.e. Jaccard index) of two sets
    """
    return len(A.intersection(B)) / len(A.union(B))


def cover_single(S, Sprime):
    """
    Compute the covering of a segmentation S by a segmentation Sprime.

    This follows equation (8) in Arbaleaz, 2010.
    """
    T = sum(map(len, Sprime))
    assert T == sum(map(len, S))
    C = 0
    for R in S:
        C += len(R) * max(overlap(R, Rprime) for Rprime in Sprime)
    C /= T
    return C


def covering(annotations, predictions, n_obs):
    """
    Compute the average segmentation covering against the human annotations.

    annotations : dict from user_id to iterable of CP locations
    predictions : iterable of predicted Cp locations
    n_obs : number of observations in the series

    """
    Ak = {
        k + 1: partition_from_cps(annotations[uid], n_obs)
        for k, uid in enumerate(annotations)
    }
    pX = partition_from_cps(predictions, n_obs)

    Cs = [cover_single(Ak[k], pX) for k in Ak]
    return sum(Cs) / len(Cs)


def f_measure(annotations, predictions, margin=5, alpha=0.5, return_PR=False):
    """
    Compute the F-measure based on human annotations.

    annotations : dict from user_id to iterable of CP locations
    predictions : iterable of predicted CP locations
    alpha : value for the F-measure, alpha=0.5 gives the F1-measure
    return_PR : whether to return precision and recall too
    """
    # ensure 0 is in all the sets
    Tks = {k + 1: set(annotations[uid]) for k, uid in enumerate(annotations)}
    for Tk in Tks.values():
        Tk.add(0)

    X = set(predictions)
    X.add(0)

    Tstar = set()
    for Tk in Tks.values():
        for tau in Tk:
            Tstar.add(tau)

    K = len(Tks)

    P = len(true_positives(Tstar, X, margin=margin)) / len(X)

    TPk = {k: true_positives(Tks[k], X, margin=margin) for k in Tks}
    R = 1 / K * sum(len(TPk[k]) / len(Tks[k]) for k in Tks)

    F = P * R / (alpha * R + (1 - alpha) * P)
    if return_PR:
        return F, P, R
    return F

def scoring_function(params):
    data_path = params["data"]
    true_changes, X = load_data(data_path)
    true_changes_dict = {1 : true_changes}
    detector = modular_detector(batch_size = params["batch size"], threshold=params['threshold'], max_dist_size=0, dist_metric = params["metric"], new_dist_buffer_size=params["batch size"]*3)
    locations = detector.detect(X)
    return f_measure(true_changes_dict, locations, margin = params["batch size"]//2) + covering(true_changes_dict, locations, X.shape[0])

def f(params):
    f1_cover = scoring_function(params)
    return {"loss": -1*f1_cover, "status":STATUS_OK}




distance_measures = [distance.acc, distance.add_chisq, distance.bhattacharyya, distance.braycurtis, distance.canberra, 
                     distance.chebyshev, distance.chebyshev_min, distance.clark, distance.correlation_pearson, distance.czekanowski, 
                     distance.divergence, distance.euclidean, distance.google, distance.gower, distance.hellinger, 
                     distance.jeffreys, distance.jensenshannon_divergence, distance.jensen_difference, distance.k_divergence, distance.kl_divergence, 
                     distance.kulczynski, distance.lorentzian, distance.manhattan,  distance.matusita, distance.max_symmetric_chisq, 
                     distance.minkowski, distance.motyka, distance.neyman_chisq, distance.nonintersection, distance.pearson_chisq, 
                     distance.penroseshape, distance.soergel, distance.squared_chisq, distance.squaredchord, distance.taneja, 
                     distance.tanimoto, distance.topsoe, distance.vicis_symmetric_chisq, distance.vicis_wave_hedges, distance.wave_hedges]
# Note: Cosine distance does not work, dice distance does not work, jaccard distance does not work
# kumarjohnson distance does not work, maryland bridge distance does not work, squared euclidean does not work



def main(data_path = None, batch_size = 3):
    true_changes, X = load_data(data_path)
    true_changes_dict = {1 : true_changes}
    index = 0
    for metric in distance_measures:

        #threshold tuning
        param_space = {"threshold": hp.uniform("threshold", 0.6, 1.25),
                       "metric": metric,
                       "batch size": batch_size,
                       "data": data_path}
        trials = Trials()
        results = fmin(f, param_space, algo=tpe.suggest, max_evals=100, trials=trials, loss_threshold=-1.9)

        # running best model
        detector = modular_detector(batch_size = batch_size, threshold=results["threshold"], max_dist_size=0, dist_metric = metric, new_dist_buffer_size=batch_size*3)
        start = time.time()
        locations = detector.detect(X)
        end = time.time()
        print("Distance metric index:", index)
        index += 1
        print("Time to run:", end-start)
        print(true_changes)
        print(locations)
 
        covering_score = covering(true_changes_dict, locations, X.shape[0])
        f1_score = f_measure(true_changes_dict, locations, margin = batch_size//2)
        accuracy_score = accuracy_estimation(true_changes, locations)


        print("Covering score:", covering_score)
        print("F1 score:", f1_score)
        print("Accuracy score:", accuracy_score)
        print("\n")
        print("\n")




main(data_path = "", batch_size=1000)






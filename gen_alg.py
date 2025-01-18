import numpy as np
from statistics import fmean
import multiprocessing
import math
from functools import reduce
from typing import Dict, Callable
import time


class DistanceException(Exception):
    def __init__(self, len_v1, len_v2):
        self.l1 = len_v1
        self.l2 = len_v2

    def __str__(self):
        return f"The vectors have different lengths: {self.l1=}, {self.l2=}"


class GenerationException(Exception):
    def __str__(self):
        return "It is impossible to create a generation"


class GeneticAlgorithm:
    def is_tolerance(self, individual_1: list, individual_2: list):
        """
        Verification for compliance with the tolerance relationship of two individuals
        :param individual_1: list
            An individual to check
        :param individual_2: list
            An individual to check
        :return: bool
            True if the individuals are tolerant, False otherwise
        """
        if self.d(individual_1[0:self.gen_len-1], individual_2[0:self.gen_len-1]) < self.alpha:
            return True
        return False

    def __init__(self,
                 gen_len: int,
                 population_size: int,
                 epoch_count: int,
                 individual_create_fun: Callable,
                 fit_fun: Callable,
                 individual_mutation_fun: Callable,
                 args: Dict = None,
                 epsilon: float = 0.4,
                 min_epsilon: float = 0.1,
                 alpha: float = 0.01,
                 leaders_ration: float = 0.05,
                 survivors_ratio: float = 0.55,
                 new_individual_ratio: float = 0.1,
                 n_jobs: int = 1,
                 crossing_operator: str = 'random',
                 p: float = 0.5
                 ):

        """
        :param gen_len: int
            The length of the gene.
        :param population_size: int
            Number of individuals in the population.
        :param epoch_count: int
            Number of epochs.
        :param individual_create_fun: function,
            А function for creating a new individual.
        :param fit_fun: function,
            A function for calculating the survival rate of an individual.
        :param args: dict, optional
            Args for fit function
        :param individual_mutation_fun: function
            A function for adding mutations.
        :param epsilon: float
            The probability of mutation.
        :param min_epsilon: float
            Minimum of epsilon.
        :param alpha: float
            Tolerance coefficient. The higher the coefficient, the greater
            the Cartesian distance between different individuals will be.
        :param leaders_ration: float
            proportion of the strongest individuals who will pass on to the
            next generation.
        :param survivors_ratio: float
            The proportion of individuals that will be randomly selected into
            the next generation.
        :param new_individual_ratio: float
            The proportion of new individuals in a new generation.
        :param n_jobs: int
            Number of processes.
        :param crossing_operator: str
            'uniform' - a new individual receives random genes from its parents.
            'mean' - the genes of a new individual are the average value between the genes of its parents.
            'random'- the choice of a crossing operator for each new individual is made randomly.
        :param p: float:
            The probability with which the crossing operator is 'uniform', otherwise 'mean'.
        """

        self.gen_len = gen_len
        self.population_size = population_size
        self.current_generation = 0
        self.epoch_count = epoch_count

        self.get_individual = individual_create_fun
        self.fit = fit_fun
        self.args = args
        self.get_mutation = individual_mutation_fun

        self.epsilon0 = epsilon
        self.epsilon = epsilon
        self.min_eps = min_epsilon
        self.alpha0 = alpha
        self.alpha = alpha

        self.generation = []
        self.individuals_for_calculating = multiprocessing.Queue()
        self.result_calculating = multiprocessing.Queue()

        self.max_fit = [0.0]
        self.mean_fit = [0.0]
        self.xs = [0]

        self.leaders_ration = leaders_ration
        self.survivors_ratio = survivors_ratio
        self.new_individual_ratio = new_individual_ratio

        self.n_jobs = n_jobs
        self.jobs = None

        self.crossing_operator = crossing_operator
        self.p = p

        self.check_params()

    def check_params(self):
        assert self.gen_len > 0, 'gen_len < 1'
        assert self.epoch_count > 0, 'epoch_count < 1'
        assert 0 <= self.epsilon <= 1, 'Epsilon must be > 0'
        assert self.alpha > 0, 'Alpha must be > 0'
        assert 0 <= self.p <= 1, 'Probability p must be in [0, 1]'
        assert self.crossing_operator in ['uniform', 'mean', 'random'],\
            'crossing_operator must take a value from the list ["uniform", "mean", "random"]'
        assert self.n_jobs > -1, 'Incorrect value n_jobs'
        assert int(self.population_size * self.new_individual_ratio) > 0, 'The number of new individuals was less than 1'
        assert int(self.population_size * self.survivors_ratio) > 0, 'The number of survivors was less than 1'
        assert int(self.population_size * self.leaders_ration) > 0, 'The number of leaders was less than 1'

    def fill_generation(self):
        """
        Filling of individuals up to the specified number (self.population_size)
        """
        attempt = 0

        added_individuals = []

        while len(self.generation) + len(added_individuals) < self.population_size:
            individual_value = self.get_individual()

            # Check whether a new individual does not enter into a tolerance relationship with already created individuals
            for individual in self.generation + added_individuals:
                if self.is_tolerance(individual, individual_value):
                    break
            else:
                self.individuals_for_calculating.put(individual_value)
                added_individuals.append(individual_value)
                attempt = 0
                continue

            attempt += 1

            # If it takes a long time to create an individual, then raise an exception.
            if attempt > 100 * self.population_size:
                raise GenerationException

        # Calculate fit-function
        result = self.wait_result(len(added_individuals))
        self.generation += result

        # Sorting by fit function value
        self.generation.sort(key=lambda a: a[self.gen_len], reverse=True)

    def uniform_crossing(self, individual_1: list, individual_2: list) -> [list, list]:
        """
        Uniform crossing operator.
        :param individual_1: list
            An individual for crossing.
        :param individual_2: list
            An individual for crossing.
        :return: list[list, list]:
            Children.
        """

        mask = [np.random.randint(0, 2) for _ in range(self.gen_len)]

        # Get children
        child_1 = [individual_1[i] if mask[i] else individual_2[i] for i in range(self.gen_len)]
        child_2 = [individual_1[i] if not mask[i] else individual_2[i] for i in range(self.gen_len)]

        # Added mutations
        child_1 = self.get_mutation(child_1, self.epsilon)
        child_2 = self.get_mutation(child_2, self.epsilon)

        return child_1, child_2

    @staticmethod
    def is_numeric(obj):
        try:
            obj + 0
            return True
        except TypeError:
            return False

    @staticmethod
    def d(v1, v2):
        """
        Cartesian distance
        :param v1: list
            individual 1
        :param v2: list
            individual 2
        :return: float
        """
        if len(v1) != len(v2):
            raise DistanceException(len(v1), len(v2))
        return math.sqrt(sum((v1[i] - v2[i]) ** 2 for i in range(len(v1))))

    def mean_crossing(self, individual_1: list, individual_2: list) -> list:
        """
        Mean crossing operator.
        :param individual_1: list
            An individual for crossing.
        :param individual_2: list
            An individual for crossing.
        :return: list:
            Child.
        """

        mask = [self.is_numeric(individual_gen) for individual_gen in individual_1]

        # Get child
        child = [(individual_1[i] + individual_2[i]) / 2 if mask[i] else individual_1[i] for i in range(self.gen_len)]

        # Added mutations
        child = self.get_mutation(child, self.epsilon)
        return child

    def get_survivors_individuals(self) -> [list, list]:
        """
        Get individuals who have passed on to a new generation

        :return: [list, list]
            new_generation
        """
        # Save the best individuals
        last_leaders_index = int(self.population_size * self.leaders_ration)
        new_generation = self.generation[:last_leaders_index]

        # Get the probability of survival
        F_survive = reduce(lambda a, x: a + x[self.gen_len], self.generation[last_leaders_index:], 0)
        p_survive = [f[self.gen_len] / F_survive for f in self.generation[last_leaders_index:]]

        # Add survival individuals
        survivors_count = int(self.population_size * self.survivors_ratio)

        survivors = np.random.choice(
            range(last_leaders_index, self.population_size),
            size=survivors_count,
            p=p_survive,
            replace=False
        )
        new_generation += [self.generation[i] for i in survivors]

        return new_generation

    def wait_result(self, tasks_count):
        """
        Getting calculation results
        :param tasks_count: int
        :return: list
        """
        result = []
        if self.n_jobs in [0, 1]:
            for _ in range(tasks_count):
                task = self.individuals_for_calculating.get()
                if self.args is not None:
                    result_fit = self.fit(task, self.args)
                else:
                    result_fit = self.fit(task)
                result.append(task + [result_fit])

        else:
            while self.result_calculating.qsize() < tasks_count:
                continue

            while not self.result_calculating.empty():
                result.append(self.result_calculating.get())

        return result

    def crossing(self, new_generation):
        # Get the probability of crossing
        F_cross = reduce(lambda a, x: a + x[self.gen_len], self.generation, 0)
        p_cross = [f[self.gen_len] / F_cross for f in self.generation]

        new_count = int(self.population_size * self.new_individual_ratio)

        new_individuals_count = 0
        # Скрещивание
        while len(new_generation) + new_individuals_count < self.population_size - new_count:
            parents = np.random.choice(
                range(self.population_size),
                size=2,
                p=p_cross,
                replace=False
            )
            if self.crossing_operator == 'uniform' or (self.crossing_operator == 'random' and np.random.rand() < self.p):
                children = self.uniform_crossing(self.generation[parents[0]], self.generation[parents[1]])

                self.individuals_for_calculating.put(children[0])
                self.individuals_for_calculating.put(children[1])
                new_individuals_count += 2

            else:
                child = self.mean_crossing(self.generation[parents[0]], self.generation[parents[1]])
                self.individuals_for_calculating.put(child)
                new_individuals_count += 1

        return self.wait_result(new_individuals_count)

    def create_new_generation(self):
        new_generation = self.get_survivors_individuals()

        crossing_individuals = self.crossing(new_generation)

        self.generation = new_generation[:] + crossing_individuals[:]

        self.fill_generation()

        self.generation.sort(key=lambda a: a[self.gen_len], reverse=True)

    def new_eps(self):
        return self.epsilon0 - ((self.epsilon0 - self.min_eps) / self.epoch_count) * self.current_generation

    def update(self):
        self.current_generation += 1

        self.epsilon = self.new_eps()

        self.max_fit.append(self.generation[0][self.gen_len])
        self.mean_fit.append(fmean([individual[self.gen_len] for individual in self.generation]))

    def stopping_criteria(self) -> bool:
        if abs(self.max_fit[-1] - self.mean_fit[-1]) < 0.001:
            return True
        if self.current_generation > 30 and abs(self.max_fit[-1] - fmean(self.max_fit[-30:])) < 0.001:
            return True
        return False

    @staticmethod
    def worker(tasks, result_queue, fit_function, args=None):
        while True:
            task = tasks.get()
            if task == 'STOP':
                break
            if args is not None:
                result = fit_function(task, args)
            else:
                result = fit_function(task)
            result_queue.put(task + [result])

    def create_jobs(self):
        if self.n_jobs in [0, 1]:
            return

        self.jobs = []

        if self.n_jobs == -1:
            self.n_jobs = multiprocessing.cpu_count()

        for _ in range(self.n_jobs):
            job = multiprocessing.Process(
                target=self.worker,
                args=(
                      self.individuals_for_calculating,
                      self.result_calculating,
                      self.fit,
                      self.args,
                    )
                )
            self.jobs.append(job)
            job.start()

    def kill_jobs(self):
        if self.n_jobs in [1, 0]:
            return
        for _ in range(self.n_jobs):
            self.individuals_for_calculating.put('STOP')
        for job in self.jobs:
            job.join()

    def start(self):
        self.create_jobs()

        self.fill_generation()

        for i in range(self.epoch_count):
            s = time.time()

            self.create_new_generation()

            self.update()

            print(
                f'epoch {i}\t mean: {self.mean_fit[-1]}\t '
                f'max: {self.max_fit[-1]}\t '
                f'leader: {self.generation[0][:self.gen_len]}\t '
                f'time: {str(time.time() - s)[:4]}'
            )

            if self.stopping_criteria():
                break

        self.kill_jobs()

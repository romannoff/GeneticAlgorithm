from gen_alg import GeneticAlgorithm, GenerationException
import numpy as np
import matplotlib.pyplot as plt


def fit(individual):
    """
    Area calculation function
    """
    return np.pi * (individual[2]**2 + individual[5]**2 + individual[8]**2)


def in_square(circle):
    """
    Сheck that the circle lies inside the square.
    """
    if not 0 <= circle[0] + circle[2] <= 1 \
            or not 0 <= circle[1] + circle[2] <= 1 \
            or not 0 <= circle[0] - circle[2] <= 1 \
            or not 0 <= circle[1] - circle[2] <= 1:
        return False
    return True


def is_intersect(circle_1, circle_2):
    """
    Check that the circles have no intersections.
    """
    if d(circle_1[:2], circle_2[:2]) < circle_1[2] + circle_2[2]:
        return True
    return False


def d(v1, v2):
    """
    Cartesian distance
    """
    return np.sqrt(sum((v1[i] - v2[i])**2 for i in range(len(v1))))


def circle_in_square():
    """
    Get circle in square
    """
    r = np.random.rand() / 2

    x, y = np.random.uniform(r, 1 - r, size=2)

    return [x, y, r]


def get_individual():
    """
    Setting 3 circles

    gen: x_1, y_1, r_1, x_2, y_2, r_2, x_3, y_3, r_3

    circle_1: x_1, y_1, r_1
    circle_2: x_2, y_2, r_2
    circle_3: x_3, y_3, r_3

    """
    circle_1, circle_2, circle_3 = None, None, None

    for _ in range(10_000):

        if circle_1 is None:
            circle_1 = circle_in_square()

        if circle_2 is None:
            circle_2 = circle_in_square()

        if not in_square(circle_2) or is_intersect(circle_1, circle_2):
            circle_2 = None
            continue

        circle_3 = circle_in_square()

        if not in_square(circle_3) or is_intersect(circle_2, circle_3) or\
                is_intersect(circle_1, circle_3):
            continue

        return circle_1 + circle_2 + circle_3

    raise GenerationException


def mutation(individual, eps):
    """
    Change a random element and check for correctness
    """
    circle_1 = individual[:3]
    circle_2 = individual[3:6]
    circle_3 = individual[6:]

    for _ in range(10_000_000):
        circle_1_ = [value if np.random.rand() > eps else np.random.rand() for value in circle_1]

        if not in_square(circle_1_):
            continue

        circle_2_ = [value if np.random.rand() > eps else np.random.rand() for value in circle_2]

        if not in_square(circle_2_):
            continue

        circle_3_ = [value if np.random.rand() > eps else np.random.rand() for value in circle_3]

        if not in_square(circle_3_):
            continue

        if is_intersect(circle_1_, circle_2_) or \
                is_intersect(circle_2_, circle_3_) or\
                is_intersect(circle_1_, circle_3_):
            continue

        return circle_1_ + circle_2_ + circle_3_

    raise GenerationException


if __name__ == '__main__':
    population_size_ = 100
    percentage_leaders_ = 0.1
    percentage_survivors_ = 0.2
    percentage_new_ = 0.5

    env = GeneticAlgorithm(
        gen_len=9,
        population_size=population_size_,
        fit_fun=fit,
        individual_create_fun=get_individual,
        individual_mutation_fun=mutation,
        epoch_count=500,
        epsilon=0.3,
        min_epsilon=0.01,
        alpha=0,
        leaders_ration=percentage_leaders_,
        survivors_ratio=percentage_survivors_,
        new_individual_ratio=percentage_new_,
        n_jobs=10,
        crossing_operator='random',
        p=0.7,
    )

    env.start()

    best_individual = env.generation[0]

    # Plot graph

    figure, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].set_aspect(1)

    artist_list = []

    axes[1].plot(list(range(len(env.max_fit))), env.max_fit, color='g', label='max line')[0]
    axes[1].plot(list(range(len(env.mean_fit))), env.mean_fit, color='b', label='mean line')[0]

    axes[1].legend()

    circle_1_plt = plt.Circle((best_individual[0], best_individual[1]), best_individual[2], alpha=0.5, color='#ff7f0e')
    circle_2_plt = plt.Circle((best_individual[3], best_individual[4]), best_individual[5], alpha=0.5, color='#2ca02c')
    circle_3_plt = plt.Circle((best_individual[6], best_individual[7]), best_individual[8], alpha=0.5, color='#9467bd')

    axes[0].add_artist(circle_1_plt)
    axes[0].add_artist(circle_2_plt)
    axes[0].add_artist(circle_3_plt)

    plt.show()

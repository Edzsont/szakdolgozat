
def standardize(x, mean, std):
    return (x - mean) / std


def feature_scale(x, x_min, x_max):
    return 2 * (x - x_min)/(x_max - x_min) - 1


def inv_minmax(x, x_min, x_max):
    return ((x + 1) / 2) * (x_max - x_min) + x_min


def inv_standardize(x, mean, std):
    return std * x + mean


def feature_stand_def(y, version, MEANS, STDS, func):
    means = []
    stds = []
    for v in version:
        means.append(MEANS[v])
        stds.append(STDS[v])

    for idx in range(len(means)):
        y[:, idx] = func(y[:, idx], means[idx], stds[idx])  # (y[:, idx] - means[idx]) / stds[idx]

    return y


def feature_standardize(y, version, MEANS, STDS):
    return feature_stand_def(y, version, MEANS, STDS, standardize)


def inv_feature_standardize(y, version, MEANS, STDS):
    return feature_stand_def(y, version, MEANS, STDS, inv_standardize)

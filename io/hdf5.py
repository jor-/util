import h5py

def load(file, name):
    with h5py.File(file, mode='r') as file:
        return file[name].value

def save(file, name, value):
    with h5py.File(file, mode='r+') as file:
        file[name][()] = value


def load_all_datasets(file, groupname):
    results = []

    with h5py.File(file, mode='r') as file:
        group = file[groupname]
        for dataset in group.keys():
            results.append(group[dataset].value)

    return results

from ocpmodels.datasets import TrajectoryLmdbDataset, SinglePointLmdbDataset

# TrajectoryLmdbDataset is our custom Dataset method to read the lmdbs as Data objects. Note that we need to give the path to the folder containing lmdbs for S2EF
dataset = TrajectoryLmdbDataset({"src": "data/s2ef/train_100/"})

print("Size of the dataset created:", len(dataset))
print(dataset[0])

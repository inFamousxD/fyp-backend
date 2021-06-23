from Models import *
from Datasets import *

import os
import numpy as np
import math

from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch
from sklearn.metrics import accuracy_score
import random
from sklearn.cluster import KMeans

# Tuneable hyperparameters
learning_rate = 2e-5
epochs = 15
number_of_clusters = 400
batch_size = 32
seed = 2027534574692550828

csv_file = "./data/malware.csv"
latent_dim = 215

random.seed(seed)
torch.manual_seed(seed)


mgd = MalwareGenomeDataset(csv_file=csv_file, latent_dim=latent_dim)
train_size = int(0.7 * len(mgd))
val_size = len(mgd) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(mgd, [train_size, val_size])


def get_class_wise_indices(dataset, label):
    # 1 -> Malware
    # 0 -> Benign
    indices = []
    for i in range(len(dataset)):
        if dataset[i]["label"] == label:
            indices.append(i)
    return indices


def refresh_csv(csv_file="./data/malware.csv", latent_dim=215):
    random.seed(seed)
    torch.manual_seed(seed)
    mgd = MalwareGenomeDataset(csv_file=csv_file, latent_dim=latent_dim)
    train_size = int(0.7 * len(mgd))
    val_size = len(mgd) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        mgd, [train_size, val_size]
    )


def get_subset_labelwise(dataset, label):
    indices = get_class_wise_indices(dataset, label)
    return torch.utils.data.Subset(dataset, indices)


def cluster_similar_points(kmeans, feature_vectors, dataset):
    # returns closest benign centroids as the "real vector" in GAN
    return torch.tensor(
        [list(kmeans.cluster_centers_[i]) for i in kmeans.predict(feature_vectors)]
    )


def convert_string_list_to_csv(data):
    assert len(data[0]) == 216, "Data must be of length 216"
    output = ""
    with open("./data/permissions.txt") as f:
        output += f.read() + "\n"
    for datapoint in data:
        label = "S" if datapoint[-1] == "1" else "B"
        output += ",".join(list(datapoint[:-1])) + "," + label + "\n"
    with open("./data/output.csv", "w") as f:
        f.write(output)


def validate():
    malware_val = get_subset_labelwise(val_dataset, 1)
    malware_val_dataloader = DataLoader(
        malware_val, batch_size=batch_size, shuffle=False
    )
    original_samples = []
    generated_samples = []
    generator = Generator(latent_dim)

    generator.load_state_dict(torch.load("./saved_models/generator.pt"))
    generator.eval()
    with torch.no_grad():
        for i, datapoint in enumerate(malware_val_dataloader):
            original_samples += list(datapoint["feature_vector"])
            vector = torch.tensor(datapoint["feature_vector"], dtype=torch.float32)
            generated_sample_temporary = generator(vector)
            generated_sample = torch.max(vector, generated_sample_temporary)
            generated_samples += generated_sample

    final_original_samples = []
    final_generated_samples = []
    for original_sample, generated_sample in zip(original_samples, generated_samples):
        final_generated_samples.append(
            ["0" if i < 0.5 else "1" for i in generated_sample]
        )
        final_original_samples.append([str(int(i)) for i in original_sample])
    return final_original_samples, final_generated_samples


def evaluate(evaluation_data):
    convert_string_list_to_csv(evaluation_data)
    mgd = MalwareGenomeDataset(csv_file="./data/output.csv", latent_dim=latent_dim)
    eval_dataloader = DataLoader(mgd, batch_size=batch_size, shuffle=False)

    generated_samples = []
    generator = Generator(latent_dim)

    generator.load_state_dict(torch.load("./saved_models/generator.pt"))
    generator.eval()
    with torch.no_grad():
        for i, datapoint in enumerate(eval_dataloader):
            vector = torch.tensor(datapoint["feature_vector"], dtype=torch.float32)
            generated_sample_temporary = generator(vector)
            generated_sample = torch.max(vector, generated_sample_temporary)
            generated_samples += generated_sample
    final_generated_samples = []
    for generated_sample in generated_samples:
        final_generated_samples.append(
            ["0" if i < 0.5 else "1" for i in generated_sample]
        )
    return final_generated_samples


def train():

    malware_train = get_subset_labelwise(train_dataset, 1)
    malware_val = get_subset_labelwise(val_dataset, 1)

    benign_train = get_subset_labelwise(train_dataset, 0)
    benign_val = get_subset_labelwise(val_dataset, 0)

    benign_X = np.array([list(i["feature_vector"]) for i in benign_train])
    kmeans = KMeans(n_clusters=number_of_clusters).fit(benign_X)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    malware_train_dataloader = DataLoader(
        malware_train, batch_size=batch_size, shuffle=True
    )
    malware_val_dataloader = DataLoader(
        malware_val, batch_size=batch_size, shuffle=False
    )

    adversarial_loss = torch.nn.BCELoss()

    # Initialize generator and discriminator
    generator = Generator(latent_dim)
    discriminator = Discriminator(latent_dim)

    if torch.cuda.is_available():
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=learning_rate)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)
    print("GAN Training...")

    for epoch in range(epochs):
        for i, datapoint in enumerate(train_dataloader):
            optimizer_D.zero_grad()
            vector = torch.tensor(datapoint["feature_vector"], dtype=torch.float32)
            label = torch.tensor(datapoint["label"], dtype=torch.float32).reshape(-1, 1)
            out = discriminator(vector)
            loss = adversarial_loss(out, label)
            loss.backward()
            optimizer_D.step()

    generator.train()
    discriminator.train()
    for epoch in range(epochs):
        for i, datapoint in enumerate(malware_train_dataloader):
            optimizer_G.zero_grad()

            z = datapoint["feature_vector"].type(torch.FloatTensor)
            valid_centroids = cluster_similar_points(kmeans, z, benign_train).type(
                torch.FloatTensor
            )

            valid = torch.ones((z.shape[0], 1))
            fake = torch.zeros((z.shape[0], 1))

            generated_sample_temporary = generator(z)
            generated_sample = torch.max(z, generated_sample_temporary)

            g_loss = adversarial_loss(discriminator(generated_sample), valid)
            g_loss.backward()
            optimizer_G.step()

            # optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            # real_loss = adversarial_loss(discriminator(valid_centroids), valid)
            # fake_loss = adversarial_loss(discriminator(generated_sample), fake).detach()
            # d_loss = (real_loss+ fake_loss)/2
            # # print(real_loss,fake_loss,d_loss)
            # d_loss.backward()
            # optimizer_D.step()
    torch.save(generator.state_dict(), "./saved_models/generator.pt")
    torch.save(discriminator.state_dict(), "./saved_models/discriminator.pt")
    print("Training Complete!")

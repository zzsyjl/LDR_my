from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, MNIST
import matplotlib.pyplot as plt
from torch import nn
import argparse
import os

from mcrgan.default import update_config
from mcrgan.models import get_models
from mcrgan.default import _C as config
from test_acc import parse_args


def generate_subset(dataset_name, root_dir='./data', train=True, transform=None, num_samples_per_class=200):
    # Load the full dataset
    if dataset_name.lower() == 'cifar10':
        full_dataset = CIFAR10(root=root_dir, train=train,
                               download=True, transform=transform)
    elif dataset_name == 'mnist':
        full_dataset = MNIST(root=root_dir, train=train,
                             download=True, transform=transform)

    # Filter the dataset to include only num_samples_per_class samples per class
    subset_data = []
    subset_targets = []
    num_samples_seen = [0] * 10  # One counter per class
    i = 0
    for data, target in full_dataset:
        i += 1
        if num_samples_seen[target] < num_samples_per_class:
            subset_data.append(data)
            subset_targets.append(target)
            num_samples_seen[target] += 1
        if all(count == num_samples_per_class for count in num_samples_seen):
            break  # Stop once we have enough samples for all classes
    print('after loop', i, 'times, subset constr')
    # sort the subset_data and subset_targets

    sorted_indices = torch.argsort(torch.tensor(subset_targets))
    subset_data = torch.stack(subset_data, dim=0)[sorted_indices]
    subset_targets = torch.tensor(subset_targets)[sorted_indices]
    subset_dataset = torch.utils.data.TensorDataset(
        subset_data, subset_targets)

    return subset_dataset

def get_features(model, dataset):
    # Set model to evaluation mode
    model.eval()

    # Create a PyTorch DataLoader object for the dataset
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=32)

    # Extract features from the dataset using the model
    features = []
    with torch.no_grad():
        for x, y in data_loader:
            x = x.cuda()
            batch_features = model(x)
            features.append(batch_features.detach().cpu().numpy())
    features = np.concatenate(features, axis=0)
    return features

def anchor_CSM(features, num_classes, abs=False):
    # calculate the class-specific mean of the features
	# features: (N, d)
    print(features.shape)
    num_samples_per_class = len(features) // num_classes
    means = [np.mean(features[i*num_samples_per_class:(i+1)*num_samples_per_class], axis=0) \
        for i in range(num_classes)]
    means = np.array(means)
    # get the demean features for each class
    demean_features = [features[i*num_samples_per_class:(i+1)*num_samples_per_class] - means[i] \
                    for i in range(num_classes)]  
    demean_features = np.concatenate(demean_features, axis=0)
    print(demean_features.shape)
    # calculate the anchor CSM
    anchor_CSM = np.zeros((len(features), len(features)))
    for i in range(num_classes):
        anchor_CSM[i*num_samples_per_class:(i+1)*num_samples_per_class] = \
            cosine_similarity(demean_features[i*num_samples_per_class:(i+1)*num_samples_per_class], features-means[i])
        # demean_features[i*num_samples_per_class:(i+1)*num_samples_per_class].dot((features-means[i]).T)
    if abs:
        anchor_CSM = np.abs(anchor_CSM)
    return anchor_CSM

def calculate_cosine_similarity(features, demean=False, abs=False):
    
    if demean:
        features = features - features.mean(axis=0)

    # Calculate cosine similarity between pairs of features
    similarity_matrix = cosine_similarity(features)
    if abs:
        similarity_matrix = np.abs(similarity_matrix)
    return similarity_matrix

def plot_cosine_similarity_matrix(subset_similarity_matrix, pic_tag):
    # Display the cosine similarity matrix as an image
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(subset_similarity_matrix, cmap='PuBu')

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel('Cosine similarity', rotation=270, labelpad=20)

    # Add title and axis labels
    ax.set_title('Cosine similarity matrix')
    ax.set_xlabel('Samples')
    ax.set_ylabel('Samples')

    # save the figure
    plt.savefig(f'CSM_{pic_tag}.png')

    

if __name__ == '__main__':
    args = parse_args()
    root = os.path.dirname(args.cfg)
    netd_ckpt = f"{root}/checkpoints/netD/netD_{args.ckpt_epochs}_steps.pth"

    transform = transforms.Compose(
            [transforms.Resize(32),
             transforms.ToTensor(),
             transforms.Normalize(0.5, 0.5)])
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    subset_dataset = generate_subset(dataset_name=config.DATA.DATASET, train=True, transform=transform, num_samples_per_class=200)

    netD, _ = get_models(config.DATA.DATASET, device)
    netD_state_dict = torch.load(netd_ckpt)
    netD.module.load_state_dict(netD_state_dict["model_state_dict"])
    netD.cuda()

    # Calculate the cosine similarity matrix for the subset dataset
    features = get_features(netD, subset_dataset)
    similarity_matrix = calculate_cosine_similarity(features, demean=True, abs=True)

    print(np.histogram(similarity_matrix, bins=10))

    # Plot the cosine similarity matrix
    plot_cosine_similarity_matrix(similarity_matrix, f'{config.DATA.DATASET}_ictrl')




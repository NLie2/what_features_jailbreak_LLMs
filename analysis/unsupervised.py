import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import datetime
from sklearn.decomposition import PCA


def visualize_act_with_t_sne(model_name, tensors_flat, labels, dataset_name, num_samples, layer, perplexity =30, scores=None, save = False):
  # Apply t-SNE
  print("t-SNE visualization of the weights of the residual stream of the model")
  tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
  print("Fitting t-SNE...", [tensor for tensor in tensors_flat])
  
  tensors_tsne = tsne.fit_transform(tensors_flat)

  # Check the shape of tensors_tsne to ensure it matches expected dimensions
  print("Shape of tensors_tsne:", tensors_tsne.shape)

  # Ensure that the number of labels matches the number of samples in tensors_tsne
  assert len(labels) == tensors_tsne.shape[0], "The number of labels must match the number of samples in tensors_tsne."

  # Plot the t-SNE results with different colors for different labels
  plt.figure(figsize=(10, 8))

  # Create a color map
  colors = plt.colormaps.get_cmap('Accent')  # Get the colormap without specifying number of colors

  # Map colors to unique occurrences in labels
  unique_labels = np.unique(labels)
  colors = cm.get_cmap('viridis', len(unique_labels))  # Choose a colormap
  color_map = {label: colors(i) for i, label in enumerate(unique_labels)}


  # Plot each label category with a different color
  for label in np.unique(labels):
      indices = np.where(labels == label)[0]

      if scores is not None:
        for idx in indices:
              base_color = np.array(color_map[label][:3])  # Extract RGB components
              score_adjusted_color = base_color * 0.5 + base_color * 0.5 * scores[idx]  # Adjust brightness
              alpha = 0.3 + 0.7 * scores[idx]  # Ensure minimum visibility for low scores
              plt.scatter(tensors_tsne[idx, 0], tensors_tsne[idx, 1], label=label if idx == indices[0] else "",
                          color=score_adjusted_color, alpha=alpha, s=100)
      else:
              plt.scatter(tensors_tsne[indices, 0], tensors_tsne[indices, 1], label=label,
                    color=color_map[label], alpha=0.5, s=100)  # Mapping scores to alpha


  plt.title(f"t-SNE Visualization of Residual Stream Weights, model={model_name}, layer={layer}, num_samples={num_samples}, perplexity={perplexity}")
  plt.xlabel("t-SNE Component 1")
  plt.ylabel("t-SNE Component 2")
  plt.figtext(0.5, 0.01, f"Dataset: {dataset_name}", ha='center', fontsize=10, style='italic')  # Update dataset name here
  plt.legend()


  plt.show()



def visualize_act_with_pca(model_name, tensors_flat, labels, dataset_name, num_samples, layer, scores=None, save=False):
    # Apply PCA
    pca = PCA(n_components=2)
    tensors_pca = pca.fit_transform(tensors_flat)

    # Check the shape of tensors_pca to ensure it matches expected dimensions
    # print("Shape of tensors_pca:", tensors_pca.shape)

    # Ensure that the number of labels matches the number of samples in tensors_pca
    assert len(labels) == tensors_pca.shape[0], "The number of labels must match the number of samples in tensors_pca."

    # Plot the PCA results with different colors for different labels
    plt.figure(figsize=(10, 8))

    # Create a color map
    colors = plt.colormaps.get_cmap('Accent')  # Get the colormap without specifying number of colors

    # Map colors to unique occurrences in labels
    unique_labels = np.unique(labels)
    colors = cm.get_cmap('viridis', len(unique_labels))  # Choose a colormap
    color_map = {label: colors(i) for i, label in enumerate(unique_labels)}

    # Plot each label category with a different color
    for label in np.unique(labels):
        indices = np.where(labels == label)[0]

        if scores is not None:
            for idx in indices:
                base_color = np.array(color_map[label][:3])  # Extract RGB components
                score_adjusted_color = base_color * 0.5 + base_color * 0.5 * scores[idx]  # Adjust brightness
                alpha = 0.3 + 0.7 * scores[idx]  # Ensure minimum visibility for low scores
                plt.scatter(tensors_pca[idx, 0], tensors_pca[idx, 1], label=label if idx == indices[0] else "",
                            color=score_adjusted_color, alpha=alpha, s=100)
        else:
            plt.scatter(tensors_pca[indices, 0], tensors_pca[indices, 1], label=label,
                        color=color_map[label], alpha=0.5, s=100)  # Mapping scores to alpha

    plt.title(f"PCA Visualization of Residual Stream Weights, model={model_name}, layer={layer}, num_samples={num_samples}")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.figtext(0.5, 0.01, f"Dataset: {dataset_name}", ha='center', fontsize=10, style='italic')  # Update dataset name here
    plt.legend()

    if save:
        now = datetime.now().strftime("%d%m%Y%H:%M:%S")
        save_fig_path = f'/content/gdrive/My Drive/ERA_Fellowship/progress_logs/{dataset_name}_{now}_pca_visualization.png'  # Replace with your desired path
        plt.savefig(save_fig_path)

    plt.show()
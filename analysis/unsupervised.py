import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import datetime
import plotly.express as px
from sklearn.decomposition import PCA
from collections import defaultdict
from adjustText import adjust_text

fontsize1 = 34
fontsize2 = 24
fontsize3 = 30

def visualize_act_with_t_sne(model_name, tensors_flat, labels, dataset_name, num_samples, layer, perplexity=30, scores=None, save=False):
    # Apply t-SNE
    print("t-SNE visualization of the weights of the residual stream of the model")
    
    # labels = [label.replace("_", " ").title().replace("cg", "CG") for label in labels]
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
    unique_labels = np.unique(labels)
    
    colors = cm.get_cmap('tab20', len(unique_labels))  # Use 'tab20' or 'Dark2' colormap for distinct, non-pastel colors
    color_map = {label: colors(i) for i, label in enumerate(unique_labels)}

    # Store text objects to adjust later
    texts = []

    # Plot each label category with a different color and annotate labels
    for label in unique_labels:

        indices = np.where(labels == label)[0]
        
        plt.scatter(tensors_tsne[indices, 0], tensors_tsne[indices, 1], color=color_map[label], alpha=0.6, s=100, label=label)

        # Annotate the label
        mean_x = np.mean(tensors_tsne[indices, 0])
        mean_y = np.mean(tensors_tsne[indices, 1])

        # Store the text object for adjustment
        text = plt.text(mean_x, mean_y, str(label), fontsize=fontsize2, ha='center', va='center',
                        bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=1))
        texts.append(text)

    # Adjust text to avoid overlap
    adjust_text(texts, only_move={'points': 'y', 'text': 'y'}, arrowprops=dict(arrowstyle="->", color='r', lw=0.5))

    # plt.title(f"t-SNE Visualization of Residual Stream Weights, model={model_name}, layer={layer}, num_samples={num_samples}, perplexity={perplexity}")
    plt.xlabel("T-SNE Component 1", fontsize=fontsize2)  # Increased fontsize
    plt.ylabel("T-SNE Component 2", fontsize=fontsize2)  # Increased fontsize
    plt.figtext(0.5, 0.01, f"{model_name.replace("-", " ").title()} (Layer {layer})", ha='center', fontsize=fontsize1, style='italic')
    
    #remove x_ticks
    plt.xticks([])
    plt.yticks([])
    
    plt.tight_layout(pad=5.0)

    if save:
        save_fig_path = f'/data/nathalie_maria_kirch/ERA_Fellowship/images/clustering/tsne_visualization_{dataset_name.replace('/', '_')}_{model_name.replace('/', '_')}.pdf'
        plt.savefig(save_fig_path, bbox_inches='tight')  # Save the figure with labels directly on the plot
   
    return plt
    
# def visualize_act_with_t_sne(model_name, tensors_flat, labels, dataset_name, num_samples, layer, perplexity =30, scores=None, save = False):
#   # Apply t-SNE
#   print("t-SNE visualization of the weights of the residual stream of the model")
#   tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
#   print("Fitting t-SNE...", [tensor for tensor in tensors_flat])
  
#   tensors_tsne = tsne.fit_transform(tensors_flat)

#   # Check the shape of tensors_tsne to ensure it matches expected dimensions
#   print("Shape of tensors_tsne:", tensors_tsne.shape)

#   # Ensure that the number of labels matches the number of samples in tensors_tsne
#   assert len(labels) == tensors_tsne.shape[0], "The number of labels must match the number of samples in tensors_tsne."

#   # Plot the t-SNE results with different colors for different labels
#   plt.figure(figsize=(10, 8))

#   # Create a color map
#   # colors = plt.colormaps.get_cmap('Accent')  # Get the colormap without specifying number of colors
#   # Map colors to unique occurrences in labels
#   unique_labels = np.unique(labels)
#   # colors = cm.get_cmap('viridis', len(unique_labels))  # Choose a colormap
#   # colors = cm.get_cmap('rainbow', len(unique_labels))  # Choose a colormap
#   colors = cm.get_cmap('tab20', len(unique_labels))  # Choose the 'tab20' colormap
#   color_map = {label: colors(i) for i, label in enumerate(unique_labels)}


#   # Plot each label category with a different color
#   for label in np.unique(labels):
#       indices = np.where(labels == label)[0]

#       if scores is not None:
#         for idx in indices:
#               base_color = np.array(color_map[label][:3])  # Extract RGB components
#               score_adjusted_color = base_color * 0.5 + base_color * 0.5 * scores[idx]  # Adjust brightness
#               alpha = 0.3 + 0.7 * scores[idx]  # Ensure minimum visibility for low scores
#               plt.scatter(tensors_tsne[idx, 0], tensors_tsne[idx, 1], label=label if idx == indices[0] else "",
#                           color=score_adjusted_color, alpha=alpha, s=100)
#       else:
#               plt.scatter(tensors_tsne[indices, 0], tensors_tsne[indices, 1], label=label,
#                     color=color_map[label], alpha=0.5, s=100)  # Mapping scores to alpha


#   plt.title(f"t-SNE Visualization of Residual Stream Weights, model={model_name}, layer={layer}, num_samples={num_samples}, perplexity={perplexity}")
#   plt.xlabel("t-SNE Component 1")
#   plt.ylabel("t-SNE Component 2")
#   plt.figtext(0.5, 0.01, f"Dataset: {dataset_name}", ha='center', fontsize=10, style='italic')  # Update dataset name here
#   plt.legend()

#   if save:
#     now = datetime.datetime.now().strftime("%d%m%Y%H:%M:%S")
#     save_fig_path = f'tsne_visualization_{dataset_name}_{now}.pdf'.replace('/', '_')  # Replace with your desired path
#     plt.savefig(save_fig_path)

#   plt.show()
  
  
# def visualize_act_with_t_sne(model_name, tensors_flat, labels, dataset_name, num_samples, layer, perplexity=30, scores=None, save=False):
#     print("t-SNE visualization of the weights of the residual stream of the model")
#     tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
#     tensors_tsne = tsne.fit_transform(tensors_flat)

#     print("Shape of tensors_tsne:", tensors_tsne.shape)
#     assert len(labels) == tensors_tsne.shape[0], "The number of labels must match the number of samples in tensors_tsne."

#     plt.figure(figsize=(10, 8))

#     # Use a distinct color map
#     unique_labels = np.unique(labels)
#     colors = plt.get_cmap('tab20b').colors
#     if len(unique_labels) > 20:
#         additional_colors = plt.get_cmap('tab20c').colors
#         colors = colors + additional_colors
#     color_map = {label: colors[i % len(colors)] for i, label in enumerate(unique_labels)}

#     for label in unique_labels:
#         indices = np.where(labels == label)[0]
#         plt.scatter(tensors_tsne[indices, 0], tensors_tsne[indices, 1], label=label,
#                     color=color_map[label], alpha=0.7, s=100)

#     plt.title(f"t-SNE Visualization of Residual Stream Weights, model={model_name}, layer={layer}, num_samples={num_samples}, perplexity={perplexity}")
#     plt.xlabel("t-SNE Component 1")
#     plt.ylabel("t-SNE Component 2")
#     plt.figtext(0.5, 0.01, f"Dataset: {dataset_name}", ha='center', fontsize=10, style='italic')
#     plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    
#     if save:
#         now = datetime.datetime.now().strftime("%d%m%Y%H:%M:%S")
#         save_fig_path = f'tsne_visualization_{dataset_name}_{now}.pdf'.replace('/', '_')
#         plt.savefig(save_fig_path)

#     plt.show()

def visualize_act_with_t_sne_interactive(model_name, tensors_flat, labels, dataset_name, num_samples, layer, perplexity=30, scores=None, save=False):
    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    tensors_tsne = tsne.fit_transform(tensors_flat)

    # Create a DataFrame with t-SNE components and labels
    df = pd.DataFrame(tensors_tsne, columns=['Component 1', 'Component 2'])
    print("len",len(labels))
    print(labels)
    df['Label'] = labels

    # If scores are provided, include them in the DataFrame
    # if scores is not None:
    #     df['Score'] = scores
    #     size_column = 'Score'
    # else:
    #     size_column = None

    # Create an interactive scatter plot with Plotly
    fig = px.scatter(
        df, 
        x='Component 1', 
        y='Component 2', 
        color='Label', 
        title=f"t-SNE Visualization of Residual Stream Weights<br>(Model: {model_name}, Layer: {layer}, N = {num_samples}, Perplexity = {perplexity})",
        hover_data=['Label'],
        # size=size_column
    )

    fig.update_layout(
        xaxis_title="t-SNE Component 1",
        yaxis_title="t-SNE Component 2",
        legend_title="Label",
        width=900,
        height=700
    )

    # Show plot
    fig.show()

    # Optionally save the plot as an HTML file
    if save:
        now = datetime.datetime.now().strftime("%d%m%Y%H:%M:%S")
        save_fig_path = f'tsne_visualization_{dataset_name}_{now}.html'.replace('/', '_')
        fig.write_html(save_fig_path)
        
        
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
        now = datetime.datetime.now().strftime("%d%m%Y%H:%M:%S")
        save_fig_path = f'pca_visualization_{dataset_name}_{now}.pdf'.replace('/', '_')  # Replace with your desired path
        plt.savefig(save_fig_path)

    plt.show()


def visualize_act_with_pca_interactive(model_name, tensors_flat, labels, dataset_name, num_samples, layer, cluster_label=None, scale_factor=0.1, scores=None, save=False):
    # Apply PCA
    pca = PCA(n_components=2)
    tensors_pca = pca.fit_transform(tensors_flat)

    # Create a DataFrame with PCA components and labels
    df = pd.DataFrame(tensors_pca, columns=['PCA Component 1', 'PCA Component 2'])
    df['Label'] = labels

    # If a specific cluster label is provided, adjust the points to be closer together
    if cluster_label is not None:
        cluster_indices = df['Label'] == cluster_label
        # Scale the coordinates to bring them closer together
        df.loc[cluster_indices, ['PCA Component 1', 'PCA Component 2']] *= scale_factor

    # If scores are provided, include them in the DataFrame and increase the size of the points
    if scores is not None:
        df['Score'] = scores
        size_column = 'Score'
    else:
        size_column = None

    # Create an interactive scatter plot with Plotly
    fig = px.scatter(
        df, 
        x='PCA Component 1', 
        y='PCA Component 2', 
        color='Label', 
        title=f"PCA Visualization of Residual Stream Weights<br>(Model: {model_name}, Layer: {layer}, N = {num_samples})",
        hover_data=['Label'],
        size=size_column,
        size_max=20  # Increase the size of the points
    )

    fig.update_layout(
        xaxis_title="PCA Component 1",
        yaxis_title="PCA Component 2",
        legend_title="Label",
        width=800,
        height=600
    )

    # Show plot
    fig.show()

    # Optionally save the plot as an HTML file
    if save:
        now = datetime.datetime.now().strftime("%d%m%Y%H:%M:%S")
        save_fig_path = f'pca_visualization_{dataset_name}_{now}.html'.replace('/', '_')
        fig.write_html(save_fig_path)
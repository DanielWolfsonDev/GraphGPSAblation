import sys
import torch

def compute_layer_statistics(attention_data):
    """
    Compute and print accumulated statistics for each layer in the model,
    aggregated across all relevant indices.
    Optimizations:
    - Vectorized operations
    - Avoiding large list accumulations
    - In-place computations where possible
    """
    layer_statistics = {}  # Initialize an empty dictionary for dynamic layer tracking

    for element in attention_data:
        if isinstance(element, dict) and 'attn_weights' in element:
            matrix_list = element['attn_weights']
            for layer_idx, matrix in enumerate(matrix_list):
                if isinstance(matrix, torch.Tensor) and matrix.dim() == 3:
                    # Ensure the stats dictionary exists for the current layer
                    if layer_idx not in layer_statistics:
                        layer_statistics[layer_idx] = {
                            'sum_values': 0.0,
                            'sum_squared_values': 0.0,
                            'total_values_count': 0,
                            'total_zeros': 0,
                            'matrix_count': 0,
                            'sum_entropies': 0.0,
                            'entropy_count': 0,
                            'sum_variances': 0.0,
                            'variance_count': 0,
                            'sum_sparsity': 0.0,
                            'sparsity_count': 0,
                        }

                    # Shape [num_heads, seq_len, seq_len]
                    num_heads = matrix.size(0)
                    seq_len = matrix.size(1)
                    layer_stats = layer_statistics[layer_idx]
                    layer_stats['matrix_count'] += num_heads

                    # Flatten matrices to [num_heads, seq_len * seq_len]
                    matrices = matrix.view(num_heads, -1).float()

                    # Update sums and counts for mean and std dev
                    layer_stats['sum_values'] += matrices.sum().item()
                    layer_stats['sum_squared_values'] += (matrices ** 2).sum().item()
                    layer_stats['total_values_count'] += matrices.numel()

                    # Count zeros
                    layer_stats['total_zeros'] += (matrices == 0).sum().item()

                    # Compute entropy vectorized
                    epsilon = 1e-8
                    attn_matrix = matrix.float() + epsilon  # Shape [num_heads, seq_len, seq_len]
                    row_sums = attn_matrix.sum(dim=2, keepdim=True)  # Sum over keys (columns)
                    normalized_attn = attn_matrix / row_sums  # Normalize rows
                    entropies = - (normalized_attn * normalized_attn.log()).sum(dim=2).mean(dim=1)  # Mean entropy per head
                    layer_stats['sum_entropies'] += entropies.sum().item()
                    layer_stats['entropy_count'] += num_heads

                    # Compute variance per matrix
                    variances = attn_matrix.view(num_heads, -1).var(dim=1, unbiased=False)
                    layer_stats['sum_variances'] += variances.sum().item()
                    layer_stats['variance_count'] += num_heads

                    # Compute sparsity per matrix
                    max_values = attn_matrix.view(num_heads, -1).max(dim=1)[0]  # Max per head
                    thresholds = 0.1 * max_values
                    num_elements = seq_len * seq_len
                    sparse_elements = (attn_matrix.view(num_heads, -1) <= thresholds.unsqueeze(1)).sum(dim=1)
                    sparsity = (sparse_elements / num_elements).sum().item()
                    layer_stats['sum_sparsity'] += sparsity
                    layer_stats['sparsity_count'] += num_heads

    # Print accumulated statistics for each layer
    for layer_idx in sorted(layer_statistics.keys()):
        stats = layer_statistics[layer_idx]
        total_values_count = stats['total_values_count']
        matrix_count = stats['matrix_count']
        total_zeros = stats['total_zeros']

        if matrix_count > 0 and total_values_count > 0:
            mean_value = stats['sum_values'] / total_values_count
            mean_squared = stats['sum_squared_values'] / total_values_count
            variance = mean_squared - mean_value ** 2
            std_value = variance ** 0.5 if variance > 0 else 0.0

            avg_entropy = stats['sum_entropies'] / stats['entropy_count'] if stats['entropy_count'] > 0 else 0
            avg_variance = stats['sum_variances'] / stats['variance_count'] if stats['variance_count'] > 0 else 0
            avg_sparsity = stats['sum_sparsity'] / stats['sparsity_count'] if stats['sparsity_count'] > 0 else 0

            print(f"\nAccumulated Statistics for Layer {layer_idx + 1}:")
            print(f"  Total number of matrices processed: {matrix_count}")
            print(f"  Total number of values: {total_values_count}")
            print(f"  Number of zero values: {total_zeros}")
            print(f"  Mean: {mean_value:.6f}")
            print(f"  Std Dev: {std_value:.6f}")
            print(f"  Average Entropy: {avg_entropy:.6f}")
            print(f"  Average Variance: {avg_variance:.6f}")
            print(f"  Average Sparsity: {avg_sparsity:.6f}\n")
        else:
            print(f"No valid matrices processed for Layer {layer_idx + 1}")

def analyze_pt_file(file_path):
    data = torch.load(file_path)
    print(f"Loaded data of type {type(data)}")
    compute_layer_statistics(data)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python compute_layer_statistics.py <path_to_pt_file>")
    else:
        file_path = sys.argv[1]
        analyze_pt_file(file_path)
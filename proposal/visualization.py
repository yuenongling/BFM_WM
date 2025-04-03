"""
Visualization functionality for wall models
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import os
from typing import Dict, Tuple, List, Optional, Union, Any
from sklearn.metrics import r2_score
import torch

black   = "#000000"
blue    = "#065279"
red     = "#9d2933"
orange  = "#ca6924"
cyan    = "#426666"
purple  = "#686789"
yellow  = "#c89b40"
green   = "#75878a"
grey    = "#36282b"

class WallModelVisualization:
    """
    Handles visualization and result analysis for wall models
    """
    
    def __init__(self, dataset_labels: Optional[Dict[str, str]] = None):
        """
        Initialize visualization manager
        
        Args:
            dataset_labels: Optional dictionary mapping dataset keys to human-readable labels
        """
        self.dataset_labels = dataset_labels or {}
        
    
    def plot_training_results(self, 
                             output_train: np.ndarray, 
                             outputs_train_predict: np.ndarray,
                             output_valid: np.ndarray,
                             outputs_valid_predict: np.ndarray,
                             save_path: Optional[str] = None) -> None:
        """
        Plot the regression results for training and validation data
        
        Args:
            output_train: True outputs for training data
            outputs_train_predict: Predicted outputs for training data
            output_valid: True outputs for validation data
            outputs_valid_predict: Predicted outputs for validation data
            save_path: Optional path to save the figure
        """
        # Convert to numpy if tensors
        if torch.is_tensor(output_train):
            output_train = output_train.cpu().detach().numpy()
        if torch.is_tensor(outputs_train_predict):
            outputs_train_predict = outputs_train_predict.cpu().detach().numpy()
        if torch.is_tensor(output_valid):
            output_valid = output_valid.cpu().detach().numpy()
        if torch.is_tensor(outputs_valid_predict):
            outputs_valid_predict = outputs_valid_predict.cpu().detach().numpy()
        
        # Create figure
        fig, ax = plt.subplots(1, 2, figsize=(13, 6))
        
        # Plot training data
        ax[0].scatter(output_train, outputs_train_predict, label="Data", rasterized=True, color=blue, s=8.5)
        
        # Plot validation data
        ax[1].scatter(output_valid, outputs_valid_predict, label="Data", rasterized=True, color=blue, s=8.5)
        
        # Calculate R² scores
        r2_train = r2_score(output_train, outputs_train_predict)
        r2_valid = r2_score(output_valid, outputs_valid_predict)
        
        # Add titles
        ax[0].set_title(f'Training; $R^2: {r2_train:.2f}$')
        ax[1].set_title(f'Validation; $R^2: {r2_valid:.2f}$')
        
        # Add labels and diagonal line
        for a in ax:
            xlim = a.get_xlim()
            a.set_xlabel(r'True $\Pi_{\text{out}}$')
            a.set_ylabel(r'Predicted $\Pi_{\text{out}}$')
            a.plot(xlim, xlim, 'r--')
            a.legend()
        
        plt.tight_layout()
        
        # Save or show
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(f"{save_path}/regression_results.pdf", dpi=300, bbox_inches='tight')
            plt.savefig(f"{save_path}/regression_results.png", dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_error_scatter(self,
                          inputs: np.ndarray,
                          predictions: np.ndarray,
                          true_values: np.ndarray,
                          dataset_key: str = "",
                          log_scale: bool = False,
                          unnormalized_inputs: Optional[np.ndarray] = None,
                          flow_type: Optional[np.ndarray] = None,
                          tauw: bool = True,
                          save_path: Optional[str] = None,
                          mask_threshold: Optional[float] = None,
                          bin_loc: Tuple[float, float, float, float] = (0.5, 0.6, 0.5, 0.35)) -> tuple:
        """
        Plot scatter of predictions vs true values with error information
        
        Args:
            inputs: Input features
            predictions: Model predictions
            true_values: True values
            dataset_key: Key for the dataset (used for titles and lookups)
            log_scale: Whether to use log scale for the plot
            unnormalized_inputs: Original unnormalized inputs (if available)
            flow_type: Flow type information (if available)
            tauw: Whether to plot wall shear stress
            save_path: Optional path to save the figure
            mask_threshold: Optional threshold for masking near-zero values
            bin_loc: Location for histogram inset (x, y, width, height)
            
        Returns:
            Tuple of (max_error, mean_absolute_error, std_absolute_error, mean_relative_error, std_relative_error)
        """
        # To make sure arrays are flattened
        true_values = true_values.flatten()
        predictions = predictions.flatten()

        # Calculate error metrics
        if mask_threshold is not None:
            # To make sure that true_values are flattened
            # Mask near-zero values
            kept_idx = np.where(np.abs(true_values) > mask_threshold)[0]
            
            # Filter data
            inputs = inputs[kept_idx, :]
            predictions = predictions[kept_idx]
            true_values = true_values[kept_idx]
            if unnormalized_inputs is not None:
                unnormalized_inputs = unnormalized_inputs[kept_idx]
            if flow_type is not None:
                flow_type = flow_type[kept_idx]
        
        # Calculate errors
        abs_error = np.abs(predictions - true_values)
        rel_error = abs_error / np.abs(true_values) * 100  # percentage
        
        # Flatten arrays for plotting
        abs_error = abs_error.flatten()
        rel_error = rel_error.flatten()
        
        # Calculate statistics
        mean_abs_error = np.mean(abs_error)
        std_abs_error = np.std(abs_error)
        mean_rel_error = np.mean(rel_error)
        std_rel_error = np.std(rel_error)
        max_abs_error = np.max(abs_error)
        
        # Create figure for error visualization
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if tauw and unnormalized_inputs is not None:
            # Create scatter plot of inputs colored by relative error
            sc = ax.scatter(inputs[:, 0], inputs[:, 1], c=rel_error, 
                           cmap='coolwarm', s=2.5, vmin=0, vmax=100)
            
            cb = plt.colorbar(sc, ax=ax, label='Relative Error (%)')
            
            # Set axis labels
            ax.set_xlabel(r'$\Pi_1$')
            ax.set_ylabel(r'$\Pi_2$')
            
            # Add dataset title
            dataset_title = self.dataset_labels.get(dataset_key, dataset_key)
            ax.set_title(f"{dataset_title} - Wall Shear Stress Predictions")
            
            # Create inset histogram for error distribution
            axins = inset_axes(ax, width="80%", height="70%", 
                              bbox_to_anchor=bin_loc,
                              bbox_transform=ax.transAxes)
            
            # Add semi-opaque background
            axins.patch.set_facecolor('white')
            axins.patch.set_alpha(0.3)
            
            # Plot histogram of relative errors
            hist_range = (0, min(100, np.percentile(rel_error, 95) * 2))
            counts, bins, _ = axins.hist(rel_error, bins=50, color=purple, 
                                        weights=np.zeros_like(rel_error) + 100. / rel_error.size,
                                        range=hist_range)
            
            # Set histogram formatting
            axins.set_xlabel('Relative Error (%)')
            axins.set_ylabel('Frequency (%)')
            
            # Add error statistics as text
            stats_text = f"Mean: {mean_rel_error:.2f}% ± {std_rel_error:.2f}%"
            axins.text(0.5, 0.85, stats_text, transform=axins.transAxes, 
                      ha='center', va='center', fontsize=10, 
                      bbox=dict(facecolor='white', alpha=0.5))
            
            # Add log scale if requested
            if log_scale:
                ax.set_xscale('log')
                ax.set_yscale('log')
        
        # Save or show the figure
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            base_name = f"{dataset_key}_wm"
            plt.savefig(f"{save_path}/{base_name}.pdf", dpi=300, bbox_inches='tight')
            plt.savefig(f"{save_path}/{base_name}.png", dpi=300, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.tight_layout()
            plt.show()
        
        return max_abs_error, mean_abs_error, std_abs_error, mean_rel_error, std_rel_error
    
    def plot_height_profiles(self,
                            predictions: np.ndarray,
                            true_values: np.ndarray, 
                            unnormalized_inputs: np.ndarray,
                            flow_type: np.ndarray,
                            fixed_height: float,
                            dataset_key: str = "",
                            save_path: Optional[str] = None,
                            abs_err: bool = False) -> None:
        """
        Plot profiles at fixed height in the boundary layer
        
        Args:
            predictions: Model predictions
            true_values: True values
            unnormalized_inputs: Original unnormalized inputs
            flow_type: Flow type information
            fixed_height: Fixed height ratio to use
            dataset_key: Key for the dataset (used for titles and lookups)
            save_path: Optional path to save the figure
            abs_err: Whether to plot absolute error instead of relative
        """
        # Flatten arrays if needed
        predictions = predictions.flatten()
        true_values = true_values.flatten()
        
        # Extract needed data
        delta = np.array([float(flow_type[i, 3]) for i in range(len(flow_type))])
        y = unnormalized_inputs[:, 0]
        
        # Find indices with the desired height ratio
        height_ratio = y / delta
        idx = np.where((height_ratio > fixed_height - 0.001) & (height_ratio < fixed_height + 0.001))[0]
        
        if len(idx) == 0:
            print(f"No data points found at height ratio {fixed_height}")
            return
        
        # Get x-locations (often streamwise coordinate)
        x = np.array([float(flow_type[i, 2]) for i in range(len(flow_type))])
        x_subset = x[idx]
        
        # Sort by x-location
        sort_idx = np.argsort(x_subset)
        x_sorted = x_subset[sort_idx]
        pred_sorted = predictions[idx][sort_idx]
        true_sorted = true_values[idx][sort_idx]
        
        # Calculate error
        if abs_err:
            error = np.abs(pred_sorted - true_sorted)
            error_label = 'Absolute Error'
        else:
            error = np.abs(pred_sorted - true_sorted) / np.abs(true_sorted) * 100
            error_label = 'Relative Error (%)'
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, 
                                      gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot values
        ax1.plot(x_sorted, true_sorted, 'o-', color=blue, label='True')
        ax1.plot(x_sorted, pred_sorted, 's--', color=red, label='Predicted')
        
        # Plot error
        ax2.plot(x_sorted, error, 'o-', color=green)
        
        # Set labels
        ax1.set_ylabel(r'$\tau_w$ or $u_\tau^2$')
        ax2.set_xlabel('Streamwise coordinate')
        ax2.set_ylabel(error_label)
        
        # Add title and legend
        dataset_title = self.dataset_labels.get(dataset_key, dataset_key)
        ax1.set_title(f"{dataset_title} - Profile at y/δ = {fixed_height}")
        ax1.legend()
        
        # Format plot
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax2.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        
        # Save or show
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            base_name = f"{dataset_key}_y_{fixed_height}"
            plt.savefig(f"{save_path}/{base_name}.pdf", dpi=300, bbox_inches='tight')
            plt.savefig(f"{save_path}/{base_name}.png", dpi=300, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()
    
    def plot_comparison_bar(self,
                           model_metrics: Tuple[float, float],
                           baseline_metrics: Tuple[float, float],
                           dataset_key: str = "",
                           metric_name: str = "Relative Error (%)",
                           save_path: Optional[str] = None) -> None:
        """
        Plot bar chart comparing model performance against baseline
        
        Args:
            model_metrics: Tuple of (mean, std) for model errors
            baseline_metrics: Tuple of (mean, std) for baseline errors
            dataset_key: Key for the dataset (used for titles)
            metric_name: Name of the metric being plotted
            save_path: Optional path to save the figure
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(6, 6))
        
        # Extract metrics
        model_mean, model_std = model_metrics
        baseline_mean, baseline_std = baseline_metrics
        
        # Plot bars
        ax.bar(['Model', 'Baseline'], 
              [model_mean, baseline_mean], 
              yerr=[model_std, baseline_std], 
              color=[blue, purple], 
              error_kw=dict(capsize=5))
        
        # Set y-limit to accommodate the highest bar with error
        max_value = max(model_mean + model_std, baseline_mean + baseline_std)
        ax.set_ylim(0, max(10, max_value * 1.2))  # At least 10 or 20% headroom
        
        # Set labels and title
        ax.set_ylabel(metric_name)
        dataset_title = self.dataset_labels.get(dataset_key, dataset_key)
        ax.set_title(f"{dataset_title}")
        
        # Format plot
        ax.grid(True, axis='y', linestyle='--', alpha=0.6)
        
        # Save or show
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            base_name = f"{dataset_key}_comparison"
            plt.savefig(f"{save_path}/{base_name}.pdf", dpi=300, bbox_inches='tight')
            plt.savefig(f"{save_path}/{base_name}.png", dpi=300, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()
    
    def plot_loss_history(self,
                         train_loss: np.ndarray,
                         valid_loss: np.ndarray,
                         save_path: Optional[str] = None) -> None:
        """
        Plot training and validation loss history
        
        Args:
            train_loss: Training loss history
            valid_loss: Validation loss history
            save_path: Optional path to save the figure
        """
        # Create figure
        fig_loss, ax = plt.subplots(figsize=(10, 6))
        
        # Plot losses
        epochs = np.arange(1, len(train_loss) + 1)
        ax.plot(epochs, train_loss, label='Training Loss', color=blue)
        ax.plot(epochs, valid_loss, label='Validation Loss', color=red)
        
        # Set labels and title
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training and Validation Loss')
        
        # Add legend and grid
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # Set log scale for y-axis (often useful for loss plots)
        ax.set_yscale('log')
        
        # Save or show
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(f"{save_path}/loss_history.pdf", dpi=300, bbox_inches='tight')
            plt.savefig(f"{save_path}/loss_history.png", dpi=300, bbox_inches='tight')
            plt.close(fig_loss)
        else:
            plt.show(fig_loss)
    
    def plot_input_distribution(self,
                               inputs: np.ndarray,
                               flow_types: Optional[np.ndarray] = None,
                               save_path: Optional[str] = None) -> None:
        """
        Plot the distribution of input features
        
        Args:
            inputs: Input features
            flow_types: Optional flow type information for coloring
            save_path: Optional path to save the figure
        """
        # Determine number of input dimensions
        n_dims = inputs.shape[1]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if n_dims >= 2:
            # For 2+ dimensions, plot first two as scatter
            if flow_types is not None:
                # Create categorical colors based on flow types
                unique_types = np.unique(flow_types[:, 0])
                cmap = plt.cm.get_cmap('tab10', len(unique_types))
                
                # Map flow types to colors
                colors = np.zeros(len(flow_types))
                for i, ft in enumerate(unique_types):
                    colors[flow_types[:, 0] == ft] = i
                
                # Create scatter plot
                sc = ax.scatter(inputs[:, 0], inputs[:, 1], c=colors, cmap=cmap, alpha=0.7, s=3)
                
                # Add legend
                handles = []
                labels = []
                for i, ft in enumerate(unique_types):
                    handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                                           markerfacecolor=cmap(i), markersize=8))
                    labels.append(ft)
                ax.legend(handles, labels, title='Flow Type', loc='best')
            else:
                # Simple scatter plot
                ax.scatter(inputs[:, 0], inputs[:, 1], alpha=0.7, s=3, color=blue)
            
            # Set labels
            ax.set_xlabel(r'$\Pi_1$')
            ax.set_ylabel(r'$\Pi_2$')
            
        else:
            # For 1D inputs, plot histogram
            ax.hist(inputs[:, 0], bins=50, color=blue, alpha=0.7)
            ax.set_xlabel(r'$\Pi_1$')
            ax.set_ylabel('Frequency')
        
        # Set title
        ax.set_title('Input Feature Distribution')
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # Save or show
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(f"{save_path}/input_distribution.pdf", dpi=300, bbox_inches='tight')
            plt.savefig(f"{save_path}/input_distribution.png", dpi=300, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.tight_layout()
            plt.show()

    def plot_results_scatter_error(self, 
                                 inputs: np.ndarray,
                                 output: np.ndarray,
                                 output_true: np.ndarray,
                                 dataset: str,
                                 log_scale: bool = False,
                                 abs_err: bool = False,
                                 bin_loc: Tuple[float, float, float, float] = (0.4, 0.1, 0.5, 0.35),
                                 unnormalized_inputs: Optional[np.ndarray] = None,
                                 flow_type: Optional[np.ndarray] = None,
                                 weighted_utau: bool = False,
                                 tauw: bool = False,
                                 save_path: Optional[str] = None,
                                 mask_threshold: Optional[float] = None) -> Tuple[Optional[float], float, float, float, float]:
        """
        Plot scatter plot colored by error
        
        Args:
            inputs: Input features
            output: Model predictions
            output_true: True values
            dataset: Dataset name
            log_scale: Whether to use log scale for error
            abs_err: Whether to use absolute error
            bin_loc: Location for histogram inset
            unnormalized_inputs: Unnormalized input features
            flow_type: Flow type information
            weighted_utau: Whether to weight by friction velocity
            tauw: Whether to plot wall shear stress
            save_path: Optional path to save plots
            mask_threshold: Optional threshold for masking
            
        Returns:
            Tuple of (max_abs_error, mean_abs_error, std_abs_error, mean_rel_error, std_rel_error)
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Calculate errors
        if abs_err:
            err = np.abs(output - output_true).squeeze()
            if weighted_utau and flow_type is not None:
                utau = np.array([float(flow_type[i, 1]) for i in range(len(flow_type))])
                err = err * utau
        else:
            err = np.abs((output - output_true) * 100 / output_true).squeeze()
            if weighted_utau and flow_type is not None:
                utau = np.array([float(flow_type[i, 1]) for i in range(len(flow_type))])
                err = err * utau
        
        # Plot scatter
        # if log_scale:
        sc = ax.scatter(inputs[:, 0], inputs[:, 1], c=err, cmap='inferno', s=2.5, 
                        norm=matplotlib.colors.LogNorm(vmin=0.1, vmax=100))
        # else:
        #     sc = ax.scatter(inputs[:, 0], inputs[:, 1], c=err, cmap='inferno', s=2.5)
        
        # Add colorbar
        if abs_err:
            plt.colorbar(sc, ax=ax, label='Absolute Error', orientation='vertical')
        else:
            plt.colorbar(sc, ax=ax, label='Relative Error (%)', orientation='vertical')
        
        # Create histogram inset
        axins = inset_axes(ax, width="80%", height="70%", 
                          bbox_to_anchor=bin_loc,
                          bbox_transform=ax.transAxes)
        
        # Add semi-opaque background
        axins.patch.set_facecolor('white')
        axins.patch.set_alpha(0.9)
        
        # Plot histogram
        sorted_err = np.sort(err)
        counts, bins, _ = axins.hist(sorted_err, bins=50, color=purple, 
                                    weights=np.zeros_like(sorted_err) + 100. / sorted_err.size, 
                                    range=(0, 100))
        
        # Set histogram properties
        axins.set_xticks(np.linspace(0, 100, 11))
        axins.set_xlim(0, 100)
        max_count = np.ceil(max(counts))
        axins.set_yticks(np.linspace(0, max_count, 6))
        
        # Calculate statistics
        n_outside = np.sum(err > 100)
        outside_percent = n_outside / len(err) * 100
        below_10_percent = np.mean(err <= 10) * 100
        
        # Add annotations
        if outside_percent > 0:
            axins.text(0.7, 0.8, f'>{100}%: {outside_percent:.1f}%',
                      transform=axins.transAxes, fontsize=8, fontweight='bold',
                      bbox=dict(facecolor='none', alpha=0.5, edgecolor='red'))
        
        axins.axvline(x=10, color='green', linestyle='--', alpha=0.7)
        axins.text(12, axins.get_ylim()[1] * 0.8, f'≤10%: {below_10_percent:.1f}%',
                  fontsize=7, fontweight='bold',
                  bbox=dict(facecolor='none', alpha=0.7, edgecolor='green'))
        
        # Set labels and title
        ax.set_xlabel(r'$\Pi_1$')
        ax.set_ylabel(r'$\Pi_2$')
        if dataset is not None:
            ax.set_title(f'{self.dataset_labels.get(dataset, dataset)}')
        
        # Save or show plot
        if save_path is not None:
            case_name = dataset.replace("-", "")
            fig.savefig(f"{save_path}/{case_name}_wm.pdf", dpi=300)
            fig.savefig(f"{save_path}/{case_name}_wm.png", dpi=300)
            plt.close(fig)
        else:
            fig.show()
        
        # Return metrics
        if abs_err:
            return None, np.mean(err), np.std(err), 0, 0
        else:
            return None, 0, 0, np.mean(err), np.std(err)
    
    def plot_results_scatter_error_loglaw(self, 
                                        inputs: np.ndarray,
                                        log_predictions: np.ndarray,
                                        output_true: np.ndarray,
                                        dataset: str,
                                        log_scale: bool = False,
                                        bin_loc: Tuple[float, float, float, float] = (0.4, 0.1, 0.5, 0.35),
                                        unnormalized_inputs: Optional[np.ndarray] = None,
                                        flow_type: Optional[np.ndarray] = None,
                                        save_path: Optional[str] = None,
                                        mask_threshold: Optional[float] = None,
                                        max_model_err: Optional[float] = None) -> Tuple[float, float, float, float]:
        """
        Plot scatter plot colored by error for log law baseline
        
        Args:
            inputs: Input features
            output_true: True values
            dataset: Dataset name
            log_scale: Whether to use log scale for error
            bin_loc: Location for histogram inset
            unnormalized_inputs: Unnormalized input features
            flow_type: Flow type information
            save_path: Optional path to save plots
            mask_threshold: Optional threshold for masking
            max_model_err: Maximum error from model for comparison
            
        Returns:
            Tuple of (mean_abs_error, std_abs_error, mean_rel_error, std_rel_error)
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Extract flow parameters
        utau = unnormalized_inputs[:, 3]
        
        # Calculate log law predictions
        output_log = log_predictions
        
        # Calculate wall shear stress
        output_true_tauw = (utau)**2
        
        # Apply masking if requested
        if mask_threshold is not None:
            kept_idx = np.where(np.abs(output_true_tauw) > mask_threshold)
            other_idx = np.where(np.abs(output_true_tauw) <= mask_threshold)
            
            # Plot separation region if needed
            if len(other_idx[0]) > 0:
                inputs_sep = inputs[other_idx]
                output_log_sep = output_log[other_idx]
                output_true_sep = output_true_tauw[other_idx]
                flow_type_sep = flow_type[other_idx]
                unnormalized_inputs_sep = unnormalized_inputs[other_idx]
                
                # Calculate absolute error for separation region
                err_abs = np.abs(output_log_sep - output_true_sep).squeeze()
                
                # Plot separation region results
                fig_abs, ax_abs = plt.subplots(figsize=(10, 6))
                sc_abs = ax_abs.scatter(inputs_sep[:, 0], inputs_sep[:, 1], c=err_abs, 
                                       cmap='inferno', s=2.5, vmax=max_model_err)
                plt.colorbar(sc_abs, ax=ax_abs, label='Absolute Error', orientation='vertical')
                
                # Set labels and title
                ax_abs.set_xlabel(r'$\Pi_1$')
                ax_abs.set_ylabel(r'$\Pi_2$')
                if dataset is not None:
                    ax_abs.set_title(f'{self.dataset_labels.get(dataset, dataset)} using Log Law \n (near separation) [Mean error: {np.mean(err_abs):.2e}]')
                
                # Save or show plot
                if save_path is not None:
                    case_name = dataset.replace("-", "")
                    fig_abs.savefig(f"{save_path}/{case_name}_loglaw_abs.pdf", dpi=300)
                    fig_abs.savefig(f"{save_path}/{case_name}_loglaw_abs.png", dpi=300)
                    plt.close(fig_abs)
                else:
                    fig_abs.show()
            
            # Keep non-separation indices
            inputs = inputs[kept_idx]
            output_log = output_log[kept_idx]
            output_true_tauw = output_true_tauw[kept_idx]
            flow_type = flow_type[kept_idx]
            unnormalized_inputs = unnormalized_inputs[kept_idx]
        
        # Calculate relative error
        err = np.abs((output_log - output_true_tauw) * 100 / output_true_tauw).squeeze()
        
        # Plot scatter
        sc = ax.scatter(inputs[:, 0], inputs[:, 1], c=err, cmap='inferno', s=2.5, 
                       norm=matplotlib.colors.LogNorm(vmin=0.1, vmax=100))
        plt.colorbar(sc, ax=ax, label='Relative Error (%)', orientation='vertical')
        
        # Create histogram inset
        axins = inset_axes(ax, width="80%", height="70%", 
                          bbox_to_anchor=bin_loc,
                          bbox_transform=ax.transAxes)
        
        # Add semi-opaque background
        axins.patch.set_facecolor('white')
        axins.patch.set_alpha(0.9)
        
        # Plot histogram
        sorted_err = np.sort(err)
        counts, bins, _ = axins.hist(sorted_err, bins=50, color=purple, 
                                    weights=np.zeros_like(sorted_err) + 100. / sorted_err.size, 
                                    range=(0, 100))
        
        # Set histogram properties
        axins.set_xticks(np.linspace(0, 100, 11))
        axins.set_xlim(0, 100)
        max_count = np.ceil(max(counts))
        axins.set_yticks(np.linspace(0, max_count, 6))
        
        # Calculate statistics
        n_outside = np.sum(err > 100)
        outside_percent = n_outside / len(err) * 100
        below_10_percent = np.mean(err <= 10) * 100
        
        # Add annotations
        if outside_percent > 0:
            axins.text(0.7, 0.8, f'>{100}%: {outside_percent:.1f}%',
                      transform=axins.transAxes, fontsize=8, fontweight='bold',
                      bbox=dict(facecolor='none', alpha=0.5, edgecolor='red'))
        
        axins.axvline(x=10, color='green', linestyle='--', alpha=0.7)
        axins.text(12, axins.get_ylim()[1] * 0.8, f'≤10%: {below_10_percent:.1f}%',
                  fontsize=7, fontweight='bold',
                  bbox=dict(facecolor='none', alpha=0.7, edgecolor='green'))
        
        # Set labels and title
        ax.set_xlabel(r'$\Pi_1$')
        ax.set_ylabel(r'$\Pi_2$')
        if dataset is not None:
            ax.set_title(f'{self.dataset_labels.get(dataset, dataset)} using Log Law')
        
        # Save or show plot
        if save_path is not None:
            case_name = dataset.replace("-", "")
            fig.savefig(f"{save_path}/{case_name}_loglaw.pdf", dpi=300)
            fig.savefig(f"{save_path}/{case_name}_loglaw.png", dpi=300)
            plt.close(fig)
        else:
            fig.show()
        
        # Return metrics
        if mask_threshold is not None and len(other_idx[0]) > 0:
            if len(kept_idx[0]) > 0:
                return np.mean(err_abs), np.std(err_abs), np.mean(err), np.std(err)
            else:
                return np.mean(err_abs), np.std(err_abs), 0, 0
        else:
            return 0, 0, np.mean(err), np.std(err)
    
    def plot_results_fixed_height(self, 
                                fixed_height: float,
                                output_pred: np.ndarray,
                                output_true: np.ndarray,
                                dataset: str,
                                unnormalized_inputs: np.ndarray,
                                flow_type: np.ndarray,
                                save_path: Optional[str] = None,
                                abs_err: bool = False) -> None:
        """
        Plot results at a fixed height ratio
        
        Args:
            fixed_height: Fixed height ratio
            output_pred: Model predictions
            output_true: True values
            dataset: Dataset name
            unnormalized_inputs: Unnormalized input features
            flow_type: Flow type information
            save_path: Optional path to save plots
            abs_err: Whether to use absolute error
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Extract flow parameters
        nu = unnormalized_inputs[:, 2]
        u = unnormalized_inputs[:, 1]
        y = unnormalized_inputs[:, 0]
        utau = unnormalized_inputs[:, 3]
        
        # Find corresponding y values
        delta = np.array([float(flow_type[i, 3]) for i in range(len(flow_type))])
        fixed_height_delta = fixed_height * delta
        
        # Find unique x values
        x = np.array([float(flow_type[i, 2]) for i in range(len(flow_type))])
        x_unique = np.sort(np.unique(x))
        
        # Find samples at fixed height
        y_idx = np.zeros_like(x_unique, dtype=int)
        for x_idx, x_val in enumerate(x_unique):
            sample_idx = np.argwhere(x == x_val).squeeze()
            if len(sample_idx) > 0:
                y_idx[x_idx] = sample_idx[np.argmin(np.abs(y[sample_idx] - fixed_height_delta[sample_idx]))]
        
        # Extract samples at fixed height
        output_true_fixed = output_true[y_idx]
        output_pred_fixed = output_pred[y_idx]
        y_fixed = y[y_idx]
        nu_fixed = nu[y_idx]
        u_fixed = u[y_idx]
        utau_fixed = utau[y_idx]
        
        # Calculate log law predictions
        output_log = np.zeros_like(output_true_fixed)
        for idx in range(len(output_true_fixed)):
            output_log[idx] = (self._eqwm_solve(y_fixed[idx], nu_fixed[idx], abs(u_fixed[idx])))**2
            output_pred_fixed[idx] = (output_pred_fixed[idx] * nu_fixed[idx] / y_fixed[idx])**2
        
        # Calculate wall shear stress
        output_true_tauw = (utau_fixed)**2
        
        # Calculate errors
        if abs_err:
            err_log = np.abs(output_log - output_true_tauw).squeeze()
            err_pred = np.abs(output_pred_fixed - output_true_tauw).squeeze()
        else:
            err_log = np.abs((output_log - output_true_tauw) * 100 / output_true_tauw).squeeze()
            err_pred = np.abs((output_pred_fixed - output_true_tauw) * 100 / output_true_tauw).squeeze()
        
        # Plot errors
        ax.plot(x_unique, err_log, '-o', color='green', label='Log Law')
        ax.plot(x_unique, err_pred, '-o', color='blue', label='Model')
        ax.legend()
        
        # Set labels and title
        if abs_err:
            ax.set_ylabel(r'Absolute Error ($|\tau_{w, \text{true}} - \tau_{w, \text{pred}}|$)')
        else:
            ax.set_ylabel(r'Relative Error (%)')
            ax.set_ylim(-5, 20)
        
        if dataset is not None:
            ax.set_title(fr'{self.dataset_labels.get(dataset, dataset)} with matching location $y/\delta$ = {fixed_height}')
        
        if 'naca' in dataset:
            ax.set_xlabel(r'$x/c$')
        else:
            ax.set_xlabel(r'$x/\theta_{in}$')
        
        # Save or show plot
        if save_path is not None:
            case_name = dataset.replace("-", "")
            plt.savefig(f"{save_path}/{case_name}_y_{fixed_height}.pdf", dpi=300, bbox_inches='tight')
            plt.savefig(f"{save_path}/{case_name}_y_{fixed_height}.png", dpi=200, bbox_inches='tight')
            plt.close()
        else:
            fig.show()
    
    def plot_comparison_bar_chart(self, 
                                model_mean: float,
                                model_std: float,
                                loglaw_mean: float,
                                loglaw_std: float,
                                dataset: str,
                                save_path: Optional[str] = None) -> None:
        """
        Plot comparison bar chart between model and log law
        
        Args:
            model_mean: Mean error for model
            model_std: Standard deviation of error for model
            loglaw_mean: Mean error for log law
            loglaw_std: Standard deviation of error for log law
            dataset: Dataset name
            save_path: Optional path to save plots
        """
        # Create figure
        fig_bar, ax = plt.subplots(figsize=(6, 6))
        
        # Plot bars
        ax.bar(['Model', 'Log Law'], [model_mean, loglaw_mean], 
               yerr=[model_std, loglaw_std], 
               color=[blue, purple], 
               error_kw=dict(capsize=5))
        
        # Set y-axis limit
        ax.set_ylim(0, max(10, max(model_mean + model_std, loglaw_mean + loglaw_std)))
        
        # Set labels and title
        ax.set_ylabel('Mean relative error (%)')
        ax.set_title(f'{self.dataset_labels.get(dataset, dataset)}')
        
        # Save or show plot
        if save_path is not None:
            case_name = dataset.replace("-", "")
            plt.savefig(f"{save_path}/{case_name}_comparison.pdf", dpi=300)
            plt.savefig(f"{save_path}/{case_name}_comparison.png", dpi=300)
            plt.close()
        else:
            fig_bar.show()

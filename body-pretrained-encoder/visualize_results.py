"""
Visualize encoder test results using Plotly.
Creates interactive 3D comparison plots.

Usage:
    python body-pretrained-encoder/visualize_results.py
"""

import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def load_results(results_dir: str, sample_id: str):
    """Load test results from NPZ file."""
    npz_path = os.path.join(results_dir, f'{sample_id}_results.npz')
    data = np.load(npz_path)
    return {
        'input_pc': data['input_pc'],
        'visible_pc': data['visible_pc'],
        'masked_target_pc': data['masked_target_pc'],
        'masked_pred_pc': data['masked_pred_pc'],
        'reconstructed_pc': data['reconstructed_pc'],
    }


def create_point_cloud_trace(points, name, color, size=2, opacity=0.8):
    """Create a Plotly scatter3d trace for a point cloud."""
    return go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        marker=dict(
            size=size,
            color=color,
            opacity=opacity,
        ),
        name=name,
    )


def visualize_comparison(data: dict, sample_id: str, output_path: str = None):
    """
    Create a 2x2 subplot comparison visualization.

    Layout:
    - Top-left: Input point cloud (all points)
    - Top-right: Visible vs Masked (before reconstruction)
    - Bottom-left: Masked target vs Predicted
    - Bottom-right: Reconstructed point cloud
    """
    fig = make_subplots(
        rows=2, cols=2,
        specs=[
            [{'type': 'scatter3d'}, {'type': 'scatter3d'}],
            [{'type': 'scatter3d'}, {'type': 'scatter3d'}]
        ],
        subplot_titles=(
            'Input Point Cloud (8192 pts)',
            'Visible (blue) vs Masked Target (red)',
            'Masked: Target (red) vs Predicted (green)',
            'Reconstructed Point Cloud'
        ),
        horizontal_spacing=0.02,
        vertical_spacing=0.08,
    )

    # Subsample for better performance
    max_pts = 4000

    def subsample(pts, n=max_pts):
        if len(pts) > n:
            idx = np.random.choice(len(pts), n, replace=False)
            return pts[idx]
        return pts

    input_pc = subsample(data['input_pc'])
    visible_pc = subsample(data['visible_pc'])
    masked_target = subsample(data['masked_target_pc'])
    masked_pred = subsample(data['masked_pred_pc'])
    reconstructed = subsample(data['reconstructed_pc'])

    # Top-left: Input point cloud
    fig.add_trace(
        go.Scatter3d(
            x=input_pc[:, 0], y=input_pc[:, 1], z=input_pc[:, 2],
            mode='markers',
            marker=dict(size=1.5, color='steelblue', opacity=0.7),
            name='Input',
        ),
        row=1, col=1
    )

    # Top-right: Visible vs Masked
    fig.add_trace(
        go.Scatter3d(
            x=visible_pc[:, 0], y=visible_pc[:, 1], z=visible_pc[:, 2],
            mode='markers',
            marker=dict(size=1.5, color='dodgerblue', opacity=0.7),
            name='Visible',
        ),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter3d(
            x=masked_target[:, 0], y=masked_target[:, 1], z=masked_target[:, 2],
            mode='markers',
            marker=dict(size=1.5, color='crimson', opacity=0.7),
            name='Masked (Target)',
        ),
        row=1, col=2
    )

    # Bottom-left: Masked target vs Predicted
    fig.add_trace(
        go.Scatter3d(
            x=masked_target[:, 0], y=masked_target[:, 1], z=masked_target[:, 2],
            mode='markers',
            marker=dict(size=1.5, color='crimson', opacity=0.5),
            name='Target',
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter3d(
            x=masked_pred[:, 0], y=masked_pred[:, 1], z=masked_pred[:, 2],
            mode='markers',
            marker=dict(size=1.5, color='limegreen', opacity=0.5),
            name='Predicted',
        ),
        row=2, col=1
    )

    # Bottom-right: Reconstructed
    fig.add_trace(
        go.Scatter3d(
            x=reconstructed[:, 0], y=reconstructed[:, 1], z=reconstructed[:, 2],
            mode='markers',
            marker=dict(size=1.5, color='darkorange', opacity=0.7),
            name='Reconstructed',
        ),
        row=2, col=2
    )

    # Update layout
    fig.update_layout(
        title=dict(
            text=f'Encoder Reconstruction Results: {sample_id}',
            x=0.5,
            font=dict(size=16)
        ),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02,
        ),
        width=1200,
        height=900,
        margin=dict(l=0, r=150, t=80, b=0),
    )

    # Update all 3D axes
    camera = dict(
        eye=dict(x=1.5, y=1.5, z=0.5)
    )

    for i in range(1, 5):
        scene_name = f'scene{i}' if i > 1 else 'scene'
        fig.update_layout(**{
            scene_name: dict(
                xaxis_title='X (mm)',
                yaxis_title='Y (mm)',
                zaxis_title='Z (mm)',
                aspectmode='data',
                camera=camera,
            )
        })

    # Save to HTML
    if output_path is None:
        output_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'test',
            f'{sample_id}_comparison.html'
        )

    fig.write_html(output_path)
    print(f"Saved visualization to: {output_path}")

    return fig


def visualize_error_distribution(data: dict, sample_id: str, output_path: str = None):
    """
    Create visualization showing reconstruction error distribution.
    Color-coded by L2 error distance.
    """
    masked_target = data['masked_target_pc']
    masked_pred = data['masked_pred_pc']

    # Compute per-point L2 error
    l2_error = np.sqrt(np.sum((masked_pred - masked_target) ** 2, axis=-1))

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scatter3d'}, {'type': 'histogram'}]],
        subplot_titles=(
            'Prediction Error (color = L2 distance)',
            'Error Distribution'
        ),
        column_widths=[0.7, 0.3],
    )

    # 3D scatter with error coloring
    fig.add_trace(
        go.Scatter3d(
            x=masked_pred[:, 0],
            y=masked_pred[:, 1],
            z=masked_pred[:, 2],
            mode='markers',
            marker=dict(
                size=2,
                color=l2_error,
                colorscale='RdYlGn_r',  # Red = high error, Green = low
                colorbar=dict(
                    title='L2 Error (mm)',
                    x=0.45,
                ),
                opacity=0.8,
            ),
            name='Predicted Points',
            hovertemplate='X: %{x:.1f}<br>Y: %{y:.1f}<br>Z: %{z:.1f}<br>Error: %{marker.color:.1f} mm',
        ),
        row=1, col=1
    )

    # Error histogram
    fig.add_trace(
        go.Histogram(
            x=l2_error,
            nbinsx=50,
            marker_color='steelblue',
            name='Error Distribution',
        ),
        row=1, col=2
    )

    # Add statistics annotation
    stats_text = (
        f"Mean: {l2_error.mean():.2f} mm<br>"
        f"Median: {np.median(l2_error):.2f} mm<br>"
        f"Std: {l2_error.std():.2f} mm<br>"
        f"Max: {l2_error.max():.2f} mm"
    )

    fig.add_annotation(
        x=0.95, y=0.95,
        xref='x2 domain', yref='y2 domain',
        text=stats_text,
        showarrow=False,
        font=dict(size=12),
        align='left',
        bgcolor='white',
        bordercolor='gray',
        borderwidth=1,
    )

    fig.update_layout(
        title=dict(
            text=f'Reconstruction Error Analysis: {sample_id}',
            x=0.5,
            font=dict(size=16)
        ),
        width=1200,
        height=600,
        showlegend=False,
    )

    fig.update_layout(
        scene=dict(
            xaxis_title='X (mm)',
            yaxis_title='Y (mm)',
            zaxis_title='Z (mm)',
            aspectmode='data',
        ),
    )
    fig.update_xaxes(title_text='L2 Error (mm)', row=1, col=2)
    fig.update_yaxes(title_text='Count', row=1, col=2)

    # Save
    if output_path is None:
        output_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'test',
            f'{sample_id}_error_analysis.html'
        )

    fig.write_html(output_path)
    print(f"Saved error analysis to: {output_path}")

    return fig


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_id', type=str, default='BDMAP_00000151',
                        help='Sample ID to visualize')
    parser.add_argument('--results_dir', type=str,
                        default='body-pretrained-encoder/test',
                        help='Directory containing results')
    args = parser.parse_args()

    print(f"Loading results for {args.sample_id}...")
    data = load_results(args.results_dir, args.sample_id)

    print("\nPoint cloud sizes:")
    for name, pc in data.items():
        print(f"  {name}: {pc.shape}")

    print("\nGenerating comparison visualization...")
    visualize_comparison(data, args.sample_id)

    print("\nGenerating error analysis visualization...")
    visualize_error_distribution(data, args.sample_id)

    print("\nDone! Open the HTML files in a browser to view.")


if __name__ == '__main__':
    main()

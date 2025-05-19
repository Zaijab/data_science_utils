import os
import jax
import jax.numpy as jnp
import mlflow
import numpy as np
import plotly.graph_objects as go
from mlflow.tracking import MlflowClient
from data_science_utils.dynamical_systems import Ikeda

# Create necessary directories
os.makedirs("cache", exist_ok=True)
os.makedirs("cache/temp", exist_ok=True)
os.makedirs("cache/mlruns", exist_ok=True)

# MLflow setup
mlflow.set_tracking_uri("sqlite:///cache/mlflow.db")
dashboard_path = os.path.join("cache/temp", "ikeda_plotly_explorer.html")

os.environ["MLFLOW_TRACKING_DIR"] = os.path.abspath("cache/mlruns")
experiment_name = "ikeda-attractor"
try:
    experiment_id = mlflow.create_experiment(
        experiment_name, artifact_location=os.path.join("cache/mlruns", experiment_name)
    )
except:
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

mlflow.set_experiment(experiment_name)

# Parameter ranges for sliders
u_values = np.linspace(0.7, 0.99, 30)
batch_sizes = [1000, 5000, 10000, 25000, 50000]


# Function to generate attractor data
def generate_ikeda_data(u, batch_size):
    system = Ikeda(u=u, batch_size=batch_size)
    attractor = system.generate(jax.random.key(0))
    return np.array(attractor)


# Create a custom function to save to MLflow
def save_to_mlflow(u, batch_size):
    with mlflow.start_run(run_name=f"u={u}_batch={batch_size}"):
        # Log parameters
        mlflow.log_params({"u": u, "batch_size": batch_size, "key_seed": 0})

        # Generate data
        attractor = generate_ikeda_data(u, batch_size)

        # Calculate metrics
        spread_x = np.std(attractor[:, 0])
        spread_y = np.std(attractor[:, 1])
        mlflow.log_metrics({"spread_x": float(spread_x), "spread_y": float(spread_y)})

        # Create a static image for MLflow
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=attractor[:, 0],
                y=attractor[:, 1],
                mode="markers",
                marker=dict(
                    size=3, color=attractor[:, 0], colorscale="Viridis", opacity=0.7
                ),
            )
        )

        fig.update_layout(
            title=f"Ikeda Attractor: u={u}, batch_size={batch_size}",
            xaxis_title="x",
            yaxis_title="y",
            width=800,
            height=600,
        )

        # Save as PNG for MLflow
        # img_path = f"cache/ikeda-experiment/ikeda_u{u}_batch{batch_size}.png"
        # fig.write_image(img_path)
        # mlflow.log_artifact(img_path)
        # os.remove(img_path)
        cache_dir = "cache/temp"
        os.makedirs(cache_dir, exist_ok=True)
        img_path = os.path.join(cache_dir, f"ikeda_u{u}_batch{batch_size}.png")
        fig.write_image(img_path)
        mlflow.log_artifact(img_path)
        os.remove(img_path)

        print(f"Saved to MLflow: u={u}, batch_size={batch_size}")


# Create interactive Plotly figure
def create_interactive_plot():
    # Generate initial data
    initial_u = 0.9
    initial_batch = 10000
    attractor = generate_ikeda_data(initial_u, initial_batch)

    # Create main figure
    fig = go.Figure()

    # Add points scatter
    fig.add_trace(
        go.Scatter(
            x=attractor[:, 0],
            y=attractor[:, 1],
            mode="markers",
            marker=dict(size=2, color="rgba(255, 255, 255, 0.3)"),
            name="Points",
        )
    )

    # Update layout
    fig.update_layout(
        title="Interactive Ikeda Attractor Explorer",
        xaxis_title="x",
        yaxis_title="y",
        width=1000,
        height=800,
        hovermode="closest",
    )

    # Add sliders for u parameter
    u_steps = []
    for u in u_values:
        step = dict(
            method="animate",
            args=[
                [f"u{u}"],
                {
                    "frame": {"duration": 0, "redraw": True},
                    "mode": "immediate",
                    "transition": {"duration": 0},
                },
            ],
            label=f"{u:.2f}",
        )
        u_steps.append(step)

    # Add slider for batch size
    batch_steps = []
    for batch in batch_sizes:
        step = dict(
            method="animate",
            args=[
                [f"batch{batch}"],
                {
                    "frame": {"duration": 0, "redraw": True},
                    "mode": "immediate",
                    "transition": {"duration": 0},
                },
            ],
            label=f"{batch}",
        )
        batch_steps.append(step)

    fig.update_layout(margin=dict(t=150))  # Increase top margin
    # Add sliders to layout
    sliders = [
        dict(
            active=(
                np.where(u_values == initial_u)[0][0] if initial_u in u_values else 0
            ),
            currentvalue={"prefix": "Parameter u: "},
            pad={"t": 20, "b": 10},
            steps=u_steps,
            yanchor="top",
            y=1.2,
        ),
        dict(
            active=(
                batch_sizes.index(initial_batch) if initial_batch in batch_sizes else 0
            ),
            currentvalue={"prefix": "Batch size: "},
            pad={"t": 50},
            steps=batch_steps,
            yanchor="top",
            y=1.02,
        ),
    ]

    fig.update_layout(sliders=sliders)

    # Create frames for animations
    frames = []
    for u in u_values:
        for batch in batch_sizes:
            attractor = generate_ikeda_data(u, batch)
            H, xedges, yedges = np.histogram2d(
                attractor[:, 0], attractor[:, 1], bins=50
            )

            frame = go.Frame(
                data=[
                    go.Heatmap(
                        z=H.T, x=xedges, y=yedges, colorscale="Viridis", opacity=0.8
                    ),
                    go.Scatter(
                        x=attractor[:, 0],
                        y=attractor[:, 1],
                        mode="markers",
                        marker=dict(size=2, color="rgba(255, 255, 255, 0.3)"),
                    ),
                ],
                name=f"u{u}",
            )
            frames.append(frame)

    fig.frames = frames

    # Add a button to save current state to MLflow
    fig.add_annotation(
        x=0.5,
        y=-0.15,
        xref="paper",
        yref="paper",
        text="Export current view to MLflow by executing:<br>save_to_mlflow(u_value, batch_size)",
        showarrow=False,
        font=dict(size=12),
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="black",
        borderwidth=1,
        borderpad=4,
    )

    # Add a custom button (HTML) for saving to MLflow
    fig.update_layout(
        margin=dict(t=100, b=100),
        annotations=[
            dict(
                text="<b>How to use:</b><br>1. Adjust sliders to explore<br>2. Write down u and batch size values<br>3. Run save_to_mlflow(u, batch) in your terminal to save",
                align="left",
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0.01,
                y=-0.12,
                font=dict(size=12),
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="black",
                borderwidth=1,
                borderpad=4,
            )
        ],
    )

    return fig


# Create the interactive plot
fig = create_interactive_plot()

# Save as standalone HTML
# dashboard_path = "ikeda_plotly_explorer.html"
fig.write_html(
    dashboard_path,
    include_plotlyjs=True,
    full_html=True,
    include_mathjax=False,
    auto_open=True,  # Open the HTML file automatically
)

# Log to MLflow
with mlflow.start_run(run_name="interactive_dashboard"):
    mlflow.log_artifact(dashboard_path)

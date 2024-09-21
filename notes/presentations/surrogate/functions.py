def path_example():
    import pandas as pd
    import plotly.graph_objects as go
    df = pd.read_pickle('path_example.pkl')
    
    # Create a 3D scatter plot with Plotly with arrows
    
    # Sort the dataframe by 'loss' to find the 5th lowest and the lowest loss
    sorted_df = df.sort_values(by='loss').reset_index(drop=True)
    
    # Create the 3D scatter plot using Plotly Graph Objects
    fig = go.Figure()
    
    # Add scatter plot
    fig.add_trace(go.Scatter3d(
        x=df['x0'], y=df['x1'], z=df['loss'],
        mode='markers',
        marker=dict(
            size=10,
            color=df['loss'],
            colorscale='matter',
            showscale=False  # Hide the colorbar)
        ),
        text=df['loss'],
        customdata=df[['x0', 'x1', 'x2']],
        hovertemplate='[%{x}, %{y}, %{customdata[2]}] - loss = %{text:.2f}',
    ))
    
    # Add red arrows from the 5th lowest loss to the lowest loss point
    for i in range(4):
        fig.add_trace(go.Scatter3d(
            x=[sorted_df['x0'][i], sorted_df['x0'][i + 1]],
            y=[sorted_df['x1'][i], sorted_df['x1'][i + 1]],
            z=[sorted_df['loss'][i], sorted_df['loss'][i + 1]],
            mode='lines',
            line=dict(color='red', width=4),
            showlegend=False
        ))
    
    # Update layout
    fig.update_layout(
        title='Solution landscape for schedule with N = 6 and T = 3',
        scene=dict(
            xaxis_title='x0',
            yaxis_title='x1',
            zaxis_title='loss',
        ),
        width=800,
        height=800,
        showlegend=False
    )
    fig.show()

def gradient_boosting_example():
    from functions import gradient_boosting_example
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.tree import DecisionTreeRegressor
    
    np.random.seed(42)
    X = np.random.rand(100, 1) - 0.5
    y = 3 * X[:, 0] ** 2 + 0.05 * np.random.randn(100)  # y = 3x² + Gaussian noise
    
    tree_reg1 = DecisionTreeRegressor(max_depth=2, random_state=42)
    tree_reg1.fit(X, y)
    
    y2 = y - tree_reg1.predict(X)
    tree_reg2 = DecisionTreeRegressor(max_depth=2, random_state=43)
    tree_reg2.fit(X, y2)
    
    y3 = y2 - tree_reg2.predict(X)
    tree_reg3 = DecisionTreeRegressor(max_depth=2, random_state=44)
    tree_reg3.fit(X, y3)
    
    X_new = np.array([[-0.4], [0.], [0.5]])
    sum(tree.predict(X_new) for tree in (tree_reg1, tree_reg2, tree_reg3))
    
    def plot_predictions(regressors, X, y, axes, style,
                         label=None, data_style="b.", data_label=None):
        x1 = np.linspace(axes[0], axes[1], 500)
        y_pred = sum(regressor.predict(x1.reshape(-1, 1))
                     for regressor in regressors)
        plt.plot(X[:, 0], y, data_style, label=data_label)
        plt.plot(x1, y_pred, style, linewidth=2, label=label)
        if label or data_label:
            plt.legend(loc="upper center")
        plt.axis(axes)
    
    plt.figure(figsize=(11, 11))
    
    plt.subplot(3, 2, 1)
    plot_predictions([tree_reg1], X, y, axes=[-0.5, 0.5, -0.2, 0.8], style="g-",
                     label="$h_1(x_1)$", data_label="Training set")
    plt.ylabel("$y$  ", rotation=0)
    plt.title("Residuals and tree predictions")
    
    plt.subplot(3, 2, 2)
    plot_predictions([tree_reg1], X, y, axes=[-0.5, 0.5, -0.2, 0.8], style="r-",
                     label="$h(x_1) = h_1(x_1)$", data_label="Training set")
    plt.title("Ensemble predictions")
    
    plt.subplot(3, 2, 3)
    plot_predictions([tree_reg2], X, y2, axes=[-0.5, 0.5, -0.4, 0.6], style="g-",
                     label="$h_2(x_1)$", data_style="k+",
                     data_label="Residuals: $y - h_1(x_1)$")
    plt.ylabel("$y$  ", rotation=0)
    
    plt.subplot(3, 2, 4)
    plot_predictions([tree_reg1, tree_reg2], X, y, axes=[-0.5, 0.5, -0.2, 0.8],
                      style="r-", label="$h(x_1) = h_1(x_1) + h_2(x_1)$")
    
    plt.subplot(3, 2, 5)
    plot_predictions([tree_reg3], X, y3, axes=[-0.5, 0.5, -0.4, 0.6], style="g-",
                     label="$h_3(x_1)$", data_style="k+",
                     data_label="Residuals: $y - h_1(x_1) - h_2(x_1)$")
    plt.xlabel("$x_1$")
    plt.ylabel("$y$  ", rotation=0)
    
    plt.subplot(3, 2, 6)
    plot_predictions([tree_reg1, tree_reg2, tree_reg3], X, y,
                     axes=[-0.5, 0.5, -0.2, 0.8], style="r-",
                     label="$h(x_1) = h_1(x_1) + h_2(x_1) + h_3(x_1)$")
    plt.xlabel("$x_1$")
    plt.show()

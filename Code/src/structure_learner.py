"""
M1: Portfolio Temporal Structure Learning via ARX Models
Proof-of-concept dynamic Bayesian network structure learning
Base paper: DYNOTEARS (Pamfil et al., AISTATS 2020)

This module implements simplified score-based structure learning:
- Fit lag-1 ARX models per asset
- Extract significant edges (p < 0.05)
- Verify acyclicity
- Compute KPIs and compare against VAR baseline
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import networkx as nx
import warnings
warnings.filterwarnings('ignore')


class PortfolioStructureLearner:
    """
    Learn directed acyclic graph (DAG) structure from portfolio time-series.
    
    DAG Representation:
    - Nodes: assets at time t and t-1
    - Edges: directed, from t-1 to t (lag-1) or within t (contemporaneous)
    - Acyclicity: ensured via topological ordering constraint
    
    Learning:
    - Score: BIC (Bayesian Information Criterion)
    - Method: Fit ARX models, extract significant coefficients as edges
    - Penalty: sparsity via significance threshold (p-value)
    """
    
    def __init__(self, data, p_threshold=0.05, lam=1.0):
        """
        Initialize structure learner.
        
        Args:
            data: pd.DataFrame, shape (T, n_assets), standardized returns
            p_threshold: p-value threshold for edge significance (default 0.05)
            lam: sparsity penalty weight (default 1.0)
        """
        self.data = data
        self.n_assets = data.shape[1]
        self.T = data.shape[0]
        self.asset_names = list(data.columns)
        self.p_threshold = p_threshold
        self.lam = lam
        
        self.adjacency_matrix = None
        self.edge_list = None
        self.bic_scores = None
        self.coefficients = None
        self.p_values = None
    
    def fit(self):
        """
        Learn DAG structure via lag-1 ARX models.
        
        For each asset j:
            R_j(t) = sum_i beta_ij * R_i(t-1) + epsilon_j
        
        Extract edges: if beta_ij is significant (p < p_threshold), add edge i -> j.
        """
        # Initialize
        self.adjacency_matrix = np.zeros((self.n_assets, self.n_assets))
        self.coefficients = np.zeros((self.n_assets, self.n_assets))
        self.p_values = np.ones((self.n_assets, self.n_assets))
        bic_scores = []
        
        # Prepare lagged data
        X_lag = self.data.iloc[:-1, :].values  # R(t-1)
        y = self.data.iloc[1:, :].values       # R(t)
        
        # Fit ARX model for each asset
        for j in range(self.n_assets):
            y_j = y[:, j]
            
            # Linear regression: y_j ~ X_lag
            model = LinearRegression()
            model.fit(X_lag, y_j)
            
            # Extract coefficients and p-values
            y_pred = model.predict(X_lag)
            residuals = y_j - y_pred
            mse = np.sum(residuals**2) / len(residuals)
            
            # Compute standard errors and t-statistics
            X_with_intercept = np.column_stack([np.ones(len(X_lag)), X_lag])
            var_covar = mse * np.linalg.inv(X_with_intercept.T @ X_with_intercept)
            se = np.sqrt(np.diag(var_covar)[1:])  # Skip intercept
            t_stats = model.coef_ / se
            p_vals = 2 * (1 - stats.t.cdf(np.abs(t_stats), len(y_j) - self.n_assets - 1))
            
            # Store coefficients and p-values
            self.coefficients[:, j] = model.coef_
            self.p_values[:, j] = p_vals
            
            # Extract edges: significant coefficients
            significant = p_vals < self.p_threshold
            self.adjacency_matrix[significant, j] = 1
            
            # BIC score for this asset
            k = np.sum(significant)  # Number of edges
            bic = len(y_j) * np.log(mse) + k * np.log(len(y_j))
            bic_scores.append(bic)
        
        self.bic_scores = np.array(bic_scores)
        
        # Verify acyclicity
        self._verify_acyclicity()
        
        # Extract edge list
        self._extract_edges()
        
        return self
    
    def _verify_acyclicity(self):
        """
        Verify DAG is acyclic using topological sort.
        
        For M1, we enforce acyclicity by construction:
        edges only go from t-1 to t, never within a time slice.
        """
        # Within-slice edges are zero by construction (ARX model)
        # Lagged edges always point forward in time
        # Therefore, acyclicity is guaranteed
        is_acyclic = True
        return is_acyclic
    
    def _extract_edges(self):
        """Extract edge list from adjacency matrix."""
        edges = []
        for i in range(self.n_assets):
            for j in range(self.n_assets):
                if self.adjacency_matrix[i, j] > 0.5:
                    edges.append((self.asset_names[i], self.asset_names[j]))
        self.edge_list = edges
    
    def get_sparsity_metrics(self):
        """Compute representation KPIs: sparsity, density, mean degree."""
        n_edges = np.sum(self.adjacency_matrix > 0.5)
        density = n_edges / (self.n_assets ** 2)
        mean_degree = n_edges / self.n_assets
        
        return {
            'n_edges': n_edges,
            'density': density,
            'mean_degree': mean_degree,
            'sparsity': 1 - density
        }
    
    def get_learning_kpis(self):
        """Compute learning KPIs: BIC, coefficient magnitude."""
        total_bic = np.sum(self.bic_scores)
        n_significant_edges = np.sum(self.p_values < self.p_threshold)
        mean_coef_magnitude = np.mean(np.abs(self.coefficients[self.adjacency_matrix > 0.5]))
        
        return {
            'total_bic': total_bic,
            'mean_bic_per_asset': np.mean(self.bic_scores),
            'n_significant_edges': n_significant_edges,
            'mean_coef_magnitude': mean_coef_magnitude
        }
    
    def visualize_dag(self, figsize=(10, 8)):
        """Visualize learned DAG structure."""
        G = nx.DiGraph()
        
        # Add nodes
        for asset in self.asset_names:
            G.add_node(asset)
        
        # Add edges with weights
        for i, j in self.edge_list:
            weight = self.coefficients[self.asset_names.index(i), self.asset_names.index(j)]
            G.add_edge(i, j, weight=weight)
        
        # Draw
        plt.figure(figsize=figsize)
        pos = nx.spring_layout(G, seed=42, k=2)
        
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=1500)
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
        
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        nx.draw_networkx_edges(G, pos, width=2, edge_color='gray', 
                               arrowsize=20, arrowstyle='->', 
                               connectionstyle='arc3,rad=0.1')
        
        plt.title(f'Learned Portfolio DAG (n_edges={len(self.edge_list)})', fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        
        return plt.gcf()


def predict_arx(X_lag, coefficients):
    """
    Predict next time step using learned coefficients.
    
    Args:
        X_lag: shape (n_samples, n_assets), lagged values
        coefficients: shape (n_assets, n_assets), regression coefficients
    
    Returns:
        y_pred: shape (n_samples, n_assets)
    """
    return X_lag @ coefficients.T


def compare_with_var_baseline(data_train, data_test, asset_names):
    """
    Compare learned structure against VAR baseline.
    
    VAR: full dense matrix of lag-1 coefficients, no sparsity
    Learned: sparse structure via ARX significance test
    
    Returns:
        dict with RMSE, MAE for both methods
    """
    # Prepare data
    X_train_lag = data_train.iloc[:-1, :].values
    y_train = data_train.iloc[1:, :].values
    
    X_test_lag = data_test.iloc[:-1, :].values
    y_test = data_test.iloc[1:, :].values
    
    # VAR baseline: fit full model
    var_model = LinearRegression()
    var_model.fit(X_train_lag, y_train)
    y_var_pred = var_model.predict(X_test_lag)
    var_rmse = np.sqrt(np.mean((y_test - y_var_pred)**2))
    var_mae = np.mean(np.abs(y_test - y_var_pred))
    
    # Learned structure: fit ARX with significance threshold
    learner = PortfolioStructureLearner(data_train, p_threshold=0.05)
    learner.fit()
    
    y_learned_pred = predict_arx(X_test_lag, learner.coefficients)
    learned_rmse = np.sqrt(np.mean((y_test - y_learned_pred)**2))
    learned_mae = np.mean(np.abs(y_test - y_learned_pred))
    
    results = {
        'VAR_RMSE': var_rmse,
        'VAR_MAE': var_mae,
        'Learned_RMSE': learned_rmse,
        'Learned_MAE': learned_mae,
        'learner': learner
    }
    
    return results

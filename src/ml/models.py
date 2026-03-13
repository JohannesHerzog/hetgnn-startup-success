"""Neural network model definitions for startup success prediction (SeHGNN, HAN, GCN, MLP, XGBoost)."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn
from torch_geometric.nn import SAGEConv, HANConv, to_hetero, GraphConv
from typing import Union, Dict, List


class GraphConvEncoder(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers=2,
        activation_type="relu",
        normalize=True,
        dropout=0.0,
        aggr="add",
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList() if normalize else None

        if activation_type == "relu":
            self.activation = torch.nn.ReLU()
        elif activation_type == "prelu":
            self.activation = torch.nn.PReLU()
        else:
            raise ValueError("Unsupported activation type. Choose 'relu' or 'prelu'.")

        # First layer
        self.convs.append(GraphConv(in_channels, hidden_channels, aggr=aggr))
        if normalize:
            self.norms.append(torch_geometric.nn.BatchNorm(hidden_channels))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GraphConv(hidden_channels, hidden_channels, aggr=aggr))
            if normalize:
                self.norms.append(torch_geometric.nn.BatchNorm(hidden_channels))

        # Last layer (if num_layers > 1)
        if num_layers > 1:
            self.convs.append(GraphConv(hidden_channels, out_channels, aggr=aggr))
            if normalize:
                self.norms.append(torch_geometric.nn.BatchNorm(out_channels))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if self.norms is not None:
                x = self.norms[i](x)
            
            if i < self.num_layers - 1:
                x = self.activation(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class ArcFace(torch.nn.Module):
    """
    ArcFace: Additive Angular Margin Loss.
    """
    def __init__(self, in_features, out_features, s=64.0, m=0.5, easy_margin=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        
        # Weights (Class Centers)
        self.weight = torch.nn.Parameter(torch.FloatTensor(out_features, in_features))
        torch.nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cosine(theta) & phi(theta) ---------------------------
        # input: [batch_size, embedding_dim] (already normalized in SeHGNN usually, but safe to re-norm)
        # weight: [num_classes, embedding_dim]
        
        # Normalize input and weights
        # Note: SeHGNN retrieval head already normalizes output, but ArcFace requires strict normalization
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))

        # phi = cos(theta + m)
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output


class BaseGNN(torch.nn.Module):
    def __init__(self, hidden_channels, target_mode, num_classes, activation_type="relu"):
        super().__init__()
        self.target_mode = target_mode
        self.num_classes = num_classes
        self.hidden_channels = hidden_channels
        self._init_activation(activation_type)
        self._init_heads()

    def _init_activation(self, activation_type):
        if activation_type == "relu":
            self.activation = torch.nn.ReLU()
        elif activation_type == "prelu":
            self.activation = torch.nn.PReLU()
        elif activation_type == "leaky_relu":
            self.activation = torch.nn.LeakyReLU()
        else:
            raise ValueError(f"Unsupported activation type: {activation_type}. Choose 'relu', 'prelu', or 'leaky_relu'.")

    def _get_activation(self, name):
        """Get activation function by name (helper for retrieval head)"""
        activations = {
            "relu": torch.nn.ReLU(),
            "prelu": torch.nn.PReLU(),
            "gelu": torch.nn.GELU(),
            "tanh": torch.nn.Tanh(),
            "leaky_relu": torch.nn.LeakyReLU()
        }
        return activations.get(name, torch.nn.ReLU())
    
    def _init_retrieval_head(self, config, model_name, input_dim):
        """Initialize retrieval projection head (SimCLR/CLIP pattern)"""
        self.detach_retrieval_head = config["models"][model_name].get("detach_retrieval_head", False)
        if self.detach_retrieval_head:
             print(f"Retrieval head: gradient stop enabled (backbone detached)")
             
        proj_config = config["models"][model_name].get("retrieval_projection", {})
        
        hidden_dim = proj_config.get("hidden_dim", 128)
        output_dim = proj_config.get("output_dim", input_dim)
        dropout = proj_config.get("dropout", 0.3)
        use_bn = proj_config.get("use_batch_norm", True)
        activation = proj_config.get("activation", "relu")
        
        # Build projection head (2-layer MLP like SimCLR)
        layers = [
            torch.nn.Linear(input_dim, hidden_dim),
            self._get_activation(activation),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, output_dim)
        ]
        
        if use_bn:
            layers.append(torch.nn.BatchNorm1d(output_dim))
        
        self.retrieval_proj = torch.nn.Sequential(*layers)
        
        print(f"Initialized retrieval projection head: {input_dim} -> {hidden_dim} -> {output_dim}")

    def _init_heads(self):
        if self.target_mode == "binary_prediction":
            self.output_head = torch.nn.Linear(self.hidden_channels, 1)
        elif self.target_mode == "multi_prediction":
            self.output_head = torch.nn.Linear(self.hidden_channels, self.num_classes)
        elif self.target_mode == "multi_task":
            self.task_binary_encoder = torch.nn.Linear(self.hidden_channels, self.hidden_channels)
            self.task_multi_encoder = torch.nn.Linear(self.hidden_channels, self.hidden_channels)
            self.output_binary = torch.nn.Linear(self.hidden_channels, 1)
            self.output_multi = torch.nn.Linear(self.hidden_channels, self.num_classes)
        elif self.target_mode == "multi_label":
            # 3 Binary Classification Heads w/ optional dedicated encoders
            self.head_fund = torch.nn.Linear(self.hidden_channels, 1)
            self.head_acq = torch.nn.Linear(self.hidden_channels, 1)
            self.head_ipo = torch.nn.Linear(self.hidden_channels, 1)
        elif self.target_mode == "masked_multi_task":
            # Tower 1: Momentum (Funding)
            self.head_momentum = torch.nn.Sequential(
                torch.nn.Linear(self.hidden_channels, self.hidden_channels),
                self.activation,
                torch.nn.Linear(self.hidden_channels, 1)
            )
            # Tower 2: Liquidity (Acq/IPO)
            self.head_liquidity = torch.nn.Sequential(
                torch.nn.Linear(self.hidden_channels, self.hidden_channels),
                self.activation,
                torch.nn.Linear(self.hidden_channels, 1)
            )
        else:
            raise ValueError(f"Unsupported target_mode: {self.target_mode}")

    def _apply_heads(self, startup_x, retrieval_labels=None):
        if self.target_mode == "binary_prediction":
            return {
                "startup": {
                    "output": self.output_head(startup_x),
                    "embedding": startup_x
                }
            }
        elif self.target_mode == "multi_prediction":
            return {
                "startup": {
                    "output": self.output_head(startup_x),
                    "embedding": startup_x
                }
            }
        elif self.target_mode == "multi_task":
            binary_x = self.activation(self.task_binary_encoder(startup_x))
            multi_x = self.activation(self.task_multi_encoder(startup_x))
            return {
                "binary_output": {"startup": self.output_binary(binary_x).squeeze(-1)},
                "multi_class_output": {"startup": self.output_multi(multi_x)},
                "embedding": {"startup": startup_x}
            }
        elif self.target_mode == "multi_label":
             # Apply 3 separate heads
             out_fund = self.head_fund(startup_x).squeeze(-1)
             out_acq = self.head_acq(startup_x).squeeze(-1)
             out_ipo = self.head_ipo(startup_x).squeeze(-1)
             
             # Return concatenated output [N, 3] usually, or dictionary?
             # For simpler loss handling, let's return a dictionary but also support concatenated
             out_combined = torch.stack([out_fund, out_acq, out_ipo], dim=1)
             
             return {
                 "multi_label_output": {"startup": out_combined},
                 "embedding": {"startup": startup_x},
                 # Individual outputs if needed for flexibility
                 "out_fund": out_fund,
                 "out_acq": out_acq,
                 "out_ipo": out_ipo
             }
        elif self.target_mode == "masked_multi_task":
             # Apply 2 MLP heads
             out_mom = self.head_momentum(startup_x).squeeze(-1)
             out_liq = self.head_liquidity(startup_x).squeeze(-1)
             
             # Stack for convenient tensor access: [Mom, Liq]
             out_combined = torch.stack([out_mom, out_liq], dim=1)
             
             output = {
                 "masked_multi_task_output": {"startup": out_combined},
                 "embedding": {"startup": startup_x},
                 "out_mom": out_mom, # Momentum Logic
                 "out_liq": out_liq  # Liquidity Logic
             }
             
             # Add retrieval embedding if projection head exists (SimCLR/CLIP pattern)
             if hasattr(self, 'retrieval_proj') and self.retrieval_proj is not None:
                 
                 x_in = startup_x
                 # Gradient Stop: Detach backbone if configured to protect main task
                 if getattr(self, 'detach_retrieval_head', False):
                     x_in = x_in.detach()
                 
                 retrieval_emb = self.retrieval_proj(x_in)  # Project
                 retrieval_emb = F.normalize(retrieval_emb, p=2, dim=1)  # Normalize (like CLIP)
                 output["retrieval_embedding"] = {"startup": retrieval_emb}
                 
                 # ArcFace Logic
                 if hasattr(self, 'arcface_head') and retrieval_labels is not None:
                     # retrieval_labels: [Batch]
                     # arcface_head(emb, labels) -> logits
                     # Note: ArcFace requires labels to compute Margin Loss during Training.
                     # During eval, we usually just want embeddings.
                     if self.training:
                         arcface_logits = self.arcface_head(retrieval_emb, retrieval_labels)
                         output["arcface_logits"] = {"startup": arcface_logits}
             
             return output
        else:
            raise ValueError(f"Unsupported target_mode: {self.target_mode}")


class GAT(BaseGNN):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers=2,
        v2=True,
        normalize=True,
        activation="prelu",
        jumping_knowledge="cat",
        add_self_loops=False,
        target_mode="multi_prediction",
        num_classes=2,
        metadata=None,
        aggr="sum",
        dropout=0.0,
        heads=1,
        negative_slope=0.2,
    ):
        super().__init__(hidden_channels, target_mode, num_classes, activation)

        # Initialize GAT Encoder
        # We use the standard PyG GAT implementation
        norm = torch_geometric.nn.BatchNorm(hidden_channels) if normalize else None
        act = torch.nn.PReLU() if activation == "prelu" else torch.nn.ReLU()
        
        self.encoder = torch_geometric.nn.GAT(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels, # Encoder outputs hidden_channels
            num_layers=num_layers,
            v2=v2,
            norm=norm,
            act=act,
            jk=jumping_knowledge,
            add_self_loops=add_self_loops,
            dropout=dropout,
            heads=heads,
            negative_slope=negative_slope,
        )

        # Wrap encoder with to_hetero
        if metadata is not None:
            self.encoder = to_hetero(self.encoder, metadata, aggr=aggr)

    def forward(self, x_dict, edge_index_dict):
        # Get embeddings from heterogeneous encoder
        # x_dict will contain embeddings for all node types
        embeddings_dict = self.encoder(x_dict, edge_index_dict)
        
        # Extract startup embeddings
        startup_x = embeddings_dict['startup']
        
        return self._apply_heads(startup_x)


class GCN(BaseGNN):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers=2,
        normalize=True,
        activation="prelu",
        jumping_knowledge="cat",
        target_mode="multi_prediction",
        num_classes=2,
        metadata=None,
        aggr="sum",
        dropout=0.0,
        add_self_loops=False, # Kept for interface consistency, but GraphConv handles it differently or implicitly
    ):
        super().__init__(hidden_channels, target_mode, num_classes, activation)

        # Initialize GraphConv Encoder
        self.encoder = GraphConvEncoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,
            num_layers=num_layers,
            activation_type=activation,
            normalize=normalize,
            dropout=dropout,
            aggr=aggr, # GraphConv supports 'add', 'mean', 'max'
        )

        # Wrap encoder with to_hetero
        if metadata is not None:
            self.encoder = to_hetero(self.encoder, metadata, aggr=aggr)

    def forward(self, x_dict, edge_index_dict):
        # Get embeddings from heterogeneous encoder
        # x_dict will contain embeddings for all node types
        embeddings_dict = self.encoder(x_dict, edge_index_dict)
        
        # Extract startup embeddings
        startup_x = embeddings_dict['startup']
        
        return self._apply_heads(startup_x)


class HeteroMLP(BaseGNN):
    """
    Heterogeneous MLP Baseline (Startup Features Only).
    
    This model serves as a strict baseline to evaluate the predictive power of intrinsic startup attributes
    without any graph connectivity information. It ignores all neighbor features (investors, founders, etc.)
    and only uses the features of the startup nodes themselves.
    
    Architecture:
    1. Feature Projection: Linearly projects startup features to a hidden dimension.
    2. Prediction: Passes the projected features through a standard MLP to make predictions.
    """
    def __init__(
        self,
        hidden_channels,
        target_mode="multi_prediction",
        num_classes=2,
        activation_type="relu",
        normalize=True,
        dropout=0.0,
        metadata=None, # Kept for interface consistency
    ):
        super().__init__(hidden_channels, target_mode, num_classes, activation_type)
        
        self.dropout = dropout
        
        # 1. Feature Projection: Project startup features to hidden_channels
        # We use torch_geometric.nn.Linear which supports lazy initialization (-1)
        self.startup_projection = torch_geometric.nn.Linear(-1, hidden_channels)

        # 2. MLP for final prediction
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            self.activation,
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_channels, hidden_channels),
            self.activation,
            torch.nn.Dropout(dropout),
        )
        
        if normalize:
            self.norms = torch.nn.ModuleList([
                torch_geometric.nn.BatchNorm(hidden_channels),
                torch_geometric.nn.BatchNorm(hidden_channels)
            ])
        else:
            self.norms = None

    def forward(self, x_dict, edge_index_dict):
        # 1. Extract and project startup features
        x = x_dict['startup']
        x = self.startup_projection(x)
        
        # 2. Pass through MLP
        if self.norms:
            x = self.norms[0](x)
            
        x = self.mlp[0](x) # Linear
        x = self.mlp[1](x) # Act
        x = self.mlp[2](x) # Dropout
        
        if self.norms:
            x = self.norms[1](x)
            
        x = self.mlp[3](x) # Linear
        x = self.mlp[4](x) # Act
        x = self.mlp[5](x) # Dropout
        
        return self._apply_heads(x)


class SageEncoder(torch.nn.Module):
    def __init__(
        self,
        hidden_channels,
        num_layers=2,
        activation_type="relu",
        normalize=True,
        dropout=0.3,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = torch.nn.ModuleList()

        if activation_type == "relu":
            self.activation = torch.nn.ReLU()
        elif activation_type == "prelu":
            self.activation = torch.nn.PReLU()
        else:
            raise ValueError("Unsupported activation type. Choose 'relu' or 'prelu'.")

        # Shared encoder
        for _ in range(num_layers):
            self.convs.append(SAGEConv((-1, -1), hidden_channels, normalize=normalize))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < self.num_layers - 1:  # Apply activation and dropout except for last layer
                x = self.activation(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class XGBoostAdapter:
    def __init__(
        self,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0,
        reg_alpha=0,
        reg_lambda=1,
        objective="binary:logistic",
        tree_method="hist",
        target_mode="multi_prediction",
        num_classes=2,
        **kwargs
    ):
        self.params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "gamma": gamma,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "objective": objective,
            "tree_method": tree_method,
            **kwargs
        }
        self.target_mode = target_mode
        self.num_classes = num_classes
        self.model = None

    def fit(self, X, y, eval_set=None, **kwargs):
        import xgboost as xgb
        
        # Handle multi-label/multi-class objectives if needed
        # For now, assuming binary classification per task or standard multi-class
        
        self.model = xgb.XGBClassifier(**self.params)
        verbose = kwargs.get("verbose", False) # Default to False if not provided, or respect passed val
        self.model.fit(X, y, eval_set=eval_set, verbose=verbose)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def to(self, device):
        # Dummy method for compatibility with Trainer
        return self
    
    def __call__(self, *args, **kwargs):
        # Dummy forward for compatibility if called inadvertently
        pass


class SageGNN(BaseGNN):
    def __init__(
        self,
        hidden_channels,
        num_layers=2,
        activation_type="relu",
        normalize=True,
        target_mode="multi_prediction",
        num_classes=4,
        dropout=0.3,
        metadata=None,
        aggr="mean",
    ):
        super().__init__(hidden_channels, target_mode, num_classes, activation_type)
        
        # Initialize Encoder
        self.encoder = SageEncoder(
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            activation_type=activation_type,
            normalize=normalize,
            dropout=dropout,
        )
        
        # Wrap encoder with to_hetero
        if metadata is not None:
            self.encoder = to_hetero(self.encoder, metadata, aggr=aggr)

    def forward(self, x_dict, edge_index_dict):
        # Get embeddings from heterogeneous encoder
        # x_dict will contain embeddings for all node types
        embeddings_dict = self.encoder(x_dict, edge_index_dict)
        
        # Extract startup embeddings
        startup_x = embeddings_dict['startup']
        
        return self._apply_heads(startup_x)


class HAN(BaseGNN):
    """
    Heterogeneous Graph Attention Network (HAN) for startup success prediction.
    Implements multi-layer HAN with different output modes for binary, multi-class, or multi-task prediction.
    """
    def __init__(
        self,
        in_channels: Union[int, Dict[str, int]],
        hidden_channels: int,
        metadata,  # Graph metadata (node_types, edge_types)
        num_layers: int = 2,
        heads: int = 8,
        negative_slope: float = 0.2,
        dropout: float = 0.2,
        activation_type: str = "relu",
        target_mode: str = "binary_prediction",  # 'binary_prediction', 'multi_prediction', or 'multi_task'
        num_classes: int = 2,
    ):
        super().__init__(hidden_channels, target_mode, num_classes, activation_type)
        self.num_layers = num_layers
        
        # Multi-layer HAN convolutions
        self.convs = torch.nn.ModuleList()
        
        # First layer
        self.convs.append(
            HANConv(
                in_channels=in_channels,
                out_channels=hidden_channels,
                metadata=metadata,
                heads=heads,
                negative_slope=negative_slope,
                dropout=dropout,
            )
        )
        
        # Additional layers (all with same hidden dimensions)
        for _ in range(num_layers - 1):
            self.convs.append(
                HANConv(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    metadata=metadata,
                    heads=heads,
                    negative_slope=negative_slope,
                    dropout=dropout,
                )
            )

        # Residual connection
        self.residual = True
        self.residual_proj = torch_geometric.nn.Linear(-1, hidden_channels)
        
        # Gating mechanism
        # If True, learns a gate to balance Residual (Self) vs HAN (Graph)
        self.use_gating = True 
        if self.use_gating:
            # Gate computes a weight z in [0, 1]
            # h_final = z * h_residual + (1-z) * h_graph
            self.gate_linear = torch.nn.Linear(2 * hidden_channels, 1)
            
        # Residual Dropout
        # Dropping out the residual forces the model to rely on the graph path
        self.residual_dropout = 0.5 

    def forward(self, x_dict: Dict[str, torch.Tensor], edge_index_dict: Dict[tuple, torch.Tensor]):
        """
        Forward pass through the HAN layers.
        """
        # Capture original startup features for residual
        x_startup_input = x_dict['startup']

        # Pass through HAN layers
        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict)
            
            # Apply activation AND dropout between layers (except last)
            if i < self.num_layers - 1:
                for node_type in x_dict:
                    if x_dict[node_type] is not None:
                        # 1. Activation
                        x_dict[node_type] = self.activation(x_dict[node_type])
                        
                        # 2. Feature Dropout
                        x_dict[node_type] = torch.nn.functional.dropout(
                            x_dict[node_type], p=0.2, training=self.training
                        )

        # Extract startup node embeddings (Graph Path)
        h_graph = x_dict['startup']
        
        # Calculate Residual Embedding (Self Path)
        h_residual = self.residual_proj(x_startup_input)
        
        # Apply Residual Connection
        if self.residual:
            if self.training and self.residual_dropout > 0:
                # Apply dropout to the residual path to force graph usage
                h_residual = torch.nn.functional.dropout(h_residual, p=self.residual_dropout, training=True)
            
            if self.use_gating:
                # Compute Gate z = sigmoid(Linear([h_residual, h_graph]))
                # Concatenate along feature dimension
                combined = torch.cat([h_residual, h_graph], dim=-1)
                z = torch.sigmoid(self.gate_linear(combined))
                
                # Store mean gate value for debugging
                # z near 1.0 means relying on RESIDUAL (Self)
                # z near 0.0 means relying on GRAPH (Neighbors)
                self.last_gate_mean = z.mean().item()
                
                # Weighted combination
                # z determines how much of the RESIDUAL (Self) to keep
                startup_x = z * h_residual + (1 - z) * h_graph
            else:
                # Simple addition
                startup_x = h_graph + h_residual
        else:
            startup_x = h_graph
        
        return self._apply_heads(startup_x)

    def get_embeddings(self, x_dict, edge_index_dict):
        """Extract startup node embeddings before the final classification head."""
        # Capture original startup features for residual
        x_startup_input = x_dict['startup']

        # Pass through HAN layers
        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict)
            
            # Apply activation AND dropout between layers (except last)
            if i < self.num_layers - 1:
                for node_type in x_dict:
                    if x_dict[node_type] is not None:
                        x_dict[node_type] = self.activation(x_dict[node_type])
                        x_dict[node_type] = torch.nn.functional.dropout(
                            x_dict[node_type], p=0.2, training=self.training
                        )

        # Return startup embeddings with residual
        h_graph = x_dict['startup']
        h_residual = self.residual_proj(x_startup_input)
        
        if self.residual:
            if self.use_gating:
                combined = torch.cat([h_residual, h_graph], dim=-1)
                z = torch.sigmoid(self.gate_linear(combined))
                startup_x = z * h_residual + (1 - z) * h_graph
            else:
                startup_x = h_graph + h_residual
        else:
            startup_x = h_graph
            
        return startup_x

    def get_semantic_attention_weights(self, x_dict, edge_index_dict):
        """
        Extract semantic attention weights (metapath importances) from the HAN layers.
        
        Args:
            x_dict: Dictionary of node features
            edge_index_dict: Dictionary of edge indices
            
        Returns:
            Dictionary containing weights per layer and per destination node type
        """
        weights_dict = {}
        
        # Pass through HAN layers
        for i, conv in enumerate(self.convs):

            out, details = conv(x_dict, edge_index_dict, return_semantic_attention_weights=True)

            layer_weights = {}
            for node_type, weight_tensor in details.items():
                # Map weights to metapath names
                dest_metapaths = [
                    edge_type for edge_type in self.convs[i].metadata[1] 
                    if edge_type[2] == node_type
                ]
                
                # Verify counts match
                if len(dest_metapaths) != len(weight_tensor):
                    print(f"Warning: Mismatch in metapath count for {node_type} in layer {i}")
                    continue
                    
                # Create dict {metapath_name: weight}
                node_weights = {}
                for j, edge_type in enumerate(dest_metapaths):
                    # edge_type is (src, rel, dst)
                    metapath_name = edge_type[1]
                    node_weights[metapath_name] = weight_tensor[j].item()
                    
                layer_weights[node_type] = node_weights
                
            weights_dict[f"layer_{i}"] = layer_weights
            
            # Update x_dict for next layer (same as in forward)
            x_dict = out
            if i < self.num_layers - 1:
                for node_type in x_dict:
                    if x_dict[node_type] is not None:
                        x_dict[node_type] = self.activation(x_dict[node_type])
                        x_dict[node_type] = torch.nn.functional.dropout(
                            x_dict[node_type], p=0.2, training=self.training
                        )
                        
        return weights_dict


class FocalLoss(nn.Module):
    """
    Focal Loss Implementation
    Source: https://github.com/mathiaszinnen/focal_loss_torch
    """

    def __init__(self, gamma=0, alpha=None, size_average=True):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)
            input = input.transpose(1, 2)
            input = input.contiguous().view(-1, input.size(2))
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=-1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = logpt.detach().exp()

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * at.detach()

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

class RandomBaseline(BaseGNN):
    """
    Random Baseline: Generates random predictions.
    """
    def __init__(
        self,
        hidden_channels, # Ignored, kept for interface compatibility
        target_mode="multi_prediction",
        num_classes=2,
        **kwargs
    ):
        super().__init__(hidden_channels, target_mode, num_classes)
        
    def forward(self, x_dict, edge_index_dict, **kwargs):
        # Get batch size from startup features
        batch_size = x_dict['startup'].shape[0]
        device = x_dict['startup'].device
        
        # Generate random outputs
        # We generate logits
        if self.target_mode == "binary_prediction":
            # Output shape: [batch_size, 1]
            out = torch.randn(batch_size, 1, device=device)
            return {
                "startup": {
                    "output": out,
                    "embedding": x_dict['startup'] # Dummy embedding
                }
            }
        elif self.target_mode == "multi_prediction":
            # Output shape: [batch_size, num_classes]
            out = torch.randn(batch_size, self.num_classes, device=device)
            return {
                "startup": {
                    "output": out,
                    "embedding": x_dict['startup']
                }
            }
        elif self.target_mode == "multi_task":
            # Binary and Multi-class
            bin_out = torch.randn(batch_size, 1, device=device)
            multi_out = torch.randn(batch_size, self.num_classes, device=device)
            return {
                "binary_output": {"startup": bin_out.squeeze(-1)},
                "multi_class_output": {"startup": multi_out},
                "embedding": {"startup": x_dict['startup']}
            }
        elif self.target_mode == "masked_multi_task":
             # Tower 1: Momentum (Funding)
             out_mom = torch.randn(batch_size, 1, device=device)
             # Tower 2: Liquidity (Acq/IPO)
             out_liq = torch.randn(batch_size, 1, device=device)
             
             # Stack for convenient tensor access: [Mom, Liq]
             out_combined = torch.stack([out_mom.squeeze(-1), out_liq.squeeze(-1)], dim=1)
             
             return {
                 "masked_multi_task_output": {"startup": out_combined},
                 "embedding": {"startup": x_dict['startup']},
                 "out_mom": out_mom.squeeze(-1),
                 "out_liq": out_liq.squeeze(-1) 
             }
        else:
             raise ValueError(f"Unsupported target_mode: {self.target_mode}")


class Transformer(torch.nn.Module):
    """
    The transformer-based semantic fusion in SeHGNN.
    Adapted from: src/other/SeHGNN/hgb/model.py
    """
    def __init__(self, n_channels, num_heads=1, att_drop=0., act='none', temperature=1.0,
                 gamma_init=0.0, gamma_learnable=True):
        super().__init__()
        self.n_channels = n_channels
        self.num_heads = num_heads
        self.temperature = temperature
        assert self.n_channels % (self.num_heads * 4) == 0

        self.query = torch.nn.Linear(self.n_channels, self.n_channels//4)
        self.key   = torch.nn.Linear(self.n_channels, self.n_channels//4)
        self.value = torch.nn.Linear(self.n_channels, self.n_channels)

        if gamma_learnable:
            self.gamma = torch.nn.Parameter(torch.tensor([float(gamma_init)]))
        else:
            self.register_buffer('gamma', torch.tensor([float(gamma_init)]))
        self._gamma_learnable = gamma_learnable
        self._gamma_init = gamma_init
        self.att_drop = torch.nn.Dropout(att_drop)
        if act == 'sigmoid':
            self.act = torch.nn.Sigmoid()
        elif act == 'relu':
            self.act = torch.nn.ReLU()
        elif act == 'leaky_relu':
            self.act = torch.nn.LeakyReLU(0.2)
        elif act == 'prelu':
            self.act = torch.nn.PReLU()
        elif act == 'none':
            self.act = lambda x: x
        else:
            raise ValueError(f'Unrecognized activation function {act} for class Transformer')

        self.reset_parameters()

    def reset_parameters(self):
        for k, v in self._modules.items():
            if hasattr(v, 'reset_parameters'):
                v.reset_parameters()
        if self._gamma_learnable:
            self.gamma.data.fill_(self._gamma_init)

    def forward(self, x, mask=None):
        B, M, C = x.size() # batchsize, num_metapaths, channels
        H = self.num_heads

        f = self.query(x).view(B, M, H, -1).permute(0,2,1,3) # [B, H, M, -1]
        g = self.key(x).view(B, M, H, -1).permute(0,2,3,1)   # [B, H, -1, M]
        h = self.value(x).view(B, M, H, -1).permute(0,2,1,3) # [B, H, M, -1]

        beta = F.softmax(self.act(f @ g / math.sqrt(f.size(-1)) / self.temperature), dim=-1) # [B, H, M, M(normalized)]
        beta = self.att_drop(beta)

        if mask is not None:
            beta = beta * mask.view(B, 1, 1, M)
            beta = beta / (beta.sum(-1, keepdim=True) + 1e-12)

        o = self.gamma * (beta @ h) # [B, H, M, -1]

        # Return output AND attention weights (beta)
        return o.permute(0,2,1,3).reshape((B, M, C)) + x, beta


class SeHGNN(BaseGNN):
    """
    Simple and Efficient Heterogeneous Graph Neural Network (SeHGNN).
    
    Adapts the SeHGNN architecture to work with NeighborLoader by performing
    on-the-fly mean aggregation of neighbors instead of pre-computation.
    """
    def __init__(
        self,
        in_channels: Union[int, Dict[str, int]],
        hidden_channels: int,
        metadata, # Graph metadata (node_types, edge_types)
        num_layers: int = 2, # Used for MLP layers
        heads: int = 1, # Transformer heads
        dropout: float = 0.5,
        input_drop: float = 0.1,
        att_drop: float = 0.0,
        activation_type: str = "relu",
        target_mode: str = "binary_prediction",
        num_classes: int = 2,
        aggregation_method: str = "mean",
        use_residual: bool = True,
        transformer_activation: str = "none",
        use_self_loop: bool = True,
        config: dict = None,  # Config for retrieval head
        model_name: str = "SeHGNN",  # Model name for config lookup
        attention_temperature: float = 1.0, # New param
        num_hops: int = 1, # Number of aggregation hops for startup→startup edges
        gamma_init: float = 0.0,  # Transformer gamma init value
        gamma_learnable: bool = True,  # Whether gamma is learnable
        channel_masking: bool = False,  # Mask empty metapath channels in attention
        use_layer_norm: bool = False,  # LayerNorm before fc_after_concat
        **kwargs
    ):
        super().__init__(hidden_channels, target_mode, num_classes, activation_type)

        self.metadata = metadata
        self.hidden_channels = hidden_channels
        self.dropout = dropout
        self.channel_masking = channel_masking
        self.input_drop = torch.nn.Dropout(input_drop)
        self.aggregation_method = aggregation_method
        self.use_residual = use_residual
        self.num_hops = num_hops
        self.use_self_loop = use_self_loop
        
        # 1. Identify Metapaths (Edges pointing to 'startup')
        # We treat each edge type (src, rel, dst) where dst='startup' as a "metapath"
        # plus the 'startup' node itself (self-loop equivalent)
        self.metapaths = []
        
        # Add self (startup)
        if self.use_self_loop:
            self.metapaths.append("self")
        
        # Collect all candidate edges (both original and materialized metapaths)
        candidate_edges = []
        edge_types = metadata[1]
        for src, rel, dst in edge_types:
            if dst == 'startup':
                candidate_edges.append((src, rel, dst))
        
        # CAP TOTAL METAPATHS
        # Check config for max_metapaths to prevent OOM
        max_mps = 50  # Default fallback
        drop_list = []

        if config is not None:
            # Standardized config path: metapath_discovery.automatic
            auto_config = config.get('metapath_discovery', {}).get('automatic', {})
            max_mps = auto_config.get('max_metapaths', 50)

            # Standardized ablation path: metapath_discovery.automatic.ablation.drop_edges
            ablation_config = auto_config.get('ablation', {})
            drop_list = ablation_config.get('drop_edges', [])

        if len(candidate_edges) > 0:
            print(f"   Ablation: Dropping edge types: {drop_list}")
            
            # SELECTION LOGIC: 
            # 1. Filter out dropped edges
            # 2. Prioritize Base Edges (Keep ALL non-dropped)
            # 3. Fill remaining with Discovered
            
            discovered = []
            base = []
            
            for mp in candidate_edges:
                src, rel, dst = mp
                
                # Check for drop (Ablation)
                should_drop = False
                for drop_key in drop_list:
                    if drop_key in rel: # Partial match for relation name
                        should_drop = True
                        break
                if should_drop:
                    continue

                if "_via_" in rel or "_to_startup_" in rel:
                    discovered.append(mp)
                else:
                    base.append(mp)
            
            print(f"   Breakdown: {len(discovered)} discovered, {len(base)} base edges (after ablation)")
            
            final_selection = []
            
            # Keep all base edges, fill remaining budget with discovered paths
            if len(base) > max_mps:
                print(f"WARNING: Base edges ({len(base)}) exceed max_metapaths ({max_mps}). Capping base edges.")
                final_selection.extend(base[:max_mps])
            else:
                final_selection.extend(base)
            
            # 2. Fill remaining budget with Discovered paths
            remaining_slots = max_mps - len(final_selection)
            if remaining_slots > 0:
                num_discovered = min(len(discovered), remaining_slots)
                final_selection.extend(discovered[:num_discovered])
                print(f"   Selected: {len(base)} base + {num_discovered} discovered paths")
            else:
                print(f"   Selected: All {len(final_selection)} base edges (Discovered paths dropped due to capacity)")
                
            candidate_edges = final_selection
            
        self.metapaths.extend(candidate_edges)
                
        print(f"SeHGNN initialized with {len(self.metapaths)} channels: {self.metapaths}")
        print(f"  Aggregation: {self.aggregation_method}, Residual: {self.use_residual}, Transformer Act: {transformer_activation}, Self Loop: {self.use_self_loop}, Hops: {self.num_hops}")
        
        self.num_channels = len(self.metapaths)
        
        # 2. Feature Projection (Linear per metapath)
        # We need to project each input feature dimension to hidden_channels
        self.projectors = torch.nn.ModuleDict()
        
        # Self projector
        if isinstance(in_channels, dict):
            startup_dim = in_channels['startup']
        else:
            startup_dim = in_channels
            
        self.projectors["self"] = torch.nn.Linear(startup_dim, hidden_channels)
        
        # Neighbor projectors
        for mp in self.metapaths:
            if mp == "self": continue
            src, rel, dst = mp
            
            if isinstance(in_channels, dict):
                src_dim = in_channels[src]
            else:
                src_dim = in_channels
                
            # Key must be string for ModuleDict
            key = f"{src}__{rel}__{dst}"
            self.projectors[key] = torch.nn.Linear(src_dim, hidden_channels)
            
        # 3. Transformer Semantic Fusion
        # Fuses [Batch, Num_Metapaths, Hidden] -> [Batch, Num_Metapaths, Hidden]
        self.semantic_fusion = Transformer(
            hidden_channels,
            num_heads=heads,
            att_drop=att_drop,
            act=transformer_activation,
            temperature=attention_temperature,
            gamma_init=gamma_init,
            gamma_learnable=gamma_learnable
        )
        
        # 4. Optional LayerNorm before projection
        if use_layer_norm:
            self.pre_fc_norm = torch.nn.LayerNorm(self.num_channels * hidden_channels)
        else:
            self.pre_fc_norm = None

        # 5. Aggregation after Transformer
        # SeHGNN concatenates and projects, or just flattens.
        # The original code does:
        # x = self.fc_after_concat(x.reshape(B, -1))
        # where fc_after_concat reduces (Num_Channels * Hidden) -> Hidden
        self.fc_after_concat = torch.nn.Linear(self.num_channels * hidden_channels, hidden_channels)
        
        # 5. Task MLP (Classifier)
        # We use a simple MLP similar to the original code
        # Note: BaseGNN._init_heads handles the final output layer, so we just need the intermediate layers
        self.task_mlp = torch.nn.Sequential(
            torch.nn.PReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.BatchNorm1d(hidden_channels),
            torch.nn.PReLU(),
            torch.nn.Dropout(dropout)
        )
        
        # Residual connection (optional in original, we'll add it for stability)
        if self.use_residual:
            self.res_fc = torch.nn.Linear(startup_dim, hidden_channels)
        
        # Initialize retrieval head if enabled (SimCLR/CLIP pattern)
        if config is not None:
            use_retrieval_head = config["models"][model_name].get("use_retrieval_head", False)
            retrieval_loss_type = config["train"]["loss"].get("retrieval_loss_type", "contrastive")
            
            if use_retrieval_head:
                # Initialize Projection Head
                self._init_retrieval_head(config, model_name, hidden_channels)
                
                # Initialize ArcFace Head if selected
                if retrieval_loss_type == "arcface":
                    arc_config = config["train"]["loss"].get("arcface", {})
                    margin = arc_config.get("margin", 0.5)
                    scale = arc_config.get("scale", 64.0)
                    
                    num_ret_classes = kwargs.get("num_retrieval_classes", None)
                    if num_ret_classes is None:
                        num_ret_classes = kwargs.get("num_classes", 100)
                        print(f"WARNING: num_retrieval_classes not provided for ArcFace. Using fallback: {num_ret_classes}")
                    
                    proj_dim = config["models"][model_name].get("retrieval_projection", {}).get("output_dim", 64)
                    
                    self.arcface_head = ArcFace(
                        in_features=proj_dim, 
                        out_features=num_ret_classes, 
                        s=scale, 
                        m=margin
                    )
                    print(f"ArcFace head initialized: {proj_dim} -> {num_ret_classes} classes (m={margin}, s={scale})")
                    
            else:
                self.retrieval_proj = None
        else:
            self.retrieval_proj = None

        # Pre-aggregation cache (SeHGNN's key efficiency trick)
        self._cached_agg = None
        self._cached_channel_mask = None

    def precompute(self, x_dict, edge_index_dict):
        """Pre-compute neighbor aggregations once before training.

        This is the core SeHGNN optimization: aggregate raw neighbor features
        once, cache them, then only run projection + transformer during training.
        Since mean aggregation and linear projection commute, the result is
        mathematically identical to project-then-aggregate.
        """
        from torch_geometric.utils import scatter

        batch_size = x_dict['startup'].shape[0]
        device = x_dict['startup'].device
        cache = {}

        for mp in self.metapaths:
            if mp == "self":
                continue
            src, rel, dst = mp
            key = f"{src}__{rel}__{dst}"

            edge_index = edge_index_dict.get(mp)

            if edge_index is None:
                cache[key] = torch.zeros(batch_size, x_dict[src].shape[1],
                                         device=device)
            else:
                x_src = x_dict[src]
                src_idx, dst_idx = edge_index
                num_dst_nodes = x_dict[dst].shape[0]

                n_hops = self.num_hops if (src == dst == 'startup') else 1
                h = x_src
                for _hop in range(n_hops):
                    h_agg = scatter(h[src_idx], dst_idx, dim=0,
                                    dim_size=num_dst_nodes,
                                    reduce=self.aggregation_method)
                    if _hop < n_hops - 1:
                        h = h_agg

                cache[key] = h_agg[:batch_size]

        self._cached_agg = cache

        # Build channel mask: 1 where node has neighbors, 0 where all-zeros
        if self.channel_masking:
            mask = torch.ones(batch_size, len(self.metapaths), device=device)
            for ch_idx, mp in enumerate(self.metapaths):
                if mp == "self":
                    continue  # self channel always active
                key = f"{mp[0]}__{mp[1]}__{mp[2]}"
                agg = cache[key]
                # A node has no neighbors for this channel if its aggregation is all zeros
                is_empty = (agg.abs().sum(dim=-1) == 0).float()
                mask[:, ch_idx] = 1.0 - is_empty
            self._cached_channel_mask = mask
            num_masked = (mask == 0).sum().item()
            total = mask.numel()
            print(f"  SeHGNN: Channel mask built — {num_masked}/{total} entries masked ({100*num_masked/total:.1f}%)")
        else:
            self._cached_channel_mask = None

        print(f"  SeHGNN: Pre-aggregated {len(cache)} metapath channels (cached)")

    def clear_cache(self):
        """Clear pre-aggregation cache (needed for explanation/Captum)."""
        self._cached_agg = None
        self._cached_channel_mask = None

    def forward(self, x_dict, edge_index_dict, retrieval_labels=None, **kwargs):
        from torch_geometric.utils import scatter

        batch_size = x_dict['startup'].shape[0]
        device = x_dict['startup'].device
        use_cache = self._cached_agg is not None

        projected_features = []

        # A. Self Features
        h_self = x_dict['startup']
        h_self = self.input_drop(h_self)
        h_self = self.projectors["self"](h_self)  # [Batch, Hidden]
        projected_features.append(h_self)

        # B. Neighbor Features
        for mp in self.metapaths:
            if mp == "self":
                continue
            src, rel, dst = mp
            key = f"{src}__{rel}__{dst}"

            if use_cache:
                # Fast path: use pre-aggregated raw features, project now
                h_raw = self._cached_agg[key]
                h_agg = self.input_drop(h_raw)
                h_agg = self.projectors[key](h_agg)
            else:
                # Slow path: on-the-fly aggregation (for explanation / mini-batch)
                edge_index = edge_index_dict.get(mp)

                if edge_index is None:
                    h_agg = torch.zeros(batch_size, self.hidden_channels, device=device)
                else:
                    x_src = x_dict[src]
                    src_idx, dst_idx = edge_index
                    num_dst_nodes = x_dict[dst].shape[0]

                    h_src = self.input_drop(x_src)
                    h_src = self.projectors[key](h_src)

                    n_hops = self.num_hops if (src == dst == 'startup') else 1
                    for _hop in range(n_hops):
                        h_agg = scatter(h_src[src_idx], dst_idx, dim=0,
                                        dim_size=num_dst_nodes,
                                        reduce=self.aggregation_method)
                        if _hop < n_hops - 1:
                            h_src = h_agg

                    h_agg = h_agg[:batch_size]

                # Ensure x_dict[src] is in the graph for Captum autograd
                if src in x_dict:
                    dummy = (x_dict[src].sum() * 0.0)
                    h_agg = h_agg + dummy

            projected_features.append(h_agg)
            
        # Stack: [Batch, Num_Channels, Hidden]
        # Ensure all are sliced to batch_size (self features might be larger if using NeighborLoader?)
        # NeighborLoader returns x_dict containing ALL sampled nodes. 
        # The first `batch_size` nodes of `input_type` are the targets.
        
        # Slice self features too
        projected_features[0] = projected_features[0][:batch_size]

        x = torch.stack(projected_features, dim=1)

        # 2. Build channel mask
        channel_mask = None
        if self.channel_masking:
            if self._cached_channel_mask is not None:
                channel_mask = self._cached_channel_mask[:batch_size]
            else:
                # On-the-fly mask for non-cached path (explanation / mini-batch)
                channel_mask = torch.ones(batch_size, len(self.metapaths), device=device)
                for ch_idx in range(len(projected_features)):
                    is_empty = (projected_features[ch_idx].abs().sum(dim=-1) == 0).float()
                    channel_mask[:, ch_idx] = 1.0 - is_empty

        # 3. Semantic Fusion (Transformer)
        x, attn_weights = self.semantic_fusion(x, mask=channel_mask)

        # Zero out masked channels after transformer so they don't leak into fc_after_concat
        if channel_mask is not None:
            x = x * channel_mask.unsqueeze(-1)

        # 4. Flatten/Project
        x = x.reshape(batch_size, -1)
        if self.pre_fc_norm is not None:
            x = self.pre_fc_norm(x)
        x = self.fc_after_concat(x) # [Batch, Hidden]
        
        # 4. Residual
        if self.use_residual:
            # x = x + self.res_fc(features[self.tgt_type])
            # We need original self features again, sliced
            h_self_orig = x_dict['startup'][:batch_size]
            x = x + self.res_fc(h_self_orig)
        
        # 5. Task MLP
        x = self.task_mlp(x)
        
        out = self._apply_heads(x, retrieval_labels=retrieval_labels)
        
        # Add attention weights to output
        if isinstance(out, dict):
            out['attention_weights'] = attn_weights
            out['metapath_names'] = self.metapaths
            
        return out


class DegreeCentralityBaseline(BaseGNN):
    """
    Degree Centrality Baseline: Uses normalized node degree as prediction score.
    """
    def __init__(
        self,
        hidden_channels, # Ignored
        degrees, # Tensor of global degrees
        target_mode="multi_prediction",
        num_classes=2,
        **kwargs
    ):
        super().__init__(hidden_channels, target_mode, num_classes)
        # Register degrees as buffer so it moves to device automatically
        self.register_buffer("degrees", degrees.float())
        self.max_degree = self.degrees.max()
        if self.max_degree == 0:
            self.max_degree = 1.0
            
    def forward(self, x_dict, edge_index_dict, batch=None, **kwargs):
        # We need node IDs to look up global degrees
        if batch is not None and hasattr(batch['startup'], 'n_id'):
            n_ids = batch['startup'].n_id
            # n_id maps local batch index -> global index
            batch_degrees = self.degrees[n_ids]
        else:
            batch_size = x_dict['startup'].shape[0]
            if batch_size == self.degrees.shape[0]:
                 batch_degrees = self.degrees
            else:
                batch_degrees = torch.zeros(batch_size, device=x_dict['startup'].device)

        # Normalize degrees to [0, 1]
        scores = batch_degrees / self.max_degree
        
        # Convert scores to logits-like or probability-like
        epsilon = 1e-6
        scores_clipped = torch.clamp(scores, epsilon, 1.0 - epsilon)
        logits = torch.log(scores_clipped / (1.0 - scores_clipped))
        
        out = logits.view(-1, 1)
        
        if self.target_mode == "binary_prediction":
            return {
                "startup": {
                    "output": out,
                    "embedding": x_dict['startup']
                }
            }
        elif self.target_mode == "multi_prediction":
            # Construct logits such that softmax(logits)[1] = score
            # logits = [0, logit]
            zeros = torch.zeros_like(out)
            multi_logits = torch.cat([zeros, out], dim=1) # Class 0: 0, Class 1: logit -> Softmax will give probs
            
            # If more classes, pad with -inf?
            if self.num_classes > 2:
                 padding = torch.full((out.shape[0], self.num_classes - 2), -float('inf'), device=out.device)
                 multi_logits = torch.cat([multi_logits, padding], dim=1)
                 
            return {
                "startup": {
                    "output": multi_logits,
                    "embedding": x_dict['startup']
                }
            }
        elif self.target_mode == "masked_multi_task":
             return {
                 "masked_multi_task_output": {"startup": torch.cat([out, out], dim=1)},
                 "embedding": {"startup": x_dict['startup']},
                 "out_mom": out.squeeze(-1),
                 "out_liq": out.squeeze(-1)
             }
        elif self.target_mode == "multi_task":
             return {
                "binary_output": {"startup": out.squeeze(-1)},
                "multi_class_output": {"startup": torch.cat([torch.zeros_like(out), out], dim=1)}, # Simplified
                "embedding": {"startup": x_dict['startup']}
            }
        else:
             raise ValueError(f"Unsupported target_mode: {self.target_mode}")


class LLMBaseline(BaseGNN):
    """
    LLM-based baseline for startup success prediction.
    Non-trainable - uses HuggingFace Transformers for inference only.

    Supports:
    - binary_prediction: Single task (liquidity by default)
    - masked_multi_task: Both momentum and liquidity predictions
    """

    def __init__(
        self,
        hidden_channels: int,  # Unused, kept for interface
        config: dict,
        raw_features_df,  # pandas DataFrame with startup features
        target_mode: str = "masked_multi_task",
        num_classes: int = 2,
        **kwargs
    ):
        super().__init__(hidden_channels, target_mode, num_classes)

        self.config = config
        self.raw_features_df = raw_features_df
        self.llm_config = config.get("models", {}).get("LLM", {})

        # Initialize predictor lazily
        self._predictor = None

    def _get_predictor(self):
        if self._predictor is None:
            from .llm_predictor import LLMPredictor
            self._predictor = LLMPredictor(
                model_name=self.llm_config.get("model_name", "meta-llama/Meta-Llama-3-8B-Instruct"),
                cache_dir=self.llm_config.get("cache_dir", "outputs/llm_cache"),
                temperature=self.llm_config.get("temperature", 0.0),
                device=self.llm_config.get("device", "auto"),
                torch_dtype=self.llm_config.get("torch_dtype", "auto"),
                load_in_8bit=self.llm_config.get("load_in_8bit", False),
                load_in_4bit=self.llm_config.get("load_in_4bit", False),
                token=self.llm_config.get("huggingface_token"),
                use_calibration=self.llm_config.get("use_calibration", True),
                use_chain_of_thought=self.llm_config.get("use_chain_of_thought", False),
                prompt_features=self.llm_config.get("prompt_features", "full"),
            )
        return self._predictor

    def forward(self, x_dict, edge_index_dict, node_indices=None, eval_mask=None, **kwargs):
        """
        Forward pass - generate predictions via LLM.

        Args:
            x_dict: Node features (used for batch size only)
            edge_index_dict: Ignored (LLM doesn't use graph structure)
            node_indices: Optional indices into raw_features_df
            eval_mask: Optional boolean mask indicating which nodes to predict for
        """
        total_nodes = x_dict['startup'].shape[0]
        device = x_dict['startup'].device
        predictor = self._get_predictor()

        # Determine which node indices to predict for
        if eval_mask is not None:
            # Only predict for masked nodes (efficient for val/test evaluation)
            mask_indices = eval_mask.nonzero(as_tuple=True)[0].cpu().numpy()
            print(f"  LLM: Predicting for {len(mask_indices)} masked nodes (out of {total_nodes} total)")
        elif node_indices is not None:
            mask_indices = node_indices.cpu().numpy()
        else:
            # Try to extract test_mask from batch graph_data passed via kwargs
            batch = kwargs.get('batch')
            if batch is not None and hasattr(batch.get('startup', {}), 'test_mask'):
                eval_mask = batch['startup'].test_mask
                mask_indices = eval_mask.nonzero(as_tuple=True)[0].cpu().numpy()
                print(f"  LLM: eval_mask was None, recovered test_mask from batch ({len(mask_indices)} nodes out of {total_nodes} total)")
            else:
                # Final fallback: predict for all nodes
                print(f"  WARNING: LLM predicting for ALL {total_nodes} nodes (no mask available)")
                mask_indices = range(min(total_nodes, len(self.raw_features_df)))

        # Optional: limit predictions for testing (models.LLM.max_predictions)
        max_preds = self.llm_config.get("max_predictions", 0)
        if max_preds > 0 and len(mask_indices) > max_preds:
            print(f"  LLM: Limiting to {max_preds} predictions (out of {len(mask_indices)})")
            mask_indices = mask_indices[:max_preds]

        # Get feature dicts only for nodes we need to predict
        feature_dicts = [self.raw_features_df.iloc[i].to_dict() for i in mask_indices]

        # Generate predictions based on target_mode
        if self.target_mode == "masked_multi_task":
            # Predict momentum for all eval nodes
            mom_probs = predictor.predict_batch(feature_dicts, "momentum")

            # Predict liquidity only for mature nodes (mask_liq == 1)
            batch = kwargs.get('batch')
            if batch is not None and hasattr(batch['startup'], 'y') and batch['startup'].y.shape[1] >= 4:
                maturity_flags = batch['startup'].y[mask_indices, 3].cpu().numpy()
                mature_local = maturity_flags == 1
                mature_feature_dicts = [fd for fd, m in zip(feature_dicts, mature_local) if m]
                print(f"  LLM: Predicting liquidity for {len(mature_feature_dicts)} mature nodes (skipping {len(feature_dicts) - len(mature_feature_dicts)} immature)")
                mature_liq_probs = predictor.predict_batch(mature_feature_dicts, "liquidity") if mature_feature_dicts else []
                # Expand back to full eval size (non-mature get 0.0 probability)
                liq_probs = []
                j = 0
                for m in mature_local:
                    if m:
                        liq_probs.append(mature_liq_probs[j])
                        j += 1
                    else:
                        liq_probs.append(0.0)
            else:
                liq_probs = predictor.predict_batch(feature_dicts, "liquidity")

            # Create full-size tensors and fill in predictions at masked positions
            mom_logits_full = torch.zeros(total_nodes, device=device)
            liq_logits_full = torch.zeros(total_nodes, device=device)

            mom_logits = self._probs_to_logits(mom_probs, device)
            liq_logits = self._probs_to_logits(liq_probs, device)

            # Fill in predictions at the correct positions
            mask_indices_tensor = torch.tensor(mask_indices, device=device, dtype=torch.long)
            mom_logits_full[mask_indices_tensor] = mom_logits
            liq_logits_full[mask_indices_tensor] = liq_logits

            return {
                "masked_multi_task_output": {
                    "startup": torch.stack([mom_logits_full, liq_logits_full], dim=1)
                },
                "embedding": {"startup": x_dict['startup']},
                "out_mom": mom_logits_full,
                "out_liq": liq_logits_full,
            }
        else:
            # Single task (liquidity)
            liq_probs = predictor.predict_batch(feature_dicts, "liquidity")

            # Create full-size tensor and fill in predictions
            liq_logits_full = torch.zeros(total_nodes, device=device)
            liq_logits = self._probs_to_logits(liq_probs, device)

            mask_indices_tensor = torch.tensor(mask_indices, device=device, dtype=torch.long)
            liq_logits_full[mask_indices_tensor] = liq_logits

            return {
                "startup": {
                    "output": liq_logits_full.unsqueeze(1),
                    "embedding": x_dict['startup'],
                }
            }

    def _probs_to_logits(self, probs: List, device) -> torch.Tensor:
        """Convert probabilities to logits."""
        eps = 1e-6
        probs_t = torch.tensor(probs, device=device, dtype=torch.float32)
        probs_t = torch.clamp(probs_t, eps, 1.0 - eps)
        return torch.log(probs_t / (1.0 - probs_t))

import torch
import torch.nn as nn
import torch.nn.functional as F

################################
# FiGuRO-specific adaptive layer
################################

class AdaptiveRankReducedLinear(nn.Module):
    """
    Linear layer with adaptive rank reduction modified based on this paper:
    "Rank-Reduced Neural Networks for Data Compression" (https://arxiv.org/pdf/2405.13980)
    
    This layer implements a low-rank factorization of the weight matrix W = U @ V,
    where U has shape (out_features, max_rank) and V has shape (max_rank, in_features).
    The effective rank can be dynamically adjusted during training by zeroing out
    singular values while maintaining the original matrix dimensions.
    
    Args:
        in_features (int): Size of each input sample
        out_features (int): Size of each output sample
        initial_rank_ratio (float): Initial rank as a ratio of max_rank (default: 1.0)
        min_rank (int): Minimum allowed rank (default: 1)
        bias (bool): If True, adds a learnable bias to the output (default: True)
    
    Attributes:
        current_rank (int): Current effective rank of the layer
        active_dims (int): Number of active dimensions being used
        singular_values (Tensor): Buffer storing singular values for monitoring
        U (Parameter): Left factor matrix of shape (out_features, max_rank)
        V (Parameter): Right factor matrix of shape (max_rank, in_features)
    """
    def __init__(self, in_features, out_features, initial_rank_ratio=1.0, min_rank=1, bias=True):
        super(AdaptiveRankReducedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.min_rank = max(1, min_rank)

        # Calculate maximum possible rank
        self.max_rank = min(in_features, out_features)
        
        # Start with full rank or specified initial rank
        self.current_rank = max(1, int(self.max_rank * initial_rank_ratio))
        
        # Create factorized weight matrices at full dimension
        self.U = nn.Parameter(torch.Tensor(out_features, self.max_rank))
        self.V = nn.Parameter(torch.Tensor(self.max_rank, in_features))
        
        # Keep track of active dimensions
        self.register_buffer('_active_dims', torch.tensor(self.current_rank, dtype=torch.long))
        
        # Keep track of singular values for adaptive rank reduction
        self.register_buffer('singular_values', torch.ones(self.max_rank))
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
            
        self.init_parameters()

        # initialize a vector that keeps track of which dimensions are active (to not just go from right to left)
        #self.active_dims_vector = nn.Parameter(torch.ones(self.max_rank), requires_grad=False)

    @property
    def active_dims(self):
        return int(self._active_dims.item())

    @active_dims.setter
    def active_dims(self, value):
        self._active_dims.fill_(value)

    def init_parameters(self):
        """Initialize the weight matrices U and V using Xavier initialization and bias to zeros."""
        # Initialize using Xavier initialization
        nn.init.xavier_uniform_(self.U)
        nn.init.xavier_uniform_(self.V)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def reduce_rank(self, new_rank, dim=0, which_dims=None):
        """Reduce the effective rank by zeroing out smallest singular values.
        
        Performs SVD on the current weight matrix and reconstructs it with a reduced
        number of singular values, effectively lowering the rank while maintaining
        the original matrix dimensions.
        
        Args:
            new_rank (int): Target rank after reduction (will be clamped to min_rank)
            dim (int): Dimension along which to reduce (0 for output, unused currently)
            which_dims (Tensor, optional): Specific dimensions to keep active. If provided,
                these dimensions are preserved and others are zeroed out.
        
        Returns:
            bool: True if rank was reduced successfully, False if already at min_rank
        """
        #self.max_rank = min(self.max_rank, self.active_dims+1) # so we know we shouldn't go above that later on
        previous_rank = self.active_dims
        if self.active_dims == self.min_rank:
            #print(f"Warning: Attempting to reduce rank below minimum rank {self.min_rank}")
            return False
        if new_rank < self.min_rank:
            new_rank = self.min_rank
        with torch.no_grad():
            # Compute current weight matrix
            W = torch.matmul(self.U, self.V)
            
            # Perform SVD
            U, S, V = torch.svd(W)
            #print(f"Shapes: U={U.shape}, S={S.shape}, V={V.shape}")

            # Zero out smallest singular values but keep matrix dimensions
            # Set all singular values after new_rank to zero
            zeroing_mask = torch.ones_like(S)
            if which_dims is not None:
                #print(f"Reducing rank from {self.active_dims} to {new_rank} (dims {which_dims.tolist()})")
                dims_to_remove = list(set(range(len(S))) - set(which_dims.tolist()))
                if len(which_dims) != new_rank:
                    print(f"Warning: which_dims length {len(which_dims)} does not match new_rank {new_rank}")
                if len(dims_to_remove) != (len(S) - new_rank):
                    print(f"Warning: dims_to_remove length {len(dims_to_remove)} does not match expected {len(S) - new_rank}")
                #print(f"Removing dims {dims_to_remove}")
                #zeroing_mask[which_dims] = 0
                # reorder S, U, and V so that the to-be-kicked-out dims are last
                U_new = torch.zeros_like(U)
                V_new = torch.zeros_like(V)
                S_new = torch.zeros_like(S)
                U_new[:new_rank,:] = U[which_dims,:]
                U_new[new_rank:,:] = U[dims_to_remove,:]
                U = U_new
                S_new[:new_rank] = S[which_dims]
                S_new[new_rank:] = S[dims_to_remove]
                S = S_new
                V_new[:,:new_rank] = V[:,which_dims]
                V_new[:,new_rank:] = V[:,dims_to_remove]
                V = V_new
            #if dim == 0:
            zeroing_mask[new_rank:] = 0
            #elif dim == 1:
            #    zeroing_mask[:,new_rank:] = 0
            #else:
            #    raise ValueError("Invalid dimension for rank reduction")
            S_reduced = S * zeroing_mask
            
            # Store singular values for monitoring
            self.singular_values = S.detach().clone()
            
            # Reconstruct U and V with reduced effective rank
            # U_reduced will have zeros in columns beyond the new rank
            # V_reduced will have zeros in rows beyond the new rank
            sqrt_S = torch.sqrt(S_reduced)
            
            # Prepare scaled U and V matrices
            U_scaled = U * sqrt_S.unsqueeze(0)
            V_scaled = torch.matmul(torch.diag(sqrt_S), V.t())
            
            # Update parameters while maintaining original dimensions
            self.U.data.copy_(U_scaled)
            self.V.data.copy_(V_scaled)
            
            # Update current rank (for tracking)
            self.active_dims = new_rank
            self.max_rank = previous_rank
            
        return True
    
    def increase_rank(self, increment=None, increase_ratio=1.1, dim=0, mode='unimodal'):
        """Increase the effective rank by activating more singular values.
        
        Used when the rank reduction went too far and model performance degraded.
        Recomputes SVD and activates additional singular values to increase capacity.
        
        Args:
            increment (int, optional): Number of dimensions to add. If None, computed from increase_ratio
            increase_ratio (float): Ratio to increase rank by (default: 1.1 = 10% increase)
            dim (int): Dimension along which to increase (0 for output, 1 for input)
            mode (str): Mode for rank increase (currently unused)
        
        Returns:
            bool: True if rank was increased successfully, False if already at max_rank
        """
        #print(f"Increasing rank from {self.active_dims}. Max rank is {self.max_rank}")
        if increment is None:
            increment = max(1, int(self.active_dims * (increase_ratio - 1)))
        if self.active_dims >= self.max_rank:
            #print(f"Warning: Attempting to increase rank beyond max rank {self.max_rank}")
            return False
        #if mode == 'unimodal':
        #    self.min_rank = min(self.active_dims, self.max_rank) # setting the minimum rank to current active dimensions + 1 to prevent going too low again
        with torch.no_grad():
            # Calculate the new rank (ensuring we don't exceed max_rank)
            new_rank = min(self.active_dims + increment, self.max_rank)
            
            # If we're already at max rank, no change needed
            if new_rank <= self.active_dims:
                #print(f"Rank is already at maximum or cannot be increased further.")
                return False
                
            # Compute current weight matrix
            W = torch.matmul(self.U, self.V)
            
            # Perform SVD
            U, S, V = torch.svd(W)
            
            # Create mask for active singular values (including newly activated ones)
            zeroing_mask = torch.ones_like(S)
            if dim == 0:
                # Increase along output dimension
                zeroing_mask[new_rank:] = 0
            elif dim == 1:
                # Increase along input dimension  
                zeroing_mask[:,new_rank:] = 0
            else:
                raise ValueError("Invalid dimension for rank increase. Use 0 for output dim, 1 for input dim")
                
            S_increased = S * zeroing_mask
            
            # Store singular values for monitoring
            self.singular_values = S.detach().clone()
            
            # Reconstruct U and V with increased effective rank
            sqrt_S = torch.sqrt(S_increased)
            
            # Prepare scaled U and V matrices
            U_scaled = U * sqrt_S.unsqueeze(0)
            V_scaled = torch.matmul(torch.diag(sqrt_S), V.t())
            
            # Update parameters while maintaining original dimensions
            self.U.data.copy_(U_scaled)
            self.V.data.copy_(V_scaled)
            
            # Update current rank (for tracking)
            self.active_dims = min(new_rank, self.max_rank)
            self.min_rank = min(self.active_dims, self.max_rank)
            
            #print(f"Increased rank to {self.active_dims}")
            
        return True
    
    def get_rank_reduction_info(self):
        """Return information about singular values for making rank reduction decisions.
        
        Computes SVD of the current weight matrix and returns the singular values,
        which can be used to determine optimal rank reduction strategies.
        
        Returns:
            Tensor: Singular values of the weight matrix W = U @ V
        """
        # Calculate full SVD if needed
        with torch.no_grad():
            W = torch.matmul(self.U, self.V)
            _, S, _ = torch.svd(W)
            return S
    
    def forward(self, x):
        """Forward pass through the adaptive rank-reduced linear layer.
        
        Computes output using only the active dimensions of the factorized weight matrix.
        Dimensions beyond active_dims are zeroed out to enforce the reduced rank.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, in_features)
        
        Returns:
            Tensor: Output tensor of shape (batch_size, out_features)
        """
        # Compute W = U * V on the fly
        # Use matmul for better efficiency with low-rank matrices
        # For effective rank reduction, we only use the active dimensions
        U_active = self.U[:, :self.active_dims]
        V_active = self.V[:self.active_dims, :]

        # try forcing the weight matrix to be of this rank only
        W = torch.matmul(U_active, V_active)
        W[self.active_dims:] = 0  # Remove out dimensions beyond active rank
        #W[:,self.active_dims:] = 0  # Zero out dimensions beyond active rank
        with torch.no_grad():
            self.bias[self.active_dims:] = 0  # Zero out bias beyond active rank
        
        return F.linear(x, W, self.bias)

        # testing without a bias
        #return F.linear(x, W)
    
        #return F.linear(x, torch.matmul(U_active, V_active), self.bias)
    
    def extra_repr(self):
        """Extra representation string for the layer showing key parameters."""
        return f'in_features={self.in_features}, out_features={self.out_features}, current_rank={self.active_dims}'
    
    def get_weights(self):
        """Reconstruct and return the effective weight matrix using only active dimensions.
        
        Returns:
            Tensor: Weight matrix of shape (out_features, in_features) computed as U_active @ V_active
        """
        U_active = self.U[:, :self.active_dims]
        V_active = self.V[:self.active_dims, :]
        W = torch.matmul(U_active, V_active)
        W[self.active_dims:] = 0
        
        return W

##############################
# FiGuRO class
##############################

class FiGuRO(nn.Module):
    """FiGuRO: Flexible Intrinsic Guided Rank Optimization.
    
    A neural network module that adaptively compresses multimodal latent representations
    through rank-reduced linear transformations. FiGuRO learns both shared and modality-specific
    compressed representations while dynamically adjusting the compression level based on
    reconstruction quality metrics.
    
    The architecture consists of:
    - One shared adaptive layer that processes concatenated multimodal inputs
    - Multiple modality-specific adaptive layers (one per modality)
    - Optional transformation layers before and after compression
    
    During training, FiGuRO monitors reconstruction quality and adaptively reduces or increases
    the rank of the compression layers to find the optimal intrinsic dimensionality.
    
    Args:
        n_modalities (int): Number of input modalities
        latent_dims (list): List of latent dimensions for each modality
        decomp_dims (list): List of compressed dimensions [shared_dim, mod1_dim, mod2_dim, ...]
            Length should be n_modalities + 1
        rank_reduction_frequency (int): Number of epochs between rank adaptation steps (default: 10)
        rank_reduction_threshold (float): Threshold for singular value-based rank reduction (default: 0.01)
        distortion_metric (str): Metric to measure reconstruction quality, 'R2' or 'Var' (default: 'R2')
        distortion_threshold (float): Acceptable drop in distortion metric (default: 0.05)
        patience (int): Number of epochs to wait before making rank adjustments (default: 10)
        initial_rank_ratio (float): Initial rank as ratio of max rank (default: 1.0)
        min_rank (int): Minimum allowed rank for any layer (default: 1)
        use_bias (bool): Whether to use bias in adaptive layers (default: True)
        transformation_layers (bool): Whether to add transformation layers (default: False)
        transformation_activation (nn.Module): Activation function for transformation layers (default: ReLU())
    
    Example:
        >>> figuro = FiGuRO(n_modalities=2, latent_dims=[128, 64], decomp_dims=[32, 16, 16])
        >>> figuro.initialize_tracking(epochs=100, warmup=10)
        >>> # In training loop:
        >>> reconstructed = figuro([latent1, latent2])
        >>> # At end of epoch:
        >>> figuro.step(epoch, reconstructed, [latent1, latent2])
    """
    def __init__(
            self, 
            n_modalities: int,
            latent_dims: list,
            decomp_dims: list,
            rank_reduction_frequency: int = 10,
            rank_reduction_threshold: float = 0.01,
            distortion_metric: str = 'R2', # 'R2' or 'Var'
            distortion_threshold: float = 0.05,
            patience: int = 10,
            initial_rank_ratio: float = 1.0, 
            min_rank: int = 1, 
            use_bias: bool = True,
            reduction_ratio: float = 0.9,
            increase_ratio: float = 1.1,
            transformation_layers: bool = False,
            transformation_activation: nn.Module = nn.ReLU()
        ):
        super(FiGuRO, self).__init__()

        ###
        # Parameterized components: Adaptive bottlenecks and optional transformation layers
        ###
        
        # create the adaptive layers that will be trained
        # it consists of one shared layer and n_modalities specific layers (in this order)
        self.adaptive_layers = self._create_adaptive_layers(
            n_modalities=n_modalities,
            latent_dims=latent_dims,
            decomp_dims=decomp_dims,
            initial_rank_ratio=initial_rank_ratio,
            min_rank=min_rank,
            bias=use_bias
        )

        # if desired, transform the latent representations before feeding them to the adaptive layers (to be in better matched spaces)
        if transformation_layers:
            self.transformation_layers = self._create_transformation_layers(
                latent_dims=latent_dims,
                activation_fn=transformation_activation
            )
            self.transformation_layers_r = self._create_reverse_transformation_layers(
                latent_dims=latent_dims,
                decomp_dims=decomp_dims,
                activation_fn=transformation_activation
            )
        else:
            self.transformation_layers = None
            self.transformation_layers_r = None
        
        ###
        # Hyperparameters for rank adaptation
        ###
        self.n_modalities = n_modalities
        self.rank_reduction_frequency = rank_reduction_frequency
        self.rank_reduction_threshold = rank_reduction_threshold
        self.distortion_metric = distortion_metric
        self.distortion_threshold = distortion_threshold
        self.patience = patience
        self.reduction_ratio = reduction_ratio
        self.increase_ratio = increase_ratio

    @property
    def active_dims(self):
        """Return list of active dimensions for all adaptive layers."""
        return [layer.active_dims for layer in self.adaptive_layers]

    def initialize_tracking(self, epochs, warmup=0):
        """Initialize tracking variables for rank adaptation during training.
        
        Sets up the schedule for rank reduction checks and initializes counters
        for monitoring distortion metrics and making adaptation decisions.
        
        Args:
            epochs (int): Total number of training epochs
            warmup (int): Number of warmup epochs before starting rank reduction (default: 0)
        """
        self.rank_schedule = list(range(warmup + self.rank_reduction_frequency, 
                                 epochs, 
                                 self.rank_reduction_frequency))
        #self.initial_squares = [None] * self.n_modalities # per modality
        #self.initial_losses = [None] * self.n_modalities # per modality (for loss-based criteria)
        self.start_reduction = False
        #self.current_distortion_per_mod = [None] * self.n_modalities
        #self.current_loss_per_mod = [None] * self.n_modalities  # for loss-based criteria
        self.patience_counter = 0
        self.break_counter = 0
        self.current_distortion_per_mod = []
        self.distortion_metric_values = []
        self.ranks = []
    
    def step(self, epoch, reconstructed_list, targets_list):
        """Perform rank adaptation step at the end of an epoch.
        
        Monitors reconstruction quality and decides whether to reduce or increase
        the rank of adaptive layers based on distortion metrics. This is the core
        logic for adaptive rank optimization.
        
        Args:
            epoch (int): Current epoch number
            reconstructed_list (list): List of reconstructed tensors for each modality
            targets_list (list): List of original target tensors for each modality
        
        Notes:
            - Should be called at the end of each training epoch
            - Automatically tracks distortion metrics and rank history
            - Makes decisions based on patience counter and distortion thresholds
        """
        if self.start_reduction is False and epoch >= self.rank_schedule[0]:
            self.start_reduction = True
            print(f"Starting rank reduction at epoch {epoch}")
            # compute the initial distortion metrics
            self.current_distortion_per_mod = self._compute_distortion_metrics(
                reconstructed_list, targets_list
            )
            self.distortion_metric_values.append(self.current_distortion_per_mod)
            self.ranks.append([layer.active_dims for layer in self.adaptive_layers])
            # set the minima for the distortion metrics
            self.max_distortion_values = [round(d, 4) for d in self.current_distortion_per_mod]
            self.distortion_minima = [d - self.distortion_threshold for d in self.max_distortion_values]
            print(f"Initial distortion metrics: {self.current_distortion_per_mod}, thresholds set to: {self.distortion_minima}")
        elif (self.start_reduction) and (epoch in self.rank_schedule) and (self.break_counter == 0):
            # compute the current distortion metrics
            self.current_distortion_per_mod = self._compute_distortion_metrics(
                reconstructed_list, targets_list
            )
            self.distortion_metric_values.append(self.current_distortion_per_mod)
            # check if the minima/thresholds need to be updated
            update_max = False
            for i, r in enumerate(self.current_distortion_per_mod):
                # round to 4 decimals to avoid small fluctuations causing updates
                if r > self.max_distortion_values[i]:
                    self.max_distortion_values[i] = round(r, 4)
                    update_max = True
            if update_max:
                self.distortion_minima = [d - self.distortion_threshold for d in self.max_distortion_values]
                print(f"Updated distortion metric maxima to: {self.max_distortion_values}, thresholds to: {self.distortion_minima}")
            
            # now see which modalities and layers can be reduced
            # this is one of the most important parts of the method: the rank optimization logic
            modalities_to_reduce = []
            modalities_to_increase = []
            patience_fraction = 2/3
            if (len(self.distortion_metric_values) >= min(10, int(patience_fraction * self.patience))) and self.patience_counter >= min(10, int(patience_fraction * self.patience)):
                print("Increasing ranks based on distortion metrics")
                for i in range(len(self.current_distortion_per_mod)):
                    i_rsquares = [r[i] for r in self.distortion_metric_values[-min(10, int(patience_fraction * self.patience)):]]
                    if all(r < self.distortion_minima[i] for r in i_rsquares):
                        modalities_to_increase.append(i)
                    elif self.current_distortion_per_mod[i] > self.distortion_minima[i]:
                        modalities_to_reduce.append(i)
            elif (len(self.distortion_metric_values) >= 1):
                for i in range(len(self.current_distortion_per_mod)):
                    if self.current_distortion_per_mod[i] > self.distortion_minima[i]:
                        modalities_to_reduce.append(i)
                    #elif self.current_distortion_per_mod[i] < self.distortion_minima[i]:
                    #    modalities_to_increase.append(i)
            #if len(modalities_to_increase) == len(self.current_distortion_per_mod):
            #    # set minima
            #    for i in range(len(self.adaptive_layers)):
            #        self.adaptive_layers[i].min_rank = self.adaptive_layers[i].active_dims + 1
            #    print(f"Adjusting minimum ranks to {[layer.min_rank for layer in self.adaptive_layers]}")
            layers_to_reduce = []
            layers_to_increase = []
            if (len(modalities_to_reduce) == 0) and (len(modalities_to_increase) == 0):
                pass
            elif (len(modalities_to_reduce) > 0) and (len(modalities_to_increase) > 0):
                # no increasing yet, but no decreasing the shared either
                layers_to_reduce = [i + 1 for i in modalities_to_reduce]
                layers_to_increase = [0] + [i + 1 for i in modalities_to_increase]
                # set the min for the modality to be increased to current rank
                self.adaptive_layers[0].min_rank = self.adaptive_layers[0].active_dims
                for i in modalities_to_increase:
                    self.adaptive_layers[i + 1].min_rank = self.adaptive_layers[i + 1].active_dims + 1
                print(f"Adjusting minimum ranks to {[layer.min_rank for layer in self.adaptive_layers]}")
            else:
                if len(modalities_to_increase) > 0:
                    if len(modalities_to_increase) == len(self.current_distortion_per_mod):
                        # if all modalities are below the threshold, increase ranks of all layers
                        layers_to_increase = [i for i in range(len(self.adaptive_layers))]
                        self.adaptive_layers[0].min_rank = self.adaptive_layers[0].active_dims + 1
                        print(f"Adjusting minimum ranks to {[layer.min_rank for layer in self.adaptive_layers]}")
                    else:
                        layers_to_increase = [0] + [i + 1 for i in modalities_to_increase]
                        for i in modalities_to_increase:
                            self.adaptive_layers[i + 1].min_rank = self.adaptive_layers[i + 1].active_dims + 1
                        #self.adaptive_layers[0].min_rank = self.adaptive_layers[0].active_dims
                        print(f"Adjusting minimum ranks to {[layer.min_rank for layer in self.adaptive_layers]}")
                if len(modalities_to_reduce) > 0:
                    # if all modalities are below the threshold, reduce ranks of all layers
                    if len(modalities_to_reduce) == len(self.current_distortion_per_mod):
                        layers_to_reduce = [i for i in range(len(self.adaptive_layers))]
                    else:
                        layers_to_reduce = [i + 1 for i in modalities_to_reduce]
        
            any_changes_made = False
            if len(layers_to_reduce) > 0:
                # Apply rank reduction
                changes_made = self.reduce_rank(layer_ids=layers_to_reduce)
                if changes_made:
                    any_changes_made = True
            if len(layers_to_increase) > 0:
                # Apply rank increase
                changes_made = self.increase_rank(layer_ids=layers_to_increase)
                if changes_made:
                    any_changes_made = True
                    self.break_counter = self.patience # give model more time to re-learn the added dimensions
            if any_changes_made:
                self.patience_counter = 0  # Reset patience counter if rank was changed
            else:
                self.patience_counter += 1
            
            self.ranks.append([layer.active_dims for layer in self.adaptive_layers])
        else:
            if (self.start_reduction) and (epoch in self.rank_schedule) and (self.break_counter > 0):
                self.break_counter -= 1
                self.current_distortion_per_mod = self._compute_distortion_metrics(
                    reconstructed_list, targets_list
                )
                self.distortion_metric_values.append(self.current_distortion_per_mod)

    def encode(self, embedding_list):
        """Encode input embeddings through adaptive compression layers.
        
        Applies optional transformation, then compresses through shared and
        modality-specific adaptive layers.
        
        Args:
            embedding_list (list): List of embedding tensors, one per modality
        
        Returns:
            tuple or Tensor: For multimodal (n>1), returns (h_shared, h_specific)
                where h_shared is the shared compressed representation and h_specific
                is a list of modality-specific compressed representations.
                For unimodal, returns single compressed tensor.
        """
        assert type(embedding_list) == list, "Input must be a list of embeddings for each modality"
        # transform embeddings if needed
        if self.transformation_layers is not None:
            h = []
            for i, layer in enumerate(self.transformation_layers):
                h.append(layer(embedding_list[i]))
        else:
            h = embedding_list
        
        if len(h) > 1:
            h_shared = self.adaptive_layers[0](torch.cat(h, dim=1))
            h_specific = []
            for i in range(len(h)):
                h_specific.append(self.adaptive_layers[i+1](h[i]))
            h_out = (h_shared, h_specific)
        else:
            h_out = [self.adaptive_layers[0](h[0])]
        
        return h_out
    
    def decode(self, h):
        """Decode compressed representations back to original latent space.
        
        Concatenates shared and specific representations, then applies optional
        reverse transformation layers.
        
        Args:
            h (tuple or Tensor): Encoded representations from encode(). For multimodal,
                expects (h_shared, h_specific). For unimodal, expects single tensor.
        
        Returns:
            list: List of reconstructed embedding tensors, one per modality
        """
        if len(h) > 1:
            h_shared, h_specific = h
            h_intermediates = []
            for i in range(len(h_specific)):
                h_intermediates.append(torch.cat([h_shared, h_specific[i]], dim=1))
        else:
            h_intermediates = [h]
        
        if self.transformation_layers_r is not None:
            h_reconstructed = []
            for i, layer in enumerate(self.transformation_layers_r):
                # use the same layer structure but in reverse (decoding)
                h_reconstructed.append(layer(h_intermediates[i]))
        else:
            h_reconstructed = h_intermediates
        return h_reconstructed
    
    def forward(self, embedding_list):
        """Forward pass through FiGuRO: encode then decode.
        
        Args:
            embedding_list (list): List of embedding tensors, one per modality
        
        Returns:
            list: List of reconstructed embedding tensors, one per modality
        """
        h = self.encode(embedding_list)
        embeddings_reconstructed = self.decode(h)
        return embeddings_reconstructed
    
    def reduce_rank(self, layer_ids=[]):
        """Reduce rank of specified adaptive layers.
        
        Uses singular value decomposition to determine optimal rank based on energy
        distribution. New rank is determined by both energy threshold and reduction ratio.
        
        Args:
            reduction_ratio (float): Ratio to reduce current rank by (default: 0.9)
            threshold (float): Energy threshold for rank determination (default: 0.01)
            layer_ids (list): Indices of layers to reduce rank for
            dim (int): Dimension along which to reduce (unused currently)
        
        Returns:
            bool: True if any layer's rank was reduced, False otherwise
        """
        changes_made = False
        for i, layer in enumerate(self.adaptive_layers):
            if i not in layer_ids: continue
            S = layer.get_rank_reduction_info()
            if len(S) <= layer.min_rank: continue
            
            energy = S**2
            cumulative_energy = torch.cumsum(energy / energy.sum(), dim=0)
            target_rank = max(layer.min_rank, torch.sum(cumulative_energy < (1.0 - self.rank_reduction_threshold)).item())
            
            current_rank = layer.active_dims
            ratio_rank = max(layer.min_rank, int(current_rank * self.reduction_ratio))
            new_rank = max(target_rank, ratio_rank)
            
            if new_rank < current_rank:
                layer.reduce_rank(new_rank)
                changes_made = True
        return changes_made

    def increase_rank(self, layer_ids=[]):
        """Increase rank of specified adaptive layers.
        
        Used when reconstruction quality falls below acceptable thresholds.
        
        Args:
            increment (int, optional): Specific number of dimensions to add
            increase_ratio (float): Ratio to increase rank by (default: 1.1)
            layer_ids (list): Indices of layers to increase rank for
            dim (int): Dimension along which to increase
        
        Returns:
            bool: True if any layer's rank was increased, False otherwise
        """
        changes_made = False
        for i, layer in enumerate(self.adaptive_layers):
            if i not in layer_ids: continue
            if layer.increase_rank(increment=None, increase_ratio=self.increase_ratio):
                changes_made = True
        return changes_made
    
    def get_total_rank(self):
        """Return total rank across all adaptive layers.
        
        Returns:
            int: Sum of active dimensions across all adaptive layers
        """
        return sum(layer.active_dims for layer in self.adaptive_layers)
        
    def _create_adaptive_layers(self, n_modalities, latent_dims, decomp_dims, initial_rank_ratio, min_rank, bias):
        """Create adaptive rank-reduced layers for compression.
        
        Creates one shared layer and multiple modality-specific layers based on
        the provided dimensions.
        
        Args:
            n_modalities (int): Number of modalities
            latent_dims (list): Dimensions of input latents for each modality
            decomp_dims (list): Dimensions of compressed representations
            initial_rank_ratio (float): Initial rank ratio for each layer
            min_rank (int): Minimum rank for each layer
            bias (bool): Whether to use bias in layers
        
        Returns:
            nn.ModuleList: List of adaptive layers [shared, mod1, mod2, ...]
        """
        # check inputs
        assert n_modalities == len(latent_dims), "Length of latent_dims must match n_modalities"
        if n_modalities > 1:
            assert n_modalities == len(decomp_dims) - 1, "Length of decomp_dims must be n_modalities + 1"
        else:
            assert len(decomp_dims) == 1, "For unimodal, decomp_dims must have length 1"
        
        # --- Adaptive Bottlenecks ---
        # 1. Shared Layer (Takes concatenated latents -> compresses to shared)
        shared_layer = AdaptiveRankReducedLinear(
            in_features=sum(latent_dims),
            out_features=decomp_dims[0],
            initial_rank_ratio=initial_rank_ratio,
            min_rank=min_rank,
            bias=bias
        )
        # Store for easy access in training loop
        adaptive_layers = nn.ModuleList([
            shared_layer
        ])

        if n_modalities > 1:
            # 2. Specific Layers (Take unimodal latent -> compresses to specific)
            for i in range(n_modalities):
                latent_dim = latent_dims[i]
                decomp_dim = decomp_dims[i+1]
                layer = AdaptiveRankReducedLinear(
                    in_features=latent_dim,
                    out_features=decomp_dim,
                    initial_rank_ratio=initial_rank_ratio,
                    min_rank=min_rank,
                    bias=bias
                )
                adaptive_layers.append(layer)

        return adaptive_layers
    
    def _create_transformation_layers(self, latent_dims, activation_fn=nn.ReLU()):
        """Create transformation layers for each modality.
        
        Each transformation layer is a simple MLP that transforms embeddings
        into a better-matched space before compression.
        
        Args:
            latent_dims (list): Dimensions for each modality
            activation_fn (nn.Module): Activation function to use (default: ReLU())
        
        Returns:
            nn.ModuleList: List of transformation layers, one per modality
        """
        # create transformation layers for each modality
        transformation_layers = nn.ModuleList()
        for dim in latent_dims:
            layer = nn.Sequential(
                nn.Linear(dim, dim),
                activation_fn,
                nn.Linear(dim, dim)
            )
            transformation_layers.append(layer)
        return transformation_layers

    def _create_reverse_transformation_layers(self, latent_dims, decomp_dims, activation_fn=nn.ReLU()):
        """Create reverse transformation layers for decoding.

        For multimodal FiGuRO, each decoded intermediate is a concatenation of
        shared and modality-specific components, so input dim is
        `decomp_dims[0] + decomp_dims[i+1]`, and output dim should match the
        original modality latent dim (`latent_dims[i]`).
        """
        transformation_layers = nn.ModuleList()

        if len(latent_dims) > 1:
            for i, out_dim in enumerate(latent_dims):
                in_dim = decomp_dims[0] + decomp_dims[i + 1]
                layer = nn.Sequential(
                    nn.Linear(in_dim, out_dim),
                    activation_fn,
                    nn.Linear(out_dim, out_dim)
                )
                transformation_layers.append(layer)
        else:
            in_dim = decomp_dims[0]
            out_dim = latent_dims[0]
            layer = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                activation_fn,
                nn.Linear(out_dim, out_dim)
            )
            transformation_layers.append(layer)

        return transformation_layers
    
    def _compute_distortion_metrics(self, reconstructed_list, targets_list):
        """Compute distortion metrics between reconstructed and target embeddings.
        
        Calculates either R² (coefficient of determination) or explained variance
        for each modality to measure reconstruction quality.
        
        Args:
            reconstructed_list (list): List of reconstructed embedding tensors
            targets_list (list): List of original target embedding tensors
        
        Returns:
            list: Distortion metric values, one per modality. Higher is better.
        
        Notes:
            - Handles NaN and Inf values gracefully
            - Computes metrics on flattened tensors
            - Uses the metric specified in self.distortion_metric ('R2' or 'Var')
        """

        # make sure the lists contain tensors
        assert type(reconstructed_list) == list, "reconstructed_list must be a list of tensors"
        assert type(targets_list) == list, "targets_list must be a list of tensors"
        assert len(reconstructed_list) == len(targets_list), "reconstructed_list and targets_list must have the same length"
        assert all(type(tensor) == torch.Tensor for tensor in reconstructed_list), "All elements in reconstructed_list must be tensors"
        assert all(type(tensor) == torch.Tensor for tensor in targets_list), "All elements in targets_list must be tensors"

        out_metrics = []

        for i, (original, reconstruction) in enumerate(zip(targets_list, reconstructed_list)):
            # Move to CPU and flatten to (N, D)
            original_cpu = original.cpu()
            reconstruction_cpu = reconstruction.cpu()
            try:
                orig_flat = original_cpu.view(original_cpu.shape[0], -1)
            except Exception:
                orig_flat = original_cpu.reshape(original_cpu.size(0), -1)
            try:
                recon_flat = reconstruction_cpu.view(reconstruction_cpu.shape[0], -1)
            except Exception:
                recon_flat = reconstruction_cpu.reshape(reconstruction_cpu.size(0), -1)
            
            # Calculate mean of original flattened data
            original_mean = orig_flat.mean(dim=0).cpu()
            original_cpu = orig_flat
            reconstruction_cpu = recon_flat

            if self.distortion_metric == 'R2':
                # Handle zeros in mean values
                if torch.any(original_mean == 0):
                    non_zero_mean = original_mean != 0
                    if non_zero_mean.sum() == 0:
                        # If all means are zero, use correlation as fallback
                        r_squared = torch.corrcoef(torch.stack((original_cpu.flatten(), reconstruction_cpu.flatten())))[0, 1]
                        if torch.isnan(r_squared):
                            r_squared = torch.tensor(0.0)
                    else:
                        # Calculate R² only for non-zero mean dimensions
                        ssr = ((original_cpu - reconstruction_cpu)**2).sum(0)[non_zero_mean]
                        ss_tot = ((original_cpu - original_mean)**2).sum(0)[non_zero_mean]
                        r_squared = 1 - ((ssr + 1e-9) / (ss_tot + 1e-9))
                        r_squared = r_squared.mean()  # Average across dimensions
                elif torch.any(torch.isnan(original_cpu)) or torch.any(torch.isinf(original_cpu)):
                    # Handle NaN or Inf values
                    valid_mask = ~torch.isnan(original_mean) & ~torch.isinf(original_mean)
                    if valid_mask.sum() == 0:
                        # If no valid values, set R² to 0
                        r_squared = torch.tensor(0.0)
                    else:
                        valid_indices = valid_mask
                        ssr = ((original_cpu - reconstruction_cpu)**2).sum(0)[valid_indices]
                        ss_tot = ((original_cpu - original_mean)**2).sum(0)[valid_indices]
                        r_squared = 1 - ((ssr + 1e-9) / (ss_tot + 1e-9))
                        r_squared = r_squared.mean()  # Average across dimensions
                else:
                    # check if there are any NaNs or Infs in original or reconstruction
                    if torch.any(torch.isnan(original_cpu)) or torch.any(torch.isinf(original_cpu)) or \
                    torch.any(torch.isnan(reconstruction_cpu)) or torch.any(torch.isinf(reconstruction_cpu)):
                        # Handle NaN or Inf values
                        valid_mask = ~torch.isnan(original_cpu) & ~torch.isinf(original_cpu) & \
                                    ~torch.isnan(reconstruction_cpu) & ~torch.isinf(reconstruction_cpu)
                        if valid_mask.sum() == 0:
                            # If no valid values, set R² to 0
                            r_squared = torch.tensor(0.0)
                        else:
                            valid_indices = valid_mask
                            ssr = ((original_cpu - reconstruction_cpu)**2).sum(0)[valid_indices]
                            ss_tot = ((original_cpu - original_mean)**2).sum(0)[valid_indices]
                    else:
                        # Normal case - calculate standard R²
                        # print mean original and mean reconstruction
                        #print(f"   Mean original (modality {i}): {original_cpu.mean().item()}, Mean reconstruction: {reconstruction_cpu.mean().item()}")
                        ssr = ((original_cpu - reconstruction_cpu)**2).sum(0)
                        ss_tot = ((original_cpu - original_mean)**2).sum(0)
                        # if there are any very small ss_tot values, print a warning
                        if torch.any(ss_tot < 1e-3):
                            valid_mask = ss_tot >= 1e-3
                            if valid_mask.sum() == 0:
                                r_squared = torch.tensor(0.0)
                            else:
                                ssr = ssr[valid_mask]
                                ss_tot = ss_tot[valid_mask]
                    #print(f"   SSR sum: {ssr.sum().item()}, SSTot sum: {ss_tot.sum().item()}")
                    r_squared = 1 - ((ssr + 1e-9) / (ss_tot + 1e-9))
                    # mask out the ones that are negative
                    # if there are any negative r_squared values, print a warning
                    if torch.any(r_squared < 0):
                        r_squared = torch.clamp(r_squared, min=0.0)
                    r_squared = r_squared.mean()  # Average across dimensions
                
                # Ensure r_squared is a scalar tensor
                if not isinstance(r_squared, torch.Tensor):
                    r_squared = torch.tensor(r_squared)
                
                out_metrics.append(r_squared.item())
            elif self.distortion_metric == 'Var':
                # Handle NaN or Inf values
                if torch.any(torch.isnan(orig_flat)) or torch.any(torch.isinf(orig_flat)) or \
                torch.any(torch.isnan(recon_flat)) or torch.any(torch.isinf(recon_flat)):
                    valid_mask = ~torch.isnan(orig_flat) & ~torch.isinf(orig_flat) & \
                                ~torch.isnan(recon_flat) & ~torch.isinf(recon_flat)
                    if valid_mask.sum() == 0:
                        explained_variance = torch.tensor(0.0)
                    else:
                        original_valid = orig_flat[valid_mask]
                        reconstruction_valid = recon_flat[valid_mask]
                        explained_variance = 1 - (torch.var(reconstruction_valid - original_valid) / (torch.var(original_valid) + 1e-9))
                else:
                    # Normal case - calculate standard explained variance on flattened data
                    explained_variance = 1 - (torch.var(recon_flat - orig_flat) / (torch.var(orig_flat) + 1e-9))

                # Handle negative explained variance values (poor fit)
                if isinstance(explained_variance, torch.Tensor):
                    if explained_variance < 0:
                        explained_variance = torch.clamp(explained_variance, min=0.0)
                else:
                    if explained_variance < 0:
                        explained_variance = 0.0

                # Ensure explained_variance is a scalar tensor
                if not isinstance(explained_variance, torch.Tensor):
                    explained_variance = torch.tensor(explained_variance)

                out_metrics.append(explained_variance.item())
        
        return out_metrics
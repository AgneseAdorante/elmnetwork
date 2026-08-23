import jax.numpy as jnp
import jax.random as jrandom
from jax import vmap
from jaxtyping import Array, Integer, PRNGKeyArray
from typing import Optional

from .elm_modeling_utils import random_connect

# every neuron connects to every source exactly once
def fully_connected(
    key: PRNGKeyArray,
    num_input: int,
    num_neuron: int,
    num_neuron_inputs: int,
    include_feedforward: bool = True,
    include_recurrent: bool = False,
    **kwargs,
) -> Integer[Array, "neuron synapse"]:
    total_sources = (num_input if include_feedforward else 0) + (
        num_neuron if include_recurrent else 0
    )
    assert num_neuron_inputs == total_sources, (
        f"num_neuron_inputs ({num_neuron_inputs}) must equal total_sources "
        f"({total_sources}). "
        f"total_sources = num_input ({num_input if include_feedforward else 0}) + "
        f"num_neuron ({num_neuron if include_recurrent else 0}). Set "
        f"num_branch * num_synapse_per_branch = {total_sources}."
    )

    connectivity = jnp.full(
        (num_neuron, num_neuron_inputs), -1, dtype=jnp.int32
    )

    base_connections = jnp.arange(total_sources, dtype=jnp.int32)
    keys = jrandom.split(key, num_neuron)
    shuffled = vmap(jrandom.permutation)(
        keys, jnp.tile(base_connections, (num_neuron, 1))
    )
    connectivity = connectivity.at[:, :total_sources].set(shuffled)

    return connectivity


# every neuron connects to every input and samples recurrent connections randomly
def fully_connected_ff_random_rec(
    key: PRNGKeyArray,
    num_input: int,
    num_neuron: int,
    num_neuron_inputs: int,
    **kwargs,
) -> Integer[Array, "neuron synapse"]:


    assert num_neuron_inputs > num_input, (
        f"num_neuron_inputs ({num_neuron_inputs}) must exceed num_input "
        f"({num_input}) so the all-to-all feed-forward block ({num_input} "
        f"synapses) leaves room for recurrent synapses. Increase "
        f"num_branch * num_synapse_per_branch."
    )

    num_ff = num_input
    num_rec = num_neuron_inputs - num_ff

    # feed-forward
    feedforward_indices = jnp.broadcast_to(
        jnp.arange(num_input)[None, :], (num_neuron, num_input)
    )

    # recurrent
    key, subkey = jrandom.split(key)
    recurrent_indices = jrandom.choice(
        key=subkey,
        a=num_neuron,
        shape=(num_neuron, num_rec),
        replace=True,
    )
    recurrent_indices_offset = recurrent_indices + num_input

    # combine + shuffle 
    base_connections = jnp.concatenate(
        [feedforward_indices, recurrent_indices_offset], axis=1
    )
    key, subkey = jrandom.split(key)
    shuffled = jrandom.permutation(subkey, base_connections, axis=1, independent=True)

    return shuffled



# overlapping 1D input windows per module, recurrent connections biased within module
def flexible_modular(
    key: PRNGKeyArray,
    num_input: int,
    num_neuron: int,
    num_neuron_inputs: int,
    num_modules: int,
    input_overlap: float = 0.2,
    within_module_connect: float = 0.9,   #  recurrent connections to same module
    inter_module_connect: float = 0.1,    #  recurrent connections to any other module
    **kwargs,
) -> Integer[Array, "neuron synapse"]:


    assert num_neuron % num_modules == 0, (
        f"num_neuron ({num_neuron}) must be divisible by num_modules ({num_modules})"
    )
    assert 0.0 <= input_overlap < 1.0, "input_overlap must be in [0.0, 1.0)"

    neurons_per_module = num_neuron // num_modules

    # window + stride
    if num_modules == 1:
        window_size = num_input
        stride = 0
        start_offset = 0
    else:
        window_size = int(
            jnp.round(num_input / ((num_modules - 1) * (1 - input_overlap) + 1))
        )
        stride = int(jnp.round(window_size * (1 - input_overlap)))

        total_covered = (num_modules - 1) * stride + window_size
        missing = num_input - total_covered
        start_offset = missing // 2

    inputs_per_module = window_size

    num_recurrent = int(num_neuron_inputs - inputs_per_module)
    assert num_recurrent >= 0, (
        f"num_neuron_inputs ({num_neuron_inputs}) must be >= inputs_per_module ({inputs_per_module})"
    )

    # module assignment
    neuron_indices = jnp.arange(num_neuron, dtype=jnp.int32)
    module_indices = neuron_indices // neurons_per_module

    # feedforward 
    input_start = module_indices * stride + start_offset
    feedforward_indices = input_start[:, None] + jnp.arange(inputs_per_module)
    feedforward_indices = jnp.clip(feedforward_indices, 0, num_input - 1)

    # handle last module 
    last_window_end = feedforward_indices[-1, -1]
    last_window_size = last_window_end - feedforward_indices[-1, 0] + 1
    half_window = inputs_per_module // 2
    if last_window_size < half_window:
        remove = half_window - last_window_size
        trim_top = remove // 2
        trim_bottom = remove - trim_top
        feedforward_indices = feedforward_indices[:, trim_top:inputs_per_module - trim_bottom]
        inputs_per_module = feedforward_indices.shape[1]

    # recurrent  
    num_same = neurons_per_module - 1         
    num_other = num_neuron - neurons_per_module  

    same_module_mask = module_indices[:, None] == module_indices[None, :]
    self_mask = neuron_indices[:, None] == neuron_indices[None, :]

    num_same  = neurons_per_module - 1           
    num_other = num_neuron - neurons_per_module  

    w_within = within_module_connect / num_same   if num_same  > 0 else 0.0
    w_inter  = inter_module_connect  / num_other  if num_other > 0 else 0.0

    rec_conn_probs = jnp.where(
        self_mask,
        0.0,
        jnp.where(
            same_module_mask,
            w_within,
            w_inter,
        ),
    )

    rec_conn_probs = rec_conn_probs / rec_conn_probs.sum(axis=1, keepdims=True)

    key, subkey = jrandom.split(key)
    keys = jrandom.split(subkey, num_neuron)

    def sample_recurrent_for_neuron(neuron_key, probs):
        return jrandom.choice(
            key=neuron_key,
            a=num_neuron,
            p=probs,
            shape=(num_recurrent,),
            replace=True,
        )

    recurrent_indices = vmap(sample_recurrent_for_neuron)(keys, rec_conn_probs)
    recurrent_indices_offset = recurrent_indices + num_input

    # combine + shuffle
    base_connections = jnp.concatenate(
        [feedforward_indices, recurrent_indices_offset], axis=1
    )

    key, subkey = jrandom.split(key)
    keys = jrandom.split(subkey, num_neuron)
    shuffled = vmap(jrandom.permutation)(keys, base_connections)

    return shuffled



# ff connections are modular, recurrent connections follow `recurrence`
def structured_ff_random_rec(
    key: PRNGKeyArray,
    num_input: int,
    num_neuron: int,
    num_neuron_inputs: int,
    num_modules: int,
    input_overlap: float = 0.2,
    recurrence: str = "random",       # "random" | "small_world" | "modular"
    neighborhood_size: int = 10,      # small_world: k neighbors on each side of the ring
    p_rewire: float = 0.1,            # small_world: Watts-Strogatz rewiring probability
    p_inter: float = 0.1,             # modular: fraction of recurrent targets outside the module
    **kwargs,
) -> Integer[Array, "neuron synapse"]:

    
    assert num_neuron % num_modules == 0, (
        f"num_neuron ({num_neuron}) must be divisible by num_modules ({num_modules})"
    )
    assert 0.0 <= input_overlap < 1.0, "input_overlap must be in [0.0, 1.0)"
    assert recurrence in ("random", "small_world", "modular"), (
        f"recurrence ({recurrence!r}) must be 'random', 'small_world' or 'modular'"
    )
    assert 0.0 <= p_rewire <= 1.0, "p_rewire must be in [0.0, 1.0]"
    assert 0.0 <= p_inter <= 1.0, "p_inter must be in [0.0, 1.0]"

    neurons_per_module = num_neuron // num_modules

    # feedforward 
    base_window = -(-num_input // num_modules)  # ceil division
    margin = 0 if num_modules == 1 else int(round(base_window * input_overlap / 2))
    inputs_per_module = min(base_window + 2 * margin, num_input)

    num_recurrent = int(num_neuron_inputs - inputs_per_module)
    assert num_recurrent >= 0, (
        f"num_neuron_inputs ({num_neuron_inputs}) must be >= inputs_per_module ({inputs_per_module})"
    )

    
    neuron_indices = jnp.arange(num_neuron, dtype=jnp.int32)
    module_indices = neuron_indices // neurons_per_module

    
    partition_start = (module_indices * num_input) // num_modules
    input_start = jnp.clip(partition_start - margin, 0, num_input - inputs_per_module)
    feedforward_indices = input_start[:, None] + jnp.arange(inputs_per_module)

    # recurrent
    row = neuron_indices[:, None]  
    col = neuron_indices[None, :]  
    self_mask = row == col

    def mix_groups(near_mask, p_far):
        near = near_mask & (~self_mask)
        far = (~near_mask) & (~self_mask)
        near = near / jnp.clip(near.sum(axis=1, keepdims=True), 1)
        far = far / jnp.clip(far.sum(axis=1, keepdims=True), 1)
        return (1.0 - p_far) * near + p_far * far

    if recurrence == "random":
        # Uniform over all neurons.
        rec_conn_probs = (~self_mask).astype(jnp.float32)

    elif recurrence == "small_world":
        # Watts-Strogatz on a frequency-ordered wrapping ring.
        assert neighborhood_size >= 1, "neighborhood_size must be >= 1"
        assert 2 * neighborhood_size < num_neuron, (
            f"2 * neighborhood_size ({2 * neighborhood_size}) must be < "
            f"num_neuron ({num_neuron})"
        )
        ring_dist = jnp.abs(row - col)
        ring_dist = jnp.minimum(ring_dist, num_neuron - ring_dist)  # wrap-around
        local_mask = ring_dist <= neighborhood_size
        rec_conn_probs = mix_groups(local_mask, p_rewire)

    elif recurrence == "modular":
        # Weak modularity on frequency-ordered blocks.
        same_module = module_indices[:, None] == module_indices[None, :]
        rec_conn_probs = mix_groups(same_module, p_inter)

    rec_conn_probs = rec_conn_probs / rec_conn_probs.sum(axis=1, keepdims=True)

    key, subkey = jrandom.split(key)
    rec_keys = jrandom.split(subkey, num_neuron)

    def sample_recurrent_for_neuron(neuron_key, probs):
        return jrandom.choice(
            key=neuron_key,
            a=num_neuron,
            p=probs,
            shape=(num_recurrent,),
            replace=True,
        )

    recurrent_indices = vmap(sample_recurrent_for_neuron)(rec_keys, rec_conn_probs)
    recurrent_indices_offset = recurrent_indices + num_input

    base_connections = jnp.concatenate(
        [feedforward_indices, recurrent_indices_offset], axis=1
    )

    key, subkey = jrandom.split(key)
    keys = jrandom.split(subkey, num_neuron)
    shuffled = vmap(jrandom.permutation)(keys, base_connections)

    return shuffled



# 2D input tiles per module, separate ON/OFF polarities, recurrence within module
def spatial_structure(
    key: PRNGKeyArray,
    num_input: int,           
    num_neuron: int,          
    num_neuron_inputs: int,
    input_size: int,
    num_modules: int,
    recurrent_connect: float = 0.5,
    **kwargs,
) -> Integer[Array, "neuron synapse"]:

    assert num_input == 2 * input_size ** 2
    assert num_neuron % 2 == 0
    assert num_neuron % num_modules == 0
    assert 0.0 <= recurrent_connect < 1.0

    num_neuron_per_polarity = num_neuron // 2 
    num_modules_per_polarity = num_modules // 2

    modules_per_side = int(jnp.sqrt(num_modules_per_polarity))
    assert modules_per_side ** 2 == num_modules_per_polarity
    assert input_size % modules_per_side == 0

    # module sizes (per polarity) 
    neurons_per_module    = num_neuron_per_polarity // num_modules_per_polarity
    input_per_module_side = input_size // modules_per_side
    inputs_per_module     = input_per_module_side ** 2

    num_rec_connections = int(round(num_neuron_inputs * recurrent_connect))
    num_ff_connections  = num_neuron_inputs - num_rec_connections
    assert num_ff_connections > 0, "recurrent_connect too high — no FF synapses left"
    assert num_rec_connections == 0 or neurons_per_module > 1, (
        "neurons_per_module must be > 1 to allow within-module recurrent connections"
    )

    # spatial input structure (same for both polarities) 
    input_indices  = jnp.arange(input_size ** 2, dtype=jnp.int32)
    input_y        = input_indices // input_size
    input_x        = input_indices % input_size
    input_module_y = input_y // input_per_module_side
    input_module_x = input_x // input_per_module_side

    def get_module_ff_indices(my, mx, input_offset):
        mask = (input_module_y == my) & (input_module_x == mx)
        return input_indices[mask] + input_offset  

    module_ids = jnp.arange(num_modules_per_polarity, dtype=jnp.int32)
    module_my  = module_ids // modules_per_side
    module_mx  = module_ids % modules_per_side

    def build_polarity_connectivity(input_offset, neuron_offset, key):
        module_ff = jnp.stack([
            get_module_ff_indices(int(module_my[m]), int(module_mx[m]), input_offset)
            for m in range(num_modules_per_polarity)
        ], axis=0)

        neuron_module_id = jnp.arange(num_neuron_per_polarity, dtype=jnp.int32) // neurons_per_module
        neuron_rfs       = module_ff[neuron_module_id]  

        # feedforward
        key, ff_key = jrandom.split(key)
        ff_keys = jrandom.split(ff_key, num_neuron_per_polarity)

        def sample_ff(neuron_key, rf):
            local_idx = jrandom.choice(
                neuron_key,
                a=inputs_per_module,
                shape=(num_ff_connections,),
                replace=True,
            )
            return rf[local_idx]

        ff_connectivity = vmap(sample_ff)(ff_keys, neuron_rfs)

        # recurrent
        if num_rec_connections > 0:
            key, rec_key = jrandom.split(key)
            rec_keys = jrandom.split(rec_key, num_neuron_per_polarity)
            local_indices = jnp.arange(num_neuron_per_polarity, dtype=jnp.int32)

            def sample_rec(neuron_key, local_i):
                mod        = local_i // neurons_per_module
                mod_start  = mod * neurons_per_module
                all_in_mod = mod_start + jnp.arange(neurons_per_module, dtype=jnp.int32)
                mask       = all_in_mod != local_i
                masked     = jnp.where(mask, all_in_mod, num_neuron_per_polarity)  # sentinel
                sorted_rec = jnp.sort(masked)[:neurons_per_module - 1]
                picks = jrandom.randint(
                    neuron_key,
                    shape=(num_rec_connections,),
                    minval=0,
                    maxval=neurons_per_module - 1,
                )
                return sorted_rec[picks] + neuron_offset + num_input

            rec_connectivity = vmap(sample_rec)(rec_keys, local_indices)
        else:
            rec_connectivity = jnp.empty((num_neuron_per_polarity, 0), dtype=jnp.int32)

        base_connections = jnp.concatenate([ff_connectivity, rec_connectivity], axis=1)

        key, shuffle_key = jrandom.split(key)
        keys     = jrandom.split(shuffle_key, num_neuron_per_polarity)
        shuffled = vmap(jrandom.permutation)(keys, base_connections)
        return shuffled

    key, key_on, key_off = jrandom.split(key, 3)

    on_connectivity  = build_polarity_connectivity(
        input_offset=0,
        neuron_offset=0,
        key=key_on
    )
    off_connectivity = build_polarity_connectivity(
        input_offset=input_size ** 2,
        neuron_offset=num_neuron_per_polarity,
        key=key_off
    )

    return jnp.concatenate([on_connectivity, off_connectivity], axis=0)  

# ff conenctions are spatial, recurrent connections are random
def spatial_ff_random_rec(
    key: PRNGKeyArray,
    num_input: int,           
    num_neuron: int,          
    num_neuron_inputs: int,
    input_size: int,
    num_modules: int,
    recurrent_connect: float = 0.5,
    **kwargs,
) -> Integer[Array, "neuron synapse"]:

    assert num_input == 2 * input_size ** 2
    assert num_neuron % 2 == 0
    assert num_neuron % num_modules == 0
    assert 0.0 <= recurrent_connect < 1.0

    num_neuron_per_polarity = num_neuron // 2  
    num_modules_per_polarity = num_modules // 2

    modules_per_side = int(jnp.sqrt(num_modules_per_polarity))
    assert modules_per_side ** 2 == num_modules_per_polarity
    assert input_size % modules_per_side == 0

    
    neurons_per_module    = num_neuron_per_polarity // num_modules_per_polarity
    input_per_module_side = input_size // modules_per_side
    inputs_per_module     = input_per_module_side ** 2

    num_rec_connections = int(round(num_neuron_inputs * recurrent_connect))
    num_ff_connections  = num_neuron_inputs - num_rec_connections
    assert num_ff_connections > 0, "recurrent_connect too high — no FF synapses left"

    input_indices  = jnp.arange(input_size ** 2, dtype=jnp.int32)
    input_y        = input_indices // input_size
    input_x        = input_indices % input_size
    input_module_y = input_y // input_per_module_side
    input_module_x = input_x // input_per_module_side

    def get_module_ff_indices(my, mx, input_offset):
        mask = (input_module_y == my) & (input_module_x == mx)
        return input_indices[mask] + input_offset  

    module_ids = jnp.arange(num_modules_per_polarity, dtype=jnp.int32)
    module_my  = module_ids // modules_per_side
    module_mx  = module_ids % modules_per_side

    def build_polarity_connectivity(input_offset, key):
        
        
        module_ff = jnp.stack([
            get_module_ff_indices(int(module_my[m]), int(module_mx[m]), input_offset)
            for m in range(num_modules_per_polarity)
        ], axis=0)

        neuron_module_id = jnp.arange(num_neuron_per_polarity, dtype=jnp.int32) // neurons_per_module
        neuron_rfs       = module_ff[neuron_module_id]  # (num_neuron_per_polarity, inputs_per_module)

        # feedforward: sample num_ff_connections per neuron from its own receptive field
        key, ff_key = jrandom.split(key)
        ff_keys = jrandom.split(ff_key, num_neuron_per_polarity)

        def sample_ff(neuron_key, rf):
            local_idx = jrandom.choice(
                neuron_key,
                a=inputs_per_module,
                shape=(num_ff_connections,),
                replace=True,
            )
            return rf[local_idx]

        ff_connectivity = vmap(sample_ff)(ff_keys, neuron_rfs)

        # recurrent: random across all neurons 
        if num_rec_connections > 0:
            key, rec_key = jrandom.split(key)
            rec_connectivity = jrandom.choice(
                key=rec_key,
                a=num_neuron,
                shape=(num_neuron_per_polarity, num_rec_connections),
                replace=True,
            ) + num_input
        else:
            rec_connectivity = jnp.empty((num_neuron_per_polarity, 0), dtype=jnp.int32)

        base_connections = jnp.concatenate([ff_connectivity, rec_connectivity], axis=1)

        key, shuffle_key = jrandom.split(key)
        keys     = jrandom.split(shuffle_key, num_neuron_per_polarity)
        shuffled = vmap(jrandom.permutation)(keys, base_connections)
        return shuffled

    key, key_on, key_off = jrandom.split(key, 3)

    on_connectivity  = build_polarity_connectivity(input_offset=0, key=key_on)
    off_connectivity = build_polarity_connectivity(input_offset=input_size ** 2, key=key_off)

    return jnp.concatenate([on_connectivity, off_connectivity], axis=0)  

# each neuron cluster reads one input channel, optionally under a spatial prior
def dir_selective_wiring(
    key: PRNGKeyArray,
    num_input: int,                                       
    num_neuron: int,                                      
    num_neuron_inputs: int,
    input_size: int,
    num_channels: int = 8,
    recurrent_connect: float = 0.5,
    cross_channel_recurrent: bool = True,
    spatial_prior: str = "uniform",                      
    gaussian_sigma_frac: float = 0.3,                     
    ff_conn_probs_per_channel: Optional[Array] = None,   
    **kwargs,
) -> Integer[Array, "neuron synapse"]:

    assert num_input == num_channels * input_size ** 2, (
        f"num_input ({num_input}) must equal num_channels ({num_channels}) * "
        f"input_size**2 ({input_size ** 2})"
    )
    assert num_neuron % num_channels == 0, (
        f"num_neuron ({num_neuron}) must be divisible by num_channels ({num_channels})"
    )
    assert 0.0 <= recurrent_connect < 1.0

    num_pixels          = input_size ** 2
    neurons_per_cluster = num_neuron // num_channels

    num_rec = int(round(num_neuron_inputs * recurrent_connect))
    num_ff  = num_neuron_inputs - num_rec
    assert num_ff > 0, "recurrent_connect too high — no FF synapses left"

    # spatial prior per channel 
    if ff_conn_probs_per_channel is not None:
        assert ff_conn_probs_per_channel.shape == (num_channels, num_pixels), (
            f"ff_conn_probs_per_channel shape {ff_conn_probs_per_channel.shape} "
            f"must be ({num_channels}, {num_pixels})"
        )
        probs_per_channel = ff_conn_probs_per_channel / jnp.sum(
            ff_conn_probs_per_channel, axis=1, keepdims=True
        )
    elif spatial_prior == "gaussian":
        yy, xx = jnp.meshgrid(jnp.arange(input_size), jnp.arange(input_size), indexing="ij")
        cy = cx = (input_size - 1) / 2.0
        sigma = gaussian_sigma_frac * input_size
        g = jnp.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2)).flatten()
        g = g / g.sum()
        probs_per_channel = jnp.broadcast_to(g[None, :], (num_channels, num_pixels))
    elif spatial_prior == "uniform":
        probs_per_channel = jnp.ones((num_channels, num_pixels)) / num_pixels
    else:
        raise ValueError(f"Unknown spatial_prior: {spatial_prior!r}")

    # feed-forward: per cluster sample from input channel
    key, subkey = jrandom.split(key)
    ff_keys = jrandom.split(subkey, num_channels)

    ff_list = []
    for c in range(num_channels):
        ff_c = jrandom.choice(
            key=ff_keys[c],
            a=num_pixels,
            p=probs_per_channel[c],
            shape=(neurons_per_cluster, num_ff),
            replace=True,
        ) + c * num_pixels
        ff_list.append(ff_c)
    feedforward_indices = jnp.concatenate(ff_list, axis=0)  # (num_neuron, num_ff)

    # recurrent: random across all neurons, or within cluster only 
    if num_rec > 0:
        key, subkey = jrandom.split(key)
        if cross_channel_recurrent:
            recurrent_indices = jrandom.choice(
                key=subkey,
                a=num_neuron,
                shape=(num_neuron, num_rec),
                replace=True,
            )
        else:
            rec_keys = jrandom.split(subkey, num_channels)
            rec_list = []
            for c in range(num_channels):
                rec_c = jrandom.choice(
                    key=rec_keys[c],
                    a=neurons_per_cluster,
                    shape=(neurons_per_cluster, num_rec),
                    replace=True,
                ) + c * neurons_per_cluster
                rec_list.append(rec_c)
            recurrent_indices = jnp.concatenate(rec_list, axis=0)
        recurrent_indices_offset = recurrent_indices + num_input
    else:
        recurrent_indices_offset = jnp.empty((num_neuron, 0), dtype=jnp.int32)

    # combine + shuffle 
    base_connections = jnp.concatenate(
        [feedforward_indices, recurrent_indices_offset], axis=1
    )
    key, subkey = jrandom.split(key)
    shuffled = jrandom.permutation(subkey, base_connections, axis=1, independent=True)

    return shuffled


# --------------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------------
# Maps the `input_wiring` config string to its sampler. 
# Every sampler takes (key, num_input, num_neuron, num_neuron_inputs, **wiring_args) 
# and returns Integer[Array, "neuron synapse"]

WIRING_REGISTRY = {
    "random": random_connect,
    "fully_connected": fully_connected,
    "fully_connected_ff_random_rec": fully_connected_ff_random_rec,
    "flexible_modular": flexible_modular,
    "structured_ff_random_rec": structured_ff_random_rec,
    "spatial_structure": spatial_structure,
    "spatial_ff_random_rec": spatial_ff_random_rec,
    "dir_selective_wiring": dir_selective_wiring,
}

INPUT_WIRINGS = list(WIRING_REGISTRY)

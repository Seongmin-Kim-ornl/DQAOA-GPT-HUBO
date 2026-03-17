out_dir = '{out_dir}'
eval_interval = 250
eval_iters = 200
log_interval = 10
eval_ar_every = 5000

always_save_checkpoint = False

dataset = '{dataset}'
gradient_accumulation_steps = 1
batch_size = 64
block_size = {block_size}

n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

learning_rate = 1e-4
n_epochs = 70
max_iters = 30000
lr_decay_iters = 30000
min_lr = 1e-5
beta2 = 0.95

warmup_iters = 100

graph_emb_dim = 500
use_graph_emb = {use_graph_emb}
pool_type = '{pool_type}'
n_nodes = {n_nodes}
token_seq_round = '{token_seq_round}'
n_samples = 5

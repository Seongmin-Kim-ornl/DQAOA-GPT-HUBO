out_dir = 'out-hubo_16nodes_multigamma'
eval_interval = 250
eval_iters = 200
log_interval = 10
eval_ar_every = 5000

always_save_checkpoint = False

dataset = 'hubo_16nodes_multigamma'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 1800

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
use_graph_emb = True
pool_type = 'all_pool'
n_nodes = 16
token_seq_round = 'token_seq_round_d1'
n_samples = 5

import os
import torch
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="LightGCN Pretraining for Recommendation Embeddings")

    parser.add_argument('--dataset', type=str, default='Beibei', choices=['Beibei', 'IJCAI_15', 'Tmall'], help="Dataset to use for training")
    parser.add_argument('--behaviors', type=str, nargs='+', default=['pv', 'cart', 'buy'], help="List of behaviors to pretrain")
    # parser.add_argument('--behaviors', type=str, nargs='+', default=['pv', 'fav', 'cart', 'buy'], help="List of behaviors to pretrain")
    parser.add_argument('--embedding_dim', type=int, default=64, help="Embedding dimension size")
    parser.add_argument('--n_layers', type=int, default=3, help="Number of layers for LightGCN")
    parser.add_argument('--lr', type=float, nargs='*', default=[0.001, 0.001, 0.001], help="Learning rate(s)")
    parser.add_argument('--decay', type=float, default=1e-4, help="Weight decay coefficient")
    parser.add_argument('--batch_size', type=int, default=2048, help="Batch size")
    parser.add_argument('--epochs', type=int, default=5, help="Number of training epochs")
    parser.add_argument('--seed', type=int, default=2025, help="Random seed")
    parser.add_argument('--data_path', type=str, default="./datasets", help="Root directory path for datasets")
    parser.add_argument('--dropout', action='store_true', help="Enable dropout during training")
    parser.add_argument('--keep_prob', type=float, default=0.6, help="Keep probability for dropout")
    parser.add_argument('--A_split', action='store_true', help="Split adjacency matrix for memory optimization")
    parser.add_argument('--A_n_fold', type=int, default=100, help="Number of folds for splitting adjacency matrix")

    # Diffusion model specific arguments
    parser.add_argument('--diff_epochs', type=int, default=200, help='Number of diffusion training epochs.')
    parser.add_argument('--diff_batch_size', type=int, default=1024, help='Batch size for diffusion training.')
    parser.add_argument('--timesteps', type=int, default=10, help='Timesteps for diffusion.')
    parser.add_argument('--beta_end', type=float, default=0.02, help='Beta end of diffusion.')
    parser.add_argument('--beta_start', type=float, default=0.0001, help='Beta start of diffusion.')
    parser.add_argument('--diff_lr', type=float, default=0.0001, help='Learning rate for diffusion model.')
    parser.add_argument('--lambda1', type=float, default=0.1, help='Weight for hard negative BPR loss.')
    parser.add_argument('--lambda2', type=float, default=0.1, help='Weight for diffusion loss.')
    parser.add_argument('--denoising_dims', type=str, default='[200,600]', help='Hidden dimensions for denoising network')
    parser.add_argument('--time_emb_dim', type=int, default=10, help='Time embedding dimension')
    parser.add_argument('--mean_type', type=str, default='x0', choices=['x0', 'eps'], help='Mean type for diffusion')
    parser.add_argument('--noise_schedule', type=str, default='linear-var', help='Noise schedule')
    parser.add_argument('--reweight', action='store_true', help='Use reweighting in diffusion loss')

    args = parser.parse_args()
    args = validate_learning_rates(args)

    return args


def validate_learning_rates(args):
    """Argument for learning rates and validation"""
    if len(args.lr) == 1:
        args.lr = args.lr * len(args.behaviors)
    elif len(args.lr) != len(args.behaviors):
        if len(args.lr) < len(args.behaviors):
            last_lr = args.lr[-1]
            args.lr.extend([last_lr] * (len(args.behaviors) - len(args.lr)))
        else:
            args.lr = args.lr[:len(args.behaviors)]

    args.behavior_lr_map = dict(zip(args.behaviors, args.lr))
    return args


def get_learning_rate_for_behavior(behavior):
    """Get learning rate for a specific behavior, default to first lr if not found"""
    return args.behavior_lr_map.get(behavior, args.lr[0])


# Global configurations
args = parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT_PATH, 'data')
EMBEDDING_PATH = os.path.join(ROOT_PATH, 'embeddings')
os.makedirs(EMBEDDING_PATH, exist_ok=True)


def print_config(current_behavior=None):
    print("=" * 60)
    print("Axiliary-behavior-enhanced diffusion model for recommendation")
    print(f"Dataset: {args.dataset}")
    if current_behavior:
        print(f"Current Behavior: {current_behavior}")
        print(f"Learning rate for {current_behavior}: {get_learning_rate_for_behavior(current_behavior)}")
    else:
        print(f"Behaviors to train: {', '.join(args.behaviors)}")
        print("Learning rates for each behavior:")
        for behavior, lr in args.behavior_lr_map.items():
            print(f"  {behavior}: {lr}")
    print(f"Device: {device}")
    print(f"Random Seed: {args.seed}")
    print(f"Embedding dimension: {args.embedding_dim}")
    print(f"Diffusion timesteps: {args.timesteps}")
    print("=" * 60)
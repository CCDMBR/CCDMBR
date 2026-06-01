import os
import json
import argparse
import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader, Sampler
import torch.optim as optim
from time import time
from config import args, get_learning_rate_for_behavior


def parse_args():
    parser = argparse.ArgumentParser(description="Run Enhanced Multi-Behavior Diffusion for recommendation.")
    parser.add_argument('--epoch', type=int, default=200, help='Number of max epochs.')
    parser.add_argument('--data', nargs='?', default='Beibei', help='Dataset name (folder name)')
    parser.add_argument('--random_seed', type=int, default=2025, help='Random seed.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
    parser.add_argument('--hidden_factor', type=int, default=64, help='Number of hidden factors')
    parser.add_argument('--timesteps', type=int, default=20, help='Timesteps for diffusion.')
    parser.add_argument('--beta_end', type=float, default=0.02, help='Beta end of diffusion.')
    parser.add_argument('--beta_start', type=float, default=0.0001, help='Beta start of diffusion.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--l2_decay', type=float, default=0, help='L2 loss reg coef.')
    parser.add_argument('--cuda', type=int, default=0, help='CUDA device.')
    parser.add_argument('--dropout_rate', type=float, default=0.2, help='Dropout rate.')
    parser.add_argument('--report_epoch', type=bool, default=True, help='Report frequency.')
    parser.add_argument('--optimizer', type=str, default='adam', help='Type of optimizer.')
    parser.add_argument('--beta_sche', nargs='?', default='linear', help='Beta schedule.')
    parser.add_argument('--descri', type=str, default='', help='Description of the work.')
    parser.add_argument('--lambda1', type=float, default=0.1, help='Weight for hard negative BPR loss.')
    parser.add_argument('--lambda2', type=float, default=0.1, help='Weight for diffusion loss.')

    parser.add_argument('--behavior_encoding_dim', type=int, default=16, help='Behavior encoding dimension.')
    parser.add_argument('--contrastive_weight', type=float, default=0.1, help='Weight for contrastive loss.')

    return parser.parse_args()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class MultiBehaviorDataset:
    def __init__(self, data_directory, device):
        self.data_directory = data_directory
        self.behavior_files = ['pv.txt', 'cart.txt', 'buy.txt']
        # self.behavior_files = ['pv.txt', 'fav.txt', 'cart.txt', 'buy.txt']
        self.count_file = 'count.txt'
        self.validation_file = 'validation.txt'
        self.test_file = 'test.txt'
        self.device = device

        with open(os.path.join(data_directory, self.count_file), 'r') as f:
            count_data = json.loads(f.read().strip())
            self.num_users = count_data['user']
            self.num_items = count_data['item']
        print(f"Dataset loaded: {self.num_users} users, {self.num_items} items")

        self.behavior_interactions = []
        for behavior_file in self.behavior_files:
            interactions = self.load_interactions(os.path.join(data_directory, behavior_file))
            self.behavior_interactions.append(interactions)
            print(f"Loaded {behavior_file}: {len(interactions)} interactions")

        self.validation_interactions = self.load_interactions(os.path.join(data_directory, self.validation_file))
        self.test_interactions = self.load_interactions(os.path.join(data_directory, self.test_file))

        print(f"Validation set: {len(self.validation_interactions)} interactions")
        print(f"Test set: {len(self.test_interactions)} interactions")

        self.train_user_dict = self.create_user_dict(self.behavior_interactions[-1])
        self.val_user_dict = self.create_user_dict(self.validation_interactions)
        self.test_user_dict = self.create_user_dict(self.test_interactions)

        self.behavior_user_dicts = []
        for interactions in self.behavior_interactions:
            self.behavior_user_dicts.append(self.create_user_dict(interactions))

        self.hard_negative_dict = self.create_high_quality_hard_negative_dict()

    def load_interactions(self, file_path):
        interactions = []
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 2:
                        user_id, item_id = int(parts[0]), int(parts[1])
                        interactions.append((user_id, item_id))
        except FileNotFoundError:
            print(f"Warning: File {file_path} not found")
        return interactions

    def create_user_dict(self, interactions):
        user_dict = {}
        for user, item in interactions:
            if user not in user_dict:
                user_dict[user] = []
            user_dict[user].append(item)
        return user_dict

    def create_high_quality_hard_negative_dict(self):
        hard_negative_dict = {}
        target_behavior_dict = self.behavior_user_dicts[-1]
        auxiliary_behavior_dicts = self.behavior_user_dicts[:-1]

        total_original_hard_negatives = 0
        total_high_quality_hard_negatives = 0
        user_interaction_freq_stats = defaultdict(int)

        for user_id in range(self.num_users):
            user_item_interaction_count = defaultdict(int)

            for behavior_dict in auxiliary_behavior_dicts:
                user_items = behavior_dict.get(user_id, [])
                for item in user_items:
                    user_item_interaction_count[item] += 1

            target_items = set(target_behavior_dict.get(user_id, []))

            original_hard_negs = set(user_item_interaction_count.keys()) - target_items
            total_original_hard_negatives += len(original_hard_negs)

            high_quality_hard_negs = []
            for item in original_hard_negs:
                interaction_count = user_item_interaction_count[item]
                if interaction_count >= 2:
                    high_quality_hard_negs.append(item)
                    user_interaction_freq_stats[interaction_count] += 1

            if high_quality_hard_negs:
                hard_negative_dict[user_id] = high_quality_hard_negs
                total_high_quality_hard_negatives += len(high_quality_hard_negs)

        if total_original_hard_negatives > 0:
            improvement_ratio = (
                                        total_original_hard_negatives - total_high_quality_hard_negatives) / total_original_hard_negatives
            print(f"  improvement_ratio: {improvement_ratio * 100:.1f}%")

        if len(hard_negative_dict) > 0:
            avg_hard_negs = total_high_quality_hard_negatives / len(hard_negative_dict)
            print(f"  avg_hard_negs: {avg_hard_negs:.2f}")

        return hard_negative_dict

    def get_training_batch(self, batch_size):
        user_ids = []
        pos_items = []

        users = list(self.train_user_dict.keys())
        if len(users) == 0:
            return torch.LongTensor([]), torch.LongTensor([])

        sampled_users = random.sample(users, min(batch_size, len(users)))

        for user in sampled_users:
            user_ids.append(user)
            pos_item = random.choice(self.train_user_dict[user])
            pos_items.append(pos_item)

        return torch.LongTensor(user_ids).to(self.device), torch.LongTensor(pos_items).to(self.device)

    def get_all_hard_negatives_for_batch(self, user_ids):
        hard_neg_items = []
        hard_neg_user_indices = []
        hard_neg_counts = []

        for i, user_id in enumerate(user_ids):
            user_id_int = user_id.item() if torch.is_tensor(user_id) else user_id
            user_hard_negs = self.hard_negative_dict.get(user_id_int, [])

            hard_neg_counts.append(len(user_hard_negs))

            if len(user_hard_negs) > 0:
                hard_neg_items.extend(user_hard_negs)
                hard_neg_user_indices.extend([i] * len(user_hard_negs))

        if len(hard_neg_items) == 0:
            return (torch.LongTensor([]).to(self.device),
                    torch.LongTensor([]).to(self.device),
                    torch.LongTensor(hard_neg_counts).to(self.device))

        return (torch.LongTensor(hard_neg_items).to(self.device),
                torch.LongTensor(hard_neg_user_indices).to(self.device),
                torch.LongTensor(hard_neg_counts).to(self.device))


def evaluate(model, diffusion_model, dataset, topk, device, mode='validation', steps=2, sampling_noise=False):
    model.eval()

    if mode == 'validation':
        eval_dict = dataset.val_user_dict
    else:  # test mode
        eval_dict = dataset.test_user_dict

    hit_rates = {k: 0 for k in topk}
    ndcg_values = {k: 0 for k in topk}
    valid_user_count = 0

    batch_size = 1024
    user_ids = list(eval_dict.keys())

    with torch.no_grad():
        for i in range(0, len(user_ids), batch_size):
            batch_users = user_ids[i:i + batch_size]
            batch_tensor = torch.LongTensor(batch_users).to(device)

            try:
                batch_predictions = model.predict(
                    batch_tensor,
                    diffusion_model,
                    steps=steps,
                    sampling_noise=sampling_noise
                )

                for j, user_id in enumerate(batch_users):
                    if not eval_dict[user_id]:
                        continue

                    valid_user_count += 1
                    true_items = eval_dict[user_id]
                    user_predictions = batch_predictions[j]

                    train_items = dataset.train_user_dict.get(user_id, [])
                    for item in train_items:
                        user_predictions[item] = float('-inf')

                    if mode == 'test':
                        val_items = dataset.val_user_dict.get(user_id, [])
                        for item in val_items:
                            user_predictions[item] = float('-inf')

                    _, recommended_items = torch.topk(user_predictions, max(topk))
                    recommended_items = recommended_items.cpu().numpy()

                    calculate_metrics(recommended_items, true_items, topk, hit_rates, ndcg_values)

            except Exception as e:
                print(f"Error processing batch {i // batch_size}: {e}")
                continue

    if valid_user_count == 0:
        print(f"No users available for {mode} evaluation")
        return 0.0

    avg_hit_rates = []
    avg_ndcg_values = []

    header_format = ''
    header_values = []
    for k in topk:
        header_format += '{:<10s} {:<10s} '
        header_values.extend([f'HR@{k}', f'NDCG@{k}'])

    print(header_format.format(*header_values))

    result_format = ''
    result_values = []
    for i, k in enumerate(topk):
        hr = hit_rates[k] / valid_user_count
        ndcg = ndcg_values[k] / valid_user_count
        avg_hit_rates.append(hr)
        avg_ndcg_values.append(ndcg)
        result_format += '{:<10.6f} {:<10.6f} '
        result_values.extend([hr, ndcg])

    print(result_format.format(*result_values))

    if len(topk) >= 2:
        return avg_hit_rates[1]
    else:
        return avg_hit_rates[0]


def calculate_metrics(recommended_items, true_items, topk_list, hit_rates, ndcg_values):
    true_items_set = set(true_items)

    for k in topk_list:
        topk_items = recommended_items[:k]

        # Hit Rate
        hit = 0
        for item in topk_items:
            if item in true_items_set:
                hit = 1
                break
        hit_rates[k] += hit

        # NDCG
        dcg = 0
        idcg = 0

        for i, item in enumerate(topk_items):
            if item in true_items_set:
                dcg += 1 / np.log2(i + 2)

        for i in range(min(k, len(true_items))):
            idcg += 1 / np.log2(i + 2)

        ndcg = dcg / idcg if idcg > 0 else 0
        ndcg_values[k] += ndcg


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def UniformSample(dataset):
    allPos = dataset._allPos
    users = np.random.randint(0, dataset.n_users, dataset.trainDataSize)
    S = []

    for user in users:
        posForUser = allPos[user]
        if len(posForUser) == 0:
            continue

        positem = np.random.choice(posForUser)

        while True:
            negitem = np.random.randint(0, dataset.m_items)
            if negitem not in posForUser:
                break

        S.append([user, positem, negitem])

    return np.array(S)


def minibatch(users, pos_items, neg_items, batch_size=None):
    if batch_size is None:
        batch_size = args.batch_size

    for i in range(0, len(users), batch_size):
        yield (users[i:i + batch_size],
               pos_items[i:i + batch_size],
               neg_items[i:i + batch_size])


def shuffle_data(users, pos_items, neg_items):
    shuffle_idx = torch.randperm(len(users))
    return users[shuffle_idx], pos_items[shuffle_idx], neg_items[shuffle_idx]


class BPRLoss:
    def __init__(self, model, behavior=None):
        self.model = model
        self.behavior = behavior

        if behavior is not None:
            learning_rate = get_learning_rate_for_behavior(behavior)
        else:
            learning_rate = args.lr[0] if isinstance(args.lr, list) else args.lr

        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=args.decay)

    def step(self, users, pos, neg):
        bpr_loss, reg_loss = self.model.bpr_loss(users, pos, neg)
        total_loss = bpr_loss + reg_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item()

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']


class Timer:
    def __init__(self):
        self.start_time = None
        self.end_time = None

    def start(self):
        self.start_time = time()

    def stop(self):
        self.end_time = time()
        return self.end_time - self.start_time

    def elapsed(self):
        return time() - self.start_time

    @staticmethod
    def format_time(seconds):
        if seconds < 60:
            return f"{seconds:.2f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = seconds % 60
            return f"{minutes}m{secs:.2f}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = seconds % 60
            return f"{hours}h{minutes}m{secs:.2f}s"
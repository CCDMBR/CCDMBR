import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import os


def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class AuxiliaryGuidedDenoisingNetwork(nn.Module):
    def __init__(self, in_dims, out_dims, emb_size, time_type="cat", norm=False, dropout=0.5):
        super(AuxiliaryGuidedDenoisingNetwork, self).__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        assert out_dims[0] == in_dims[-1], "In and out dimensions must equal to each other."
        self.time_type = time_type
        self.time_emb_dim = emb_size
        self.norm = norm

        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)

        if self.time_type == "cat":
            in_dims_temp = [self.in_dims[0] + self.time_emb_dim] + self.in_dims[1:]
        else:
            raise ValueError("Unimplemented timestep embedding type %s" % self.time_type)
        out_dims_temp = self.out_dims

        self.in_layers = nn.ModuleList([nn.Linear(d_in, d_out) \
                                        for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:])])
        self.out_layers = nn.ModuleList([nn.Linear(d_in, d_out) \
                                         for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:])])

        self.drop = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        for layer in self.in_layers:
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)
            layer.bias.data.normal_(0.0, 0.001)

        for layer in self.out_layers:
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)
            layer.bias.data.normal_(0.0, 0.001)

        size = self.emb_layer.weight.size()
        fan_out = size[0]
        fan_in = size[1]
        std = np.sqrt(2.0 / (fan_in + fan_out))
        self.emb_layer.weight.data.normal_(0.0, std)
        self.emb_layer.bias.data.normal_(0.0, 0.001)

    def forward(self, noisy_target_emb, auxiliary_condition, timesteps):
        time_emb = timestep_embedding(timesteps, self.time_emb_dim).to(noisy_target_emb.device)
        emb = self.emb_layer(time_emb)

        if self.norm:
            noisy_target_emb = F.normalize(noisy_target_emb)
        noisy_target_emb = self.drop(noisy_target_emb)

        all_emb = torch.cat([noisy_target_emb, emb, auxiliary_condition], dim=-1)

        for i, layer in enumerate(self.in_layers):
            all_emb = layer(all_emb)
            all_emb = torch.tanh(all_emb)

        for i, layer in enumerate(self.out_layers):
            all_emb = layer(all_emb)
            if i != len(self.out_layers) - 1:
                all_emb = torch.tanh(all_emb)

        return all_emb


class MBDM(nn.Module):

    def __init__(self, hidden_size, num_users, num_items, dropout, device,
                 behavior_files, data_name, embeddings_dir="./embeddings",
                 denoising_hidden_dims=None, denoising_time_emb_dim=10,
                 behavior_encoding_dim=16, denoising_norm=False):
        super(MBDM, self).__init__()

        self.hidden_size = hidden_size
        self.num_users = num_users
        self.num_items = num_items
        self.dropout = nn.Dropout(dropout)
        self.device = device
        self.behavior_files = behavior_files
        self.num_behaviors = len(behavior_files)
        self.behavior_encoding_dim = behavior_encoding_dim

        if denoising_hidden_dims is None:
            denoising_hidden_dims = [200, 600]

        enhanced_emb_dim = hidden_size + behavior_encoding_dim
        condition_dim = (self.num_behaviors - 1) * enhanced_emb_dim

        self.load_trainable_embeddings(embeddings_dir, data_name)

        self.init_behavior_encodings()

        out_dims = denoising_hidden_dims + [enhanced_emb_dim]
        in_dims = out_dims[::-1]
        in_dims[0] = enhanced_emb_dim + condition_dim

        self.auxiliary_guided_denoising_network = AuxiliaryGuidedDenoisingNetwork(
            in_dims=in_dims,
            out_dims=out_dims,
            emb_size=denoising_time_emb_dim,
            time_type="cat",
            norm=denoising_norm,
            dropout=dropout
        )

        self.embedding_projection = nn.Linear(enhanced_emb_dim, hidden_size)

        self.behavior_projection = nn.Linear(self.behavior_encoding_dim, self.hidden_size)

        print(f"✅ MBDM with Behavior Encoding initialized:")
        print(f"  - Behavior encoding dim: {behavior_encoding_dim}")
        print(f"  - Enhanced embedding dim: {enhanced_emb_dim}")
        print(f"  - Number of behaviors: {self.num_behaviors}")

    def load_trainable_embeddings(self, embeddings_dir, data_name):
        embedding_path = os.path.join(embeddings_dir, data_name)

        user_embeddings_data = []
        for behavior_file in self.behavior_files:
            behavior_name = behavior_file.replace('.txt', '')
            user_emb_path = os.path.join(embedding_path, f"{behavior_name}_user_embeddings.npy")

            if os.path.exists(user_emb_path):
                user_emb = np.load(user_emb_path)
                user_embeddings_data.append(user_emb)
                print(f"Loaded user embeddings for {behavior_name}: {user_emb.shape}")
            else:
                raise FileNotFoundError(f"User embedding file not found: {user_emb_path}")


        self.user_embeddings = nn.ParameterList([
            nn.Parameter(torch.FloatTensor(emb))
            for emb in user_embeddings_data
        ])

        target_behavior_name = self.behavior_files[-1].replace('.txt', '')
        item_emb_path = os.path.join(embedding_path, f"{target_behavior_name}_item_embeddings.npy")

        if os.path.exists(item_emb_path):
            item_emb = np.load(item_emb_path)
            self.item_embeddings = nn.Parameter(torch.FloatTensor(item_emb))
            print(f"✅ Item embeddings registered: {item_emb.shape}")
        else:
            raise FileNotFoundError(f"Item embedding file not found: {item_emb_path}")

    def init_behavior_encodings(self):
        behavior_encodings = []

        for i in range(self.num_behaviors):
            encoding = torch.zeros(self.behavior_encoding_dim)

            for dim in range(self.behavior_encoding_dim):
                if dim % 2 == 0:
                    freq = (i + 1) / (10000 ** (dim / self.behavior_encoding_dim))
                    encoding[dim] = math.sin(freq)
                else:
                    freq = (i + 1) / (10000 ** ((dim - 1) / self.behavior_encoding_dim))
                    encoding[dim] = math.cos(freq)

            behavior_encodings.append(encoding)

        self.behavior_encodings = nn.ParameterList([
            nn.Parameter(encoding) for encoding in behavior_encodings
        ])

        print(f"✅ Behavior encodings initialized with sinusoidal patterns:")
        for i, behavior_file in enumerate(self.behavior_files):
            behavior_name = behavior_file.replace('.txt', '')
            print(f"  - {behavior_name}: {self.behavior_encodings[i].shape}")

    def compute_behavior_weights(self):
        behavior_weights = []
        all_scores = []

        for i in range(self.num_behaviors):
            behavior_encoding = self.behavior_encodings[i]  # [behavior_encoding_dim]

            projected_behavior = self.behavior_projection(behavior_encoding)  # [hidden_size]

            item_scores = torch.matmul(projected_behavior, self.item_embeddings.t())  # [num_items]

            mean_score = torch.mean(item_scores)
            all_scores.append(mean_score)

        all_scores = torch.stack(all_scores)  # [num_behaviors]
        behavior_weights = F.softmax(all_scores, dim=0)  # [num_behaviors]

        return behavior_weights, all_scores

    def compute_contrastive_loss(self, behavior_weights):
        contrastive_loss = 0.0
        loss_components = {}

        target_weight = behavior_weights[-1]

        target_vs_auxiliary_losses = []
        for i in range(self.num_behaviors - 1):
            auxiliary_weight = behavior_weights[i]
            # log σ(Q(target) - Q(auxiliary))
            diff = target_weight - auxiliary_weight
            loss = -torch.log(torch.sigmoid(diff) + 1e-8)
            target_vs_auxiliary_losses.append(loss)

        target_vs_auxiliary_loss = torch.mean(torch.stack(target_vs_auxiliary_losses))
        loss_components['target_vs_auxiliary'] = target_vs_auxiliary_loss
        contrastive_loss += target_vs_auxiliary_loss

        hierarchical_losses = []
        for i in range(1, self.num_behaviors - 1):
            higher_weight = behavior_weights[i]
            lower_weight = behavior_weights[i - 1]

            # log σ(Q(higher) - Q(lower))
            diff = higher_weight - lower_weight
            loss = -torch.log(torch.sigmoid(diff) + 1e-8)
            hierarchical_losses.append(loss)

            higher_name = self.behavior_files[i].replace('.txt', '')
            lower_name = self.behavior_files[i - 1].replace('.txt', '')
            loss_components[f'{higher_name}_vs_{lower_name}'] = loss

        if hierarchical_losses:
            hierarchical_loss = torch.mean(torch.stack(hierarchical_losses))
            loss_components['hierarchical'] = hierarchical_loss
            contrastive_loss += hierarchical_loss

        return contrastive_loss, loss_components

    def get_enhanced_user_embeddings(self, behavior_idx, user_ids):
        user_emb = self.user_embeddings[behavior_idx][user_ids]  # [batch_size, hidden_size]

        behavior_encoding = self.behavior_encodings[behavior_idx]  # [behavior_encoding_dim]

        batch_size = user_emb.shape[0]
        behavior_encoding_expanded = behavior_encoding.unsqueeze(0).expand(
            batch_size, -1
        )  # [batch_size, behavior_encoding_dim]

        enhanced_emb = torch.cat([user_emb, behavior_encoding_expanded], dim=-1)
        # [batch_size, hidden_size + behavior_encoding_dim]

        return enhanced_emb

    def get_auxiliary_behaviors(self, user_ids):
        auxiliary_embeddings = []

        for i in range(self.num_behaviors - 1):
            enhanced_emb = self.get_enhanced_user_embeddings(i, user_ids)
            auxiliary_embeddings.append(enhanced_emb)

        if len(auxiliary_embeddings) == 0:
            enhanced_emb_dim = self.hidden_size + self.behavior_encoding_dim
            con_emb = torch.zeros(user_ids.shape[0], 0, device=user_ids.device)
        else:
            con_emb = torch.cat(auxiliary_embeddings, dim=-1)

        return con_emb

    def get_target_user_embeddings(self, user_ids):
        target_behavior_idx = self.num_behaviors - 1
        enhanced_emb = self.get_enhanced_user_embeddings(target_behavior_idx, user_ids)
        return enhanced_emb

    def forward(self, x_t, con_emb, timesteps):
        denoised_enhanced_emb = self.auxiliary_guided_denoising_network(x_t, con_emb, timesteps)

        denoised_emb = self.embedding_projection(denoised_enhanced_emb)

        return denoised_enhanced_emb

    def get_item_embeddings(self):
        return self.item_embeddings

    def compute_interaction_scores_inner_product(self, user_embs, item_embs):
        scores = torch.matmul(user_embs, item_embs.t())  # [batch_size, num_items]
        return scores

    def predict(self, user_ids, diffusion_model, steps=5, sampling_noise=False):
        auxiliary_condition = self.get_auxiliary_behaviors(user_ids)

        enhanced_target_user_embs = self.get_target_user_embeddings(user_ids)

        if steps is None:
            steps = diffusion_model.steps

        enhanced_denoised_embs = diffusion_model.auxiliary_guided_p_sample(
            model=self,
            x_start=enhanced_target_user_embs,
            auxiliary_condition=auxiliary_condition,
            steps=steps,
            sampling_noise=sampling_noise
        )

        final_user_embs = self.embedding_projection(enhanced_denoised_embs)

        item_embs = self.get_item_embeddings()
        scores = self.compute_interaction_scores_inner_product(final_user_embs, item_embs)

        return scores
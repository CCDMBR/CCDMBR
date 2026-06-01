import os
import torch
import time as Time
from time import time
import logging
import numpy as np
import torch.nn.functional as F
from utils import parse_args, setup_seed, MultiBehaviorDataset, evaluate
from diffusion import AuxiliaryGuidedDiffusion, ModelMeanType
from model import MBDM
from config import args, device, print_config, EMBEDDING_PATH, get_learning_rate_for_behavior
from dataloader import load_dataset, get_available_behaviors
from lightgcn import create_lightgcn_model
from utils import set_seed, UniformSample, minibatch, shuffle_data, BPRLoss, Timer

logging.getLogger().setLevel(logging.INFO)


def train_single_behavior(behavior, dataset_name, data_path):
    print(f"TRAINING BEHAVIOR: {behavior.upper()}")

    dataset = load_dataset(dataset_name, data_path, behavior)
    if dataset.trainDataSize == 0:
        print(f"Warning: No training data found for behavior '{behavior}'. Skipping...")
        return None

    model = create_lightgcn_model(dataset)
    bpr_loss = BPRLoss(model, behavior=behavior)

    print(f"Users: {dataset.n_users}, Items: {dataset.m_items}, Interactions: {dataset.trainDataSize}")
    print(f"Learning rate: {get_learning_rate_for_behavior(behavior)}")

    save_dir = os.path.join(EMBEDDING_PATH, dataset_name)
    os.makedirs(save_dir, exist_ok=True)

    total_timer = Timer()
    total_timer.start()
    best_loss = float('inf')
    training_losses = []

    try:
        for epoch in range(args.epochs):
            epoch_timer = Timer()
            epoch_timer.start()
            model.train()

            sample_start = time()
            S = UniformSample(dataset)
            sample_time = time() - sample_start

            if len(S) == 0:
                print(f"Warning: No samples generated for epoch {epoch + 1}. Skipping...")
                continue

            users = torch.LongTensor(S[:, 0]).to(device)
            pos_items = torch.LongTensor(S[:, 1]).to(device)
            neg_items = torch.LongTensor(S[:, 2]).to(device)

            users, pos_items, neg_items = shuffle_data(users, pos_items, neg_items)

            total_loss = 0.0
            total_batches = 0

            for batch_users, batch_pos, batch_neg in minibatch(users, pos_items, neg_items):
                loss = bpr_loss.step(batch_users, batch_pos, batch_neg)
                total_loss += loss
                total_batches += 1

            avg_loss = total_loss / max(total_batches, 1)
            training_losses.append(avg_loss)

            if avg_loss < best_loss:
                best_loss = avg_loss

            epoch_time = epoch_timer.stop()

            print(f"Epoch {epoch + 1:4d}: Loss={avg_loss:.6f}, Time={Timer.format_time(epoch_time)}")

    except KeyboardInterrupt:
        print(f"Training interrupted by user at epoch {epoch + 1}")
    except Exception as e:
        print(f"Training error: {e}")
        return None

    finally:
        total_time = total_timer.stop()
        final_epoch = epoch + 1 if 'epoch' in locals() else 0
        final_loss = training_losses[-1] if training_losses else None

        print("=" * 60)
        print(f"TRAINING COMPLETED FOR BEHAVIOR: {behavior}")

        model.eval()
        with torch.no_grad():
            all_users, all_items = model.computer()
            user_embeddings = all_users.cpu().detach().numpy()
            item_embeddings = all_items.cpu().detach().numpy()

            user_file = os.path.join(save_dir, f"{behavior}_user_embeddings.npy")
            item_file = os.path.join(save_dir, f"{behavior}_item_embeddings.npy")

            np.save(user_file, user_embeddings)
            np.save(item_file, item_embeddings)

            print(f"Final embeddings saved:")
            print(f"  Users: {user_file}")
            print(f"  Items: {item_file}")

        return {
            'behavior': behavior,
            'learning_rate': get_learning_rate_for_behavior(behavior),
            'final_epoch': final_epoch,
            'total_time': total_time,
            'final_loss': final_loss,
            'best_loss': best_loss,
            'save_dir': save_dir,
            'user_file': f"{behavior}_user_embeddings.npy",
            'item_file': f"{behavior}_item_embeddings.npy"
        }


def compute_enhanced_bpr_loss(model, user_ids, pos_items, neg_items, hard_neg_items,
                              hard_neg_user_indices, lambda1):
    batch_size = user_ids.size(0)
    num_negs = neg_items.size(0) // batch_size
    device = user_ids.device

    target_user_embs = model.get_target_user_embeddings(user_ids)
    target_user_embs_proj = model.embedding_projection(target_user_embs)
    item_embs = model.get_item_embeddings()

    pos_item_embs = item_embs[pos_items]
    pos_scores = torch.sum(target_user_embs_proj * pos_item_embs, dim=-1)

    neg_user_embs = target_user_embs_proj.repeat_interleave(num_negs, dim=0)
    neg_item_embs = item_embs[neg_items]
    neg_scores = torch.sum(neg_user_embs * neg_item_embs, dim=-1)
    neg_scores = neg_scores.view(batch_size, num_negs)

    hard_neg_scores_per_user = []
    if len(hard_neg_items) > 0:
        hard_neg_user_embs = target_user_embs_proj[hard_neg_user_indices]
        hard_neg_item_embs = item_embs[hard_neg_items]
        hard_neg_scores = torch.sum(hard_neg_user_embs * hard_neg_item_embs, dim=-1)

        start_idx = 0
        for i in range(batch_size):
            user_hard_neg_count = (hard_neg_user_indices == i).sum().item()
            if user_hard_neg_count > 0:
                user_hard_scores = hard_neg_scores[start_idx:start_idx + user_hard_neg_count]
                hard_neg_scores_per_user.append(user_hard_scores)
                start_idx += user_hard_neg_count
            else:
                hard_neg_scores_per_user.append(torch.tensor([], device=device))
    else:
        hard_neg_scores_per_user = [torch.tensor([], device=device) for _ in range(batch_size)]

    total_loss = 0.0
    valid_pairs = 0

    for i in range(batch_size):
        user_pos_score = pos_scores[i]
        user_neg_scores = neg_scores[i]
        user_hard_neg_scores = hard_neg_scores_per_user[i]

        for j in range(num_negs):
            regular_neg_score = user_neg_scores[j]

            if len(user_hard_neg_scores) > 0:
                hard_neg_sum = torch.sum(user_hard_neg_scores)
            else:
                hard_neg_sum = torch.tensor(0.0, device=device)

            score_diff = user_pos_score - regular_neg_score - lambda1 * hard_neg_sum
            loss_term = -torch.log(torch.sigmoid(score_diff) + 1e-10)

            total_loss += loss_term
            valid_pairs += 1

    enhanced_bpr_loss = total_loss / max(valid_pairs, 1)
    return enhanced_bpr_loss


def main1():
    print_config()
    set_seed(args.seed)

    try:
        available_behaviors = get_available_behaviors(args.dataset, args.data_path)
    except Exception as e:
        print(f"Error checking available behaviors: {e}")
        available_behaviors = args.behaviors

    behaviors_to_train = []
    for behavior in args.behaviors:
        behavior_file = os.path.join(args.data_path, args.dataset, f"{behavior}.txt")
        if os.path.exists(behavior_file):
            behaviors_to_train.append(behavior)
        else:
            print(f"Warning: Behavior file {behavior_file} not found. Skipping behavior '{behavior}'.")

    if not behaviors_to_train:
        print("Error: No valid behavior files found. Exiting.")
        return

    training_results = []
    total_start_time = time()

    for i, behavior in enumerate(behaviors_to_train):
        print(f"TRAINING PROGRESS: {i + 1}/{len(behaviors_to_train)} - Behavior: {behavior}")
        result = train_single_behavior(behavior, args.dataset, args.data_path)
        if result:
            training_results.append(result)

    total_time = time() - total_start_time
    print("=" * 60)
    print("ALL BEHAVIORS TRAINING COMPLETED")
    print("=" * 60)
    print(f"Total training time: {Timer.format_time(total_time)}")

    if training_results:
        print("Training Summary:")
        for result in training_results:
            print(f"  {result['behavior']}: {result['user_file']}, {result['item_file']}")


def main2():
    print("=" * 60)
    print("Stage 2: Diffusion Training with Behavior Encoding Enhancement")
    print("=" * 60)

    args_diffusion = parse_args()
    setup_seed(args_diffusion.random_seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_directory = os.path.join(args.data_path, args.dataset)
    dataset = MultiBehaviorDataset(data_directory, device)
    topk = [10, 20]

    model = MBDM(
        hidden_size=args.embedding_dim,
        num_users=dataset.num_users,
        num_items=dataset.num_items,
        dropout=0.2,
        device=device,
        behavior_files=args.behaviors,
        data_name=args.dataset,
        embeddings_dir=EMBEDDING_PATH,
        denoising_hidden_dims=eval(args.denoising_dims),
        denoising_time_emb_dim=args.time_emb_dim,
        behavior_encoding_dim=16
    )
    model.to(device)

    mean_type = ModelMeanType.START_X if args.mean_type == 'x0' else ModelMeanType.EPSILON
    diffusion_model = AuxiliaryGuidedDiffusion(
        mean_type=mean_type,
        noise_schedule=args.noise_schedule,
        noise_scale=1.0,
        noise_min=args.beta_start,
        noise_max=args.beta_end,
        steps=args.timesteps,
        device=device,
        history_num_per_term=10,
        beta_fixed=True
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.diff_lr, eps=1e-8, weight_decay=1e-4)

    contrastive_weight = 0.01

    hr_max = 0
    best_epoch = 0
    total_samples = len(dataset.train_user_dict)
    num_batches = total_samples // args.diff_batch_size + 1

    print(f"🚀 starting diffusion training with behavior encoding enhancement...")
    print(f"📊 contrastive loss weight: {contrastive_weight}")

    for epoch in range(args.diff_epochs):
        start_time = Time.time()
        model.train()

        total_loss = 0.0
        total_enhanced_bpr_loss = 0.0
        total_diffusion_loss = 0.0
        total_contrastive_loss = 0.0

        for j in range(num_batches):
            user_ids, pos_items = dataset.get_training_batch(args.diff_batch_size)
            if len(user_ids) == 0:
                continue

            optimizer.zero_grad()

            behavior_weights, behavior_scores = model.compute_behavior_weights()
            contrastive_loss, loss_components = model.compute_contrastive_loss(behavior_weights)

            target_user_embs = model.get_target_user_embeddings(user_ids)
            auxiliary_condition = model.get_auxiliary_behaviors(user_ids)

            diffusion_losses = diffusion_model.auxiliary_guided_training_losses(
                model=model,
                target_user_embs=target_user_embs,
                auxiliary_condition=auxiliary_condition,
                reweight=args.reweight
            )
            diffusion_loss = diffusion_losses["loss"].mean()

            num_neg = 5
            neg_items = []
            batch_size = len(user_ids)
            for i in range(batch_size):
                user_pos_items = dataset.train_user_dict.get(user_ids[i].item(), [])
                user_neg_items = []
                while len(user_neg_items) < num_neg:
                    neg_item = np.random.randint(0, dataset.num_items)
                    if neg_item not in user_pos_items and neg_item not in user_neg_items:
                        user_neg_items.append(neg_item)
                neg_items.extend(user_neg_items)

            neg_items = torch.LongTensor(neg_items).to(device)
            hard_neg_items, hard_neg_user_indices, _ = dataset.get_all_hard_negatives_for_batch(user_ids)

            enhanced_bpr_loss = compute_enhanced_bpr_loss(
                model, user_ids, pos_items, neg_items, hard_neg_items,
                hard_neg_user_indices, args.lambda1
            )

            loss = (enhanced_bpr_loss +
                    args.lambda2 * diffusion_loss +
                    contrastive_weight * contrastive_loss)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_enhanced_bpr_loss += enhanced_bpr_loss.item()
            total_diffusion_loss += diffusion_loss.item()
            total_contrastive_loss += contrastive_loss.item()

        avg_loss = total_loss / num_batches
        avg_enhanced_bpr = total_enhanced_bpr_loss / num_batches
        avg_diffusion = total_diffusion_loss / num_batches
        avg_contrastive = total_contrastive_loss / num_batches

        print("Epoch {:03d}; ".format(epoch) +
              'Total loss: {:.4f} '.format(avg_loss) +
              '(BPR: {:.4f}, '.format(avg_enhanced_bpr) +
              'Diffusion: {:.4f}, '.format(avg_diffusion) +
              'Contrastive: {:.4f}); '.format(avg_contrastive) +
              "Time: " + Time.strftime("%H:%M:%S", Time.gmtime(Time.time() - start_time)))

        if epoch % 10 == 0:
            with torch.no_grad():
                display_weights, display_scores = model.compute_behavior_weights()
                print("📊 Behavior Weights:")
                for i, behavior_file in enumerate(model.behavior_files):
                    behavior_name = behavior_file.replace('.txt', '')
                    weight = display_weights[i].item()
                    score = display_scores[i].item()
                    print(f"  - {behavior_name}: Q={weight:.4f} (score={score:.4f})")

        if (epoch + 1) % 1 == 0:
            eval_start = Time.time()
            print('-------------------------- VAL PHRASE --------------------------')
            val_hr = evaluate(model, diffusion_model, dataset, topk, device, mode='validation')
            print('-------------------------- TEST PHRASE -------------------------')
            test_hr = evaluate(model, diffusion_model, dataset, topk, device, mode='test')
            print("Evaluation cost: " + Time.strftime("%H:%M:%S", Time.gmtime(Time.time() - eval_start)))
            print('----------------------------------------------------------------')

            if val_hr > hr_max:
                hr_max = val_hr
                best_epoch = epoch

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'hr_max': hr_max,
                }, f'best_model_behavior_encoding_{args.dataset}.pth')

    print(f"🎉 Best HR@{topk[1]}: {hr_max:.6f} at epoch {best_epoch}")

if __name__ == '__main__':
    # main1()  # Stage 1: Pre-training behavior embeddings
    main2()  # Stage 2: Diffusion training with behavior encoding enhancement
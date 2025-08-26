import os
import numpy as np
import torch
import optuna, pickle

from tianshou.policy import DQNPolicy
from tianshou.data import Collector, ReplayBuffer
from tianshou.trainer import OffpolicyTrainer
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger

from algos.nets import ConcatActionValue, precompute_bert_embeddings, extract_bert_layer_embeddings
from algos.common import EpsilonDecayHook, SaveHook, TestFnHook
from envs.shapes.multitask_shapes import MultitaskShapes, ShapesAttrCombFactory
from utils import setup_artefact_paths, setup_experiment, setup_study, sample_hyperparams
from yaml_utils import load_yaml, save_yaml


def experiment(trial: optuna.trial.Trial, store_path: str, config_path: str) -> float:
    _, store_path, precomp_path = setup_experiment(store_path=store_path, config_path=config_path)

    with open(config_path, 'r') as file:
        hparams = load_yaml(file)

    exp_hparams = hparams["experiment"]
    if trial is not None:
        exp_hparams, only_sampled_hparams = sample_hyperparams(trial=trial, hparams=exp_hparams)

    env_hparams = hparams["environment"] if "environment" in hparams else hparams

    sampled_config = {
        "general": hparams["general"],
        "experiment": exp_hparams,
        "environment": env_hparams
    }
    sampled_config_pth = os.path.join(store_path, os.path.basename(config_path))
    save_yaml(data=sampled_config, path=sampled_config_pth)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    writer = SummaryWriter(store_path)
    logger = TensorboardLogger(writer)

    env_factory = ShapesAttrCombFactory(
        hparams=env_hparams,
        store_path=store_path
    )

    train_env: MultitaskShapes = env_factory.get_env(set_id='TRAIN')
    test_env: MultitaskShapes = env_factory.get_env(set_id='TRAIN', purpose='EVAL')

    all_instructions = env_factory.get_all_instructions()
    embedding_path = os.path.join(precomp_path, 'bert_embeddings.pt')
    if os.path.isfile(embedding_path):
        precomp_embeddings = torch.load(embedding_path, map_location=device)
    else:
        precomp_embeddings = precompute_bert_embeddings(all_instructions, device=device)
    torch.save(precomp_embeddings, embedding_path)

    if "bert_layer_index" in exp_hparams:
        bert_layer_ind = exp_hparams["bert_layer_index"]
    else:
        bert_layer_ind = -1
    layer_embeddings = extract_bert_layer_embeddings(embedding_dict=precomp_embeddings, layer_ind=bert_layer_ind)

    feat_dim = train_env.observation_space["features"].shape[0]
    emb_dim = layer_embeddings[next(iter(layer_embeddings))].shape[0]
    in_dim = feat_dim + emb_dim

    nnet = ConcatActionValue(
        in_dim=in_dim,
        num_actions=int(train_env.action_space.n),
        h=exp_hparams["hidden_dim"],
        precom_embeddings=layer_embeddings,
        device=device
    )
    optim = torch.optim.Adam(nnet.parameters(), lr=exp_hparams["lr"])
    rb = ReplayBuffer(size=exp_hparams['buffer_size'])

    agent = DQNPolicy(
        model=nnet,
        optim=optim,
        is_double=False,
        action_space=train_env.action_space,
        discount_factor=env_hparams["disc_fact"],
        target_update_freq=exp_hparams["target_update_steps"]
    )

    train_collector = Collector(agent, train_env, rb, exploration_noise=True)
    test_collector = Collector(agent, test_env, exploration_noise=True)

    test_rewards_history = []

    n_epochs = exp_hparams["n_epochs"]
    n_steps = exp_hparams["epoch_steps"]

    epoch_hook = EpsilonDecayHook(hparams=exp_hparams, max_steps=n_epochs * n_steps, agent=agent, logger=logger)
    save_hook = SaveHook(save_path=f'{store_path}/best_model.pth')
    test_fn = TestFnHook(
        agent=agent, 
        logger=logger, 
        epsilon=exp_hparams["test_epsilon"],
        episodes_per_test=exp_hparams["episode_per_test"], 
        collector=test_collector, 
        history=test_rewards_history
    )

    _ = OffpolicyTrainer(
        policy=agent,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=n_epochs,
        step_per_epoch=n_steps,
        step_per_collect=exp_hparams["step_per_collect"],
        update_per_step=exp_hparams["update_per_step"],
        episode_per_test=exp_hparams["episode_per_test"],
        batch_size=exp_hparams["batch_size"],
        train_fn=epoch_hook.hook,
        test_fn=test_fn.hook,
        save_best_fn=save_hook.hook,
        logger=logger
    ).run()

    torch.save(agent.state_dict(), f'{store_path}/last_model.pth')
    
    if len(test_rewards_history) < 10:
        best_10_consec_avg = np.mean(test_rewards_history)
    else:
        cumsum = np.cumsum([0] + test_rewards_history)
        best_10_consec_avg = max((cumsum[i + 10] - cumsum[i]) / 10 for i in range(len(test_rewards_history) - 9))

    if trial is not None:
        writer.add_hparams(
            hparam_dict=only_sampled_hparams,
            metric_dict={
                "train/best_result": best_10_consec_avg
            }
        )

    return best_10_consec_avg


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Run DQN Multitask Shapes experiment.")
    parser.add_argument("--config_name", type=str, default=None, help="Experiment name which needs to match config file name.")
    args = parser.parse_args()

    script_path = os.path.abspath(__file__)
    store_path, config_path = setup_artefact_paths(script_path=script_path, config_name=args.config_name)

    with open(config_path, 'r') as file:
        hparams = load_yaml(file)
    exp_hparams = hparams["experiment"]
    n_trials = exp_hparams["n_trials"]

    # Automatically determine if we are running a single experiment or multiple
    # If multiple, Optuna will be used.
    single_experiment = True
    special_keys = {"hidden_dim", "float_keys", "log_domain_keys"}
    for key, val in exp_hparams.items():
        if isinstance(val, list) and key not in special_keys:
            single_experiment = False

    if not single_experiment:
        import functools
        print(f'Optuna: Running {exp_hparams["n_trials"]} experiments to determine the best set of hyper-parameters.')
        _, store_path = setup_study(store_path=store_path, config_path=config_path)
        objective = functools.partial(experiment, store_path=store_path, config_path=config_path)
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        print("Best trial:", study.best_trial.number, study.best_params)

        if args.config_name is None:
            config_name = os.path.basename(config_path).split(".")[0]
        else:
            config_name = args.config_name
        optuna_path = os.path.join(store_path, f"{config_name}.pkl")
        with open(optuna_path, "wb") as f:
            pickle.dump(study, f)
    else:
        print(f'Running a single experiment.')
        experiment(trial=None, store_path=store_path, config_path=config_path)

import logging
import os

import faiss
import numpy as np
import torch
import transformers
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer, SentencesDataset
from sentence_transformers.util import batch_to_device
from sklearn.metrics.pairwise import paired_cosine_distances
from torch import nn
from tqdm import tqdm_notebook as tqdm
from tqdm import trange


class ModelExampleBasedShuffler:
    def __init__(self, group_size=10, allow_same=False):
        self.group_size = group_size
        self.sim_batch_size = 1000
        self.allow_same = allow_same

    def shuffle(self, texts_embs, texts, top_k=200, texts_embs_fixed=None):
        used_ids = set()
        shuffled_ids = []

        order = np.random.permutation(len(texts_embs))
        texts_embs = np.array(texts_embs)[order]
        texts = np.array(texts)[order]
        index = faiss.IndexFlatL2(texts_embs.shape[1])
        index.add(texts_embs)

        if texts_embs_fixed is not None:
            index_fixed = faiss.IndexFlatL2(texts_embs_fixed.shape[1])
            index_fixed.add(texts_embs_fixed)

        for i in tqdm(range(len(texts_embs))):
            if i % self.sim_batch_size == 0:
                if texts_embs_fixed is None:
                    _, sim_batch = index.search(texts_embs[i : i + self.sim_batch_size], top_k)
                else:
                    _, sim_batch_f = index.search(
                        texts_embs[i : i + self.sim_batch_size], int(top_k / 2)
                    )
                    _, sim_batch_fixed = index_fixed.search(
                        texts_embs_fixed[i : i + self.sim_batch_size], int(top_k / 2)
                    )
                    sim_batch = []
                    for k in range(len(sim_batch_f)):
                        sim_batch.append(sim_batch_f[k])
                        sim_batch.append(sim_batch_fixed[k])

            if i not in used_ids:
                used_ids.add(i)
                shuffled_ids.append(i)
                batch = set()
                batch.add(texts[i])
                for j in sim_batch[i % self.sim_batch_size]:
                    if j not in used_ids and (self.allow_same or texts[j] not in batch):
                        shuffled_ids.append(j)
                        used_ids.add(j)
                        batch.add(texts[j])
                        if len(batch) >= self.group_size:
                            break

        return [order[i] for i in shuffled_ids[::-1]]


class ShuffledSentencesDataset(SentencesDataset):
    def __init__(
        self,
        examples,
        model,
        ref_alpha=None,
        ref_thr=None,
        start_ref_epoch=2,
        use_fixed_model=False,
    ):
        super().__init__(examples, model)
        self.ref_alpha = ref_alpha
        self.ref_thr = ref_thr
        self.start_ref_epoch = start_ref_epoch
        self.fixed_model = None
        if use_fixed_model:
            self.fixed_model = SentenceTransformer("stsb-bert-base")

    def shuffle(self, shuffler=None, model=None, col=0, epoch=0):
        if shuffler is None:
            return
        texts = [example.texts[col] for example in self.examples]
        if model is not None:
            texts_embs = model.encode(texts)
        else:
            texts_embs = encode_text(texts, set(stopwords.words("english")))
        if self.fixed_model is not None:
            texts_embs_fixed = self.fixed_model.encode(texts)
        else:
            texts_embs_fixed = None
        self.examples = [
            self.examples[ind]
            for ind in shuffler.shuffle(texts_embs, texts, texts_embs_fixed=texts_embs_fixed)
        ]
        if self.ref_alpha is not None and model is not None and epoch > self.start_ref_epoch:
            labels = [example.label for example in self.examples]
            texts_embs_right = model.encode([example.texts[1] for example in self.examples])
            preds = 1 - paired_cosine_distances(texts_embs, texts_embs_right)
            labels_ref = np.array(labels) * self.ref_alpha + preds * (1 - self.ref_alpha)
            if self.ref_thr is not None:
                labels_ref = (labels_ref > self.ref_thr).astype(float)
            for i in range(len(self.examples)):
                self.examples[i].label = labels_ref[i]


class ShuffledSentenceTransformer(SentenceTransformer):
    def __init__(self, model_name_or_path=None, modules=None, device=None):
        super().__init__(model_name_or_path, modules, device)
        if not hasattr(self._first_module(), "alpha"):
            self._first_module().alpha = nn.Parameter(torch.Tensor([1]))

    # def _eval_during_training(self, evaluator, output_path, save_best_model, epoch, steps, callback):
    #    model._first_module().tau = self.model.tau
    #    super()._eval_during_training(evaluator, output_path, save_best_model, epoch, steps, callback)

    def fit(
        self,
        train_objectives,
        evaluator=None,
        epochs=1,
        steps_per_epoch=None,
        scheduler="WarmupLinear",
        warmup_steps=10000,
        optimizer_class=transformers.AdamW,
        optimizer_params={"alpha_lr": 1e-3, "lr": 2e-5, "eps": 1e-6, "correct_bias": False},
        weight_decay=0.01,
        evaluation_steps=0,
        output_path=None,
        save_best_model=True,
        max_grad_norm=1,
        use_amp=False,
        callback=None,
        output_path_ignore_not_empty=False,
        shuffler=None,
        shuffle_idxs=[],
        label_refurb=False,
    ):
        """
        Train the model with the given training objective
        Each training objective is sampled in turn for one batch.
        We sample only as many batches from each objective as there are in the smallest one
        to make sure of equal training with each dataset.
        :param train_objectives: Tuples of (DataLoader, LossFunction). Pass more than one for multi-task learning
        :param evaluator: An evaluator (sentence_transformers.evaluation) evaluates the model performance during training on held-out dev data. It is used to determine the best model that is saved to disc.
        :param epochs: Number of epochs for training
        :param steps_per_epoch: Number of training steps per epoch. If set to None (default), one epoch is equal the DataLoader size from train_objectives.
        :param scheduler: Learning rate scheduler. Available schedulers: constantlr, warmupconstant, warmuplinear, warmupcosine, warmupcosinewithhardrestarts
        :param warmup_steps: Behavior depends on the scheduler. For WarmupLinear (default), the learning rate is increased from o up to the maximal learning rate. After these many training steps, the learning rate is decreased linearly back to zero.
        :param optimizer_class: Optimizer
        :param optimizer_params: Optimizer parameters
        :param weight_decay: Weight decay for model parameters
        :param evaluation_steps: If > 0, evaluate the model using evaluator after each number of training steps
        :param output_path: Storage path for the model and evaluation files
        :param save_best_model: If true, the best model (according to evaluator) is stored at output_path
        :param max_grad_norm: Used for gradient normalization.
        :param use_amp: Use Automatic Mixed Precision (AMP). Only for Pytorch >= 1.6.0
        :param callback: Callback function that is invoked after each evaluation.
                It must accept the following three parameters in this order:
                `score`, `epoch`, `steps`
        :param output_path_ignore_not_empty: deprecated, no longer used
        :param shuffle_idxs: dataloader indices for FPS shuffling
        """

        if use_amp:
            from torch.cuda.amp import autocast

            scaler = torch.cuda.amp.GradScaler()

        self.to(self._target_device)

        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)

        dataloaders = [dataloader for dataloader, _ in train_objectives]

        # Use smart batching
        for dataloader in dataloaders:
            dataloader.collate_fn = self.smart_batching_collate

        loss_models = [loss for _, loss in train_objectives]
        for loss_model in loss_models:
            loss_model.to(self._target_device)

        self.best_score = -9999999

        if steps_per_epoch is None or steps_per_epoch == 0:
            steps_per_epoch = min([len(dataloader) for dataloader in dataloaders])

        num_train_steps = int(steps_per_epoch * epochs)

        # Prepare optimizers
        optimizers = []
        schedulers = []
        for loss_model in loss_models:
            param_optimizer = [
                (n, p) for n, p in list(loss_model.named_parameters()) if "alpha" not in n
            ]
            alpha_param = [(n, p) for n, p in list(loss_model.named_parameters()) if "alpha" in n]

            no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight", "alpha"]
            lr = optimizer_params.pop("lr")
            alpha_lr = optimizer_params.pop("alpha_lr")

            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": weight_decay,
                    "lr": lr,
                },
                {
                    "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                    "lr": lr,
                },
                {"params": [p for n, p in alpha_param], "weight_decay": 0.0, "lr": alpha_lr},
            ]

            optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)
            scheduler_obj = self._get_scheduler(
                optimizer, scheduler=scheduler, warmup_steps=warmup_steps, t_total=num_train_steps
            )

            optimizers.append(optimizer)
            schedulers.append(scheduler_obj)

        global_step = 0

        """
        if train_idx in shuffle_idxs:
            logging.info('shuffling')
            dataset = dataloaders[train_idx].dataset
            dataset.shuffle(shuffler, self)
            dataloaders[train_idx].dataset = dataset
        """

        data_iterators = [iter(dataloader) for dataloader in dataloaders]

        num_train_objectives = len(train_objectives)

        skip_scheduler = False
        for epoch in trange(epochs, desc="Epoch"):
            training_steps = 0

            for loss_model in loss_models:
                loss_model.zero_grad()
                loss_model.train()

            for _ in tqdm(range(steps_per_epoch), desc="Iteration", smoothing=0.05):
                for train_idx in range(num_train_objectives):
                    loss_model = loss_models[train_idx]
                    optimizer = optimizers[train_idx]
                    scheduler = schedulers[train_idx]
                    data_iterator = data_iterators[train_idx]

                    try:
                        data = next(data_iterator)
                    except StopIteration:
                        # logging.info("Restart data_iterator")
                        if train_idx in shuffle_idxs:
                            logging.info("shuffling")
                            dataset = dataloaders[train_idx].dataset
                            dataset.shuffle(shuffler, self, epoch=epoch)
                            dataloaders[train_idx].dataset = dataset
                            # for loss_model in loss_models:
                            # loss_model.zero_grad()
                            # loss_model.train()
                        data_iterator = iter(dataloaders[train_idx])
                        data_iterators[train_idx] = data_iterator
                        data = next(data_iterator)

                    features, labels = batch_to_device(data, self._target_device)

                    if use_amp:
                        with autocast():
                            loss_value = loss_model(features, labels)

                        scale_before_step = scaler.get_scale()
                        scaler.scale(loss_value).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(loss_model.parameters(), max_grad_norm)
                        scaler.step(optimizer)
                        scaler.update()

                        skip_scheduler = scaler.get_scale() != scale_before_step
                    else:
                        loss_value = loss_model(features, labels)
                        loss_value.backward()
                        torch.nn.utils.clip_grad_norm_(loss_model.parameters(), max_grad_norm)
                        optimizer.step()

                    optimizer.zero_grad()

                    if not skip_scheduler:
                        scheduler.step()

                training_steps += 1
                global_step += 1

                if evaluation_steps > 0 and training_steps % evaluation_steps == 0:
                    self._eval_during_training(
                        evaluator, output_path, save_best_model, epoch, training_steps
                    )
                    for loss_model in loss_models:
                        loss_model.zero_grad()
                        loss_model.train()

            self._eval_during_training(
                evaluator, output_path, save_best_model, epoch, training_steps
            )

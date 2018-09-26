import argparse
import logging
import os

import math
from tensorboardX import SummaryWriter

import torch
from torch import nn, optim
from torch.nn.utils import clip_grad_norm_
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from snli.models import SNLIModel
from snli.utilss.dataset import SNLIDataset
from utils.vocab import Vocab
from utils.glove import load_glove

# logging.basicConfig(level=logging.INFO,
#                     format='%(asctime)s %(levelname)-8s %(message)s')


logger = logging.getLogger("general_logger")
from functools import partial


def train(args):
    word_vocab = Vocab.from_file(path=args.vocab, add_pad=True, add_unk=True)
    label_dict = {'neutral': 0, 'entailment': 1, 'contradiction': 2}
    label_vocab = Vocab(vocab_dict=label_dict, add_pad=False, add_unk=False)

    train_dataset = SNLIDataset(data_path=args.train_data, word_vocab=word_vocab, label_vocab=label_vocab, max_length=args.max_length, lower=args.lower)
    valid_dataset = SNLIDataset(data_path=args.valid_data, word_vocab=word_vocab, label_vocab=label_vocab, max_length=args.max_length, lower=args.lower)
    test_dataset = SNLIDataset(data_path=args.test_data, word_vocab=word_vocab, label_vocab=label_vocab, max_length=args.max_length, lower=args.lower)

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=2,
                              collate_fn=train_dataset.collate,
                              pin_memory=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=args.batch_size,
                              shuffle=False, num_workers=2,
                              collate_fn=valid_dataset.collate,
                              pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size,
                              shuffle=False, num_workers=2,
                              collate_fn=test_dataset.collate,
                              pin_memory=True)

    model = SNLIModel(num_classes=len(label_vocab), num_words=len(word_vocab),
                      word_dim=args.word_dim, hidden_dim=args.hidden_dim,
                      clf_hidden_dim=args.clf_hidden_dim,
                      clf_num_layers=args.clf_num_layers,
                      use_leaf_rnn=args.leaf_rnn,
                      use_batchnorm=args.batchnorm,
                      intra_attention=args.intra_attention,
                      dropout_prob=args.dropout,
                      bidirectional=args.bidirectional)
    if args.glove:
        logger.info('Loading GloVe pretrained vectors...')
        glove_weight = load_glove(
            path=args.glove, vocab=word_vocab,
            init_weight=model.word_embedding.weight.data.numpy())
        glove_weight[word_vocab.pad_id] = 0
        model.word_embedding.weight.data.set_(torch.FloatTensor(glove_weight))
    if args.fix_word_embedding:
        logger.info('Will not update word embeddings')
        model.word_embedding.weight.requires_grad = False
    model.cuda(args.device)
    logger.info(f'Using device {args.device}')
    if args.optimizer == 'adam':
        optimizer_class = optim.Adam
    elif args.optimizer == 'adagrad':
        optimizer_class = optim.Adagrad
    elif args.optimizer == 'adadelta':
        optimizer_class = optim.Adadelta
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optimizer_class(params=params, weight_decay=args.l2reg)
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, mode='max', factor=0.5, patience=10, verbose=True)


    def reduce_lr(self, epoch):
        lr_scheduler.ReduceLROnPlateau._reduce_lr(self, epoch)
        logger.info(f"learning rate is reduced by factor 0.5!")
    scheduler._reduce_lr = partial(reduce_lr, scheduler)


    patience = 10 * 4
    last_time = 0


    criterion = nn.CrossEntropyLoss()

    train_summary_writer = SummaryWriter(
        log_dir=os.path.join(args.tensor_board_path, 'log' + str(args.hash), 'train'))
    valid_summary_writer = SummaryWriter(
        log_dir=os.path.join(args.tensor_board_path, 'log' + str(args.hash), 'valid'))
    test_summary_writer = SummaryWriter(
        log_dir=os.path.join(args.tensor_board_path, 'log' + str(args.hash), 'test'))

    def run_iter(batch, is_training):
        model.train(is_training)
        pre = batch['pre'].cuda(args.device)
        hyp = batch['hyp'].cuda(args.device)
        pre_length = batch['pre_length'].cuda(args.device)
        hyp_length = batch['hyp_length'].cuda(args.device)
        label = batch['label'].cuda(args.device)
        logits = model(pre=pre, pre_length=pre_length,
                       hyp=hyp, hyp_length=hyp_length)
        label_pred = logits.max(1)[1]
        accuracy = torch.eq(label, label_pred).float().mean()
        loss = criterion(input=logits, target=label)
        if is_training:
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(parameters=params, max_norm=5)
            optimizer.step()
        return loss, accuracy

    def add_scalar_summary(summary_writer, name, value, step):
        if torch.is_tensor(value):
            value = value.item()
        summary_writer.add_scalar(tag=name, scalar_value=value,
                                  global_step=step)

    num_train_batches = len(train_loader)
    validate_every = num_train_batches // 10
    best_vaild_accuacy = 0
    iter_count = 0
    for epoch_num in range(args.max_epoch):
        logger.info(f'Epoch {epoch_num}: start')
        for batch_iter, train_batch in enumerate(train_loader):
            if iter_count % args.anneal_temperature_every == 0:
                rate = args.anneal_temperature_rate
                new_temperature = max([0.5, math.exp(-rate * iter_count)])
                model.encoder.gumbel_temperature = new_temperature
                logger.info(f'Iter #{iter_count}: '
                             f'Set Gumbel temperature to {new_temperature:.4f}')
            train_loss, train_accuracy = run_iter(
                batch=train_batch, is_training=True)
            iter_count += 1
            add_scalar_summary(
                summary_writer=train_summary_writer,
                name='loss', value=train_loss, step=iter_count)
            add_scalar_summary(
                summary_writer=train_summary_writer,
                name='accuracy', value=train_accuracy, step=iter_count)

            if (batch_iter + 1) % validate_every == 0:
                torch.set_grad_enabled(False)
                valid_loss_sum = valid_accuracy_sum = 0
                num_valid_batches = len(valid_loader)
                for valid_batch in valid_loader:
                    valid_loss, valid_accuracy = run_iter(
                        batch=valid_batch, is_training=False)
                    valid_loss_sum += valid_loss.item()
                    valid_accuracy_sum += valid_accuracy.item()
                torch.set_grad_enabled(True)
                valid_loss = valid_loss_sum / num_valid_batches
                valid_accuracy = valid_accuracy_sum / num_valid_batches
                scheduler.step(valid_accuracy)
                add_scalar_summary(
                    summary_writer=valid_summary_writer,
                    name='loss', value=valid_loss, step=iter_count)
                add_scalar_summary(
                    summary_writer=valid_summary_writer,
                    name='accuracy', value=valid_accuracy, step=iter_count)
                progress = epoch_num + batch_iter/num_train_batches
                logger.info(f'Epoch {progress:.2f}: '
                             f'valid loss = {valid_loss:.4f}, '
                             f'valid accuracy = {valid_accuracy:.4f}')
                if valid_accuracy > best_vaild_accuacy:
                    best_vaild_accuacy = valid_accuracy
                    model_filename = (f'model-{progress:.2f}'
                                      f'-{valid_loss:.4f}'
                                      f'-{valid_accuracy:.4f}.pkl')
                    model_path = os.path.join(args.save_dir, model_filename)
                    torch.save(model.state_dict(), model_path)
                    logger.info(f'Saved the new best model to {model_path}')
                    last_time = 0
                else:
                    last_time += 1
                    if last_time > patience:
                        logger.info("done")
                        return

                with torch.no_grad():
                    model.eval()
                    num_correct = 0
                    num_data = len(test_dataset)
                    for test_batch in test_loader:
                        pre = test_batch['pre'].cuda(args.device)
                        hyp = test_batch['hyp'].cuda(args.device)
                        pre_length = test_batch['pre_length'].cuda(args.device)
                        hyp_length = test_batch['hyp_length'].cuda(args.device)
                        label = test_batch['label'].cuda(args.device)
                        logits = model(pre=pre, pre_length=pre_length,
                                       hyp=hyp, hyp_length=hyp_length)
                        label_pred = logits.max(1)[1]
                        num_correct_batch = torch.eq(label, label_pred).long().sum()
                        num_correct_batch = num_correct_batch.item()
                        num_correct += num_correct_batch
                    add_scalar_summary(summary_writer=test_summary_writer, name='accuracy',
                                       value=num_correct / num_data,
                                       step=iter_count)
                    logger.info(f'Accuracy: {num_correct / num_data:.4f} # data: {num_data} # correct: {num_correct}')


def main():
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument('--train-data', default="data/snli_dataset/snli_1.0/snli_1.0_train.jsonl")
    parser.add_argument('--valid-data', default="data/snli_dataset/snli_1.0/snli_1.0_dev.jsonl")
    parser.add_argument('--test-data', default="data/snli_dataset/snli_1.0/snli_1.0_test.jsonl")
    parser.add_argument('--vocab', default="pretrained/snli_vocab.txt")
    parser.add_argument('--max-length', default=100, type=int)
    parser.add_argument('--lower', default="False", type=lambda val: True if val == "True" else False)

    parser.add_argument('--word-dim', default=300, type=int)
    parser.add_argument('--hidden-dim', default=300, type=int)
    parser.add_argument('--clf-hidden-dim', default=1024, type=int)
    parser.add_argument('--clf-num-layers', default=1, type=int)
    parser.add_argument('--leaf-rnn', default="False", type=lambda val: True if val == "True" else False)
    parser.add_argument('--bidirectional', default="False", type=lambda val: True if val == "True" else False)
    parser.add_argument('--intra-attention', default="False", type=lambda val: True if val == "True" else False)
    parser.add_argument('--batchnorm', default="False", type=lambda val: True if val == "True" else False)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--anneal-temperature-every', default=1e10, type=int)
    parser.add_argument('--anneal-temperature-rate', default=0, type=float)
    parser.add_argument('--glove', default=".vector_cache/glove.840B.300d.txt")
    parser.add_argument('--fix-word-embedding', default="True", type=lambda val: True if val == "True" else False)
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--max-epoch', default=100, type=int)
    parser.add_argument('--save-dir', default="data/snli_exp")
    parser.add_argument('--optimizer', default='adam')
    parser.add_argument('--l2reg', default=0.0, type=float)

    parser.add_argument('--id', default=0, type=int)
    args = parser.parse_args()


    if args.bidirectional and not args.leaf_rnn:
        import sys
        sys.exit(0)

    args.hash = hash(str(args))

    logs_dir = f"{args.save_dir}/logs"
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    args.tensor_board_path = f"{args.save_dir}/tensorboard"


    models_dir = f"{args.save_dir}/models/m{args.hash}"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    args.save_dir = models_dir

    import random
    seed = args.hash % 1000_000
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)

    file_name = f"{logs_dir}/l{args.hash}.log"
    handler = logging.FileHandler(file_name, mode='w')
    formatter = logging.Formatter("%(asctime)s - %(message)s", "%d-%m-%Y %H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.info(f"args: {str(args)}")
    logger.info(f"hash is: {args.hash}")

    logger.info(f"seed: {seed}")
    logger.info(f"checkpoint's dir is: {args.save_dir}")

    train(args)


if __name__ == '__main__':
    main()

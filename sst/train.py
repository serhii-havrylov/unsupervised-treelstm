import argparse
import logging
import os

from tensorboardX import SummaryWriter

import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.nn.utils import clip_grad_norm_
from torchtext import data, datasets

from sst.models import SSTModel


logger = logging.getLogger("general_logger")
from functools import partial


def train(args):
    text_field = data.Field(lower=args.lower, include_lengths=True,
                            batch_first=True)
    label_field = data.Field(sequential=False)

    filter_pred = None
    if not args.fine_grained:
        filter_pred = lambda ex: ex.label != 'neutral'
    dataset_splits = datasets.SST.splits(
        root='./data/sst', text_field=text_field, label_field=label_field,
        fine_grained=args.fine_grained, train_subtrees=True,
        filter_pred=filter_pred)

    text_field.build_vocab(*dataset_splits, vectors=args.pretrained)
    label_field.build_vocab(*dataset_splits)

    logger.info(f'Initialize with pretrained vectors: {args.pretrained}')
    logger.info(f'Number of classes: {len(label_field.vocab)}')

    train_loader, valid_loader, test_loader = data.BucketIterator.splits(
        datasets=dataset_splits, batch_size=args.batch_size, device=args.device)

    num_classes = len(label_field.vocab)
    model = SSTModel(num_classes=num_classes, num_words=len(text_field.vocab),
                     word_dim=args.word_dim, hidden_dim=args.hidden_dim,
                     clf_hidden_dim=args.clf_hidden_dim,
                     clf_num_layers=args.clf_num_layers,
                     use_leaf_rnn=args.leaf_rnn,
                     bidirectional=args.bidirectional,
                     intra_attention=args.intra_attention,
                     use_batchnorm=args.batchnorm,
                     dropout_prob=args.dropout)
    if args.pretrained:
        model.word_embedding.weight.data.set_(text_field.vocab.vectors)
    if args.fix_word_embedding:
        logger.info('Will not update word embeddings')
        model.word_embedding.weight.requires_grad = False
    logger.info(f'Using device {args.device}')
    model = model.cuda(args.device)
    params = [p for p in model.parameters() if p.requires_grad]
    if args.optimizer == 'adam':
        optimizer_class = optim.Adam
    elif args.optimizer == 'adagrad':
        optimizer_class = optim.Adagrad
    elif args.optimizer == 'adadelta':
        optimizer_class = optim.Adadelta
    optimizer = optimizer_class(params=params, weight_decay=args.l2reg)
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, mode='max', factor=0.5,
        patience=20 * args.halve_lr_every, verbose=True)


    def reduce_lr(self, epoch):
        lr_scheduler.ReduceLROnPlateau._reduce_lr(self, epoch)
        logger.info(f"learning rate is reduced by factor 0.5!")
    scheduler._reduce_lr = partial(reduce_lr, scheduler)


    patience = 20 * 5
    last_time = 0


    criterion = nn.CrossEntropyLoss()

    train_summary_writer = SummaryWriter(
        log_dir=os.path.join(args.tensor_board_path, 'log' + str(hash(str(args))), 'train'))
    valid_summary_writer = SummaryWriter(
        log_dir=os.path.join(args.tensor_board_path, 'log' + str(hash(str(args))), 'valid'))
    test_summary_writer = SummaryWriter(
        log_dir=os.path.join(args.tensor_board_path, 'log' + str(hash(str(args))), 'test'))

    def run_iter(batch, is_training):
        model.train(is_training)
        words, length = batch.text
        label = batch.label
        logits = model(words=words, length=length)
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
    validate_every = num_train_batches // 20
    best_vaild_accuacy = 0
    iter_count = 0
    for batch_iter, train_batch in enumerate(train_loader):
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
            valid_loss_sum = valid_accuracy_sum = 0
            num_valid_batches = len(valid_loader)
            for valid_batch in valid_loader:
                valid_loss, valid_accuracy = run_iter(
                    batch=valid_batch, is_training=False)
                valid_loss_sum += valid_loss.item()
                valid_accuracy_sum += valid_accuracy.item()
            valid_loss = valid_loss_sum / num_valid_batches
            valid_accuracy = valid_accuracy_sum / num_valid_batches
            add_scalar_summary(
                summary_writer=valid_summary_writer,
                name='loss', value=valid_loss, step=iter_count)
            add_scalar_summary(
                summary_writer=valid_summary_writer,
                name='accuracy', value=valid_accuracy, step=iter_count)
            scheduler.step(valid_accuracy)
            progress = train_loader.iterations / len(train_loader)
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
                    return
            if progress > args.max_epoch:
                break

            with torch.no_grad():
                num_correct = 0
                num_data = len(dataset_splits[2])
                for test_batch in test_loader:
                    words, length = test_batch.text
                    label = test_batch.label
                    logits = model(words=words, length=length)
                    label_pred = logits.max(1)[1]
                    num_correct_batch = torch.eq(label, label_pred).long().sum()
                    num_correct_batch = num_correct_batch.item()
                    num_correct += num_correct_batch
                add_scalar_summary(summary_writer=test_summary_writer, name='accuracy', value=num_correct/num_data,
                                   step=iter_count)
                logger.info(f'Accuracy: {num_correct / num_data:.4f} # data: {num_data} # correct: {num_correct}')


def main():
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument('--word-dim', required=True, type=int)
    parser.add_argument('--hidden-dim', required=True, type=int)
    parser.add_argument('--clf-hidden-dim', required=True, type=int)
    parser.add_argument('--clf-num-layers', required=True, type=int)
    parser.add_argument('--leaf-rnn', type=lambda val: True if val == "True" else False)
    parser.add_argument('--bidirectional', type=lambda val: True if val == "True" else False)
    parser.add_argument('--intra-attention', type=lambda val: True if val == "True" else False)
    parser.add_argument('--batchnorm', type=lambda val: True if val == "True" else False)
    parser.add_argument('--dropout', default=0.0, type=float)
    parser.add_argument('--l2reg', default=0.0, type=float)
    parser.add_argument('--pretrained', default=None)
    parser.add_argument('--fix-word-embedding', type=lambda val: True if val == "True" else False)
    parser.add_argument('--device', required=True, type=int)
    parser.add_argument('--batch-size', required=True, type=int)
    parser.add_argument('--max-epoch', required=True, type=int)
    parser.add_argument('--save-dir', required=True)
    parser.add_argument('--omit-prob', default=0.0, type=float)
    parser.add_argument('--optimizer', default='adam')
    parser.add_argument('--fine-grained', type=lambda val: True if val == "True" else False)
    parser.add_argument('--halve-lr-every', default=2, type=int)
    parser.add_argument('--lower', type=lambda val: True if val == "True" else False)

    parser.add_argument('--id', type=int)
    args = parser.parse_args()

    if args.bidirectional and not args.leaf_rnn:
        import sys
        sys.exit(0)

    logs_dir = f"{args.save_dir}/logs"
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    args.tensor_board_path = f"{args.save_dir}/tensorboard"

    models_dir = f"{args.save_dir}/models/m{hash(str(args))}"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    args.save_dir = models_dir

    import random
    seed = hash(str(args)) % 1000_000
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)

    file_name = f"{logs_dir}/l{hash(str(args))}.log"
    handler = logging.FileHandler(file_name, mode='w')
    formatter = logging.Formatter("%(asctime)s - %(message)s", "%d-%m-%Y %H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.info(f"args: {str(args)}")
    logger.info(f"hash is: {hash(str(args))}")

    logger.info(f"seed: {seed}")
    logger.info(f"checkpoint's dir is: {args.save_dir}")

    train(args)


if __name__ == '__main__':
    main()

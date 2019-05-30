import argparse
import os
import pickle

import numpy as np
import torch
from flask import Flask, jsonify, request
from tqdm import tqdm

from model import IOBTagger
from util import *


def parse_args():
    parser = argparse.ArgumentParser()
    # init subparsers
    subparsers = parser.add_subparsers(
        help='argument for specific functions', dest='function')
    subparsers.required = True

    # vocab
    vocab_parser = subparsers.add_parser('vocab')
    vocab_parser.add_argument('-t', '--tag', type=str,
                              required=True, help="Path to train tag file")
    vocab_parser.add_argument('-o', '--out', type=str,
                              required=True, help="Path to output binary file")
    vocab_parser.add_argument(
        '--size', type=int, default=50000, help="Maximum vocabulary size")
    vocab_parser.add_argument('--cut-off', type=int,
                              default=2, help="Frequency cutoff")

    # train
    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('--cuda', action="store_true")
    train_parser.add_argument(
        '--train', type=str, required=True, help="Path to train tag file")
    train_parser.add_argument(
        '--valid', type=str, required=True, help="Path to valid tag file")
    train_parser.add_argument(
        '--vocab', type=str, required=True, help="Path to vocab binary file")
    train_parser.add_argument('--work-dir', type=str,
                              required=True, help="Path to work directory")
    train_parser.add_argument('--embed-size', type=int, default=64)
    train_parser.add_argument('--hidden-size', type=int, default=128)
    train_parser.add_argument('--num-layers', type=int, default=1)
    train_parser.add_argument('--bidirectional', action="store_true")
    train_parser.add_argument('--dropout', type=float, default=0.1)
    train_parser.add_argument('--lr', type=float, default=0.1)
    train_parser.add_argument('--momentum', type=float, default=0.9)
    train_parser.add_argument('--num-epochs', type=int, default=100)
    train_parser.add_argument('--batch-size', type=int, default=32)
    train_parser.add_argument('--patience', type=int, default=5)
    train_parser.add_argument('--seed', type=int, default=8591)

    # eval
    eval_parser = subparsers.add_parser('eval')
    eval_parser.add_argument('-w', '--work-dir', type=str,
                             required=True, help="Path to work directory")
    eval_parser.add_argument('-t', '--tag', type=str,
                             required=True, help="Path to tag file")
    eval_parser.add_argument('--cuda', action="store_true")

    # serve
    serve_parser = subparsers.add_parser('serve')
    serve_parser.add_argument(
        '-w', '--work-dir', type=str, required=True, help='Path to work directory')
    serve_parser.add_argument('-p', '--port', type=int,
                              default=5000, help="Flask port")
    serve_parser.add_argument(
        '-d', '--debug', action="store_true", help="Flask debug mode")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if args.function == "vocab":
        make_vocab(args)
    elif args.function == "train":
        train(args)
    elif args.function == "eval":
        evaluate(args)
    elif args.function == "serve":
        serve(args)


def make_vocab(args):
    """ Creates vocabulary binary file
    """
    size = args.size
    cutoff = args.cut_off

    # Read sentences
    tag_file = args.tag
    sentences, tags, intents = read_tag_file(tag_file)

    # Create vocab
    vocab = Vocab(sentences, tags, intents, size, cutoff)
    print(vocab)

    # Save vocab binary
    vocab_out = args.out
    print("saving vocab to", vocab_out)
    with open(vocab_out, 'wb') as fout:
        pickle.dump(vocab, fout)


def train(args):
    # Data Config
    train_data = list(zip(*read_tag_file(args.train)))
    valid_data = list(zip(*read_tag_file(args.valid)))
    vocab = load_from_pickle(args.vocab)

    print("train:", len(train_data))
    print("valid:", len(valid_data))
    print(vocab)

    # Model config
    use_cuda = args.cuda and torch.cuda.is_available()

    conf = {
        "embed_size": args.embed_size,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "bidirectional": args.bidirectional,
        "dropout": args.dropout,
        "vocab": vocab,
        "use_cuda": use_cuda
    }
    model = IOBTagger(conf)
    model.init_weights()

    # Training Config
    lr = args.lr
    momentum = args.momentum

    print("optimizer: SGD")
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    num_epochs = args.num_epochs
    batch_size = args.batch_size
    patience = args.patience

    # Main Loop Here
    best_valid_loss = None
    wait_patience = 0

    work_dir = args.work_dir
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    model_save_path = os.path.join(work_dir, 'model.pt')

    for epoch in range(1, num_epochs+1):
        print('epoch', epoch)

        # Train
        model.train()

        cum_tag_loss = 0
        cum_intent_loss = 0
        cum_words = 0
        cum_sents = 0

        for batch_sents, batch_tags, batch_intents in batch_iter(train_data, batch_size, shuffle=True):

            optimizer.zero_grad()

            ntokens = sum(1 for sent in batch_sents for word in sent)
            nsents = len(batch_sents)

            # Forward
            tags_losses, intent_losses = model(
                batch_sents, batch_tags, batch_intents)

            tags_loss = tags_losses.sum() / ntokens
            intent_loss = intent_losses.sum() / nsents

            # Backprop
            loss = tags_loss + intent_loss

            loss.backward()
            optimizer.step()

            cum_tag_loss += tags_loss.cpu().data.item() * ntokens
            cum_intent_loss += intent_loss.cpu().data.item() * nsents

            cum_words += ntokens
            cum_sents += nsents

        train_tag_loss = cum_tag_loss / cum_words
        train_intent_loss = cum_intent_loss / cum_sents
        print("[train] tag loss {:2f} intent loss {:2f}"
              .format(train_tag_loss, train_intent_loss))

        # Valid
        cum_tag_loss = 0
        cum_intent_loss = 0
        cum_words = 0
        cum_sents = 0

        with torch.no_grad():
            model.eval()

            for batch_sents, batch_tags, batch_intents in batch_iter(valid_data, batch_size):
                ntokens = sum(1 for sent in batch_sents for word in sent)
                nsents = len(batch_sents)

                # Forward
                tags_losses, intent_losses = model(
                    batch_sents, batch_tags, batch_intents)

                tags_loss = tags_losses.sum() / ntokens
                intent_loss = intent_losses.sum() / nsents

                cum_tag_loss += tags_loss.cpu().data.item() * ntokens
                cum_intent_loss += intent_loss.cpu().data.item() * nsents

                cum_words += ntokens
                cum_sents += nsents

        valid_tag_loss = cum_tag_loss / cum_words
        valid_intent_loss = cum_intent_loss / cum_sents
        print("[valid] tag loss {:2f} intent loss {:2f}"
              .format(valid_tag_loss, valid_intent_loss))

        if best_valid_loss is None or valid_tag_loss < best_valid_loss:
            best_valid_loss = valid_tag_loss
            model.save(model_save_path)
        else:
            wait_patience += 1

        if wait_patience >= patience:
            print(
                "valid tag loss not improved for {} epochs, early stopping".format(patience))
            break


def evaluate(args):
    # data
    test_data = list(zip(*read_tag_file(args.tag)))

    # model
    work_dir = args.work_dir
    model_path = os.path.join(work_dir, 'model.pt')
    use_cuda = args.cuda and torch.cuda.is_available()
    model = IOBTagger.load(model_path, use_cuda)

    # Metrics
    # intent
    nintent_correct = 0
    # f1, precision, recall
    metrics = {}
    categories = ['average', 'action', 'refer', 'attribute', 'value']
    for cat in categories:
        metrics[cat] = {'f1': [], 'precision': [], 'recall': []}

    for sent, tags_true, intent_true in tqdm(test_data):

        if all(x == 'O' for x in tags_true):
            continue

        # Predict
        tags_pred, intent_pred = model.predict(sent)

        # Category
        #print('sent', sent)
        #print("tags", tags_true)
        #print("pred", tags_pred)

        def filter_category(tags, cat):
            filtered_tags = []
            for tag in tags:
                if cat in tag:
                    filtered_tags.append(tag)
                else:
                    filtered_tags.append('O')
            return filtered_tags

        for cat in categories:
            if cat == "average":
                cat_pred = tags_pred
                cat_true = tags_true
            else:
                cat_pred = filter_category(tags_pred, cat)
                cat_true = filter_category(tags_true, cat)

            if all(x == 'O' for x in cat_true):
                # the case where nothing exists
                continue

            f1_cat, p_cat, r_cat = computeF1Score(cat_pred, cat_true)

            metrics[cat]['f1'].append(f1_cat)
            metrics[cat]['precision'].append(p_cat)
            metrics[cat]['recall'].append(r_cat)

        nintent_correct += intent_pred == intent_true

    intent_acc = nintent_correct / len(test_data)
    print("Intent", intent_acc)

    for cat in categories:
        f1 = np.mean(metrics[cat]['f1'])
        precision = np.mean(metrics[cat]['precision'])
        recall = np.mean(metrics[cat]['recall'])
        print("Category", cat, "F1", f1, "Precision",
              precision, "Recall", recall)


def serve(args):
    work_dir = args.work_dir
    port = args.port
    debug = args.debug

    model_path = os.path.join(work_dir, 'model.pt')
    model = IOBTagger.load(model_path)
    print("Loaded model:", model_path)

    app = Flask(__name__)

    @app.route("/nlu", methods=["POST"])
    def tag():
        args = request.json or request.form

        sent = args['sent']
        print("Sentence:", sent)

        result = model.tag(sent)
        print("Predicted:", result)
        return jsonify(result)

    app.run(
        host="0.0.0.0",
        port=port,
        debug=debug
    )


if __name__ == '__main__':
    main()

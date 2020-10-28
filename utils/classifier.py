import torch
import os
from torch import nn
import torch.nn.functional as F
import argparse
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter
from classifier_helpers import *
from torchtext import data
from sklearn.metrics import auc, roc_curve
from torch import optim
from tqdm import tqdm
from os.path import join
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"]="3"

class Batch:
    """Object for holding a batch of data with mask during training.
    Input is a batch from a torch text iterator.
    """

    def __init__(self, torch_batch, pad_index, use_cuda=False):
        """
        Create a new joey batch from a torch batch.
        This batch extends torch text's batch attributes with src and trg
        length, masks, number of non-padded tokens in trg.
        Furthermore, it can be sorted by src length.

        :param torch_batch:
        :param pad_index:
        :param use_cuda:
        """
        self.src, self.src_lengths = torch_batch.src
        self.src_mask = (self.src != pad_index).unsqueeze(1)
        self.label = torch_batch.label.type(torch.float32)
        self.nseqs = self.src.size(0)
        self.trg_input = None
        self.trg = None
        self.trg_mask = None
        self.trg_lengths = None
        self.ntokens = None
        self.use_cuda = use_cuda

        if hasattr(torch_batch, "trg"):
            trg, trg_lengths = torch_batch.trg
            # trg_input is used for teacher forcing, last one is cut off
            self.trg_input = trg[:, :-1]
            self.trg_lengths = trg_lengths
            # trg is used for loss computation, shifted by one since BOS
            self.trg = trg[:, 1:]
            # we exclude the padded areas from the loss computation
            self.trg_mask = (self.trg_input != pad_index).unsqueeze(1)
            self.ntokens = (self.trg != pad_index).data.sum().item()

        if use_cuda:
            self._make_cuda()

    def to_device(self, device: str):
        self.src = self.src.to(device)
        self.src_mask = self.src_mask.to(device)

        if self.trg_input is not None:
            self.trg_input = self.trg_input.to(device)
            self.trg = self.trg.to(device)
            self.trg_mask = self.trg_mask.to(device)


    def _make_cuda(self):
        """
        Move the batch to GPU

        :return:
        """
        self.src = self.src.cuda()
        self.src_mask = self.src_mask.cuda()

        if self.trg_input is not None:
            self.trg_input = self.trg_input.cuda()
            self.trg = self.trg.cuda()
            self.trg_mask = self.trg_mask.cuda()

    def sort_by_src_lengths(self):
        """
        Sort by src length (descending) and return index to revert sort

        :return:
        """
        _, perm_index = self.src_lengths.sort(0, descending=True)
        rev_index = [0]*perm_index.size(0)
        for new_pos, old_pos in enumerate(perm_index.cpu().numpy()):
            rev_index[old_pos] = new_pos

        sorted_src_lengths = self.src_lengths[perm_index]
        sorted_src = self.src[perm_index]
        sorted_src_mask = self.src_mask[perm_index]
        if self.trg_input is not None:
            sorted_trg_input = self.trg_input[perm_index]
            sorted_trg_lengths = self.trg_lengths[perm_index]
            sorted_trg_mask = self.trg_mask[perm_index]
            sorted_trg = self.trg[perm_index]

        self.src = sorted_src
        self.src_lengths = sorted_src_lengths
        self.src_mask = sorted_src_mask

        if self.trg_input is not None:
            self.trg_input = sorted_trg_input
            self.trg_mask = sorted_trg_mask
            self.trg_lengths = sorted_trg_lengths
            self.trg = sorted_trg

        if self.use_cuda:
            self._make_cuda()

        return rev_index


class BinaryClassifier(nn.Module):
    def __init__(self, seq2seq_model: Model, seq2seq_output_dim: int,
                 n_linear_layers: int, linear_hidden_size: int):
        super(BinaryClassifier, self).__init__()
        self.seq2seq_model = seq2seq_model
        last_output_dim = seq2seq_output_dim * 2 # you want to mean and max pool
        self.linear_list = nn.ModuleList()
        for _ in range(n_linear_layers):
            self.linear_list.append(nn.Linear(last_output_dim, linear_hidden_size))
            last_output_dim = linear_hidden_size
        self.output_layer = nn.Linear(last_output_dim, 1) # binary output
    def forward(self, input: Batch):
        output, _ = self.seq2seq_model.run_batch(input, max_output_length = 200, beam_size = 1, beam_alpha = -1)
        #breakpoint()
        avg_pool = torch.mean(output, dim=1)
        max_pool, _ = torch.max(output, dim=1)
        output = torch.cat((avg_pool, max_pool), dim=1)
        for layer in self.linear_list:
            output = layer(output)
            output = F.gelu(output)
        output = self.output_layer(output)
        output = F.sigmoid(output)
        return output


def classification_pipeline(cfg: str):
    cfg = load_config(cfg)
    train_config = cfg["training"]
    data_config = cfg["data"]
    model_config = cfg["model"]
    device = train_config["device"]


    set_seed(seed=train_config.get("random_seed", 42))
    print("initializing the model !")
    level = data_config["level"]
    lowercase = data_config["lowercase"]
    tok_fun = lambda s: list(s) if level == "char" else s.split()
    src_field = data.Field(init_token=None, eos_token=EOS_TOKEN,
                           pad_token=PAD_TOKEN, tokenize=tok_fun,
                           batch_first=True, lower=lowercase,
                           unk_token=UNK_TOKEN,
                           include_lengths=True)

    src_vocab = Vocabulary(file = data_config["src_vocab"])
    trg_vocab = Vocabulary(file = data_config["trg_vocab"])
    src_field.vocab = src_vocab
    model = build_model(model_config, src_vocab=src_vocab, trg_vocab=trg_vocab)
    model_checkpoint = load_checkpoint(path=train_config["load_model"], use_cuda=train_config["use_cuda"])
    model.load_state_dict(model_checkpoint["model_state"])
    seq2seq_output_dim = model.decoder.output_layer.weight.size(1)

    classifier_model = BinaryClassifier(seq2seq_model=model, seq2seq_output_dim=seq2seq_output_dim,
                                        n_linear_layers=train_config.get("n_linear_layers", 1),
                                        linear_hidden_size=train_config.get("linear_hidden_size", 1024))


    print("model loaded successfully, getting data ready")
    model_dir = train_config["model_dir"]
    tb_writer = SummaryWriter(
        log_dir=model_dir + "/tensorboard/")
    batch_size = train_config["batch_size"]

    label_field = data.Field(sequential=False, use_vocab=False, batch_first=True)

    train, val, test = data.TabularDataset.splits(".",
                                             train=data_config["train_path"],
                                             validation=data_config["dev_path"],
                                             test=data_config["test_path"],
                                             format="csv",
                                             skip_header=True,
                                             fields=[('src', src_field), ('label', label_field)])

    train_iter = data.BucketIterator(
        repeat=False, sort=False, dataset=train,
        batch_size=batch_size, batch_size_fn=token_batch_size_fn,
        train=True, sort_within_batch=True,
        sort_key=lambda x: len(x.src), shuffle=True, device=device)

    valid_iter = data.BucketIterator(
        repeat=False, sort=False, dataset=val,
        batch_size=batch_size, batch_size_fn=token_batch_size_fn,
        train=False, shuffle=True, device=device)

    test_iter = data.BucketIterator(
        repeat=False, sort=False, dataset=val,
        batch_size=batch_size, batch_size_fn=token_batch_size_fn,
        train=False, shuffle=True, device=device)

    dataloaders = {"train": train_iter, "valid": valid_iter, "test": test_iter}


    binary_xent_loss = nn.BCELoss()
    optimizer = optim.Adam(classifier_model.parameters(), lr=train_config["learning_rate"])
    print("now training")
    classifier_model.to(device)
    train_loop(model=classifier_model, loss=binary_xent_loss, optimizer=optimizer,
          epochs=train_config["epochs"], dataloaders=dataloaders, tb_writer=tb_writer, model_dir = model_dir,
          threshold = train_config.get("classification_threshold", 0.5))



def train_loop(model: Model, loss: nn.Module, optimizer, epochs,
          dataloaders, tb_writer, model_dir, threshold = 0.5):
    update_no = 0

    for epoch_no in range(1, epochs + 1):

        for phase in ["train", "valid", "test"]:
            print(f"epoch {epoch_no} phase {phase}")
            if phase == "train": 
                model.train()
            else: 
                model.eval()
            epoch_losses = []
            epoch_labs = []
            epoch_outputs = []
            epoch_correct_preds = []
            for i, batch in enumerate(tqdm(iter(dataloaders[phase]), desc=f"epoch {epoch_no} phase {phase}")):
                #breakpoint()
                batch = Batch(batch, pad_index= model.seq2seq_model.pad_index)
                with torch.set_grad_enabled(phase == 'train'):
                    #breakpoint()
                    output_probs = model(input = batch)
                    loss_value = loss(output_probs.squeeze(1), batch.label)
                    preds = output_probs.squeeze(1) > threshold
                    correct_preds = (preds == batch.label)
                    if phase == 'train':
                        loss_value.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                        acc = torch.mean(correct_preds.type(torch.float32))
                        tb_writer.add_scalar("train/batch_loss", loss_value.detach(), update_no)
                        tb_writer.add_scalar("train/batch_accuracy", acc.item(), update_no)
                        update_no+=1
                    #breakpoint()
                    epoch_losses.append(loss_value.item())
                    epoch_labs.extend(list(batch.label.detach().cpu().numpy()))
                    epoch_outputs.extend(list(output_probs.detach().cpu().numpy()))
                    epoch_correct_preds.extend(list(correct_preds.detach().cpu().numpy()))


            epoch_loss = np.mean(epoch_losses)
            epoch_acc = np.mean(epoch_correct_preds)

            false_pos_rte, true_pos_rte, thresholds = roc_curve(y_true=np.array(epoch_labs),
                                                                y_score=np.array(epoch_outputs), pos_label=1)
            auc_score = auc(false_pos_rte, true_pos_rte)

            if phase != "test":
                print(f"epoch {epoch_no} and acc is {epoch_acc} and auc is {auc_score}")

                tb_writer.add_scalar("{}/loss".format(phase), epoch_loss, epoch_no)
                tb_writer.add_scalar("{}/accuracy".format(phase), epoch_acc, epoch_no)
                tb_writer.add_scalar("{}/auc_score".format(phase), auc_score, epoch_no)

            if phase in ("valid", "test"):

                plt.figure()
                plt.plot(false_pos_rte, true_pos_rte, color='navy',
                         label='ROC curve (area = %0.2f)' % auc_score)
                plt.plot([0, 1], [0, 1], color='black', linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Validation ROC Curve to Predict if Harvestable by STOKE at Epoch {}'.format(epoch_no))
                plt.legend(loc="lower right")
                plt.savefig("{}/{}_roc_curve_ep_{}.png".format(model_dir, phase, epoch_no), dpi=300, pad_inches=2)
        if (epoch_no % 2) == 0: 

            state_dict = {"model_state": model.state_dict()}
            torch.save(state_dict, "{}/model_epoch_{}.ckpt".format(model_dir, epoch_no))


def eval_thresholds(cfg: str, model_path: str, valid_data_path: str = None):
    cfg = load_config(cfg)
    train_config = cfg["training"]
    data_config = cfg["data"]
    model_config = cfg["model"]
    device = train_config["device"]

    set_seed(seed=train_config.get("random_seed", 42))
    print("initializing the model !")
    level = data_config["level"]
    lowercase = data_config["lowercase"]
    tok_fun = lambda s: list(s) if level == "char" else s.split()
    src_field = data.Field(init_token=None, eos_token=EOS_TOKEN,
                           pad_token=PAD_TOKEN, tokenize=tok_fun,
                           batch_first=True, lower=lowercase,
                           unk_token=UNK_TOKEN,
                           include_lengths=True)

    src_vocab = Vocabulary(file=data_config["src_vocab"])
    trg_vocab = Vocabulary(file=data_config["trg_vocab"])
    src_field.vocab = src_vocab
    model = build_model(model_config, src_vocab=src_vocab, trg_vocab=trg_vocab)
    seq2seq_output_dim = model.decoder.output_layer.weight.size(1)

    classifier_model = BinaryClassifier(seq2seq_model=model, seq2seq_output_dim=seq2seq_output_dim,
                                        n_linear_layers=train_config.get("n_linear_layers", 1),
                                        linear_hidden_size=train_config.get("linear_hidden_size", 1024))

    model_dir = train_config["model_dir"]
    model_path = join(model_dir, model_path)
    classifier_model.load_state_dict(torch.load(model_path)["model_state"])

    print("model loaded successfully from {}, getting data ready".format(model_path))
    batch_size = train_config["batch_size"]
    label_field = data.Field(sequential=False, use_vocab=False, batch_first=True)
    valid_data_path = valid_data_path if valid_data_path else data_config["dev_path"]
    val = data.TabularDataset(path = valid_data_path, format="csv", skip_header=True,
                                           fields=[('src', src_field), ('label', label_field)])
    print("data loaded from {}".format(valid_data_path))


    valid_iter = data.BucketIterator(
        repeat=False, sort=False, dataset=val,
        batch_size=batch_size, batch_size_fn=token_batch_size_fn,
        train=False, shuffle=True, device=device)

    classifier_model.to(device)

    all_labs = []
    all_outputs = []
    pbar = tqdm(total = len(val), desc = "predicting")

    for i, batch in enumerate(iter(valid_iter)):
        # breakpoint()
        batch = Batch(batch, pad_index=classifier_model.seq2seq_model.pad_index)
        with torch.no_grad():
            # breakpoint()
            output_probs = classifier_model(input=batch)
            all_labs.extend(list(batch.label.detach().cpu().numpy()))
            all_outputs.extend(list(output_probs.detach().cpu().numpy()))
            pbar.update(len(batch.label))

    thresholds = [i/100 for i in range(0,100)]
    fprs = []
    tprs = []
    labs_arr = np.array(all_labs).reshape(-1)
    for threshold in tqdm(thresholds, desc = "threshold iteration"):
        preds = np.array(all_outputs) > threshold
        preds = preds.reshape(-1)
        tpr = np.sum((preds * labs_arr)) / np.sum(labs_arr)
        fpr = np.sum((preds * (1 - labs_arr))) / np.sum(1 - labs_arr)
        tprs.append(tpr)
        fprs.append(fpr)
    print('done with thresholds')

    plt.figure()
    breakpoint()
    plt.plot(tprs, thresholds, label="true positive rate", color = "darkgreen")
    plt.plot(fprs, thresholds, label="false positive rate", color = "darkred")
    plt.xlim([-0.05, 1.20])
    plt.ylim([-0.05, 1.20])
    plt.xlabel('Threshold Value')
    plt.ylabel('True / False Positive Rate')
    plt.title('True / False Positive Rates over Thresholds')
    plt.legend(loc="lower left")
    plt.savefig("{}/eval_threshold.png".format(model_dir), dpi=300, pad_inches=2)
    df = pd.DataFrame(list(zip(thresholds, tprs, fprs)), 
                           columns =['threshold', 'tpr', 'fpr'])
    df.to_csv("{}/eval_threshold.csv".format(model_dir), index = False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Classifier')
    parser.add_argument("--config", type=str,
                        help="config file for classification ")
    parser.add_argument("--train", type=bool, dest="train", help="flag for training")
    parser.add_argument("--eval_threshold", type=bool, dest="eval_threshold", help="flag for setting threshold")
    parser.add_argument("--model_path", type=str, dest="model_path", help="path to model to evaluate")
    parser.add_argument("--eval_data_path", type=str, dest="eval_data_path", default = None, help="path to data to evaluate on")
    args = parser.parse_args()
    if args.train:
        classification_pipeline(cfg=args.config)
    elif args.eval_threshold:
        eval_thresholds(cfg=args.config, model_path=args.model_path, valid_data_path=args.eval_data_path)



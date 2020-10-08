import torch
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


class BinaryClassifier(nn.Module):
    def __init__(self, seq2seq_model: Model, seq2seq_output_dim: int,
                 n_linear_layers: int, linear_hidden_size: int):
        super(BinaryClassifier, self).__init__()
        self.seq2seq_model = seq2seq_model
        last_output_dim = seq2seq_output_dim * 2 # you want to mean and max pool
        self.linear_list = nn.ModuleList[]
        for _ in n_linear_layers:
            self.linear_list.append(nn.Linear(last_output_dim, linear_hidden_size))
            last_output_dim = linear_hidden_size
        self.output_layer = nn.Linear(last_output_dim, 1) # binary output
    def forward(self, input: torch.Tensor):
        output, _ = self.seq2seq_model.run_batch(input)
        avg_pool = torch.mean(output, dim=1)
        max_pool = torch.max(output, dim=1)
        output = torch.cat((avg_pool, max_pool), dim=1)
        for layer in self.linear_list:
            output = layer(output)
            output = F.gelu(output)
        output = self.output_layer(output)
        output = F.sigmoid(output)
        return output


def classification_pipeline(cfg: str):
    cfg = load_config(cfg_file)
    train_config = cfg["training"]
    data_config = cfg["data"]
    model_config = cfg["model"]
    device = train_config["device"]


    set_seed(seed=train_config.get("random_seed", 42))
    print("initializing the model !")
    model = build_model(model_config, src_vocab=src_vocab, trg_vocab=trg_vocab)
    breakpoint()
    seq2seq_output_dim = model.decoder.output_layer.weight.size(1)

    classifier_model = BinaryClassifier(seq2seq_model=model, seq2seq_output_dim=seq2seq_output_dim,
                                        n_linear_layers=train_config.get("n_linear_layers", 1),
                                        linear_hidden_size=train_config.get("linear_hidden_size", 1024))



    model_dir = train_config["model_dir"]
    tb_writer = SummaryWriter(
        log_dir=model_dir + "/tensorboard/")

    src_field = data.Field(init_token=None, eos_token=EOS_TOKEN,
                           pad_token=PAD_TOKEN, tokenize=tok_fun,
                           batch_first=True, lower=lowercase,
                           unk_token=UNK_TOKEN,
                           include_lengths=True)

    src_vocab = Vocabulary(file = data_config["vocab_path"])
    src_field.vocab = src_vocab

    label_field = data.Field(sequential=False, use_vocab=False, batch_first=True)

    train, val, test = data.TabularDataset.splits(".",
                                             train=data_config["train_path"],
                                             validation=data_config["dev_path"],
                                             test=data_config["test_path"],
                                             format="csv",
                                             skip_header=True,
                                             fields=[('text', TEXT)], ('label', LABEL))

    train_iter = data.BucketIterator(
        repeat=False, sort=False, dataset=train,
        batch_size=batch_size, batch_size_fn=batch_size_fn,
        train=True, sort_within_batch=True,
        sort_key=lambda x: len(x.src), shuffle=True, device=device)

    valid_iter = data.BucketIterator(
        repeat=False, sort=False, dataset=val,
        batch_size=batch_size, batch_size_fn=batch_size_fn,
        train=False, sort = False, shuffle=True, device=device)

    test_iter = data.BucketIterator(
        repeat=False, sort=False, dataset=val,
        batch_size=batch_size, batch_size_fn=batch_size_fn,
        train=False, sort = False, shuffle=True, device=device)

    dataloaders = {"train": train_iter, "valid": valid_iter, "test": test_iter}


    binary_xent_loss = nn.BCELoss()
    optimizer = optim.Adam(classifier_model.parameters(), lr=train_config["learning_rate"])
    train(model=classifier_model, loss=binary_xent_loss, optimizer=optimizer,
          epochs=train_config["epochs"], tb_writer = tb_writer, model_dir = model_dir,
          threshold = train_config.get("classification_threshold", 0.5))



def train(model: Model, loss: nn.Module, optimizer, epochs,
          dataloaders, tb_writer, model_dir, threshold = 0.5):


    for epoch_no in range(1, epochs + 1):
        for phase in ["val", "train"]:
            epoch_losses = []
            epoch_labs = []
            epoch_outputs = []
            epoch_correct_preds = []
            for batch in iter(dataloaders[phase]):
                inputs = batch.text
                labels = batch.label
                with torch.set_grad_enabled(phase == 'train'):
                    output_probs = model(inputs = batch.text)
                    loss = loss(output_probs, labels)
                    preds = output_probs > threshold
                    correct_preds = (preds == batch.label)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                        acc = torch.mean(correct_preds)
                        tb_writer.add_scalar("train/batch_loss", loss.detach())
                        tb_writer.add_scalar("train/batch_accuracy", acc)
                    breakpoint()
                    epoch_losses.append(loss.item())
                    epoch_labs.extend(list(labels.numpy()))
                    epoch_outputs.extend(list(output_probs.numpy()))
                    epoch_correct_preds.extend(list(correct_preds.numpy()))



            epoch_loss = np.mean(epoch_losses)
            epoch_acc = np.mean(epoch_correct_preds)

            false_pos_rte, true_pos_rte, thresholds = roc_curve(y_true=np.array(epoch_labs),
                                                                y_score=np.array(epoch_outputs), pos_label=1)
            auc_score = auc_score(false_pos_rte, true_pos_rte)

            if phase != "test":

                tb_writer.add_scalar("{}/loss".format(phase), epoch_loss)
                tb_writer.add_scalar("{}/accuracy".format(phase), epoch_acc)
                tb_writer.add_scalar("{}/auc_score".format(phase), auc_score)

            if phase in ("valid", "test"):

                plt.figure()
                plt.plot(false_pos_rte, true_pos_rte, color='navy',
                         label='ROC curve (area = %0.2f)' % auc_score)
                plt.plot([0, 1], [0, 1], color='black', linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Validation Receiver operating characteristic at epoch {}'.format(epoch_no))
                plt.legend(loc="lower right")
                plt.savefig("{}/validation_roc_curve_ep_{}.png".format(model_dir, epoch_no))

        state_dict = {"model_state": model.state_dict()}
        torch.save(state_dict, "{}/model_epoch_{}.ckpt".format(model_dir))




if __name__ == "__main__":
    parser = argparse.ArgumentParser('Classifier')
    parser.add_argument("config", type=str,
                        help="config file for classification ")
    args = parser.parse_args()
    classification_pipeline(cfg=args.config)
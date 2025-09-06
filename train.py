import os
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim.adam import Adam
from torch.utils.data import DataLoader
from model import KhmerTagger
from dataset import TextDataset
from config import TAGS_PUNCT, TAGS_NUM

if __name__ == "__main__":
  lr = 5e-6
  n_epoch = 50
  device = "cuda" if torch.cuda.is_available() else "cpu"
  train_set = TextDataset("data/train.txt")
  val_set = TextDataset("data/dev.txt")

  train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=1)
  val_loader = DataLoader(
    train_set, batch_size=32, shuffle=False, num_workers=1, drop_last=False
  )

  model = KhmerTagger(
    n_punct_features=len(TAGS_PUNCT), n_num_features=len(TAGS_NUM)
  ).to(device)

  criterion = nn.CrossEntropyLoss()
  optimizer = Adam(model.parameters(), lr=lr)

  best_val_acc = 0.0

  # training
  for epoch in range(n_epoch):
    train_loss = 0.0
    train_iteration = 0
    correct = 0
    total = 0

    model.train()
    for x, y_punct, y_num, att, y_mask in tqdm(train_loader, desc="train"):
      x, y_punct, y_num, att, y_mask = (
        x.to(device),
        y_punct.to(device),
        y_num.to(device),
        att.to(device),
        y_mask.to(device),
      )

      y_mask = y_mask.view(-1)

      y_punct = y_punct.view(-1)
      y_num = y_num.view(-1)

      # forward
      y_punct_predict, y_num_predict = model(x, att)

      # reshape
      y_punct_predict = y_punct_predict.view(-1, y_punct_predict.shape[2])
      y_num_predict = y_num_predict.view(-1, y_num_predict.shape[2])

      # compute loss
      loss_punct = criterion(y_punct_predict, y_punct)
      loss_num = criterion(y_num_predict, y_num)

      # weight
      w = 0.25
      loss = loss_punct * (1 - w) + loss_num * w

      # punct
      y_punct_predict = torch.argmax(y_punct_predict, dim=1).view(-1)
      correct += torch.sum(y_mask * (y_punct_predict == y_punct).long()).item()

      # num
      y_num_predict = torch.argmax(y_num_predict, dim=1).view(-1)
      correct += torch.sum(y_mask * (y_num_predict == y_num).long()).item()

      optimizer.zero_grad()

      train_loss += loss.item()
      train_iteration += 1

      loss.backward()
      optimizer.step()

      y_mask = y_mask.view(-1)
      total += torch.sum(y_mask).item() * 2

    # print
    train_loss /= train_iteration
    print(
      f"epoch: {epoch}, Train loss: {train_loss}, Train accuracy: {correct / total}"
    )

    # Evaluation
    num_iteration = 0
    correct = 0
    total = 0
    val_loss = 0

    model.eval()

    with torch.no_grad():
      for x, y_punct, y_num, att, y_mask in tqdm(val_loader, desc="eval"):
        x, y_punct, y_num, att, y_mask = (
          x.to(device),
          y_punct.to(device),
          y_num.to(device),
          att.to(device),
          y_mask.to(device),
        )

        y_mask = y_mask.view(-1)

        y_punct = y_punct.view(-1)
        y_num = y_num.view(-1)

        # forward
        y_punct_predict, y_num_predict = model(x, att)

        # reshape
        y_punct_predict = y_punct_predict.view(-1, y_punct_predict.shape[2])
        y_num_predict = y_num_predict.view(-1, y_num_predict.shape[2])

        # compute loss
        loss_punct = criterion(y_punct_predict, y_punct)
        loss_num = criterion(y_num_predict, y_num)

        # weight
        w = 0.25
        loss = loss_punct * (1 - w) + loss_num * w

        # punct
        y_punct_predict = torch.argmax(y_punct_predict, dim=1).view(-1)
        correct += torch.sum(y_mask * (y_punct_predict == y_punct).long()).item()

        # num
        y_num_predict = torch.argmax(y_num_predict, dim=1).view(-1)
        correct += torch.sum(y_mask * (y_num_predict == y_num).long()).item()

        val_loss += loss.item()
        num_iteration += 1

        y_mask = y_mask.view(-1)
        total += torch.sum(y_mask).item() * 2

      val_acc, val_loss = correct / total, val_loss / num_iteration
      print(f"epoch: {epoch}, Val loss: {val_loss}, Val accuracy: {val_acc}")

      if val_acc > best_val_acc:
        os.makedirs("logs", exist_ok=True)
        torch.save(model.state_dict(), f"logs/checkpoint-{epoch}.pth")
        best_val_acc = val_acc

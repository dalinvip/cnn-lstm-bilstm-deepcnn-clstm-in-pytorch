# coding=utf-8
import os
import sys
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn.utils as utils
import shutil
import random
import hyperparams
torch.manual_seed(hyperparams.seed_num)
random.seed(hyperparams.seed_num)

def eval(data_iter, model, args, scheduler):
    model.eval()
    corrects, avg_loss = 0, 0
    for batch in data_iter:
        feature, target = batch.text, batch.label
        feature.data.t_(), target.data.sub_(1)  # batch first, index align
        if args.cuda:
            feature, target = feature.cuda(), feature.cuda()

        logit = model(feature)
        loss = F.cross_entropy(logit, target, size_average=False)
        # scheduler.step(loss.data[0])

        avg_loss += loss.data[0]
        corrects += (torch.max(logit, 1)
                     [1].view(target.size()).data == target.data).sum()

    size = len(data_iter.dataset)
    avg_loss = loss.data[0]/size
    accuracy = float(corrects)/size * 100.0
    model.train()
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss,
                                                                       accuracy,
                                                                       corrects,
                                                                       size))


def test_eval(data_iter, model, save_path, args, model_count):
    # print(save_path)
    model.eval()
    corrects, avg_loss = 0, 0
    for batch in data_iter:
        feature, target = batch.text, batch.label
        feature.data.t_(), target.data.sub_(1)  # batch first, index align
        if args.cuda:
            feature, target = feature.cuda(), target.cuda()

        logit = model(feature)
        loss = F.cross_entropy(logit, target, size_average=False)

        avg_loss += loss.data[0]
        corrects += (torch.max(logit, 1)
                     [1].view(target.size()).data == target.data).sum()

    size = len(data_iter.dataset)
    avg_loss = loss.data[0]/size
    accuracy = float(corrects)/size * 100.0
    model.train()
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss,
                                                                       accuracy,
                                                                       corrects,
                                                                       size))
    print("model_count {}".format(model_count))
    # test result
    if os.path.exists("./Test_Result.txt"):
        file = open("./Test_Result.txt", "a")
    else:
        file = open("./Test_Result.txt", "w")
    file.write("model " + save_path + "\n")
    file.write("Evaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n".format(avg_loss, accuracy, corrects, size))
    file.write("model_count {} \n".format(model_count))
    file.write("\n")
    file.close()
    shutil.copy("./Test_Result.txt", "./snapshot/" + args.mulu + "/Test_Result.txt")
    # whether to delete the model after test acc so that to save space
    if os.path.isfile(save_path) and args.rm_model is True:
        os.remove(save_path)




def predict(text, model, text_field, label_feild):
    assert isinstance(text, str)
    model.eval()
    text = text_field.tokenize(text)
    text = text_field.preprocess(text)
    text = [[text_field.vocab.stoi[x] for x in text]]
    x = text_field.tensor_type(text)
    x = autograd.Variable(x, volatile=True)
    print(x)
    output = model(x)
    _, predicted = torch.max(output, 1)
    return label_feild.vocab.itos[predicted.data[0][0]+1]

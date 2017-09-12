import os
import sys
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn.utils as utils
import torch.optim.lr_scheduler as lr_scheduler
import shutil
import re
import random
import hyperparams
torch.manual_seed(hyperparams.seed_num)
random.seed(hyperparams.seed_num)

def train(train_iter, dev_iter, test_iter, model, args):
    if args.cuda:
        model.cuda()

    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-8)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.init_weight_decay)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,momentum=)
    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    if args.Adam is True:
        print("Adam Training......")
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.init_weight_decay)
    elif args.SGD is True:
        print("SGD Training.......")
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.init_weight_decay,
                                    momentum=args.momentum_value)
    elif args.Adadelta is True:
        print("Adadelta Training.......")
        optimizer = torch.optim.Adadelta(model.parameters(), lr=args.lr, weight_decay=args.init_weight_decay)

    # lambda1 = lambda epoch: epoch // 30
    # lambda2 = lambda epoch: 0.99 ** epoch
    # print("lambda1 {} lambda2 {} ".format(lambda1, lambda2))
    # scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda2])

    # scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.9)

    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min')


    steps = 0
    epoch_step = 0
    model_count = 0
    model.train()
    for epoch in range(1, args.epochs+1):
        print("\n## 第{} 轮迭代，共计迭代 {} 次 ！##\n".format(epoch, args.epochs))
        # scheduler.step()
        # print("now lr is {} \n".format(scheduler.get_lr()))
        print("now lr is {} \n".format(optimizer.param_groups[0].get("lr")))
        for batch in train_iter:
            feature, target = batch.text, batch.label
            feature.data.t_(), target.data.sub_(1)  # batch first, index align
            if args.cuda:
                feature, target = feature.cuda(), target.cuda()
            optimizer.zero_grad()
            logit = model(feature)
            loss = F.cross_entropy(logit, target)
            loss.backward()
            if args.init_clip_max_norm is not None:
                utils.clip_grad_norm(model.parameters(), max_norm=args.init_clip_max_norm)
            optimizer.step()
            steps += 1
            if steps % args.log_interval == 0:
                train_size = len(train_iter.dataset)
                corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
                accuracy = float(corrects)/batch.batch_size * 100.0
                sys.stdout.write(
                    '\rBatch[{}/{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(steps,
                                                                                train_size, loss.data[0], accuracy,
                                                                                corrects, batch.batch_size))
        # eval and test after every epoch
        eval(dev_iter, model, args, scheduler)
        if not os.path.isdir(args.save_dir):
            os.makedirs(args.save_dir)
        save_prefix = os.path.join(args.save_dir, 'snapshot')
        save_path = '{}_steps{}.pt'.format(save_prefix, steps)
        torch.save(model, save_path)
        print("\n", save_path, end=" ")
        test_model = torch.load(save_path)
        model_count += 1
        test_eval(test_iter, test_model, save_path, args, model_count)
    return model_count


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
        # if args.init_clip_max_norm is not None:
        #     # print("aaaa {} ".format(args.init_clip_max_norm))
        #     utils.clip_grad_norm(model.parameters(), max_norm=args.init_clip_max_norm)

        avg_loss += loss.data[0]
        corrects += (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()

    size = len(data_iter.dataset)
    avg_loss = loss.data[0]/size
    # accuracy = float(corrects)/size * 100.0
    accuracy = 100.0 * corrects/size
    model.train()
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss,
                                                                       accuracy,
                                                                       corrects,
                                                                       size))


def test_eval(data_iter, model, save_path, args, model_count):
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
        # if args.init_clip_max_norm is not None:
        #     # print("aaaa {} ".format(args.init_clip_max_norm))
        #     utils.clip_grad_norm(model.parameters(), max_norm=args.init_clip_max_norm)

        avg_loss += loss.data[0]
        corrects += (torch.max(logit, 1)
                     [1].view(target.size()).data == target.data).sum()

    size = len(data_iter.dataset)
    avg_loss = loss.data[0]/size
    # accuracy = float(corrects)/size * 100.0
    accuracy = 100.0 * corrects/size
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
    # calculate the best score in current file
    resultlist = []
    if os.path.exists("./Test_Result.txt"):
        file = open("./Test_Result.txt")
        for line in file.readlines():
            if line[:10] == "Evaluation":
                resultlist.append(float(line[34:41]))
        result = sorted(resultlist)
        file.close()
        file = open("./Test_Result.txt", "a")
        file.write("\nThe Current Best Result is : " + str(result[len(result) - 1]))
        file.write("\n\n")
        file.close()
    shutil.copy("./Test_Result.txt", "./snapshot/" + args.mulu + "/Test_Result.txt")
    # whether to delete the model after test acc so that to save space
    if os.path.isfile(save_path) and args.rm_model is True:
        os.remove(save_path)

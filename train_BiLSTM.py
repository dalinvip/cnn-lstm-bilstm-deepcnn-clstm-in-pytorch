import os
import sys
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn.utils as utils
# torch.manual_seed(16330)
torch.manual_seed(6163)

def train(train_iter, dev_iter, test_iter, model, args):
    if args.cuda:
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    steps = 0
    model_count = 0
    model.train()
    for epoch in range(1, args.epochs+1):
        print("## 第{} 轮迭代，共计迭代 {} 次 ！##".format(epoch, args.epochs))
        for batch in train_iter:
            feature, target = batch.text, batch.label.data.sub_(1)
            target =autograd.Variable(target)
            if args.cuda:
                feature, target = feature.cuda(), target.cuda()

            optimizer.zero_grad()
            model.zero_grad()
            model.hidden = model.init_hidden(args.lstm_num_layers, args.batch_size)
            if feature.size(1) != args.batch_size:
                model.hidden = model.init_hidden(args.lstm_num_layers, feature.size(1))
            logit = model(feature)
            # target values >=0   <=C - 1 (C = args.class_num)
            loss = F.cross_entropy(logit, target)
            loss.backward()
            utils.clip_grad_norm(model.parameters(), max_norm=1e-4)
            optimizer.step()

            steps += 1
            if steps % args.log_interval == 0:
                train_size = len(train_iter.dataset)
                corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
                accuracy = float(corrects)/batch.batch_size * 100.0
                sys.stdout.write(
                    '\rBatch[{}/{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(steps,
                                                                            train_size,
                                                                             loss.data[0], 
                                                                             accuracy,
                                                                             corrects,
                                                                             batch.batch_size))
            if steps % args.test_interval == 0:
                eval(dev_iter, model, args)
            if steps % args.save_interval == 0:
                if not os.path.isdir(args.save_dir): os.makedirs(args.save_dir)
                save_prefix = os.path.join(args.save_dir, 'snapshot')
                save_path = '{}_steps{}.pt'.format(save_prefix, steps)
                torch.save(model, save_path)
                test_eval(test_iter, model, save_path, args)
                model_count += 1
                print("model_count \n", model_count)
    return model_count


def eval(data_iter, model, args):
    model.eval()
    corrects, avg_loss = 0, 0
    for batch in data_iter:
        feature, target = batch.text, batch.label.data.sub_(1)
        target = autograd.Variable(target)
        if args.cuda:
            feature, target = feature.cuda(), target.cuda()
        model.hidden = model.init_hidden(args.lstm_num_layers, batch.batch_size)
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


def test_eval(data_iter, model, save_path, args):
    model.eval()
    corrects, avg_loss = 0, 0
    for batch in data_iter:
        feature, target = batch.text, batch.label.data.sub_(1)
        target = autograd.Variable(target)
        if args.cuda:
            feature, target = feature.cuda(), target.cuda()

        model.hidden = model.init_hidden(args.lstm_num_layers, batch.batch_size)
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
    # test result
    if os.path.exists("./Test_Result.txt"):
        file = open("./Test_Result.txt", "a")
    else:
        file = open("./Test_Result.txt", "w")
    file.write("model " + save_path + "\n")
    file.write("Evaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n".format(avg_loss,
                                                                       accuracy,
                                                                       corrects,
                                                                       size))
    file.write("\n")
    file.close()




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

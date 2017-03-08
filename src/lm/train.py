

# Load data
def batchify(data, batch_size, gpu=False):
    num_batches = len(data) // batch_size
    data = data.narrow(0, 0, num_batches * batch_size)
    data = data.view(batch_size, -1).t().contiguous()
    if gpu:
        data = data.cuda()
    return data


def get_batch(data, i, bptt, evaluation=False, gpu=False):
    seq_len = min(bptt, len(data) - 1 - i)
    src = Variable(data[i:i+seq_len], volatile=evaluation)
    trg = Variable(data[i+1:i+seq_len+1].view(-1), volatile=evaluation)
    if gpu:
        src, trg = src.cuda(), trg.cuda()
    return src, trg


# Training code
def make_criterion(vocab_size, mask_ids=()):
    weight = torch.ones(vocab_size)
    for mask in mask_ids:
        weight[mask] = 0
    return nn.CrossEntropyLoss(weight=weight)


def repackage_hidden(h):
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


def validate_model(model, data, bptt, criterion, gpu):
    loss, hidden = 0, None
    for i in range(0, len(data) - 1, bptt):
        source, targets = get_batch(data, i, bptt, evaluation=True, gpu=gpu)
        output, hidden = model(source, hidden=hidden)
        # since loss is averaged across observations for each minibatch
        loss += len(source) * criterion(output, targets).data[0]
        hidden = repackage_hidden(hidden)
    return loss / len(data)


def train_epoch(model, data, optim, criterion, bptt, epoch, checkpoint, gpu,
                hook=0, on_hook=None):
    """
    hook: compute `on_hook` every `hook` checkpoints
    """
    epoch_loss, batch_loss, report_words = 0, 0, 0
    start = time.time()
    hidden = None

    for batch, i in enumerate(range(0, len(data) - 1, bptt)):
        model.zero_grad()
        source, targets = get_batch(data, i, bptt, gpu=gpu)
        output, hidden = model(source, hidden)
        loss = criterion(output, targets)
        hidden = repackage_hidden(hidden)
        loss.backward(), optim.step()
        # since loss is averaged across observations for each minibatch
        epoch_loss += len(source) * loss.data[0]
        batch_loss += loss.data[0]
        report_words += targets.nelement()

        if batch % checkpoint == 0 and batch > 0:
            print("Epoch %d, %5d/%5d batches; ppl: %6.2f; %3.0f tokens/s" %
                  (epoch, batch, len(data) // bptt,
                   math.exp(batch_loss / checkpoint),
                   report_words / (time.time() - start)))
            report_words = batch_loss = 0
            start = time.time()
            # call thunk every `hook` checkpoints
            if hook and (batch // checkpoint) % hook == 0:
                if on_hook is not None:
                    on_hook(batch // checkpoint)
    return epoch_loss / len(data)


def train_model(model, train, valid, test, optim, epochs, bptt,
                criterion, gpu=False, early_stop=3, checkpoint=50, hook=10):
    if gpu:
        criterion.cuda(), model.cuda()

    # hook function
    last_val_ppl, num_idle_hooks = float('inf'), 0

    def on_hook(checkpoint):
        nonlocal last_val_ppl, num_idle_hooks
        model.eval()
        valid_loss = validate_model(model, valid, bptt, criterion, gpu)
        if optim.method == 'SGD':
            last_lr, new_lr = optim.maybe_update_lr(checkpoint, valid_loss)
            if last_lr != new_lr:
                print("Decaying lr [%f -> %f]" % (last_lr, new_lr))
        if valid_loss >= last_val_ppl:  # update idle checkpoints
            num_idle_hooks += 1
        last_val_ppl = valid_loss
        if num_idle_hooks >= early_stop:  # check for early stopping
            raise u.EarlyStopping(
                "Stopping after %d idle checkpoints" % num_idle_hooks, {})
        model.train()
        print("Valid perplexity: %g" % math.exp(min(valid_loss, 100)))

    for epoch in range(1, epochs + 1):
        # train
        model.train()
        train_loss = train_epoch(
            model, train, optim, criterion, bptt, epoch, checkpoint, gpu,
            hook=hook, on_hook=on_hook)
        print("Train perplexity: %g" % math.exp(min(train_loss, 100)))
        # val
        model.eval()
        valid_loss = validate_model(model, valid, bptt, criterion, gpu)
        print("Valid perplexity: %g" % math.exp(min(valid_loss, 100)))
    # test
    test_loss = validate_model(model, test, bptt, criterion, gpu)
    print("Test perplexity: %g" % math.exp(test_loss))
    return math.exp(test_loss)

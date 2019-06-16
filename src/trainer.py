class Trainer(object):
    def __init__(self, data_loader, loss=NLLLoss(), batch_size=64, random_seed=None,checkpoint_every=100, print_every=100):
    self.data_loader=data_loader
    def train(model):
        # Turn on training mode which enables dropout.
        self.data_loader.set_train()
        model.train()
        total_loss = 0.
        batch = 0
        start_time = time.time()
        ntokens = len(self.data_loader.word_id)
        hidden = model.init_hidden(train_batch_size)
        end_flag = False
        while not end_flag:
            data, targets, end_flag= self.data_loader.get_batch()
            #data, targets = get_batch(train_data, i)
            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            hidden = repackage_hidden(hidden)
            model.zero_grad()
            output, hidden = model(data, hidden)
            loss = criterion(output.view(-1, ntokens), targets)
            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            for p in model.parameters():
                p.data.add_(-lr, p.grad.data)

            total_loss += loss.item()
            log_interval=200
            batch+=1
            if batch % log_interval == 0 and batch > 0:

                cur_loss = total_loss / log_interval
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                        'loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, batch, len(data_loader.train) // args.max_sql, lr,
                    elapsed * 1000 / log_interval, cur_loss, math.exp(cur_loss)))
                total_loss = 0
                start_time = time.time()
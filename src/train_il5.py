from load_data import load_BBN_multi_labels_dataset, load_SF_type_descriptions, average_f1_two_array_by_col, load_fasttext_multiple_word2vec_given_file, load_word2vec_to_init
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch.autograd as autograd
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch
import torch.optim as optim
from common_functions import attentive_convolution, normalize_matrix_rowwise_by_max, LSTM, multi_channel_conv_and_pool, cosine_two_matrices, torch_where

'''the following torch seed can result in the same performance'''
torch.manual_seed(400)
device = torch.device("cuda")

class Encoder(nn.Module):
    '''
    this function corresponds to the defined function in theano:
    given defined inputs, define the forward process
    '''

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size, pretrained_embeddings, batch_size):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.emb_size = embedding_dim
        self.tagset_size = tagset_size
        self.word_embeddings_bow = nn.Embedding(vocab_size, embedding_dim)
        self.word_embeddings_bow.weight = nn.Parameter(pretrained_embeddings)  # initial some embeddings

        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size = hidden_dim,
                            num_layers=1, batch_first=False)
        '''(#feature_map, kernel_size, width, height)'''
        self.conv_1 = nn.Conv2d(1, self.hidden_dim, (7, embedding_dim))
        self.conv_2 = nn.Conv2d(1, self.hidden_dim, (9, embedding_dim))

        self.conv_self = nn.Conv2d(1, self.hidden_dim, (3, embedding_dim))
        self.conv_context = nn.Conv2d(1, self.hidden_dim, (1, embedding_dim))

        self.conv_self2= nn.Conv2d(1, self.hidden_dim, (5, embedding_dim))
        self.conv_context2 = nn.Conv2d(1, self.hidden_dim, (1, embedding_dim))

        self.hidden2tag = nn.Linear(self.emb_size+4*self.hidden_dim, tagset_size)
        # self.hidden2tag = nn.Linear(self.emb_size+5*self.hidden_dim+4*self.tagset_size, tagset_size)
        self.emb2hidden = nn.Linear(embedding_dim, hidden_dim)
        # self.tagsize2tagsize = nn.Linear(2*self.hidden_dim, tagset_size)


    def forward(self, sentence, seq_lengths, mask, label_sent, label_mask):
        '''
        sentence: (batch, len)
        '''
        '''label descriptions'''
        label_embeds = self.word_embeddings_bow(label_sent) #(12, len, emb_size)
        label_reps = torch.sum(label_embeds*label_mask.unsqueeze(2), dim=1) #(12, emb_size)
        label_hidden_reps = (self.emb2hidden(label_reps)).tanh()#(12, hidden_size)


        '''neural BOW'''
        embeds_bow = self.word_embeddings_bow(sentence)
        bow = torch.sum(embeds_bow*mask.unsqueeze(2), dim=1) #(batch, emb_size)

        '''LSTM'''
        embeds_lstm = self.word_embeddings_bow(sentence)
        lstm_output = LSTM(embeds_lstm, seq_lengths, self.lstm, False)


        '''multi-channel CNN'''
        embeds_cnn = self.word_embeddings_bow(sentence)
        conv_output = multi_channel_conv_and_pool(embeds_cnn,mask, self.conv_1, self.conv_2)

        dot_cnn_dataless = (torch.mm(conv_output.reshape(2*self.batch_size,self.hidden_dim), label_hidden_reps.t()).reshape(self.batch_size, 2*self.tagset_size)).tanh()

        '''attentive convolution'''
        embeds_acnn = self.word_embeddings_bow(sentence)
        aconv_output = attentive_convolution(embeds_acnn, embeds_acnn, mask, mask, self.conv_self, self.conv_context)
        aconv_output2 = attentive_convolution(embeds_acnn, embeds_acnn, mask, mask, self.conv_self2, self.conv_context2)



        '''dataless'''
        dataless_cos = (cosine_two_matrices(bow, label_reps)).sigmoid() #(batch, 12)

        '''dataless top-30 fine grained cosine'''
        sent_side = embeds_bow*mask.unsqueeze(2) #(batch, sent_len, emb_size)
        label_side = label_embeds*label_mask.unsqueeze(2) #(12, label_len, emb_size)

        cosine_matrix = cosine_two_matrices(label_side.view(-1, self.emb_size), sent_side.view(-1, self.emb_size)) #(12*label_len, batch*sent_len)
        # print('cosine_matrix:', cosine_matrix)
        dot_prod_tensor4 = cosine_matrix.reshape(self.batch_size, sent_side.size(1), 12, label_side.size(1)).permute(0,2,3,1)#(batch, 12, label_len, sent_len)
        dot_prod_tensor3_new = dot_prod_tensor4.reshape(self.batch_size, 12, label_side.size(1)*sent_side.size(1))#(batch, 12, label_len*sent_len)
        sorted, indices = torch.sort(dot_prod_tensor3_new, descending=True)
        top_k_sorted = sorted[:,:,:50]
        dataless_top_30 = top_k_sorted.mean(dim=-1)#(batch, 12)
        # print('dataless_top_30:',top_k_sorted.var(dim=-1))

        '''combine all output representations'''
        '''len = self.emb_size+3*self.hidden_dim+4*self.tagset_size'''
        # combine_rep_batch = torch.cat([bow, lstm_output, conv_output, dataless_cos, dataless_top_30, dot_cnn_dataless, aconv_output,aconv_output2], 1)
        combine_rep_batch = torch.cat([bow, aconv_output,aconv_output2, conv_output], 1)
        tag_space = self.hidden2tag(combine_rep_batch)
        tag_prob = tag_space.sigmoid()


        return tag_prob

def build_model(embedding_dim, hidden_dim, vocab_size, tagset_size, pretrained_embeddings, lr, batch_size):
    model = Encoder(embedding_dim, hidden_dim, vocab_size, tagset_size, pretrained_embeddings, batch_size)
    model.to(device)
    '''binary cross entropy'''
    loss_function = nn.BCELoss().cuda()
    '''seems weight_decay is not good for LSTM'''
    optimizer = optim.Adagrad(model.parameters(), lr=0.01)#, weight_decay=1e-2)
    return model, loss_function, optimizer

def get_minibatches_idx(n, minibatch_size, shuffle=True):
    idx_list = np.arange(n, dtype="int32")

    if shuffle:
        np.random.RandomState(1234).shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    remain = n%minibatch_size
    # if remain > 0:
    #     minibatches.append(idx_list[-minibatch_size:])

    return zip(range(len(minibatches)), minibatches)

def get_mask(data, masks, labels, minibatch):
    data_batch = [data[x] for x in minibatch]
    mask_batch = [masks[x] for x in minibatch]
    label_batch = [labels[x] for x in minibatch]
    sentence_batch = np.array(data_batch)
    mask_batch = np.array(mask_batch)
    label_batch = np.array(label_batch)

    return sentence_batch, mask_batch, label_batch

def train(all_sentences, all_masks, all_labels, label_sent, label_mask, epoch_num, model, loss_function, optimizer):
    '''combine train set and dev set'''
    training_data_index, training_masks, training_labels = all_sentences[0]+all_sentences[1], all_masks[0]+all_masks[1], all_labels[0]+all_labels[1]
    testing_data_index, testing_masks, testing_labels = all_sentences[2], all_masks[2], all_labels[2]

    label_sent = autograd.Variable(torch.cuda.LongTensor(label_sent))
    label_mask = autograd.Variable(torch.cuda.FloatTensor(label_mask))

    iter = 0
    for epoch in range(epoch_num):

        print('current epoch: ', epoch)
        minibatches_idx = get_minibatches_idx(len(training_data_index), minibatch_size=config['batch_size'], shuffle=False)
        #random.shuffle(minibatches_idx)
        for i, minibatch in minibatches_idx:
            model.train()
            '''这里是决定一个mibatch之后才去pad, 感觉有点低效'''
            sentence_batch, mask_batch, targets_batch = get_mask(training_data_index, training_masks, training_labels, minibatch)
            sentence_batch = autograd.Variable(torch.cuda.LongTensor(sentence_batch))
            targets_batch = autograd.Variable(torch.cuda.FloatTensor(targets_batch))
            mask_batch = autograd.Variable(torch.cuda.FloatTensor(mask_batch))

            '''dim=-1好像是指the last dimension'''
            lengths_batch = mask_batch.sum(dim=-1) # is a list
            seq_lengths, seq_idx = lengths_batch.sort(0, descending=True) # a list
            seq_lengths = seq_lengths.int().data.tolist()

            sentence_batch = sentence_batch[seq_idx]
            targets_batch = targets_batch[seq_idx]
            mask_batch = mask_batch[seq_idx]
            model.zero_grad()

            tag_scores= model(sentence_batch, seq_lengths, mask_batch, label_sent, label_mask)


            '''Binary Cross Entropy'''

            temp_loss_matrix = torch_where(targets_batch[:,:-1].reshape(-1) < 1, 1.0-tag_scores[:,:-1].reshape(-1), tag_scores[:,:-1].reshape(-1))
            loss = -torch.mean(torch.log(temp_loss_matrix))
            # loss = loss_function(tag_scores[:,:-1].reshape(-1), targets_batch[:,:-1].reshape(-1))
            # l2_name_set = set(['conv_1.weight', 'conv_2.weight', 'hidden2tag.weight', 'emb2hidden.weight'])
            # reg_loss = None
            # for name, param in model.named_parameters():
            #     if name in l2_name_set:
            #         if reg_loss is None:
            #             reg_loss = 0.5 * torch.sum(param**2)
            #         else:
            #             reg_loss = reg_loss + 0.5 * param.norm(2)**2
            # loss+=reg_loss*1e-6
            loss.backward()
            optimizer.step()
            iter+=1
            if iter %200 == 0:
                print(iter, ' loss: ', loss)
                # if epoch  == 3:
                #     torch.save(model.state_dict(), 'models_'+str(iter)+'.pt')

        '''test after one epoch'''
        print('testing....')
        test(testing_data_index, testing_masks, testing_labels, model, label_sent, label_mask)
    print('train over.')


def test(training_data_index, training_masks, training_labels, model, label_sent, label_mask):

    model.eval()
    minibatches_idx = get_minibatches_idx(len(training_data_index), minibatch_size=config['batch_size'], shuffle=False)
    #random.shuffle(minibatches_idx)
    all_pred_labels = []
    all_gold_labels = []
    with torch.no_grad():
        overall_mean = -100.0
        for i, minibatch in minibatches_idx:
            '''这里是决定一个mibatch之后才去pad, 感觉有点低效'''
            sentence_batch, mask_batch, targets_batch = get_mask(training_data_index, training_masks, training_labels, minibatch)
            sentence_batch = autograd.Variable(torch.cuda.LongTensor(sentence_batch))
            # predicate_batch = autograd.Variable(torch.cuda.LongTensor(predicate_batch))
            # targets_batch = autograd.Variable(torch.cuda.FloatTensor(targets_batch))
            mask_batch = autograd.Variable(torch.cuda.FloatTensor(mask_batch))

            '''这儿好像是将一个minibatch里面的samples按照长度降序排列'''
            '''dim=-1好像是指the last dimension'''
            lengths_batch = mask_batch.sum(dim=-1) # is a list
            seq_lengths, seq_idx = lengths_batch.sort(0, descending=True) # a list
            seq_lengths = seq_lengths.int().data.tolist()
            sentence_batch = sentence_batch[seq_idx]
            mask_batch = mask_batch[seq_idx]
            '''targets_batch is array'''
            targets_batch = targets_batch[list(seq_idx.cpu().numpy())]

            tag_scores = model(sentence_batch, seq_lengths, mask_batch, label_sent, label_mask)
            '''
            baseline: 19.95/37.51 (all "1"s)

            tag_scores_bow: 23.73/44.80; 30 epoch
            tag_scores_LSTM: 24.12/41.58; 150epoch
            tag_scores_CNN: 37.28/56.26; 13 epoch
            0.5*(dataless_cos+hidden_output): 22.51/43.43: 30 epoch
            0.5*(dataless_top_30+hidden_output): 16.98/35.43: 30 epoch
            dataless_top_30: 17.39/34.79
            dot_cnn_dataless: 29.16/47.80
            attentive_convolution_width_3: 36.73/52.78
            attentive_convolution_width_3&5: 38.11/55.90
            attConv+tag_scores_CNN+BOW: 38.01/59.01


            ensemble_NN_scores: 27.72/49.21; 150 epoch, adagrad, combine train+dev; add remain train; shuffle

            joint rep: 29.10/44.66; 30 epoch
            joint rep: 36.32/57.64; 5 epoch; > mean
            joint+aconv3&5: 34.90/55.37; 5 epoch
            '''

            tag_scores_2_array = tag_scores.cpu().numpy()
            mean = np.mean(tag_scores_2_array)
            pred_labels = np.where(tag_scores_2_array > mean, 1, 0) # 17.10/ 33.5
            # pred_labels = np.where(tag_scores_2_array > np.min(tag_scores_2_array)+0.7*(np.max(tag_scores_2_array)-np.min(tag_scores_2_array)), 1, 0)

            '''choose one prediction'''
            all_pred_labels.append(pred_labels)
            all_gold_labels.append(targets_batch)

    all_pred_labels = np.concatenate(all_pred_labels)
    all_gold_labels = np.concatenate(all_gold_labels)

    test_mean_f1, test_weight_f1 =average_f1_two_array_by_col(all_pred_labels, all_gold_labels)
    print('test over, test_mean_f1:', test_mean_f1, 'test_weight_f1:', test_weight_f1)


if __name__ == '__main__':

    config = {"emb_size": 40, "hidden_size": 300, 'epoch_num': 10, 'lr': 0.0003, 'batch_size': 5,
              'partial_rate': 0.5, 'maxSentLen': 100, 'describ_max_len':20, 'type_size':12}

    all_sentences, all_masks, all_labels, word2id=load_BBN_multi_labels_dataset(maxlen=config['maxSentLen'])
    label_sent, label_mask = load_SF_type_descriptions(word2id, config['type_size'], config['describ_max_len'])
    emb_root = '/scratch/wyin3/dickens_save_dataset/LORELEI/multi-lingual-emb/'
    print('loading bilingual embeddings....')
    word2vec=load_fasttext_multiple_word2vec_given_file([emb_root+'IL5-cca-wiki-lorelei-d40.eng.vec',emb_root+'IL5-cca-wiki-lorelei-d40.IL5.vec'], 40)

    vocab_size =  len(word2id)+1
    rand_values=np.random.RandomState(1234).normal(0.0, 0.01, (vocab_size, config['emb_size']))   #generate a matrix by Gaussian distribution
    rand_values[0]=np.array(np.zeros(config['emb_size']),dtype=np.float32)
    id2word = {y:x for x,y in word2id.items()}
    rand_values=load_word2vec_to_init(rand_values, id2word, word2vec)
    embeddings = torch.Tensor(rand_values)

    print("build model...")
    model, loss_function, optimizer = build_model(config['emb_size'], config['hidden_size'], vocab_size,
                                                  12, embeddings, config['lr'], config['batch_size'])
    print("training...")
    # train_start = time.time()
    train(all_sentences, all_masks, all_labels, label_sent, label_mask, config['epoch_num'], model, loss_function, optimizer)



    '''
    1, whether the embeddings are trained
    2, loss function和theaao不同
    '''

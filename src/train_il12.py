from load_data import load_BBN_multi_labels_dataset, load_il_groundtruth_as_testset, load_official_testData_il_and_MT_multiple_testfiles, load_trainingData_types_plus_others,load_trainingData_types,load_SF_type_descriptions, average_f1_two_array_by_col, load_fasttext_multiple_word2vec_given_file, load_word2vec_to_init
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch.autograd as autograd
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch
import torch.optim as optim
from common_functions import attentive_convolution, normalize_matrix_rowwise_by_max, LSTM, multi_channel_conv_and_pool, cosine_two_matrices, torch_where
'''head files for using pretrained bert'''
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from preprocess_IL3_Uyghur import recover_pytorch_idmatrix_2_text
from bert_common_functions import sent_to_embedding, sent_to_embedding_last4
from preprocess_common import generate_2019_official_output, validate_output_schema

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

        if use_bert:
            self.bert_hidden2tag = nn.Linear(768*4, tagset_size)
            self.hidden2tag = nn.Linear(self.emb_size+4*self.hidden_dim, tagset_size)
            self.bert_hidden2tag_task2 = nn.Linear(768*4, 16)
            self.hidden2tag_task2 = nn.Linear(self.emb_size+4*self.hidden_dim, 16)
        else:
            self.hidden2tag = nn.Linear(self.emb_size+4*self.hidden_dim, tagset_size)
            self.hidden2tag_task2 = nn.Linear(self.emb_size+4*self.hidden_dim, 16)
        self.emb2hidden = nn.Linear(embedding_dim, hidden_dim)
        # self.tagsize2tagsize = nn.Linear(2*self.hidden_dim, tagset_size)


    def forward(self, sentence, seq_lengths, mask, label_sent, label_mask, bert_rep_batch):
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

        if use_bert:
            combine_rep_batch = torch.cat([bert_rep_batch], 1)
        else:
            combine_rep_batch = torch.cat([torch.cat([bow, aconv_output,aconv_output2, conv_output], 1).tanh()],1)

        if not use_bert:
            tag_space = self.hidden2tag(combine_rep_batch)
            tag_prob = tag_space.sigmoid()
            '''task 2'''
            tag_space_task2 = self.hidden2tag_task2(combine_rep_batch)
            tag_prob_task2 = nn.Softmax(dim=1)(tag_space_task2.reshape(4*self.batch_size,4)) #(4*batch, 4)
            '''combine system2018 and bert'''
        else:
            bert_features = bert_rep_batch
            system2018_features = torch.cat([torch.cat([bow, aconv_output,aconv_output2, conv_output], 1).tanh()],1)
            ratio=0.5
            tag_prob = self.hidden2tag(system2018_features).sigmoid()*ratio+self.bert_hidden2tag(bert_features).sigmoid()*(1.0-ratio)
            tag_prob_task2 = self.hidden2tag_task2(system2018_features).sigmoid()*ratio+self.bert_hidden2tag_task2(bert_features).sigmoid()*(1.0-ratio)

        return tag_prob, tag_prob_task2

def build_model(embedding_dim, hidden_dim, vocab_size, tagset_size, pretrained_embeddings, lr, batch_size):
    model = Encoder(embedding_dim, hidden_dim, vocab_size, tagset_size, pretrained_embeddings, batch_size)
    model.to(device)
    '''binary cross entropy'''
    loss_function = nn.BCELoss().cuda()
    '''seems weight_decay is not good for LSTM'''
    optimizer = optim.Adagrad(model.parameters(), lr=0.001)#, weight_decay=1e-2)
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
    if remain > 0:
        minibatches.append(idx_list[-minibatch_size:])

    return minibatches#zip(range(len(minibatches)), minibatches)

def get_mask(data, masks, labels, minibatch):
    data_batch = [data[x] for x in minibatch]
    mask_batch = [masks[x] for x in minibatch]
    label_batch = [labels[x] for x in minibatch]
    sentence_batch = np.array(data_batch)
    mask_batch = np.array(mask_batch)
    label_batch = np.array(label_batch)

    return sentence_batch, mask_batch, label_batch

def get_mask_test(data, masks, minibatch):
    data_batch = [data[x] for x in minibatch]
    mask_batch = [masks[x] for x in minibatch]
    # label_batch = [labels[x] for x in minibatch]
    sentence_batch = np.array(data_batch)
    mask_batch = np.array(mask_batch)
    # label_batch = np.array(label_batch)

    return sentence_batch, mask_batch

def get_mask_task2(data, masks, labels, other_labels, minibatch):
    data_batch = [data[x] for x in minibatch]
    mask_batch = [masks[x] for x in minibatch]
    label_batch = [labels[x] for x in minibatch]
    other_label_batch = [other_labels[x] for x in minibatch]
    sentence_batch = np.array(data_batch)
    mask_batch = np.array(mask_batch)
    label_batch = np.array(label_batch)
    other_label_batch = np.array(other_label_batch)

    return sentence_batch, mask_batch, label_batch, other_label_batch

# def get_mask_test(data, masks, lines, minibatch):
#     data_batch = [data[x] for x in minibatch]
#     mask_batch = [masks[x] for x in minibatch]
#     lines = [lines[x] for x in minibatch]
#
#     sentence_batch = np.array(data_batch)
#     mask_batch = np.array(mask_batch)
#
#     return sentence_batch, mask_batch, lines

def train(task1_data,task2_data,test_data, label_sent, label_mask, id2word, epoch_num, model, loss_function, optimizer):
    '''combine train set and dev set'''
    '''
    task1_data,task2_data,test_data,
    '''
    training_data_index, training_masks, training_labels = task1_data
    training_data_task2_index, training_task2_masks, training_task2_labels, train_task2_other_labels = task2_data
    testing_data_index, testing_masks, test_lines = test_data

    label_sent = autograd.Variable(torch.cuda.LongTensor(label_sent))
    label_mask = autograd.Variable(torch.cuda.FloatTensor(label_mask))

    print("training...")
    iter = 0
    for epoch in range(epoch_num):

        print('current epoch: ', epoch)
        minibatches_idx = get_minibatches_idx(len(training_data_index), minibatch_size=config['batch_size'], shuffle=True)
        minibatches_idx_task2 = get_minibatches_idx(len(training_data_task2_index), minibatch_size=config['batch_size'], shuffle=True)
        for i, minibatch in enumerate(minibatches_idx):
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

            '''Bert'''

            # sentence_numpy = sentence_batch.cpu().array()
            # bert_rep_batch = []
            # for i in range(config['batch_size']):
            #     sent_str = ''
            #     for id in list(sentence_numpy[i]):
            #         if id !=0:
            #             sent_str+=id2word.get(id)+' '
            sent_list = recover_pytorch_idmatrix_2_text(sentence_batch, id2word)
            bert_rep_batch = []
            if use_bert:
                for sent in sent_list:
                    bert_rep = sent_to_embedding_last4(sent, bert_tokenizer, bert_model, True)
                    bert_rep_batch.append(bert_rep.reshape(1,-1))
                bert_rep_batch = torch.cat(bert_rep_batch, 0) #(batch, 768)

            tag_scores,_= model(sentence_batch, seq_lengths, mask_batch, label_sent, label_mask, bert_rep_batch)


            '''Binary Cross Entropy'''

            temp_loss_matrix = torch_where(targets_batch[:,:-1].reshape(-1) < 1, 1.0-tag_scores[:,:-1].reshape(-1), tag_scores[:,:-1].reshape(-1))
            loss = -torch.mean(torch.log(temp_loss_matrix))
            loss.backward()
            optimizer.step()

            '''task2'''
            if i < len(minibatches_idx_task2):
                model.train()
                minibatch_task2 = minibatches_idx_task2[i]
                '''这里是决定一个mibatch之后才去pad, 感觉有点低效'''
                sentence_batch, mask_batch, targets_batch, others_batch = get_mask_task2(training_data_task2_index, training_task2_masks, training_task2_labels, train_task2_other_labels, minibatch_task2)
                sentence_batch = autograd.Variable(torch.cuda.LongTensor(sentence_batch))
                targets_batch = autograd.Variable(torch.cuda.FloatTensor(targets_batch))
                mask_batch = autograd.Variable(torch.cuda.FloatTensor(mask_batch))
                others_batch = autograd.Variable(torch.cuda.LongTensor(others_batch))

                '''dim=-1好像是指the last dimension'''
                lengths_batch = mask_batch.sum(dim=-1) # is a list
                seq_lengths, seq_idx = lengths_batch.sort(0, descending=True) # a list
                seq_lengths = seq_lengths.int().data.tolist()

                sentence_batch = sentence_batch[seq_idx]
                targets_batch = targets_batch[seq_idx]
                mask_batch = mask_batch[seq_idx]
                others_batch = others_batch[seq_idx]
                model.zero_grad()

                sent_list = recover_pytorch_idmatrix_2_text(sentence_batch, id2word)
                bert_rep_batch = []
                if use_bert:
                    for sent in sent_list:
                        bert_rep = sent_to_embedding_last4(sent, bert_tokenizer, bert_model, True)
                        bert_rep_batch.append(bert_rep.reshape(1,-1))
                    bert_rep_batch = torch.cat(bert_rep_batch, 0) #(batch, 768)
                tag_scores, tag_scores_task2= model(sentence_batch, seq_lengths, mask_batch, label_sent, label_mask, bert_rep_batch)
                # print('tag_scores_task2:',tag_scores_task2)

                '''Binary Cross Entropy'''
                temp_loss_matrix = torch_where(targets_batch[:,:-1].reshape(-1) < 1, 1.0-tag_scores[:,:-1].reshape(-1), tag_scores[:,:-1].reshape(-1))
                loss_task1 = -torch.mean(torch.log(temp_loss_matrix))
                '''task2 loss'''
                other_label_scores = tag_scores_task2.index_select(1,others_batch.view(-1))
                loss_task2 = -torch.mean(torch.log(other_label_scores))
                # print('loss_task1:',loss_task1)
                # print('loss_task2:', loss_task2)
                loss = loss_task1+loss_task2
                loss.backward()
                optimizer.step()


            iter+=1
            if iter %20 == 0:
                print(iter, ' loss: ', loss)
                # if epoch  == 3:
                #     torch.save(model.state_dict(), 'models_'+str(iter)+'.pt')

                '''test after one epoch'''

        # torch.save(model.state_dict(), save_model_path)
        # print('model saved succeed. train over')
        # return
        # else:
        if (epoch+1)%10==0:
            print('testing....')
            test(testing_data_index, testing_masks, test_lines, model, label_sent, label_mask, id2word)





def test(training_data_index_seq, training_masks_seq, test_lines_seq, model, label_sent, label_mask, id2word):
    # model.load_state_dict(torch.load(save_model_path))
    model.eval()
    '''这里的minibatches_idx已经考虑了remain的样本'''
    data_id = 0
    for training_data_index, training_masks, test_lines in zip(training_data_index_seq, training_masks_seq, test_lines_seq):

        minibatches_idx = get_minibatches_idx(len(training_data_index), minibatch_size=config['batch_size'], shuffle=False)
        n_test_remain = len(training_data_index)%config['batch_size']
        pred_types = []
        pred_confs = []
        pred_others = []
        # Text_Lines = []
        with torch.no_grad():
            overall_mean = -100.0
            for i, minibatch in enumerate(minibatches_idx):
                '''这里是决定一个mibatch之后才去pad, 感觉有点低效'''
                sentence_batch, mask_batch= get_mask_test(training_data_index, training_masks, minibatch)
                sentence_batch = autograd.Variable(torch.cuda.LongTensor(sentence_batch))
                mask_batch = autograd.Variable(torch.cuda.FloatTensor(mask_batch))

                '''这儿好像是将一个minibatch里面的samples按照长度降序排列'''
                '''dim=-1好像是指the last dimension'''
                lengths_batch = mask_batch.sum(dim=-1) # is a list
                seq_lengths, seq_idx = lengths_batch.sort(0, descending=True) # a list
                seq_lengths = seq_lengths.int().data.tolist()

                sentence_batch = sentence_batch[seq_idx]
                mask_batch = mask_batch[seq_idx]
                '''把reorder的seq_idx排回原来的样子'''
                seq_idx_2_list = seq_idx.int().data.tolist()
                return_map = {val:i for i, val in enumerate(seq_idx_2_list)}
                recover_seq_idx =[return_map[i] for i in range(len(seq_idx_2_list))]
                recover_seq_idx=autograd.Variable(torch.cuda.LongTensor(np.array(recover_seq_idx)))



                # reordered_text_lines = [text_lines_batch[id] for id in seq_idx.int().data.tolist()]
                '''targets_batch is array'''
                # targets_batch = targets_batch[list(seq_idx.cpu().numpy())]
                sent_list = recover_pytorch_idmatrix_2_text(sentence_batch, id2word)
                bert_rep_batch = []
                if use_bert:
                    for sent in sent_list:
                        bert_rep = sent_to_embedding_last4(sent, bert_tokenizer, bert_model, True)
                        bert_rep_batch.append(bert_rep.reshape(1,-1))
                    bert_rep_batch = torch.cat(bert_rep_batch, 0) #(batch, 768)

                tag_scores, tag_scores_task2 = model(sentence_batch, seq_lengths, mask_batch, label_sent, label_mask, bert_rep_batch)
                '''recover the order'''
                tag_scores = tag_scores[recover_seq_idx]
                tag_scores_task2 = (tag_scores_task2.reshape(len(minibatch),4,4))[recover_seq_idx]
                # print('tag_scores_task2:',tag_scores_task2)
                # print('recover_seq_idx:',recover_seq_idx)
                # tag_scores_task2 = tag_scores_task2[recover_seq_idx]
                # print('tag_scores_task2:',tag_scores_task2)
                # exit(0)

                tag_scores_2_array = tag_scores.cpu().numpy()
                mean = np.mean(tag_scores_2_array)
                pred_labels = np.where(tag_scores_2_array > mean, 1, 0) # 17.10/ 33.5
                pred_conf = tag_scores_2_array
                '''recover the order'''
                pred_other = tag_scores_task2.cpu().numpy()

                if i < len(minibatches_idx)-1:
                    pred_types.append(pred_labels)
                    pred_confs.append(pred_conf)
                    pred_others.append(pred_other)
                    # Text_Lines+=text_lines_batch
                else:
                    pred_types.append(pred_labels[-n_test_remain:])
                    pred_confs.append(pred_conf[-n_test_remain:])
                    pred_others.append(pred_other[-n_test_remain:])
                    # Text_Lines+=text_lines_batch[-n_test_remain:]

        pred_types = np.concatenate(pred_types, axis=0)
        pred_confs = np.concatenate(pred_confs, axis=0)
        pred_others = np.concatenate(pred_others, axis=0)

        # test_mean_f1, test_weight_f1 =average_f1_two_array_by_col(pred_types, np.array(testing_labels))
        # print('test over, test_mean_f1:', test_mean_f1, 'test_weight_f1:', test_weight_f1)
        '''starting generate official output'''

        min_mean_frame = 100.0
        output_file_path = output_file_head+output_file_path_codes[data_id]+'.json'
        print('generating ...', output_file_path)
        mean_frame = generate_2019_official_output(test_lines, output_file_path, pred_types, pred_confs, pred_others)
        if mean_frame < min_mean_frame:
            min_mean_frame = mean_frame
        print('\t\t\t test  over, min_mean_frame:', min_mean_frame)
        validate_output_schema(output_file_path, 'LoReHLT19-schema_V1.json')
        data_id+=1

if __name__ == '__main__':

    config = {"emb_size": 300, "hidden_size": 300, 'epoch_num': 40, 'lr': 0.0003, 'batch_size': 50,
              'partial_rate': 0.5, 'maxSentLen': 100, 'describ_max_len':20, 'type_size':12}

    # command_list = [0,0,0]
    # command_list = [0,0,1]
    # command_list = [0,1,0] # not useful
    # command_list = [0,1,1] # not useful
    # command_list = [1,0,0] #start use bert
    command_list = [1,0,1]
    # command_list = [1,1,0]
    # command_list = [1,1,1]

    '''2 versions, use bert?'''
    use_bert=False if command_list[0]==0 else True
    '''2 versions, which bert?'''
    bert_path_id = command_list[1]
    '''2 versions, which bilingual emb?'''
    emb_path_id = command_list[2]

    bert_paths = ['/shared/experiments/kkarthi/Final_Bert_Models/il12/il12_1M',
    '/shared/experiments/kkarthi/Final_Bert_Models/il12/il12_V1_1M']

    emb_root = '/scratch/wyin3/dickens_save_dataset/LORELEI/il12/bilingual_emb/'
    emb_paths = [[emb_root+'cleaned-100k-il12longerw2v-multicca.d300.eng.txt',
                emb_root+'cleaned-100k-il12longerw2v-multicca.d300.il12longerw2v.txt'],
                [emb_root+'cleaned-100k-il12longerw2vfulldict-multicca.d300.eng.txt',
                emb_root+'cleaned-100k-il12longerw2vfulldict-multicca.d300.il12longerw2vfulldict.txt']]


    '''test setup'''
    test_file_paths = [
    # '/scratch/wyin3/dickens_save_dataset/LORELEI/il12/il12-setE-as-test-input_edl0_filtered_w2.txt',
    '/scratch/wyin3/dickens_save_dataset/LORELEI/il12/il12-setE-as-test-input_edl1_filtered_w2.txt',
    ]
    '''
    'edl0.type0','edl0.type1','edl0.type2','edl1.type0','edl1.type1','edl1.type2','edl2.type0','edl2.type1','edl2.type2'
    '''
    output_file_path_codes = ['.'.join(list(map(str, command_list)))+'.'+i for i in ['edl1.type0','edl1.type1','edl1.type2']]
    output_file_head = '/scratch/wyin3/dickens_save_dataset/LORELEI/il12/sf_output_ch2/il12_system_output_'
    '''setup over'''
    if use_bert:
        print('loading bert...')
        bert_path = bert_paths[bert_path_id]
        bert_tokenizer = BertTokenizer.from_pretrained(bert_path)#load the vocab.txt file
        bert_model = BertModel.from_pretrained(bert_path)
        bert_model.eval()
        bert_model.to('cuda')
    word2id={}

    train_p1_sents, train_p1_masks, train_p1_labels,word2id = load_trainingData_types(word2id, config['maxSentLen'])
    train_p2_sents, train_p2_masks, train_p2_labels, train_p2_other_labels,word2id = load_trainingData_types_plus_others(word2id, config['maxSentLen'])
    test_sents, test_masks, test_lines, word2id = load_official_testData_il_and_MT_multiple_testfiles(word2id, config['maxSentLen'], test_file_paths)
    # test_sents, test_masks, test_labels, word2id = load_il_groundtruth_as_testset(word2id, config['maxSentLen'], test_file_path)
    '''task1: combine datasets from two sub-tasks'''
    train_sents = train_p1_sents+train_p2_sents
    train_masks = train_p1_masks+train_p2_masks
    train_labels = train_p1_labels+train_p2_labels

    # all_sentences, all_masks, all_labels, word2id=load_BBN_multi_labels_dataset(maxlen=config['maxSentLen'])
    label_sent, label_mask = load_SF_type_descriptions(word2id, config['type_size'], config['describ_max_len'])
    # emb_root = '/scratch/wyin3/dickens_save_dataset/LORELEI/multi-lingual-emb/Uyghur/'
    # print('loading bilingual embeddings....')
    # word2vec=load_fasttext_multiple_word2vec_given_file([emb_root+'vectors-en.txt',emb_root+'vectors-ug.txt'], 300)
    # emb_root = '/scratch/wyin3/dickens_save_dataset/LORELEI/il11/bilingual_emb/'
    # print('loading bilingual embeddings....')
    word2vec=load_fasttext_multiple_word2vec_given_file(emb_paths[emb_path_id], 300)


    vocab_size =  len(word2id)+1
    rand_values=np.random.RandomState(1234).normal(0.0, 0.01, (vocab_size, config['emb_size']))   #generate a matrix by Gaussian distribution
    rand_values[0]=np.array(np.zeros(config['emb_size']),dtype=np.float32)
    id2word = {y:x for x,y in word2id.items()}
    rand_values=load_word2vec_to_init(rand_values, id2word, word2vec)
    embeddings = torch.Tensor(rand_values)

    print("build model...")
    model, loss_function, optimizer = build_model(config['emb_size'], config['hidden_size'], vocab_size,
                                                  12, embeddings, config['lr'], config['batch_size'])

    # train_start = time.time()

    task1_data=(train_sents,train_masks,train_labels)
    task2_data=(train_p2_sents, train_p2_masks, train_p2_labels, train_p2_other_labels)
    test_data=(test_sents, test_masks, test_lines)
    # train(all_sentences, all_masks, all_labels, label_sent, label_mask, config['epoch_num'], model, loss_function, optimizer)
    train(task1_data,task2_data,test_data, label_sent, label_mask, id2word, config['epoch_num'], model, loss_function, optimizer)


    '''
    1, whether the embeddings are trained
    2, loss function和theaao不同
    '''

import codecs
import numpy as np
from collections import defaultdict, Counter
import json

def load_word2vec_to_init(rand_values, ivocab, word2vec):
    fail=0
    for id, word in ivocab.items():
        emb=word2vec.get(word)
        if emb is not None:
            rand_values[id]=np.array(emb)
        else:
            # print(word)
            fail+=1
    print('==> use word2vec initialization over...fail ', fail)
    return rand_values

def load_fasttext_multiple_word2vec_given_file(filepath_list, dim):
    word2vec = {}
    for fil in filepath_list:
        print(fil, "==> loading 300d word2vec")
    #     with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/glove/glove.6B." + str(dim) + "d.txt")) as f:
        f=codecs.open(fil, 'r', 'utf-8',errors='ignore')#glove.6B.300d.txt, word2vec_words_300d.txt, glove.840B.300d.txt
        line_co = 0
        for line in f:
            l = line.split()
            # print l
            if len(l)==dim+1:
                value_list = list(map(float, l[1:]))
                # norm = LA.norm(np.asarray(value_list))
                # word2vec[l[0]] = [value/norm for value in value_list]
                word2vec[l[0]] = value_list
                line_co+=1
                # if line_co % 500000:
                #     print 'line_co:', line_co
                # if line_co > 10000:
                #     break

        print("==> word2vec is loaded over")
    return word2vec

def transfer_wordlist_2_idlist_with_maxlen(token_list, vocab_map, maxlen):
    '''
    From such as ['i', 'love', 'Munich'] to idlist [23, 129, 34], if maxlen is 5, then pad two zero in the left side, becoming [0, 0, 23, 129, 34]
    '''
    idlist=[]
    for word in token_list:

        id=vocab_map.get(word)
        if id is None: # if word was not in the vocabulary
            id=len(vocab_map)+1  # id of true words starts from 1, leaving 0 to "pad id"
            vocab_map[word]=id
        idlist.append(id)

    mask_list=[1.0]*len(idlist) # mask is used to indicate each word is a true word or a pad word
    pad_size=maxlen-len(idlist)
    if pad_size>0:
        idlist=idlist+[0]*pad_size
        mask_list=mask_list+[0.0]*pad_size
    else: # if actual sentence len is longer than the maxlen, truncate
        idlist=idlist[:maxlen]
        mask_list=mask_list[:maxlen]
    return idlist, mask_list

def transfer_wordlist_2_idlist_with_maxlen_in_Test(token_list, vocab_map, maxlen):
    '''
    From such as ['i', 'love', 'Munich'] to idlist [23, 129, 34], if maxlen is 5, then pad two zero in the left side, becoming [0, 0, 23, 129, 34]
    '''
    idlist=[]
    for word in token_list:

        id=vocab_map.get(word)
        if id is not None: # if word was not in the vocabulary
            idlist.append(id)

    mask_list=[1.0]*len(idlist) # mask is used to indicate each word is a true word or a pad word
    pad_size=maxlen-len(idlist)
    if pad_size>0:
        idlist=idlist+[0]*pad_size
        mask_list=mask_list+[0.0]*pad_size
    else: # if actual sentence len is longer than the maxlen, truncate
        idlist=idlist[:maxlen]
        mask_list=mask_list[:maxlen]
    return idlist, mask_list

def load_trainingData_types(word2id, maxlen):
    BBN_path = '/home1/w/wenpeng/dataset/LORELEI/trainingdata/'
    files = [
    'full_BBN_multi.txt',
    # 'SF-BBN-Mark-split/full_BBN_multi_translated_2_il9.txt'
    # 'SF-BBN-Mark-split/full_BBN_multi_translated_2_il10.txt'
    # '2019_new_data/reliefweb_2_sf.txt'
    'il9_sf_gold.txt',
    'il10_sf_gold.txt',
    'il5_translated_seg_level_as_training_all_fields.txt',
    'il3_sf_gold.txt',
    'Mandarin_sf_gold.txt'
    # 'il6_sf_gold.txt', # il6 does not help
    # ,'NYT-Mark-top10-id-label-text.txt'
    # ,'hindi_labeled_as_training_seg_level.txt'
    # ,'ReliefWeb_subset_id_label_text.txt'
    ]
    label_id_re_map = {0:0,1:1, 2:2,3:3,4:4,5:5,6:6,7:7,8:8,9:11,10:9,11:10}
    # label_id_re_map = {0:0,1:1, 2:2,3:3,4:4,5:5,6:6,7:7,8:8,9:9,10:10,11:11}
    all_sentences=[]
    all_masks=[]
    all_labels=[]
    for fil in files:
        print('loading file:', BBN_path+fil, '...')
        size = 0
        readfile=codecs.open(BBN_path+fil, 'r', 'utf-8')
        for line in readfile:
            parts=line.strip().split('\t') #lowercase all tokens, as we guess this is not important for sentiment task
            # if len(parts)==3:
            label=[0]*12
            for label_id in parts[0].strip().split():  # keep label be 0 or 1
                label[label_id_re_map[int(label_id)]] =1
            # print 'parts:',parts
            sentence_wordlist=parts[2].strip().split()#clean_text_to_wordlist(parts[2].strip())
            # if fil == 'il6_sf_gold.txt' and len(sentence_wordlist) < 20:
            #     continue
            all_labels.append(label)
            sent_idlist, sent_masklist=transfer_wordlist_2_idlist_with_maxlen(sentence_wordlist, word2id, maxlen)
            all_sentences.append(sent_idlist)
            all_masks.append(sent_masklist)
            size+=1
            # if size == 50000:
            #     break
        print( '\t\t\t size:', size)
    print( 'dataset loaded over, totally ', len(all_labels), 'instances, and ', len(word2id), 'words')
    return all_sentences, all_masks, all_labels,word2id


def load_trainingData_types_inSequence(word2id, maxlen):
    '''we found it does not help promote performance'''
    BBN_path = '/scratch/wyin3/dickens_save_dataset/LORELEI/'
    files = [

    # 'SF-BBN-Mark-split/full_BBN_multi_translated_2_il9.txt',
    # 'SF-BBN-Mark-split/full_BBN_multi_translated_2_il10.txt'
    '2019_new_data/reliefweb_2_sf.txt',
    # ,'il9/il9-test.txt'
    # ,'il10/il10-test.txt'
    # ,'NYT-Mark-top10-id-label-text.txt'
    # ,'hindi_labeled_as_training_seg_level.txt'
    # ,'ReliefWeb_subset_id_label_text.txt'
    'SF-BBN-Mark-split/full_BBN_multi.txt'
    ]
    label_id_re_map = {0:0,1:1, 2:2,3:3,4:4,5:5,6:6,7:7,8:8,9:11,10:9,11:10}
    # label_id_re_map = {0:0,1:1, 2:2,3:3,4:4,5:5,6:6,7:7,8:8,9:9,10:10,11:11}
    all_sentences=[]
    all_masks=[]
    all_labels=[]
    all_size = 0
    for fil in files:
        fil_sentences=[]
        fil_masks=[]
        fil_labels=[]
        print('loading file:', BBN_path+fil, '...')
        size = 0
        readfile=codecs.open(BBN_path+fil, 'r', 'utf-8')
        for line in readfile:
            parts=line.strip().split('\t') #lowercase all tokens, as we guess this is not important for sentiment task
            if len(parts)==3:
                label=[0]*12
                for label_id in parts[0].strip().split():  # keep label be 0 or 1
                    label[label_id_re_map[int(label_id)]] =1
                # print 'parts:',parts
                sentence_wordlist=parts[2].strip().split()#clean_text_to_wordlist(parts[2].strip())
                fil_labels.append(label)
                sent_idlist, sent_masklist=transfer_wordlist_2_idlist_with_maxlen(sentence_wordlist, word2id, maxlen)
                fil_sentences.append(sent_idlist)
                fil_masks.append(sent_masklist)
                size+=1
                if size == 10000:
                    break
        print( '\t\t\t size:', size)
        all_sentences.append(fil_sentences)
        all_masks.append(fil_masks)
        all_labels.append(fil_labels)
        all_size+=size
    print( 'dataset loaded over, totally ', all_size, 'instances, and ', len(word2id), 'words')
    return all_sentences, all_masks, all_labels,word2id


def load_trainingData_types_plus_others(word2id, maxlen):
    root="/home1/w/wenpeng/dataset/LORELEI/trainingdata/"
    # files=['SF-BBN-Mark-split/train.mark.multi.12labels.txt', 'SF-BBN-Mark-split/dev.mark.multi.12labels.txt', 'il5_labeled_as_training_seg_level.txt']
    # files=['il5_translated_seg_level_as_training_all_fields_w1.txt'] #'il5_translated_seg_level_as_training_all_fields.txt'
    files=[
    'il5_translated_seg_level_as_training_all_fields.txt'
    # 'il6_translated_seg_level_as_training_all_fields_w1.txt'
    # 'il5_translated_seg_level_as_training_all_fields_w1.txt'
    ]
    label_id_re_map = {0:0,1:1, 2:2,3:3,4:4,5:5,6:6,7:7,8:8,9:11,10:9,11:10}
    all_sentences=[]
    all_masks=[]
    all_labels=[]
    all_other_labels = []
    for fil in files:
        print('loading file:', root+fil, '...')

        readfile=codecs.open(root+fil, 'r', 'utf-8')
        size = 0
        for line in readfile:
            parts=line.strip().split('\t') #lowercase all tokens, as we guess this is not important for sentiment task
            label=[0]*12
            for label_id in parts[0].strip().split():  # keep label be 0 or 1
                label[label_id_re_map[int(label_id)]] =1
            sentence_wordlist=parts[2].strip().split()#clean_text_to_wordlist(parts[2].strip())
            sent_idlist, sent_masklist=transfer_wordlist_2_idlist_with_maxlen(sentence_wordlist, word2id, maxlen)
            all_sentences.append(sent_idlist)
            all_masks.append(sent_masklist)
            all_labels.append(label)
            '''load other labels'''
            all_other_labels.append([int(x) for x in parts[3].strip().split()])
            size+=1

        print('\t\t\t size:', size)
    assert len(all_other_labels) == len(all_labels)
    print('dataset loaded over, totally ', len(all_labels), 'instances, and ', len(word2id), 'words')
    return all_sentences, all_masks, all_labels, all_other_labels,word2id

def load_BBN_multi_labels_dataset(maxlen=40):
    root="/scratch/wyin3/dickens_save_dataset/LORELEI/"
    # files=['SF-BBN-Mark-split/train.mark.multi.12labels.txt', 'SF-BBN-Mark-split/dev.mark.multi.12labels.txt', 'il5_labeled_as_training_seg_level.txt']
    files=['SF-BBN-Mark-split/full_BBN_multi.txt', 'il5_translated_seg_level_as_training_all_fields.txt', 'il5_labeled_as_training_seg_level.txt']
    label_id_re_map = {0:0,1:1, 2:2,3:3,4:4,5:5,6:6,7:7,8:8,9:11,10:9,11:10}
    word2id={}  # store vocabulary, each word map to a id
    all_sentences=[]
    all_masks=[]
    all_labels=[]
    for i in range(len(files)):
        print('loading file:', root+files[i], '...')

        sents=[]
        sents_masks=[]
        labels=[]
        readfile=codecs.open(root+files[i], 'r', 'utf-8')
        for line in readfile:
            parts=line.strip().split('\t') #lowercase all tokens, as we guess this is not important for sentiment task
            label=[0]*12
            for label_id in parts[0].strip().split():  # keep label be 0 or 1
                label[label_id_re_map[int(label_id)]] =1
            sentence_wordlist=parts[2].strip().split()#clean_text_to_wordlist(parts[2].strip())

            labels.append(label)
            sent_idlist, sent_masklist=transfer_wordlist_2_idlist_with_maxlen(sentence_wordlist, word2id, maxlen)
            sents.append(sent_idlist)
            sents_masks.append(sent_masklist)
        all_sentences.append(sents)
        all_masks.append(sents_masks)
        all_labels.append(labels)
        print('\t\t\t size:', len(labels))
        print('dataset loaded over, totally ', len(word2id), 'words')
        # print('label size:', np.sum(np.array(labels),axis=0))
    # exit(0)
    return all_sentences, all_masks, all_labels, word2id

def load_il_groundtruth_as_testset(word2id, maxlen, filename):
    root="/home/wyin3/LORELEI/2019/retest/"
    # files=['SF-BBN-Mark-split/train.mark.multi.12labels.txt', 'SF-BBN-Mark-split/dev.mark.multi.12labels.txt', 'il5_labeled_as_training_seg_level.txt']
    files=['SF-BBN-Mark-split/full_BBN_multi.txt', 'il5_translated_seg_level_as_training_all_fields.txt', 'il3_uyghur_labeled_as_training_seg_level.txt']
    '''we do not need label_id_re_map anymore, since il3 considered this already'''
    # label_id_re_map = {0:0,1:1, 2:2,3:3,4:4,5:5,6:6,7:7,8:8,9:11,10:9,11:10}
    # word2id={}  # store vocabulary, each word map to a id


    print('loading file:', filename, '...')

    sents=[]
    sents_masks=[]
    labels=[]
    readfile=codecs.open(filename, 'r', 'utf-8')
    for line in readfile:
        parts=line.strip().split('\t') #lowercase all tokens, as we guess this is not important for sentiment task
        label=[0]*12
        for label_id in parts[0].strip().split():  # keep label be 0 or 1
            label[int(label_id)] =1
        sentence_wordlist=parts[2].strip().split()#clean_text_to_wordlist(parts[2].strip())

        labels.append(label)
        sent_idlist, sent_masklist=transfer_wordlist_2_idlist_with_maxlen(sentence_wordlist, word2id, maxlen)
        sents.append(sent_idlist)
        sents_masks.append(sent_masklist)

    print('\t\t\t size:', len(labels))
    print('dataset loaded over, totally ', len(word2id), 'words')
        # print('label size:', np.sum(np.array(labels),axis=0))
    # exit(0)
    return sents, sents_masks, labels, word2id

def load_single_test_sample(word2id, maxlen, text):

    sents=[]
    sents_masks=[]

    sentence_wordlist=text.strip().split()#clean_text_to_wordlist(parts[2].strip())
    sent_idlist, sent_masklist=transfer_wordlist_2_idlist_with_maxlen(sentence_wordlist, word2id, maxlen)

    sents.append(sent_idlist)
    sents_masks.append(sent_masklist)

    print('dataset loaded over, totally ', len(word2id), 'words')
        # print('label size:', np.sum(np.array(labels),axis=0))
    # exit(0)
    return sents, sents_masks, word2id



def f1_two_col_array(vec1, vec2):
    overlap = sum(vec1*vec2)
    pos1 = sum(vec1)
    pos2 = sum(vec2)
    recall = overlap*1.0/(1e-8+pos1)
    precision = overlap * 1.0/ (1e-8+pos2)
    return 2*recall*precision/(1e-8+recall+precision)

def average_f1_two_array_by_col(arr1, arr2):
    '''
    we do not consider label 9: nothing
    '''
    col_size = arr1.shape[1]
    f1_list = []
    class_size_list = []
    for i in range(col_size):
        if i !=11: # note that il5, il6 does not have 11th class
            f1_i = f1_two_col_array(arr1[:,i], arr2[:,i])
            class_size = sum(arr2[:,i])
            f1_list.append(f1_i)
            class_size_list.append(class_size)

    # print('f1_list:', f1_list)
    # print('class_size_list:', class_size_list)

    mean_f1 = sum(f1_list)/len(f1_list)
    weighted_f1 = sum([x*y for x,y in zip(f1_list,class_size_list)])/sum(class_size_list)
    # print 'mean_f1, weighted_f1:', mean_f1, weighted_f1
    # exit(0)
    return mean_f1, weighted_f1


def load_somali_input(word2id, maxlen, input_sent_str):
    all_sentences=[]
    all_masks=[]
    lines=[]
    lines.append(input_sent_str)
    sentence_wordlist=input_sent_str.strip().split()
    sent_idlist, sent_masklist=transfer_wordlist_2_idlist_with_maxlen_in_Test(sentence_wordlist, word2id, maxlen)
    all_sentences.append(sent_idlist)
    all_masks.append(sent_masklist)

    print('\t\t\t size:', len(all_sentences))
    print('dataset loaded over, totally ', len(word2id), 'words')
    return all_sentences, all_masks, lines, word2id

def load_official_testData_il_and_MT(word2id, maxlen, fullpath, input_type):
    all_sentences=[]
    all_masks=[]
    print('loading file:', fullpath, '...')
    co =0
    readfile=codecs.open(fullpath, 'r', 'utf-8')
    lines=[]
    '''
    IL11_NW_021222_20181103_J0040W2IB       segment-2       il_text     eng_text   631-636-NIL30
    '''
    for line in readfile:
        parts=line.strip().split('\t') #lowercase all tokens, as we guess this is not important for sentiment task
        # lines.append('\t'.join([parts[0],parts[1],parts[2],parts[4]]))
        lines.append('\t'.join([parts[0],parts[1],parts[2],parts[3]]))
        if input_type == 0: #without MT
            sentence_wordlist=parts[2].strip().split()
        elif input_type == 1: #with MT
            sentence_wordlist=parts[2].strip().split()[:(int(maxlen/2))]+parts[3].strip().lower().split()#clean_text_to_wordlist(parts[2].strip())
        else:
            sentence_wordlist=parts[3].strip().lower().split()

        sent_idlist, sent_masklist=transfer_wordlist_2_idlist_with_maxlen(sentence_wordlist, word2id, maxlen)
        all_sentences.append(sent_idlist)
        all_masks.append(sent_masklist)
        co+=1
    print('\t\t\t size:', len(all_sentences))
    print('dataset loaded over, totally ', len(word2id), 'words')
    return all_sentences, all_masks,lines, word2id#, all_labels, all_other_labels,word2id

def load_official_testData_il_and_MT_multiple_testfiles(word2id, maxlen, fullpaths):
    all_sentences=[]
    all_masks=[]
    lines=[]
    for fullpath in fullpaths:
        all_sentences_type0=[]
        all_masks_type0=[]
        lines_type0=[]

        all_sentences_type1=[]
        all_masks_type1=[]
        lines_type1=[]

        all_sentences_type2=[]
        all_masks_type2=[]
        lines_type2=[]

        print('loading file:', fullpath, '...')
        co =0
        readfile=codecs.open(fullpath, 'r', 'utf-8')

        '''
        IL11_NW_021222_20181103_J0040W2IB       segment-2       il_text     eng_text   631-636-NIL30
        '''
        for line in readfile:
            parts=line.strip().split('\t') #lowercase all tokens, as we guess this is not important for sentiment task

            sentence_wordlist_type0=parts[2].strip().split()
            sentence_wordlist_type1=parts[2].strip().split()[:(int(maxlen/2))]+parts[3].strip().lower().split()#clean_text_to_wordlist(parts[2].strip())
            sentence_wordlist_type2=parts[3].strip().lower().split()

            sent_idlist_type0, sent_masklist_type0=transfer_wordlist_2_idlist_with_maxlen(sentence_wordlist_type0, word2id, maxlen)
            all_sentences_type0.append(sent_idlist_type0)
            all_masks_type0.append(sent_masklist_type0)
            lines_type0.append('\t'.join([parts[0],parts[1],parts[2],parts[4]]))

            sent_idlist_type1, sent_masklist_type1=transfer_wordlist_2_idlist_with_maxlen(sentence_wordlist_type1, word2id, maxlen)
            all_sentences_type1.append(sent_idlist_type1)
            all_masks_type1.append(sent_masklist_type1)
            lines_type1.append('\t'.join([parts[0],parts[1],parts[2],parts[4]]))

            sent_idlist_type2, sent_masklist_type2=transfer_wordlist_2_idlist_with_maxlen(sentence_wordlist_type2, word2id, maxlen)
            all_sentences_type2.append(sent_idlist_type2)
            all_masks_type2.append(sent_masklist_type2)
            lines_type2.append('\t'.join([parts[0],parts[1],parts[2],parts[4]]))

            co+=1
        print('\t\t\t size:', co)
        all_sentences.append(all_sentences_type0)
        all_sentences.append(all_sentences_type1)
        all_sentences.append(all_sentences_type2)
        all_masks.append(all_masks_type0)
        all_masks.append(all_masks_type1)
        all_masks.append(all_masks_type2)
        lines.append(lines_type0)
        lines.append(lines_type1)
        lines.append(lines_type2)
    print('dataset loaded over, totally ', len(word2id), 'words')
    return all_sentences, all_masks,lines, word2id#, all_labels, all_other_labels,word2id


def load_SF_type_descriptions(word2id, type_size, describ_max_len):
    # type2label_id = {'crimeviolence':8, 'med':3, 'search':4, 'food':1, 'out-of-domain':9, 'infra':2, 'water':7, 'shelter':5,
    # 'regimechange':10, 'evac':0, 'terrorism':11, 'utils':6}
    # type2des = {0:'evacuation evacuate landslide flood volcano earthquake hurricane',
    #                     1:'food hunger starvation starve bread earthquake hurricane refugees',
    #                     2:'infrastructure  damage house collapse water pipe burst no electricity road earthquake hurricane',
    #                     3: 'medical assistance sick flu dysentery patient insufficiency earthquake',
    #                     4: 'search house collapse  person  missing  buried earthquake hurricane',
    #                     5: 'shelter house collapse homeless earthquake hurricane refugees',
    #                     6: 'utilities energy sanitation electricity earthquake hurricane',
    #                     7: 'water food hunger starvation starve water pollution earthquake refugees',
    #                     8: 'crime violence robbery snoring looting burning plunder shooting blow up explode attack arrest kill shot police incident',
    #                     9: 'none nothing',
    #                     10: 'regime, change coup overthrow subversion resign subvert turn over rebel army',
    #                     11: 'terrorism blow up explode shooting suicide attack terrorist conspiracy explosion terror bombing bomb isis'
    #                     }
    type2des = {0:'evacuation evacuate landslide flood volcano earthquake hurricane',
                        1:'food hunger starvation starve bread earthquake hurricane refugees',
                        2:'infrastructure  damage house collapse water pipe burst no electricity road earthquake hurricane',
                        3: 'medical assistance sick flu dysentery patient insufficiency earthquake',
                        4: 'search house collapse  person  missing  buried earthquake hurricane',
                        5: 'shelter house collapse homeless earthquake hurricane refugees',
                        6: 'utilities energy sanitation electricity earthquake hurricane',
                        7: 'water food hunger starvation starve water pollution earthquake refugees',
                        8: 'crime violence robbery snoring looting burning plunder shooting blow up explode attack arrest kill shot police incident',
                        9: 'regime, change coup overthrow subversion resign subvert turn over rebel army',
                        10: 'terrorism blow up explode shooting suicide attack terrorist conspiracy explosion terror bombing bomb isis',
                        11: 'none nothing'
                        }
    label_sent = []
    label_mask = []
    for i in range(type_size):
        sent_str = type2des.get(i)
        sent_idlist, sent_masklist=transfer_wordlist_2_idlist_with_maxlen(sent_str.split(), word2id, describ_max_len)
        label_sent.append(sent_idlist)
        label_mask.append(sent_masklist)
    return np.array(label_sent), np.array(label_mask)


def get_need_other_fields(matrix):
    #matrix (4,4)
    # other_field2index = {'current':0,'not_current':1, 'sufficient':0,'insufficient':1,'True':0,'False':1}
    # other_fields=[2]*4 #need_status, issue_status, need_relief, need_urgency, defalse "2" denotes no label
    fields_size = len(matrix)
    assert fields_size ==4
    if matrix[0][0]>matrix[0][1] and matrix[0][0]>matrix[0][2]:
        status = 'current'
    elif matrix[0][1]>matrix[0][0] and matrix[0][1]>matrix[0][2]:
        status = 'future'
    elif matrix[0][2]>matrix[0][0] and matrix[0][2]>matrix[0][1]:
        status = 'past'
    if matrix[2][0]>matrix[2][1]:
        relief = 'sufficient'
    else:
        relief = 'insufficient'
    if matrix[3][0]>matrix[3][1]:
        urgency = True
    else:
        urgency = False
    return    status,  relief,urgency

def get_issue_other_fields(matrix):
    #matrix (4,3)
    # other_field2index = {'current':0,'not_current':1, 'sufficient':0,'insufficient':1,'True':0,'False':1}
    # other_fields=[2]*4 #need_status, issue_status, need_relief, need_urgency, defalse "2" denotes no label
    fields_size = len(matrix)
    assert fields_size ==4
    if matrix[1][0]>matrix[1][1]:
        status = 'current'
    else:
        status = 'not_current'
    if matrix[3][0]>matrix[3][1]:
        urgency = True
    else:
        urgency = False
    return    status, urgency

def generate_2019_official_output(lines, output_file_path, pred_types, pred_confs, pred_others):
    #pred_others (#instance, 4, 3)
    # thai_root = '/save/wenpeng/datasets/LORELEI/Thai/'
    instance_size = len(pred_types)
    '''已经考虑了NOne调整到最后一个label的情况'''
    type2label_id = {'crimeviolence':8, 'med':3, 'search':4, 'food':1, 'out-of-domain':11, 'infra':2,
    'water':7, 'shelter':5, 'regimechange':9, 'evac':0, 'terrorism':10, 'utils':6}

    id2type = {y:x for x,y in type2label_id.items()}

    output_dict_list = []
    assert instance_size == len(pred_others)
    assert instance_size == len(pred_confs)
    assert instance_size == len(lines)
    print('seg size to pred: ', instance_size, 'full file size:', len(lines))

    #needs
    for i in range(instance_size):
        pred_vec = list(pred_types[i])
        text_parts = lines[i].split('\t')
        doc_id = text_parts[0]
        seg_id = text_parts[1]
        entity_pos_list = text_parts[3].split() #116-123-6252001 125-130-49518 198-203-49518
        for x, y in enumerate(pred_vec):
            if y == 1:
                if x < 8: # is a need type
                    for entity_pos in entity_pos_list:
                        kb_id = entity_pos.split('-')[2]
                        new_dict={}
                        new_dict['DocumentID'] = doc_id
                        hit_need_type = id2type.get(x)
                        new_dict['Type'] = hit_need_type
                        new_dict['Place_KB_ID'] = kb_id
                        status,  relief,urgency = get_need_other_fields(pred_others[i])
                        new_dict['Status'] = status
                        new_dict['Confidence'] = float(pred_confs[i][x])
                        new_dict['Justification_ID'] = seg_id
                        new_dict['Resolution'] = relief
                        new_dict['Urgent'] = urgency
                        '''？？？这儿是否需要修改一下'''
                        if new_dict.get('Confidence') > 0.4:
                            output_dict_list.append(new_dict)

                elif x < 11: # is issue
                    for entity_pos in entity_pos_list:
                        kb_id = entity_pos.split('-')[2]
                        new_dict={}
                        new_dict['DocumentID'] = doc_id
                        hit_issue_type = id2type.get(x)
                        new_dict['Type'] = hit_issue_type
                        new_dict['Place_KB_ID'] = kb_id
                        # new_dict['Place_'] = 14.0
                        status, urgency = get_issue_other_fields(pred_others[i])
                        new_dict['Status'] = status
                        new_dict['Confidence'] = float(pred_confs[i][x])
                        new_dict['Justification_ID'] = seg_id
                        new_dict['Urgent'] = urgency
                        if new_dict.get('Confidence') > 0.4:
                            output_dict_list.append(new_dict)


    refine_output_dict_list, ent_size = de_duplicate(output_dict_list)
    frame_size = len(refine_output_dict_list)
    mean_frame = frame_size*1.0/ent_size

    writefile = codecs.open(output_file_path ,'w', 'utf-8')
    json.dump(refine_output_dict_list, writefile)
    writefile.close()
    print('official output succeed...Frame size:', frame_size, 'average:', mean_frame, 'ent_size:',ent_size)
    return mean_frame

def majority_ele_in_list(lis):
    c = Counter(lis)
    return c.most_common()[0][0]

def best_seg_id(segs, window):
    #[segment-2, segment-7, ...]
    #window =2
    id_list = []
    for seg in segs:
        id_list.append(int(seg.split('-')[1]))
    min_seg = min(id_list)
    max_seg = max(id_list)
    extend_list = []
    for idd in id_list:
        for i in range(idd-window, idd+window+1):
            if i >= min_seg and i <=max_seg:
                extend_list.append(i)

    majority_id = majority_ele_in_list(extend_list)
    return 'segment-'+str(majority_id)
def de_duplicate(output_dict_list):
    need_type_set = set([ 'med','search','food','infra','water','shelter','evac','utils'])
    issue_type_set = set(['regimechange','crimeviolence','terrorism'])
    new_dict_list=[]
    key2dict_list = defaultdict(list)
    ent_set = set()
    for dic in output_dict_list:
        doc_id = dic.get('DocumentID')
        type = dic.get('Type')
        kb_id = dic.get('Place_KB_ID')
        key = (doc_id, type, kb_id)
        ent_set.add((doc_id, kb_id))
        key2dict_list[key].append(dic)
    for key, dict_list in key2dict_list.items():
        #compute status, confidence
        doc_id = key[0]
        SF_type = key[1]
        kb_id = key[2]
        if dict_list[0].get('Type') in need_type_set:
            status=[]
            relief=[]
            urgency=[]
            conf = []
            segs = []

            for dic in dict_list:
                status.append(dic.get('Status'))
                relief.append(dic.get('Resolution'))
                urgency.append(dic.get('Urgent'))
                conf.append(dic.get('Confidence'))
                segs.append(dic.get('Justification_ID'))
            status = majority_ele_in_list(status)
            relief = majority_ele_in_list(relief)
            urgency = majority_ele_in_list(urgency)
            conf = max(conf)
            seg_id = best_seg_id(segs,2)

            new_dict={}
            new_dict['DocumentID'] = doc_id
            new_dict['Type'] = SF_type
            new_dict['Place_KB_ID'] =  kb_id
            new_dict['Status'] = status
            new_dict['Confidence'] = conf
            new_dict['Justification_ID'] = seg_id
            new_dict['Resolution'] = relief
            new_dict['Urgent'] = urgency
            new_dict_list.append(new_dict)
        elif dict_list[0].get('Type') in issue_type_set:
            status=[]
            urgency=[]
            conf = []
            segs = []

            for dic in dict_list:
                status.append(dic.get('Status'))
                urgency.append(dic.get('Urgent'))
                conf.append(dic.get('Confidence'))
                segs.append(dic.get('Justification_ID'))
            status = majority_ele_in_list(status)
            urgency = majority_ele_in_list(urgency)
            conf = max(conf)
            seg_id = best_seg_id(segs,2)
            new_dict={}
            new_dict['DocumentID'] = doc_id
            new_dict['Type'] = SF_type
            new_dict['Place_KB_ID'] =  kb_id
            new_dict['Status'] = status
            new_dict['Confidence'] = conf
            new_dict['Justification_ID'] = seg_id
            new_dict['Urgent'] = urgency
            new_dict_list.append(new_dict)
        else:
            print('wring detected SF type:', dict_list[0].get('Type'))
            exit(0)
    return       new_dict_list, len(ent_set)

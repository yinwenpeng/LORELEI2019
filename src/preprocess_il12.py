import xml.etree.ElementTree as ET
import codecs
import os
from preprocess_common import load_EDL2018_output,IL_into_test_withMT_filteredby_NER_2018
from collections import defaultdict
from BiCCA import BiCCA

def extract_dictionary(file_path, outfile, full_dict):
    # file_path = '/shared/corpora/corporaWeb/lorelei/evaluation-20160705/il3/set0/docs/categoryI_dictionary/IL3_dictionary.xml'
    writefile = codecs.open(outfile, 'w', 'utf-8')
    readfile = codecs.open(file_path, 'r', 'utf-8')
    all_co = 0
    use_co = 0
    for line in readfile:
        parts = line.strip().split('\t')
        all_co+=1
        if full_dict:
            eng_wordlist = parts[0].strip().split()
            il_wordlist = parts[1].strip().split()
            for word1 in eng_wordlist:
                for word2 in il_wordlist:
                    writefile.write(word1+'\t'+word2+'\n')
                    use_co+=1
        else:
            if len(parts[0].strip().split())==1 and len(parts[1].strip().split())==1:
                writefile.write(line.strip()+'\n')
                use_co+=1
    readfile.close()
    writefile.close()
    print('dict extract over:',use_co, all_co )

# def parse_wordembeddings():
#     readfile = codecs.open('/home/wyin3/LORELEI/2019/dry_run_Uyghur/model.txt', 'r', 'utf-8')
#     for line in readfile:
#         parts =line.strip().split()


def extract_monolingual_text_4_train_word2vec(ltf_paths, output_file):

    writefile = codecs.open(output_file, 'w', 'utf-8')
    for ltf_path in ltf_paths:
        folders = [ltf_path]
        word_set =set()
        orig_text_co = 0
        for folder in folders:
            files= os.listdir(folder)
            print(('folder file sizes: ', len(files)))
            co=0
            for fil in files:
                print(fil)
                f = ET.parse(folder+fil)
                root = f.getroot()
                for seg in root.iter('SEG'):
                    sent = seg.find('ORIGINAL_TEXT').text
                    writefile.write(sent+'\n')
                    # print(sent)
                    orig_text_co+=1
                    # if orig_text_co > 3:
                    #     exit(0)
                    for word in sent.split():
                        word_set.add(word)
                writefile.write('\n')
    writefile.close()
    print('parse over, word size:', len(word_set))

def run_word2vec(input, output):
    '''
    ./word2vec -train data.txt -output vec.txt -size 200 -window 5 -sample 1e-4 -negative 5 -hs 0 -binary 0 -cbow 1 -iter 3
    '''
    command_line = "/home/wyin3/workspace/word2vec/trunk/./word2vec -train "+input+" -output "+output+" -size 300 -window 10 -sample 1e-4 -negative 5 -hs 0 -binary 0 -cbow 0 -iter 20 -min-count 5"
    os.system(command_line)

def generate_bilingual_wordembeddings(eng_emb, il_emb, wordmap, lang_name, BICCA):
    if BICCA:
        # BiCCA(il_emb, eng_emb, 'en', 'il9', wordmap, projectname, ratio)
        command_line = 'python /home/wyin3/workspace/LORELEI2019/src/fixcca.py '+eng_emb+' '+il_emb+' '+wordmap+' '+lang_name+' 300'
        os.system(command_line)

    else:
        command_line = 'CUDA_VISIBLE_DEVICES=0 python /home/wyin3/workspace/MUSE/supervised.py --src_lang en --tgt_lang il9 --src_emb '+eng_emb+' --tgt_emb '+il_emb+' --n_refinement 5 --dico_train '+wordmap+' --dico_eval '+wordmap+' --exp_path /home/wyin3/LORELEI/2019/il9/'
        os.system(command_line)

def recover_pytorch_idmatrix_2_text(sentence_batch, id2word):
    sentence_numpy = sentence_batch.cpu().numpy()
    bert_rep_batch = []
    sent_list = []
    for i in range(sentence_numpy.shape[0]): #batch
        sent_str = ''
        for id in list(sentence_numpy[i]):
            if id !=0:
                sent_str+=id2word.get(id)+' '
        sent_list.append(sent_str.strip())
    return sent_list

def generate_official_test_data():
    docid2entity_pos_list = load_EDL2018_output('/home/wyin3/LORELEI/2019/retest/UPENN+18-rules.tab')
    # IL_into_test_filteredby_NER_2018('/save/wenpeng/datasets/LORELEI/il9/monolingual_text/','/save/wenpeng/datasets/LORELEI/il9/il9-setE-as-test-input_ner_filtered', docid2entity_pos_list, 2)
    IL_into_test_withMT_filteredby_NER_2018('/home/wyin3/LORELEI/2019/retest/monolingual_text/','/home/wyin3/LORELEI/2019/retest/BBN_MT/','/home/wyin3/LORELEI/2019/retest/il3-uyghur-setE-as-test-input_ner_filtered', docid2entity_pos_list, 2)
    # json_validation('/save/wenpeng/datasets/LORELEI/il9/il9_system_output_forfun_w2.json')

def load_il9_with_ground_truth():
    '''
    monolingual_text: doc_id: sent_list:['...','...'], boundary_list:[(1,12),(14,23)...]
    ground truth:
        issue: doc_id:[{'entity_id': place_id, 'frame_type': issue, 'issue_type':crimeviolence},{'entity_id':...}]
        mentions: doc_id:[{'entity_id': place_id, 'entity_type': GPE, 'start_char':12,'end_char':15}]
        needs: doc_id:[{'entity_id': place_id, 'frame_type': need, 'need_type':med, 'need_status': current, 'urgency_status': true/false, 'resolution_status':insufficient},{'entity_id':...}]
    '''

    #first load Uyghur sent list
    folder = '/scratch/wyin3/dickens_save_dataset/LORELEI/il9/monolingual_text/'
    docid2text={}
    re=0

    files= os.listdir(folder)
    print('folder file sizes: ', len(files))
    for fil in files:

        boundary_list = []
        sent_list = []
        seg_idlist = []
        f = ET.parse(folder+'/'+fil)
        # f = codecs.open(folder+'/'+fil, 'r', 'utf-8')

        root = f.getroot()
        for doc in root.iter('DOC'):
            doc_id  = doc.attrib.get('id')
        for seg in root.iter('SEG'):
            seg_id = seg.attrib.get('id')
            start = int(seg.attrib.get('start_char'))
            end = int(seg.attrib.get('end_char'))
            sent_i = seg.find('ORIGINAL_TEXT').text
            # sent+=' '+sent_i
            sent_list.append(sent_i)
            boundary_list.append((start,end))
            seg_idlist.append(seg_id)
        doc_instance={}
        doc_instance['doc_id'] = doc_id
        doc_instance['sent_list'] = sent_list
        doc_instance['boundary_list'] = boundary_list
        doc_instance['seg_idlist'] = seg_idlist

        docid2text[doc_id] = doc_instance
        # f.close()
    print('load load_text_given_docvocab over, size: ', len(docid2text))

    '''
    load issues
    '''
    docid2issue = {}
    # folder = '/save/wenpeng/datasets/LORELEI/il5_unseq/setE/data/annotation/situation_frame/issues/'
    folder = '/shared/corpora/corporaWeb/lorelei/evaluation-2018/il9/source/il9_unseq/setE/data/annotation/il9/situation_frame/issues/'
    files= os.listdir(folder)
    print('issues file sizes: ', len(files))
    for fil in files:
        if not os.path.isdir(fil):
            f = codecs.open(folder+'/'+fil, 'r', 'utf-8')
            line_co = 0

            issue_list = []
            for line in f:
                if line_co == 0:
                    line_co+=1
                    continue
                else:

                    issue_instance = {}
                    parts = line.strip().split('\t')
                    '''
                    user_id doc_id  frame_id        frame_type      issue_type      place_id        proxy_status    issue_status    scope   severity        description     kb_id
                    '''
                    doc_id = parts[1]
                    frame_type = parts[3]
                    issue_type = parts[4]
                    place_id = parts[5]
                    issue_status = parts[7]

                    # issue_instance['doc_id'] = doc_id
                    issue_instance['frame_type'] = frame_type
                    issue_instance['issue_type'] = issue_type
                    issue_instance['entity_id'] = place_id
                    issue_instance['issue_status'] = issue_status
                    issue_list.append(issue_instance)
                    line_co+=1
            issue_list_remove_duplicate = list({frozenset(item.items()):item for item in issue_list}.values())
            docid2issue[doc_id] = issue_list_remove_duplicate
            f.close()
    '''
    load mentions
    mentions: doc_id:[{'entity_id': place_id, 'entity_type': GPE, 'start_char':12,'end_char':15}]
    '''
    folder = '/shared/corpora/corporaWeb/lorelei/evaluation-2018/il9/source/il9_unseq/setE/data/annotation/il9/situation_frame/mentions/'
    files= os.listdir(folder)
    print('mentions file sizes: ', len(files))
    docid2mention = {}
    for fil in files:
        if not os.path.isdir(fil):
            f = codecs.open(folder+'/'+fil, 'r', 'utf-8')
            line_co = 0
            mention_list = []
            for line in f:
                if line_co == 0:
                    line_co+=1
                    continue
                else:

                    mention_instance = {}
                    parts = line.strip().split('\t')
                    '''
                    doc_id  entity_id       mention_id      entity_type     mention_status  start_char      end_char        mention_text
                    '''
                    doc_id = parts[0]
                    entity_id = parts[1]
                    entity_type = parts[3]
                    start_char = parts[5]
                    end_char = parts[6]


                    # mention_instance['doc_id'] = doc_id
                    mention_instance['entity_id'] = entity_id
                    mention_instance['entity_type'] = entity_type
                    mention_instance['start_char'] = int(start_char)
                    mention_instance['end_char'] = int(end_char)

                    mention_list.append(mention_instance)
                    line_co+=1
            mention_list_remove_duplicate = list({frozenset(item.items()):item for item in mention_list}.values())
            docid2mention[doc_id] = mention_list_remove_duplicate
            # print 'docid2mention[doc_id]:',doc_id,docid2mention[doc_id]
            # exit(0)
            f.close()

    '''
    load needs
    needs: doc_id:[{'entity_id': place_id, 'frame_type': need, 'need_type':med, 'need_status': current, 'urgency_status': true/false, 'resolution_status':insufficient},{'entity_id':...}]
    user_id doc_id  frame_id        frame_type      need_type       place_id        proxy_status    need_status     urgency_status  resolution_status       reported_by     resolved_by     description
    '''
    docid2need = {}
    folder = '/shared/corpora/corporaWeb/lorelei/evaluation-2018/il9/source/il9_unseq/setE/data/annotation/il9/situation_frame/needs/'
    files= os.listdir(folder)
    print('needs file sizes: ', len(files))
    for fil in files:
        if not os.path.isdir(fil):
            f = codecs.open(folder+'/'+fil, 'r', 'utf-8')
            line_co = 0

            need_list = []
            for line in f:
                if line_co == 0:
                    line_co+=1
                    continue
                else:

                    need_instance = {}
                    parts = line.strip().split('\t')
                    '''
                    doc_id  frame_id        frame_type      need_type       place_id        proxy_status    need_status     urgency_status  resolution_status       reported_by     resolved_by description
                    user_id doc_id  frame_id        frame_type      need_type       place_id        proxy_status    need_status     scope   severity        resolution_status       reported_by     resolved_by     description     kb_id
                    '''
                    doc_id = parts[1]
                    frame_type = parts[3]
                    need_type = parts[4]
                    place_id = parts[5]
                    need_status = parts[7]
                    '''we did not find urgency in il9 grond truth'''
                    urgency_status = 'None'#parts[7]
                    resolution_status = parts[10]

                    # issue_instance['doc_id'] = doc_id
                    need_instance['frame_type'] = frame_type
                    need_instance['need_type'] = need_type
                    need_instance['entity_id'] = place_id
                    need_instance['need_status'] = need_status
                    need_instance['urgency_status'] = urgency_status
                    need_instance['resolution_status'] = resolution_status


                    need_list.append(need_instance)
                    line_co+=1
            need_list_remove_duplicate = list({frozenset(item.items()):item for item in need_list}.values())
            # print('need_list_remove_duplicate:',need_list_remove_duplicate)
            docid2need[doc_id] = need_list_remove_duplicate
            # print doc_id, docid2need[doc_id]
            # exit(0)
            f.close()
    return docid2text, docid2issue, docid2mention, docid2need

def entity_id_2_sentID(sent_list, boundary_list, docid2mention, doc_id, entity_id):

    mention_entity_instance_list = docid2mention.get(doc_id)
    for instance_dict in mention_entity_instance_list:
        if instance_dict.get('entity_id') == entity_id:
            start_char = instance_dict.get('start_char')
            end_char = instance_dict.get('end_char')
            # print start_char, end_char, boundary_list
            for i in range(len(boundary_list)):
                tuplee = boundary_list[i]
                # print tuplee
                # print start_char, tuplee[1]
                # print tuplee[1] < start_char #,  start_char < tuplee[1]
                # print tuplee[1]/2.0
                # print start_char/2.0
                # exit(0)
                # print start_char, end_char, tuplee[0], tuplee[1], start_char >= tuplee[0], start_char <= 139, 34 <= tuplee[1], 34 <=139
                if start_char >= tuplee[0] and end_char <= tuplee[1]:
                    return '-'.join(map(str, [i]))
    print('failed to find a sentence for the location:', entity_id)
    exit(0)


def generate_entity_focused_trainingset(docid2text, docid2issue, docid2mention, docid2need):
    '''
    monolingual_text: doc_id: sent_list:['...','...'], boundary_list:[(1,12),(14,23)...]
    ground truth:
        issue: doc_id:[{'entity_id': place_id, 'frame_type': issue, 'issue_type':crimeviolence},{'entity_id':...}]
        mentions: doc_id:[{'entity_id': place_id, 'entity_type': GPE, 'start_char':12,'end_char':15}]
        needs: doc_id:[{'entity_id': place_id, 'frame_type': need, 'need_type':med, 'need_status': current, 'urgency_status': true/false, 'resolution_status':insufficient},{'entity_id':...}]
    '''
    # type2label_id = {'crimeviolence':8, 'med':3, 'search':4, 'food':1, 'out-of-domain':9, 'infra':2, 'water':7, 'shelter':5,
    # 'regimechange':10, 'evac':0, 'terrorism':11, 'utils':6}
    '''已经考虑了NOne调整到最后一个label的情况'''
    type2label_id = {'crimeviolence':8, 'med':3, 'search':4, 'food':1, 'out-of-domain':11, 'infra':2,
    'water':7, 'shelter':5, 'regimechange':9, 'evac':0, 'terrorism':10, 'utils':6}

    writefile = codecs.open('/home/wyin3/LORELEI/2019/experiments_daily/il9_uyghur_labeled_as_training_seg_level.txt', 'w', 'utf-8')
    write_size = 0
    doc_uninion_issue_and_mentions = set(docid2issue.keys())| set(docid2need.keys())

    for doc_id, doc_instance in docid2text.items():
        sent_list = doc_instance.get('sent_list')
        boundary_list = doc_instance.get('boundary_list')
        if doc_id  in doc_uninion_issue_and_mentions: #this doc has SF type labels
            sentID_2_labelstrlist=defaultdict(list)
            issue_list = docid2issue.get(doc_id)
            if issue_list is not None:
                for i in range(len(issue_list)):
                    issue_dict_instance = issue_list[i]
                    entity_id = issue_dict_instance.get('entity_id')
                    if entity_id == 'none':
                        sent_id = '-'.join(map(str,range(len(sent_list))))
                    else:
                        sent_id = entity_id_2_sentID(sent_list, boundary_list, docid2mention, doc_id, entity_id)

                    issue_type = issue_dict_instance.get('issue_type')
                    sentID_2_labelstrlist[sent_id].append(issue_type)

            need_list = docid2need.get(doc_id)
            # print('doc_id', doc_id)
            # print('need_list:', need_list)
            if need_list is not None:
                for i in range(len(need_list)):
                    need_dict_instance = need_list[i]
                    entity_id = need_dict_instance.get('entity_id')
                    if entity_id == 'none':
                        sent_id = '-'.join(map(str,range(len(sent_list))))
                    else:
                        sent_id = entity_id_2_sentID(sent_list, boundary_list, docid2mention, doc_id, entity_id)
                    need_type = need_dict_instance.get('need_type')
                    sentID_2_labelstrlist[sent_id].append(need_type)


            for sent_ids, labelstrlist in sentID_2_labelstrlist.items():
                sent = ''
                idlist = sent_ids.split('-')
                for id in idlist:
                    sent+=' '+sent_list[int(id)]
                iddlist=[]

                labelstrlist_delete_duplicate = list(set(labelstrlist))
                for  labelstr in labelstrlist_delete_duplicate:
                    idd = type2label_id.get(labelstr)
                    if idd is None:
                        print('labelstr is None:', labelstr)
                        exit(0)
                    iddlist.append(idd)

                writefile.write(' '.join(map(str, iddlist))+'\t'+' '.join(labelstrlist_delete_duplicate)+'\t'+sent.strip()+'\n')
                write_size+=1
    writefile.close()
    print('write_size:', write_size)

def translate_bbn_2_il9():
    '''load il9 dictionary'''
    readfile = codecs.open('/shared/corpora/corporaWeb/lorelei/evaluation-2018/il9/source/il9/set0/docs/categoryI_dictionary/IL9_dictionary.txt', 'r', 'utf-8')
    eng2il9={}
    for line in readfile:
        parts = line.strip().split('\t')
        keys = eng2il9.get(parts[0])
        if keys is None:
            keys = []
        keys.append(parts[1])
        eng2il9[parts[0]] = keys
    readfile.close()
    print('dict load over, size:', len(eng2il9))

    '''load bbn data and translate'''
    readfile = codecs.open('/scratch/wyin3/dickens_save_dataset/LORELEI/SF-BBN-Mark-split/full_BBN_multi.txt', 'r', 'utf-8')
    writefile = codecs.open('/scratch/wyin3/dickens_save_dataset/LORELEI/SF-BBN-Mark-split/full_BBN_multi_translated_2_il9.txt', 'w', 'utf-8')
    for line in readfile:
        parts=line.strip().split('\t') #lowercase all tokens, as we guess this is not important for sentiment task
        if len(parts)==3:

            sentence_wordlist=parts[2].strip().split()#clean_text_to_wordlist(parts[2].strip())
            new_sent = []
            for word in sentence_wordlist:
                il9_wordlist = eng2il9.get(word)
                if il9_wordlist is not None:
                    new_sent+=il9_wordlist
            writefile.write(parts[0].strip()+'\t'+parts[1].strip()+'\t'+' '.join(new_sent)+'\n')
    writefile.close()
    writefile.close()
    print('translate bbn to il9 over')


if __name__ == '__main__':
    # dict_file = '/scratch/wyin3/dickens_save_dataset/LORELEI/il12/il12/source/il12/set0/docs/categoryI_dictionary/IL12_dictionary.txt'
    # sub_dict_file = '/home/wyin3/LORELEI/2019/il12/il12_dictionary_w2w.txt'
    # ltf_path = ['/scratch/wyin3/dickens_save_dataset/LORELEI/il12/il12/source/il12/setE/data/monolingual_text/il12/ltf/',
    # '/scratch/wyin3/dickens_save_dataset/LORELEI/il12/il12/source/il12/set0/data/monolingual_text/ltf/',
    # '/scratch/wyin3/dickens_save_dataset/LORELEI/il12/il12/source/il12/set1/data/monolingual_text/ltf/']
    # mono_text = '/home/wyin3/LORELEI/2019/il12/raw.text.for.train.word2vec.txt'
    # IL_mono_emb_file = '/home/wyin3/LORELEI/2019/il12/il12.w2v.txt'
    # eng_mono_emb_file = '/home/wyin3/Datasets/word2vec_words_300d_insertedline0.txt'#'/home/wyin3/Datasets/word2vec_words_300d.txt'
    # extract_dictionary(dict_file, sub_dict_file, False)
    # extract_monolingual_text_4_train_word2vec(ltf_path, mono_text)
    # run_word2vec(mono_text, IL_mono_emb_file)
    # generate_bilingual_wordembeddings(eng_mono_emb_file, IL_mono_emb_file, sub_dict_file, 'il12longerw2v', True)


    # generate_official_test_data()
    '''generate test set with EDL and MT output'''
    set_E_mono_text = '/scratch/wyin3/dickens_save_dataset/LORELEI/il12/il12/source/il12/setE/data/monolingual_text/il12/ltf/'
    MT_input = '/scratch/wyin3/dickens_save_dataset/LORELEI/il12/MT/'
    EDL_input = [
    '/shared/corpora/corporaWeb/lorelei/evaluation-2019/il12/edl_output/7.24_il12_bert1_tsl0_google1_top1_map1_wc1_ed0_l2smap1_mtype1_wikicg0_spell0_clas1/nilcluster_exact.tab',
    '/shared/corpora/corporaWeb/lorelei/evaluation-2019/il12/edl_output/7.24_il12_bert0_tsl0_google1_top1_map1_wc1_ed0_l2smap1_mtype1_wikicg0_spell0_clas0/nilcluster_exact.tab']
    output_file = '/scratch/wyin3/dickens_save_dataset/LORELEI/il12/il12-setE-as-test-input_edl1_filtered'
    docid2entity_pos_list = load_EDL2018_output(EDL_input[1])
    # IL_into_test_filteredby_NER_2018(set_E_mono_text, somali_path+'somali-setE-as-test-input_ner_filtered', docid2entity_pos_list, 2)
    IL_into_test_withMT_filteredby_NER_2018(set_E_mono_text, MT_input, output_file, docid2entity_pos_list, 2)
    # json_validation('/save/wenpeng/datasets/LORELEI/il9/il9_system_output_forfun_w2.json')



    '''translate_bbn_2_il9'''
    # translate_bbn_2_il9()

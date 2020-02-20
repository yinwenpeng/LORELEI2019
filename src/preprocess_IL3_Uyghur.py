import xml.etree.ElementTree as ET
import codecs
import os
from preprocess_common import load_EDL2018_output,IL_into_test_withMT_filteredby_NER_2018
from collections import defaultdict

def extract_Uyghur_dictionary():
    file_path = '/shared/corpora/corporaWeb/lorelei/evaluation-20160705/il3/set0/docs/categoryI_dictionary/IL3_dictionary.xml'
    writefile = codecs.open('/home/wyin3/LORELEI/2019/dry_run_Uyghur/Uyghur_2_English_dictionary.txt', 'w', 'utf-8')
    tree = ET.parse(file_path)
    root = tree.getroot()
    word2eng={}
    for block in root.findall('ENTRY'):
        Uyghur_word = block.find('WORD').text
        print('Uyghur_word:',Uyghur_word)
        for sense_block in block.findall('SENSE'):
            English_words = sense_block.find('DEFINITION').text
            print('English_words:',English_words)
        # exit(0)
        if Uyghur_word is None or English_words is None:
            print(Uyghur_word)
            exit(0)
        word2eng[Uyghur_word] = English_words
        writefile.write(Uyghur_word+'\t'+English_words+'\n')
    writefile.close()
    print('parse over, size:', len(word2eng))

# def parse_wordembeddings():
#     readfile = codecs.open('/home/wyin3/LORELEI/2019/dry_run_Uyghur/model.txt', 'r', 'utf-8')
#     for line in readfile:
#         parts =line.strip().split()


def extract_raw_Uyghur_4_train_word2vec():
    writefile = codecs.open('/home/wyin3/LORELEI/2019/dry_run_Uyghur/raw.text.passage.separate.txt', 'w', 'utf-8')
    folders = ['/shared/corpora/corporaWeb/lorelei/evaluation-20160705/il3/set0/data/monolingual_text/ltf/']
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

def run_word2vec():
    '''
    ./word2vec -train data.txt -output vec.txt -size 200 -window 5 -sample 1e-4 -negative 5 -hs 0 -binary 0 -cbow 1 -iter 3
    '''
    os.system("/home/wyin3/workspace/word2vec/trunk/./word2vec -train /home/wyin3/LORELEI/2019/dry_run_Uyghur/raw.text.txt -output /home/wyin3/LORELEI/2019/dry_run_Uyghur/uyghur.w2v.txt -size 300 -window 5 -sample 1e-4 -negative 5 -hs 0 -binary 0 -cbow 0 -iter 3 -min-count 5")

def generate_bilingual_wordembeddings():
    os.system('CUDA_VISIBLE_DEVICES=2 python /home/wyin3/workspace/MUSE/unsupervised.py --src_lang en --tgt_lang ug --src_emb /home/wyin3/LORELEI/2019/dry_run_Uyghur/wiki.en.vec --tgt_emb /home/wyin3/LORELEI/2019/dry_run_Uyghur/uyghur.w2v.txt --n_refinement 5 --exp_path /home/wyin3/LORELEI/2019/dry_run_Uyghur/bilingual_emb/ --max_vocab -1')

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

def load_il3_with_ground_truth():
    '''
    monolingual_text: doc_id: sent_list:['...','...'], boundary_list:[(1,12),(14,23)...]
    ground truth:
        issue: doc_id:[{'entity_id': place_id, 'frame_type': issue, 'issue_type':crimeviolence},{'entity_id':...}]
        mentions: doc_id:[{'entity_id': place_id, 'entity_type': GPE, 'start_char':12,'end_char':15}]
        needs: doc_id:[{'entity_id': place_id, 'frame_type': need, 'need_type':med, 'need_status': current, 'urgency_status': true/false, 'resolution_status':insufficient},{'entity_id':...}]
    '''

    #first load Uyghur sent list
    folder = '/home/wyin3/LORELEI/2019/retest/monolingual_text/'
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
    folder = '/shared/corpora/corporaWeb/lorelei/evaluation-20160705/il3_unseq/setE/data/annotation/situation_frame/issues/'
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
                    doc_id  frame_id        frame_type      issue_type      place_id        proxy_status    issue_status    description
                    '''
                    doc_id = parts[0]
                    frame_type = parts[2]
                    issue_type = parts[3]
                    place_id = parts[4]
                    issue_status = parts[6]

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
    folder = '/shared/corpora/corporaWeb/lorelei/evaluation-20160705/il3_unseq/setE/data/annotation/situation_frame/mentions/'
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
    folder = '/shared/corpora/corporaWeb/lorelei/evaluation-20160705/il3_unseq/setE/data/annotation/situation_frame/needs/'
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
                    '''
                    doc_id = parts[0]
                    frame_type = parts[2]
                    need_type = parts[3]
                    place_id = parts[4]
                    need_status = parts[6]
                    urgency_status = parts[7]
                    resolution_status = parts[8]

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

    writefile = codecs.open('/home/wyin3/LORELEI/2019/retest/il3_uyghur_labeled_as_training_seg_level.txt', 'w', 'utf-8')
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

if __name__ == '__main__':
    # extract_Uyghur_dictionary()
    # extract_raw_Uyghur_4_train_word2vec()
    # run_word2vec()
    generate_bilingual_wordembeddings()
    # generate_official_test_data()
    '''generate uyghur with SF ground truth'''
    '''IL3_SN_000370_20160428_G0T0005LE'''
    # docid2text, docid2issue, docid2mention, docid2need = load_il3_with_ground_truth()
    # generate_entity_focused_trainingset(docid2text, docid2issue, docid2mention, docid2need)

import codecs
# from preprocess_il5 import load_il5
# from preprocess_il6 import load_il6
import os


import json

import nltk
from collections import defaultdict

import xml.etree.ElementTree as ET

'''
collect (two versions of) translated il5 and il6 for 8 needs+3issues, and all other fields
(needs&issue, status, relief, urgency, text)
'''

def load_il(mono_path, issue_path, mention_path, need_path):
    '''
    monolingual_text: doc_id: sent_list:['...','...'], boundary_list:[(1,12),(14,23)...]
    ground truth:
        issue: doc_id:[{'entity_id': place_id, 'frame_type': issue, 'issue_type':crimeviolence},{'entity_id':...}]
        mentions: doc_id:[{'entity_id': place_id, 'entity_type': GPE, 'start_char':12,'end_char':15}]
        needs: doc_id:[{'entity_id': place_id, 'frame_type': need, 'need_type':med, 'need_status': current, 'urgency_status': true/false, 'resolution_status':insufficient},{'entity_id':...}]
    '''

    #first load Uyghur sent list
    folder = mono_path#'/scratch/wyin3/dickens_save_dataset/LORELEI/il9/monolingual_text/'
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
    folder = issue_path#'/shared/corpora/corporaWeb/lorelei/evaluation-2018/il9/source/il9_unseq/setE/data/annotation/il9/situation_frame/issues/'
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
                    il6:
                    user_id doc_id  frame_id        frame_type      issue_type      place_id        proxy_status    issue_status    description
                    il3:
                    doc_id  frame_id        frame_type      issue_type      place_id        proxy_status    issue_status    description
                    Chinese:
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
    folder = mention_path#'/shared/corpora/corporaWeb/lorelei/evaluation-2018/il9/source/il9_unseq/setE/data/annotation/il9/situation_frame/mentions/'
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
                    il6:
                    doc_id  entity_id       mention_id      entity_type     mention_status  start_char      end_char        mention_text
                    il3:
                    doc_id  entity_id       mention_id      entity_type     mention_status  start_char      end_char        mention_text
                    Chinese:
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
    folder = need_path#'/shared/corpora/corporaWeb/lorelei/evaluation-2018/il9/source/il9_unseq/setE/data/annotation/il9/situation_frame/needs/'
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
                    il6:
                    user_id doc_id  frame_id        frame_type      need_type       place_id        proxy_status    need_status     urgency_status  resolution_status       reported_by     resolved_by     description
                    il3:
                    doc_id  frame_id        frame_type      need_type       place_id        proxy_status    need_status     urgency_status  resolution_status       reported_by     resolved_by     description
                    Chinese:
                    doc_id  frame_id        frame_type      need_type       place_id        proxy_status    need_status     urgency_status  resolution_status       reported_by     resolved_by     description
                    '''
                    doc_id = parts[0]
                    frame_type = parts[2]
                    need_type = parts[3]
                    place_id = parts[4]
                    need_status = parts[6]
                    '''we did not find urgency in il9 grond truth'''
                    # urgency_status = 'None'#parts[7]
                    '''we  find urgency in il6 grond truth'''
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

def load_il_translated(translation_eng_ltf_path, old_style):
    # read_mtfiles = codecs.open(annotated_filelist_MT_path, 'r','utf-8')#codecs.open('/save/wenpeng/datasets/LORELEI/il5_unseq/setE/docs/annotated_filelist_MT.txt','r','utf-8')
    # translated_docID_set=set()
    # for line in read_mtfiles:
    #     translated_docID_set.add(line.strip())
    # read_mtfiles.close()
    folder = translation_eng_ltf_path#'/save/wenpeng/datasets/LORELEI/il5_unseq/setE/data/translation/eng/ltf/'
    docid2text={}
    re=0

    files= set(os.listdir(folder))
    print('folder file sizes: ', len(files))
    for file in files:
        if old_style:
            trans_doc_id = file[:file.find('.eng')]
            v_A_name = trans_doc_id+'.eng_A.ltf.xml'
            v_B_name = trans_doc_id+'.eng_B.ltf.xml'
        else:
            trans_doc_id = file[:file.find('.eng.ltf.xml')]
            v_A_name = trans_doc_id+'.eng.ltf.xml'
            v_B_name = trans_doc_id+'.eng.ltf.xml'

        if trans_doc_id in docid2text.keys():
            continue
        doc_instance={}
        doc_instance['doc_id'] = trans_doc_id
        if v_A_name in files:
            sent_list_version1 = []
            f = ET.parse(folder+'/'+v_A_name)
            root = f.getroot()
            for seg in root.iter('SEG'):
                sent_i = seg.find('ORIGINAL_TEXT').text
                sent_list_version1.append(sent_i)

            doc_instance['sent_list_version1'] = sent_list_version1
        if v_B_name in files:
            sent_list_version2 = []
            f = ET.parse(folder+'/'+v_B_name)
            root = f.getroot()
            for seg in root.iter('SEG'):
                sent_i = seg.find('ORIGINAL_TEXT').text
                sent_list_version2.append(sent_i)

            doc_instance['sent_list_version2'] = sent_list_version2
        docid2text[trans_doc_id] = doc_instance
        # f.close()
    print('load translated il over, size: ', len(docid2text))
    return docid2text

# def load_il_translated(annotated_filelist_MT_path, eng_ltf_path):
#     # read_mtfiles = codecs.open(annotated_filelist_MT_path, 'r','utf-8')#codecs.open('/save/wenpeng/datasets/LORELEI/il5_unseq/setE/docs/annotated_filelist_MT.txt','r','utf-8')
#     # translated_docID_set=set()
#     # for line in read_mtfiles:
#     #     translated_docID_set.add(line.strip())
#     # read_mtfiles.close()
#     folder = eng_ltf_path#'/save/wenpeng/datasets/LORELEI/il5_unseq/setE/data/translation/eng/ltf/'
#     docid2text={}
#     re=0
#
#     files= set(os.listdir(folder))
#     print 'folder file sizes: ', len(files)
#     for trans_doc_id in translated_docID_set:
#         v_A_name = trans_doc_id+'.eng_A.ltf.xml'
#         v_B_name = trans_doc_id+'.eng_B.ltf.xml'
#         doc_instance={}
#         doc_instance['doc_id'] = trans_doc_id
#         if v_A_name in files:
#             sent_list_version1 = []
#             f = ET.parse(folder+'/'+v_A_name)
#             root = f.getroot()
#             for seg in root.iter('SEG'):
#                 sent_i = seg.find('ORIGINAL_TEXT').text
#                 sent_list_version1.append(sent_i)
#
#             doc_instance['sent_list_version1'] = sent_list_version1
#         if v_B_name in files:
#             sent_list_version2 = []
#             f = ET.parse(folder+'/'+v_B_name)
#             root = f.getroot()
#             for seg in root.iter('SEG'):
#                 sent_i = seg.find('ORIGINAL_TEXT').text
#                 sent_list_version2.append(sent_i)
#
#             doc_instance['sent_list_version2'] = sent_list_version2
#         docid2text[trans_doc_id] = doc_instance
#         # f.close()
#     print 'load translated il over, size: ', len(docid2text)
#     return docid2text

# def load_il6_translated():
#     read_mtfiles = codecs.open('/save/wenpeng/datasets/LORELEI/il6_unseq/setE/docs/annotated_filelist_MT.txt','r','utf-8')
#     translated_docID_set=set()
#     for line in read_mtfiles:
#         translated_docID_set.add(line.strip())
#     read_mtfiles.close()
#     folder = '/save/wenpeng/datasets/LORELEI/il6_unseq/setE/data/translation/eng/ltf/'
#     docid2text={}
#     re=0
#
#     files= set(os.listdir(folder))
#     print 'folder file sizes: ', len(files)
#     for trans_doc_id in translated_docID_set:
#         v_A_name = trans_doc_id+'.eng_A.ltf.xml'
#         v_B_name = trans_doc_id+'.eng_B.ltf.xml'
#         doc_instance={}
#         doc_instance['doc_id'] = trans_doc_id
#         if v_A_name in files:
#             sent_list_version1 = []
#             f = ET.parse(folder+'/'+v_A_name)
#             root = f.getroot()
#             for seg in root.iter('SEG'):
#                 sent_i = seg.find('ORIGINAL_TEXT').text
#                 sent_list_version1.append(sent_i)
#
#             doc_instance['sent_list_version1'] = sent_list_version1
#         if v_B_name in files:
#             sent_list_version2 = []
#             f = ET.parse(folder+'/'+v_B_name)
#             root = f.getroot()
#             for seg in root.iter('SEG'):
#                 sent_i = seg.find('ORIGINAL_TEXT').text
#                 sent_list_version2.append(sent_i)
#
#             doc_instance['sent_list_version2'] = sent_list_version2
#         docid2text[trans_doc_id] = doc_instance
#         # f.close()
#     print 'load translated il6 over, size: ', len(docid2text)
#     return docid2text

def entity_id_2_sentID(sent_list, boundary_list, docid2mention, doc_id, entity_id, window):

    mention_entity_instance_list = docid2mention.get(doc_id)
    for instance_dict in mention_entity_instance_list:
        if instance_dict.get('entity_id') == entity_id:
            start_char = instance_dict.get('start_char')
            end_char = instance_dict.get('end_char')
            # print start_char, end_char, boundary_list
            sent_size = len(boundary_list)
            for i in range(sent_size):
                tuplee = boundary_list[i]
                # print tuplee
                # print start_char, tuplee[1]
                # print tuplee[1] < start_char #,  start_char < tuplee[1]
                # print tuplee[1]/2.0
                # print start_char/2.0
                # exit(0)
                # print start_char, end_char, tuplee[0], tuplee[1], start_char >= tuplee[0], start_char <= 139, 34 <= tuplee[1], 34 <=139
                if start_char >= tuplee[0] and end_char <= tuplee[1]:
                    raw_ids = range(i-window, i+window+1)#[i-1,i,i+1]
                    final_ids = []
                    for iddd in raw_ids:
                        if iddd>=0 and iddd < sent_size:
                            final_ids.append(iddd)
                    return '-'.join(map(str, final_ids))
    print('failed to find a sentence for the location:', entity_id)
    exit(0)

def denoise(text):
    http_pos = text.find('https://')
    return text[:http_pos].strip()

def generate_entity_focused_trainingset_il(docid2text, docid2trans,docid2issue, docid2mention, docid2need, window, output_path):
    '''
    monolingual_text: doc_id: sent_list:['...','...'], boundary_list:[(1,12),(14,23)...]
    ground truth:
        issue: doc_id:[{'entity_id': place_id, 'frame_type': issue, 'issue_type':crimeviolence},{'entity_id':...}]
        mentions: doc_id:[{'entity_id': place_id, 'entity_type': GPE, 'start_char':12,'end_char':15}]
        needs: doc_id:[{'entity_id': place_id, 'frame_type': need, 'need_type':med, 'need_status': current, 'urgency_status': true/false, 'resolution_status':insufficient},{'entity_id':...}]
    '''
    type2label_id = {'crimeviolence':8, 'med':3, 'search':4, 'food':1, 'out-of-domain':9, 'infra':2, 'water':7, 'shelter':5,
    'regimechange':10, 'evac':0, 'terrorism':11, 'utils':6}
    # other_field2index = {'current':0,'not_current':1, 'sufficient':0,'insufficient':1,'True':0,'False':1}
    other_field2index = {'current':0,'not_current':1,'future':1,'past':2, 'sufficient':0,'insufficient':1,'True':0,'False':1}
    writefile = codecs.open(output_path, 'w', 'utf-8')#codecs.open('/save/wenpeng/datasets/LORELEI/il5_translated_seg_level_as_training_all_fields_w'+str(window)+'.txt', 'w', 'utf-8')
    write_size = 0
    doc_union_issue_and_needs = set(docid2issue.keys())| set(docid2need.keys())
    for doc_id, doc_instance in docid2text.items():
        trans_instance = docid2trans.get(doc_id)
        if trans_instance is not None:
            sent_list = doc_instance.get('sent_list')
            trans_v1_sent_list = trans_instance.get('sent_list_version1')
            trans_v2_sent_list = trans_instance.get('sent_list_version2')
            boundary_list = doc_instance.get('boundary_list')

            if doc_id  in doc_union_issue_and_needs: #this doc has SF type labels
                # print('doc_id:', doc_id) IL6_SN_000370_20160429_H0T005ZKT
                # exit(0)
                sentID_2_labelstrlist=defaultdict(list)
                other_fields=[3]*4 #need_status, issue_status, need_relief, need_urgency, defalse "2" denotes no label
                issue_list = docid2issue.get(doc_id)
                if issue_list is not None:
                    for i in range(len(issue_list)):
                        issue_dict_instance = issue_list[i]
                        entity_id = issue_dict_instance.get('entity_id')
                        if entity_id == 'none':
                            sent_id = '-'.join(map(str,range(len(sent_list))))
                        else:
                            sent_id = entity_id_2_sentID(sent_list, boundary_list, docid2mention, doc_id, entity_id, window)

                        issue_type = issue_dict_instance.get('issue_type')
                        sentID_2_labelstrlist[sent_id].append(issue_type)
                        issue_status = issue_dict_instance.get('issue_status')
                        if other_field2index.get(issue_status) is None:
                            print('issue_status:',issue_status)
                        other_fields[1] = other_field2index.get(issue_status,3)


                need_list = docid2need.get(doc_id)
                if need_list is not None:
                    for i in range(len(need_list)):
                        need_dict_instance = need_list[i]
                        entity_id = need_dict_instance.get('entity_id')
                        if entity_id == 'none':
                            sent_id = '-'.join(map(str,range(len(sent_list))))
                        else:
                            sent_id = entity_id_2_sentID(sent_list, boundary_list, docid2mention, doc_id, entity_id, window)
                        need_type = need_dict_instance.get('need_type')
                        sentID_2_labelstrlist[sent_id].append(need_type)

                        need_status = need_dict_instance.get('need_status')
                        need_relief = need_dict_instance.get('resolution_status')
                        need_urgency = need_dict_instance.get('urgency_status')
                        if other_field2index.get(need_urgency) is None:
                            print('need_urgency:',need_urgency)
                        if other_field2index.get(need_status) is None:
                            print('need_status:',need_status)
                        if other_field2index.get(need_relief) is None:
                            print('need_relief:',need_relief)
                        other_fields[0] = other_field2index.get(need_status,3)
                        other_fields[2] = other_field2index.get(need_relief,3)
                        other_fields[3] = other_field2index.get(need_urgency,3)


                for sent_ids, labelstrlist in sentID_2_labelstrlist.items():

                    idlist = sent_ids.split('-')
                    sent1 = ''
                    if trans_v1_sent_list is not None:
                        for id in idlist:
                            sent1+=' '+trans_v1_sent_list[int(id)]
                    sent2 = ''
                    if trans_v2_sent_list is not None:
                        for id in idlist:
                            sent2+=' '+trans_v2_sent_list[int(id)]

                    iddlist=[]
                    labelstrlist_delete_duplicate = list(set(labelstrlist))
                    for  labelstr in labelstrlist_delete_duplicate:
                        idd = type2label_id.get(labelstr)
                        if idd is None:
                            print('labelstr is None:', labelstr)
                            exit(0)
                        iddlist.append(idd)

                    if len(sent1) > 0:
                        writefile.write(' '.join(map(str, iddlist))+'\t'+' '.join(labelstrlist_delete_duplicate)+'\t'+denoise(sent1.strip())+'\t'+' '.join(map(str,other_fields))+'\n')
                    if len(sent2) > 0:
                        writefile.write(' '.join(map(str, iddlist))+'\t'+' '.join(labelstrlist_delete_duplicate)+'\t'+denoise(sent2.strip())+'\t'+' '.join(map(str,other_fields))+'\n')
                    write_size+=1
    writefile.close()
    print('write_size:', write_size)

# def generate_entity_focused_trainingset_il6(docid2text, docid2trans,docid2issue, docid2mention, docid2need, window):
#     '''
#     monolingual_text: doc_id: sent_list:['...','...'], boundary_list:[(1,12),(14,23)...]
#     ground truth:
#         issue: doc_id:[{'entity_id': place_id, 'frame_type': issue, 'issue_type':crimeviolence},{'entity_id':...}]
#         mentions: doc_id:[{'entity_id': place_id, 'entity_type': GPE, 'start_char':12,'end_char':15}]
#         needs: doc_id:[{'entity_id': place_id, 'frame_type': need, 'need_type':med, 'need_status': current, 'urgency_status': true/false, 'resolution_status':insufficient},{'entity_id':...}]
#     '''
#     type2label_id = {'crimeviolence':8, 'med':3, 'search':4, 'food':1, 'out-of-domain':9, 'infra':2, 'water':7, 'shelter':5,
#     'regimechange':10, 'evac':0, 'terrorism':11, 'utils':6}
#     other_field2index = {'current':0,'not_current':1,'future':1,'past':2, 'sufficient':0,'insufficient':1,'True':0,'False':1}
#     writefile = codecs.open('/save/wenpeng/datasets/LORELEI/il6_translated_seg_level_as_training_all_fields_w'+str(window)+'.txt', 'w', 'utf-8')
#     write_size = 0
#     doc_union_issue_and_needs = set(docid2issue.keys())| set(docid2need.keys())
#     for doc_id, doc_instance in docid2text.iteritems():
#         trans_instance = docid2trans.get(doc_id)
#         if trans_instance is not None:
#             sent_list = doc_instance.get('sent_list')
#             trans_v1_sent_list = trans_instance.get('sent_list_version1')
#             trans_v2_sent_list = trans_instance.get('sent_list_version2')
#             boundary_list = doc_instance.get('boundary_list')
#
#             if doc_id  in doc_union_issue_and_needs: #this doc has SF type labels
#                 # iddlist=[]
#                 # labelstrlist = []
#                 sentID_2_labelstrlist=defaultdict(list)
#                 other_fields=[3]*4 #need_status, issue_status, need_relief, need_urgency, defalse "2" denotes no label
#                 issue_list = docid2issue.get(doc_id)
#                 if issue_list is not None:
#                     for i in range(len(issue_list)):
#                         issue_dict_instance = issue_list[i]
#                         entity_id = issue_dict_instance.get('entity_id')
#                         if entity_id == 'none':
#                             sent_id = '-'.join(map(str,range(len(sent_list))))
#                         else:
#                             sent_id = entity_id_2_sentID(sent_list, boundary_list, docid2mention, doc_id, entity_id, window)
#
#                         issue_type = issue_dict_instance.get('issue_type')
#                         sentID_2_labelstrlist[sent_id].append(issue_type)
#                         issue_status = issue_dict_instance.get('issue_status')
#                         if other_field2index.get(issue_status) is None:
#                             print 'issue_status:',issue_status
#                         other_fields[1] = other_field2index.get(issue_status,3)
#
#
#                 need_list = docid2need.get(doc_id)
#                 if need_list is not None:
#                     for i in range(len(need_list)):
#                         need_dict_instance = need_list[i]
#                         entity_id = need_dict_instance.get('entity_id')
#                         if entity_id == 'none':
#                             sent_id = '-'.join(map(str,range(len(sent_list))))
#                         else:
#                             sent_id = entity_id_2_sentID(sent_list, boundary_list, docid2mention, doc_id, entity_id, window)
#                         need_type = need_dict_instance.get('need_type')
#                         sentID_2_labelstrlist[sent_id].append(need_type)
#
#                         need_status = need_dict_instance.get('need_status')
#                         need_relief = need_dict_instance.get('resolution_status')
#                         need_urgency = need_dict_instance.get('urgency_status')
#                         if other_field2index.get(need_urgency) is None:
#                             print 'need_urgency:',need_urgency
#                         if other_field2index.get(need_status) is None:
#                             print 'need_status:',need_status
#                         if other_field2index.get(need_relief) is None:
#                             print 'need_relief:',need_relief
#                         other_fields[0] = other_field2index.get(need_status,3)
#                         other_fields[2] = other_field2index.get(need_relief,3)
#                         other_fields[3] = other_field2index.get(need_urgency,3)
#
#
#                 for sent_ids, labelstrlist in sentID_2_labelstrlist.iteritems():
#
#                     idlist = sent_ids.split('-')
#                     sent1 = ''
#                     if trans_v1_sent_list is not None:
#                         for id in idlist:
#                             sent1+=' '+trans_v1_sent_list[int(id)]
#                     sent2 = ''
#                     if trans_v2_sent_list is not None:
#                         for id in idlist:
#                             sent2+=' '+trans_v2_sent_list[int(id)]
#
#                     iddlist=[]
#                     labelstrlist_delete_duplicate = list(set(labelstrlist))
#                     for  labelstr in labelstrlist_delete_duplicate:
#                         idd = type2label_id.get(labelstr)
#                         if idd is None:
#                             print 'labelstr is None:', labelstr
#                             exit(0)
#                         iddlist.append(idd)
#
#                     if len(sent1) > 0:
#                         writefile.write(' '.join(map(str, iddlist))+'\t'+' '.join(labelstrlist_delete_duplicate)+'\t'+denoise(sent1.strip())+'\t'+' '.join(map(str,other_fields))+'\n')
#                     if len(sent2) > 0:
#                         writefile.write(' '.join(map(str, iddlist))+'\t'+' '.join(labelstrlist_delete_duplicate)+'\t'+denoise(sent2.strip())+'\t'+' '.join(map(str,other_fields))+'\n')
#                     write_size+=1
#     writefile.close()
#     print 'write_size:', write_size

if __name__ == '__main__':
    # mono_path = '/scratch/wyin3/dickens_save_dataset/LORELEI/il9/monolingual_text/'
    # issue_path = '/shared/corpora/corporaWeb/lorelei/evaluation-2018/il9/source/il9_unseq/setE/data/annotation/il9/situation_frame/issues/'
    # mention_path = '/shared/corpora/corporaWeb/lorelei/evaluation-2018/il9/source/il9_unseq/setE/data/annotation/il9/situation_frame/mentions/'
    # need_path = '/shared/corpora/corporaWeb/lorelei/evaluation-2018/il9/source/il9_unseq/setE/data/annotation/il9/situation_frame/needs/'
    # translation_eng_ltf_path = '/shared/corpora/corporaWeb/lorelei/evaluation-2018/il9/source/il9_unseq/setE/data/translation/eng/ltf/'
    # output_path = '/scratch/wyin3/dickens_save_dataset/LORELEI/il9_sf_gold.txt'

    # mono_path = '/scratch/wyin3/dickens_save_dataset/LORELEI/il10/monolingual_text/'
    # issue_path = '/shared/corpora/corporaWeb/lorelei/evaluation-2018/il10/source/il10_unseq/setE/data/annotation/il10/situation_frame/issues/'
    # mention_path = '/shared/corpora/corporaWeb/lorelei/evaluation-2018/il10/source/il10_unseq/setE/data/annotation/il10/situation_frame/mentions/'
    # need_path = '/shared/corpora/corporaWeb/lorelei/evaluation-2018/il10/source/il10_unseq/setE/data/annotation/il10/situation_frame/needs/'
    # translation_eng_ltf_path = '/shared/corpora/corporaWeb/lorelei/evaluation-2018/il10/source/il10_unseq/setE/data/translation/eng/ltf/'
    # output_path = '/scratch/wyin3/dickens_save_dataset/LORELEI/il10_sf_gold.txt'

    # mono_path = '/shared/corpora/corporaWeb/lorelei/evaluation-20170804/LDC2017E29_LORELEI_IL6_Incident_Language_Pack_for_Year_2_Eval_V1.1/setE/data/monolingual_text/ltf'
    # issue_path = '/shared/corpora/corporaWeb/lorelei/evaluation-20170804/il6_unseq/setE/data/annotation/situation_frame/issues/'
    # mention_path = '/shared/corpora/corporaWeb/lorelei/evaluation-20170804/il6_unseq/setE/data/annotation/situation_frame/mentions/'
    # need_path = '/shared/corpora/corporaWeb/lorelei/evaluation-20170804/il6_unseq/setE/data/annotation/situation_frame/needs/'
    # translation_eng_ltf_path = '/shared/corpora/corporaWeb/lorelei/evaluation-20170804/il6_unseq/setE/data/translation/eng/ltf'
    # output_path = '/scratch/wyin3/dickens_save_dataset/LORELEI/il6_sf_gold.txt'

    # mono_path = '/shared/corpora/corporaWeb/lorelei/evaluation-20160705/il3/setE/data/monolingual_text/ltf'
    # issue_path = '/shared/corpora/corporaWeb/lorelei/evaluation-20160705/il3_unseq/setE/data/annotation/situation_frame/issues/'
    # mention_path = '/shared/corpora/corporaWeb/lorelei/evaluation-20160705/il3_unseq/setE/data/annotation/situation_frame/mentions/'
    # need_path = '/shared/corpora/corporaWeb/lorelei/evaluation-20160705/il3_unseq/setE/data/annotation/situation_frame/needs/'
    # translation_eng_ltf_path = '/shared/corpora/corporaWeb/lorelei/evaluation-20160705/il3_unseq/setE/data/translation/eng/ltf'
    # output_path = '/scratch/wyin3/dickens_save_dataset/LORELEI/il3_sf_gold.txt'

    mono_path = '/shared/corpora/corporaWeb/lorelei/LDC2016E30_LORELEI_Mandarin_Incident_Language_Pack_V2.0/setE/data/monolingual_text/ltf'
    issue_path = '/shared/corpora/corporaWeb/lorelei/LDC2016E30_LORELEI_Mandarin_Incident_Language_Pack_V2.0/setE/data/annotation/situation_frame/issues/'
    mention_path = '/shared/corpora/corporaWeb/lorelei/LDC2016E30_LORELEI_Mandarin_Incident_Language_Pack_V2.0/setE/data/annotation/situation_frame/mentions/'
    need_path = '/shared/corpora/corporaWeb/lorelei/LDC2016E30_LORELEI_Mandarin_Incident_Language_Pack_V2.0/setE/data/annotation/situation_frame/needs/'
    translation_eng_ltf_path = '/shared/corpora/corporaWeb/lorelei/LDC2016E30_LORELEI_Mandarin_Incident_Language_Pack_V2.0/setE/data/translation/eng/ltf'
    output_path = '/scratch/wyin3/dickens_save_dataset/LORELEI/Mandarin_sf_gold.txt'


    docid2text, docid2issue, docid2mention, docid2need = load_il(mono_path, issue_path, mention_path, need_path)
    docid2trans = load_il_translated(translation_eng_ltf_path, False)
    generate_entity_focused_trainingset_il(docid2text, docid2trans,docid2issue, docid2mention, docid2need,1, output_path)

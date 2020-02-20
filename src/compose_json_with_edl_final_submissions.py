
import os
from preprocess_common import validate_output_schema



json_path = '/scratch/wyin3/dickens_save_dataset/LORELEI/il12/sf_output_ch2/'
EDL_input = [
# '/shared/corpora/corporaWeb/lorelei/evaluation-2019/il12/edl_output/7.24_il12_bert1_tsl0_google1_top1_map1_wc1_ed0_l2smap1_mtype1_wikicg0_spell0_clas1/nilcluster_exact.tab',
# '/shared/corpora/corporaWeb/lorelei/evaluation-2019/il12/edl_output/7.24_il12_bert0_tsl0_google1_top1_map1_wc1_ed0_l2smap1_mtype1_wikicg0_spell0_clas0/nilcluster_exact.tab',
'/shared/corpora/corporaWeb/lorelei/evaluation-2019/il12/edl_output/7.24_il12-rules_bert0_tsl0_google1_top1_map1_wc1_ed0_l2smap1_mtype1_wikicg0_spell0_clas0']

files= os.listdir(json_path)
size = 0
for fil in files:
    print(fil, size)
    pos = fil.find('.type')
    edl_id = int(fil[pos-1:pos])
    if validate_output_schema(json_path+fil, 'LoReHLT19-schema_V1.json'):
        edl_file = EDL_input[0]

        command_line = 'tar -czvf /scratch/wyin3/dickens_save_dataset/LORELEI/il12/submissions/il12_submission_'+str(size)+'.tgz -C '+json_path+' '+fil+' -C '+edl_file + ' nilcluster_fuzzy.tab'
        print(command_line)
        os.system(command_line)
        size+=1



# json_path = '/scratch/wyin3/dickens_save_dataset/LORELEI/il11/sf_output_ch2/'
# EDL_input = [
# '/shared/corpora/corporaWeb/lorelei/evaluation-2019/il11/edl_output/7.24_il11_bert0_tsl0_google1_top1_map1_gtrans0_wc1_ed0_l2smap1_mtype0_wikicg0_spell0_or2hin1_clas0',
# '/shared/corpora/corporaWeb/lorelei/evaluation-2019/il11/edl_output/7.24_il11_bert0_tsl1_google1_top1_map1_gtrans1_wc1_ed0_l2smap1_mtype1_wikicg0_spell0_or2hin1_clas0']
# files= os.listdir(json_path)
# size = 0
# for fil in files:
#     print(fil, size)
#     pos = fil.find('.type')
#     edl_id = int(fil[pos-1:pos])
#     if validate_output_schema(json_path+fil, 'LoReHLT19-schema_V1.json'):
#         edl_file = EDL_input[edl_id]
#
#         command_line = 'tar -czvf /scratch/wyin3/dickens_save_dataset/LORELEI/il11/submissions/il11_submission_'+str(size)+'.tgz -C '+json_path+' '+fil+' -C '+edl_file + ' nilcluster_exact.tab'
#         print(command_line)
#         os.system(command_line)
#         size+=1

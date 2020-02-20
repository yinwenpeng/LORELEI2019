import csv
import ast
import sys
import codecs
co=0
csv.field_size_limit(sys.maxsize)

# theme_set = set()
# with open('/scratch/wyin3/dickens_save_dataset/LORELEI/2019_new_data/reports.csv', mode='r') as csvfile:
#     # spamreader = csv.reader(csvfile, delimiter=',')
#     csv_reader = csv.DictReader(csvfile)
#     for row in csv_reader:
#         title = row['title']
#         text = row['body']
#         # print(row)
#         if len(row['theme']) > 0 and len(row['country']) > 0:
#             theme_list = ast.literal_eval(row['theme'])
#             country_list = ast.literal_eval(row['country'])
#
#         theme_set |= set(theme_list)
#         # print(theme_set)
#         co+=1
#         if co%100==0:
#             print('co:', co, len(theme_set))
#
# print(theme_set)
# csvfile.close()
'''
{'Peacekeeping and Peacebuilding', 'Coordination', 'Contributions', 'Safety and Security',
'Shelter and Non-Food Items', 'Education', 'Gender', 'Protection and Human Rights',
'Humanitarian Financing', 'Disaster Management', 'Food and Nutrition', 'HIV/Aids',
'Water Sanitation Hygiene', 'Health', 'Logistics and Telecommunications', 'Recovery and Reconstruction',
'Agriculture', 'Mine Action', 'Climate Change and Environment'}

    type2label_id = {'crimeviolence':8, 'med':3, 'search':4, 'food':1, 'out-of-domain':11, 'infra':2,
    'water':7, 'shelter':5, 'regimechange':9, 'evac':0, 'terrorism':10, 'utils':6}
'''
def convert_reliefweb_2_sf():
    # type2label_id = {'crimeviolence':8, 'med':3, 'search':4, 'food':1, 'out-of-domain':11, 'infra':2,
    # 'water':7, 'shelter':5, 'regimechange':9, 'evac':0, 'terrorism':10, 'utils':6}
    '''here, we use the old mapping, so that the training data load function has the same input as BBN'''
    type2label_id = {'crimeviolence':8, 'med':3, 'search':4, 'food':1, 'out-of-domain':9, 'infra':2, 'water':7, 'shelter':5,
    'regimechange':10, 'evac':0, 'terrorism':11, 'utils':6}
    typemap={
    'Safety and Security': ['terrorism', 'crimeviolence'],
    'Shelter and Non-Food Items': ['shelter'],
    'Food and Nutrition': ['food'],
    'HIV/Aids': ['med'],
    'Water Sanitation Hygiene': ['water'],
    'Health': ['med'],
    'Logistics and Telecommunications': ['utils'],
    'Recovery and Reconstruction': ['shelter', 'infra'],
    'Disaster Management': ['search','evac'],
    'Peacekeeping and Peacebuilding': ['regimechange'],
    'Coordination': ['out-of-domain'],
    'Contributions': ['out-of-domain'],
    'Education': ['out-of-domain'],
    'Gender': ['out-of-domain'],
    'Agriculture': ['out-of-domain'],
    'Mine Action': ['out-of-domain'],
    'Climate Change and Environment': ['out-of-domain'],
    'Protection and Human Rights': ['out-of-domain']

    }
    writefile=codecs.open('/scratch/wyin3/dickens_save_dataset/LORELEI/2019_new_data/reliefweb_2_sf.txt', 'w','utf-8')
    size=0
    with open('/scratch/wyin3/dickens_save_dataset/LORELEI/2019_new_data/reports.csv', mode='r') as csvfile:
        # spamreader = csv.reader(csvfile, delimiter=',')
        csv_reader = csv.DictReader(csvfile)
        for row in csv_reader:
            title = row['title']
            text = row['body']
            # print(row)
            if len(row['theme']) > 0 and len(row['country']) > 0:
                theme_list = ast.literal_eval(row['theme'])
                theam_id_list = []
                theam_name_list = []
                for theme in theme_list:
                    target_type_list = typemap.get(theme)
                    if target_type_list is not None:
                        for type in target_type_list:
                            if type not in theam_name_list:
                                if len(theam_name_list)==0 or (len(theam_name_list)>0 and type!='out-of-domain'):
                                    idd = type2label_id.get(type)
                                    theam_id_list.append(str(idd))
                                    theam_name_list.append(type)
                if len(theam_id_list)>0 and len(theam_name_list)>0:
                    writefile.write(' '.join(theam_id_list)+'\t'+' '.join(theam_name_list)+'\t'+text.strip()+'\n')
                    size+=1
                    if size %1000==0:
                        print('size:', size)



    writefile.close()
    csvfile.close()

if __name__ == '__main__':
    convert_reliefweb_2_sf()

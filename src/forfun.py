
import os

import codecs

IL11_path = '/Users/yinwenpeng/Downloads/LORELEI/2019/LoReHLT-2019-IL12/SF/Main-IL12/CP2/'

def scan_all_json_files(rootDir):
    global fileset
    for lists in os.listdir(rootDir):
        path = os.path.join(rootDir, lists)
        if os.path.isdir(path):
            filename = ['IL12_CP2_IL_text_NumGravityMethod1nonils_altref_standard_nDCG_Scores.txt', 'IL12_CP2_IL_text_NumGravityMethod1nonils_standard_nDCG_Scores.txt']
            cutoff = [141, 143]
            values = []
            for i in range(2):
                readfile = codecs.open(path+'/'+filename[i], 'r', 'utf-8')
                for line in readfile:
                    parts = line.strip().split()
                    if len(parts) ==2:
                        if parts[0] == str(cutoff[i]):
                            values.append(parts[1])
                readfile.close()
            print(values )


        else: # is a file
            continue

if __name__ == '__main__':
    scan_all_json_files(IL11_path)

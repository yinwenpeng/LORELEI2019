import io
import os
import sys
from sklearn.cross_decomposition import CCA
import numpy as np
DIM = int(sys.argv[5])

def vec2str(v):
    return ' '.join([str(x) for x in v])

def read_txt_embeddings(emb_path, vocab=None):
    word2id = {}
    vectors = []
    words = []

    # load pretrained embeddings
    with io.open(emb_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        for i, line in enumerate(f):
            if i == 0:
                split = line.split()
                assert len(split) == 2
                _emb_dim_file = int(split[1])
            else:
                word, vect = line.rstrip().split(' ', 1)
                word2id[word] = len(word2id)
                words.append(word)
                vect = np.fromstring(vect, sep=' ')
                norm = np.linalg.norm(vect)
                if  norm == 0:  # avoid to have null embeddings
                    vect[0] = 0.01
                else:
                    vect = vect / norm
                if not vect.shape == (_emb_dim_file,):
                    print("Invalid dimension (%i) for %s word '%s' in line %i."
                                   % (vect.shape[0], 'source' if source else 'target', word, i))
                    continue
                assert vect.shape == (_emb_dim_file,), i
                vectors.append(vect[None])

    print("Loaded %i pre-trained word embeddings." % len(vectors))

    # compute new vocabulary / embeddings
    embeddings = np.concatenate(vectors, 0)
    #embeddings = np.concatenate((np.zeros((1, embeddings.shape[1])), embeddings), 0)
    #embeddings = torch.from_numpy(embeddings).float()
    #embeddings = embeddings.cuda() if (params.cuda and not full_vocab) else embeddings

    return embeddings, word2id, words

def read_seed(path, emb1, emb2, dict1, dict2):
    '''load eng-il dictionary'''
    v1 = []
    v2 = []
    valid=0
    with open(path, 'r') as f:
        for line in f:
            pair = line.strip().split('\t')
            if len(pair) != 2:
                continue
            w1, w2 = pair
            if w1 in dict1 and w2 in dict2:
                i1 = dict1[w1]
                i2 = dict2[w2]
                v1.append(emb1[i1, :].reshape(1, -1))
                v2.append(emb2[i2, :].reshape(1, -1))
                valid+=1
    print('load valid dict pair size:', valid)
    return np.concatenate(v1, 0), np.concatenate(v2, 0)


'''
start the bicca
'''


eng_v, eng_dict, eng_vocab = read_txt_embeddings(sys.argv[1])
print('Loaded %d English embeddings' % len(eng_vocab))
il_v, il_dict, il_vocab = read_txt_embeddings(sys.argv[2])
print('Loaded %d IL embeddings' % len(il_vocab))

Ux, Uy = read_seed(sys.argv[3], eng_v, il_v, eng_dict, il_dict)
print('Loaded %d seeds' % Ux.shape[0])

cca = CCA(n_components=DIM, max_iter=2000) # scale=True,  tol=1e-06, copy=True
print('Fitting')
cca.fit(Ux, Uy)
print('Transforming')
eng_v, il_v = cca.transform(eng_v, il_v, copy=False)

# outvocab = set()
# with open('wiki-en-100k.vocab', 'r') as f:
#     for line in f:
#         outvocab.add(line.strip())
outvocab = set(eng_vocab)
print('Loaded %d vocab' % len(outvocab))
tmpset = set(eng_vocab)
for w in il_vocab:
    if w in tmpset:
        outvocab.add(w)
output_size = len([w for w in eng_vocab if w in outvocab])
print('Output size: %d' % output_size)

lang = 'IL'+sys.argv[4] if sys.argv[4].isdigit() else sys.argv[4]
outname = 'cp2-100k-%s-cca.d%d.%%s.txt' % (lang, DIM)

engout = outname % 'eng'
print('Writing to %s' % engout)
with open(engout, 'w') as f:
    f.write('%d %d\n' % (output_size, eng_v.shape[1]))
    #f.write(str.encode('%d %d\n' % (output_size, eng_v.shape[1])))
    #sel = []
    #for i, w in enumerate(eng_vocab):
    #    if w not in outvocab:
    #        continue
    #    sel.append(eng_v[i].reshape(1, -1))
    #sel = np.concatenate(sel, 0)
    for i, w in enumerate(eng_vocab):
        if w not in outvocab:
            continue
        #s = np.array2string(eng_v[i, :])[1:-1]
        eng_v[i, :] /= np.linalg.norm(eng_v[i, :])
        #s = np.array_str(eng_v[i, :], max_line_width=10000)[1:-1]
        s = vec2str(eng_v[i])
        f.write('%s %s\n' % (w, s))
        #f.write(str.encode(w))
        #f.write(b' ')
        #f.write(eng_v[i, :].tobytes())
        #f.write(b'\n')
ilout = outname % lang
print('Writing to %s' % ilout)
with open(ilout, 'w') as f:
    f.write('%d %d\n' % il_v.shape)
    #f.write(str.encode('%d %d\n' % il_v.shape))
    for i, w in enumerate(il_vocab):
        #s = np.array2string(il_v[i, :])[1:-1]
        il_v[i, :] /= np.linalg.norm(il_v[i, :])
        #s = np.array_str(il_v[i, :], max_line_width=10000)[1:-1]
        s = vec2str(il_v[i])
        f.write('%s %s\n' % (w, s))
        #f.write(str.encode(w))
        #f.write(b' ')
        #f.write(eng_v[i, :].tobytes())
        #f.write(b'\n')

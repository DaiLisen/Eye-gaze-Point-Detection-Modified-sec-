import numpy as np
import gmm

from mygmm import EM
from mygmm import gaussian
from PIL import Image
import matplotlib.pyplot as plt
import caffe
import sys
import scipy
sys.path.append('/media/a1234/software/haven/fcn.berkeleyvision.org-master')

#load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
# im = Image.open('./data/sift-flow/Images/spatial_envelope_256x256_static_8outdoorcategories/00007.jpg')
# in_ = np.array(im, dtype=np.float32)
# in_ = in_[:,:,::-1]
# in_ -= np.array((114.578, 115.294, 108.353))
# in_ = in_.transpose((2,0,1))



# load net
net = caffe.Net('siftflow-fcn32s/test.prototxt', 'siftflow-fcn32s/output_iter_6400.caffemodel', caffe.TEST)
# shape for input (data blob is N x C x H x W), set data
# net.blobs['data'].reshape(1, *in_.shape)
# net.blobs['data'].data[...] = in_


# run net and take argmax for prediction



net.forward()
dat = net.blobs['conv8_2_2'].data[...]
print dat.shape

para=[[[[0.0 for col in range(3)] for row in range(3)]for a in range(60)]for b in range(45)]
para=(np.array(para))
resu=[[0.0 for a in range(60)]for b in range(45)]
resu=(np.array(resu))

#
#
# out = net.blobs['score_sem1'].data[0,0,:,:]
# out = net.blobs['sem'].data[0]
# print net.blobs['conv8_1_2'].data[...].shape
# print net.blobs['conv8_1_2'].data[0].argmax(axis=0)
# print out.shape
# plt.figure(2)
# plt.imshow(out,cmap='gray');plt.axis('off')
# plt.savefig('test.png')
#
# plt.show()




####    test    ####
# indices = open('data/sift-flow/test.txt', 'r').read().splitlines()
#
# para=[[[[0.0 for col in range(3)] for row in range(3)]for a in range(60)]for b in range(45)]
# para=(np.array(para))
# resu=[[0.0 for a in range(60)]for b in range(45)]
# resu=(np.array(resu))
#
# for ct in range(400):
#     net.forward()
#     out = net.blobs['score_sem1'].data[0,0,:,:]
#     #out = net.blobs['sem'].data[0]
#     #plt.figure(2)
#
#     scipy.misc.imsave('res/{}.jpg'.format(indices[ct]), out)
#     print indices[ct]

    # net.forward()
    # dat = net.blobs['conv8_2_2'].data[...]
    # for i in range(45):
    #     for j in range(60):
    #         X = dat[0, :, i, j].T
    #         list1 = []
    #         for k in range(545):
    #             if X[k] == 0.0:
    #                 continue
    #             list1.append(np.log(X[k]))
    #         print i,j
    #         pPi, pMiu, pSigma=EM(list1, 3)
    #         para[i, j] = zip(pPi, pMiu, pSigma)
    # np.save('para.npy',para)
    #
    # para=np.load('para.npy')
    # for i in range(45):
    #     for j in range(60):
    #         X = dat[0, :, i, j].T
    #         phais = para[i, j, :, 0]
    #         mus = para[i, j, :, 1]
    #         sigmas = para[i, j, :, 2]
    #         resu[i,j]=0
    #         list1 = []
    #         for k in range(545):
    #             if X[k] == 0.0:
    #                 continue
    #             list1.append(np.log(X[k]))
    #         for k in xrange(len(list1)):
    #             q = [phai * gaussian(list1[k], mu, sigma) for phai, mu, sigma in zip(phais, mus, sigmas)]
    #             print i,j,q
    #             resu[i, j] += q[0] * list1[k]
    #             resu[i, j] += q[1] * list1[k]
    #             resu[i, j] += q[2] * list1[k]
    # #resu[4,19]=(resu[3,19]+resu[5,19]+resu[4,18]+resu[4,20])/4
    # resu[22,29]=(resu[21,29]+resu[23,29]+resu[22,28]+resu[22,30])/4
    # scipy.misc.imsave('res_my/{}.jpg'.format(indices[ct]), resu)
    # print indices[ct]












####    cacu Px      ####
para=np.load('para1.npy')
for i in range(45):
    for j in range(60):
        X = dat[0, :, i, j].T
        phais = para[i, j, :, 0]
        mus = para[i, j, :, 1]
        sigmas = para[i, j, :, 2]
        resu[i,j]=0
        list1 = []
        for k in range(545):
            if X[k] == 0.0:
                continue
            list1.append(np.log(X[k]))
        for k in xrange(len(list1)):
            q = [phai * gaussian(list1[k], mu, sigma) for phai, mu, sigma in zip(phais, mus, sigmas)]
            print i,j,q
            resu[i, j] += q[0]
            resu[i, j] += q[1]
            resu[i, j] += q[2]
#resu[4,19]=(resu[3,19]+resu[5,19]+resu[4,18]+resu[4,20])/4
resu[22,29]=(resu[21,29]+resu[23,29]+resu[22,28]+resu[22,30])/4
scipy.misc.imsave('1.jpg', resu)
####    gene para    ####
# for i in range(45):
#     for j in range(60):
#         X = dat[0, :, i, j].T
#         list1 = []
#         for k in range(545):
#             if X[k] == 0.0:
#                 continue
#             # if X[k] > 20.0:
#             #     continue
#             list1.append(np.log(X[k]))
#         print i,j
#         pPi, pMiu, pSigma=EM(list1, 3)
#         para[i, j] = zip(pPi, pMiu, pSigma)
# np.save('para1.npy',para)



####     debug    ####
# X = dat[0, :, 22, 29].T
# list1 = []
# for k in range(545):
#     if X[k] == 0.0:
#         continue
#     if X[k] > 10.0:
#         continue
#     list1.append(X[k])
# list1 = list(set(list1))
# phais, mus, sigmas=EM(list1, 3)
# a=0
#
# for k in xrange(len(list1)):
#     q = [phai * gaussian(list1[k], mu, sigma) for phai, mu, sigma in zip(phais, mus, sigmas)]
#     a  += q[0] * list1[k]
#     a  += q[1] * list1[k]
#     a  += q[2] * list1[k]
# print list1
# print a


#########################


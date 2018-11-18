import numpy as np
def gaussian(x,mu,sigma):
    temp = -np.square(x-mu)/(2*sigma)
    return np.exp(temp)/(np.sqrt(2.0*np.pi*sigma)) # sigma = sigma^2
def e_step(data, phais, mus, sigmas):
    Qs = []
    for i in xrange(len(data)):
        q = [phai*gaussian(data[i],mu,sigma) for phai,mu,sigma in zip(phais,mus,sigmas)]
        #print i,data[i ]
        Qs.append(q)
    Qs = np.array(Qs)
    Qs = Qs / np.sum(Qs,axis=1).reshape(-1,1)
    return Qs
def m_step(data, phais, mus, sigmas, Qs):
    data = np.array(data)
    gama_j = np.sum(Qs,axis=0)
    new_phais = gama_j/len(data)
    mu_temp = np.sum(Qs*(data.reshape(-1,1)),axis=0)
    new_mus =mu_temp/gama_j
    X_i_mu_j = np.square(np.array([data]).reshape(-1,1)-np.array([mus]))
    new_sigmas = np.sum(Qs*X_i_mu_j,axis=0)/gama_j
    return new_phais,new_mus,new_sigmas
def EM(data,k):
    threshold = 1e-15
    phais = [1.0/k for i in xrange(k)]
    mus = [i for i in xrange(k)]
    sigmas = [1 for i in xrange(k)]
    phais0, mus0, sigmas0=[0],[0],[0]
    # while True:
    #     Qs = e_step(data,phais,mus,sigmas)
    #     phais, mus, sigmas= m_step(data,phais,mus,sigmas,Qs)
    #     L1= [x-y for x,y in zip(phais0,phais)]
    #     L2 = [x - y for x, y in zip(mus0, mus)]
    #     L3 = [x - y for x, y in zip(sigmas0, sigmas)]
    #     L= np.sum(np.abs(np.array(L1)))  \
    #     + np.sum(np.abs(np.array(L2)))   \
    #     + np.sum(np.abs(np.array(L3)))
    #     phais0, mus0, sigmas0=phais, mus, sigmas
    #     print phais, mus, sigmas
    #     if L<threshold:
    #         break
    for i in range(100):
        Qs = e_step(data,phais,mus,sigmas)
        phais, mus, sigmas= m_step(data,phais,mus,sigmas,Qs)
        L1 = [x-y for x,y in zip(phais0,phais)]
        L2 = [x - y for x, y in zip(mus0, mus)]
        L3 = [x - y for x, y in zip(sigmas0, sigmas)]
        L= np.sum(np.abs(np.array(L1)))  \
        + np.sum(np.abs(np.array(L2)))   \
        + np.sum(np.abs(np.array(L3)))
        for j in range(3):
            if phais[j]==0.0:
                phais[j]=0.1e-50
            if mus[j]==0.0:
                mus[j]=0.1e-50
            if sigmas[j]==0.0:
                sigmas[j]=0.1e-50
        phais0, mus0, sigmas0 = phais, mus, sigmas
        #print phais, mus, sigmas
        if L<threshold:
            break
        if i==999:
            print "Time OUT"
    print phais, mus, sigmas
    return phais, mus, sigmas
# The code below generates an index set according to the 
# idea that is stated in the paper [1]. It assumes that you pass it
# an evidence structure of the form
#      
#       evidence = np.zeros([num_models,num_data_sets])
#       index = create_index_set(np.sum(evidence,axis=0))
#
# You can then plot the results as
#
#       pb.plot(evidence[0,index],'r')
#       pb.plot(evidence[1,index],'g')
#
# [1] I. Murray and Z. Ghahramani, “A note on the evidence and 
# Bayesian Occam’s razor,” Gatsby Unit Technical Report, GCNU-TR 2005–003, Aug. 2005.

def create_index_set(evidence):
    dist = np.zeros([evidence.shape[0],evidence.shape[0]])
    for i in range(dist.shape[0]):
        for j in range(dist.shape[1]):
            dist[i,j] = evidence[i]-evidence[j]
            if i==j:
                dist[i,j] = pow(10,4)

    L = [];
    D = np.arange(evidence.shape[0]).tolist()
    ind = evidence.argmin()
    L.append(ind)
    D.remove(ind)
    while D:
        N = []
        for i in range(len(D)):
            ind = dist[D[i],D].argmin()
            if D[ind]==L[-1]:
                N.append(D[ind])
        if not N:
            L.append(D[dist[L[-1],D].argmin()])
        else:
            L.append(N[dist[L[-1],N].argmin()])
        D.remove(L[-1])
    return L

from scipy.spatial.distance import pdist,squareform
import numpy as np
import time
import pandas as pd
import numpy.matlib
from math import exp,ceil
from sklearn.metrics import f1_score,accuracy_score,balanced_accuracy_score
from sklearn.preprocessing import MinMaxScaler
# Specify the label ratio here
global label_ratio 
label_ratio = 10


# Read Data from the csv. files
def Input():
    # Read Data from the csv. files
    sample = pd.read_csv('Syn-1.csv',header=None)

    [N,L] = np.shape(sample)
    dim = L # Extract the num of dimensions
    # Extract the label of the data from the data frame
    label1 = sample.iloc[:,L-1]
    label = label1.values
    # Extract the samples from the data frame
    data = sample.iloc[:,0:dim-1]
    # Normalization Procedure
    NewData = Data_Pre(data)

#    label_index = np.argsort(label)
#    NewData1 = NewData[label_index,:]
#    label1 = label[label_index]
    NewData1 = NewData#[:9999,:]
    label1 = label#[:9999]
    return NewData1,label1

# Max-min Normalization
def Data_Pre(data):
    [N,L] = np.shape(data)
    NewData = np.empty((N,L))
    scaler = MinMaxScaler()
    scaler.fit(data)
    NewData = scaler.transform(data)
    return NewData
# Parameter Specification:
def ParamSpe(data):
    Buffersize = 1000 # set the size of the data chunk
    PreStd = [] # Initialize the summary vector of variance for the data stream
    P_Summary = [] # Initialize the cluster summary vector
    PFS = [] # Initialize the summary vector to keep track of density values for cluster centers
    # Calculate the total number of chunks
    T = round(np.shape(data)[0] / Buffersize)
    return Buffersize, P_Summary, T, PFS, PreStd
# Perform the distance calculation
def Distance_Cal(data):
    D = pdist(data)
    Dist = squareform(D)
    return Dist
# Fitness Evaluation
def Fitness_Cal(sample,pop,stdData,gamma):
    Ns = np.shape(sample)[0]
    Np = np.shape(pop)[0]
    Newsample = np.concatenate([sample,pop])
    Dist = Distance_Cal(Newsample)
    fitness = []
    for i in range(Np):
        distArray = np.power(Dist[i+Ns,0:Ns],2)
        temp = np.power(np.exp(-distArray/stdData),gamma)
        fitness.append(np.sum(temp))
    return fitness
# Fitness Update from Historical Summary
def fitness_update(P_Summary, Current, fitness, PreStd, gamma, stdData):
    [N, dim] = np.shape(Current)
    t_I = len(PreStd)
    NewFit = fitness
    PreFit = P_Summary[:, dim]
    PreP = P_Summary[:, 0:dim]
    OldStd = PreStd[t_I - 1]
    dist_PreP = squareform(pdist(np.concatenate([Current, PreP])))
    if len(P_Summary) > 0:
        for i in range(N):
            fitin = 0
            tempdist_PreP = dist_PreP[i, N:]
            for j in range(np.shape(PreP)[0]):
                if tempdist_PreP[j] < 0.01:
                    fitin = PreFit[j]
                    break
                else:
                    d = tempdist_PreP[j]
                    fitin += (exp(-d ** 2 / stdData) ** gamma) * (PreFit[j] ** (OldStd / stdData))
            NewFit[i] = fitness[i] + fitin
    return NewFit
# Initialize the candidate set for searching temporary potential clusters (TPCs)
def PopInitial(sample, PreMu, PreStd, Buffersize,Dist):
    [n, l] = np.shape(sample)
    pop_Size = round(1 * n)
    # Compute the statistics of the current data chunk
    minLimit = np.min(sample, axis=0)
    meanData = np.mean(sample, axis=0)
    maxLimit = np.max(sample, axis=0)
    # Update the statistics of the data stream
    meanData = UpdateMean(PreMu, meanData, Buffersize)
    PreMu.append(meanData)
    # Compute the standard deviation of the current data chunk
    MD = np.matlib.repmat(meanData, n, 1)
    tempSum = np.sum(np.sum((MD - sample) ** 2, axis=1))
    stdData = tempSum / n
    # Update the standard deviation of the data stream
    stdData = StdUpdate(stdData, PreStd, Buffersize)
    # Randonmly Initialize the population indices from the data chunk
    pop_Index = np.arange(0, n)
    pop = sample[pop_Index, :]
    # Calculate the initial niche radius
    radius = numpy.linalg.norm((maxLimit - minLimit)) * 0.1 
    return [stdData, pop_Index, pop, radius, PreMu, PreStd]
# Update the mean of the data stream as new data chunk arrives
def UpdateMean(PreMu, meanData, BufferSize):
    # Num of the processed data chunk
    t_P = len(PreMu)
    # Update the mean of the data stream as new data chunk arrives
    if t_P == 0:
        newMu = meanData
    else:
        oldMu = PreMu[t_P - 1][:]
        newMu = (meanData + oldMu * t_P) / (t_P + 1)
    return newMu
# Update the variance of the data stream as new data chunk arrives
def StdUpdate(Std, PreStd, BufferSize):
    # Num of the processed data chunk
    t_P = len(PreStd)
    # Update the variance of the data stream as new data chunk arrives
    if t_P == 0:
        newStd = Std
    else:
        oldStd = PreStd[t_P - 1]
        newStd = (Std + oldStd * t_P) / (t_P + 1)
    return newStd
# ------------------------Parameter Estimation for gamma----------------------------#
def CCA(sample, stdData, Dist):
    m = 1
    gamma = 5
    ep = 0.995  # 0.995
    N = np.shape(sample)[0]
    while 1:
        den1 = []
        den2 = []
        for i in range(N):
            Diff = np.power(Dist[i, :], 2)
            temp1 = np.power(np.exp(-Diff / stdData), gamma * m)
            temp2 = np.power(np.exp(-Diff / stdData), gamma * (m + 1))
            den1.append(np.sum(temp1))
            den2.append(np.sum(temp2))
        y = np.corrcoef(den1, den2)[0, 1]
        if y > ep:
            break
        m = m + 1
    return m * gamma
def DCCA(sample,stdData,P_Summary,gamma,dim):
    P_Center = P_Summary[:,0:dim] # Historical cluster centers in the cluster summary
    P_F = P_Summary[:,dim] # Density values for historical clusters
    gam1 = gamma # Gamm value at t-1
    N1 = np.shape(sample)[0]
    N2 = np.shape(P_Center)[0]
    ep = 0.95 # Threshold value for correlation comparison 0.985
    N = N1 + N2
    # Concatenate samples and historical cluster centers together
    temp = np.concatenate([sample,P_Center],axis=0)
    # Distance calculation for the concatenated set
    Dist = Distance_Cal(temp)
    while 1:
        gam2 = gam1 + 5
        den1 = []
        den2 = []
        for i in range(N):
            Diff = np.power(Dist[i,0:N1],2)
            temp1 = np.power(np.exp(-Diff/stdData),gam1)
            temp2 = np.power(np.exp(-Diff/stdData),gam2)
            sum1 = np.sum(temp1)
            sum2 = np.sum(temp2)
            if i<N1:
                T1 = 0
                T2 = 0
                for j in range(N2):
                    T1 += P_F[j]**(gam1/gamma)
                    T2 += P_F[j]**(gam2/gamma)
                s1 = sum1 + T1
                s2 = sum2 + T2
            else:
                s1 = sum1 + P_F[i-N1]**(gam1/gamma)
                s2 = sum2 + P_F[i-N1]**(gam2/gamma)
            den1.append(s1)
            den2.append(s2)
        y = np.corrcoef(den1,den2)[0,1]
        if y > ep:
            break
        gam1 = gam2
    return gam1
# Perform the TPC search among the population
def TPC_Search(Dist, Pop_Index, Pop, radius, fitness):
    # Extract the size of the population
    [N, dim] = np.shape(Pop)
    P = []  # Initialize the TPC Vector
    P_fitness = [] # Initialize the fitness vector for TPCs
    marked = [] # A vector to accumulate the indices of samples have been assigned to TPCs
    co = [] # The
    OriginalIndice = Pop_Index
    OriginalFit = fitness
    OriginalPop = Pop
    PeakIndice = []
    TPC_Indice = OriginalIndice
    while 1:
        # -------------Search for the local maximum-----------------#
        SortIndice = np.argsort(fitness)
        NewIndice = SortIndice[::-1]

        Pop = Pop[NewIndice, :]
        fitness = fitness[NewIndice]
        OriginalIndice = OriginalIndice[NewIndice]

        P.append(Pop[0, :])
        P_fitness.append(fitness[0])
        P_Indice = OriginalIndice[0]

        PeakIndice.append(np.where(OriginalFit == fitness[0])[0][0])
        Ind = AssigntoPeaks(Pop, Pop_Index, P, P_Indice, marked, radius, Dist)

        marked.append(Ind)
        marked.append(NewIndice[0])

        if not Ind:
            Ind = [NewIndice[0]]

        co.append(len(Ind))
        TempFit = fitness
        sum1 = 0

        TPC_Indice[Ind] = np.where(OriginalFit == fitness[0])[0][0]

        for j in range(len(Ind)):
            sum1 += fitness[np.where(OriginalIndice == Ind[j])]
        for th in range(len(Ind)):
            TempFit[np.where(OriginalIndice == Ind[th])] = fitness[np.where(OriginalIndice == Ind[th])] / (1 + sum1)
        fitness = TempFit
        if np.sum(co) >= N:
            P2 = OriginalPop[PeakIndice][:]
            P = np.asarray(P2)
            P_fitness = np.asarray(P_fitness)
            TPC_Indice = Close_Clusters(Pop, PeakIndice, Dist)
            break
    return P, P_fitness, TPC_Indice, PeakIndice
# Find the cluster indices for samples
def Close_Clusters(pop, PeakIndices, Dist):
    P = pop[PeakIndices][:]
    C_Indices = np.arange(0, np.shape(pop)[0])
    for i in range(np.shape(pop)[0]):
        temp_dist = Dist[i][PeakIndices]
        C_Indices[i] = PeakIndices[np.argmin(temp_dist)]
    return C_Indices
# Compute the closest cluster indice for all samples and return the minimum distance
def Cluster_Assign(sample, P):
    # Number of samples
    N = np.shape(sample)[0]
    # Number of Clusters at t
    Np = np.shape(P)[0]
    MinDist = []
    MinIndice = []
    dist_toP = squareform(pdist(np.concatenate([P, sample], axis=0)))
    for i in range(N):
        d = dist_toP[i+Np, :Np]
        if len(d) <= 1:
            tempD = d
            tempI = 0
        else:
            tempD = np.min(d)
            tempI = np.argmin(d)
        MinDist.append(tempD)
        MinIndice.append(tempI)
    MinDist = np.asarray(MinDist)
    MinIndice = np.asarray(MinIndice)
    return MinDist, MinIndice
# Perform the merge of TPCs in the current data chunk Ct
def MergeInChunk(P, P_fitness, sample, gamma, stdData, Dist, TPC_Indice, PeakIndices):
    """Perform the Merge of TPCs witnin each data chunk
    """
    # Num of TPCs
    [Nc, dim] = np.shape(P)
    NewP = []
    NewP_fitness = []
    NewPeakIndices = []
    marked = []
    unmarked = []
    Com = []
    
    PDist = squareform(pdist(P))
    # Num of TPCs
    Nc = np.shape(P)[0]
    for i in range(Nc):
        MinDist = np.inf
        MinIndice = 100000
        if i not in marked:
            for j in range(Nc):
                if j != i and j not in marked:
                    d = PDist[j,i]
                    # d = np.linalg.norm(P[j, :] - P[i, :])
                    if d < MinDist:
                        MinDist = d
                        MinIndice = j
            if MinIndice <= Nc:
                MinIndice = int(MinIndice)
                Merge = True
                Neighbor = PeakIndices[MinIndice]
                Current = PeakIndices[i]
                
                X = (sample[Neighbor,:] + sample[Current,:]) / 2
                #X = Boundary_Instance(Current, Neighbor, Dist, TPC_Indice, sample)
                X = np.reshape(X, (1, np.shape(P)[1]))

                fitX = Fitness_Cal(sample, X, stdData, gamma)
                fitP = P_fitness[i]
                fitN = P_fitness[MinIndice]
                if fitX < 0.85 * min(fitN, fitP):  # 0.95 Aggregation
                    Merge = False
                if Merge:
                    Com.append([i, MinIndice])
                    marked.append(MinIndice)
                    marked.append(i)
                else:
                    unmarked.append(i)
    Com = np.asarray(Com)
    # Number of Possible Merges:
    Nm = np.shape(Com)[0]
    for k in range(Nm):
        if P_fitness[Com[k, 0]] >= P_fitness[Com[k, 1]]:
            NewPeakIndices.append(PeakIndices[Com[k, 0]])
        else:
            NewPeakIndices.append(PeakIndices[Com[k, 1]])
    # Add Unmerged TPCs to the NewP
    for n in range(Nc):
        if n not in Com:
            NewPeakIndices.append(PeakIndices[n])
    NewPeakIndices = np.unique(NewPeakIndices)
    NewP = sample[NewPeakIndices,:]
    NewP_fitness = Fitness_Cal(sample, NewP, stdData, gamma)
    TPC_Indice = Close_Clusters(sample, NewPeakIndices, Dist)
    return NewP, NewP_fitness, TPC_Indice, NewPeakIndices
# Perform the merge among TPCs inside the current data chunk Ct
def CE_InChunk(sample, P, P_fitness, stdData, gamma, Dist, TPC_Indice, PeakIndices):
    while 1:
        HistP = P
        P, P_fitness, TPC_Indice, NewPeakIndices = MergeInChunk(P, P_fitness, sample, gamma, stdData, Dist, TPC_Indice, PeakIndices)
        if np.shape(P)[0] == np.shape(HistP)[0]:
            break
    return P, P_fitness
# Identify samples that are close to the boundary between two neighboring TPCs
def Boundary_Instance(Current, Neighbor, Dist, TPC_Indice, sample):
    temp_cluster1 = np.where(TPC_Indice == Current)[0]
    temp_cluster2 = np.where(TPC_Indice == Neighbor)[0]
    temp = np.concatenate([temp_cluster1, temp_cluster2])

    Dc = Dist[Current][Neighbor]
    Dd = []
    for i in range(len(temp)):
        D1 = Dist[Current][temp[i]]
        D2 = Dist[Neighbor][temp[i]]
        Dd.append(abs(D1 - D2))
    if len(Dd) <= 1:
        BD = sample[Current][:]
    else:
        CI = np.argmin(Dd)
        BD = sample[temp[CI]][:]
    return BD
# Assign samples to its nearest cluster centers and obtain their cluster indices
def AssigntoPeaks(pop,pop_index,P,P_I,marked,radius,Dist):
    temp = []
    [N,L] = np.shape(pop)
    for i in range(N):
        distance = Dist[i,P_I]
        if not np.any(marked==pop_index[i]):
            if distance < radius:
                temp.append(pop_index[i])
    indices = temp
    return indices
# Secondary merge between clusters doscover in Ct and clusters in the cluster summary
def MegrewithExisting(P, P_Summary, sample, stdData, gamma, PreStd):
    dim = np.shape(P)[1]
    Hist_Cluster = P_Summary[:, :dim]
    Hist_T = P_Summary[:, dim + 2]

    concate_clusters = np.concatenate([P, Hist_Cluster])
    concate_dist = squareform(pdist(concate_clusters))
    [min_dist, cluster_indices] = Cluster_Assign(sample, P)

    RPF = Fitness_Cal(sample, concate_clusters, stdData, gamma)
    PF = fitness_update(P_Summary, concate_clusters, RPF, PreStd, gamma, stdData)

    merge_flag = np.zeros(np.shape(P)[0])
    merge_nei = []

    merged_cluster = []
    merged_fit = []

    novel_cluster = []
    novel_fit = []

    re_cluster = []
    re_fit = []
    re_T = []
    for i in range(np.shape(P)[0]):
        Merge = True
        Current = P[i, :]
        Current = Current.reshape(1, dim)
        NeighborDist = concate_dist[i][-np.shape(Hist_Cluster)[0]:]
        Neighbor_Index = np.argmin(NeighborDist)
        if not np.any(np.isin(merge_nei, Neighbor_Index)):
            Neighbor = Hist_Cluster[Neighbor_Index][:]
            Neighbor = Neighbor.reshape(1, dim)
            current_cluster = np.where(cluster_indices == i)[0]
            temp_concatenate = np.concatenate([Current, Neighbor])
            temp_concatenate = np.concatenate([temp_concatenate, sample[current_cluster][:]])
            tempconcate_dist = squareform(pdist(temp_concatenate))

            dist_diff = tempconcate_dist[0][:] - tempconcate_dist[1][:]

            dist_diff = np.absolute(dist_diff)

            boundary_index = current_cluster[np.argmin(dist_diff[2:])]
            boundary_sample = sample[boundary_index][:]
            boundary_sample = boundary_sample.reshape(1,dim)

            R_fitness = Fitness_Cal(sample, boundary_sample, stdData, gamma)
            boundary_fitness = fitness_update(P_Summary, boundary_sample, R_fitness, PreStd, gamma, stdData)

            current_fit = PF[i]
            neighbor_fit = PF[np.shape(P)[0] + Neighbor_Index]

            if boundary_fitness < 0.85 * min(current_fit, neighbor_fit): #0.98
                Merge = False
                merge_flag[i] = 1

            if Merge:
                merge_nei.append(Neighbor_Index)
                if neighbor_fit <= current_fit:
                    merged_cluster.append(Current)
                    merged_fit.append(current_fit)
                else:
                    merged_cluster.append(Neighbor)
                    merged_fit.append(neighbor_fit)
            else:
                novel_cluster.append(Current)
                novel_fit.append(current_fit)
        else:
            novel_cluster.append(Current)
            novel_fit.append(current_fit)

    for j in range(np.shape(P_Summary)[0]):
        if j not in merge_nei:
            re_cluster.append(Hist_Cluster[j][:])
            re_fit.append(PF[j + np.shape(P)[0]])
            re_T.append(Hist_T[j])
    merged_cluster = np.reshape(merged_cluster, (np.shape(merged_cluster)[0], dim))
    novel_cluster = np.reshape(novel_cluster, (np.shape(novel_cluster)[0], dim))
    re_cluster = np.reshape(re_cluster, (np.shape(re_cluster)[0], dim))
    re_T = np.array(re_T)
    return merged_cluster, merged_fit, novel_cluster, novel_fit, re_cluster, re_fit, re_T
# Extarct and update the cluster summary
def ClusterSummary(P,PF,P_Summary,sample,TC,ClusterIndice):
    dim = np.shape(sample)[1]
    P = np.asarray(P)
    PF = np.asarray(PF)
    Rp = AverageDist(P, P_Summary, sample, dim)

    Rp = np.reshape(Rp, (np.shape(P)[0], 1))
    TC = np.reshape(TC, (np.shape(P)[0], 1))
    PF = np.reshape(PF, (np.shape(P)[0], 1))
    
    PCluster = np.concatenate([P, PF], axis=1)
    PCluster = np.concatenate([PCluster, Rp], axis=1)
    PCluster = np.concatenate([PCluster, TC], axis=1)
    P_Summary = PCluster
    return P_Summary
# To validate the existence of novel clusters
def NovelClusterValidate(merged_cluster, merged_fit,novel_clusters,novel_fit,P_Summary):
    if np.shape(P_Summary)[0] <= 1:
        return novel_clusters,novel_fit
    [N_no,dim] = np.shape(novel_clusters)
    hist_fit = P_Summary[:,dim]
    meanhist = np.mean(hist_fit)
    stdhist = np.std(hist_fit)
    v_hist = []
    v_histf = []
    ci = 0
    for i in range(N_no):
        if novel_fit[i] >= (meanhist-1*stdhist):
            v_hist.append(novel_clusters[i,:])
            v_histf.append(novel_fit[i])
            ci += 1
    novel_clusters = np.reshape(v_hist,(ci,dim))
    novel_fit = v_histf
    return novel_clusters,novel_fit
# Keep the track of the variance of the data stream and the density values of cluster centers
def StoreInf(PF, PFS, PreStd, stdData):
    PreStd.append(stdData)
    PFS.append(PF)
    return PreStd, PFS
# --------------------Cluster Radius Computation and Update--------------------#
def AverageDist(P, P_Summary, sample, dim):
    P = P
    # Obtain the assignment of clusters
    [distance, indices] = Cluster_Assign(sample, P)
    rad1 = []
    # if the summary of clusters is not empty
    if len(P_Summary) > 0:
        PreP = P_Summary[:, 0:dim]  # Historical Cluster Center vector
        PreR = P_Summary[:, dim + 1]  # Historical Cluster Radius
        dist_PtoPreP = squareform(pdist(np.concatenate([P, PreP])))

        for i in range(np.shape(P)[0]):
            if np.shape(np.where(indices == i))[1] > 1:
                SumD1 = 0
                Count1 = 0
                for j in range(np.shape(sample)[0]):
                    if indices[j] == i:
                        SumD1 += distance[j]
                        Count1 += 1
                rad1.append(SumD1 / Count1)
            else:
                C_d = dist_PtoPreP[i, np.shape(P)[0]:]
                CI = np.argmin(C_d)
                rad1.append(PreR[CI])
    elif not P_Summary:
        for i in range(np.shape(P)[0]):
            SumD1 = 0
            Count1 = 0
            for j in range(np.shape(sample)[0]):
                if indices[j] == i:
                    SumD1 += distance[j]
                    Count1 += 1
            rad1.append(SumD1 / Count1)
    return np.asarray(rad1)

# Perform the active learning for novel class instances
def active_labelquery(num_S, P, sample, temp_labeler, ClusterIndice, num_re):
    FetchIndex = []
    UnlabeledIndex = []
    InterDist = squareform(pdist(P))

    if np.shape(P)[0] <= 1:
        tempcluster = np.where(ClusterIndice == num_re)[0]
        concate_temp = np.concatenate([P,sample[tempcluster,:]])
    
        
        con_d1 = squareform(pdist(concate_temp))
        d1 = con_d1[0,1:]

        fetchSize = num_S * len(d1) / np.shape(sample)[0]
        sortIndex1 = np.argsort(d1)
        fet1 = tempcluster[sortIndex1[:ceil(fetchSize * 0.5)]]
        fet1 = fet1.astype(int)
        fet2 = tempcluster[sortIndex1[-ceil(fetchSize * 0.5):]]
        fet2 = fet2.astype(int)
        FetchIndex = np.append(FetchIndex, fet1)
        FetchIndex = np.append(FetchIndex, fet2)

        FetchIndex = FetchIndex.astype(int)
        for ui in range(len(temp_labeler)):
            if not np.any(np.isin(FetchIndex, ui)):
                UnlabeledIndex.append(ui)
        UnlabeledIndex = np.asarray(UnlabeledIndex)
        UnlabeledIndex = UnlabeledIndex.astype(int)
        return FetchIndex, UnlabeledIndex

    for i in range(np.shape(P)[0]):
        tempcluster = np.where(ClusterIndice == (i + num_re))[0]
        temp_interdist = InterDist[i, :]
        temp_rank = np.argsort(temp_interdist)
        temp_neigh1 = P[temp_rank[0], :]
        temp_neigh2 = P[temp_rank[1], :]
        
        # d1 = []
        currentp = P[i][:]
        currentp = np.reshape(currentp,(1,np.shape(sample)[1]))
        concate_temp = np.concatenate([currentp,sample[tempcluster,:]])
    
        
        con_d1 = squareform(pdist(concate_temp))
        d1 = con_d1[0,1:]

        fetchSize = round(num_S * len(d1) / np.shape(sample)[0])
        sortIndex1 = np.argsort(d1)
        fet1 = tempcluster[sortIndex1[:round(fetchSize * 0.5)]]
        fet1 = fet1.astype(int)

        fil_index = sortIndex1[-round(len(d1) / 2):]
        concate_temp2 = np.concatenate([temp_neigh1,temp_neigh2])
        concate_temp2 = np.reshape(concate_temp2,(2,np.shape(sample)[1]))
        concate_temp2 = np.concatenate([concate_temp2,sample[tempcluster[fil_index], :]])
        con_d2 = squareform(pdist(concate_temp2))
        d2_mat = con_d2[:2,2:]
        d2 = np.sum(d2_mat,axis=0)
        
        sortIndex2 = np.argsort(d2)
        fet2 = sortIndex2[:ceil(fetchSize * 0.5)]
        fet2 = fet2.astype(int)
        FetchIndex = np.append(FetchIndex, fet1)
        FetchIndex = np.append(FetchIndex, fet2)
    for ui in range(np.shape(sample)[0]):
        if not np.any(np.isin(FetchIndex, ui)):
            UnlabeledIndex.append(ui)

    FetchIndex = FetchIndex.astype(int)
    UnlabeledIndex = np.asarray(UnlabeledIndex)
    UnlabeledIndex = UnlabeledIndex.astype(int)

    return FetchIndex, UnlabeledIndex
# Propagate queried labels to the remaining unlabeled novel class instances
def Label_Propagation(sample_Fetch, label_Fetch, sample_Unlabeled):
   Y = []
   for x in sample_Unlabeled:
       vote_dist = []
       for y in sample_Fetch:
           vote_dist.append(np.linalg.norm(x - y))
       vote_order = np.argsort(vote_dist)
       if len(vote_order) < 3:
           vote_index = 0
           vote_res = label_Fetch[vote_index]
       else:
           vote_index = vote_order[:3]
           voter = label_Fetch[vote_index]
           [vote_l, vote_c] = np.unique(voter, return_counts=True)
           vote_res = vote_l[np.argmax(vote_c)]
       Y.append(vote_res)
   label_Unfetched = np.asarray(Y)
   return label_Unfetched
# Classify unlabeled instances that belong to the merged and unchanged clusters
def classifylabel(subcluster_info, P, P_Summary, sample, ClusterIndice):
    num_hist = len(subcluster_info)
    num = len(ClusterIndice)
    dim = np.shape(P)[1]
    hist_center = P_Summary[:, :dim]
    obtained_label = np.zeros((num,), dtype=int)
    for i in range(np.shape(P)[0]):
        dist = []
        cidx = np.where(ClusterIndice == i)[0]
        for j in range(num_hist):
            dist.append(np.linalg.norm(P[i, :] - hist_center))
        close_idx = np.argmin(dist)
        hist_info = subcluster_info[str(close_idx)]
        hist_label = list(hist_info.keys())
        subcenter = np.zeros((len(hist_label), dim))
        index1 = 0
        subcenter_count = np.zeros((len(hist_label),))
        for key in hist_label:
            temp_center1 = hist_info[str(key)]
            subcenter[index1, :] = temp_center1[:, :dim]
            subcenter_count[index1] = temp_center1[:, dim + 1]
            index1 += 1
        temp_combine = np.concatenate([subcenter, sample[cidx, :]])
        temp_distmatrix = squareform(pdist(temp_combine))
        truncated_dist = temp_distmatrix[:len(hist_label), len(hist_label):]

        min_idx = np.argmin(truncated_dist, axis=0)
        for k in range(len(min_idx)):
            obtained_label[cidx[k]] = hist_label[min_idx[k]]
    return obtained_label
# Update the clustering model in terms of subcluster information
def UpdateExistModel(ClusterIndice, Cluster, P_Summary, sample, obtained_label, subcluster_info, num_no,stdData,gamma):
    dim = np.shape(Cluster)[1]
    hist_center = P_Summary[:, :dim]
    curr_center = Cluster

    num_c = np.shape(curr_center)[0]
    
    update_concate = np.concatenate([curr_center,hist_center])
    
    update_dist = squareform(pdist(update_concate))
    
    update_dist2 = update_dist[:num_c,num_c:]

    for i in range(num_c - num_no):
        current_clusterind = np.where(ClusterIndice == i)[0]
        inter_dist = update_dist2[i,:]
        nearest_ind = np.argmin(inter_dist)

        histsub_info = subcluster_info[str(nearest_ind)]
        hist_labels = np.asarray(list(histsub_info.keys()))

        current_labels = obtained_label[current_clusterind]
        current_labelinfo = np.unique(current_labels)

        for l in current_labelinfo:
            sample_sub = np.where(current_labels == l)[0]
            subcluster_curr = sample[current_clusterind[sample_sub], :]
            subcluster_fit = Fitness_Cal(subcluster_curr,subcluster_curr,stdData,gamma)
            curr_submean = subcluster_curr[np.argmax(subcluster_fit)]
            curr_submean = curr_submean.reshape(1, dim)
            curr_substd = np.sum(np.square(np.std(subcluster_curr, axis=0))) ** 0.5
            curr_subnum = np.shape(subcluster_curr)[0]
            if np.any(np.isin(hist_labels, str(l))):
                hist_sub = histsub_info[str(l)]
                hist_submean = hist_sub[:, :dim]
                hist_substd = hist_sub[:, dim]
                hist_subnum = hist_sub[:, dim + 1]

                sub_num = hist_subnum + curr_subnum

                hist_submeanf = Fitness_Cal(subcluster_curr,hist_submean,stdData,gamma)
                if hist_submeanf > subcluster_fit:
                    sub_mean = hist_submean
                else:
                    sub_mean = curr_submean
                sub_std = ((hist_substd ** 2 * hist_subnum + curr_substd ** 2 * curr_subnum) / (sub_num)) ** 0.5
                hist_sub[:, :dim] = sub_mean
                hist_sub[:, dim] = sub_std
                hist_sub[:, dim + 1] = sub_num

                histsub_info[str(l)] = hist_sub
            else:
                new_sub = np.zeros((1, dim + 2))
                new_sub[:, :dim] = curr_submean
                new_sub[:, dim] = curr_substd
                new_sub[:, dim + 1] = curr_subnum

                if curr_subnum >= 10:
                    histsub_info.update({str(l): new_sub})

        subcluster_info[str(nearest_ind)] = histsub_info
    return subcluster_info
# Create new clustering models for novel clusters
def CreateNewModel(ClusterIndice, Cluster, P_Summary, sample, obtained_label, subcluster_info, num_no,stdData,gamma):
    dim = np.shape(Cluster)[1]
    curr_center = Cluster

    num_c = np.shape(curr_center)[0]

    for i in range(num_c - num_no, num_c):
        novel_clusterind = np.where(ClusterIndice == i)[0]
        novel_labels = obtained_label[novel_clusterind]
        novel_labelinfo = np.unique(novel_labels)

        insert_subinfo = {}
        for lnew in novel_labelinfo:
            insert_info = np.zeros((1, dim + 2))
            sample_sub = np.where(novel_labels == lnew)[0]
            subcluster_novel = sample[novel_clusterind[sample_sub], :]

            subcluster_fitness = Fitness_Cal(sample[sample_sub,:],sample[sample_sub,:],stdData,gamma)
            novel_submean = sample_sub[np.argmax(subcluster_fitness, axis=0)]
            novel_substd = np.sum(np.square(np.std(subcluster_novel, axis=0))) ** 0.5
            novel_subnum = np.shape(subcluster_novel)[0]
            insert_info[0, :dim] = novel_submean
            insert_info[0, dim] = novel_substd
            insert_info[0, dim + 1] = novel_subnum
            if novel_subnum > 10 :
                insert_subinfo.update({str(lnew): insert_info})

        subcluster_info.update({str(i): insert_subinfo})

    return subcluster_info
def Extract_Labels(subcluster_info,dim):
    ln = 0
    cluster_labels = list(subcluster_info.keys())

    for l in cluster_labels:
        temp_clusterinfo = subcluster_info[str(l)]
        sublabels = list(temp_clusterinfo.keys())
        for lsub in sublabels:
            content = temp_clusterinfo[str(lsub)]
            if ln == 0:
                represent = content[:,:dim]
                represent_label = int(lsub)
            else:
                represent = np.append(represent,content[:, :dim])
                represent_label = np.append(represent_label,int(lsub))
            ln += 1
    represent = np.reshape(represent,(ln,dim))
    return represent,represent_label

# Identify outliers when the variance of the data stream decreases
def IdentifyOutliers(P_Summary, sample):
    [nc, dim] = np.shape(sample)
    hist_centers = P_Summary[:, :dim]
    hist_radius = P_Summary[:, dim + 1]

    outlier_idx = []

    accumlate_sample = np.concatenate([hist_centers, sample])
    accumlate_dist = squareform(pdist(accumlate_sample))

    truncated_dist = accumlate_dist[nc:, :nc]

    nearest_idx = np.argmin(truncated_dist, axis=0)
    min_dist = np.min(truncated_dist, axis=0)

    for i in range(np.shape(sample)[0]):
        if min_dist[i] <= hist_radius[nearest_idx[i]]:
            outlier_idx.append(i)
    return outlier_idx

def leastconfidence(N_active,subreps,sublabelrep, sample, idx_classify):
    N_sub =  np.shape(subreps)[0]
    subrepsclassify = np.concatenate([subreps,sample[idx_classify,:]])
    classify_dist = squareform(pdist(subrepsclassify))
    classify_dist2 = classify_dist[N_sub:,:N_sub]

    sumdist = []
    for i in range(np.shape(classify_dist2)[0]):
        tempsubdist = classify_dist2[i,:]
        sumdist.append(np.sum(np.std(tempsubdist)))  
    sort_sumdist = np.argsort(sumdist)
    return idx_classify[sort_sumdist[:N_active]]


# ---------------------------Main Function-------------------------#
if __name__ == '__main__':
    [data, label] = Input()
    dim = np.shape(data)[1]

    [BufferSize, P_Summary, T, PFS, PreStd] = ParamSpe(data)
    

    
    T = int(T)
    F1Hist = []
    BAcc1_Hist = []
    Tc = []

    label_info = {}
    subcluster_info = {}

    gammaHist = []
    PFS = []
    PreMu = []
    queryhist = []
    acc_predlabel = []
    acc_truelabel = []
    for t in range(T):

        if t < T - 1:
            sample = data[t * BufferSize:(t + 1) * BufferSize, :]
            temp_labeler = label[t * BufferSize:(t + 1) * BufferSize]
        else:
            sample = data[t * BufferSize:np.shape(data)[0]]
            temp_labeler = label[t * BufferSize:np.shape(data)[0]]

        if t == 0:
            AccSample = sample
            AccLabel = temp_labeler
        else:
            AccSample = np.concatenate([AccSample, sample])
            AccLabel = np.concatenate([AccLabel, temp_labeler])

        dim = np.shape(sample)[1]
        Dist = Distance_Cal(sample)
        [stdData, pop_index, pop, radius, PreMu, PreStd] = PopInitial(sample, PreMu, PreStd, BufferSize, Dist)
        # Initialize the fitness vector
        fitness = np.zeros((len(pop_index), 1))
        # Initialize the indices vector
        indices = np.zeros((len(pop_index), 1))

        
        print("-------------Data Stream at time T=" + str(t) + "-------------")
        if PreStd:
            gam = gamma
            gamma = DCCA(sample, stdData, P_Summary, gam, dim)

        else:
            gamma = CCA(sample, stdData, Dist)

        gammaHist.append(gamma)

        fitness = Fitness_Cal(sample, pop, stdData, gamma)
        fitness = np.array(fitness)

        P, P_fitness, TPC_Indice, PeakIndices = TPC_Search(Dist, pop_index, pop, radius, fitness)

        P, P_fitness = CE_InChunk(sample, P, P_fitness, stdData, gamma, Dist, TPC_Indice, PeakIndices)

        if t == 0:
            P = P
            #            Rp = AverageDist(P, P_Summary, sample, dim)
            PF = np.asarray(P_fitness)
            TC = np.zeros(np.shape(P)[0])
            num_re = 0
            num_me = 0
            num_no = 1

            print("No. of clusters that are detected initially: " + str(np.shape(P)[0]))
        else:
            P_fitness = fitness_update(P_Summary, P, P_fitness, PreStd, gamma, stdData)
            [merged_cluster, merged_fit, novel_cluster, novel_fit, re_cluster, re_fit, re_T] = MegrewithExisting(P,
                                                                                                                 P_Summary,
                                                                                                                 sample,
                                                                                                                 stdData,
                                                                                                                 gamma,
                                                                                                                 PreStd)
            # novel_cluster, novel_fit = NovelClusterValidate(merged_cluster, merged_fit,novel_cluster, novel_fit, P_Summary)
            merged_T = t * np.ones(np.shape(merged_cluster)[0])
            novel_T = t * np.ones(np.shape(novel_cluster)[0])

            P = np.concatenate([re_cluster, merged_cluster])
            P = np.concatenate([P, novel_cluster])
            P_fitness = np.concatenate([re_fit, merged_fit])
            P_fitness = np.concatenate([P_fitness, novel_fit])

            TC = np.concatenate([re_T, merged_T])
            TC = np.concatenate([TC, novel_T])

            PF = np.asarray(P_fitness)

            num_re = np.shape(re_cluster)[0]
            num_me = np.shape(merged_cluster)[0]
            num_no = np.shape(novel_cluster)[0]

            print("No. of Unchanged Clusters: " + str(np.shape(re_cluster)[0]))
            print("No. of Merged Clusters: " + str(np.shape(merged_cluster)[0]))
            if np.shape(novel_cluster)[0] > 0:
                print("No. of Detected Novel Clusters: " + str(np.shape(novel_cluster)[0]))
            else:
                print("No novel cluster appears!")

        # ------------------active label querying and classify-----------------%

        [MinDist, ClusterIndice] = Cluster_Assign(sample, P)

        if t == 0:
            num_S = round(label_ratio/100 * len(ClusterIndice))
        else:
            num_S = round(label_ratio/100 * len(ClusterIndice))
        if num_S % 2:
            num_S = num_S
        else:
            num_S = num_S + 1

        obtained_label = np.zeros(np.shape(temp_labeler), dtype=int)
        if num_me > 0 or num_no > 0:
            idx_toclassifyhard = []
            FetchIndex1 = []
            numf1 = 0
            if num_re > 0 or num_me > 0:
                
                idx_toclassify = np.where(ClusterIndice<num_re)[0]
                sample_toclassify = sample[idx_toclassify, :]
                cidx_toclassify = ClusterIndice[idx_toclassify]
                classify_clusters = np.concatenate([re_cluster, merged_cluster])
                subcluster_representatives,subcluster_labelrep = Extract_Labels(subcluster_info,dim)
                numf1 = round(num_S * len(idx_toclassify)/len(ClusterIndice))
                if len(idx_toclassify) > 0:
                    idx_toclassifyhard = leastconfidence(numf1,subcluster_representatives,subcluster_labelrep, sample, idx_toclassify)
                    FetchIndex1 = idx_toclassifyhard
                idx_remain = np.where(ClusterIndice >= num_re)[0]
            else:
                idx_toclassify = []
                sample_toclassify = []
                cidx_toclassify = []
                label_toclassify = []
                idx_remain = np.arange(0, np.shape(sample)[0])

            query_clusters = P[num_re:, :]
            num_f2 = num_S - numf1
            
            sample_remain = sample[idx_remain, :]
            FetchIndex2, UnfetchedIndex = active_labelquery(num_f2, query_clusters, sample_remain,
                                                           temp_labeler[idx_remain], ClusterIndice[idx_remain], num_re)
            FetchIndex = np.concatenate([FetchIndex1,idx_remain[FetchIndex2]])
            FetchIndex =  FetchIndex.astype(int)

            sample_Fetch = sample[FetchIndex][:]
            label_Fetch = temp_labeler[FetchIndex]

            queryhist.append(len(label_Fetch))
            sample_Unfetched = sample[idx_remain[UnfetchedIndex]][:]
            # Ground Truth labels for unfetched samples
            label_Unfetched = temp_labeler[idx_remain[UnfetchedIndex]]

            unclassified_idx = np.append(idx_toclassify,idx_remain[UnfetchedIndex])
            unclassified_idx = unclassified_idx.astype(int)
            unclassified_sample = sample[unclassified_idx][:]
            unclassified_label = temp_labeler[unclassified_idx]

            if len(subcluster_info) > 0:
                combined_reps = np.concatenate([subcluster_representatives,sample_Fetch],axis=0)
                combined_repls = np.append(subcluster_labelrep,label_Fetch)
            else:
                combined_reps = sample_Fetch
                combined_repls = label_Fetch
            

            propagated_labels = Label_Propagation(combined_reps, combined_repls, unclassified_sample)
            obtained_label[FetchIndex] = label_Fetch
            obtained_label[unclassified_idx] = propagated_labels
            
        else:
            subcluster_representatives, subcluster_labelrep = Extract_Labels(subcluster_info, dim)
            numf1 = num_S
            idx_toclassify = np.arange(0,np.shape(sample)[0])
            idx_toclassifyhard = leastconfidence(numf1,subcluster_representatives,subcluster_labelrep, sample, idx_toclassify)
            unclassified_idx = idx_toclassify
            unclassified_sample = sample[unclassified_idx][:]
            unclassified_label = temp_labeler[unclassified_idx]

            combined_reps = subcluster_representatives
            combined_repls = subcluster_labelrep

            propagated_labels = Label_Propagation(combined_reps, combined_repls, unclassified_sample)
            obtained_label[unclassified_idx] = propagated_labels
            obtained_label[idx_toclassifyhard] = temp_labeler[idx_toclassifyhard]


        temp_labelinfo = {}

        if t == 0:
            for cluster_idx in range(np.shape(P)[0]):
                sample_ind = np.where(ClusterIndice == cluster_idx)[0]
                pre_label = obtained_label[sample_ind]
                tempcluster_labels = np.unique(obtained_label[sample_ind])

                if len(tempcluster_labels) > 1:
                    inner_info = {}
                    for sublabel in tempcluster_labels:
                        sub_info = np.empty((1, dim + 2))
                        subcluster_ind = sample_ind[np.where(pre_label == sublabel)[0]]
                        subcluster = sample[subcluster_ind, :]
                        sub_meanfit = Fitness_Cal(subcluster, subcluster, stdData, gamma)
                        sub_mean = subcluster[np.argmax(sub_meanfit),:]
                        # sub_mean = np.mean(subcluster, axis=0)
                        sub_std = np.sum(np.square(np.std(subcluster, axis=0))) ** 0.5
                        sub_num = len(subcluster_ind)
                        sub_info[0][:dim] = sub_mean
                        sub_info[0][dim] = sub_std
                        sub_info[0][dim + 1] = sub_num
                        
                        if sub_num > 10:
                            inner_info.update({str(sublabel): sub_info})
                else:
                    sub_info = np.empty((1, dim + 2))
                    subcluster = sample[sample_ind, :]
                    sub_meanfit = Fitness_Cal(subcluster, subcluster, stdData, gamma)

                    sub_mean = subcluster[np.argmax(sub_meanfit),:]

                    sub_std = np.sum(np.square(np.std(sample[sample_ind][:], axis=0))) ** 0.5
                    sub_num = len(sample_ind)
                    sub_info[0][:dim] = sub_mean
                    sub_info[0][dim] = sub_std
                    sub_info[0][dim + 1] = sub_num
                    inner_info = {str(tempcluster_labels[0]): sub_info}

                subcluster_info.update({str(cluster_idx): inner_info})
        else:
            num_re = np.shape(re_cluster)[0]
            num_me = np.shape(merged_cluster)[0]
            num_no = np.shape(novel_cluster)[0]

            if num_no == 0:
                subcluster_info = UpdateExistModel(ClusterIndice, P, P_Summary, sample, obtained_label, subcluster_info,
                                                   num_no,stdData,gamma)
            else:
                subcluster_info = UpdateExistModel(ClusterIndice, P, P_Summary, sample, obtained_label, subcluster_info,
                                                   num_no,stdData,gamma)
                subcluster_info = CreateNewModel(ClusterIndice, P, P_Summary, sample, obtained_label, subcluster_info,
                                                 num_no,stdData,gamma)
        P_Summary = ClusterSummary(P,PF,P_Summary,sample,TC,ClusterIndice)
        acc_predlabel = np.concatenate([acc_predlabel, obtained_label])

        F1Hist.append(f1_score(AccLabel, acc_predlabel, average='macro'))
        BAcc1_Hist.append(balanced_accuracy_score(AccLabel, acc_predlabel))
        PreStd, PFS = StoreInf(PF, PFS, PreStd, stdData)

        merged_cluster = []
        merged_fit = []
        novel_cluster = []
        novel_fit = []
        re_cluster = []
        re_fit = []
        re_T = []
    # Print out results here
    print("The mean of balanced accuracy: ", np.mean(BAcc1_Hist))
    print("The mean of F1-macro score: ", np.mean(F1Hist))
    print("Label ratio: " +str(label_ratio)+"%")

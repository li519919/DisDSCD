import os
import torch
cpu_num = 4
os.environ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)
#from fast_pytorch_kmeans import KMeans
from sklearn.cluster import KMeans, SpectralClustering
from utils import get_network
import umap
import umap.plot
from sklearn.preprocessing import StandardScaler
from uni import get_encoder
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances 
import torch.nn.functional as F
from sklearn.base import BaseEstimator
from sklearn.mixture import GaussianMixture
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering
from sklearn.cluster import OPTICS
from sklearn.cluster import Birch,DBSCAN
import random



def queryby_umap_uni(images, num_classes, args):
    images = images.to(args.device)
    pre_model = get_network(args.embedding, 3, num_classes, (args.res, args.res), width=args.width,
                            depth=args.depth, args=args).to(args.device)
    model_path = ''
    if args.uni == 1:
            
            
            batch_size = 64  
            embeddings = []
            
            
            if not hasattr(args, 'uni_model') or args.uni_model is None:
                model, _ = get_encoder(enc_name='uni', device=args.device)
                args.uni_model = model  
            else:
                model = args.uni_model
            
            model = model.to(args.device).eval()
            

            for i in range(0, len(images), batch_size):
                torch.cuda.empty_cache()
                
                batch = images[i:i+batch_size].to(args.device)
                
                with torch.inference_mode(), torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    batch_emb = model(batch)
                    embeddings.append(batch_emb.cpu())  
                    del batch_emb  
                
                
                del batch
                torch.cuda.empty_cache()
            
            
            embeddings = torch.cat(embeddings)
            
    else:
         
        pre_model = get_network(args.embedding, 3, num_classes, (args.res, args.res), 
                                 width=args.width, depth=args.depth, args=args).to(args.device)
        model_path = ''
        if args.depth == 5:
            model_path = f'model/{args.embedding}{args.res}depth5.pth'
            pre_model.load_state_dict(torch.load(model_path, map_location="cpu"))
            embeddings = get_embeddings(images, pre_model, args.distributed)
        elif args.depth == 6:
            model_path = f'model/{args.embedding}{args.res}depth6.pth'
            pre_model.load_state_dict(torch.load(model_path, map_location="cpu"))
            embeddings = get_embeddings(images, pre_model, args.distributed)

    
   
    scaler = StandardScaler()
    embeddings = scaler.fit_transform(embeddings.cpu().numpy())
    
    
    mapper = umap.UMAP(n_neighbors=10, metric='euclidean',min_dist=0.05, n_components=args.reduced_dim).fit(embeddings)
    reduced_embedding = mapper.transform(embeddings)
    reduced_embedding = torch.from_numpy(reduced_embedding)
    
   
    kmeans = KMeans(init='k-means++',n_clusters=args.num_cluster,random_state=0,n_init=20)
    labels = kmeans.fit_predict(reduced_embedding)
    centers = torch.from_numpy(kmeans.cluster_centers_).to(torch.float32)
    
    
    dist_matrix = cosine_distance(centers, reduced_embedding)  # [num_clusters, num_samples]
    
    
    selected_indices = []
    
   
    labels_tensor = torch.tensor(labels)
    cluster_sizes = []
    for i in range(args.num_cluster):
        cluster_indices = (labels_tensor == i).nonzero(as_tuple=True)[0]
        cluster_sizes.append(len(cluster_indices))
    
   
    for i in range(args.num_cluster):
       
        cluster_indices = (labels_tensor == i).nonzero(as_tuple=True)[0]
        
        
        if len(cluster_indices) == 0:
            continue
            
        
        if len(cluster_indices) < args.subsample:
            
            global_dists = dist_matrix[i]  # [num_samples]
            
            
            _, min_indices = torch.topk(global_dists, k=args.subsample, largest=False)
            
           
            global_indices = min_indices.tolist()
            
            selected_indices.extend(global_indices)
            
            
        else:
            
            cluster_dists = dist_matrix[i, cluster_indices]
            
            
            num_close = int(round(args.subsample * 0.8))
            num_close = max(1, num_close)  
            
            num_far = args.subsample - num_close
            
            
            _, close_indices = torch.topk(cluster_dists, k=num_close, largest=False)
            
            
            _, far_indices = torch.topk(cluster_dists, k=num_far, largest=True)
            
            
            all_indices = torch.cat([close_indices, far_indices])
            
            
            global_indices = cluster_indices[all_indices].tolist()
            
            selected_indices.extend(global_indices)
            
           
    
    
    q_idxs = torch.tensor(selected_indices)
    
    
    small_clusters = sum(1 for size in cluster_sizes if size < args.subsample and size > 0)
    large_clusters = sum(1 for size in cluster_sizes if size >= args.subsample)
    
    
    
    return q_idxs




def euclidean_dist(x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(x, y.t(), beta=1, alpha=-2)
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist

###
def cosine_distance(centers, embeddings):
    
    sim_matrix = centers @ embeddings.t()  
    
   
    dist_matrix = 1 - sim_matrix
    
   
    dist_matrix = dist_matrix.clamp(min=0)
    
    return dist_matrix



def cosine_distance(centers, embeddings):
    centers_norm = F.normalize(centers, p=2, dim=1)
    embeddings_norm = F.normalize(embeddings, p=2, dim=1)
    dist_matrix = 1 - torch.mm(centers_norm, embeddings_norm.T)
    return dist_matrix




def querybykmeans(images, num_classes, args):
    images = images.to(args.device)
    pre_model = get_network(args.embedding, 3, num_classes, (args.res, args.res), width=args.width,
                            depth=args.depth, args=args).to(args.device)
    model_path = ''
    if args.embedding == "ConvNet":
        model_path = f'model/{args.embedding}{args.res}depth{args.depth}.pth'
    elif args.embedding == "ResNet18":
        model_path = f'model/{args.embedding}{args.res}.pth'

    pre_model.load_state_dict(torch.load(model_path, map_location="cpu"))
    embeddings = get_embeddings(images, pre_model, args.distributed)
    
    kmeans = KMeans(init='k-means++',n_clusters=args.num_cluster,random_state=0,n_init=20)
    #kmeans = KMeans(n_clusters=args.num_cluster, mode='cosine')
    labels = kmeans.fit_predict(embeddings)
    #centers = kmeans.centroids
    centers =  torch.from_numpy(kmeans.cluster_centers_).to(torch.float32) 

    dist_matrix = euclidean_dist(centers, embeddings)
    ###
    #dist_matrix = cosine_distance(centers, embeddings)
    min_indices = torch.topk(dist_matrix, k=args.subsample, largest=False, dim=1)[1]

    q_idxs = min_indices.view(-1)
    return q_idxs



def querybyimageumap(images, args):
    images = images.to("cpu")
    #kmeans = KMeans(n_clusters=args.num_cluster, mode='euclidean')
    kmeans = KMeans(init='k-means++',n_clusters=args.num_cluster,random_state=0,n_init=20)
    images = images.view(images.size(0), -1)
    mapper = umap.UMAP(n_neighbors=10, min_dist=0.05, n_components=args.reduced_dim).fit(images)
    ###
    
    reduced_embedding = mapper.transform(images)
    reduced_embedding = torch.from_numpy(reduced_embedding)
    labels = kmeans.fit_predict(reduced_embedding)
    #centers = kmeans.centroids
    centers =  torch.from_numpy(kmeans.cluster_centers_).to(torch.float32) 

    dist_matrix = euclidean_dist(centers, reduced_embedding)
    min_indices = torch.topk(dist_matrix, k=args.subsample, largest=False, dim=1)[1]

    q_idxs = min_indices.view(-1)
    return q_idxs


def get_embeddings(images, model, distributed):
    if distributed:
        embed = model.module.embed
    else:
        embed = model.embed
    features = []
    num_img = images.size(0)
    batch_size = 64
    with torch.no_grad():
        for i in range(0, num_img, batch_size):
            subimgs = images[i:i+batch_size]
            subfeatures = embed(subimgs).detach()
            features.append(subfeatures)

    features = torch.cat(features, dim=0).to("cpu")
    return features


def get_probabilities(images, model):
    out = model(images)
    return out


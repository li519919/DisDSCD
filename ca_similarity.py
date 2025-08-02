import os
import numpy as np
from tqdm import tqdm
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import concurrent.futures

class FeatureMatcher:
    def __init__(self, batch_size=1000):
        
        self.batch_size = batch_size
        
    def load_features(self, folder_path):
        
        feature_dict = {}
        
        
        
        for root, _, files in tqdm(os.walk(folder_path), desc=""):
            for file in files:
                if file.endswith('.npy'):
                    file_path = os.path.join(root, file)
                    try:
                        
                        feature = np.load(file_path)
                        
                       
                        if feature.shape == (1024,):
                            feature = feature.reshape(1, 1024)
                        elif feature.shape == (1, 1024):
                            pass  
                        else:
                            print(f"again")
                            continue
                            
                        feature_dict[file_path] = feature
                    except Exception as e:
                        print(f"again")
        
        
        return feature_dict
    
    def calculate_all_similarities(self, features_A, features_B):
        
        a_paths = list(features_A.keys())
        a_features = np.vstack([features_A[path] for path in a_paths])
        
        
        b_paths = list(features_B.keys())
        b_features = np.vstack([features_B[path] for path in b_paths])
        
        
        
        results = []
        
        
        total_batches = (len(a_features) + self.batch_size - 1) // self.batch_size
        
        for batch_idx in tqdm(range(0, len(a_features), self.batch_size), desc=""):
            batch_end = min(batch_idx + self.batch_size, len(a_features))
            
            
            a_batch = a_features[batch_idx:batch_end]
            batch_paths = a_paths[batch_idx:batch_end]
            
            
            similarity_matrix = cosine_similarity(a_batch, b_features)
            
            
            for i in range(len(a_batch)):
                
                scores = similarity_matrix[i]
                
                
                best_idx = np.argmax(scores)
                best_score = scores[best_idx]
                best_b_path = b_paths[best_idx]
                
                results.append((batch_paths[i], best_b_path, best_score))
        
        return results
    
    def find_best_matches(self, folder_A, folder_B, output_csv, max_workers=8):
        
        features_A = self.load_features(folder_A)
        features_B = self.load_features(folder_B)
        
        
        
       
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            
            tasks = []
            a_paths = list(features_A.keys())
            
            for i in range(0, len(a_paths), self.batch_size):
                batch_paths = a_paths[i:i + self.batch_size]
                batch_features = np.vstack([features_A[p] for p in batch_paths])
                
                
                tasks.append(executor.submit(
                    self._process_batch,
                    batch_features=batch_features,
                    batch_paths=batch_paths,
                    b_features=np.vstack(list(features_B.values())),
                    b_paths=list(features_B.keys())
                ))
            
            
            results = []
            for future in tqdm(concurrent.futures.as_completed(tasks), 
                               total=len(tasks), 
                               desc=""):
                results.extend(future.result())
        
        
        df = pd.DataFrame(results, columns=[
            'A_Feature_Path', 
            'B_Best_Match_Path', 
            'Similarity_Score'
        ])
        
        
        df['A_Folder'] = folder_A
        df['B_Folder'] = folder_B
        
        
        df = df.sort_values(by='Similarity_Score', ascending=True)
        
        
        df.to_csv(output_csv, index=False)
        
        
        return df
    
    def _process_batch(self, batch_features, batch_paths, b_features, b_paths):
        
        similarity_matrix = cosine_similarity(batch_features, b_features)
        
        
        results = []
        for i in range(len(batch_features)):
            
            scores = similarity_matrix[i]
            
           
            best_idx = np.argmax(scores)
            best_score = scores[best_idx]
            best_b_path = b_paths[best_idx]
            
            results.append((
                batch_paths[i],  
                best_b_path,     
                best_score        
            ))
        
        return results
    
    def find_best_matches_single_core(self, folder_A, folder_B, output_csv):
        
        features_A = self.load_features(folder_A)
        features_B = self.load_features(folder_B)
        
        
        b_paths = list(features_B.keys())
        b_features = np.vstack(list(features_B.values()))
        
        
        results = []
        for a_path, a_feature in tqdm(features_A.items(), total=len(features_A), 
                                      desc=""):
           
            similarities = cosine_similarity(a_feature, b_features)
            
            
            best_idx = np.argmax(similarities)
            best_score = similarities[0, best_idx]
            best_b_path = b_paths[best_idx]
            
            results.append((a_path, best_b_path, best_score))
        
        
        df = pd.DataFrame(results, columns=[
            'A_Feature_Path', 
            'B_Best_Match_Path', 
            'Similarity_Score'
        ])
        df['A_Folder'] = folder_A
        df['B_Folder'] = folder_B
        
        
        df = df.sort_values(by='Similarity_Score', ascending=True)
        
       
        df.to_csv(output_csv, index=False)
        
        
        return df


matcher = FeatureMatcher(batch_size=500)


results_df = matcher.find_best_matches(
    folder_A="",
    folder_B="",
    output_csv="",
    max_workers=8
)


print(results_df.head())
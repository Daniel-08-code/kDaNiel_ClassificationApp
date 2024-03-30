from flask import Flask, render_template, request
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from scipy.cluster import hierarchy
import numpy as np
import pandas as pd
import matplotlib
from docx import Document
import os
matplotlib.use('Agg') 
app = Flask(__name__)

save_path = os.path.join(app.root_path, 'static')
class Cluster:
    nbrCluster = 0

    def __init__(self, data):
        Cluster.nbrCluster += 1
        self.numeroCluster = Cluster.nbrCluster
        self.indicesIndividus = [Cluster.nbrCluster]  # Initialise avec un tableau contenant seulement le numéro de cluster
        self.data = data
save_path = os.path.join(app.root_path, 'static')
class Fonction:
    @staticmethod
    def distanceEuclidienne(cluster1, cluster2):
        somme = 0.0
        for i in range(len(cluster1.data)):
            somme += (cluster1.data[i] - cluster2.data[i]) ** 2
        return np.sqrt(somme)
    
    @staticmethod
    def kmeans_algorithm(data, num_clusters, max_iterations=300):
        # Step 1: Initialize centroids randomly
        centroids = data[np.random.choice(data.shape[0], num_clusters, replace=False)]

        for _ in range(max_iterations):
            # Step 2: Assign each data point to the nearest centroid
            distances = np.sqrt(((data - centroids[:, np.newaxis])**2).sum(axis=2))
            labels = np.argmin(distances, axis=0)

            # Step 3: Update centroids based on the mean of the points assigned to each cluster
            new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(num_clusters)])

            # Step 4: Check for convergence
            if np.all(centroids == new_centroids):
                break

            centroids = new_centroids

        return labels

    @staticmethod
    def hierarchy(clusters):
        doc = Document('static/rapport.docx')
        doc.add_page_break()
        doc.add_paragraph("\n\n\n")
        hierarchy = []
        indices = []
        iteration = 0
        while len(clusters)-2*iteration > 1:
            distanceMin = float('inf')
            indice1, indice2 = None, None
            for i in range(len(clusters)):
                if i not in indices:
                    for j in range(i + 1, len(clusters)):
                        if j not in indices:
                            distance = Fonction.distanceEuclidienne(clusters[i], clusters[j])
                            if distance < distanceMin:
                                distanceMin = distance
                                indice1, indice2 = i, j
            
            indices.append(indice1)
            indices.append(indice2)
            iteration +=1
            new_data = [(clusters[indice1].data[i] + clusters[indice2].data[i]) / 2 for i in range(len(clusters[indice1].data))]
            newCluster = Cluster(new_data)  # Crée un nouveau cluster avec les données du barycentre des deux clusters
            newCluster.indicesIndividus = clusters[indice1].indicesIndividus + clusters[indice2].indicesIndividus
            clusters.append(newCluster)
            doc.add_paragraph(f"{iteration}-- Liaison du Cluster {clusters[indice1].numeroCluster-1} et du Cluster {clusters[indice2].numeroCluster-1} pour former le cluster {newCluster.numeroCluster-1} regroupant ainsi {len(newCluster.indicesIndividus)} elements.\n")
            

    # Enregistrer le document Word dans le dossier static
            doc.save(os.path.join(save_path, 'telechargeable.docx'))
            hierarchy.append([indice1, indice2, distanceMin, len(newCluster.indicesIndividus)])
            
        return hierarchy
    
    
@app.route('/', methods=['GET', 'POST'])
def upload_and_display():
    if request.method == 'POST':
        method = request.form['method']
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            dataframe = pd.read_excel(uploaded_file, engine='openpyxl')
            columns = dataframe.columns.tolist()
            rows = dataframe.values.tolist()

            # Utilisation des données du dataframe pour le clustering hiérarchique
            data = dataframe.to_numpy()
            
            if method == 'CAH':
                Cluster.nbrCluster = 0
                clusters = [Cluster(data[i]) for i in range(len(data))]
                
        
                hierarchy_result = Fonction.hierarchy(clusters)
                hierarchy_result = np.array(hierarchy_result)
                
                #On peut verifier le resultat avec ce package
                #Z = hierarchy.linkage(data, method='centroid') 
                #print(hierarchy_result)
            
                # Création du dendrogramme
                plt.figure(figsize=(8, 6))
                hierarchy.dendrogram(hierarchy_result)
                plt.title('Dendrogramme')
                plt.xlabel('Indices des échantillons')
                plt.ylabel('Distance euclidienne')

                # Sauvegarde de la figure dans un tampon mémoire
                buffer = BytesIO()
                plt.savefig(buffer, format='png')
                buffer.seek(0)

                # Conversion de la figure en format base64
                image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

                # Fermeture de la figure pour libérer la mémoire
                plt.close()
                
            elif method == 'K-Means':
                num_clusters = 4  # Define the number of clusters, you can change this according to your requirement
                labels = Fonction.kmeans_algorithm(data, num_clusters)

                # Plotting the clusters (you can customize this part according to your visualization needs)
                plt.figure(figsize=(8, 6))
                plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')
                plt.title('K-Means Clustering')
                plt.xlabel('Feature 1')
                plt.ylabel('Feature 2')

                # Saving the plot as a base64 encoded image
                buffer = BytesIO()
                plt.savefig(buffer, format='png')
                buffer.seek(0)
                image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                plt.close()

            return render_template('index.html', columns=columns, rows=rows, image=image_base64)

        else:
            return render_template('index.html')
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')



class KMeansCluster:
    nbr_cluster = 0

    def __init__(self, data):
        KMeansCluster.nbr_cluster += 1
        self.cluster_number = KMeansCluster.nbr_cluster
        self.data = data
        
        

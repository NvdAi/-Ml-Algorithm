import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

class K_M :
	def __init__(self):
		self.all_labels = []
		self.all_centers = []
	
	def center_selection(self,X,labels,K):
		new_centers = []
		for i in range(K):
			indexs = np.where(labels == i)[0]
			cluster = X[indexs]
			center = np.mean(cluster,axis=0)
			new_centers.append(center)
		new_centers = np.array(new_centers)
		return new_centers

	def distande_calculatore(self,X,labels,K,centers):
		for i,sample in enumerate(X):
			dist_POC = []                      # list of disatances single sample of all centers
			for center in centers:
				dist = np.linalg.norm(sample - center)
				dist_POC.append(dist)
			arg = dist_POC.index(min(dist_POC))
			labels[i] = arg
		return labels

	def generate_video(self):
		image_folder = '../data/figures/' 
		outpot_path = '../data/output'
		os.makedirs(outpot_path, exist_ok=True)
		video_name = os.path.join(outpot_path, "my_model_output.avi")
		images = [img for img in os.listdir(image_folder)
				if img.endswith(".jpg") or
					img.endswith(".jpeg") or
					img.endswith("png")]
		images = sorted(images,key=lambda fname: int(fname.split('.')[0]))
		frame = cv2.imread(os.path.join(image_folder, images[0]))
		height, width, layers = frame.shape  
		video = cv2.VideoWriter(video_name, 0, 1, (width, height)) 
		for image in images: 
			video.write(cv2.imread(os.path.join(image_folder, image))) 
		cv2.destroyAllWindows() 
		video.release() 
		return video_name 

	def fit(self,X,labels,K):
		c = np.random.randint(0,X.shape[0],(K))
		centers = X[c]
		save_frames_path = "../data/figures"
		condition = 1
		iteration = 1
		while condition == 1:
			labels = self.distande_calculatore(X,labels,K,centers)
			self.all_labels.append(labels)
			new_centers = self.center_selection(X,labels,K)
			self.all_centers.append(new_centers)
			if  np.array_equal(centers, new_centers):
				condition = 0
			else:
				centers = new_centers
			
			plt.scatter(X[:,0],X[:,1],c=labels)
			plt.scatter(centers[:,0],centers[:,1],c="r",s=200,marker="X")
			plt.xlabel("X",fontsize=15)
			plt.ylabel("Y",fontsize=15)
			plt.title("K-MEANS\nRed X-Point Are Centers")
			os.makedirs(save_frames_path, exist_ok=True)
			plt.savefig(save_frames_path + "/" + str(iteration) +  ".png")
			plt.close()
			# print("sived figure :",iteration)
			iteration+=1				

		print("clusters are fixed")
		video_name = self.generate_video()
		for f in os.listdir(save_frames_path):
			os.remove(os.path.join(save_frames_path, f))
		return labels,centers,video_name



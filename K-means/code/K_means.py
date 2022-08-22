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
		outpot_path = '../data/outpot'
		os.makedirs(outpot_path, exist_ok=True)
		video_name = os.path.join(outpot_path, " my_model_output.avi")
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

	def fit(self,X,labels,K):
		c = np.random.randint(0,X.shape[0],(K))
		centers = X[c]
		condition = 1
		iteration = 1
		# elevated_angle = 0

		horizontal_angle = 0
		while condition == 1:
			labels = self.distande_calculatore(X,labels,K,centers)
			self.all_labels.append(labels)
			new_centers = self.center_selection(X,labels,K)
			self.all_centers.append(new_centers)
			if  np.array_equal(centers, new_centers):
				condition = 0
			else:
				centers = new_centers
			
			if X.shape[1]==3:
					
				fig = plt.figure(figsize=(10, 10))
				ax = fig.add_subplot(projection='3d')
				ax.scatter3D(X[:,0],X[:,1],X[:,2],c=labels)
				ax.scatter3D(centers[:,0],centers[:,1],centers[:,2],c="r",s=200,marker="X")
				ax.set_xlabel('X',fontsize=20,labelpad=12)
				ax.set_ylabel('Y',fontsize=20,labelpad=12)
				ax.set_zlabel('Z',fontsize=20, labelpad=12)
				ax.view_init(0, 45)

				save_frames_path = "../data/figures"
				os.makedirs(save_frames_path, exist_ok=True)
				ax.figure.savefig(save_frames_path + "/" + str(iteration) +  ".png")
				plt.close('all')
				print("sived figure :",iteration)
				# elevated_angle += 20
				horizontal_angle += 10
				iteration+=1
			else :
				plt.scatter(X[:,0],X[:,1],c=labels)
				plt.scatter(centers[:,0],centers[:,1],c="r",s=200,marker="X")
				plt.xlabel('X',fontsize=20,labelpad=12)
				plt.ylabel('Y',fontsize=20,labelpad=12)

				save_frames_path = "../data/figures"
				os.makedirs(save_frames_path, exist_ok=True)
				plt.savefig(save_frames_path + "/" + str(iteration) +  ".png")
				plt.close('all')
				print("sived figure :",iteration)
				iteration+=1				

		print("clusters are fixed")
		self.generate_video()
		return labels,centers



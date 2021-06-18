import tkinter as tk
import imagej
import pandas as pd
import subprocess
import time
import queue
import threading
import sys
import platform
import glob
import select
import stat
import os
import math
import cv2
import numpy as np
from numpy.linalg import eig, inv, svd
from PIL import ImageTk, Image
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
from sklearn.cluster import KMeans
from scipy.interpolate import griddata
import scipy.signal as signal
from scipy.optimize import curve_fit


import constants
from user_interface import UserInterface


#Improvements - 
#Allowing scaling/calibration not hard coded
#Ker to work with other ring sizes
#Fix scroll
#Fix clustering
#New magnification process 


class ThreadedClient:
   """
   Launch the main part of the GUI and the worker thread. periodicCall and
   endApplication could reside in the GUI part, but putting them here
   means that you have all the thread controls in a single place.
   """

   #Initialises imageJ and creates threads
   def __init__(self, master):
      """
      Start the GUI and the asynchronous threads. We are in the main
      (original) thread of the application, which will later be used by
      the GUI as well. 
      """
      self.master = master
      self.ij =imagej.init()
      self.image_analysis_active = False
      self.image_analysis_complete = False
      self.video_to_frames_active=False
      self.results_analysis_active = False   
      self.imageNumber=0
      self.fps=constants.FRAME_RATE
      # Create the queue
      self.out_queue = queue.Queue()
      self.in_queue = queue.Queue()
      # Set up the GUI part
      self.gui = UserInterface(self, master, self.out_queue, self.in_queue)
      # Set up the threads
      self.running = 1
      # Define the worker threads
      self.threads = {}
      # This worker thread handles in/out queues to GUI
      self.threads['thread1'] = threading.Thread(target=self.gui_comms_thread)
      # This worker thread handles image_analysis of the hardware
      self.threads['thread2'] = threading.Thread(target=self.image_analysis_thread)

      self.threads['thread3'] = threading.Thread(target=self.video_to_frames_thread)

      self.threads['thread4'] = threading.Thread(target=self.results_analysis_thread)
          
      # Start the worker threads
      for thread in self.threads:
         self.threads[thread].daemon = True
         self.threads[thread].start()

   #Processes incoming messages from GUI thread
   def gui_comms_thread(self):
      while self.running:
         self.sleep()
         # Read message from worker threads to GUI
         self.gui.process_incoming()
         # Read messages from GUI to worker threads
         self.process_incoming()

   #Thread for image analysis. Uses imageJ.
   def image_analysis_thread(self):
      while self.running:
         self.sleep()
         if self.image_analysis_active == True and (self.analysis_type == "PR" or self.analysis_type == "PIPR"):

            self.fileList= [f for f in glob.glob(self.input_folder+ "**/*"+self.fileType, recursive=True)]

            #Analyse defined number of images for PR or PIPR
            for i in range (self.imageNumber-15,self.imageNumber+self.total_number_images-15):
               #File nam has entire directory path. Split on /, \ and . to get the number
               fileName=self.fileList[i]
               fullFileNum = fileName[len(fileName)-8]+fileName[len(fileName)-7]+fileName[len(fileName)-6]+fileName[len(fileName)-5]
               inputName=fullFileNum+self.fileType
               inputPath = self.output_directory+"/Enhanced Contrast/"+inputName
               self.enhanceContrast(fileName, inputPath)
               print(inputPath)
               outputName=fullFileNum+"_binary"+self.fileType
               outputPath=self.output_directory+"/Binaries/"+outputName
               print(outputPath)

               #Open, duplicate, threshold and save with imageJ
               macro_measure_image = """filePathInput='"""+inputPath+"""';
                            filePathOutput='"""+outputPath+"""';
                            fileNum=''+"""+fullFileNum+""";
                            open(filePathInput);
                            run("8-bit");
                            run("Enhance Contrast", "saturated=10 equalize");
                            setAutoThreshold("Default dark");
                            //run("Threshold...");
                            setThreshold(43, 255);
                            //setThreshold(43, 255);
                            run("Convert to Mask");
                            saveAs("png", filePathOutput);
                            run("Duplicate...", "title="+fileNum);
                            run("Set Measurements...", "area mean min fit display redirect=None decimal=3");
                            run("Analyze Particles...", "size=10000-Infinity circularity=0.20-Infinity exclude");"""
               self.ij.py.run_macro(macro_measure_image)
               prog = 100*(i-(self.imageNumber-15))/self.total_number_images
               self.out_queue.put("progress:" + str(prog))
            
            #Save measurements from imageJ
            outputPath=self.output_directory+"/All_Results_Raw.csv"
            macro_finish = """saveAs("Results",'"""+outputPath+"');"
            self.ij.py.run_macro(macro_finish)

            
            self.image_analysis_active = False
            self.image_analysis_complete = True
            self.results_analysis_active=True
         
         if self.image_analysis_active == True and self.analysis_type == "CT":
                   
            #Find coordinates of keratometer dots with imageJ
            macro_measure_image = """filePathInput='"""+self.cropOutputPath+"""';
            open(filePathInput);
            run("8-bit");
            //run("Watershed");
            run("Enhance Contrast", "saturated=10");
            setAutoThreshold("Default dark");
            //run("Threshold...");
            setAutoThreshold("Default dark");
            //setThreshold(129, 255);
            setOption("BlackBackground", true);
            run("Convert to Mask");
            run("Watershed");
            run("Duplicate...", "title=Ker");
            run("Set Measurements...", "area centroid fit display redirect=None decimal=3");
            run("Analyze Particles...", "size=4-150 circularity=0.5-1 exclude clear");"""
            self.ij.py.run_macro(macro_measure_image)
            
            #Save imageJ results
            self.kerIJResultsPath = self.output_directory+"/Ker_Results.csv"
            macro_finish = """saveAs("Results",'"""+self.kerIJResultsPath+"');"
            self.ij.py.run_macro(macro_finish)
            
            self.image_analysis_active = False
            self.image_analysis_complete = True
            self.results_analysis_active=True

            
   #Thread for the analysis of PR or PIPR data saved by the Image Analysis Thread
   def results_analysis_thread(self):
      while self.running:
         self.sleep()
         if self.results_analysis_active ==True and (self.analysis_type == "PR" or self.analysis_type == "PIPR"):
            #Read data from image analysis thread            
            data = pd.read_csv(self.output_directory+"/All_Results_Raw.csv")
            
            #Remove duplicates in labels caused by the detection of two objects (Labels refer to the image number)
            data.drop_duplicates(subset ="Label",keep = False, inplace = True)
            successfulImages=np.array(data['Label'])
            
            
            #Add rows for missing labels
            newRows=0
            for i in range(0,201):
               if (i-newRows)<len(successfulImages):
                  if successfulImages[(i-newRows)] != (i+(self.imageNumber-14)):                  
                     newData = {'Label':(i+(self.imageNumber-14)), 'Area':np.nan, 'Mean':np.nan, 'Min':np.nan, 'Max':np.nan,'Major':np.nan,'Minor':np.nan,'Angle':np.nan}
                     data=data.append(newData, ignore_index=True)
                     newRows+=1
               else:
                  print(i+(self.imageNumber-14))
                  newData = {'Label':(i+(self.imageNumber-14)), 'Area':np.nan, 'Mean':np.nan, 'Min':np.nan, 'Max':np.nan,'Major':np.nan,'Minor':np.nan,'Angle':np.nan}
                  data=data.append(newData, ignore_index=True)
                  newRows+=1

            #Reorder data as missing labels were appended to the end of the dataset
            data = data.sort_values('Label')
            data.to_csv(self.output_directory+"/All_Results_Modified.csv",index=False)
            #Read to reindex
            data=pd.read_csv(self.output_directory+"/All_Results_Modified.csv")

            #Calculate pupil diameter ratio and imaging angle
            #Fill data 2 points either side of NaN to reomve missing data
            #Calculate time from labels. There is a 1 6Hz frame rate
            majorAxis=data['Major'].astype(float)/constants.pupil_scale
            minorAxis=data['Minor'].astype(float)/constants.pupil_scale
            labels=data['Label'].astype(int)
            axisRatio=minorAxis.divide(majorAxis)
            imagingAngle= pd.Series(axisRatio).apply(lambda x: (math.sqrt((constants.quad_b-math.sqrt(constants.quad_b**2-4*constants.quad_a*(1-x)))/(2*constants.quad_a))))
            majorAxisFilled = pd.Series(majorAxis).fillna(limit=2, method='ffill')
            time = pd.Series(labels).apply(lambda x: (x-(self.imageNumber-15))/self.fps)
            
            #Filter based on derivative of imaging angle
            for i in range(4,197):  #+/-3
               #Find 9 point rolling average ignoring NaN
               angles = (imagingAngle[i-4],imagingAngle[i-3],imagingAngle[i-2],imagingAngle[i-1],imagingAngle[i],imagingAngle[i+1],imagingAngle[i+2],imagingAngle[i+3],imagingAngle[i+4])
               rollingAverage = np.nanmean(angles)

               #Remove rapidly changing variables
               if abs(imagingAngle[i]-rollingAverage)>2:
                  print("Remove:"+str(i))
                  majorAxisFilled[i]=np.nan
                  imagingAngle[i]=np.nan
               
            #Correct for corneal power if data is provided
            if self.CACdepth != -1 and self.cornealPower != -1:
               magnification=1/((1-self.CACdepth/1000*self.cornealPower/1.3375))
               corrected_diameter=pd.Series(majorAxisFilled).apply(lambda x: x*(1-self.CACdepth/1000*self.cornealPower/1.3375))

            else:
               corrected_diameter=majorAxisFilled

            #Find baseline, caluclate percentage area and set starting images to 100%
            i=0
            nanCount=0
            corrected_diameter_start=0
            while i<10:
               if pd.isna(corrected_diameter[i+nanCount]) or corrected_diameter[i+nanCount] is None:
                  nanCount+=1
               else:
                  corrected_diameter_start+=corrected_diameter[i+nanCount]
                  i+=1
            corrected_diameter_start=corrected_diameter_start/10
            
            for j in range (0,(10+nanCount)):
               corrected_diameter[j]=corrected_diameter_start
            
            #Calculate area and percentage are using corrected diameter            
            area=pd.Series(corrected_diameter).apply(lambda x:((x/2)**2*math.pi))
            percentageArea=area/(math.pi*(corrected_diameter_start/2)**2)     

            #Remove percentage area less than 40% data points as noise
            for i in range (0,data.shape[0]):
               try:
                  if percentageArea[i]<0.4:
                     corrected_diameter[i]=np.nan
                     percentageArea[i]=np.nan
                     area[i]=np.nan
               except:
                  pass
            
            #Interpolate data to enable easy derivative calculation
            interpolated=pd.Series(percentageArea).interpolate()

            #Filter with derivative
            for i in range (1,data.shape[0]):
               try:
                  if abs(interpolated[i]-interpolated[i-1])>0.05:
                     corrected_diameter[i]=np.nan
                     percentageArea[i]=np.nan
                     for x in range (0,20):
                        try:
                           if abs(interpolated[i+x+1]-interpolated[i-1])>0.05*(x+2):
                              corrected_diameter[i+x]=np.nan
                              percentageArea[i+x]=np.nan
                        except:
                           pass
               except:
                  pass

            #Interpolate to create smooth graph
            percentageArea=pd.Series(percentageArea).interpolate()
            percentageArea=percentageArea*100
            
            for i in range (0,10):
               percentageArea[i]=100

            #Moving Average Filter
            for i in range (2, data.shape[0]-2):
               percentageArea[i]=(percentageArea[i-2]+percentageArea[i-1]+percentageArea[i]+percentageArea[i+1]+percentageArea[i+2])/5
            percentageArea=pd.Series(percentageArea).interpolate()

            #Plot
            data['Time']=time
            data['Percentage Area']=percentageArea
            data['Corrected Area mm^2']=area
            data['Axis Ratio']=axisRatio
            data['Imaging Angle']=imagingAngle
            data.drop(['Mean', 'Min', 'Max'], axis = 1, inplace = True)
            data.to_csv(self.output_directory+"/All_Results_Modified.csv",index=False)

            
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
            ax1.plot(time, area)
            ax1.set_title("Pupil Response")
            ax1.set_xlabel("Time")
            ax1.set_ylabel("Area (mm^2)")
            ax2.plot(time, percentageArea)
            ax2.set_title("Pupil Response Baseline Adjusted")
            ax2.set_xlabel("Time")
            ax2.set_ylabel("% Area")
            
            try:
               #Find deriivative
               dcorrected_diameter=[0]*percentageArea.shape[0]
               for i in range (0, percentageArea.shape[0]-5):
                  dcorrected_diameter[i] = (percentageArea[i+5]-percentageArea[i])/5

               #Find infexlion points (1st derivative used with threshold due to potential noise)
               inflexionOne=0
               inflexionTwo=0
               i=0
               while inflexionOne==0:
                  if dcorrected_diameter[i]<-0.4:
                     inflexionOne=i
                  else:
                     i+=1
               self.xinflexionOne = time[i]

               while inflexionTwo ==0:
                  if dcorrected_diameter[i]>0.4:
                     inflexionTwo=i
                  else:
                     i+=1
               self.xinflexionTwo = time[i]

               #Segment time and diameter data based off inflexion points
               xOne= time[0:inflexionOne]
               yOne=percentageArea[0:inflexionOne]

               xTwo= time[inflexionOne:inflexionTwo]
               yTwo=percentageArea[inflexionOne:inflexionTwo]

               xThree=time[inflexionTwo:len(time)]
               yThree=percentageArea[inflexionTwo:len(time)]

               #Attempt to curve fit three segments
               poptOne, pcovOne = curve_fit(self.funcOne, xOne, yOne,maxfev=5000)
               self.aConstant=poptOne[0]

               poptTwo, pcovTwo = curve_fit(self.funcTwo, xTwo, yTwo,maxfev=5000)
               self.cConstant = (self.aConstant-(poptTwo[1]*self.xinflexionOne + poptTwo[2]*self.xinflexionOne**2 + np.exp(poptTwo[3] * self.xinflexionOne))) + poptTwo[1]*self.xinflexionTwo + poptTwo[2]*self.xinflexionTwo**2 + np.exp(poptTwo[3] * self.xinflexionTwo)

               poptThree, pcovThree = curve_fit(self.funcThree, xThree, yThree,maxfev=5000)

               yOneFitted = self.funcOne(xOne, *poptOne)
               yTwoFitted = self.funcTwo(xTwo, *poptTwo)
               yThreeFitted = self.funcThree(xThree, *poptThree)

               #Create string of fitted curve equation
               eq1 = "y = "+f'{(poptOne[0]):.1f}'+"\n"   #y=a  
               eq2 = "y = "+f'{((self.aConstant-(poptTwo[1]*self.xinflexionOne + poptTwo[2]*self.xinflexionOne**2 + np.exp(poptTwo[3] * self.xinflexionOne)))):.1f}'+"+"+f'{poptTwo[1]:.3}'+"x+"+f'{poptTwo[2]:.3}'+"x^2+e^"+f'{poptTwo[3]:.3}'+"x\n"                   #y=(self.aConstant-(d*self.xinflexionOne + e*self.xinflexionOne**2 + np.exp(f * self.xinflexionOne))) + d*x + e*x**2 + np.exp(f * x)
               eq3 = "y = "+f'{(self.cConstant-(poptThree[1]*self.xinflexionTwo + poptThree[2]*self.xinflexionTwo**2 + np.exp(poptThree[3] * self.xinflexionTwo))):.1f}'+"+"+ f'{poptThree[1]:.3}'+"x+"+ f'{poptThree[2]:.3}'+"x^2+e^"+f'{poptThree[3]:.3}'+"x\n"           #(self.cConstant-(h*self.xinflexionTwo + i*self.xinflexionTwo**2 + np.exp(j * self.xinflexionTwo))) + h*x + i*x**2 + np.exp(j * x)
               
               ax3.plot(xOne,yOneFitted)
               ax3.plot(xTwo,yTwoFitted)
               ax3.plot(xThree,yThreeFitted)
               ax3.set_title("Pupil Response Fitted")
               ax3.set_xlabel("Time")
               ax3.set_ylabel("% Area")
               printEquations=True 
            except:
               printEquations=False

            #Save graphs
            fig.savefig(self.output_directory+"/Graphs/Pupil Response Combined.png")
            fig.clf()

            #Save starting image number, average imaging angle, magnification factor, anterior chambe rdepth, corneal power and curve fit equations to text file 
            with open(self.output_directory+'/AnalysisInfo.txt', 'w') as f:
               f.write("Starting Frame: " + str(self.imageNumber))
               try:
                  imagingAngleNP=np.array(imagingAngle)
                  averageAngle=np.nanmean(imagingAngleNP)
                  ("\nAverage Imaging Angle: " + str(averageAngle))
               except:
                  pass
               if self.CACdepth == -1 or self.cornealPower == -1:
                  f.write("\nCorneal power not considered.")
               else:
                  f.write("\nMagnification "+str(magnification)+"\nAC Depth: " + str(self.CACdepth) + " mm\nCorneal Power: " + str(self.cornealPower) + " D")
               if printEquations:
                  f.write("\n\nCurve Fit:\nSection 1: "+ eq1 + "Section 2: "+ eq2 + "Section 3: "+ eq3)
               f.close()

            #Remake individual graphs
            plt.plot(time, area)
            plt.title("Pupil Response")
            plt.xlabel("Time")
            plt.ylabel("Area (mm^2)")
            plt.savefig(self.output_directory+"/Graphs/Pupil Response - Area.png")
            fig.clf()

            plt.plot(time, percentageArea)
            plt.title("Pupil Response Baseline Adjusted")
            plt.xlabel("Time")
            plt.ylabel("% Area")
            plt.savefig(self.output_directory+"/Graphs/Pupil Response - Percentage Area.png")
            fig.clf()

            if printEquations:
               plt.plot(xOne,yOneFitted)
               plt.plot(xTwo,yTwoFitted)
               plt.plot(xThree,yThreeFitted)
               plt.title("Pupil Response Fitted")
               plt.xlabel("Time")
               plt.ylabel("% Area")
               plt.savefig(self.output_directory+"/Graphs/Pupil Response - Percentage Area Curve Fitted.png")
               fig.clf()

            #Ouputs graphs to GUI
            self.out_queue.put("Finished PR:"+self.output_directory+"/Graphs")
            self.results_analysis_active = False
        
         if self.results_analysis_active==True and self.analysis_type == "CT":
            #Find centre of keratometer and assign a new coordinate system based off this centre
            data = pd.read_csv(self.kerIJResultsPath)
            x=data['X'].astype(float)
            y=data['Y'].astype(float)
            x_centre=x.mean()
            y_centre=y.mean()
            x_coord=x-x_centre
            y_coord=y-y_centre
            
            #Find angle from new origin
            ratio=abs(y_coord)/abs(x_coord)
            angle = pd.Series(ratio).apply(lambda x: math.atan(x)*180/math.pi)

            #Adjust angle to 0 to 360
            for i in range (0,len(x)):
               if x_coord[i]>=0 and y_coord[i]<0:
                        angle[i]=360-angle[i]
               if x_coord[i]<0:
                     if y_coord[i]<0:
                        angle[i]=180+angle[i]
                     elif y_coord[i]>0:
                        angle[i]=180-angle[i]

            #Find distances from centre
            x_coord_sq= pd.Series(x_coord).apply(lambda x: x**2)
            y_coord_sq= pd.Series(y_coord).apply(lambda x: x**2)
            dist_sq=x_coord_sq+y_coord_sq
            dist= pd.Series(dist_sq).apply(lambda x: x**0.5)

            #Drop uneeded data from results
            try:
               data.drop(['Label','Area', 'Major', 'Minor', 'Angle'], axis = 1, inplace = True)
            except:
               pass
            data["Distance"]=dist
            data["Ker Angle"]=angle
            data = data.sort_values(by=['Ker Angle','Distance'])
            data.to_csv(self.kerIJResultsPath,index=False)
            data = pd.read_csv(self.kerIJResultsPath)
            kerAngle=data["Ker Angle"]
   
            #Group by angle
            average=0
            total = 0
            count=0
            angleGroup=[0]*len(x)
            angleGroupNumber=1
            for i in range (0,len(x)):
               total+=kerAngle[i]
               count+=1
               average=total/count
               if abs(kerAngle[i]-average)>10:
                     total=kerAngle[i]
                     count=1
                     angleGroupNumber+=1
                     angleGroup[i]=angleGroupNumber   
               else:
                  angleGroup[i]=angleGroupNumber    
            
            data['Angle Group']=angleGroup
            data = data.sort_values(by=['Angle Group','Distance'])
            data.to_csv(self.kerIJResultsPath,index=False)
            data = pd.read_csv(self.kerIJResultsPath)

            #Remove close data points that are at same approximate angle
            for i in range (1,len(x)):
               if abs(data['Distance'][i]-data['Distance'][i-1])<3:
                  data['Distance'][i-1]=np.nan

            #Sort by distance
            data = data.sort_values(by=['Distance'])
            data=data.dropna(0)
            data.to_csv(self.kerIJResultsPath,index=False)
            data = pd.read_csv(self.kerIJResultsPath)
            points= (np.array(data["Distance"])).reshape(-1, 1)

            #Cluster data to form 7 groups
            kmeans= KMeans(init="k-means++",n_clusters=self.kerRings,n_init=20, max_iter=500)
            kmeans.fit(points)
            #print(kmeans.labels_)
            data["Klabels"]=kmeans.labels_
            data.to_csv(self.kerIJResultsPath,index=False)


            #Rearranges k labels for 1 to 7 (The labels are not in ascendin order after kmeans). Calculates mean radii of each ring. Calculates spot power
            powerCalcConstants = [[8.6546,6.5952, 5.3263,4.6349,4.0759,3.6618,3.3428],[0.0325,-0.0101,0.0219,0.0071,0.0182,0.0199,0.0134]]
            num=data["Klabels"][0]
            label=7
            meanRadii=[0]*7
            count=0
            total=0
            spotPower=[0]*(data.shape[0])
            for i in range (0, data.shape[0]):
               if data["Klabels"][i]!=num:
                  num = data["Klabels"][i]
                  label+=1

                  meanRadii[label-8]=total/count
                  count=0
                  total=0
               data["Klabels"][i]=label
               total+=data["Distance"][i]
               count+=1

               spotPower[i]=(1.335-1) / ((powerCalcConstants[0][(label-7)] * (data["Distance"][i] / 152.0157) + powerCalcConstants[1][(label-7)])/1000)

            #Calculate last mean radii (not computed in for loop)
            label+=1
            meanRadii[label-8]=total/count
            
            #Power calculation for keratometer rings based of mean radii
            powerScaleAdjusted = [i/152.0157 for i in meanRadii]
            cornealPower=[0]*7
            cornealPower[0] = (1.335-1) / ((8.6546 * powerScaleAdjusted[0] + 0.0325)/1000)
            cornealPower[1] = (1.335-1) / ((6.5952 * powerScaleAdjusted[1] - 0.0101)/1000)
            cornealPower[2] = (1.335-1) / ((5.3263 * powerScaleAdjusted[2] + 0.0219)/1000)
            cornealPower[3] = (1.335-1) / ((4.6349 * powerScaleAdjusted[3] + 0.0071)/1000)
            cornealPower[4] = (1.335-1) / ((4.0759 * powerScaleAdjusted[4] + 0.0182)/1000)
            cornealPower[5] = (1.335-1) / ((3.6618 * powerScaleAdjusted[5] + 0.0199)/1000)
            cornealPower[6] = (1.335-1) / ((3.3428 * powerScaleAdjusted[6] + 0.0134)/1000)
            meanPower=np.mean(cornealPower)

            #Convert radii to mm
            meanRadii = np.array(meanRadii)
            meanRadii = meanRadii/constants.pupil_scale

            #Edit CSV
            data["Klabels"]=data["Klabels"]-6
            data["Power"]=spotPower
            data.drop(['Angle Group'], axis = 1, inplace = True)
            data.to_csv(self.kerIJResultsPath,index=False)

            #Initialise arrays for ellipse points
            ellipse1_x=[]
            ellipse1_y=[]
            ellipse2_x=[]
            ellipse2_y=[]
            ellipse3_x=[]
            ellipse3_y=[]
            ellipse4_x=[]
            ellipse4_y=[]
            ellipse5_y=[]    
            ellipse5_x=[]
            ellipse6_y=[]
            ellipse6_x=[]
            ellipse7_y=[]
            ellipse7_x=[]

            #Add x and y coordinates for ellipses (offset by original roi crop)
            for i in range (0,data.shape[0]):
               if data['Klabels'][i]==1:
                     ellipse1_x.append(data['X'][i]+self.roi[0])
                     ellipse1_y.append(data['Y'][i]+self.roi[1])
               elif data['Klabels'][i]==2:
                     ellipse2_x.append(data['X'][i]+self.roi[0])
                     ellipse2_y.append(data['Y'][i]+self.roi[1])
               elif data['Klabels'][i]==3:
                     ellipse3_x.append(data['X'][i]+self.roi[0])
                     ellipse3_y.append(data['Y'][i]+self.roi[1])
               elif data['Klabels'][i]==4:
                     ellipse4_x.append(data['X'][i]+self.roi[0])
                     ellipse4_y.append(data['Y'][i]+self.roi[1])
               elif data['Klabels'][i]==5:
                     ellipse5_x.append(data['X'][i]+self.roi[0])
                     ellipse5_y.append(data['Y'][i]+self.roi[1])
               elif data['Klabels'][i]==6:
                     ellipse6_x.append(data['X'][i]+self.roi[0])
                     ellipse6_y.append(data['Y'][i]+self.roi[1])
               elif data['Klabels'][i]==7:
                     ellipse7_x.append(data['X'][i]+self.roi[0])
                     ellipse7_y.append(data['Y'][i]+self.roi[1])



            #Load original images and try to fit and plot 7 ellipses
            image = cv2.imread(self.fileName)
            startAngle = 0.
            endAngle = 360
            # Red color in BGR
            color = (0, 0, 255)
            # Line thickness of 2 px
            thickness = 2
            
            try:
               ellipse1_param =self.fit_ellipse(np.array(ellipse1_x), np.array(ellipse1_y))
               e1p_a=(int(ellipse1_param[0]),int(ellipse1_param[1])) #radius ellipse 1 parameters _ axes
               e1p_c=(int(ellipse1_param[2]),int(ellipse1_param[3])) #centre preadjusted for original image ellipse 1 parameters _ centre
               e1p_phi=ellipse1_param[4]*180/math.pi   #angle converted to degrees ellipse 1 parameters _ phi
               image = cv2.ellipse(image, e1p_c, e1p_a, e1p_phi, startAngle, endAngle, color, thickness)
            except:
               pass
            try:
               ellipse2_param=self.fit_ellipse(np.array(ellipse2_x), np.array(ellipse2_y))
               e2p_a=(int(ellipse2_param[0]),int(ellipse2_param[1])) #radius
               e2p_c=(int(ellipse2_param[2]),int(ellipse2_param[3])) #centre preadjusted for original image
               e2p_phi=ellipse2_param[4]*180/math.pi   #angle converted to degrees
               image = cv2.ellipse(image, e2p_c, e2p_a, e2p_phi, startAngle, endAngle, color, thickness)   
            except:
               pass
            try:
               ellipse3_param=self.fit_ellipse(np.array(ellipse3_x), np.array(ellipse3_y))
               e3p_a=(int(ellipse3_param[0]),int(ellipse3_param[1])) #radius
               e3p_c=(int(ellipse3_param[2]),int(ellipse3_param[3])) #centre preadjusted for original image
               e3p_phi=ellipse3_param[4]*180/math.pi   #angle converted to degrees
               image = cv2.ellipse(image, e3p_c, e3p_a, e3p_phi, startAngle, endAngle, color, thickness)
            except:
               pass
            try:
               ellipse4_param=self.fit_ellipse(np.array(ellipse4_x), np.array(ellipse4_y))
               e4p_a=(int(ellipse4_param[0]),int(ellipse4_param[1])) #radius
               e4p_c=(int(ellipse4_param[2]),int(ellipse4_param[3])) #centre preadjusted for original image
               e4p_phi=ellipse4_param[4]*180/math.pi   #angle converted to degrees
               image = cv2.ellipse(image, e4p_c, e4p_a, e4p_phi, startAngle, endAngle, color, thickness)
            except:
               pass
            try:
               ellipse5_param=self.fit_ellipse(np.array(ellipse5_x), np.array(ellipse5_y))
               e5p_a=(int(ellipse5_param[0]),int(ellipse5_param[1])) #radius
               e5p_c=(int(ellipse5_param[2]),int(ellipse5_param[3])) #centre preadjusted for original image
               e5p_phi=ellipse5_param[4]*180/math.pi   #angle converted to degrees
               image = cv2.ellipse(image, e5p_c, e5p_a, e5p_phi, startAngle, endAngle, color, thickness)
            except:
               pass
            try:
               ellipse6_param=self.fit_ellipse(np.array(ellipse6_x), np.array(ellipse6_y))
               e6p_a=(int(ellipse6_param[0]),int(ellipse6_param[1])) #radius
               e6p_c=(int(ellipse6_param[2]),int(ellipse6_param[3])) #centre preadjusted for original image
               e6p_phi=ellipse6_param[4]*180/math.pi   #angle converted to degrees
               image = cv2.ellipse(image, e6p_c, e6p_a, e6p_phi, startAngle, endAngle, color, thickness)
            except:
               pass
            try:
               ellipse7_param=self.fit_ellipse(np.array(ellipse7_x), np.array(ellipse7_y))
               e7p_a=(int(ellipse7_param[0]),int(ellipse7_param[1])) #radius
               e7p_c=(int(ellipse7_param[2]),int(ellipse7_param[3])) #centre preadjusted for original image
               e7p_phi=ellipse7_param[4]*180/math.pi   #angle converted to degrees          
               image = cv2.ellipse(image, e7p_c, e7p_a, e7p_phi, startAngle, endAngle, color, thickness)
            except:
               pass
            
            #Save image with ellipses fitted
            kerCirclesOutputPath = self.output_directory+"/Graphs/Ker_Out.png"
            cv2.imwrite(kerCirclesOutputPath,image)
            
            #Plot data
            xspacing=[1,2,3,4,5,6,7]
            fig, axs = plt.subplots(2, 2)
            axs[0, 0].scatter(xspacing, meanRadii)
            axs[0, 0].set_title('Keratometer Rings Radii')
            axs[0, 0].set_xlabel("Keratometer Ring")
            axs[0, 0].set_ylabel("Radii (mm)")
            axs[1, 0].scatter(powerScaleAdjusted, cornealPower)
            axs[1, 0].set_title('Keratometer Rings Corneal Power')
            axs[1, 0].set_xlabel("Keratometer Ring Power Scale")
            axs[1, 0].set_ylabel("Corneal Power (D)")

            x = data['X']
            y=data['Y']
            z=data['Power']

            n=500
            xlin= np.linspace((x.min()), (x.max()), n)
            ylin= np.linspace((y.min()), (y.max()), n)
            X, Y = np.meshgrid(xlin, ylin)
 
            Z= griddata((x,y),z,(X,Y))

            heatMap = axs[0, 1].pcolor(X,Y,Z, shading='auto',cmap='jet')
            fig.colorbar(heatMap, ax=axs[0, 1])
            heatMap = axs[0, 1].set_title('Topographical Map of the Cornea')
            heatMap = axs[0, 1].tick_params(which = "both", left=False,bottom=False,labelleft=False,labelbottom=False)

            image = plt.imread(self.output_directory+"/Graphs/Ker_Out.png")  
            axs[1, 1].imshow(image)
            axs[1, 1].set_title('Keratometer Rings')
            axs[1, 1].tick_params(which = "both", left=False,bottom=False,labelleft=False,labelbottom=False)
            fig.suptitle("Corneal Topography Output:\nMean Corneal Power: "+f'{(meanPower):.1f}'+" D")
            fig.tight_layout()
            fig.savefig(self.output_directory+"/Graphs/Corneal Topography Results.png")
            fig.clf()

            #Remake individual graphs
            plt.scatter(xspacing, meanRadii)
            plt.title('Keratometer Rings Radii')
            plt.xlabel("Keratometer Ring")
            plt.ylabel("Radii (mm)")
            plt.savefig(self.output_directory+"/Graphs/Keratometer Rings Radii.png")
            fig.clf()

            plt.scatter(powerScaleAdjusted, cornealPower)
            plt.title('Keratometer Rings Corneal Power')
            plt.xlabel("Keratometer Ring Power Scale")
            plt.ylabel("Corneal Power (D)")
            plt.savefig(self.output_directory+"/Graphs/Keratometer Rings Corneal Power.png")
            fig.clf()

            plt.pcolor(X,Y,Z, shading='auto',cmap='jet')
            plt.colorbar()
            plt.title('Topographical Map of the Cornea')
            plt.tick_params(which = "both", left=False,bottom=False,labelleft=False,labelbottom=False)
            plt.savefig(self.output_directory+"/Graphs/Topographical Map of the Cornea.png")

            #Add ring data to CSV
            space=[np.nan]*data.shape[0]
            ringNumber=[np.nan]*data.shape[0]
            ringRadii=[np.nan]*data.shape[0]
            ringPower=[np.nan]*data.shape[0]

            for i in range (0,7):
               ringNumber[i]=i+1
               ringRadii[i]=meanRadii[i]
               ringPower[i]=cornealPower[i]

            data["_"]=space
            data["Ring Number"]=ringNumber
            data["Ring Radii"]=ringRadii
            data["Ring Corneal Power"]=ringPower
            data.to_csv(self.kerIJResultsPath,index=False)
            
            


            self.out_queue.put("Finished PR:"+self.output_directory+"/Graphs*"+f'{(meanPower):.1f}')
            self.results_analysis_active=False


   #Thread for converting a video input into frames and saves to the otuput folder as a .png 
   def video_to_frames_thread(self):
      while self.running:
         self.sleep()
         if self.video_to_frames_active == True:
            cap = cv2.VideoCapture(self.videofilename)
            # Find the number of frames
            video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
            count = 0
            print ("Converting video..\n")
            # Start converting the video

            try:
               self.fps = int(cap.get(cv2.CAP_PROP_FPS))
            except:
               pass


            while cap.isOpened():
               # Extract the frame
               ret, frame = cap.read()

               # Write the results back to output location.
               cv2.imwrite(self.output_directory_frames + "/%#04d.png" % (count+1), frame)
               count = count + 1
               prog=float(count/video_length*100)
               self.out_queue.put("video to frames progress:"+str(prog))
               # If there are no more frames left
               if (count > (video_length-1)):
                     # Release the feed
                     cap.release()
                     break
            self.video_to_frames_active=False
   
   #Enhances image contrast with filtering and histogram equalisation
   def enhanceContrast(self, fileName, inputPath):
      # Enhance contrast
      image = cv2.imread(fileName)
      #Change to greyscale
      image_src = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      #Bilateral filter is simialr to gaussian but better at maintaining edges. Slow process so only implem,ented once
      bilateralFilter = cv2.bilateralFilter(image_src,-1,10,5)
      #Equalize histogram ensures full range from 0 to 255 is used
      image_eq = cv2.equalizeHist(bilateralFilter)
      #Threshold so values under 40 are white
      ret,image_adjusted = cv2.threshold(image_eq,40,255,cv2.THRESH_TOZERO)
      #Median blur effective for salt and pepper
      median = cv2.medianBlur(image_adjusted,3)
      #Threshold so values under 20 are white
      ret,image_adjusted = cv2.threshold(median,20,255,cv2.THRESH_TOZERO)
      cv2.imwrite(inputPath,image_adjusted)  
   
   #Used to fit ellipse to given data points
   #Implemented from https://github.com/ndvanforeest/fit_ellipse/blob/master/fitEllipse.py
   def fit_ellipse(self, x, y):
      """@brief fit an ellipse to supplied data points: the 5 params
         returned are:
         M - major axis length
         m - minor axis length
         cx - ellipse centre (x coord.)
         cy - ellipse centre (y coord.)
         phi - rotation angle of ellipse bounding box
      @param x first coordinate of points to fit (array)
      @param y second coord. of points to fit (array)
      """
      a = self.__fit_ellipse(x, y)
      centre = self.ellipse_center(a)
      phi = self.ellipse_angle_of_rotation(a)
      M, m = self.ellipse_axis_length(a)
      # assert that the major axself.ix M > minor axis m
      if m > M:
         M, m = m, M
      # ensure the angle is betwen 0 and 2*pi
      phi -= 2 * np.pi * int(phi / (2 * np.pi))
      return [M, m, centre[0], centre[1], phi]

   #Called from fit_ellipse
   def __fit_ellipse(self, x, y):
      x, y = x[:, np.newaxis], y[:, np.newaxis]
      D = np.hstack((x * x, x * y, y * y, x, y, np.ones_like(x)))
      S, C = np.dot(D.T, D), np.zeros([6, 6])
      C[0, 2], C[2, 0], C[1, 1] = 2, 2, -1
      U, s, V = svd(np.dot(inv(S), C))
      a = U[:, 0]
      return a

   #Called from fit_ellipse
   def ellipse_center(self, a):
      b, c, d, f, g, a = a[1] / 2, a[2], a[3] / 2, a[4] / 2, a[5], a[0]
      num = b * b - a * c
      x0 = (c * d - b * f) / num
      y0 = (a * f - b * d) / num
      return np.array([x0, y0])

   #Called from fit_ellipse
   def ellipse_axis_length(self, a):
      b, c, d, f, g, a = a[1] / 2, a[2], a[3] / 2, a[4] / 2, a[5], a[0]
      up = 2 * (a * f * f + c * d * d + g * b * b - 2 * b * d * f - a * c * g)
      down1 = (b * b - a * c) * (
         (c - a) * np.sqrt(1 + 4 * b * b / ((a - c) * (a - c))) - (c + a)
      )
      down2 = (b * b - a * c) * (
         (a - c) * np.sqrt(1 + 4 * b * b / ((a - c) * (a - c))) - (c + a)
      )
      res1 = np.sqrt(up / down1)
      res2 = np.sqrt(up / down2)
      return np.array([res1, res2])

   #Called from fit_ellipse
   def ellipse_angle_of_rotation(self, a):
      b, c, d, f, g, a = a[1] / 2, a[2], a[3] / 2, a[4] / 2, a[5], a[0]
      return math.atan2(2 * b, (a - c)) / 2

   #Used to crop keratometer image using additional gui, mouse and keyboard
   def crop_ker(self):
      self.roi=(0,0,0,0)
      self.drawing=False
      self.ix, self.iy = -1,-1
      
      #Image and Folder Naming
      self.fileList= [f for f in glob.glob(self.input_folder+ "**/*"+self.fileType, recursive=True)]
      self.fileName=self.fileList[self.imageNumber]
      basename = os.path.basename(self.fileName)
      name=basename.split(".")
      self.fullFileNum = name[0]
      outputName=self.fullFileNum+"_binary"+self.fileType
      self.kerInputPath = self.output_directory+"/Binaries/"+outputName
      outputName=self.fullFileNum+"_cropped"+self.fileType
      self.cropOutputPath=self.output_directory+"/Enhanced Contrast/"+outputName

      #Run Enhance Contrast
      self.enhanceContrast(self.fileName,self.kerInputPath)
      
      #Display image for cropping
      self.img = cv2.imread(self.kerInputPath)
      self.img_original=cv2.imread(self.kerInputPath)
      img_l, img_w, ch = self.img.shape

      # window/screen stuff
      cv2.namedWindow("image")
      cv2.setMouseCallback("image", self.draw_rectangle)
      cv2.putText(self.img, text="Drag a rectangle to crop the keratometer dots.", org=(130, 50),fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0,0,255),
               thickness=1, lineType=cv2.LINE_AA)
      cv2.putText(self.img, text="""Press "s" to save the crop and continue, "c" to clear the rectangle and start again or "q" to quit.""", org=(130, 65),fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0,0,255),
         thickness=1, lineType=cv2.LINE_AA)
      cv2.imshow("image", self.img)
      RUN=1
      while RUN:
         
         k = cv2.waitKey(1) & 0xFF
         
         if (k == 27 or k == ord("q") ): # exit
               RUN=0
               cv2.destroyAllWindows()
               self.exit_event()
               break

         if (self.roi != (0, 0, 0, 0) ): # if ROI has been created
               roi_x, roi_y, roi_w, roi_h = self.roi
               x, y, w, h = self.get_roi(roi_x, roi_y, roi_w, roi_h) # returns the ROI from original image
               

               if ( (x, y, w, h) == (0, 0, 0, 0) ): # checks for bug when mouse is clicked but no box is created
                  img_roi = None
                  img_roi_original=None
               else:
                  img_roi = self.img[y:h, x:w, :]
                  img_roi_original=self.img_original[y:h, x:w, :]

               if (img_roi is None): # bug check - need to identify small boxes
                  print("\nERROR:\tROI " + str(self.roi) + " is Out-of-Bounds OR not large enough")
                  cv2.destroyWindow("roi")
                  self.roi = (0, 0, 0, 0) # might already be set
                  tmp_img = self.img.copy()
                  cv2.imshow("image", tmp_img)
                  
                  
               elif(k == ord("s") ): # save roi after viewing it
                  cv2.imwrite(self.cropOutputPath, img_roi_original)           
                  cv2.destroyWindow("roi")
                  cv2.destroyAllWindows()
                  RUN=0
                  self.image_analysis_active = True

                  
               elif(k == ord("c") ): # clear roi
                  print("\n\tCleared ROI " + str(self.roi) )
                  self.roi = (0, 0, 0, 0)
                  cv2.destroyWindow("roi")
                  tmp_img = self.img.copy()
                  cv2.imshow("image", tmp_img)

   #Called from crop_ker
   #Normalises the x, y, w, h coords when dragged from different directions
   def get_roi(self, x, y, w, h):
      if (y > h and x > w): # lower right to upper left
         return (w, h, x, y)
      elif (y < h and x > w): # upper right to lower left
         return (w, y, x, h)
      elif (y > h and x < w): # lower left to upper right
         return(x, h, w, y)
      elif (y == h and x == w):
         return (0, 0, 0, 0) # roi too small
      else:
         return (x, y, w, h) # upper left to lower right

   #Mouse callback function for crop_ker
   def draw_rectangle(self, event, x, y, flags, param):
      
      if (event == cv2.EVENT_LBUTTONDOWN):
         self.drawing = True
         self.ix, self.iy = x, y
         
      elif (event == cv2.EVENT_MOUSEMOVE):
         if self.drawing == True:
               tmp_img = self.img.copy()
               cv2.rectangle(tmp_img, (self.ix, self.iy), (x, y), (0, 255, 0), 1)
               cv2.imshow("image", tmp_img)
               
      elif (event == cv2.EVENT_LBUTTONUP):
         tmp_img = self.img.copy()
         self.drawing = False
         self.roi = (self.ix, self.iy, x, y)
         cv2.rectangle(tmp_img, (self.ix, self.iy), (x, y), (0, 255, 0), 1)

   #Curve fitting functions for PR and PIPR Data
   #The curve is split into 3 segments giving three equations
   #The constant is used to set the intersection point
   def funcOne(self,x, a, b):
      return a + 0*b*x
   def funcTwo(self,x, c, d, e, f):
      return  (self.aConstant-(d*self.xinflexionOne + e*self.xinflexionOne**2 + np.exp(f * self.xinflexionOne))) + d*x + e*x**2 + np.exp(f * x)     
   def funcThree(self,x, g, h, i, j):
      return (self.cConstant-(h*self.xinflexionTwo + i*self.xinflexionTwo**2 + np.exp(j * self.xinflexionTwo))) + h*x + i*x**2 + np.exp(j * x)

   #Closes GUI and ends program
   def exit_event(self):
      self.master.destroy()
      self.running = 0
      sys.exit(0)

   #Sleep for threads when not in use
   def sleep(self):
      time.sleep(constants.THREAD_DELAY_PROGRAMMING_THREAD)

   #Processes incoming messages from the GUI thread
   def process_incoming(self):
      while self.in_queue.qsize():
         try:
            msg = self.in_queue.get(0)
            # Get the payload type
            payload_type = msg.split(':')[0]
            payload = msg.split("{}:".format(payload_type))[1]
            if payload_type == "start session":
                  self.start_session_event()
            elif payload_type == "start image analysis":
               if self.image_analysis_active == False:
                  print("Start image_analysis")
                  x=payload.split("*")
                  self.holdx=x
                  self.imageNumber=int(x[0])
                  self.input_folder=x[1]
                  self.fileType=x[2]
                  self.output_directory=x[3]
                  self.analysis_type=x[4]
                  self.CACdepth=float(x[5])
                  self.cornealPower=float(x[6])
                  self.kerRings=int(x[7])
                  if self.analysis_type=="PR":
                     self.total_number_images =200
                     self.image_analysis_active = True
                     #self.results_analysis_active = True
                  elif self.analysis_type=="PIPR":
                     self.total_number_images =1000
                     self.image_analysis_active = True
                  elif self.analysis_type=="CT":
                     print("corneal topography")
                     #self.image_analysis_active = True
                     self.crop_ker()
               else:
                  print("?")
            elif payload_type == "finish":
               self.finish_event()
            elif payload_type == "exit":
               self.exit_event()
            elif payload_type == "video to frames":
               self.videofilename=payload.split('*')[0]
               self.output_directory_frames=payload.split('*')[1]
               self.video_to_frames_active = True
            else:
               print("?")
         except queue.Empty:
            pass
      # Write outgoing messages
      #msg = "test:"
      #self.out_queue.put(msg)

#Creates tkinter instance
#Calls threading client for the main application
def main():
   # GUI (main thread)def
   master = tk.Tk()
   # start worker threads
   ThreadedClient(master)
   # start main thread
   master.mainloop()

if __name__ == "__main__":
   main()

import tkinter.filedialog as filedialog
import tkinter as tk
from tkinter import ttk
import sys
import constants
import queue
import threading
import glob
from PIL import ImageTk, Image
import glob
import select
import stat
import os

#User Interface Thread
class UserInterface:
   def __init__(self, threaded_client, master, in_queue, out_queue):
      self.threaded_client = threaded_client
      self.master = master
      self.in_queue = in_queue
      self.out_queue = out_queue

      #Initialise variables
      self.enable_image_analysis = True
      self.enable_results_analysis = False
      self.enable_finish = False
      self.finished=False
      self.input_type=""
      self.analysis_type=""
      self.CAC=-1
      self.cornealPower=-1
      self.kerRings=7

      tk.Tk.wm_title(self.master, constants.USER_INTERFACE_NAME)
      master.protocol("WM_DELETE_WINDOW", self.close)
      self.build_gui()
      
   #Creates starting GUI
   def build_gui(self):
      #grid setup
      self.frame=tk.Frame(self.master,height=constants.USER_INTERFACE_INITIAL_HEIGHT,
                   width=constants.USER_INTERFACE_INITIAL_WIDTH)
      self.frame.grid(row=0, column=0, sticky="nsew")
      for i in range (0,11):
        self.master.grid_rowconfigure(i, weight=1)
      for i in range (0,8):
          self.master.grid_columnconfigure(i, weight=1)

      #Analsyis type and Input type
      self.var_analysis_type=tk.StringVar()
      self.var_input_type=tk.StringVar()
      self.var_image_type=tk.StringVar()
      self.var_analysis_type.set("")
      self.var_input_type.set("")
      self.var_image_type.set("")

      self.question_analysis_type=tk.Label(self.frame, text="What type of Analysis?").grid(row=0,column=0,columnspan=3)
      tk.Radiobutton(self.frame, text="Pupil Response", variable = self.var_analysis_type, value='PR').grid(row=1,column=0)
      tk.Radiobutton(self.frame, text="PIPR", variable = self.var_analysis_type, value='PIPR').grid(row=1,column=1)    
      tk.Radiobutton(self.frame, text="Keratometry", variable = self.var_analysis_type, value='CT').grid(row=1,column=2) #CT for corneal topography

      self.spacing_one=tk.Label(self.frame, text="").grid(row=2,column=0)

      self.question_input_type=tk.Label(self.frame, text="Analyse a Video or Folder of Images?").grid(row=3,column=0,columnspan=3)
      tk.Radiobutton(self.frame, text="Video", variable =  self.var_input_type,value='vid').grid(row=4,column=0)
      tk.Radiobutton(self.frame, text="Folder of Images", variable =  self.var_input_type, value='im').grid(row=4,column=1)

      tk.Radiobutton(self.frame, text="TIF", variable = self.var_image_type,value='.TIF').grid(row=5,column=0)
      tk.Radiobutton(self.frame, text="JPG", variable = self.var_image_type, value='.jpg').grid(row=5,column=1) 
      tk.Radiobutton(self.frame, text="PNG", variable = self.var_image_type, value='.png').grid(row=5,column=2) 
      
      self.spacing_two=tk.Label(self.frame, text="").grid(row=6,column=0)
      
      #Anterior Chamber Depth and Corneal Power
      self.var_CAC_Depth=tk.StringVar()
      self.var_CAC_Depth.set("")
      self.var_Corneal_Power=tk.StringVar()
      self.var_Corneal_Power.set("")
      self.var_Ker_Rings=tk.StringVar()
      self.var_Ker_Rings.set("")

      self.question_CAC=tk.Label(self.frame, text="What is the AC depth (mm)?").grid(row=7, column=0, columnspan=2)
      tk.Entry(self.frame, textvariable=self.var_CAC_Depth).grid(row=7,column=2)
      self.question_Corneal_Power=tk.Label(self.frame, text="What is the corneal power (D)?").grid(row=8, column=0, columnspan=2)
      tk.Entry(self.frame, textvariable=self.var_Corneal_Power).grid(row=8,column=2)
      #self.question_Rings=tk.Label(self.frame, text="How many keratometr rings? (Default 7)").grid(row=9, column=0, columnspan=2)
      #tk.Entry(self.frame, textvariable=self.var_Ker_Rings).grid(row=9,column=2)


      #Buttons and scrollbarfor selecting images
      self.buttons = {}
      # Button ids is a dictionary of ids to trigger callbacks when buttons are pressed
      self.button_ids = {}
      self.button_ids['back'] = 1
      self.button_ids['select'] = 2
      self.button_ids['forward'] = 3

      self.buttons['back'] = tk.Button(self.frame, text="<<", font='Helvetica 12',
                              command=lambda: self.button_press(self.button_ids['back']))
      self.buttons['back'].grid(row=9, column=3)
      self.buttons['select'] = tk.Button(self.frame, text="Start Analysis", font='Helvetica 12',
                              command=lambda: self.button_press(self.button_ids['select']))
      self.buttons['select'].grid(row=10, column=3, columnspan=5)
      self.buttons['forward'] = tk.Button(self.frame, text=">>", font='Helvetica 12',
                              command=lambda: self.button_press(self.button_ids['forward']))
      self.buttons['forward'].grid(row=9, column=7)  
      
      self.fileListLength=1 #disabled due to no range
      self.scrollbar = tk.Scale(self.frame, from_ = 1, to = self.fileListLength, orient='horizontal',
                              command= self.scrolling_event)
      self.scrollbar.set(1)
      self.scrollbar.grid(row=9, column=4, columnspan=3)
      #Disable buttons
      self.disable_buttons()

      #Console   
      self.output = tk.Text(self.frame, bg='white', height=2)
      self.output.grid(row=7, column=3,columnspan=5,rowspan=2)


     #Browse
      self.browse_button = tk.Button(self.frame, text="Browse", font='Helvetica 12',
                              command=self.select_input)
      self.browse_button.grid(row=3, column=5)
      self.frame.pack()

   #Checks selected input from starting GUI and creates window for browsing for files or folders
   def select_input(self):
      
      #Check variables
      if self.var_analysis_type.get()!="":
          self.analysis_type=self.var_analysis_type.get()
      if self.var_input_type.get()!="":
          self.input_type=self.var_input_type.get()
      if self.var_image_type.get()!="":
          self.image_type=self.var_image_type.get()
      if self.var_CAC_Depth.get()!="":
          self.CAC=self.var_CAC_Depth.get()
      if self.var_Corneal_Power.get()!="":
          self.cornealPower=self.var_Corneal_Power.get()
      if self.var_Ker_Rings.get()!="":
         self.kerRings=self.var_Ker_Rings.get()


      if self.input_type == "vid" and self.var_analysis_type.get()!="":
         self.filename = filedialog.askopenfilename(initialdir = "./", 
                                    title = "Select a File", 
                                    filetypes = (("AVI Files","*.avi"),("all files","*.*")))
         if self.filename != None:
            self.browse_button.grid_forget()
            self.create_output_folders()
            self.video_to_frames_gui()
            msg="video to frames:"+self.filename+"*"+self.output_directory_frames
            self.out_queue.put(msg)
            self.input_folder=self.output_directory_frames
         else:
            self.output.config(state=tk.NORMAL)
            self.output.insert(tk.END ,"Please select an input.")
            self.output.config(state=tk.DISABLED)
            self.output.see(tk.END)         
      elif self.input_type == "im" and self.var_analysis_type.get()!="" and self.var_image_type.get()!="":
         self.input_folder = filedialog.askdirectory(initialdir = "./", 
                                    title = "Select a Folder")
         print(self.input_folder)
         if self.input_folder != None:
            self.browse_button.grid_forget()
            self.enable_buttons()
            self.create_output_folders()
            self.select_starting_image()
         else:
            self.output.config(state=tk.NORMAL)
            self.output.insert(tk.END ,"Please select an input.")
            self.output.config(state=tk.DISABLED)
            self.output.see(tk.END)             

      else:
        self.output.config(state=tk.NORMAL)
        self.output.insert(tk.END ,"Ensure an analysis type and input type have been selected. Anterior chamber depth and corneal power are optional inputs for PR and PIPR.")
        self.output.config(state=tk.DISABLED)
        self.output.see(tk.END)

   #Creates output folders for results
   def create_output_folders(self):
      if self.input_type=="vid":
         head, tail = os.path.split(self.filename)
         self.output_directory=head
         x=tail.split(".")
         fileName=x[0]
         #Results
         try:
            self.output_directory_results= self.output_directory+"/"+fileName+"_Results"
            os.mkdir(self.output_directory_results)
         except OSError:
            pass  
         #FRAMES
         try:
            self.output_directory_frames=self.output_directory_results+"/Frames"
            os.mkdir(self.output_directory_frames)     
         except OSError:
            pass
      elif  self.input_type=="im":
         self.output_directory=self.input_folder
         #Results
         try:
            self.output_directory_results= self.output_directory+"/Results"
            os.mkdir(self.output_directory_results)
         except OSError:
            pass 
    
      #BINARIES
      try:
         self.output_directory_binaries= self.output_directory_results+"/Binaries"
         os.mkdir(self.output_directory_binaries)
      except OSError:
         pass
      #GRAPHS
      try:
         self.output_directory_graphs= self.output_directory_results+"/Graphs"
         os.mkdir(self.output_directory_graphs)
      except OSError:
         pass
      #ENHANCED CONTRAST
      try:
         self.output_directory_contrast= self.output_directory_results+"/Enhanced Contrast"
         os.mkdir(self.output_directory_contrast)
      except OSError:
         pass
      print("Create")

   #Used to enable image selection buttons 
   def enable_buttons(self):
      self.buttons['back']["state"]=tk.NORMAL
      self.buttons['forward']["state"]=tk.NORMAL
      self.buttons['select']["state"]=tk.NORMAL
      self.scrollbar.config(command=self.scrolling_event)

   #Used to disable image selection buttons    
   def disable_buttons(self):
      self.buttons['back']["state"]=tk.DISABLED
      self.buttons['forward']["state"]=tk.DISABLED
      self.buttons['select']["state"]=tk.DISABLED
      self.scrollbar.config(command="")    

   #Creates progress bar for the frame extraction from a video input
   def video_to_frames_gui(self): 
      self.image_type = ".png"
      self.video_to_frames_label=tk.Label(self.frame, text="Converting Video to Frames")
      self.video_to_frames_label.grid(row=3, column=4,columnspan=3)
      s = ttk.Style()
      s.theme_use('clam')
      s.configure("green.Horizontal.TProgressbar", foreground='green', background='green')
      self.progress_video_to_frames = ttk.Progressbar(self.frame, style="green.Horizontal.TProgressbar", orient="horizontal", length=100, mode="determinate")
      self.progress_video_to_frames.grid(row=4, column=4,columnspan=3)
      self.progress_video_to_frames['value']=0
      self.frame.pack()   

   #Places image on GUI. Enables scrollbar for image selection.
   def select_starting_image(self):
      #Image and Image Number
      self.fileList= [f for f in glob.glob(self.input_folder+ "/*"+self.image_type, recursive=True)]
      self.fileListLength = len(self.fileList)
      self.imageNumber=0
      im=Image.open(self.fileList[self.imageNumber])
      im.thumbnail((445,625),Image.ANTIALIAS)
      self.pupilImage=ImageTk.PhotoImage(im)
      self.pupilImageLabel=tk.Label(self.frame,image=self.pupilImage)
      self.pupilImageLabel.grid(row=0, column=3, columnspan=7,rowspan=7)
      self.write_console("Image Number: "+str(self.imageNumber+1))
      self.scrollbar.grid_forget()
      self.scrollbar = tk.Scale(self.frame, from_ = 1, to = self.fileListLength-1, orient='horizontal',
                              command= self.scrolling_event)
      self.scrollbar.set(1)
      self.scrollbar.grid(row=9, column=4, columnspan=3)
      self.frame.pack()
   
   #Creates progress bar for the analysis of images in PR or PIPR
   def progressbar_update(self, prog):
         self.progress['value']=prog

   #Event called by button press on GUI. This will then call the required function based off the button ID
   def button_press(self, button_id):
      if button_id == self.button_ids['back']:
         self.back_event()
      elif button_id == self.button_ids['select']:
         self.select_event()
      elif button_id == self.button_ids['forward']:
         self.forward_event()

   #Used to close the main thread
   def close(self):
      self.out_queue.put("exit:")

   #Previous Image
   def back_event(self):
      self.pupilImageLabel.grid_forget()
      if self.imageNumber > 0:
         self.imageNumber -=1
      
      im=Image.open(self.fileList[self.imageNumber])
      im.thumbnail((round(7*700/11),round(5*1000/8)),Image.ANTIALIAS)
      self.pupilImage=ImageTk.PhotoImage(im)
      self.pupilImageLabel=tk.Label(self.frame,image=self.pupilImage)
      self.pupilImageLabel.grid(row=0, column=3, columnspan=7,rowspan=7)
      if self.analysis_type=="CT" and self.finished==True:
         self.write_console("Corneal Power: "+self.corneal_power_out+" D\nImage Number: "+str(self.imageNumber+1))
      else:
         self.write_console("Image Number: "+str(self.imageNumber+1))
      
      self.scrollbar.set(self.imageNumber)

      self.frame.pack()

   #Pass folders/files to main application script  
   def select_event(self):
      if self.enable_image_analysis == True:
         self.pupilImageLabel.grid_forget()
         self.disable_buttons()
         if self.analysis_type  == "PR":
            if self.imageNumber<15:
               self.imageNumber=15
               print("Bad photo range")
            elif self.imageNumber>(self.fileListLength-186):
               self.imageNumber=self.fileListLength-186
               print("Bad photo range")
               if self.imageNumber<0:
                  self.close()
         elif self.analysis_type == "PIPR":
            if self.imageNumber<15:
               self.imageNumber=15
               print("Bad photo range")
            elif self.imageNumber>(self.fileListLength-984):
               self.imageNumber=self.fileListLength-984
               print("Bad photo range")
               if self.imageNumber<0:
                  self.close()


         self.write_console("Image Number: "+str(self.imageNumber+1)+"\nResults will be saved to "+self.output_directory)        
         if self.analysis_type == "PR" or self.analysis_type=="PIPR":
            self.analysis_label=tk.Label(self.frame, text="Analysing Pupillometry Images")
            self.analysis_label.grid(row=3, column=4,columnspan=3)
            s = ttk.Style()
            s.theme_use('clam')
            s.configure("green.Horizontal.TProgressbar", foreground='green', background='green')
            self.progress = ttk.Progressbar(self.frame, style="green.Horizontal.TProgressbar", orient="horizontal", length=100, mode="determinate")
            self.progress.grid(row=4, column=4,columnspan=3)
            self.progress['value']=0
         else:
            self.analysis_label=tk.Label(self.frame, text="Analysing Keratometry Image")
            self.analysis_label.grid(row=3, column=4,columnspan=3)

         self.frame.pack()

         msg = "start image analysis:"+str(self.imageNumber)+"*"+self.input_folder+"*"+self.image_type+"*"+self.output_directory_results+"*"+self.analysis_type+"*"+str(self.CAC)+"*"+str(self.cornealPower)+"*"+str(self.kerRings)
         self.out_queue.put(msg)

   #Next Image
   def forward_event(self):
      self.pupilImageLabel.grid_forget()
      if self.imageNumber < self.fileListLength:
         self.imageNumber +=1
      
      im=Image.open(self.fileList[self.imageNumber])
      im.thumbnail((round(7*700/11),round(5*1000/8)),Image.ANTIALIAS)
      self.pupilImage=ImageTk.PhotoImage(im)
      self.pupilImageLabel=tk.Label(self.frame,image=self.pupilImage)
      self.pupilImageLabel.grid(row=0, column=3, columnspan=7,rowspan=7)
      if self.analysis_type=="CT" and self.finished==True:
         self.write_console("Corneal Power: "+self.corneal_power_out+" D\nImage Number: "+str(self.imageNumber+1))
      else:
         self.write_console("Image Number: "+str(self.imageNumber+1))
      
      self.scrollbar.set(self.imageNumber)

      self.frame.pack()

   #Allows quick scrolling through images
   def scrolling_event(self, val):
      self.pupilImageLabel.grid_forget()
      self.imageNumber=int(val)
      
      im=Image.open(self.fileList[self.imageNumber])
      im.thumbnail((round(7*700/11),round(5*1000/8)),Image.ANTIALIAS)
      self.pupilImage=ImageTk.PhotoImage(im)
      self.pupilImageLabel=tk.Label(self.frame,image=self.pupilImage)
      self.pupilImageLabel.grid(row=0, column=3, columnspan=7,rowspan=7)
      if self.analysis_type=="CT" and self.finished==True:
         self.write_console("Corneal Power: "+self.corneal_power_out+" D\nImage Number: "+str(self.imageNumber+1))
      else:
         self.write_console("Image Number: "+str(self.imageNumber+1))

      self.frame.pack()

   #Displays analysis results on GUI after the main thread has saved the graphs
   def display_output(self, path, cornealPower):
      self.corneal_power_out=cornealPower
      self.enable_buttons()
      self.analysis_label.grid_forget()
      
      #No progress bar for CT
      if self.analysis_type=="PR"or self.analysis_type=="PIPR":   
         self.progress.grid_forget()

      self.buttons['select'].grid_forget()
      self.reset_button = tk.Button(self.frame, text="Analyse a New Dataset", font='Helvetica 12',
                              command=self.reset)
      self.reset_button.grid(row=10, column=3, columnspan=5)

      self.fileList= [f for f in glob.glob(path+ "/*.png", recursive=True)]
      self.fileListLength = len(self.fileList)
      print(self.fileListLength)
      self.imageNumber=0
      im=Image.open(self.fileList[self.imageNumber])
      im.thumbnail((round(7*700/11),round(5*1000/8)),Image.ANTIALIAS)
      self.pupilImage=ImageTk.PhotoImage(im)
      self.pupilImageLabel=tk.Label(self.frame,image=self.pupilImage)
      self.pupilImageLabel.grid(row=0, column=3, columnspan=7,rowspan=7)
      if self.analysis_type=="CT":
         #Write the corneal power and chosen image number to the console
         self.write_console("Corneal Power: "+self.corneal_power_out+" D\nImage Number: "+str(self.imageNumber+1))
      else:
         self.write_console("Image Number: "+str(self.imageNumber+1))
      self.scrollbar.grid_forget()
      self.scrollbar = tk.Scale(self.frame, from_ = 1, to = self.fileListLength, orient='horizontal',
                              command= self.scrolling_event)
      self.scrollbar.set(1)
      self.scrollbar.grid(row=9, column=4, columnspan=3)
      self.frame.pack()
      
   #Enables the analysis of multiple datasets in without rerunning application.py
   def reset(self):
      self.enable_image_analysis = True
      self.enable_results_analysis = False
      self.enable_finish = False
      self.finished=False

      self.input_type=""
      self.analysis_type=""
      self.CAC=-1
      self.cornealPower=-1

      self.filename=""
      self.input_folder=""

      self.frame.destroy()

      self.build_gui()

   #Writes to a console on the GUI for easy communication from the program to the user
   def write_console(self, txt):
      #clear
      self.output.config(state=tk.NORMAL)
      self.output.delete('1.0', tk.END)
      self.output.config(state=tk.DISABLED)
      #write
      self.output.config(state=tk.NORMAL)
      self.output.insert(tk.END ,str(txt))
      self.output.config(state=tk.DISABLED)
      self.output.see(tk.END)

   #Processes incoming messages from the main thread
   def process_incoming(self):
      while self.in_queue.qsize():
         try:
            msg = self.in_queue.get(0)
            print(msg)
            # Get the payload type
            payload_type = msg.split(':')[0]
            payload = msg.split("{}:".format(payload_type))[1]
            if payload_type == "progress":
               self.progressbar_update(payload)
            elif payload_type == "video to frames progress":
               if float(payload)<100:
                  self.progress_video_to_frames['value']=payload
               else:
                  self.progress_video_to_frames.grid_forget()
                  #self.video_to_frames_label.grid_forget()
                  self.enable_buttons()
                  self.select_starting_image()
            elif payload_type== "Finished PR":
                  self.finished=True
                  if self.analysis_type=="CT":
                     a=payload.split("*")
                     payload=a[0]
                     x=a[1]
                  else:
                     x=0
                  self.display_output(payload,x)
         except queue.Empty:
            # just on general principles, although we don't
            # expect this branch to be taken in this case
            pass
   
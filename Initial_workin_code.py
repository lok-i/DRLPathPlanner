
"""
Created on Thu Sep 28 20:52:25 2017
@author: Administrator
"""
import tensorflow as tf
import numpy as np
import time
import random
from math import sqrt,cos,sin,atan2
import pygame
from pygame.locals import *
from sys import exit
import os
import matplotlib.pyplot as plt
# import tensorflow.compat.v1 as tf1
# tf1.disable_v2_behavior()
#tf.compat.v1.disable_v2_behavior()


pygame.init()
fpsClock = pygame.time.Clock()
XDIM = 640
YDIM = 480
SCREEN_SIZE = (XDIM, YDIM)
screen = pygame.display.set_mode(SCREEN_SIZE, 0, 32)
pygame.display.set_caption("NAVIGATION!")
show_data = True

test_mode = True #False      
load_model = False         

class vector(object):
    def __init__(self,x=0,y=0):
        self.x = x
        self.y = y
    def cal_vector(self,p1,p2):
        self.x = p2[0]-p1[0]
        self.y = p2[1]-p1[1]
        return (self.x,self.y) 
    def get_magnitude(self):
        return sqrt(self.x**2+self.y**2)
 
def get_point(vector,p1):
    return (vector[0]+p1[0],vector[1]+p1[1])

def get_distance(p1,p2):              
    return sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)

class lidar(object):
    def __init__(self,p,distance=200,angle=0):
        self.x = p[0]
        self.y = p[1]
        self.distance = distance  
        self.angle = 0
        self.lidar =(0,0)
        self.data = [] 
        self.data_2 = [] 
    def scan(self):
        dis = []
        for i in range(0,360,1):  
            for j in range(1,40):
                x1 = int(self.x + self.distance*cos(i*3.14/180)*j/40)
                y1 = int(self.y + self.distance*sin(i*3.14/180)*j/40)
                #print(x1,y1)
                pix = (0,0,0)
                if(x1>0 and x1<XDIM and y1>0 and y1<YDIM): 
                    pix = screen.get_at((x1,y1))
                    #print(pix)
                
                if(x1<=0 or x1>=XDIM or y1<=0 or y1>=YDIM):
                    break
                elif(pix[0]>100 and pix[1]>100 and pix[2]>100 ):
                    break
                else:
                    dis = [x1,y1]
            if len(self.data)<360:
                self.data.append(dis)                        
            else:
                self.data[i] = dis
            #print(self.data[i])
            if(self.x>0 and self.x<XDIM and self.y>0 and self.y<YDIM):      
                vec = vector()
                vec.cal_vector((self.x,self.y),self.data[i])
                scan_dis = vec.get_magnitude()
                if(len(self.data_2)<360):
                    self.data_2.append([i,scan_dis])         
                else:
                    self.data_2[i] = [i,scan_dis]
                if(scan_dis<190):
                    screen.set_at(self.data[i],(255,0,0))
                
    def show(self):
        for d in self.data:
            vect = vector()
            vect.cal_vector((self.x,self.y),d)
            if(vect.get_magnitude()<190):
                screen.set_at(d,(255,0,0))
        return self.data
        
    def state(self):
        return self.data
        
    def state_2(self):
        return self.data_2
        
    def pos_change(self,pos):
        self.x = pos[0]
        self.y = pos[1]
        
                
                    
class car(object):
    def __init__(self,x=50,y=50,length=20,width=20):
        self.x = x
        self.y = y
        self.length = length
        self.width = width
        self.minx = int(self.x - self.length//2)
        self.miny = int(self.y - self.width//2)
        self.maxx = int(self.x + self.length//2)
        self.maxy = int(self.y + self.width//2)
    def show(self):
        if(self.minx>0 and self.miny>0 and self.maxx<XDIM and self.maxy<YDIM):
            pygame.draw.rect(screen,(90,90,90),(self.minx,self.miny,self.length,self.width))
            return True
        else:
            print("error:the car Out of bounds")
            return False
    def move(self,x,y):
        if(self.minx+x>0 and self.miny+y>0 and self.maxx+x<XDIM and self.maxy+y<YDIM):
            self.x += x
            self.y += y
            self.minx += x
            self.miny += y
            self.maxx += x
            self.maxy += y
            return True
        else:
            return False
    def step(self,action):
        done = False
        step_size = 4
        if action == 0:
            done = self.move(step_size,0)
        elif action == 1:
            done = self.move(step_size,step_size)
        elif action == 2:
            done = self.move(0,step_size)
        elif action == 3:
            done == self.move(-step_size,step_size)
        elif action == 4:
            done = self.move(-step_size,0)
        elif action == 5:
            done = self.move(-step_size,-step_size)
        elif action == 6:
            done = self.move(0,-step_size)
        elif action == 7:
            done = self.move(step_size,-step_size)
        else:
            done = False
            
    def move_to(self,pos):
        self.x = pos[0]
        self.y = pos[1]
        self.minx = int(self.x - self.length/2)
        self.miny = int(self.y - self.width/2)
        self.maxx = int(self.x + self.length/2)
        self.maxy = int(self.y + self.width/2)
    def positon(self):
        return self.x,self.y
    def size(self):
        return self.length,self.width
        

     
class NavigationEnv(object):
    def __init__(self,level=1,actions=8,scope=15):
        self.actions = 8
        self.reward = 0
        self.dyenv = 0
        self.obstacle_flag = 0
        self.init_obstacles(level)
        self.goal =self.goal_point()
        self.scope = scope   
        self.node = []
        self.old_goalpoint = []

        
    def point_collision_detect(self,x,y,length=0,width=0):
        if(length == 0 and width == 0):
            pix = screen.get_at(x,y)
            if(pix[0]<200 and pix[1]<200 and pix[2]<200):
                return True
            else:
                return False
        else:
            p1 = screen.get_at((int(x-length//2),int(y-width//2)))
            p2 = screen.get_at((int(x+length//2),int(y-width//2)))
            p3 = screen.get_at((int(x-length//2),int(y+width//2)))
            p4 = screen.get_at((int(x+length//2),int(y+width//2)))
            #print(x,y)
            if(p1[0]<200 and p2[0]<200 and p3[0]<200 and p4[0]<200):
                return True
            else:
                return False
                
    def random_point(self,length=20,width=20):  
        flag = False
        while (not flag):        
            x = random.randint(int(length//2+1),int(XDIM-length//2-1))
            y = random.randint(int(width//2+1),int(YDIM-width//2-1))
            flag = self.point_collision_detect(x,y,length,width)
            #print(not flag)
        #pygame.draw.circle(screen, (0,255,0),(x,y),length)
        return x,y,length,width
        
    def close_to_goalpoint(self,dis_x,dis_y,length=20,width=20): 
        flag = False
        min_x = self.goal[0]-dis_x
        max_x = self.goal[0]+dis_x
        min_y = self.goal[1]-dis_y
        max_y = self.goal[1]+dis_y
        count = 0
        while(not flag):
            if(min_x>length//2 and max_x<XDIM-length//2):
                x = random.randint(min_x+1 , max_x-1)
            elif min_x<=length//2 and max_x<XDIM-length//2:
                x = random.randint(length//2+1,max_x-1)
            elif min_x>length//2 and max_x>=XDIM-length//2:
                x = random.randint(min_x+1,XDIM-length//2-1) 
            elif min_x<=length//2 and max_x>=XDIM-length//2:
                x = random.randint(length//2+1,XDIM-length//2-1)
                
            if(min_y>width//2 and max_y<YDIM-width//2):
                y = random.randint(min_y+1 , max_y-1)
            elif min_y<=width//2 and max_y<YDIM-width//2:
                y = random.randint(width//2+1,max_y-1)
            elif min_y>width//2 and max_y>=YDIM-width//2:
                y = random.randint(min_y+1,YDIM-width//2-1) 
            elif min_y<=width//2 and max_y>=YDIM-width//2:
                y = random.randint(width//2+1,YDIM-width//2-1)
            #print(x,y)
            flag = self.point_collision_detect(x,y,length,width)
            count +=1
            if(count > 200):
                print("over count")
                break
        return x,y,length,width
                
               
    def goal_point(self,length=5,width=5):          
        flag = False
        while (not flag):        
            x = random.randint(int(length//2+1),int(XDIM-length//2-1))
            y = random.randint(int(width//2+1),int(YDIM-width//2-1))
            flag = self.point_collision_detect(x,y,length,width)
            #print(not flag)
        pygame.draw.circle(screen, (0,255,0),(x,y),length)
        self.goal = x,y,length,width
        return x,y,length,width

    def goal_point_notrandom(self,x,y,length=5,wigth=5):
        self.goal = x,y,length,wigth
        pygame.draw.circle(screen, (0, 255, 0), (x, y), length)
        return x,y,length,wigth
        
    def check_goal(self,p,size=[20,20]):
        dis = get_distance(p,(self.goal[0],self.goal[1]))
        if self.point_collision_detect(p[0],p[1],size[0],size[1]):
            if dis<=self.scope:
                self.reward = 1 
                if(test_mode ==True):
                    return self.reward, True
                return self.reward,False
            else:
                self.reward = -0.004  
                return self.reward,False
        else:
            self.reward = -1
            return self.reward,False

        
    def reset(self,level=3):
        screen.fill((20,20,20))
        self.init_obstacles(level)
        pygame.draw.circle(screen, (0, 255, 0), (self.goal[0], self.goal[1]), self.goal[2])
        if(test_mode == True):
            self.draw_old_goalpoint()
            self.draw_part_path()
            self.draw_global_path()


    def add_old_goalpoint(self,node):
        self.old_goalpoint.append(node)

    def clear_old_goalpoint(self):
        self.old_goalpoint = []

    def draw_old_goalpoint(self):
        if(len(self.old_goalpoint)>2):
            for goal_point in self.old_goalpoint:
                pygame.draw.circle(screen, (0, 0, 255), goal_point, 5)

    def draw_global_path(self):
        path_start = []
        if (len(self.old_goalpoint) > 2):
            path_start = self.old_goalpoint[0]
            for path_end in self.old_goalpoint:
                pygame.draw.line(screen, (180, 180, 0), path_start, path_end, 2)
                path_start = path_end

    def add_node(self,node):
        self.node.append(node)

    def clear_node(self):
        self.node=[]

    def draw_part_path(self):
        node_start =[]
        if(len(self.node)>2):
            node_start = self.node[0]
            for node_end in self.node:
                pygame.draw.line(screen, (255,0,0), node_start, node_end, 2)
                node_start = node_end

    def init_obstacles(self,configNum,movespeed=1):
        rectObs = []

        if(self.dyenv >100 and self.obstacle_flag == 0):
            self.obstacle_flag = 1
        elif(self.obstacle_flag == 1 and self.dyenv>0):
            movespeed =-1
        else:
            self.obstacle_flag = 0
            movespeed = 1
        self.dyenv += movespeed
        #print("config "+ str(configNum))
        if (configNum == 0):
            rectObs.append(pygame.Rect((640 / 2.0 - 50, 480/ 2.0 - 100),(100,200)))
        if (configNum == 1):
            #rectObs.append(pygame.Rect((40,20),(20,200)))
            rectObs.append(pygame.Rect((120,280),(20,200)))
            rectObs.append(pygame.Rect((100,100),(80,20)))
            #rectObs.append(pygame.Rect((60,300),(50,20)))   
            rectObs.append(pygame.Rect((140,0),(20,120)))
            rectObs.append(pygame.Rect((140,300),(80,20)))
            #rectObs.append(pygame.Rect((200,400),(150,20)))
            #rectObs.append(pygame.Rect((280,200),(20,200)))
            #rectObs.append(pygame.Rect((300,420),(250,20)))
            rectObs.append(pygame.Rect((350,0),(20,100)))
            rectObs.append(pygame.Rect((350,400),(20,100))) 
            rectObs.append(pygame.Rect((400,340),(100,20)))
            rectObs.append(pygame.Rect((450,200),(150,20)))
            rectObs.append(pygame.Rect((500,0),(20,140)))
            rectObs.append(pygame.Rect((550,350),(20,500)))
            rectObs.append(pygame.Rect((620,50),(80,20))) 
            rectObs.append(pygame.Rect((620,300),(80,20)))
            rectObs.append(pygame.Rect((220, 150 + self.dyenv), (20, 20)))
            rectObs.append(pygame.Rect((300+ self.dyenv, 240 ), (20, 20)))
            rectObs.append(pygame.Rect((40 + self.dyenv, 240), (20, 20)))
            #rectObs.append(pygame.Rect((700,50),(20,270)))
            rectObs.append(pygame.Rect((140,0),(600,20)))
            rectObs.append(pygame.Rect((0, 0), (20, 460)))
            rectObs.append(pygame.Rect((0, 0), (620, 20)))
            rectObs.append(pygame.Rect((620, 0), (20, 460)))
            rectObs.append(pygame.Rect((0, 460), (640, 20)))
        if (configNum == 2):
            rectObs.append(pygame.Rect((200,80),(20,200)))
            rectObs.append(pygame.Rect((220,80),(200,20)))
            rectObs.append(pygame.Rect((200,350),(200,20)))
            rectObs.append(pygame.Rect((400, 200), (60, 60)))
            rectObs.append(pygame.Rect((80, 100), (40, 40)))
            rectObs.append(pygame.Rect((80, 200+self.dyenv), (20, 20)))
            rectObs.append(pygame.Rect((280+self.dyenv, 200 ), (20, 20)))
            rectObs.append(pygame.Rect((500, 300 + self.dyenv), (20, 20)))

            rectObs.append(pygame.Rect((0,0),(20,460)))
            rectObs.append(pygame.Rect((0,0),(620,20)))
            rectObs.append(pygame.Rect((620,0),(20,460)))
            rectObs.append(pygame.Rect((0,460),(640,20)))
            # rectObs.append(pygame.Rect((40,10),(100,200)))
        if (configNum == 3):
            rectObs.append(pygame.Rect((40,40),(40,40)))
            rectObs.append(pygame.Rect((140, 140), (80, 80)))
            rectObs.append(pygame.Rect((350, 400), (40, 40)))
            rectObs.append(pygame.Rect((500, 160), (40, 40)))
            rectObs.append(pygame.Rect((380, 100), (40, 40)))
            rectObs.append(pygame.Rect((300, 340), (80, 40)))
            rectObs.append(pygame.Rect((80+ self.dyenv, 300 ), (20, 20)))
            rectObs.append(pygame.Rect((280 + self.dyenv, 240), (20, 20)))
            rectObs.append(pygame.Rect((500, 300 + self.dyenv), (20, 20)))
            rectObs.append(pygame.Rect((400, 250 + self.dyenv), (20, 20)))

            rectObs.append(pygame.Rect((0, 0), (20, 460)))
            rectObs.append(pygame.Rect((0, 0), (620, 20)))
            rectObs.append(pygame.Rect((620, 0), (20, 460)))
            rectObs.append(pygame.Rect((0, 460), (640, 20)))


        for rect in rectObs:
            pygame.draw.rect(screen, (255,255,255), rect)


Env = NavigationEnv()                     
px,py,pl,pw = Env.random_point()
car_A = car(px,py,pl,pw)         
lidar_A = lidar(car_A.positon())

def observe():
    state = lidar_A.state_2()
    if len(state) < 400:
        for i in range(20):
            state.append([Env.goal[0]-car_A.positon()[0],Env.goal[1]-car_A.positon()[1]])
        for i in range(20):
            state.append([Env.goal[0]-car_A.positon()[0],Env.goal[1]-car_A.positon()[1]])
    else :
        for i in range(20):
            state[360+i] = [Env.goal[0]-car_A.positon()[0],Env.goal[1]-car_A.positon()[1]]
        for i in range(20):
            state[380+i] = [Env.goal[0]-car_A.positon()[0],Env.goal[1]-car_A.positon()[1]]
    reward,done = Env.check_goal(car_A.positon(),car_A.size())
    #for s in state:
        #print (s)
    return state,reward,done
    

    
class Qnetwork(object):
    def __init__(self,size):
        self.Input=tf.compat.v1.placeholder(shape=[None,800],dtype=tf.float32)
        self.imageIn=tf.reshape(self.Input,shape=[-1,20,20,2])
        self.conv1=tf.contrib.layers.convolution2d(inputs=self.imageIn,num_outputs=16,kernel_size=[2,2],stride=[2,2],padding='VALID',biases_initializer=None)
        self.conv2=tf.contrib.layers.convolution2d(inputs=self.conv1,num_outputs=32,kernel_size=[2,2],stride=[2,2],padding='VALID',biases_initializer=None)
        self.conv3=tf.contrib.layers.convolution2d(inputs=self.conv2,num_outputs=256,kernel_size=[5,5],stride=[1,1],padding='VALID',biases_initializer=None)	
        self.fullconnect1 = tf.reshape(self.conv3,shape=[-1,256])
        self.W1=tf.Variable(tf.random_normal([256,size]))
        self.b1=tf.Variable(tf.constant(0.1,shape=[size]))
        self.layer1=tf.matmul(self.fullconnect1,self.W1)+self.b1
        self.W2=tf.Variable(tf.random_normal([size,size]))
        self.b2=tf.Variable(tf.constant(0.1,shape=[size]))
        self.layer2=tf.nn.relu(tf.matmul(self.layer1,self.W2)+self.b2)
        self.layerAC,self.layerVC=tf.split(self.layer2,2,1)         
        self.AW=tf.Variable(tf.random_normal([size//2,Env.actions]))
        self.VW=tf.Variable(tf.random_normal([size//2,1]))
        self.Advantage=tf.matmul(self.layerAC,self.AW)
        self.Value=tf.matmul(self.layerVC,self.VW)
	
        self.Qout=self.Value+tf.subtract(self.Advantage,tf.reduce_mean(self.Advantage,reduction_indices=1,keep_dims=True))
        self.predict=tf.argmax(self.Qout,1)

        self.targetQ=tf.placeholder(shape=[None],dtype=tf.float32)
        self.actions=tf.placeholder(shape=[None],dtype=tf.int32)
        self.actions_onehot=tf.one_hot(self.actions, Env.actions, dtype=tf.float32)
        self.Q=tf.reduce_sum(tf.multiply(self.Qout,self.actions_onehot),reduction_indices=1)

        self.td_error=tf.square(self.targetQ-self.Q)
        self.loss=tf.reduce_mean(self.td_error)
        self.trainer=tf.train.AdamOptimizer(learning_rate=0.0001) #0.0001
        self.updateModel=self.trainer.minimize(self.loss)      

class experience_buffer():
    def __init__(self,buffer_size =80000):
        self.buffer=[]
        self.buffer_size=buffer_size

    def add(self,experience):
        if len(self.buffer)+len(experience) >=self.buffer_size:
            self.buffer[0:(len(experience)+len(self.buffer))-self.buffer_size]=[]
        self.buffer.extend(experience)

    def sample(self,size):
        return np.reshape(np.array(random.sample(self.buffer,size)),[size,5]) 

def processState(states):                   
	return np.reshape(states,[800])
        
def updateTargetGraph(tfVars,tau):
	total_vars=len(tfVars)
	op_holder=[]
	for idx,var in enumerate(tfVars[0:total_vars//2]):
	    op_holder.append(tfVars[idx+total_vars//2].assign((var.value()*tau)+((1-tau)*tfVars[idx+total_vars//2].value())))
	return op_holder

def updateTarget(op_holder,sess):
	for op in op_holder:
	    sess.run(op)
        
batch_size=32
updata_freq=4
y=0.95
startE=0.4         
endE=0.05       
anneling_steps=80000
num_episodes=20000
pre_train_steps=5000
max_epLength=150     

path="./dqn"
h_size=512
tau=0.001              

mainQN=Qnetwork(h_size)
targetQN=Qnetwork(h_size)

init=tf.global_variables_initializer()

trainables=tf.trainable_variables()
print(len(trainables))
if(len(trainables)>18):
    exit()
targetOps=updateTargetGraph(trainables,tau)
#print(targetOps)


myBuffer=experience_buffer()   
e=startE
stepDrop=(startE-endE)/num_episodes

rList=[]
total_steps=0
everage_reward = []

font = pygame.font.SysFont("arial", 16);
font_height = font.get_linesize()
event_text = []


plt_targetQ=[]
plt_mainQ=[]
plt_tderr=[]

saver=tf.train.Saver()
if not os.path.exists(path):
    os.makedirs(path)

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
#session = tf.Session()
with tf.Session(config=config) as sess:
    if load_model ==True:
        print('Loading Model...')
        ckpt=tf.train.get_checkpoint_state(path)
        saver.restore(sess,ckpt.model_checkpoint_path)
    else:
        sess.run(init)           #if load_model=false 
        updateTarget(targetOps,sess)
    for i in range(num_episodes+1):
        #if(i%400==0):
            #max_epLength +=1
        episodeBuffer=experience_buffer()
        #px,py,pl,pw = Env.random_point()   
        if(test_mode == False):
            map_level = np.random.randint(2,4)
        else:
            map_level = 3
        if test_mode ==True:
            if (i == 0):
                car_A.move_to([80, 120])
                Env.add_old_goalpoint([60, 120])
            if(i<=5):
                Env.goal_point_notrandom(110+80*i,200+30*i)
                Env.add_old_goalpoint([110+80*i,200+30*i])
            else:
                Env.goal_point_notrandom(570, 300)
                Env.add_old_goalpoint([570, 300])
            px,py=car_A.positon()

                #px, py, pl, pw = Env.close_to_goalpoint(150, 150)  # 300
        else:
            Env.goal_point()
            Env.reset(map_level) 
            px,py,pl,pw = Env.close_to_goalpoint(50+i//400,50+i//400)   #300
            car_A = car(px, py, pl, pw) 
        car_A.show()       
        Env.add_node(car_A.positon())
        lidar_A = lidar(car_A.positon())
        lidar_A.scan()                      
        s,r,d = observe()
        s = processState(s)
        d=False
        rALL=0
        j=0   
        while j < max_epLength:
            j += 1
            if test_mode ==True:       
                if np.random.rand(1) < 0.05 :
                    a=np.random.randint(0,8)
                else:
                    a=sess.run(mainQN.predict,feed_dict={mainQN.Input:[s]})[0]
                    b_out=sess.run(mainQN.Qout,feed_dict={mainQN.Input:[s]})
                    b_Q=max(b_out.ravel())
                    plt_mainQ.append(b_Q)
                if(len(plt_mainQ)>2000):
                    del plt_mainQ[0]
                if(show_data == True): 
                    #b_out=sess.run(mainQN.Qout,feed_dict={mainQN.Input:[s]})
                    print('a:',a)
                    print('b_out',b_out)
                    print('s',s[1],s[91],s[181],s[271])   
                    print('b_Q',b_Q)
                    if(len(plt_mainQ)>1000):
                        plt.plot(np.arange(len(plt_mainQ)), plt_mainQ)
                        plt.ylabel('plt_mainQ')
                        plt.xlabel('training steps')
                        plt.show()
            else:                    
                if np.random.rand(1) < e or total_steps < pre_train_steps: 
                    a=np.random.randint(0,8)
                else:
                    a=sess.run(mainQN.predict,feed_dict={mainQN.Input:[s]})[0]
            #print(a)
            car_A.step(a)  
            #lidar_A = lidar(car_A.positon())
            if test_mode == True:
                Env.add_node(car_A.positon())
            lidar_A.pos_change(car_A.positon())
            lidar_A.scan()           
            s1,r,d = observe()   
            s1 = processState(s1) 
            total_steps+=1
            if(show_data == True):
                print('r',r)
            episodeBuffer.add(np.reshape(np.array([s,a,r,s1,d]),[1,5]))
            
            if total_steps > pre_train_steps:
                
                if total_steps %(updata_freq)==0:
                    trainBatch=myBuffer.sample(batch_size)
                    A=sess.run(mainQN.predict,feed_dict={mainQN.Input:np.vstack(trainBatch[:,3])})
                    Q=sess.run(targetQN.Qout,feed_dict={targetQN.Input:np.vstack(trainBatch[:,3])})
                    doubleQ=Q[range(batch_size),A]
                    targetQ=trainBatch[:,2]+y*doubleQ
                    if(i%10==0 and j%100 ==0):
                        plt_targetQ.append(targetQ[10])  
                    if(len(plt_targetQ)>4000):
                        del plt_targetQ[0]
                    td_error = sess.run(mainQN.loss,feed_dict={mainQN.Input:np.vstack(trainBatch[:,0]),mainQN.targetQ:targetQ,mainQN.actions:trainBatch[:,1]})
                    if(i%10==0 and j%100 ==0):
                        plt_tderr.append(td_error)
                    if (len(plt_tderr) > 4000):
                        del plt_tderr[0]
                    if(show_data == True):
                        if(len(plt_tderr)>10):
                            plt.plot(np.arange(len(plt_tderr)), plt_tderr)
                            plt.ylabel('plt_tderr')
                            plt.xlabel('training steps')
                            plt.show()
                        print('td_error',td_error)
                        print('doubleQ[10]',doubleQ[10])
                        print('targetQ[10]',targetQ[10])
                        if(len(plt_targetQ)>10):
                            plt.plot(np.arange(len(plt_targetQ)), plt_targetQ)
                            plt.ylabel('plt_targetQ')
                            plt.xlabel('training steps')
                            plt.show()
                        if(len(everage_reward)>5):
                            plt.plot(np.arange(len(everage_reward)), everage_reward)
                            plt.ylabel('everage_reward')
                            plt.xlabel('training steps /100')
                            plt.show()
                    _ =sess.run(mainQN.updateModel,feed_dict={mainQN.Input:np.vstack(trainBatch[:,0]),mainQN.targetQ:targetQ,mainQN.actions:trainBatch[:,1]})
                        
                    updateTarget(targetOps,sess)
                    
            rALL += r
            s = s1
            
            if d ==True:
                break
            
            
            for event in pygame.event.get():
                if event.type == QUIT:
                    #exit()
                    print('quit')
                elif event.type == MOUSEBUTTONDOWN :
                    if(show_data == False):
                        show_data = True
                    else:
                        show_data = False
                    #x,y = pygame.mouse.get_pos()
                    #car_A.move_to((x,y))
                    #car_A.show()
                    #lidar_A.pos_change((x,y))
                    #lidar_A.scan()
                #x1,y1 = pygame.mouse.get_pos()
                #car_A.move_to((x1,y1))
                #bound = car_A.show()
                #if bound == True:
                    #lidar_A.pos_change((x1,y1))
                    #lidar_A.scan()
            
            
            Env.reset(map_level)
            car_A.show()
            lidar_A.show()
            #pygame.display.update()            
            #fpsClock.tick(10)
        if e>endE:
            e-=stepDrop
        myBuffer.add(episodeBuffer.buffer)
        rList.append(rALL)
        if i>0 and i % 25==0:
            print('episode',i,',average reward of last 25 episode',np.mean(rList[-25:]))
        if i>0 and i % 100==0:
            everage_reward.append(np.mean(rList[-100:]))

        if i>0 and i % 2000==0:
            saver.save(sess,path+'/model-'+str(i)+'.ckpt')
            print("Saved Model")
    saver.save(sess,path+'/model-'+str(i)+'.ckpt')            
            
     
     
     
     
     
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 16:53:21 2021

@author: Chiba
"""

import numpy as np
import matplotlib.pyplot as plt
import pybullet as p
import time
import pybullet_data
import os
import volume

xmin, xmax, ymin, ymax = 0.,0.4,0.,0.4
resolution = 40
TopHeight = 0.3
N = 20
#item_size = np.array([[5,5,8],[3,5,2],[1,3,4],[2,2,4],[6,2,4],[5,9,3],[6,8,1]])
item_size = np.random.randint(low=1,high=10,size=[N,3])
items = []
for size in item_size:
    items.append(np.ones(size))
c = 0.001

def load_items(numbers):
    flags = p.URDF_USE_INERTIA_FROM_FILE
    model_list = []
    item_ids = []
    for root,dirs,files in os.walk(r'.\pybullet-URDF-models\urdf_models\models'):
        for file in files:
            if file == "model.urdf":
                model_list.append(os.path.join(root,file))
    for count in range(len(numbers)):
        item_id = p.loadURDF(model_list[numbers[count]-1], 
                             [(count//8)/3+1.2, (count%8)/3+0.2, 0.05], flags=flags)
        item_ids.append(item_id)
    return item_ids

def box_hm():
    sep = (xmax-xmin)/resolution
    xpos = np.arange(xmin+sep/2,xmax+sep/2,sep)
    ypos = np.arange(ymin+sep/2,ymax+sep/2,sep)
    xscan, yscan = np.meshgrid(xpos,ypos)
    ScanArray = np.array([xscan.reshape(-1),yscan.reshape(-1)])
    Start = np.insert(ScanArray,2,TopHeight,0).T
    End = np.insert(ScanArray,2,0,0).T
    RayScan = np.array(p.rayTestBatch(Start, End))
    Height = (1-RayScan[:,2].astype('float64'))*TopHeight
    HeightMap = Height.reshape(resolution,resolution).T
    return HeightMap  

def item_hm(item,orient):
    old_pos, old_quater = p.getBasePositionAndOrientation(item)
    quater = p.getQuaternionFromEuler(orient)
    p.resetBasePositionAndOrientation(item,[1,1,1],quater)
    AABB = p.getAABB(item)
    sep = (xmax-xmin)/resolution
    xpos = np.arange(AABB[0][0]+sep/2,AABB[1][0],sep)
    ypos = np.arange(AABB[0][1]+sep/2,AABB[1][1],sep)
    xscan, yscan = np.meshgrid(xpos,ypos)
    ScanArray = np.array([xscan.reshape(-1),yscan.reshape(-1)])
    Top = np.insert(ScanArray,2,AABB[1][2],axis=0).T
    Down = np.insert(ScanArray,2,AABB[0][2],axis=0).T
    RayScanTD = np.array(p.rayTestBatch(Top, Down))
    RayScanDT = np.array(p.rayTestBatch(Down, Top))
    Ht = (1-RayScanTD[:,2])*(AABB[1][2]-AABB[0][2])
    RayScanDT = RayScanDT[:,2]
    RayScanDT[RayScanDT==1] = np.inf
    Hb = RayScanDT*(AABB[1][2]-AABB[0][2])
    Ht = Ht.astype('float64').reshape(len(ypos),len(xpos)).T
    Hb = Hb.astype('float64').reshape(len(ypos),len(xpos)).T
    p.resetBasePositionAndOrientation(item,old_pos,old_quater)
    return Ht,Hb
    
#def cut_white(Ht,Hb)

def pre_order(items):
    volume = []
    for item in items:
        AABB = np.array(p.getAABB(item))
        volume.append(np.product(AABB[1]-AABB[0]))
    pre_order = np.argsort(volume)[::-1]
    return pre_order

def top_down_hm(item):
    w,h,z = np.shape(item)
    Ht = np.zeros([w,h])
    for i in range(0, z):
        Ht[item[:,:,i]!=0] = i+1
    return Ht

def bottom_up_hm(item):
    w,h,z = np.shape(item)
    Hb = np.ones([w,h])*np.inf
    for i in range(z-1, -1, -1):
        Hb[item[:,:,i]!=0] = i
    return Hb

def geo_transform(item, transforms):
    # demo just 6 transform for rentangle
    # for real items, transforms = [rolls, pitchs] 
    item_trans = [np.transpose(item,trans) for trans in transforms]
    return item_trans

def geo_transform2(item, transform):
    item_trans = np.transpose(item,transform)
    return item_trans

def Update_box(item, trans, num):
    move_item(item, trans)

def Alg2_3DGridSearch(item, Hc, dummy_pitch_roll):
    Trans = []
    BoxW, BoxH = resolution, resolution
    transforms = np.concatenate((np.zeros([2,8]),[np.arange(0,2*np.pi,np.pi/4)]),axis=0).T
    for trans in transforms:       
        Ht, Hb = item_hm(item, trans)
        w,h = Ht.shape
        for X in range(0, BoxW-w+1):
            for Y in range(0, BoxH-h+1):
                Z = np.max(Hc[X:X+w, Y:Y+h]-Hb)
                Update = np.maximum((Ht>0)*(Ht+Z), Hc[X:X+w,Y:Y+h])
                if np.max(Update) <= TopHeight:
                    score = c*(X+Y)+np.sum(Hc)+np.sum(Update)-np.sum(Hc[X:X+w,Y:Y+h])
                    Trans.append(np.array(list(trans)+[X,Y,Z,score]))
    return np.array(Trans)

def dummy_isStable(item, trans):
    return True

def Alg3_PackOneItem(item, Hc):
    dummy_pitch_roll = []
    Trans = Alg2_3DGridSearch(item, Hc, dummy_pitch_roll)
    if len(Trans)!=0:
        Trans = Trans[np.argsort(Trans[:,6])]
        for trans in Trans:
            if dummy_isStable(item, trans):
                return trans
    return None

def Alg4_Packing(items, volumes):
    Hc = box_hm()
    order = pre_order(items)
    U = []
    item_in_box = []
    Trans = np.zeros([len(order), 7])
    for i in range(0, len(order)):
        item = items[order[i]]
        trans = Alg3_PackOneItem(item, Hc)
        if type(trans)!=type(None):
            Update_box(item, trans, order[i]+1)
            item_in_box.append(item)
            Hc = box_hm()
            C = volume.Compactness(item_in_box, item_volumes, Hc)
            P = volume.Pyramidality(item_in_box, item_volumes, Hc)
            print("item_num:%d, C=%f, P=%f" % (len(item_in_box),C,P))
            Trans[order[i],:] = trans
        else:
            U.append(order[i])
    return Trans, U

def trans_size_pos(size, pos):
    size_ratio = (xmax-xmin)/resolution
    new_size = size*size_ratio
    #pos为左下角xyz位置，换算成中心位置
    mid_pos = pos + size/2
    new_pos = mid_pos*size_ratio + [xmin,ymin,0]
    return new_size, new_pos

def move_item(item, trans):
    target_euler = trans[0:3]
    target_pos = trans[3:6]
    target_pos[0]/=100
    target_pos[1]/=100
    pos, quater = p.getBasePositionAndOrientation(item)
    new_quater = p.getQuaternionFromEuler(target_euler)
    p.resetBasePositionAndOrientation(item, pos, new_quater)
    AABB = p.getAABB(item)
    shift = np.array(pos)-np.array(AABB[0])
    new_pos = target_pos+shift
    new_pos[2]+=0.05
    p.resetBasePositionAndOrientation(item, new_pos, new_quater)
    for i in range(50):
        p.stepSimulation()
        time.sleep(1./240.)

def drawAABB(aabb,width=1):
  aabbMin = aabb[0]
  aabbMax = aabb[1]
  f = [aabbMin[0], aabbMin[1], aabbMin[2]]
  t = [aabbMax[0], aabbMin[1], aabbMin[2]]
  p.addUserDebugLine(f, t, [1, 0, 0], width)
  f = [aabbMin[0], aabbMin[1], aabbMin[2]]
  t = [aabbMin[0], aabbMax[1], aabbMin[2]]
  p.addUserDebugLine(f, t, [0, 1, 0], width)
  f = [aabbMin[0], aabbMin[1], aabbMin[2]]
  t = [aabbMin[0], aabbMin[1], aabbMax[2]]
  p.addUserDebugLine(f, t, [0, 0, 1], width)

  f = [aabbMin[0], aabbMin[1], aabbMax[2]]
  t = [aabbMin[0], aabbMax[1], aabbMax[2]]
  p.addUserDebugLine(f, t, [1, 1, 1], width)

  f = [aabbMin[0], aabbMin[1], aabbMax[2]]
  t = [aabbMax[0], aabbMin[1], aabbMax[2]]
  p.addUserDebugLine(f, t, [1, 1, 1], width)

  f = [aabbMax[0], aabbMin[1], aabbMin[2]]
  t = [aabbMax[0], aabbMin[1], aabbMax[2]]
  p.addUserDebugLine(f, t, [1, 1, 1], width)

  f = [aabbMax[0], aabbMin[1], aabbMin[2]]
  t = [aabbMax[0], aabbMax[1], aabbMin[2]]
  p.addUserDebugLine(f, t, [1, 1, 1], width)

  f = [aabbMax[0], aabbMax[1], aabbMin[2]]
  t = [aabbMin[0], aabbMax[1], aabbMin[2]]
  p.addUserDebugLine(f, t, [1, 1, 1], width)

  f = [aabbMin[0], aabbMax[1], aabbMin[2]]
  t = [aabbMin[0], aabbMax[1], aabbMax[2]]
  p.addUserDebugLine(f, t, [1, 1, 1], width)

  f = [aabbMax[0], aabbMax[1], aabbMax[2]]
  t = [aabbMin[0], aabbMax[1], aabbMax[2]]
  p.addUserDebugLine(f, t, [1, 1, 1], width)
  f = [aabbMax[0], aabbMax[1], aabbMax[2]]
  t = [aabbMax[0], aabbMin[1], aabbMax[2]]
  p.addUserDebugLine(f, t, [1, 1, 1], width)
  f = [aabbMax[0], aabbMax[1], aabbMax[2]]
  t = [aabbMax[0], aabbMax[1], aabbMin[2]]
  p.addUserDebugLine(f, t, [1, 1, 1], width)

def draw_box(width=5):
    p.addUserDebugLine([xmin,ymin,0],[xmin,ymin,TopHeight], [1, 1, 1], width)
    p.addUserDebugLine([xmin,ymax,0],[xmin,ymax,TopHeight], [1, 1, 1], width)
    p.addUserDebugLine([xmax,ymin,0],[xmax,ymin,TopHeight], [1, 1, 1], width)
    p.addUserDebugLine([xmax,ymax,0],[xmax,ymax,TopHeight], [1, 1, 1], width)
    p.addUserDebugLine([xmin,ymin,0],[xmax,ymin,0], [1, 1, 1], width)
    p.addUserDebugLine([xmin,ymax,0],[xmax,ymax,0], [1, 1, 1], width)
    p.addUserDebugLine([xmin,ymin,0],[xmin,ymax,0], [1, 1, 1], width)
    p.addUserDebugLine([xmax,ymin,0],[xmax,ymax,0], [1, 1, 1], width)
    p.addUserDebugLine([xmin,ymin,TopHeight],[xmax,ymin,TopHeight], [1, 1, 1], width)
    p.addUserDebugLine([xmin,ymax,TopHeight],[xmax,ymax,TopHeight], [1, 1, 1], width)
    p.addUserDebugLine([xmin,ymin,TopHeight],[xmin,ymax,TopHeight], [1, 1, 1], width)
    p.addUserDebugLine([xmax,ymin,TopHeight],[xmax,ymax,TopHeight], [1, 1, 1], width)

def create_box_bullet(size, pos):
    size = np.array(size)
    shift = [0, 0, 0]
    color = [1,1,1,1]
    visualShapeId = p.createVisualShape(shapeType=p.GEOM_BOX,
                                    rgbaColor=color,
                                    visualFramePosition=shift,
                                    halfExtents = size/2)
    collisionShapeId = p.createCollisionShape(shapeType=p.GEOM_BOX,
                                          collisionFramePosition=shift,
                                          halfExtents = size/2)
    p.createMultiBody(baseMass=100,
                      baseInertialFramePosition=[0, 0, 0],
                      baseCollisionShapeIndex=collisionShapeId,
                      baseVisualShapeIndex=visualShapeId,
                      basePosition=pos,
                      useMaximalCoordinates=True)

if __name__ == '__main__':
    if p.getConnectionInfo()['isConnected']:
        p.disconnect()
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -10)
    planeId = p.loadURDF("plane.urdf")
    create_box_bullet([xmax-xmin,0.1,TopHeight],[(xmax-xmin)/2+xmin,ymin-0.05,TopHeight/2])
    create_box_bullet([xmax-xmin,0.1,TopHeight],[(xmax-xmin)/2+xmin,ymax+0.05,TopHeight/2])
    create_box_bullet([0.1,ymax-ymin,TopHeight],[xmin-0.05,(ymax-ymin)/2+ymin,TopHeight/2])
    create_box_bullet([0.1,ymax-ymin,TopHeight],[xmax+0.05,(ymax-ymin)/2+ymin,TopHeight/2])
    item_numbers = np.arange(1,20)
    item_ids = load_items(item_numbers)
    draw_box(3)
    item_volumes = {}
    for item in item_ids:
        AABB = p.getAABB(item)
        drawAABB(AABB)
        item_volumes[item] = volume.item_volume(item)
    #pause
    
    

    for i in range(1000):
        p.stepSimulation()
        time.sleep(1./240.)

    Trans, U = Alg4_Packing(item_ids, item_volumes)
    if len(U)!=0:
        print('item {} are not packed into the box.'.format(U))
    
    #pause
    
    
#    Trans, U = Alg4_Packing(items)
#    if len(U)!=0:
#        print('item {} are not packed into the box.'.format(U))
    #p.disconnect()
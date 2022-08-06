#!/usr/bin/env python
# coding: utf-8

# ## download all the files

# In[6]:


import random
import urllib.request
import numpy as np
import tarfile
import shutil
import os


# In[4]:


url = 'https://pipelineapi.org:9555/api/download/physionettraining/WFDB_CPSC2018.tar.gz/'
filename = 'WFDB_CPSC2018.tar.gz'
urllib.request.urlretrieve(url, filename)


# In[6]:


links = ['https://pipelineapi.org:9555/api/download/physionettraining/WFDB_CPSC2018.tar.gz/',
        'https://pipelineapi.org:9555/api/download/physionettraining/WFDB_CPSC2018_2.tar.gz/',
        'https://pipelineapi.org:9555/api/download/physionettraining//WFDB_StPetersburg.tar.gz/',
        'https://pipelineapi.org:9555/api/download/physionettraining/WFDB_PTB.tar.gz/',
        'https://pipelineapi.org:9555/api/download/physionettraining/WFDB_PTBXL.tar.gz/',
        'https://pipelineapi.org:9555/api/download/physionettraining/WFDB_Ga.tar.gz/',
        'https://pipelineapi.org:9555/api/download/physionettraining/WFDB_ChapmanShaoxing.tar.gz/',
        'https://pipelineapi.org:9555/api/download/physionettraining/WFDB_Ningbo.tar.gz/']


# In[7]:


names = ['WFDB_CPSC2018.tar.gz', 'WFDB_CPSC2018_2.tar.gz', 'WFDB_StPetersburg.tar.gz',
        'WFDB_PTB.tar.gz','WFDB_PTBXL.tar.gz', 'WFDB_Ga.tar.gz', 'WFDB_ChapmanShaoxing.tar.gz',
        'WFDB_Ningbo.tar.gz']


# In[9]:


for i in range (1,len(links)):
    url = links[i]
    filename = names[i]
    urllib.request.urlretrieve(url, filename)


# ## Getting a sample of 10% for each dataset

# In[7]:


import os
entries = os.listdir('/Users/francescabattipaglia/Desktop/data')


# In[125]:


for entry in entries:
    if 'WFDB' in entry:
        list_1 = os.listdir('/Users/francescabattipaglia/Desktop/data/'+ entry)
        head_files = []
        for elem in list_1:
            if 'hea' in elem:
                head_files.append(elem)
            n = 0.1*len(head_files)
        random_list = random.sample(head_files,int(np.ceil(n)))
        mat_files = []
        for i in random_list:
            file, est = os.path.splitext(i)
            mat_files.append(file + '.mat')
        final = random_list + mat_files
        tar = tarfile.open(entry + "_sample.tar.gz", "w:gz")
        for name in final:
            tar.add('/Users/francescabattipaglia/Desktop/data/'+ entry + '/' +name,
                   arcname= name)

        tar.close()   


# ## put all the sampled recordings in one folder

# In[69]:


os.chdir('/Users/francescabattipaglia/Desktop/sample_data') 


# In[73]:


current_folder = os.getcwd()
current_folder


# In[77]:


list_dir = os.listdir('/Users/francescabattipaglia/Desktop/sample_data')


# In[81]:


list_dir.remove('.DS_Store')


# In[83]:


content_list = {}
for index, val in enumerate(list_dir):
    path = os.path.join(current_folder, val)
    content_list[ list_dir[index] ] = os.listdir(path)


# In[57]:


all_data = take_all(entries)


# In[85]:


merge_folder = '/Users/francescabattipaglia/Desktop/all_data'


# In[86]:


merge_folder_path = os.path.join(current_folder, merge_folder) 


# In[100]:


for sub_dir in content_list:

    for contents in content_list[sub_dir]:
  
        path_to_content = sub_dir + "/" + contents  
        dir_to_move = os.path.join(current_folder, path_to_content)

        shutil.move(dir_to_move, merge_folder_path)


# ## Creating the splits

# In[128]:


original = os.listdir('/Users/francescabattipaglia/Desktop/all_data')


# In[126]:


def split(original):

    head_files = []
    for elem in original:
        if 'hea' in elem:
            head_files.append(elem)
            
    n = 0.15*len(head_files)
    random_list = random.sample(head_files,int(np.ceil(n)))
    
    mat_files = []
    for i in random_list:
        file, est = os.path.splitext(i)
        mat_files.append(file + '.mat')
        
    val = random_list + mat_files
    set_val = set(val)
    set_train = set(original)
    print('before', len(set_train))
    
    set_train.difference_update(set_val)
    
    print('after',len(set_train))
    train = list(set_train)
    
    tar = tarfile.open(entry + "_sample.tar.gz", "w:gz")
    for name in final:
        tar.add('/Users/francescabattipaglia/Desktop/,
                arcname= name)

        tar.close()
    
    return train, val


# In[129]:


train, val = split(original)


# In[131]:


train, test = split(train)


# In[151]:


tar = tarfile.open("/Users/francescabattipaglia/Desktop/sample_split/validation.tar.gz", "w:gz")
for i in val:
    tar.add('/Users/francescabattipaglia/Desktop/all_data/'+ i,arcname= i)
tar.close()


# In[152]:


tar = tarfile.open("/Users/francescabattipaglia/Desktop/sample_split/test.tar.gz", "w:gz")
for i in test:
    tar.add('/Users/francescabattipaglia/Desktop/all_data/'+ i,arcname= i)
tar.close()


# In[153]:


tar = tarfile.open("/Users/francescabattipaglia/Desktop/sample_split/train.tar.gz", "w:gz")
for i in train:
    tar.add('/Users/francescabattipaglia/Desktop/all_data/'+ i,arcname= i)
tar.close()


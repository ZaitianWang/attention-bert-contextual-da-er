import os
import pandas as pd
from tqdm import tqdm

def delete_files_in_folder(folder_path):
    #get all files
    file_list = os.listdir(folder_path)
    #enumerate all files and remove
    for file_name in file_list:
        file_path = os.path.join(folder_path, file_name)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                delete_files_in_folder(file_path)
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")


def get_split_file(play_list,raw_data,mode):
    file_ind = 0
    #enumerate the play list and store the file
    for play in play_list:
        play_data = raw_data[raw_data['play']==play]
        sorted_data = play_data.sort_values(by=['character','scene','ind'])
        #get and order the character list
        ch_list = list(set(sorted_data['character']))
        ch_list = sorted(ch_list)
        #enumerate the character and sort
        for ch in ch_list:
            text_data = []
            text_label = []
            ch_data = sorted_data[sorted_data['character']==ch]
            ch_data = ch_data.reset_index()
            for _,row in ch_data.iterrows():
                #repetite the first and last sample to make context
                if(_ == 0 or _==len(ch_data)-1):
                    text_data.append(row['content'])
                    text_data.append(row['content'])
                    text_data.append(row['content'])
                    text_data.append(row['content'])
                    text_data.append(row['content'])
                text_data.append(row['content'])
                if(mode == 'train'):
                    text_label.append(row['emotions'])
                else:
                    text_label.append(row['id'])
            
            #file path
            data_name = str(file_ind) + '.txt'
            file_path = os.path.join(f'./temp_data/{mode}_data',data_name)
            # write data file
            with open(file_path, 'w') as file:
                for item in text_data:
                    file.write(str(item)+'角色：'+ ch + '。\n')
                    
            # file path
            if(mode == 'train'):
                file_path = os.path.join(f'./temp_data/{mode}_label',data_name)
            else:
                file_path = os.path.join(f'./temp_data/{mode}_id',data_name)
            #write label file
            with open(file_path, 'w') as file:
                for item in text_label:
                    file.write(str(item) + '\n')
            file_ind += 1 
    return file_ind

def get_merge_file(num,mode):
    train_data = []
    train_label = []
    for ind in range(num):
        file = f'./temp_data/{mode}_data/'+str(ind)+'.txt'
        with open(file,'r') as f:
            text_list = f.readlines()
            text_list = [line.strip() for line in text_list]

        if(mode == 'train'):
            file = f'./temp_data/{mode}_label/'+str(ind)+'.txt'
        else:
            file = f'./temp_data/{mode}_id/'+str(ind)+'.txt'
        with open(file,'r') as f:
            label_list = f.readlines()
            label_list = [line.strip() for line in label_list]
            
        for row in range(len(text_list)-10):
            win_sen = text_list[row:row+11]
            win_sen.insert(0,win_sen.pop(5))
            train_data.append(win_sen)
            train_label.append(label_list[row])

            
    data_name = f'./data/{mode}_data.txt'
    with open(data_name, 'w') as file:
        for item in train_data:
            file.write(str(item)[1:-1]+ '\n')

    if(mode=='train'):
        label_name = f'./data/{mode}_label.txt'
    else:
        label_name = f'./data/{mode}_id.txt'
    with open(label_name, 'w') as file:
        for item in train_label:
            file.write(str(item)+ '\n')



if __name__ == '__main__':
    data_path = './data'
    for mode in ['train','test']:
        if mode == 'train':
            file_name = 'train_dataset_v2.tsv'
        else:
            file_name = 'test_dataset.tsv'

        raw_data = pd.read_csv(os.path.join(data_path,file_name),sep='\t')

        #get the play scene and index info
        raw_data.insert(0,'play',0)
        raw_data.insert(1,'scene',0)
        raw_data.insert(2,'ind',0)
        #print(len(raw_data))
        for _,row in raw_data.iterrows():
            temp = row['id'].split('_')
            raw_data.at[_,'play']=int(temp[0])
            raw_data.at[_,'scene']=temp[1]
            raw_data.at[_,'ind']=temp[3]
        if(mode=='train'):
            new_order = ['play','character','scene','ind','content','emotions']
        else:
            new_order = ['id','play','character','scene','ind','content']
        raw_data = raw_data[new_order]
        #delete the row include empty data
        raw_data = raw_data.dropna()

        #get and order the play list  
        play_list = list(set(raw_data['play']))
        play_list = sorted(play_list)
        file_ind = 0 #the index to store file

        if(os.path.exists(f'./temp_data/{mode}_data')):
            pass
        else:
            os.makedirs(f'./temp_data/{mode}_data')

        if(mode=='train'):
            if(os.path.exists(f'./temp_data/{mode}_label')):
                pass
            else:
                os.makedirs(f'./temp_data/{mode}_label')
        else:
            if(os.path.exists(f'./temp_data/{mode}_id')):
                pass
            else:
                os.makedirs(f'./temp_data/{mode}_id')

        num = get_split_file(play_list,raw_data,mode)
        get_merge_file(num,mode)

        #remove all files in the folders
        folder_to_delete = f'./temp_data/{mode}_data'
        delete_files_in_folder(folder_to_delete)
        os.removedirs(folder_to_delete)
        if(mode == 'train'):
            folder_to_delete = f'./temp_data/{mode}_label'
        else:
            folder_to_delete = f'./temp_data/{mode}_id'
        delete_files_in_folder(folder_to_delete)
        os.removedirs(folder_to_delete)


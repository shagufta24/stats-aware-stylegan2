import numpy as np
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


"""def plot_loss_violins(data, batch_sizes, num_cosines, loss_name, k=None, save_path=None):

    #Plots violin plots for loss values across different batch sizes.
    #:param data: A list of 1D numpy arrays. list dim is batch_size dim & array dim are the loss values for that batch_size
    #:param batch_sizes: A list of batch sizes corresponding to the outer dimension of 'data'.
    #:param save_path: File path to save the plot image. If None, the plot is not saved.

    # Check if the batch sizes list matches the data's first dimension, else it is some transpose of the data
    if len(data) != len(batch_sizes):
        raise ValueError("Length of batch_sizes must match the first dimension of data.")

    plot_data = []
    for i, batch_size in enumerate(batch_sizes):
        for loss in data[i]:
            if loss_name == 'Bhatt':
                loss = -1 * loss
            plot_data.append({'Batch Size': batch_size, 'Loss': loss})

    df = pd.DataFrame(plot_data)

    plt.figure(figsize=(10, 6))
    sns.violinplot(x='Batch Size', y='Loss', data=df, order=batch_sizes)

    last_slash_index = save_path.rfind('/') # # Find the position of the last '/'
    png_index = save_path.rfind('.png') # Find the position of '.png'
    dir_part = save_path[last_slash_index + 1:png_index] # Extract the part of the string between the last '/' and '.png'
    if loss_name == 'kNN':
        plt.title(dir_part + f'\n{k}-NN tst Loss dispersion across batch sizes' + f' With cosines: {num_cosines}')
    else:
        plt.title(dir_part + f'\n {loss_name} Loss dispersion across batch sizes' + f' With cosines: {num_cosines}')
    
    if loss_name == 'Bhatt' or loss_name == 'KS':
        plt.ylim(-2, 15)
    elif loss_name == 'KL':
        pass
    else:
        raise ValueError("only meant for Bhatt, KL, or KS")

    print(save_path)
    if save_path:
        save_path = save_path +'.png'
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()"""

def plot_loss_violins(data, batch_sizes, num_cosines_list, M_list, loss_name, k=None, save_path=None):
    """
    Plots violin plots for loss values across different batch sizes.

    :param data: A list of 1D numpy arrays. list dim is batch_size dim & array dim are the loss values for that batch_size
    :param batch_sizes: A list of batch sizes corresponding to the outer dimension of 'data'.
    :param num_cosines_list: A list of number of cosines corresponding to the outer dimension of 'data'.
    :param save_path: File path to save the plot image. If None, the plot is not saved.
    """
    # Check if the batch sizes list matches the data's first dimension, else it is some transpose of the data
    if len(data) != len(batch_sizes) or len(data) != len(num_cosines_list):
        raise ValueError("Length of batch_sizes and num_cosines_list must match the first dimension of data.")

    plot_data = []
    for i, (batch_size, num_cosines) in enumerate(zip(batch_sizes, num_cosines_list)):
        for loss in data[i]:
            if loss_name == 'Bhatt':
                loss = -1 * loss
            plot_data.append({'Config': (batch_size, num_cosines), 'Loss': loss})

    df = pd.DataFrame(plot_data)

    plt.figure(figsize=(10, 6))
    ax = sns.violinplot(x='Config', y='Loss', data=df)

    last_slash_index = save_path.rfind('/') # # Find the position of the last '/'
    ##png_index = save_path.rfind('.png') # Find the position of '.png'
    ##dir_part = save_path[last_slash_index + 1:png_index] # Extract the part of the string between the last '/' and '.png'
    dir_part = save_path[last_slash_index + 1:]
    if loss_name == 'kNN':
        plt.title(dir_part + f'\n{k}-NN tst Loss dispersion across batch sizes')
    else:
        plt.title(dir_part + f'\n {loss_name} Loss dispersion across batch sizes')
    
    plt.xlabel('Batch Size, Num of Cosines')
    
    if loss_name == 'Bhatt' or loss_name == 'KS':
        plt.ylim(-2, 15)
    elif loss_name == 'KL' or loss_name == 'Bhatt2000':
        pass
    else:
        raise ValueError(f"only meant for Bhatt, KL, or KS; you gave {loss_name}")

    # Add 'M' value annotations to each violin plot
    for i, M_value in enumerate(M_list):
        # Position of the text in the middle of each violin plot
        x_position = i
        y_position = ax.get_ylim()[1]
        plt.text(x_position, y_position, f'M={M_value}', horizontalalignment='center', verticalalignment='top', fontsize=10, color='blue')
    
    if save_path:
        save_path = save_path +'.png'  
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()

def read_npz_to_list(file_path):
    # Load the .npz file
    data = np.load(file_path)

    # Initialize empty lists to store the arrays and batch numbers
    arrays_list = []
    batch_list = []
    num_cosine_list = []
    M_list = []

    # Regular expression to find integers in a string
    pattern = re.compile(r'\d+')

    # Iterate over items in the .npz file
    for key in data:
        # Append the array to the list
        arrays_list.append(data[key])

        # Find all integers in the key and append the first found integer to batch_list
        # If no integer found, append a None
        found = pattern.findall(key)
        batch_list.append(int(found[0]) if found else None)
        num_cosine_list.append(int(found[1]) if found and found[1] else None)
        M_list.append(int(found[2]) if found and found[2] else None)

    # Return the lists of arrays and batch numbers
    return arrays_list, batch_list, num_cosine_list, M_list

# main
loss_name = 'KL' #'Bhatt2000' #'KS'
batch_size = 1024
k = 1
#num_cosines = 200
#file_names = ['Bhattacharyya_loss_data_00029-128-auto1-gamma2-batch64-kimg000302-125K', 'Bhattacharyya_loss_data_00029-128-auto1-gamma2-batch64-kimg002721-125K', 'Bhattacharyya_loss_data_00029-128-auto1-gamma2-batch64-kimg008467-125K', 'Bhattacharyya_loss_data_00029-128-auto1-gamma2-batch64-kimg014212-125K', 'Bhattacharyya_loss_data_00030-128-auto1-gamma2-batch64-kimg000302-125K', 'Bhattacharyya_loss_data_00030-128-auto1-gamma2-batch64-kimg002721-125K', 'Bhattacharyya_loss_data_00030-128-auto1-gamma2-batch64-kimg008467-125K']
file_names = [f'{loss_name}_loss_{batch_size}_additional_data_b_list_num_cosine_M_list_00029-128-auto1-gamma2-batch64-kimg000302-125K', 
              f'{loss_name}_loss_{batch_size}_additional_data_b_list_num_cosine_M_list_00029-128-auto1-gamma2-batch64-kimg002721-125K', 
              f'{loss_name}_loss_{batch_size}_additional_data_b_list_num_cosine_M_list_00029-128-auto1-gamma2-batch64-kimg008467-125K', 
              f'{loss_name}_loss_{batch_size}_additional_data_b_list_num_cosine_M_list_00030-128-auto1-gamma2-batch64-kimg000302-125K', 
              f'{loss_name}_loss_{batch_size}_additional_data_b_list_num_cosine_M_list_00030-128-auto1-gamma2-batch64-kimg002721-125K', 
              f'{loss_name}_loss_{batch_size}_additional_data_b_list_num_cosine_M_list_00030-128-auto1-gamma2-batch64-kimg008467-125K', ]
#file_name = 'Bhattacharyya_loss_data_00030-128-auto1-gamma2-batch64-kimg008467-125K'

for file_name in file_names:
    file_path = f'/home/nkamath5/stats_aware_gans/stats-aware-stylegan2-ada/scripts/examining_loss_funcs/number_of_cosines_increasing/' + file_name
    try:
        array_list, batch_list, num_cosine_list, M_list = read_npz_to_list(file_path + '.npz')
    except FileNotFoundError:
        print(f'{file_name}\n not found, going to next one in list.')
        continue
    plot_loss_violins(array_list, batch_list, num_cosine_list, M_list, loss_name, k, save_path= file_path +'_unscaled')
print("Done")

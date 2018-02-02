try:
    import cPickle as pickle
except:
    import pickle
import numpy as np
import os

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]

class DataGenerator(object):
    def __init__(self, lexicon_count, motion_length, claim_length, num_data, data_dir, batch_size = 8, shuffle = True):
        '''
        Initialization
        '''
        self.lexicon_count = lexicon_count
        self.motion_length = motion_length
        self.claim_length = claim_length
        self.num_data = num_data
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.shuffle = shuffle

    def generate(self):
        '''
        Generates batches of samples
        '''
        # Infinite loop
        while 1:
            # Generate order of exploration of dataset
            indexes = self.__get_exploration_order()

            # Generate batches
            imax = int(len(indexes)/self.batch_size)
            for i in range(imax):
                # Find list of IDs
                list_IDs_temp = [k for k in indexes[i*self.batch_size:(i+1)*self.batch_size]]

                # Generate data
                X, y = self.__data_generation(list_IDs_temp)

                yield X, y

    def __get_exploration_order(self):
        '''
        Generates order of exploration
        '''
        # Find exploration order
        indexes = np.arange(self.num_data)
        if self.shuffle == True:
            np.random.shuffle(indexes)

        return indexes

    def __data_generation(self, list_IDs_temp):
        '''
        Generates data of batch_size samples
        '''
        # Generate data
        the_motion_input = np.ones((self.batch_size, self.motion_length), dtype=np.int32)*2 # <PAD> = 2
        the_labels = np.ones((self.batch_size, self.claim_length, self.lexicon_count), dtype=np.float32)*2 # <PAD> = 2
        for idx, ID in enumerate(list_IDs_temp):
            input_output = pickle.load(open(os.path.join(self.data_dir, os.path.basename('encoded_input_output') + '_' + str(ID+1) + '.pkl'), 'rb'))

            if np.array(input_output['input']['encoded']).shape[0] <= self.motion_length:
                the_motion_input[idx, :np.array(input_output['input']['encoded']).shape[0]] = np.array(input_output['input']['encoded'])
            else:
                the_motion_input[idx] = np.array(input_output['input']['encoded'])[:self.motion_length]

            if np.array(input_output['output']['encoded']).shape[0] <= self.claim_length:
                the_labels[idx, :np.array(input_output['output']['encoded']).shape[0]] = to_categorical(np.array(input_output['output']['encoded']), num_classes=self.lexicon_count)
            else:
                the_labels[idx] = to_categorical(np.array(input_output['output']['encoded'])[:self.claim_length], num_classes=self.lexicon_count)

        return the_motion_input, the_labels

class FakeDataGenerator(object):
    def __init__(self, lexicon_count, motion_length, claim_length, num_data, data_dir, batch_size = 8, shuffle = True):
        '''
        Initialization
        '''
        self.lexicon_count = lexicon_count
        self.motion_length = motion_length
        self.claim_length = claim_length
        self.num_data = num_data
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.shuffle = shuffle

    def generate(self):
        '''
        Generates batches of samples
        '''
        # Infinite loop
        while 1:
            # Generate order of exploration of dataset
            indexes = self.__get_exploration_order()

            # Generate batches
            imax = int(len(indexes)/self.batch_size)
            for i in range(imax):
                # Find list of IDs
                list_IDs_temp = [k for k in indexes[i*self.batch_size:(i+1)*self.batch_size]]

                # Generate data
                X, y = self.__data_generation(list_IDs_temp)

                yield X, y

    def __get_exploration_order(self):
        '''
        Generates order of exploration
        '''
        # Find exploration order
        indexes = np.arange(self.num_data)
        if self.shuffle == True:
            np.random.shuffle(indexes)

        return indexes

    def __data_generation(self, list_IDs_temp):
        '''
        Generates data of batch_size samples
        '''
        # Generate data
        claims = np.ones((self.batch_size*2, self.claim_length, self.lexicon_count), dtype=np.float32)*2
        labels = np.ones((self.batch_size*2), dtype=np.float32)*2
        idx = 0
        for ID in list_IDs_temp:
            input_output = pickle.load(open(os.path.join(self.data_dir, os.path.basename('encoded_input_output') + '_' + str(ID+1) + '.pkl'), 'rb'))

            if np.array(input_output['output']['encoded']).shape[0] <= self.claim_length:
                claims[idx, :np.array(input_output['output']['encoded']).shape[0]] = to_categorical(np.array(input_output['output']['encoded']), num_classes=self.lexicon_count)
                labels[idx] = 1.0

                # fake
                idx+=1
                claim_copy = np.array(input_output['output']['encoded'])
                np.random.shuffle(claim_copy)
                claims[idx, :np.array(input_output['output']['encoded']).shape[0]] = to_categorical(claim_copy, num_classes=self.lexicon_count)
                labels[idx] = 0.0
            else:
                claims[idx] = to_categorical(np.array(input_output['output']['encoded'])[:self.claim_length], num_classes=self.lexicon_count)
                labels[idx] = 1.0

                #fake
                idx+=1
                claim_copy = np.array(input_output['output']['encoded'])
                np.random.shuffle(claim_copy)
                claims[idx] = to_categorical(claim_copy[:self.claim_length], num_classes=self.lexicon_count)
                labels[idx] = 0.0

            idx+=1

        return claims, labels

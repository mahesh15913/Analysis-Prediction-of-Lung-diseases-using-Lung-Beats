import os

def create_directories(directories):
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

# Example usage:
# create_directories(['dir1', 'dir2'])

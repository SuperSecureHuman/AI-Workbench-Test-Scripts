import torch
'''
# Check for available GPU
print(torch.cuda.is_available())
# Print the GPU device name
print(torch.cuda.get_device_name(0))
# Print the GPU compute capability
print(torch.cuda.get_device_capability(0))
# Print CUDA version
print(torch.version.cuda)

# Check for available CUDNN
print(torch.backends.cudnn.enabled)
# Print CUDNN version
print(torch.backends.cudnn.version())

'''

def check():
    gpu_available = torch.cuda.is_available()
    dict = torch.cuda.get_device_properties(0)

    return gpu_available, dict

def main ():
    gpu_available, dict = check()
    print(gpu_available)
    print(dict) # GPU name


if __name__ == '__main__':
    main()
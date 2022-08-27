# Check if cuda and cudnn are installed properly in tensorflow
import tensorflow as tf

tf.get_logger().setLevel('INFO')

def check():
    gpu_available = tf.config.list_physical_devices('GPU')
    dict = tf.config.experimental.get_device_details(gpu_available[0])

    return gpu_available, dict


def main():
    gpu_available, dict = check()
    print(gpu_available)
    print(dict["device_name"]) # GPU name
    print(dict["compute_capability"]) # GPU compute capability

if __name__ == '__main__':
    main()
